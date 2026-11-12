#!/usr/bin/env python3
"""Fine-tune de tinta sobre pares reais (pacote drawfmt da frente 13).

Modulo da linhagem page-clean (o corpo gerado training/train_ink3.py
e uma corrida de destilacao 3 classes sobre dado sintetico gerado no
kernel; este modulo e a corrida de FINE-TUNE sobre pares reais
rotulados, que aquela corrida nao cobre). Reusa da linhagem, sem
copiar o corpo: a arquitetura U-Net da v2 (base 8, a do modelo de
producao page_clean_unet.onnx a470bd28) e a perda binaria mascarada
(BCE pos_weight=8 + Dice, a masked_loss do corpo). A definicao da
arquitetura abaixo e transcrita de training/train_body_tri.py
(Block/UNet, byte a byte as classes da linhagem) e e guardada por
asserts executaveis: o state_dict do w8 precisa carregar strict e,
no caminho ONNX, a reconstrucao precisa concordar com o ONNX de
producao (verificado em laboratorio em 2026-11-10: .pt strict ok,
diff max 1,3e-4; unfold diff max 1,0e-3 nos logits).

Receita (config em training/configs/ink-finetune-real-v1.json):
init nos pesos do w8; alvo binario tinta = label3 em {1,2}; ignore
de verdade (valid = ~ignore); split por SESSAO — treino nas sessoes
do config, os pares temporais (unico GT independente do w8) ficam
FORA do treino e servem de regua interna; aumento fotometrico forte
(ganho/gamma/contraste — a regua F4 mediu explosao de x25 com
brilho x1,15 no w8); patches 256x256 (o contrato de inferencia do
app), amostrados com centro em tinta sobreamostrado. Aumento
opcional de cast quente (config "warm_cast_aug", default OFF,
frente 51): papel ambar sintetico (R/B 1,3-3,4, papel 70-140)
sobre pares ja rotulados, tag aug_photometric — a familia do vp44
(relatorio 53), cuja cauda de cast nao tem fonte real em branco.
Loader (frente 53): patches amostrados por workers com prefetch,
rng derivado de (seed, epoca, indice global) — serial e paralelo
produzem os mesmos patches; "sample_workers": 0 volta ao serial.
Aumento na GPU (frente 55, config "gpu_augment", default OFF):
paginas residentes no device em uint8 + crop/geometrico/
fotometrico/cast em torch nativo; selecao por coordenada segue
na CPU (plan_patch) e o caminho CPU fica byte a byte intacto.

Uso:
  python training/finetune_real.py --config <cfg.json> \
      [--data-dir <pacote>] [--out-dir <dir>] [--smoke]

Config: JSON plano (convencao do corpo irmao). Sem --config, le
DRAW_WORK/cfg.json ou a env F31_CFG_JSON (kernels). Saidas e estado
em --out-dir ou DRAW_WORK: metrics.json incremental, last.pt /
best.pt por epoca, util.json (recibo de utilizacao), ONNX final no
contrato do app (page -> ink_logits, 256 fixo, BN dobrado).
"""

import argparse
import gc
import glob
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
import struct
import subprocess
import threading
import time
import zlib

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

T0 = time.time()
PATCH = 256


# ------------------------------------------------------------------
# Arquitetura: U-Net da v2 (linhagem page-clean), base 8.
# Transcricao de training/train_body_tri.py (Block/UNet). Os asserts
# de carga strict + equivalencia no main() sao a prova de fidelidade.
# ------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.c(x)


class UNet(nn.Module):
    """A arquitetura da v2, com forward_features para a KD."""

    def __init__(self, base=16, out_ch=1):
        super().__init__()
        self.d1 = Block(3, base)
        self.d2 = Block(base, base * 2)
        self.d3 = Block(base * 2, base * 4)
        self.d4 = Block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = Block(base * 8 + base * 4, base * 4)
        self.u2 = Block(base * 4 + base * 2, base * 2)
        self.u1 = Block(base * 2 + base, base)
        self.head = nn.Conv2d(base, out_ch, 1)
        # gradient checkpointing (frente 77): default OFF; main() liga
        # so em CUDA (flag "grad_ckpt": false na config desliga).
        self.grad_ckpt = False

    def _ckpt(self, mod, x):
        """Roda um bloco com checkpointing quando ligado.

        So em treino com grad habilitado: o checkpoint descarta as
        ativacoes intermediarias do bloco e as recalcula no backward
        (use_reentrant=False). Em eval / no_grad / CPU o caminho e o
        eager de sempre, byte a byte.
        """
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=False)
        return mod(x)

    def forward_features(self, x):
        s1 = self._ckpt(self.d1, x)
        s2 = self._ckpt(self.d2, self.pool(s1))
        s3 = self._ckpt(self.d3, self.pool(s2))
        b = self._ckpt(self.d4, self.pool(s3))
        y = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        y3 = self._ckpt(self.u3, torch.cat([y, s3], 1))
        y = F.interpolate(y3, scale_factor=2, mode="bilinear", align_corners=False)
        y2 = self._ckpt(self.u2, torch.cat([y, s2], 1))
        y = F.interpolate(y2, scale_factor=2, mode="bilinear", align_corners=False)
        y1 = self._ckpt(self.u1, torch.cat([y, s1], 1))
        return self.head(y1), [s1, s2, s3, b, y3, y2, y1]

    def forward(self, x):
        return self.forward_features(x)[0]


class FoldBlock(nn.Module):
    """Bloco sem BN: o ONNX de producao da v2 saiu com o BN dobrado
    nas convolucoes (grafo sem no BatchNormalization)."""

    def __init__(self, a, b):
        super().__init__()
        self.c1 = nn.Conv2d(a, b, 3, padding=1)
        self.c2 = nn.Conv2d(b, b, 3, padding=1)

    def forward(self, x):
        return F.relu(self.c2(F.relu(self.c1(x))))


class FoldUNet(nn.Module):
    """Professor reconstruido do ONNX quando o .pt nao esta anexado.
    As ativacoes por estagio sao as mesmas do modelo de producao,
    porque a dobra do BN em eval e exata."""

    def __init__(self, base=16):
        super().__init__()
        self.d1 = FoldBlock(3, base)
        self.d2 = FoldBlock(base, base * 2)
        self.d3 = FoldBlock(base * 2, base * 4)
        self.d4 = FoldBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = FoldBlock(base * 8 + base * 4, base * 4)
        self.u2 = FoldBlock(base * 4 + base * 2, base * 2)
        self.u1 = FoldBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward_features(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.d4(self.pool(s3))
        y = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        y3 = self.u3(torch.cat([y, s3], 1))
        y = F.interpolate(y3, scale_factor=2, mode="bilinear", align_corners=False)
        y2 = self.u2(torch.cat([y, s2], 1))
        y = F.interpolate(y2, scale_factor=2, mode="bilinear", align_corners=False)
        y1 = self.u1(torch.cat([y, s1], 1))
        return self.head(y1), [s1, s2, s3, b, y3, y2, y1]

    def forward(self, x):
        return self.forward_features(x)[0]


BASE = 8  # largura do w8 (TEACHER_BASE da linhagem)
W8_RELEASE_URL = (
    "https://github.com/santleme/draw/releases/download/"
    "models-2026-11-06/page_clean_unet.onnx"
)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------
# torch.compile no passo de GPU (frente 75; port do padrao #62 de
# training/train.py — decide/maybe_compile com fallback seguro pra
# eager). So-CUDA: em CPU o corpo segue byte a byte o anterior (a
# prova OFF garante o hash 2d669a95... de 128 patches, cast ON).
# Frente 76: o compile passa a ser aplicado ANTES da sonda de lote,
# para a sonda medir o pico de memoria do passo COMPILADO (na F75 o
# compile era aplicado depois da sonda e ela media o eager). O
# warmup do compile cai na sonda (1o passo do 1o candidato), nao
# mais na epoca 0; o regime segue sendo as epocas 1-2.
# Frente 80: o modo do compile passa de "default" para
# "max-autotune" (segunda opcao pre-registrada, relatorio 79 §6).
# Ressalva: o modo GEMM do max-autotune NAO engaja neste T4 (aviso
# do Inductor no log da F78 e da F79: "Not enough SMs to use
# max_autotune_gemm mode" — T4 tem 40 SMs); o ganho, se vier, vem
# do autotune de conv. A primeira compilacao e mais longa que a do
# default — o warmup pode somar minutos ao kernel.
# Frente 82r (veredito): o reset do dynamo entre a sonda e o treino
# funciona (liberou os 12,14 GiB retidos; o OOM nao recorreu), mas o
# max-autotune NAO foi adotado — nivel 1 passou (54,04 s/epoca) e o
# nivel 2 nao (abaixo do piso de ~7% sobre a F79). Reconciliacao
# 2026-11-12: o modo volta a "default" (o regime validado da F79:
# channels_last + lote 256, 55,10 s/epoca); o recibo do modo
# (compile_mode no metrics.json) fica.
# ------------------------------------------------------------------
class _CompileFallbackLog(logging.Handler):
    """Guarda os avisos do dynamo de frame que caiu para eager."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        lines = record.getMessage().splitlines() or [""]
        errors = [
            line
            for line in lines[1:]
            if line
            and not line[0].isspace()
            and not line.startswith(("due to", "Traceback", "Set TORCHDYNAMO"))
        ]
        self.messages.append(" | ".join([lines[0].strip(), *errors])[:500])


_COMPILE_FALLBACK_LOG = _CompileFallbackLog()

# Modo do torch.compile: "default" (o max-autotune da frente 80 foi
# medido e NAO adotado — veredito da F82r acima). O recibo do modo
# vai pro log e pro metrics.json (compile_mode).
_COMPILE_MODE = "default"


def _apply_compile(model):
    """``torch.compile(model, mode=_COMPILE_MODE)``, com fallback seguro.

    Devolve ``(modelo, status, motivo)``: ``("engaged", None)`` ou
    ``("fallback", motivo)``. Falha na chamada cai para eager na
    hora; falha tardia (tracado ou compilacao do backend no 1o
    passo) cai para eager por frame via ``suppress_errors``, com o
    motivo no log do dynamo (lido por _compile_runtime_status).
    Nunca falha a corrida.
    """
    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        compiled = torch.compile(model, mode=_COMPILE_MODE)
    except Exception as exc:  # noqa: BLE001 - qualquer falha vira eager
        log(
            f"AVISO: torch.compile falhou ({type(exc).__name__}: {exc}); "
            "seguindo em eager."
        )
        return model, "fallback", f"{type(exc).__name__}: {exc}"
    _COMPILE_FALLBACK_LOG.messages.clear()
    dynamo_log = logging.getLogger("torch._dynamo.convert_frame")
    if _COMPILE_FALLBACK_LOG not in dynamo_log.handlers:
        dynamo_log.addHandler(_COMPILE_FALLBACK_LOG)
    log(
        f"torch.compile: LIGADO (modo {_COMPILE_MODE}; erro de "
        "compilacao no 1o passo cai para eager)"
    )
    return compiled, "engaged", None


def _compile_runtime_status(status, reason=None):
    """``(status, motivo)`` depois de passos reais.

    ``"engaged"`` vira ``("fallback", motivo)`` quando o dynamo
    recusou algum frame (``suppress_errors``) desde o
    ``_apply_compile``.
    """
    if status == "engaged" and _COMPILE_FALLBACK_LOG.messages:
        return "fallback", _COMPILE_FALLBACK_LOG.messages[0]
    return status, reason


def _reset_dynamo_after_probe(dev):
    """Frente 82 (opcao (b) do relatorio 80 §6): ``torch._dynamo.reset()``
    + liberacao explicita entre a sonda de lote e o treino.

    Mecanismo medido (log da F81): a sonda roda passos compilados
    para os candidatos de lote; cada candidato captura seus proprios
    grafos CUDA e a retencao acumulada chega a 12,14 GiB em private
    pools antes do 1o lote de treino — o residente cheio (2.052 MiB)
    empurra por cima do teto de 14,56 GiB. O reset apos a sonda
    escolher o lote deve destruir os grafos da sonda e deixar so os
    grafos do lote escolhido durante o treino (o 1o passo do treino
    recompila — o warmup cai na ep 0, fora do regime).
    Documentacao atual do PyTorch (2026): ``torch.compiler.reset()``
    "clears all compilation caches and restores the system to its
    initial state" e e implementado como ``torch._dynamo.reset()``;
    a implementacao do reset chama
    ``torch._inductor.cudagraph_trees.reset_cudagraph_trees()``
    ("Clear all cudagraph trees") quando CUDA esta inicializado —
    o reset destroi as arvores de grafos e os buffers estaticos
    voltam ao pool do alocador, de onde o treino realoca.
    Ressalva honesta: ``memory_reserved`` pode seguir alto (o pool
    do caching allocator e retido para reuso); o que decide e a
    memoria ALOCADA cair e o 1o lote caber — por isso os dois
    recibos (alocada/reservada) antes e depois.
    So-CUDA: em CPU devolve None sem tocar no dynamo (o corpo
    segue byte a byte o anterior; a prova OFF garante o hash).
    Devolve o dicionario de recibo (ou None em CPU).
    """
    if dev != "cuda":
        return None
    antes_aloc = torch.cuda.memory_allocated() // 2**20
    antes_res = torch.cuda.memory_reserved() // 2**20
    log(f"memoria antes do reset: alocada {antes_aloc} MiB / reservada {antes_res} MiB")
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    depois_aloc = torch.cuda.memory_allocated() // 2**20
    depois_res = torch.cuda.memory_reserved() // 2**20
    log(f"memoria apos reset: alocada {depois_aloc} MiB / reservada {depois_res} MiB")
    return {
        "antes_aloc_mib": antes_aloc,
        "antes_res_mib": antes_res,
        "depois_aloc_mib": depois_aloc,
        "depois_res_mib": depois_res,
    }


# ------------------------------------------------------------------
# Init a partir do w8: .pt (strict) ou ONNX de producao (unfold).
# ------------------------------------------------------------------
def _foldunet_from_onnx(path):
    """Reconstrucao do FoldUNet a partir dos inicializadores do ONNX
    (porta de build_teacher_from_onnx do corpo; os 28 tensores
    onnx::Conv_* vem na ordem do grafo)."""
    import re

    import onnx as onnx_lib

    m = onnx_lib.load(path)
    tensors = {}
    for t in m.graph.initializer:
        tensors[t.name] = torch.from_numpy(onnx_lib.numpy_helper.to_array(t).copy())
    names = sorted(
        (n for n in tensors if n.startswith("onnx::Conv_")),
        key=lambda n: int(re.search(r"(\d+)$", n).group(1)),
    )
    assert len(names) == 28, len(names)
    fold = FoldUNet(BASE)
    blocks = [
        (fold.d1, 0),
        (fold.d2, 2),
        (fold.d3, 4),
        (fold.d4, 6),
        (fold.u3, 8),
        (fold.u2, 10),
        (fold.u1, 12),
    ]
    with torch.no_grad():
        for blk, i in blocks:
            blk.c1.weight.copy_(tensors[names[2 * i]])
            blk.c1.bias.copy_(tensors[names[2 * i + 1]])
            blk.c2.weight.copy_(tensors[names[2 * i + 2]])
            blk.c2.bias.copy_(tensors[names[2 * i + 3]])
        fold.head.weight.copy_(tensors["head.weight"])
        fold.head.bias.copy_(tensors["head.bias"])
    return fold


def _unfold_to_unet(fold):
    """FoldUNet -> UNet com BN identidade: em eval, BN(0,1,1,0) e a
    identidade (a menos do eps), entao o UNet nasce com a mesma
    funcao do modelo dobrado e ganha BN treinavel pro fine-tune."""
    unf = UNet(base=BASE, out_ch=1)
    pairs = (
        (unf.d1, fold.d1),
        (unf.d2, fold.d2),
        (unf.d3, fold.d3),
        (unf.d4, fold.d4),
        (unf.u3, fold.u3),
        (unf.u2, fold.u2),
        (unf.u1, fold.u1),
    )
    with torch.no_grad():
        for dst, srcf in pairs:
            dst.c[0].weight.copy_(srcf.c1.weight)
            dst.c[0].bias.copy_(srcf.c1.bias)
            dst.c[3].weight.copy_(srcf.c2.weight)
            dst.c[3].bias.copy_(srcf.c2.bias)
            for bn in (dst.c[1], dst.c[4]):
                bn.weight.fill_(1.0)
                bn.bias.fill_(0.0)
                bn.running_mean.fill_(0.0)
                bn.running_var.fill_(1.0)
        unf.head.weight.copy_(fold.head.weight)
        unf.head.bias.copy_(fold.head.bias)
    return unf


_DOWNLOAD_HOSTS = (
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
)


def _check_download_url(url):
    import urllib.parse

    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or (parts.hostname or "") not in _DOWNLOAD_HOSTS:
        raise RuntimeError(f"download fora da allowlist HTTPS recusado: {url}")
    return url


def _download(url, dest):
    """Baixa só por HTTPS de hosts da Release; cada redirect é conferido."""
    import shutil
    import urllib.request

    class _Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            _check_download_url(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    _check_download_url(url)
    log(f"baixando {url}")
    opener = urllib.request.build_opener(_Redirect())
    with opener.open(url, timeout=120) as response, open(dest, "wb") as handle:
        shutil.copyfileobj(response, handle)
    return dest


def build_init_model(input_bases, work_dir):
    """Devolve (modelo UNet init no w8, info). Procura o .pt do w8
    nos inputs; senao o .onnx; senao baixa o ONNX da release do repo
    (o artefato publico do modelo de producao, sha no MANIFEST)."""
    model = UNet(base=BASE, out_ch=1)
    pt_hits = []
    onnx_hits = []
    for base in input_bases:
        pt_hits += glob.glob(f"{base}/**/page_clean_student_w8.pt", recursive=True)
        onnx_hits += glob.glob(f"{base}/**/page_clean_student_w8.onnx", recursive=True)
        onnx_hits += glob.glob(f"{base}/**/page_clean_unet.onnx", recursive=True)
    if pt_hits:
        sd = torch.load(pt_hits[0], map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (
            f"state_dict do w8 nao casa strict: {missing} {unexpected}"
        )
        log(f"init: .pt do w8 carregado strict de {pt_hits[0]}")
        return model, {"init_kind": "pt", "init_path": pt_hits[0]}
    if not onnx_hits:
        dest = os.path.join(work_dir, "page_clean_unet_release.onnx")
        _download(W8_RELEASE_URL, dest)
        onnx_hits = [dest]
    fold = _foldunet_from_onnx(onnx_hits[0])
    model = _unfold_to_unet(fold)
    log(f"init: unfold do ONNX do w8 de {onnx_hits[0]}")
    return model, {"init_kind": "onnx-unfold", "init_path": onnx_hits[0]}


# ------------------------------------------------------------------
# Dados: pacote drawfmt (input/label3/ignore/mask_total + manifest).
# ------------------------------------------------------------------
def find_package(data_dir):
    if data_dir:
        return data_dir
    for cand in glob.glob("/kaggle/input/**/manifest.jsonl", recursive=True):
        root = os.path.dirname(cand)
        if os.path.isdir(f"{root}/label3") and os.path.exists(f"{root}/sessions.json"):
            return root
    raise SystemExit("pacote drawfmt nao encontrado (use --data-dir)")


class Page:
    __slots__ = (
        "pair_id",
        "session_id",
        "route",
        "img",
        "lab",
        "valid",
        "paper_level",
        "ink_coords",
        "dark_coords",
    )

    def __init__(self, root, entry):
        import cv2

        self.pair_id = entry["pair_id"]
        self.session_id = entry["session_id"]
        self.route = entry["route"]
        self.img = cv2.imread(f"{root}/input/{self.pair_id}.png")
        assert self.img is not None, self.pair_id
        self.lab = cv2.imread(f"{root}/label3/{self.pair_id}.png", cv2.IMREAD_UNCHANGED)
        ign = cv2.imread(f"{root}/ignore/{self.pair_id}.png", cv2.IMREAD_UNCHANGED)
        self.valid = ign == 0
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.paper_level = float(np.percentile(gray, 85))
        ink = ((self.lab == 1) | (self.lab == 2)) & self.valid
        self.ink_coords = np.argwhere(ink)
        dark = (gray < self.paper_level - 30) & (self.lab == 0) & self.valid
        self.dark_coords = np.argwhere(dark)


def load_pages(root, manifest):
    return [Page(root, e) for e in manifest]


def photometric(patch, paper_level, rng):
    """Aumento fotometrico forte (so na imagem): ganho, gamma,
    contraste em torno do nivel do papel, offset e ruido. E a
    resposta a regua F4: o w8 explode/colapsa com +-15% de luz."""
    x = patch.astype(np.float32)
    gain = float(rng.uniform(0.80, 1.25))
    x = x * gain
    gamma = float(rng.uniform(0.75, 1.35))
    x = 255.0 * np.clip(x / 255.0, 0, 1) ** gamma
    contrast = float(rng.uniform(0.85, 1.15))
    x = (x - paper_level) * contrast + paper_level
    x = x + float(rng.uniform(-10, 10))
    if rng.random() < 0.4:
        x = x + rng.normal(0, float(rng.uniform(0.5, 4.0)), x.shape)
    if rng.random() < 0.12:
        import cv2

        x = cv2.GaussianBlur(np.clip(x, 0, 255).astype(np.uint8), (3, 3), 0).astype(
            np.float32
        )
    return np.clip(x, 0, 255)


def paper_stats(patch):
    """(paper_level, cast R/B) de um patch BGR: paper_level = p85 do
    cinza; cast = razao das medias R/B sobre a mascara de papel
    (cinza >= p85 - 25). E a metrica das frentes 47/50."""
    import cv2

    gray = cv2.cvtColor(
        np.clip(patch, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    pl = float(np.percentile(gray, 85))
    mask = gray >= pl - 25
    if mask.sum() < 100:
        mask = gray >= np.percentile(gray, 75)
    b = float(patch[..., 0][mask].mean())
    r = float(patch[..., 2][mask].mean())
    return pl, r / max(b, 1.0)


def warm_cast(patch, rng, rb_lo=1.3, rb_hi=3.4, paper_lo=70.0, paper_hi=140.0):
    """Aumento fotometrico de CAST QUENTE (aug_photometric sintetico,
    frente 51): reaplica a fotometria da familia do vp44 — papel
    ambar escuro — sobre um patch ja rotulado, SEM tocar no rotulo
    (e a mesma tinta sob outra luz). Amostra o cast-alvo R/B em
    [rb_lo, rb_hi] (uniforme em log; a familia real medida vive em
    1,31-1,47 e o vp44 em 3,36 — a cauda >=2,0 so existe sintetico)
    e o nivel do papel em [paper_lo, paper_hi] (a familia: 67-130;
    o treino da linhagem comeca em 149). Ganho por canal: B fica,
    G sobe com sqrt(k), R com k (k = alvo/cast atual, so esquenta,
    nunca esfria), e um fator comum poe o papel no nivel-alvo.
    NUNCA dessatura: no cinza o recall desta cena cai a 0,28
    (relatorio 53) — invariancia a cast vem de cor, nao de cinza."""
    pl, r0 = paper_stats(patch)
    if r0 >= rb_hi:
        return patch
    lo = max(rb_lo, r0 * 1.02)
    if lo >= rb_hi:
        # r0 em (rb_hi/1.02, rb_hi): o cast minimo ja passa do teto —
        # o patch ja esta no alvo; devolver sem tocar (mesma semantica
        # do early return acima). Sem isto, uniform(log(lo), log(rb_hi))
        # com lo > rb_hi levanta ValueError (matou a F52 v2 na ep 9).
        return patch
    rt = float(np.exp(rng.uniform(np.log(lo), np.log(rb_hi))))
    pt = float(rng.uniform(paper_lo, paper_hi))
    k = rt / r0
    x = patch.astype(np.float32).copy()
    x[..., 1] *= np.sqrt(k)
    x[..., 2] *= k
    # normaliza pelo p85 do cinza em FLOAT, sem clip: com cast forte
    # os canais estouram 255 e a medida em uint8 (paper_stats) satura
    # — o fator sairia errado e o papel nao escureceria (medido na
    # prova da frente 51: papel 178-207 com alvo <=140).
    gray = 0.114 * x[..., 0] + 0.587 * x[..., 1] + 0.299 * x[..., 2]
    pl2 = float(np.percentile(gray, 85))
    x *= pt / max(pl2, 1.0)
    return np.clip(x, 0, 255)


def geometric(img, lab, valid, rng):
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k)
        lab = np.rot90(lab, k)
        valid = np.rot90(valid, k)
    if rng.random() < 0.5:
        img, lab, valid = img[:, ::-1], lab[:, ::-1], valid[:, ::-1]
    if rng.random() < 0.5:
        img, lab, valid = img[::-1], lab[::-1], valid[::-1]
    return img, lab, valid


def sample_patch(page, rng, ink_frac=0.55, dark_frac=0.20, cast_cfg=None):
    """Patch 256 com centro sobreamostrado: tinta / escuro-nao-tinta
    (pauta, textura — o falso positivo do w8) / livre."""
    h, w = page.lab.shape
    r = rng.random()
    coords = None
    if r < ink_frac and len(page.ink_coords):
        coords = page.ink_coords
    elif r < ink_frac + dark_frac and len(page.dark_coords):
        coords = page.dark_coords
    if coords is not None:
        cy, cx = coords[int(rng.integers(0, len(coords)))]
        y = int(np.clip(cy - PATCH // 2 + rng.integers(-64, 65), 0, h - PATCH))
        x = int(np.clip(cx - PATCH // 2 + rng.integers(-64, 65), 0, w - PATCH))
    else:
        y = int(rng.integers(0, h - PATCH + 1))
        x = int(rng.integers(0, w - PATCH + 1))
    img = page.img[y : y + PATCH, x : x + PATCH]
    lab = page.lab[y : y + PATCH, x : x + PATCH]
    valid = page.valid[y : y + PATCH, x : x + PATCH]
    img, lab, valid = geometric(img, lab, valid, rng)
    img = photometric(img, page.paper_level, rng)
    if cast_cfg is not None and rng.random() < float(cast_cfg.get("prob", 0.25)):
        # aug_photometric sintetico (frente 51): cast quente sobre o
        # par ja rotulado; o rotulo NAO muda. Default OFF — sem a
        # flag na config, este ramo nunca executa.
        # ink_only (frente 65, opcional, default ausente=False): com a
        # flag, o cast so se aplica a patch COM tinta no rotulo —
        # positivos como base. Sem a flag, o comportamento e o da
        # linhagem, bit a bit (o sorteio do rng nao muda).
        if not (cast_cfg.get("ink_only") and not ((lab == 1) | (lab == 2)).any()):
            img = warm_cast(
                img,
                rng,
                rb_lo=float(cast_cfg.get("rb_min", 1.3)),
                rb_hi=float(cast_cfg.get("rb_max", 3.4)),
                paper_lo=float(cast_cfg.get("paper_min", 70.0)),
                paper_hi=float(cast_cfg.get("paper_max", 140.0)),
            )
    target = ((lab == 1) | (lab == 2)).astype(np.float32)
    return (
        img.transpose(2, 0, 1).astype(np.float32) / 255.0,
        target[None],
        valid[None].astype(np.float32),
    )


# ------------------------------------------------------------------
# Amostragem na GPU (frente 55).
#
# Split CPU x GPU (o portao de desenho e o split real medido pela
# F53): a SELECAO vive na CPU e os PIXELS vivem na GPU.
#
# - Na CPU (barato, escalar): plan_patch() sorteia do patch_rng
#   (seed, epoca, indice) as coordenadas do crop, o geometrico e
#   TODOS os escalares do fotometrico/cast, na mesma ordem de
#   sorteios do sample_patch. Por lote atravessam o barramento
#   ~64 x 20 floats (~5 KB): paginas, rotulos e listas de
#   coordenadas NUNCA descem por lote.
# - Na GPU (caro, pixel): as paginas moram residentes em uint8
#   (build_resident, uma vez por corrida) e o crop/geometrico/
#   fotometrico/cast rodam com ops torch nativas em lote — a
#   "direcao DALI" sem a dependencia DALI (ver relatorio 61 §6:
#   DALI nao vem na imagem do Kaggle e o torch nativo cobre
#   todas as ops; um pipeline DALI externo nao enxergaria as
#   listas de coordenadas nem o rng por indice do corpo).
# - As listas ink/dark_coords FICAM na CPU (numpy, ~370 MiB em
#   int64 no v6): move-las pra GPU custaria VRAM e exigiria
#   reescrever a selecao com outro fluxo de rng, quebrando a
#   identidade (seed, epoca, indice) provada na F53 — e a
#   selecao custa 3 sorteios escalares por patch.
#
# Equivalencia com o caminho CPU (declarada, nao prometida):
# crop, geometrico e escalares fotometricos/cast sao EXATOS por
# construcao (mesmo rng, mesma ordem). O que difere por desenho:
# (a) o ruido: o CPU tira o array normal do fluxo numpy, a GPU
# de um torch.Generator por patch semeado pelo hash de
# (seed, epoca, indice) — mesma distribuicao N(0, sigma), outra
# realizacao; (b) o flag de blur: no CPU ele e sorteado DEPOIS
# do array de ruido (o sorteio consome o fluxo), no plano ele
# vem antes do ruido — mesmo Bernoulli(0,12), valor independente
# por patch; (c) o cast-alvo: o CPU sorteia rt em
# [max(rb_lo, r0*1,02), rb_hi] com r0 do patch, o plano sorteia
# em [rb_lo, rb_hi] e o apply na GPU trava em max(rt, r0*1,02)
# — identico quando r0 <= 1,27 (o caso geral; patches ja quentes
# acumulam massa no limite inferior); (d) blur e paper_stats
# usam conv float32 / quantil float no lugar de cv2 uint8 —
# diferenca <= 1 nivel de cinza. A prova de equivalencia
# (frente55/prove_equiv_f55.py) mede o MAE e declara a
# tolerancia; o caminho GPU e distribuicao-equivalente, nunca
# bitwise. O caminho CPU NAO foi refatorado: com a flag
# desligada o codigo executado e byte a byte o antigo.
#
# O feed GPU roda serial na thread principal (workers CPU = 0):
# as ops sao kernels pequenos lancados em sequencia e a fila
# CUDA e o prefetch — threads disputando o contexto CUDA so
# somariam lancamento. O H2D por lote cai de ~84 MB (float32)
# pra ~5 KB (parametros).
# ------------------------------------------------------------------
def plan_patch(page, rng, ink_frac=0.55, dark_frac=0.20, cast_cfg=None):
    """Parametros de UM patch, sorteados de patch_rng na mesma
    ordem do sample_patch: crop, geometrico, fotometrico, cast.
    Nao toca em pixel nenhum — so escalares e coordenadas."""
    h, w = page.lab.shape
    r = rng.random()
    coords = None
    if r < ink_frac and len(page.ink_coords):
        coords = page.ink_coords
    elif r < ink_frac + dark_frac and len(page.dark_coords):
        coords = page.dark_coords
    if coords is not None:
        cy, cx = coords[int(rng.integers(0, len(coords)))]
        y = int(np.clip(cy - PATCH // 2 + rng.integers(-64, 65), 0, h - PATCH))
        x = int(np.clip(cx - PATCH // 2 + rng.integers(-64, 65), 0, w - PATCH))
    else:
        y = int(rng.integers(0, h - PATCH + 1))
        x = int(rng.integers(0, w - PATCH + 1))
    k = int(rng.integers(0, 4))
    # ordem do geometric(): o 1o sorteio vira as COLUNAS (esq-dir) e
    # o 2o as LINHAS (cima-baixo) — os nomes seguem essa ordem.
    fliplr = bool(rng.random() < 0.5)
    flipud = bool(rng.random() < 0.5)
    gain = float(rng.uniform(0.80, 1.25))
    gamma = float(rng.uniform(0.75, 1.35))
    contrast = float(rng.uniform(0.85, 1.15))
    offset = float(rng.uniform(-10, 10))
    noise_on = bool(rng.random() < 0.4)
    sigma = float(rng.uniform(0.5, 4.0)) if noise_on else 0.0
    blur = bool(rng.random() < 0.12)
    cast = None
    if cast_cfg is not None and rng.random() < float(cast_cfg.get("prob", 0.25)):
        # rt/pt sorteados aqui mesmo quando o patch ja esta quente
        # (o CPU so sortearia se r0 < rb_hi): o rng e por patch,
        # nao ha efeito cruzado, e o apply na GPU decide com r0.
        rt = float(
            np.exp(
                rng.uniform(
                    np.log(float(cast_cfg.get("rb_min", 1.3))),
                    np.log(float(cast_cfg.get("rb_max", 3.4))),
                )
            )
        )
        pt = float(
            rng.uniform(
                float(cast_cfg.get("paper_min", 70.0)),
                float(cast_cfg.get("paper_max", 140.0)),
            )
        )
        cast = {"rt": rt, "pt": pt}
    return {
        "y": y,
        "x": x,
        "k": k,
        "flipud": flipud,
        "fliplr": fliplr,
        "gain": gain,
        "gamma": gamma,
        "contrast": contrast,
        "offset": offset,
        "noise_on": noise_on,
        "sigma": sigma,
        "blur": blur,
        "cast": cast,
    }


def _patch_noise_seed(seed, epoch, idx):
    """Seed estavel (crc32, multiplataforma) do gerador de ruido
    de UM patch: o ruido e funcao pura de (seed, epoca, indice),
    como o resto do patch."""
    return zlib.crc32(struct.pack("<QQQ", int(seed) & 0xFFFFFFFF, int(epoch), int(idx)))


def build_resident(pages, dev):
    """Empilha as paginas de treino no device em uint8, uma vez
    por corrida: img (N,3,H,W) + lab (N,H,W) + valid (N,H,W).
    Exige shape unico (o v6/v7 tem 850x1200 em todas as paginas);
    sem isso o caminho GPU recusa e o treino segue no CPU."""
    shapes = {p.img.shape for p in pages}
    if len(shapes) != 1:
        raise ValueError(
            f"gpu_augment: paginas com shapes distintos {shapes} — "
            "caminho GPU exige shape unico"
        )
    imgs = torch.stack([torch.from_numpy(p.img).permute(2, 0, 1) for p in pages]).to(
        dev
    )
    labs = torch.stack([torch.from_numpy(p.lab) for p in pages]).to(dev)
    valids = torch.stack([torch.from_numpy(p.valid) for p in pages]).to(dev)
    mib = (imgs.numel() + labs.numel() + valids.numel()) / 2**20
    return {
        "img": imgs,
        "lab": labs,
        "valid": valids,
        "mib": float(mib),
        "pages": len(pages),
        "shape": tuple(pages[0].img.shape),
    }


def _gpu_warm_cast(xf, params, cast_idx, cast_cfg, dev):
    """warm_cast em lote (o equivalente GPU do warm_cast): so os
    patches com cast sorteado; r0 medido no patch em float, rt
    travado em >= r0*1,02 (nunca esfria), skip se r0 >= rb_hi."""
    ci = torch.tensor(cast_idx, device=dev)
    sub = torch.clamp(xf[ci], 0, 255)
    gray = 0.114 * sub[:, 0] + 0.587 * sub[:, 1] + 0.299 * sub[:, 2]
    flat = gray.flatten(1)
    pl = torch.quantile(flat, 0.85, dim=1)
    q75 = torch.quantile(flat, 0.75, dim=1)
    mask = gray >= (pl - 25).view(-1, 1, 1)
    cnt = mask.sum(dim=(1, 2))
    mask = torch.where((cnt < 100).view(-1, 1, 1), gray >= q75.view(-1, 1, 1), mask)
    msum = mask.sum(dim=(1, 2)).clamp(min=1).float()
    bmean = (sub[:, 0] * mask).sum(dim=(1, 2)) / msum
    rmean = (sub[:, 2] * mask).sum(dim=(1, 2)) / msum
    r0 = rmean / bmean.clamp(min=1.0)
    rb_hi = float(cast_cfg.get("rb_max", 3.4))
    rb_lo = float(cast_cfg.get("rb_min", 1.3))
    rt = torch.tensor([params[i]["cast"]["rt"] for i in cast_idx], device=dev)
    pt = torch.tensor([params[i]["cast"]["pt"] for i in cast_idx], device=dev)
    rt_eff = torch.maximum(
        rt, torch.maximum(r0 * 1.02, torch.tensor(rb_lo, device=dev))
    )
    apply = r0 < rb_hi
    k = rt_eff / r0.clamp(min=1e-6)
    warmed = sub.clone()
    warmed[:, 1] *= torch.sqrt(k).view(-1, 1, 1)
    warmed[:, 2] *= k.view(-1, 1, 1)
    gray2 = 0.114 * warmed[:, 0] + 0.587 * warmed[:, 1] + 0.299 * warmed[:, 2]
    pl2 = torch.quantile(gray2.flatten(1), 0.85, dim=1)
    warmed = warmed * (pt / pl2.clamp(min=1.0)).view(-1, 1, 1, 1)
    xf[ci] = torch.where(apply.view(-1, 1, 1, 1), torch.clamp(warmed, 0, 255), sub)
    return xf


def gpu_augment_batch(resident, pages, plan, epoch, seed, cast_cfg, g0, g1, dev, gen):
    """Um lote de patches com crop/augment na GPU, a partir das
    paginas residentes: devolve (x, target, valid) float32 ja no
    device, prontos pro forward. Os parametros vem de plan_patch
    (mesmo rng do caminho CPU); os pixels nunca descem pra CPU."""
    idxs = list(range(g0, g1))
    params = [
        plan_patch(pages[plan[g]], patch_rng(seed, epoch, g), cast_cfg=cast_cfg)
        for g in idxs
    ]
    n = len(idxs)
    pidx = [int(plan[g]) for g in idxs]
    ys = [p["y"] for p in params]
    xs = [p["x"] for p in params]
    imgs, labs, valids = (resident["img"], resident["lab"], resident["valid"])
    x = torch.stack(
        [imgs[p, :, y : y + PATCH, xx : xx + PATCH] for p, y, xx in zip(pidx, ys, xs)]
    )
    lab = torch.stack(
        [labs[p, y : y + PATCH, xx : xx + PATCH] for p, y, xx in zip(pidx, ys, xs)]
    )
    val = torch.stack(
        [valids[p, y : y + PATCH, xx : xx + PATCH] for p, y, xx in zip(pidx, ys, xs)]
    )
    for k in (1, 2, 3):
        grp = [i for i, p in enumerate(params) if p["k"] == k]
        if grp:
            gi = torch.tensor(grp, device=dev)
            x[gi] = torch.rot90(x[gi], k, dims=(2, 3))
            lab[gi] = torch.rot90(lab[gi], k, dims=(1, 2))
            val[gi] = torch.rot90(val[gi], k, dims=(1, 2))
    for flag, d_x, d_l in (("flipud", 2, 1), ("fliplr", 3, 2)):
        grp = [i for i, p in enumerate(params) if p[flag]]
        if grp:
            gi = torch.tensor(grp, device=dev)
            x[gi] = torch.flip(x[gi], dims=(d_x,))
            lab[gi] = torch.flip(lab[gi], dims=(d_l,))
            val[gi] = torch.flip(val[gi], dims=(d_l,))
    xf = x.float()
    gain = torch.tensor([p["gain"] for p in params], device=dev).view(n, 1, 1, 1)
    gamma = torch.tensor([p["gamma"] for p in params], device=dev).view(n, 1, 1, 1)
    contrast = torch.tensor([p["contrast"] for p in params], device=dev).view(
        n, 1, 1, 1
    )
    offset = torch.tensor([p["offset"] for p in params], device=dev).view(n, 1, 1, 1)
    pl = torch.tensor([pages[p].paper_level for p in pidx], device=dev).view(n, 1, 1, 1)
    xf = xf * gain
    xf = 255.0 * torch.clamp(xf / 255.0, 0, 1) ** gamma
    xf = (xf - pl) * contrast + pl
    xf = xf + offset
    for i, p in enumerate(params):
        if p["noise_on"]:
            gen.manual_seed(_patch_noise_seed(seed, epoch, idxs[i]))
            xf[i] += (
                torch.randn((3, PATCH, PATCH), generator=gen, device=dev) * p["sigma"]
            )
    blur_idx = [i for i, p in enumerate(params) if p["blur"]]
    if blur_idx:
        kern = (
            torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=dev
            )
            / 16.0
        )
        kern = kern.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        bi = torch.tensor(blur_idx, device=dev)
        # borda reflect (== cv2 GaussianBlur, BORDER_REFLECT_101):
        # padding=1 zero escurecia os cantos ate 9/16 do valor
        # (max 0,4375 medido na prova de equivalencia, F55 diag).
        xf[bi] = F.conv2d(
            F.pad(torch.clamp(xf[bi], 0, 255), (1, 1, 1, 1), mode="reflect"),
            kern,
            groups=3,
        )
    cast_idx = [i for i, p in enumerate(params) if p["cast"] is not None]
    if cast_idx:
        xf = _gpu_warm_cast(xf, params, cast_idx, cast_cfg, dev)
    tgt = ((lab == 1) | (lab == 2)).float().unsqueeze(1)
    vv = val.float().unsqueeze(1)
    return torch.clamp(xf, 0, 255) / 255.0, tgt, vv


# ------------------------------------------------------------------
# Amostragem com workers + prefetch (frente 53).
#
# Diagnostico: na corrida completa da F48 a utilizacao de GPU ficou
# em 39,6% de media (mediana 39%, p90 83%) com passo constante de
# 0,455 s — cada lote era amostrado em serie no processo principal
# (sample_patch num generator Python) e a GPU esperava a CPU.
#
# Desenho: o patch de indice global g da epoca e e FUNCAO PURA de
# (seed, e, g) via patch_rng. Quem calcula (principal ou worker) e
# em que ordem nao muda o resultado — serial e paralelo produzem
# exatamente os mesmos patches (prova por hash no relatorio 58).
# Isso mata o modo de falha classico de seed+worker_id, em que
# workers com fluxos proprios repetem os mesmos patches.
#
# Transporte: threads com anel de lotes prealocados, sem pickle.
# Medido ponta a ponta nesta VM (amostragem + transporte +
# empilhamento; teto local de ~1,5 vCPU compartilhada): pool com
# pickle de float32 = 0,81–0,93x do serial (o IPC come o ganho);
# pool com memoria compartilhada = 1,36–1,48x; threads + anel =
# 1,54–1,64x. No Kaggle sao 4 vCPUs dedicadas (spec oficial da
# sessao T4 x2, com 29 GB de RAM): 3 workers amostrando + 1 nucleo
# pro principal empilhar e alimentar a GPU.
#
# O pool nasce UMA vez por corrida (o equivalente ao
# persistent_workers=True do DataLoader), nunca por epoca, e a
# janela de prefetch e limitada em lotes: memoria limitada, sem
# fila infinita. workers=0 (config "sample_workers" ou env
# DRAW_SAMPLE_WORKERS) volta ao caminho serial, pra fiacao e
# depuracao.
# ------------------------------------------------------------------
_PLAN_SALT = 0xFFFFFFFF  # entropia do rng do plano (idx nunca chega aqui)


def patch_rng(seed, epoch, idx):
    """RNG de UM patch: SeedSequence [seed, epoca, indice global
    do patch na epoca]. Reprodutivel por construcao e independente
    da identidade do worker — a ordem de chegada nao entra."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(epoch), int(idx)])
    )


def _resolve_sample_workers(cfg):
    """Workers da amostragem: config "sample_workers" > env
    DRAW_SAMPLE_WORKERS > default min(3, cpu-1). Varredura medida
    {2, 3, 4}: 2 e 3 empatam no teto desta VM e 4 ja degrada por
    troca de contexto; o default 3 e o split da sessao de 4 vCPUs
    (3 amostrando + 1 pro principal)."""
    v = cfg.get("sample_workers")
    if v is None:
        v = os.environ.get("DRAW_SAMPLE_WORKERS")
    if v is not None:
        return max(0, int(v))
    return max(0, min(3, (os.cpu_count() or 2) - 1))


class PatchFeed:
    """Fonte de lotes da epoca, na ordem do plano.

    plan: array de posicoes em pages, ja embaralhado pela epoca
    (embaralho derivado de (seed, epoca) — o mesmo plano nos dois
    caminhos). batches() gera (xs, ys, vs) empilhados; no caminho
    paralelo, os workers amostram os lotes seguintes enquanto o
    principal treina o atual (prefetch), escrevendo num anel de
    prefetch*lote patches: slot = indice global % tamanho do anel,
    reutilizado so depois que o lote dono foi copiado pra fora.
    """

    def __init__(
        self,
        pages,
        seed,
        cast_cfg,
        workers,
        prefetch,
        gpu=False,
        dev="cpu",
        resident=None,
    ):
        self.pages = pages
        self.seed = seed
        self.cast_cfg = cast_cfg
        self.workers = workers
        self.prefetch = max(1, int(prefetch))
        self.gpu = gpu
        self.dev = dev
        self.resident = resident
        self._gen = torch.Generator(device=dev) if gpu else None
        self._pool = (
            ThreadPoolExecutor(max_workers=workers) if workers > 0 and not gpu else None
        )

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _one(self, plan, epoch, g):
        return sample_patch(
            self.pages[plan[g]], patch_rng(self.seed, epoch, g), cast_cfg=self.cast_cfg
        )

    def batches(self, plan, epoch, batch):
        n = len(plan)
        nb = max(1, math.ceil(n / batch))
        if self.gpu:
            # caminho GPU (frente 55): serial na thread principal,
            # kernels lancados em sequencia; a fila CUDA e o
            # prefetch. Devolve tensores ja no device.
            for b in range(nb):
                g0 = b * batch
                yield gpu_augment_batch(
                    self.resident,
                    self.pages,
                    plan,
                    epoch,
                    self.seed,
                    self.cast_cfg,
                    g0,
                    min(n, g0 + batch),
                    self.dev,
                    self._gen,
                )
            return
        if self._pool is None:
            for b in range(nb):
                idx = range(b * batch, min(n, (b + 1) * batch))
                xs, ys, vs = zip(*(self._one(plan, epoch, g) for g in idx))
                yield np.stack(xs), np.stack(ys), np.stack(vs)
            return
        ring = self.prefetch * batch
        rx = np.empty((ring, 3, PATCH, PATCH), np.float32)
        rt = np.empty((ring, 1, PATCH, PATCH), np.float32)
        rv = np.empty((ring, 1, PATCH, PATCH), np.float32)

        def task(g):
            x, y, v = self._one(plan, epoch, g)
            s = g % ring
            rx[s], rt[s], rv[s] = x, y, v
            return g

        pending = {}

        def submit(b):
            if b < nb:
                pending[b] = [
                    self._pool.submit(task, g)
                    for g in range(b * batch, min(n, (b + 1) * batch))
                ]

        for b in range(min(self.prefetch, nb)):
            submit(b)
        for b in range(nb):
            futs = pending.pop(b)
            m = len(futs)
            bx = np.empty((m, 3, PATCH, PATCH), np.float32)
            by = np.empty((m, 1, PATCH, PATCH), np.float32)
            bv = np.empty((m, 1, PATCH, PATCH), np.float32)
            for k, f in enumerate(futs):
                f.result()
                s = (b * batch + k) % ring
                bx[k], by[k], bv[k] = rx[s], rt[s], rv[s]
            submit(b + self.prefetch)
            yield bx, by, bv


# ------------------------------------------------------------------
# Perda: a masked_loss da linhagem (BCE pos_weight=8 + Dice).
# ------------------------------------------------------------------
def masked_loss(logits, target, valid):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=torch.tensor(8.0, device=logits.device),
    )
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


# ------------------------------------------------------------------
# Inferencia em mosaico (o protocolo do app) para val/regua.
# ------------------------------------------------------------------
@torch.no_grad()
def page_prob(model, img_bgr, dev):
    model.eval()
    h, w = img_bgr.shape[:2]
    tiles, pos = [], []
    for y in range(0, h, PATCH):
        for x in range(0, w, PATCH):
            tile = img_bgr[y : y + PATCH, x : x + PATCH]
            pad_h, pad_w = PATCH - tile.shape[0], PATCH - tile.shape[1]
            if pad_h or pad_w:
                tile = np.pad(tile, [(0, pad_h), (0, pad_w), (0, 0)])
            tiles.append(tile.transpose(2, 0, 1).astype(np.float32) / 255.0)
            pos.append((y, x))
    logits = np.zeros((h, w), np.float32)
    for i in range(0, len(tiles), 16):
        batch = torch.from_numpy(np.stack(tiles[i : i + 16])).to(dev)
        out = model(batch).cpu().numpy()[:, 0]
        for j, (y, x) in enumerate(pos[i : i + 16]):
            hh, ww = min(PATCH, h - y), min(PATCH, w - x)
            logits[y : y + hh, x : x + ww] = out[j, :hh, :ww]
    return 1.0 / (1.0 + np.exp(-logits))


def ruler_temporal(model, pages, dev):
    """A regua da frente 29 sobre os pares temporais (GT = classe 2,
    mecanismo independente do w8): prob mediana e recall sobre o GT
    dos positivos; FP@0,7 fora do GT nos pares todos."""
    out = {"pairs": {}, "pos_gt_px": 0}
    meds, fps = [], 0
    all_vals = []
    for page in pages:
        prob = page_prob(model, page.img, dev)
        gt = (page.lab == 2) & page.valid
        n_gt = int(gt.sum())
        entry = {"gt_px": n_gt}
        if n_gt:
            vals = prob[gt]
            all_vals.append(vals)
            meds.append(float(np.median(vals)))
            out["pos_gt_px"] += n_gt
        fp_mask = (~gt) & page.valid & (page.lab != 2)
        fp = int((prob[fp_mask] > 0.7).sum())
        fps += fp
        entry["fp07_px"] = fp
        out["pairs"][page.pair_id] = entry
    if all_vals:
        # a frente 29 mede sobre o GT agregado dos positivos
        agg = np.concatenate(all_vals)
        out["median_prob_agg"] = float(np.median(agg))
        out["recall05_agg"] = float((agg > 0.5).mean())
        out["recall07_agg"] = float((agg > 0.7).mean())
    out["median_prob_per_pair_median"] = float(np.median(meds)) if meds else None
    out["fp07_total_px"] = fps
    return out


# ------------------------------------------------------------------
# Recibo de utilizacao: amostrador nvidia-smi em thread.
# Cadencia de 0,25 s (medido na v4 do kernel, 2026-11-10: com
# 2,0 s, um smoke de 4 passos rendeu 2 amostras, as duas nos
# intervalos entre passos — media 0,0% com a GPU trabalhando,
# e o portao do smoke reprovou uma corrida saudavel).
# ------------------------------------------------------------------
class UtilSampler:
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                util, mem = out.strip().splitlines()[0].split(",")
                self.samples.append(
                    {
                        "t": time.time() - T0,
                        "gpu_util_pct": float(util),
                        "gpu_mem_mib": float(mem),
                    }
                )
            except Exception:
                try:
                    import torch as _t

                    if _t.cuda.is_available():
                        self.samples.append(
                            {
                                "t": time.time() - T0,
                                "gpu_util_pct": None,
                                "gpu_mem_mib": _t.cuda.memory_allocated() / 2**20,
                            }
                        )
                except Exception:
                    pass
            self._stop.wait(0.25)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        utils = [
            s["gpu_util_pct"] for s in self.samples if s.get("gpu_util_pct") is not None
        ]
        mems = [s["gpu_mem_mib"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "mean_gpu_util_pct": float(np.mean(utils)) if utils else None,
            "peak_gpu_mem_mib": float(np.max(mems)) if mems else None,
            "samples": self.samples,
        }


class _StepProfiler:
    """Perfil do passo de GPU (frente 78): eventos CUDA por fase.

    Instrumentacao apenas: registra eventos CUDA assincronos ao
    redor de cada fase do passo (data / forward / loss / backward /
    optimizer); nenhum tensor, loss, gradiente, ordem de operacao
    ou semente muda — a matematica do passo e byte a byte a
    anterior, com o perfil ligado ou desligado (prova OFF). A
    sincronizacao acontece so no fim da epoca, fora do passo,
    quando as medias sao calculadas; o `elapsed_time` de cada par
    de eventos mede o tempo de GPU da fase. Com profile OFF (ou
    em CPU), nada e registrado e o passo e o de sempre.
    """

    PHASES = ("data", "forward", "loss", "backward", "optimizer")

    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self._pairs = {p: [] for p in self.PHASES}
        self._open = {}
        self.epochs = []

    def begin(self, phase):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._open[phase] = ev

    def end(self, phase):
        if not self.enabled:
            return
        ev0 = self._open.pop(phase, None)
        if ev0 is None:
            return
        ev1 = torch.cuda.Event(enable_timing=True)
        ev1.record()
        self._pairs[phase].append((ev0, ev1))

    def drop(self, phase):
        # descarta um evento de abertura sem par (fim do iterador)
        if not self.enabled:
            return
        self._open.pop(phase, None)

    def epoch_close(self, epoch, steps, wall_s):
        """Sincroniza (fora do passo), calcula as medias da epoca."""
        if not self.enabled:
            return None
        torch.cuda.synchronize()
        rec = {
            "epoch": epoch,
            "steps": steps,
            "wall_s_per_step": wall_s / max(1, steps),
        }
        total_ms = 0.0
        for p in self.PHASES:
            pairs = self._pairs[p]
            ms = [a.elapsed_time(b) for a, b in pairs]
            mean_ms = sum(ms) / len(ms) if ms else 0.0
            rec[p] = {"mean_ms": mean_ms, "n": len(ms)}
            total_ms += mean_ms
            self._pairs[p] = []
        rec["sum_phases_ms"] = total_ms
        rec["residual_ms"] = rec["wall_s_per_step"] * 1000.0 - total_ms
        self.epochs.append(rec)
        return rec

    def summary(self):
        """Media por fase sobre as epocas + participacao + ranking."""
        if not self.enabled or not self.epochs:
            return None
        n = len(self.epochs)
        out = {"epochs": n}
        for p in self.PHASES:
            out[p] = sum(e[p]["mean_ms"] for e in self.epochs) / n
        out["sum_phases_ms"] = sum(out[p] for p in self.PHASES)
        out["wall_ms_per_step"] = (
            sum(e["wall_s_per_step"] for e in self.epochs) / n * 1000.0
        )
        out["residual_ms"] = out["wall_ms_per_step"] - out["sum_phases_ms"]
        for p in self.PHASES:
            out[p + "_share_pct"] = (
                100.0 * out[p] / out["sum_phases_ms"] if out["sum_phases_ms"] else 0.0
            )
        out["ranking"] = sorted(self.PHASES, key=lambda p: -out[p])
        return out


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args, _unknown_args = (
        ap.parse_known_args()
    )  # notebook (papermill) injeta argv do Jupyter: -f, --HistoryManager...

    cfg = {}
    cfg_path = args.config
    if os.environ.get("F31_CFG_JSON"):
        cfg = json.loads(os.environ["F31_CFG_JSON"])
    elif cfg_path and os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    work = args.out_dir or os.environ.get("DRAW_WORK", "/kaggle/working")
    os.makedirs(work, exist_ok=True)
    smoke = bool(args.smoke or cfg.get("smoke"))
    if smoke:
        cfg = {
            **cfg,
            "epochs": int(cfg.get("smoke_epochs", 2)),
            "patches_per_page": int(cfg.get("smoke_patches_per_page", 8)),
            "max_train_pages": int(cfg.get("smoke_pages", 12)),
        }

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(
        f"torch {torch.__version__} device={dev} "
        f"{torch.cuda.get_device_name(0) if dev == 'cuda' else ''}"
    )
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    workers = _resolve_sample_workers(cfg)
    prefetch = max(1, int(cfg.get("sample_prefetch", 2)))

    # aumento de cast quente (aug_photometric, frente 51): default OFF
    cast_cfg = cfg.get("warm_cast_aug") or {}
    if not cast_cfg.get("enabled"):
        cast_cfg = None
    else:
        log(
            f"aumento fotometrico de cast quente (aug_photometric): "
            f"ON prob={cast_cfg.get('prob', 0.25)} "
            f"R/B em [{cast_cfg.get('rb_min', 1.3)}, "
            f"{cast_cfg.get('rb_max', 3.4)}], papel em "
            f"[{cast_cfg.get('paper_min', 70.0)}, "
            f"{cast_cfg.get('paper_max', 140.0)}]"
        )

    # aumento na GPU (frente 55): default OFF — sem a flag na
    # config, o caminho executado e byte a byte o antigo.
    gpu_cfg = cfg.get("gpu_augment") or {}
    gpu_on = bool(gpu_cfg.get("enabled"))
    if gpu_on:
        log(
            "gpu_augment: ON (frente 55) — paginas residentes na "
            "VRAM em uint8, crop/geometrico/fotometrico/cast em "
            "torch nativo; selecao (coordenadas tinta/escuro) segue "
            "na CPU via plan_patch; workers CPU ignorados"
        )
    if gpu_on:
        workers = 0

    root = find_package(args.data_dir)
    manifest = [json.loads(ln) for ln in open(f"{root}/manifest.jsonl")]
    sessions = json.load(open(f"{root}/sessions.json"))
    holdout_sessions = set(sessions.get("holdout", []))
    assert not any(e["session_id"] in holdout_sessions for e in manifest), (
        "pacote contem sessao selada — abortando"
    )
    train_sessions = set(cfg.get("train_sessions", sessions.get("train", [])))
    ruler_entries = [e for e in manifest if e["route"] == "temporal"]
    train_entries = [
        e
        for e in manifest
        if e["session_id"] in train_sessions and e["route"] != "temporal"
    ]
    log(
        f"pacote {root}: treino {len(train_entries)} pares "
        f"(sessoes {sorted(train_sessions)}), regua temporal "
        f"{len(ruler_entries)} pares (fora do treino)"
    )
    assert train_entries, "nenhum par de treino"
    assert not {e["pair_id"] for e in train_entries} & {
        e["pair_id"] for e in ruler_entries
    }, "regua vazou pro treino"

    pages = load_pages(root, train_entries)
    ruler_pages = load_pages(root, ruler_entries)
    # val-monitor: ultimas paginas da maior sessao de treino (mesma
    # sessao — sinal de sanidade, nao julgamento; declarado)
    n_val = int(cfg.get("val_monitor_pages", 10))
    by_session = {}
    for p in pages:
        by_session.setdefault(p.session_id, []).append(p)
    main_session = max(by_session, key=lambda s: len(by_session[s]))
    session_pages = sorted(by_session[main_session], key=lambda p: p.pair_id)
    val_pages = session_pages[-n_val:] if len(session_pages) > n_val + 5 else []
    train_pages = [p for p in pages if p not in val_pages]
    if cfg.get("max_train_pages"):
        train_pages = train_pages[: int(cfg["max_train_pages"])]
        val_pages = val_pages[:2]
    log(
        f"paginas: treino {len(train_pages)}, val-monitor "
        f"{len(val_pages)}, regua {len(ruler_pages)}"
    )

    input_bases = ["/kaggle/input", root]
    model, init_info = build_init_model(input_bases, work)
    model = model.to(dev)

    # gradient checkpointing (frente 77): ataque direto a causa
    # medida do pico 12,34 GiB do 256 (relatorio 76 §3 — ativacoes
    # salvas do backward por patch, nao temporarios do Inductor).
    # Os 7 blocos da U-Net passam a rodar checkpointados
    # (torch.utils.checkpoint, use_reentrant=False): as ativacoes
    # intermediarias de cada bloco sao descartadas e recalculadas
    # no backward. So-CUDA. Veredito da F77 (relatorio 77 §4/§5):
    # o checkpointing NAO vira padrao — default OFF desde a frente
    # 78 (o corpo 87d896c trazia default ON; a F78 corrige o
    # default para o padrao decidido); ligar e decisao de desenho
    # de outra frente ("grad_ckpt": true na config). A flag e
    # ligada no modelo CRU, antes do compile, para o dynamo
    # guardar o valor. Em CPU o caminho segue byte a byte o
    # anterior (prova OFF).
    grad_ckpt_on = dev == "cuda" and bool(cfg.get("grad_ckpt", False))
    if grad_ckpt_on:
        model.grad_ckpt = True
    log(
        f"grad_ckpt: {'ON' if grad_ckpt_on else 'OFF'} (frente 77; "
        "blocos da U-Net checkpointados, use_reentrant=False; "
        "so-CUDA, default OFF desde a frente 78 — veredito F77; "
        'ligar com "grad_ckpt": true na config)'
    )

    # channels_last (frente 79): o perfil da F78 mediu o backward
    # como 63,2% do passo (246,24 ms) — o gradiente das convolucoes
    # (convs + transpostas) com ativacoes salvas em NCHW. O formato
    # channels_last (NHWC) melhora a localidade de memoria e o uso
    # dos Tensor Cores no backward em CUDA. A variavel da frente:
    # modelo convertido para channels_last E lotes de entrada
    # convertidos no passo, nas duas fases, ANTES do torch.compile
    # (o dynamo guarda o formato). So-CUDA, default OFF
    # ("channels_last": true na config liga). Em CPU o caminho
    # segue byte a byte o anterior (prova OFF). A sonda e o passo
    # usam o mesmo formato, para a sonda medir o pico do passo real.
    channels_last_on = dev == "cuda" and bool(cfg.get("channels_last", False))
    if channels_last_on:
        model = model.to(memory_format=torch.channels_last)
    log(
        f"channels_last: {'ON' if channels_last_on else 'OFF'} (frente 79; "
        "modelo U-Net + lotes de entrada em NHWC, antes do compile; "
        'so-CUDA, default OFF — ligar com "channels_last": true na config)'
    )

    # perfil do passo de GPU (frente 78): instrumentacao apenas —
    # eventos CUDA por fase (data / forward / loss / backward /
    # optimizer), em smoke E medicao. Nao muda a matematica do
    # passo (prova OFF); so-CUDA, default OFF ("profile_step":
    # true na config liga). Uma linha de recibo por fase por epoca
    # + bloco-resumo no fim de cada fase.
    profile_on = dev == "cuda" and bool(cfg.get("profile_step", False))
    log(
        f"perfil passo: {'ON' if profile_on else 'OFF'} (frente 78; "
        "eventos CUDA por fase: data/forward/loss/backward/optimizer; "
        "instrumentacao apenas, sem mudanca na matematica)"
    )

    # torch.compile no passo de GPU (frente 76): aplicado ANTES da
    # sonda (a sonda mede o pico do passo COMPILADO) e ANTES do
    # otimizador — o warmup do compile cai no 1o passo da sonda.
    # So-CUDA: em CPU o status fica "off" e o modelo segue eager,
    # byte a byte.
    compile_status, compile_reason = "off", None
    if dev == "cuda":
        model, compile_status, compile_reason = _apply_compile(model)

    # sonda de lote: maior lote da lista que cabe em <=80% da VRAM
    # (frente 76: o 256 VOLTA a lista — a F75 baixou o pico de VRAM
    # da medicao (6.499 MiB vs 6.975 da F73) e a pergunta aberta do
    # relatorio 74 §6 / relatorio 75 §6 e se o 256 cabe DENTRO do
    # teto com o passo compilado; o eager rejeitou a 12,34 GiB >
    # teto de 11,68 GiB. A sonda mede o pico por candidato com
    # torch.cuda.max_memory_allocated() + reset apos cada leitura;
    # o 1o candidato da lista compila o grafo no 1o passo.)
    batch = int(cfg.get("batch_size", 32))
    if dev == "cuda" and cfg.get("batch_probe", True):
        total = torch.cuda.get_device_properties(0).total_memory
        for cand in (256, 128, 96, 64, 48, 32, 16):
            try:
                xb = torch.rand(cand, 3, PATCH, PATCH, device=dev)
                if channels_last_on:
                    xb = xb.to(memory_format=torch.channels_last)
                yb = torch.rand(cand, 1, PATCH, PATCH, device=dev)
                vb = torch.ones_like(yb)
                model.train()
                loss = masked_loss(model(xb), yb, vb)
                loss.backward()
                model.zero_grad(set_to_none=True)
                used = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                del xb, yb, vb, loss
                torch.cuda.empty_cache()
                log(
                    f"sonda lote {cand}: pico {used / 2**30:.2f} GiB "
                    f"de {total / 2**30:.1f}"
                )
                if used <= 0.80 * total:
                    batch = cand
                    break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
    log(f"batch efetivo: {batch}")

    # Frente 82 (opcao (b) do relatorio 80 §6): reset do dynamo +
    # liberacao explicita entre a sonda e o treino — UMA variavel
    # contra a F81 (o wrapper split-process segue identico).
    reset_info = _reset_dynamo_after_probe(dev)

    epochs = int(cfg.get("epochs", 20))
    pp = int(cfg.get("patches_per_page", 48))
    lr = float(cfg.get("lr", 2e-4))
    steps_per_epoch = max(1, math.ceil(len(train_pages) * pp / batch))
    total_steps = steps_per_epoch * epochs
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=float(cfg.get("wd", 1e-4))
    )
    warmup = max(10, total_steps // 20)

    def lr_at(step):
        if step < warmup:
            return lr * step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 2e-5 + 0.5 * (lr - 2e-5) * (1 + math.cos(math.pi * t))

    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics = {
        "recipe": "finetune_real w8 + pares reais drawfmt",
        "run_kind": "smoke" if smoke else "full",
        "planned_epochs": epochs,
        "init": init_info,
        "train_pairs": [p.pair_id for p in train_pages],
        "val_monitor_pairs": [p.pair_id for p in val_pages],
        "ruler_pairs": [p.pair_id for p in ruler_pages],
        "batch": batch,
        "history": [],
        "dynamo_reset": reset_info or {"enabled": False},
        "sample_workers": workers,
        "sample_prefetch": prefetch,
        "warm_cast_aug": cast_cfg or {"enabled": False},
        "gpu_augment": ({"enabled": True} if gpu_on else {"enabled": False}),
    }

    def save_metrics():
        with open(f"{work}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    resident = None
    if gpu_on:
        resident = build_resident(train_pages, dev)
        log(
            f"gpu_augment: paginas residentes {resident['pages']} "
            f"({resident['shape']}) = {resident['mib']:.0f} MiB em "
            f"uint8 no {dev}; H2D por lote ~5 KB (parametros)"
        )
        metrics["gpu_augment"].update(
            {"resident_mib": resident["mib"], "resident_pages": resident["pages"]}
        )
    feed = PatchFeed(
        train_pages,
        seed,
        cast_cfg,
        workers,
        prefetch,
        gpu=gpu_on,
        dev=dev,
        resident=resident,
    )
    feed_desc = (
        "GPU (frente 55)"
        if gpu_on
        else "workers %d, prefetch %d lotes (workers 0 = serial)" % (workers, prefetch)
    )
    log(f"amostragem de patches: {feed_desc}; rng por (seed, epoca, indice global)")
    sampler = UtilSampler()
    sampler.start()
    best_dice = -1.0
    step = 0
    t_start = time.time()
    prof = _StepProfiler(profile_on)
    try:
        for epoch in range(epochs):
            model.train()
            plan = np.repeat(np.arange(len(train_pages)), pp)
            np.random.default_rng(
                np.random.SeedSequence([seed, epoch, _PLAN_SALT])
            ).shuffle(plan)
            ep_loss, ep_steps, t_ep = 0.0, 0, time.time()
            # O laco e um iterador explicito (frente 78): a fase
            # "data" comeca ANTES do next() — a amostragem do lote
            # (plan_patch na CPU + gpu_augment_batch na GPU) e
            # lazily executada dentro do next() do gerador — e
            # termina depois das transferencias H2D. As demais
            # fases cercam forward / loss / backward / optimizer;
            # a separacao forward|loss exige o split
            # `logits = model(xb)` + `loss = masked_loss(logits,
            # yb, vb)` em vez da expressao unica anterior — mesma
            # computacao, mesma ordem, sem mudanca na matematica
            # (prova OFF: loss e gradientes byte-identicos).
            batches = iter(feed.batches(plan, epoch, batch))
            while True:
                prof.begin("data")
                try:
                    xs, ys, vs = next(batches)
                except StopIteration:
                    prof.drop("data")
                    break
                if torch.is_tensor(xs):
                    xb, yb, vb = (
                        xs.to(dev, non_blocking=True),
                        ys.to(dev, non_blocking=True),
                        vs.to(dev, non_blocking=True),
                    )
                else:
                    xb = torch.from_numpy(xs).to(dev, non_blocking=True)
                    yb = torch.from_numpy(ys).to(dev, non_blocking=True)
                    vb = torch.from_numpy(vs).to(dev, non_blocking=True)
                if channels_last_on:
                    xb = xb.to(memory_format=torch.channels_last)
                prof.end("data")
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad(set_to_none=True)
                prof.begin("forward")
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                prof.end("forward")
                prof.begin("loss")
                loss = masked_loss(logits, yb, vb)
                prof.end("loss")
                prof.begin("backward")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                prof.end("backward")
                prof.begin("optimizer")
                scaler.step(opt)
                scaler.update()
                prof.end("optimizer")
                ep_loss += float(loss.detach())
                ep_steps += 1
                step += 1
                if ep_steps == 1 or ep_steps % 20 == 0:
                    log(
                        f"ep {epoch} passo {ep_steps}/{steps_per_epoch} "
                        f"loss {ep_loss / ep_steps:.4f} "
                        f"lr {opt.param_groups[0]['lr']:.2e}"
                    )
            # recibo do perfil da epoca (frente 78): uma linha por
            # fase, medias em ms/passo; o residuo e o tempo de
            # passo fora das 5 fases (sync do float(loss), logs).
            prec = prof.epoch_close(epoch, ep_steps, time.time() - t_ep)
            if prec is not None:
                log(
                    "perfil passo ep %(epoch)d: %(fases)s | soma "
                    "%(soma).2f ms | passo %(passo).2f ms | residuo "
                    "%(residuo).2f ms"
                    % {
                        "epoch": epoch,
                        "fases": " | ".join(
                            f"{p} {prec[p]['mean_ms']:.2f} ms"
                            for p in _StepProfiler.PHASES
                        ),
                        "soma": prec["sum_phases_ms"],
                        "passo": prec["wall_s_per_step"] * 1000.0,
                        "residuo": prec["residual_ms"],
                    }
                )
                metrics.setdefault("step_profile", {}).setdefault("epochs", []).append(
                    prec
                )
            # val-monitor: Dice@0,5 nas paginas separadas
            dice = None
            if val_pages:
                inter = union = 0
                for page in val_pages:
                    prob = page_prob(model, page.img, dev)
                    pred = (prob > 0.5) & page.valid
                    gt = ((page.lab == 1) | (page.lab == 2)) & page.valid
                    inter += int((pred & gt).sum())
                    union += int((pred | gt).sum())
                dice = (2 * inter) / max(1, inter + union)
            rec = {
                "epoch": epoch,
                "loss": ep_loss / max(1, ep_steps),
                "steps": ep_steps,
                "seconds": time.time() - t_ep,
                "val_monitor_dice": dice,
            }
            metrics["history"].append(rec)
            log(f"ep {epoch} fim: {rec}")
            torch.save(model.state_dict(), f"{work}/last.pt")
            if dice is not None and dice > best_dice:
                best_dice = dice
                torch.save(model.state_dict(), f"{work}/best.pt")
                metrics["best_epoch"] = epoch
            save_metrics()
    finally:
        feed.close()
        util = sampler.stop()
        with open(f"{work}/util.json", "w") as f:
            json.dump(util, f, indent=2)
        metrics["util"] = {
            k: util[k] for k in ("n_samples", "mean_gpu_util_pct", "peak_gpu_mem_mib")
        }
        metrics["train_seconds"] = time.time() - t_start
        save_metrics()
    log(f"utilizacao: {metrics['util']}")

    # bloco-resumo do perfil do passo (frente 78): media por fase
    # sobre as epocas + participacao + ranking — a DECISAO da F78
    # (qual fase domina o passo) sai daqui. O overhead da
    # instrumentacao se le comparando o s/epoca em regime com o
    # denominador F75 (69,51 s) / F76 (67,82 s).
    summ = prof.summary()
    if summ is not None:
        metrics.setdefault("step_profile", {})["summary"] = summ
        save_metrics()
        log(
            f"PERFIL PASSO (resumo {'smoke' if smoke else 'medicao'}): "
            f"media por epoca, {summ['epochs']} epocas, passo "
            f"{summ['wall_ms_per_step']:.2f} ms "
            f"(soma das fases {summ['sum_phases_ms']:.2f} ms, "
            f"residuo {summ['residual_ms']:.2f} ms)"
        )
        for p in _StepProfiler.PHASES:
            log(
                f"perfil passo resumo: {p} {summ[p]:.2f} ms "
                f"({summ[p + '_share_pct']:.1f}% do passo)"
            )
        log(f"perfil passo ranking: {' > '.join(summ['ranking'])}")

    # recibo do compile (frente 75): UMA linha por fase, depois de
    # passos reais — o status final reflete fallback tardio do
    # dynamo. "engaged" = compilado e sem recusa de frame;
    # "fallback" = eager (motivo junto); "off" = CPU, eager.
    # Fallback NUNCA falha a corrida, mas o veredito trata
    # fallback/off como INCONCLUSIVO (sem adocao, sem re-julgar a F73).
    compile_status, compile_reason = _compile_runtime_status(
        compile_status, compile_reason
    )
    metrics["compile_status"] = compile_status
    metrics["compile_mode"] = _COMPILE_MODE
    metrics["compile_fallback_reason"] = compile_reason
    metrics["grad_ckpt"] = "on" if grad_ckpt_on else "off"
    metrics["channels_last"] = "on" if channels_last_on else "off"
    save_metrics()
    detalhe = f" (modo {_COMPILE_MODE}, torch.compile{'; motivo: ' + compile_reason if compile_reason else ''})"
    log(f"compile_status: {compile_status}{detalhe}")

    # export: melhor (se houve val) senao o ultimo; BN dobrado pro
    # contrato do app (grafo Conv+Relu como o do w8)
    if os.path.exists(f"{work}/best.pt"):
        model.load_state_dict(
            torch.load(f"{work}/best.pt", map_location="cpu", weights_only=True)
        )
    model = model.to(dev).eval()
    fold = _fold_from_unet(model)
    onnx_out = f"{work}/page_clean_finetune_real_v1.onnx"
    _export_onnx(fold, onnx_out, dev)
    metrics["onnx"] = {"path": onnx_out}
    try:
        import onnxruntime as ort

        rng2 = np.random.default_rng(7)
        xt = rng2.random((1, 3, PATCH, PATCH)).astype(np.float32)
        with torch.no_grad():
            ref = fold(torch.from_numpy(xt).to(dev)).cpu().numpy()
        sess = ort.InferenceSession(onnx_out, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"page": xt})[0]
        metrics["onnx"]["max_abs_diff_torch_vs_ort"] = float(np.abs(ref - got).max())
        log(f"export ok: {metrics['onnx']}")
    except Exception as e:  # noqa: BLE001
        metrics["onnx"]["agreement_error"] = str(e)
        log(f"export sem conferencia ORT: {e}")
    save_metrics()

    ruler = ruler_temporal(model, ruler_pages, dev)
    metrics["ruler_temporal"] = ruler
    save_metrics()
    log(f"regua temporal (candidato): {json.dumps(ruler)[:400]}")
    w8_note = {
        "median_prob_agg": 0.0040,
        "recall05_agg": 0.12,
        "fonte": "frente 29 (w8 de producao sobre o mesmo GT)",
    }
    log(f"regua temporal (w8, referencia): {w8_note}")
    if smoke:
        metrics["smoke_finished"] = True
        save_metrics()
        ok = metrics["onnx"].get("max_abs_diff_torch_vs_ort", 9) < 1e-3 and (
            dev != "cuda"
            or (util["n_samples"] > 0 and (util["mean_gpu_util_pct"] or 0) > 0)
        )
        print(
            f"SMOKE {'OK' if ok else 'FALHOU'}: {json.dumps(metrics['util'])}",
            flush=True,
        )
        return 0 if ok else 1
    metrics["finished"] = True
    save_metrics()
    log("corrida completa: finished")
    return 0


def _fold_from_unet(model):
    """Dobra o BN de cada Block nas convolucoes (eval): devolve o
    FoldUNet equivalente, o formato do ONNX de producao."""
    fold = FoldUNet(BASE)
    pairs = (
        (model.d1, fold.d1),
        (model.d2, fold.d2),
        (model.d3, fold.d3),
        (model.d4, fold.d4),
        (model.u3, fold.u3),
        (model.u2, fold.u2),
        (model.u1, fold.u1),
    )
    with torch.no_grad():
        for src, dst in pairs:
            for conv_s, bn_s, conv_d in (
                (src.c[0], src.c[1], dst.c1),
                (src.c[3], src.c[4], dst.c2),
            ):
                scale = bn_s.weight / torch.sqrt(bn_s.running_var + bn_s.eps)
                conv_d.weight.copy_(conv_s.weight * scale.view(-1, 1, 1, 1))
                conv_d.bias.copy_((conv_s.bias - bn_s.running_mean) * scale + bn_s.bias)
        fold.head.weight.copy_(model.head.weight)
        fold.head.bias.copy_(model.head.bias)
    return fold.to(next(model.parameters()).device).eval()


def _export_onnx(fold, path, dev):
    dummy = torch.zeros(1, 3, PATCH, PATCH, device=dev)
    try:
        torch.onnx.export(
            fold,
            dummy,
            path,
            input_names=["page"],
            output_names=["ink_logits"],
            dynamic_axes={"page": {2: "h", 3: "w"}},
            opset_version=17,
            dynamo=False,
        )
    except Exception:
        torch.onnx.export(
            fold,
            dummy,
            path,
            input_names=["page"],
            output_names=["ink_logits"],
            dynamic_axes={"page": {2: "h", 3: "w"}},
            opset_version=14,
            dynamo=False,
        )
    import onnx as onnx_lib

    m = onnx_lib.load(path)
    dims = m.graph.output[0].type.tensor_type.shape.dim
    dims[2].dim_value = PATCH
    dims[3].dim_value = PATCH
    if dims[2].HasField("dim_param"):
        dims[2].ClearField("dim_param")
    if dims[3].HasField("dim_param"):
        dims[3].ClearField("dim_param")
    onnx_lib.save(m, path)
    log(f"ONNX (BN dobrado, lote fixo 1, 256 fixo): {path}")


if __name__ == "__main__":
    raise SystemExit(main())
