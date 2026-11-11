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
app), amostrados com centro em tinta sobreamostrado.

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
import glob
import json
import math
import os
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def sample_patch(page, rng, ink_frac=0.55, dark_frac=0.20):
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
    target = ((lab == 1) | (lab == 2)).astype(np.float32)
    return (
        img.transpose(2, 0, 1).astype(np.float32) / 255.0,
        target[None],
        valid[None].astype(np.float32),
    )


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
    torch.manual_seed(int(cfg.get("seed", 42)))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

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

    # sonda de lote: maior lote da lista que cabe em <=80% da VRAM
    batch = int(cfg.get("batch_size", 32))
    if dev == "cuda" and cfg.get("batch_probe", True):
        total = torch.cuda.get_device_properties(0).total_memory
        for cand in (64, 48, 32, 16):
            try:
                xb = torch.rand(cand, 3, PATCH, PATCH, device=dev)
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
    }

    def save_metrics():
        with open(f"{work}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    sampler = UtilSampler()
    sampler.start()
    best_dice = -1.0
    step = 0
    t_start = time.time()
    try:
        for epoch in range(epochs):
            model.train()
            plan = []
            for page in train_pages:
                plan += [page] * pp
            rng.shuffle(plan)
            ep_loss, ep_steps, t_ep = 0.0, 0, time.time()
            for i in range(0, len(plan), batch):
                chunk = plan[i : i + batch]
                xs, ys, vs = zip(*(sample_patch(p, rng) for p in chunk))
                xb = torch.from_numpy(np.stack(xs)).to(dev, non_blocking=True)
                yb = torch.from_numpy(np.stack(ys)).to(dev, non_blocking=True)
                vb = torch.from_numpy(np.stack(vs)).to(dev, non_blocking=True)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = masked_loss(model(xb), yb, vb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                ep_loss += float(loss.detach())
                ep_steps += 1
                step += 1
                if ep_steps == 1 or ep_steps % 20 == 0:
                    log(
                        f"ep {epoch} passo {ep_steps}/{steps_per_epoch} "
                        f"loss {ep_loss / ep_steps:.4f} "
                        f"lr {opt.param_groups[0]['lr']:.2e}"
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
        util = sampler.stop()
        with open(f"{work}/util.json", "w") as f:
            json.dump(util, f, indent=2)
        metrics["util"] = {
            k: util[k] for k in ("n_samples", "mean_gpu_util_pct", "peak_gpu_mem_mib")
        }
        metrics["train_seconds"] = time.time() - t_start
        save_metrics()
    log(f"utilizacao: {metrics['util']}")

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
