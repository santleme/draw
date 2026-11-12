#!/usr/bin/env python3
"""Fine-tune de tinta (finetune_real) em TPU Kaggle via torch_xla (frente 54).

Port do corpo versionado ``training/finetune_real.py`` (commit 05595de:
warm_cast_aug da F51 + loader PatchFeed da F53) para TPU. Este modulo
NAO reimplementa o treino: ele reusa do corpo — UNet, masked_loss,
PatchFeed/patch_rng, build_init_model, _fold_from_unet, _export_onnx,
page_prob/ruler_temporal, o plano de epoca e a receita (lr, warmup,
cosine, patches por pagina) — e so muda o encanamento que o runtime
TPU exige. Mesma regra do ``training/train_tpu.py`` (rota T7, provada
na sessao TPU do Kaggle em 2026-11-04): o pai nunca toca em device
XLA; o treino vive em ``_mp_fn`` via ``xmp.spawn``.

O que muda (cada item com o motivo no codigo):

- Spawn em N cores (``xmp.spawn``, nprocs=8 no v3-8; o mundo real
  vem de ``xr.world_size()``). Cada rank treina a sua fatia do MESMO
  plano de epoca do corpo CUDA (``plan[rank::world]``) em lotes de
  ``batch_per_core`` (default 8) — lote global 64, igual a receita
  v6, passos/epoca 316 (CUDA: 317; a diferenca e a cauda descartada,
  32 patches/epoca = 0,16%, ver ``_shard_plan``).
- Sem sonda de lote CUDA e sem GradScaler: lote fixo por core e
  ``torch.autocast(device_type="xla", dtype=torch.bfloat16)`` — bf16
  nativo da TPU; GradScaler existe pra fp16 nao estourar em underflow
  e e desnecessario em bf16 (docs PyTorch/XLA AMP, r2.6).
- Sem ``float(loss)`` a cada passo: o corpo CUDA sincroniza host a
  cada passo pra logar (barato em CUDA, fatal em XLA — cada sync
  executa o grafo lazy e quebra a fusao). Aqui a perda acumula como
  tensor XLA e so materializa via ``xm.mesh_reduce`` a cada 20
  passos e no fim da epoca (o mesmo ``mesh_reduce`` do train_tpu.py).
- BatchNorm por core: cada core normaliza o seu lote de 8 (no CUDA
  era um lote unico de 64). Divergencia de receita conhecida e
  registrada — nao ha SyncBatchNorm na API publica do torch_xla; o
  juiz (regua temporal in-kernel + f31) e quem diz se mordeu.
- Val-monitor so no rank 0, com lotes de tamanho FIXO
  (``_page_prob_fixed``): o ``page_prob`` do corpo deixa a ultima
  leva de tiles com tamanho variavel, e cada tamanho novo e um grafo
  novo pro XLA compilar (ate ~10 compilacoes extras por epoca). A
  leva final e completada repetindo o primeiro tile e os resultados
  do enchimento sao descartados — saida identica ao ``page_prob``
  (provado no ``--selftest``).
- Export ONNX em CPU no processo pai, depois do spawn (o mesmo
  padrao do train_tpu.py): tensores XLA tem que ir pra CPU antes de
  salvar/exportar (API guide, "Saving and Loading XLA Tensors").
- Recibo de utilizacao sem nvidia-smi: tempo por passo + tempo por
  epoca + ``xm.get_memory_info`` (HBM) + excerto do
  ``torch_xla.debug.metrics.metrics_report()`` (CompileTime /
  ExecuteTime). Nao existe "util %" de TPU acessivel — o s/epoca
  comparado com o T4 E a regua concordando e o portao.
- PatchFeed fica intacto (CPU/threads, amostragem nunca vai pra
  TPU). Workers por rank: config > env > default adaptativo (3 se
  o host tem >= 16 vCPUs, 0 = serial se menos — o host da TPU v3-8
  tem 96 vCPUs pela doc do Kaggle; se o host real for menor, o
  serial por rank ainda amostra so 1/8 dos patches).

Fontes (lidas em 2026-11-11, citadas no relatorio 59):
- kaggle.com/docs/tpu — TPU v3-8, 8 cores, 128 GB HBM, 20 h/semana,
  9 h por sessao; lote >= 8/core aceitavel; ``xmp.spawn(_mp_fn,
  nprocs=8, start_method='fork')``; a propria pagina avisa que foi
  escrita pra versoes mais antigas ("written for earlier TPU
  versions that Kaggle no longer supports") — o tipo real do host e
  o que o smoke logar (ver TBD-1).
- docs.pytorch.org/xla (AMP, r2.6) — ``autocast(xm.xla_device())``
  equivale a ``torch.autocast('xla')`` na TPU, ops em bf16.
- pytorch/xla master API_GUIDE.md — ``xm.optimizer_step`` =
  all_reduce dos gradientes + step + sync; ``xm.mark_step``;
  tensores XLA sao lazy; salvar so depois de ``.cpu()``.
- Memoria externa (2026-11-11): host v5e-8 com 224 vCPUs e medicao
  de fonte unica, a confirmar no smoke — NAO afirmado aqui.

Uso (sessao Kaggle com TPU, apos o preparo do kernel F54):
    python training/finetune_tpu.py --config <cfg.json> \
        [--data-dir <pacote>] [--out-dir <dir>] [--smoke]
    python training/finetune_tpu.py --selftest   # prova local em CPU
Sem --config, le a env F31_CFG_JSON (kernels), como o corpo CUDA.

Saidas em --out-dir (ou DRAW_WORK): init.pt (pesos do w8 no pai),
metrics.json incremental, last.pt / best.pt por epoca (rank 0),
util.json (recibo TPU), ONNX final no contrato do app
(page -> ink_logits, 256 fixo, BN dobrado), ruler_temporal.
Exit 0 so com SMOKE OK (smoke) ou finished + export conferido.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuso do corpo: nada aqui duplica arquitetura, perda, amostragem,
# init, export ou regua. O fallback cobre o kernel, onde os dois
# modulos sao escritos soltos em /kaggle/working/f54_port/ (sem o
# pacote `training` no path).
try:
    from training.finetune_real import (
        PATCH,
        UNet,
        _PLAN_SALT,
        _export_onnx,
        _fold_from_unet,
        build_init_model,
        find_package,
        load_pages,
        masked_loss,
        page_prob,
        PatchFeed,
        ruler_temporal,
    )
except ImportError:  # kernel: modulos soltos no mesmo diretorio
    from finetune_real import (
        PATCH,
        UNet,
        _PLAN_SALT,
        _export_onnx,
        _fold_from_unet,
        build_init_model,
        find_package,
        load_pages,
        masked_loss,
        page_prob,
        PatchFeed,
        ruler_temporal,
    )

T0 = time.time()
BATCH_PER_CORE_DEFAULT = 8  # doc do Kaggle: >= 8/core aceitavel
LOG_EVERY = 20  # passos entre syncs de log (o corpo CUDA loga a cada 20)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------
# Pedaços puros (prováveis localmente sem TPU, sem torch).
# ------------------------------------------------------------------
def _epoch_plan(n_pages, pp, seed, epoch):
    """O plano da epoca, byte a byte o do corpo CUDA: repeat das
    paginas x patches_por_pagina, embaralhado por rng (seed, epoca,
    _PLAN_SALT). Todos os ranks constroem o MESMO plano."""
    plan = np.repeat(np.arange(n_pages), pp)
    np.random.default_rng(np.random.SeedSequence([seed, epoch, _PLAN_SALT])).shuffle(
        plan
    )
    return plan


def _shard_plan(plan_len, rank, world, batch_per_core):
    """Fatia do rank: indices globais rank, rank+world, ... (strided
    — o plano ja vem embaralhado, entao strided e contiguo sao
    equivalentes em mistura, e o strided equilibra os ranks quando
    len(plan) nao e multiplo de world). Devolve (idx, passos,
    descartados): idx sao os indices GLOBAIS deste rank, na ordem
    do plano — o patch de indice g usa patch_rng(seed, epoca, g),
    entao o treino TPU ve EXATAMENTE os mesmos patches que o CUDA
    veria (mesmo plano, mesmo rng por indice), so distribuidos.
    Os passos sao o MINIMO entre os ranks (piso de len//world //
    lote): todos os ranks dao o mesmo numero de passos por epoca,
    em lockstep — sem isso o all-reduce do optimizer_step trava
    (um rank a mais que os outros). A cauda que nao fecha um lote
    e DESCARTADA (drop_last): sem ela, o ultimo lote de cada rank
    teria tamanho diferente e o XLA recompilaria o passo."""
    idx = np.arange(plan_len)[rank::world]
    steps = (plan_len // world) // batch_per_core
    dropped = len(idx) - steps * batch_per_core
    return idx, steps, dropped


def _lr_at(step, lr, warmup, total_steps):
    """Espelho exato do lr_at do corpo CUDA (warmup linear + cosine
    ate 2e-5). O step aqui e o passo GLOBAL (todos os ranks dao o
    mesmo numero de passos por epoca, em lockstep)."""
    if step < warmup:
        return lr * step / warmup
    t = (step - warmup) / max(1, total_steps - warmup)
    return 2e-5 + 0.5 * (lr - 2e-5) * (1 + math.cos(math.pi * t))


def _resolve_tpu_workers(cfg, world):
    """Workers da amostragem POR RANK: config > env > default
    adaptativo. O host da TPU v3-8 tem 96 vCPUs (doc do Kaggle) —
    8 ranks x 3 workers = 24 threads cabem com folga; num host de
    4 vCPUs (sessoes GPU) o default seria serial, porque 8 ranks
    x 3 threads disputariam 4 nucleos. Cada rank amostra so a sua
    fatia (1/8 dos patches), entao o serial por rank ja e 8x menos
    trabalho que o serial do corpo CUDA."""
    v = cfg.get("sample_workers")
    if v is None:
        v = os.environ.get("DRAW_SAMPLE_WORKERS")
    if v is not None:
        return max(0, int(v))
    ncpu = os.cpu_count() or 0
    return 3 if ncpu >= 16 else 0


class _RankFeed:
    """Prefetch por rank sobre o PatchFeed do corpo.

    Por que nao usar ``feed.batches`` direto: ele itera o plano
    inteiro com indices locais (0..n), e o rng do patch e funcao
    do indice GLOBAL — um rank que chamasse ``batches`` no seu
    sub-plano reusaria os rngs 0..k e quebraria a identidade
    (seed, epoca, indice global) provada na F53. Aqui o pool
    submete ``feed._one(plan, epoca, g)`` com o g GLOBAL, e os
    lotes saem na ordem do plano; o PatchFeed e construido com
    workers=0 (o pool dele nao e usado — o prefetch vive aqui).
    Janela de ``prefetch`` lotes a frente, como no corpo."""

    def __init__(self, feed, plan, epoch, idx, bpc, workers, prefetch):
        self.feed = feed
        self.plan = plan
        self.epoch = epoch
        self.idx = idx
        self.bpc = bpc
        self.workers = workers
        self.prefetch = max(1, int(prefetch))
        self._pool = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _one(self, g):
        return self.feed._one(self.plan, self.epoch, int(g))

    def batches(self, steps):
        if self._pool is None:
            for b in range(steps):
                gs = self.idx[b * self.bpc : (b + 1) * self.bpc]
                xs, ys, vs = zip(*(self._one(g) for g in gs))
                yield np.stack(xs), np.stack(ys), np.stack(vs)
            return
        pending = {}

        def submit(b):
            if b < steps:
                gs = self.idx[b * self.bpc : (b + 1) * self.bpc]
                pending[b] = [self._pool.submit(self._one, g) for g in gs]

        for b in range(min(self.prefetch, steps)):
            submit(b)
        for b in range(steps):
            futs = pending.pop(b)
            xs, ys, vs = zip(*(f.result() for f in futs))
            submit(b + self.prefetch)
            yield np.stack(xs), np.stack(ys), np.stack(vs)


# ------------------------------------------------------------------
# Val com lote fixo (evita recompilacao por tamanho de leva).
# ------------------------------------------------------------------
def _page_prob_fixed(model, img_bgr, dev, batch=16):
    """Mesma saida do page_prob do corpo, mas TODOS os forwards
    usam o mesmo shape (batch, 3, 256, 256): a ultima leva e
    completada repetindo o primeiro tile e os logits do enchimento
    sao descartados. Sem isso, cada tamanho de leva final (1..15)
    e um grafo novo pro XLA compilar — ate ~10 compilacoes por
    chamada de val-monitor, uma por epoca."""
    import numpy as np
    import torch

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
    with torch.no_grad():
        for i in range(0, len(tiles), batch):
            chunk = tiles[i : i + batch]
            n_real = len(chunk)
            if n_real < batch:
                chunk = chunk + [tiles[0]] * (batch - n_real)
            bt = torch.from_numpy(np.stack(chunk)).to(dev)
            out = model(bt).cpu().numpy()[:, 0]
            for j, (y, x) in enumerate(pos[i : i + batch]):
                hh, ww = min(PATCH, h - y), min(PATCH, w - x)
                logits[y : y + hh, x : x + ww] = out[j, :hh, :ww]
    return 1.0 / (1.0 + np.exp(-logits))


def _val_dice_fixed(model, val_pages, dev):
    """Dice@0,5 do val-monitor com lotes fixos (mesma conta do
    corpo CUDA, so troca page_prob por _page_prob_fixed)."""
    inter = union = 0
    for page in val_pages:
        prob = _page_prob_fixed(model, page.img, dev)
        pred = (prob > 0.5) & page.valid
        gt = ((page.lab == 1) | (page.lab == 2)) & page.valid
        inter += int((pred & gt).sum())
        union += int((pred | gt).sum())
    return (2 * inter) / max(1, inter + union)


def _xla_metrics_excerpt():
    """Excerto do relatorio de metricas XLA (CompileTime /
    ExecuteTime / contadores): a evidencia de utilizacao da TPU
    que existe. Guardado — se a API mudar de nome, o recibo traz
    o erro em vez de derrubar a corrida."""
    try:
        from torch_xla.debug import metrics as xla_metrics

        rep = xla_metrics.metrics_report()
        keep = [
            ln
            for ln in rep.splitlines()
            if re.search(
                r"CompileTime|ExecuteTime|TransferToDevice|"
                r"ExecuteReplicated|MarkStep",
                ln,
            )
        ]
        return {"compile_execute_lines": keep[:12], "report_chars": len(rep)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _xla_mem_mib(dev):
    """HBM do core (total/livre, MiB) via xm.get_memory_info.
    Guardado como o de metricas."""
    try:
        import torch_xla.core.xla_model as xm

        info = xm.get_memory_info(dev)
        return {
            "hbm_total_mib": float(info.get("kb_total", 0)) / 1024.0,
            "hbm_free_mib": float(info.get("kb_free", 0)) / 1024.0,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ------------------------------------------------------------------
# Costura XLA/CPU: 4 one-liners isolados; o resto do laco e comum.
# ------------------------------------------------------------------
class _Step:
    """Backend do passo: no XLA, optimizer_step (all-reduce dos
    gradientes + step + sync) e mesh_reduce pro log; na CPU, o
    equivalente direto. E o unico lugar que importa torch_xla."""

    def __init__(self, is_xla):
        self.is_xla = is_xla
        self.xm = None
        if is_xla:
            import torch_xla.core.xla_model as xm

            self.xm = xm

    def optimizer_step(self, opt):
        if self.is_xla:
            self.xm.optimizer_step(opt)
        else:
            opt.step()

    def mesh_mean(self, value, tag):
        """Media de `value` (tensor escalar 0-d) entre os ranks;
        na CPU, so materializa. Devolve float."""
        if self.is_xla:
            return float(self.xm.mesh_reduce(tag, value, lambda v: sum(v) / len(v)))
        return float(value.detach().cpu())

    def mesh_sum(self, value, tag):
        """Soma de `value` entre os ranks (p.ex. o dice, que so o
        rank 0 calcula e os demais contribuem 0)."""
        if self.is_xla:
            return float(self.xm.mesh_reduce(tag, value, sum))
        return float(value.detach().cpu())


# ------------------------------------------------------------------
# Treino (comum aos dois backends; o spawn so existe no XLA).
# ------------------------------------------------------------------
def _load_cfg(args):
    cfg = {}
    cfg_path = args.config
    if os.environ.get("F31_CFG_JSON"):
        cfg = json.loads(os.environ["F31_CFG_JSON"])
    elif cfg_path and os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    return cfg


def _split_package(cfg, data_dir):
    """Espelho do split do corpo CUDA (mesmos asserts: nenhuma
    sessao selada no pacote, regua temporal fora do treino)."""
    root = find_package(data_dir)
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
    return root, pages, train_pages, val_pages, ruler_pages


def _run(cfg, data_dir, out_dir, backend, rank=0, world=1, init_pt=None):
    """O treino. backend "xla" (dentro de _mp_fn, 1 rank por core)
    ou "cpu" (selftest local, sem spawn). Devolve metrics."""
    import numpy as np
    import torch

    is_xla = backend == "xla"
    step_be = _Step(is_xla)
    smoke = bool(cfg.get("smoke"))
    if smoke:
        cfg = {
            **cfg,
            "epochs": int(cfg.get("smoke_epochs", 2)),
            "patches_per_page": int(cfg.get("smoke_patches_per_page", 8)),
            "max_train_pages": int(cfg.get("smoke_pages", 12)),
        }
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if is_xla:
        import torch_xla.core.xla_model as xm

        dev = xm.xla_device()
        # prova de device (mesma do train_tpu.py): aborta se o core
        # nao responder — o corpo CUDA cairia pra CPU em silencio.
        assert float((torch.zeros(1, device=dev) + 1).cpu()) == 1.0
        try:
            hw = xm.xla_device_hw(dev)
        except Exception as e:  # noqa: BLE001
            hw = f"desconhecido ({e})"
        log(
            f"TPU rank {rank}/{world} device={dev} hw={hw} "
            f"torch {torch.__version__} host_cpus={os.cpu_count()}"
        )
    else:
        dev = "cpu"
        hw = "cpu"
        log(f"SELFTEST cpu torch {torch.__version__} host_cpus={os.cpu_count()}")

    cast_cfg = cfg.get("warm_cast_aug") or {}
    if not cast_cfg.get("enabled"):
        cast_cfg = None
    elif rank == 0:
        log(
            f"aumento fotometrico de cast quente (aug_photometric): "
            f"ON prob={cast_cfg.get('prob', 0.25)} "
            f"R/B em [{cast_cfg.get('rb_min', 1.3)}, "
            f"{cast_cfg.get('rb_max', 3.4)}], papel em "
            f"[{cast_cfg.get('paper_min', 70.0)}, "
            f"{cast_cfg.get('paper_max', 140.0)}]"
        )

    root, _pages, train_pages, val_pages, ruler_pages = _split_package(cfg, data_dir)

    # --- init no w8: no XLA o pai ja salvou init.pt (1 unfold);
    # na CPU o selftest constroi direto (mesma funcao do corpo).
    if init_pt and os.path.exists(init_pt):
        model = UNet(base=8, out_ch=1)
        sd = torch.load(init_pt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (
            f"init.pt nao casa strict: {missing} {unexpected}"
        )
        init_info = {"init_kind": "pt", "init_path": init_pt}
        if rank == 0:
            log(f"init: init.pt carregado strict de {init_pt}")
    else:
        model, init_info = build_init_model(["/kaggle/input", root], out_dir)
    model = model.to(dev)

    # --- lote: sem sonda CUDA; fixo por core, global = receita.
    bpc = int(cfg.get("batch_per_core", BATCH_PER_CORE_DEFAULT))
    batch_global = bpc * world
    if rank == 0:
        log(
            f"sonda de lote: TPU — sem sonda CUDA; lote fixo "
            f"{bpc}/core x {world} cores = {batch_global} global "
            f"(receita v6: 64)"
        )
        log(f"batch efetivo: {batch_global} ({bpc}/core x {world})")

    epochs = int(cfg.get("epochs", 20))
    pp = int(cfg.get("patches_per_page", 48))
    lr = float(cfg.get("lr", 2e-4))
    plan_len = len(train_pages) * pp
    _idx, steps_per_epoch, dropped = _shard_plan(plan_len, rank, world, bpc)
    total_steps = steps_per_epoch * epochs
    if rank == 0:
        log(
            f"passos/epoca: {steps_per_epoch} (CUDA: "
            f"{max(1, math.ceil(plan_len / batch_global))}; "
            f"descartados {dropped * world} patches/epoca por "
            f"drop_last nos ranks)"
        )
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=float(cfg.get("wd", 1e-4))
    )
    warmup = max(10, total_steps // 20)
    workers = _resolve_tpu_workers(cfg, world)
    prefetch = max(1, int(cfg.get("sample_prefetch", 2)))
    precision = str(cfg.get("tpu_precision", "bf16"))
    assert precision in ("bf16", "fp32"), precision
    if rank == 0:
        log(
            f"precisao TPU: {precision} "
            f"(autocast xla bf16; GradScaler nao se aplica em TPU)"
        )

    metrics = {
        "recipe": "finetune_tpu (finetune_real w8 + pares reais drawfmt, torch_xla)",
        "backend": backend,
        "run_kind": "smoke" if smoke else "full",
        "planned_epochs": epochs,
        "init": init_info,
        "train_pairs": [p.pair_id for p in train_pages],
        "val_monitor_pairs": [p.pair_id for p in val_pages],
        "ruler_pairs": [p.pair_id for p in ruler_pages],
        "batch_per_core": bpc,
        "batch_global": batch_global,
        "world": world,
        "steps_per_epoch": steps_per_epoch,
        "dropped_patches_per_epoch": dropped * world,
        "history": [],
        "sample_workers": workers,
        "sample_prefetch": prefetch,
        "tpu_precision": precision,
        "warm_cast_aug": cast_cfg or {"enabled": False},
    }
    if is_xla:
        import torch_xla

        metrics["torch"] = torch.__version__
        metrics["torch_xla"] = getattr(torch_xla, "__version__", "?")
        metrics["tpu_hw"] = hw

    def save_metrics():
        if rank == 0:
            with open(f"{out_dir}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    feed = PatchFeed(train_pages, seed, cast_cfg, 0, prefetch)
    if rank == 0:
        log(
            f"amostragem de patches: workers {workers} por rank, "
            f"prefetch {prefetch} lotes (workers 0 = serial); rng por "
            f"(seed, epoca, indice global)"
        )
    best_dice = -1.0
    step = 0
    t_start = time.time()
    s_per_step_windows = []
    try:
        for epoch in range(epochs):
            model.train()
            plan = _epoch_plan(len(train_pages), pp, seed, epoch)
            idx_shard, steps, _dr = _shard_plan(len(plan), rank, world, bpc)
            rfeed = _RankFeed(feed, plan, epoch, idx_shard, bpc, workers, prefetch)
            ep_loss_sum = None
            ep_loss_n = 0
            ep_steps = 0
            t_ep = time.time()
            t_win = time.time()
            win_steps = 0
            if precision == "bf16":
                autocast = torch.autocast(
                    device_type="xla" if is_xla else "cpu", dtype=torch.bfloat16
                )
            else:
                autocast = _nullcontext()
            for xs, ys, vs in rfeed.batches(steps):
                xb = torch.from_numpy(xs).to(dev)
                yb = torch.from_numpy(ys).to(dev)
                vb = torch.from_numpy(vs).to(dev)
                for g in opt.param_groups:
                    g["lr"] = _lr_at(step, lr, warmup, total_steps)
                opt.zero_grad(set_to_none=True)
                with autocast:
                    loss = masked_loss(model(xb), yb, vb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                step_be.optimizer_step(opt)
                # acumula SEM sync: a perda so materializa no
                # mesh_reduce do log (a cada 20 passos) — o corpo
                # CUDA faz float(loss) a cada passo. O acumulador
                # da epoca NAO zera no log: o mesh_reduce so le.
                ldet = loss.detach()
                ep_loss_sum = ldet if ep_loss_sum is None else ep_loss_sum + ldet
                ep_loss_n += 1
                ep_steps += 1
                step += 1
                win_steps += 1
                if ep_steps == 1 or ep_steps % LOG_EVERY == 0 or ep_steps == steps:
                    mean = step_be.mesh_mean(ep_loss_sum / ep_loss_n, f"loss_e{epoch}")
                    dt_win = time.time() - t_win
                    sps = dt_win / max(1, win_steps)
                    s_per_step_windows.append(sps)
                    if rank == 0:
                        log(
                            f"ep {epoch} passo {ep_steps}/{steps} "
                            f"loss {mean:.4f} "
                            f"lr {opt.param_groups[0]['lr']:.2e} "
                            f"({sps:.3f} s/passo na janela)"
                        )
                    t_win = time.time()
                    win_steps = 0
            rfeed.close()
            ep_loss = step_be.mesh_mean(ep_loss_sum / ep_loss_n, f"loss_e{epoch}_final")
            dice = None
            if val_pages:
                # so o rank 0 calcula; os demais contribuem 0 e o
                # mesh_reduce (soma) barra todo mundo ate o rank 0
                # acabar — nenhum rank avanca pra proxima epoca
                # antes do val terminar.
                d0 = _val_dice_fixed(model, val_pages, dev) if rank == 0 else 0.0
                dice = step_be.mesh_sum(
                    torch.tensor(d0, device=dev), f"val_dice_e{epoch}"
                )
            rec = {
                "epoch": epoch,
                "loss": ep_loss,
                "steps": ep_steps,
                "seconds": time.time() - t_ep,
                "val_monitor_dice": dice,
            }
            if rank == 0:
                metrics["history"].append(rec)
                log(f"ep {epoch} fim: {rec}")
                sd = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(sd, f"{out_dir}/last.pt")
                if dice is not None and dice > best_dice:
                    best_dice = dice
                    torch.save(sd, f"{out_dir}/best.pt")
                    metrics["best_epoch"] = epoch
                save_metrics()
    finally:
        feed.close()
    metrics["train_seconds"] = time.time() - t_start
    if rank == 0:
        util = {
            "backend": backend,
            "world": world,
            "batch_per_core": bpc,
            "batch_global": batch_global,
            "steps_per_epoch": steps_per_epoch,
            "s_per_step_median": (
                float(np.median(s_per_step_windows)) if s_per_step_windows else None
            ),
            "s_per_step_windows": [round(s, 4) for s in s_per_step_windows],
            "epochs": epochs,
        }
        if is_xla:
            util.update(_xla_mem_mib(dev))
            util["xla_metrics"] = _xla_metrics_excerpt()
        with open(f"{out_dir}/util.json", "w") as f:
            json.dump(util, f, indent=2)
        metrics["util"] = util
        save_metrics()
        log(
            f"utilizacao: {json.dumps({k: util[k] for k in util if k != 's_per_step_windows' and k != 'xla_metrics'})}"
        )
        if is_xla:
            log(f"xla_metrics: {json.dumps(util.get('xla_metrics', {}))[:600]}")
    return metrics


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ------------------------------------------------------------------
# Export + regua em CPU no processo pai (pos-spawn).
# ------------------------------------------------------------------
def _export_and_grade(cfg, data_dir, out_dir, smoke):
    """Carrega best.pt (ou last.pt), exporta o ONNX em CPU,
    confere com ORT e roda a regua temporal. Espelho do fim do
    main() do corpo CUDA — so que em CPU, porque tensores XLA
    nao exportam."""
    import numpy as np
    import torch

    metrics = json.load(open(f"{out_dir}/metrics.json"))
    _root, _pages, train_pages, val_pages, ruler_pages = _split_package(cfg, data_dir)
    ckpt = (
        f"{out_dir}/best.pt"
        if os.path.exists(f"{out_dir}/best.pt")
        else f"{out_dir}/last.pt"
    )
    model = UNet(base=8, out_ch=1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.eval()
    fold = _fold_from_unet(model)
    onnx_out = f"{out_dir}/page_clean_finetune_real_f54.onnx"
    _export_onnx(fold, onnx_out, "cpu")
    metrics["onnx"] = {"path": onnx_out}
    try:
        import onnxruntime as ort

        rng2 = np.random.default_rng(7)
        xt = rng2.random((1, 3, PATCH, PATCH)).astype(np.float32)
        with torch.no_grad():
            ref = fold(torch.from_numpy(xt)).numpy()
        sess = ort.InferenceSession(onnx_out, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"page": xt})[0]
        metrics["onnx"]["max_abs_diff_torch_vs_ort"] = float(np.abs(ref - got).max())
        log(f"export ok: {metrics['onnx']}")
    except Exception as e:  # noqa: BLE001
        metrics["onnx"]["agreement_error"] = str(e)
        log(f"export sem conferencia ORT: {e}")
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    ruler = ruler_temporal(model, ruler_pages, "cpu")
    metrics["ruler_temporal"] = ruler
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"regua temporal (candidato): {json.dumps(ruler)[:400]}")
    log(
        "regua temporal (w8, referencia): "
        '{"median_prob_agg": 0.0040, "recall05_agg": 0.12, '
        '"fonte": "frente 29 (w8 de producao sobre o mesmo GT)"}'
    )
    if smoke:
        metrics["smoke_finished"] = True
        hist = metrics.get("history", [])
        dices = [
            h.get("val_monitor_dice")
            for h in hist
            if h.get("val_monitor_dice") is not None
        ]
        ok = (
            metrics["onnx"].get("max_abs_diff_torch_vs_ort", 9) < 1e-3
            and bool(dices)
            and metrics["util"].get("s_per_step_median") is not None
            and "xla_metrics" in metrics["util"]
        )
        with open(f"{out_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(
            f"SMOKE {'OK' if ok else 'FALHOU'}: {json.dumps(metrics['util'])}",
            flush=True,
        )
        return 0 if ok else 1
    metrics["finished"] = True
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log("corrida completa: finished")
    return 0


# ------------------------------------------------------------------
# Spawn XLA.
# ------------------------------------------------------------------
def _mp_fn(rank, run_cfg, data_dir, out_dir):
    import torch_xla.runtime as xr

    world = xr.world_size()
    _run(
        run_cfg["cfg"],
        data_dir,
        out_dir,
        backend="xla",
        rank=rank,
        world=world,
        init_pt=run_cfg["init_pt"],
    )


def _preflight_tpu():
    """O pai nao importa torch_xla nem toca em device: o PJRT
    recusa o xmp.spawn se o runtime XLA ja foi inicializado
    neste processo (mesma nota do train_tpu.py). So le versoes."""
    if os.environ.get("TPU_PROCESS_ADDRESSES") == "local":
        os.environ.pop("TPU_PROCESS_ADDRESSES")
        os.environ.pop("CLOUD_TPU_TASK_ID", None)
        log("TPU_PROCESS_ADDRESSES=local removido (placeholder do Kaggle)")
    import torch

    log(f"torch {torch.__version__} host_cpus={os.cpu_count()}")
    try:
        import torch_xla

        log(f"torch_xla {getattr(torch_xla, '__version__', '?')}")
    except ImportError as e:
        raise SystemExit(f"torch_xla ausente neste runtime: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-per-core", type=int, default=None)
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="Prova local em CPU (sem TPU); nao treina a corrida.",
    )
    args, _unknown = ap.parse_known_args(argv)
    cfg = _load_cfg(args)
    work = args.out_dir or os.environ.get("DRAW_WORK", "/kaggle/working")
    os.makedirs(work, exist_ok=True)
    smoke = bool(args.smoke or cfg.get("smoke"))
    if args.batch_per_core:
        cfg["batch_per_core"] = args.batch_per_core

    if args.selftest:
        return _selftest(cfg, args.data_dir, work)

    _preflight_tpu()
    # --- init no w8 UMA vez no pai (CPU): 1 unfold do ONNX, os
    # 8 ranks carregam o mesmo init.pt (pesos identicos por
    # construcao, sem 8 downloads nem 8 unfolds).
    import torch

    root = find_package(args.data_dir)
    model, init_info = build_init_model(["/kaggle/input", root], work)
    init_pt = f"{work}/init.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, init_pt)
    log(f"init: {init_info} salvo em {init_pt} (pesos do w8, 1x)")
    del model
    run_cfg = {"cfg": cfg, "init_pt": init_pt}
    with open(f"{work}/tpu_run.json", "w") as f:
        json.dump(
            {
                "cfg_keys": sorted(cfg.keys()),
                "init": init_info,
                "batch_per_core": cfg.get("batch_per_core", BATCH_PER_CORE_DEFAULT),
                "smoke": smoke,
            },
            f,
            indent=2,
        )

    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(
        _mp_fn, args=(run_cfg, args.data_dir, work), nprocs=8, start_method="fork"
    )
    log("spawn encerrado")
    return _export_and_grade(cfg, args.data_dir, work, smoke)


def _selftest(cfg, data_dir, work):
    """Prova local em CPU: o laco REAL (mesmo _run, backend cpu)
    sobre o pacote sintetico da frente 53 + val de lote fixo ==
    page_prob + export + regua. Nao prova o XLA (spawn, optimizer
    step, mesh_reduce) — esses 4 one-liners sao auditoria estatica
    contra a doc (relatorio 59 §4) e o smoke do Kaggle."""
    import numpy as np
    import torch

    if not data_dir:
        raise SystemExit(
            "--selftest precisa de --data-dir (pacote sintetico da frente 53)"
        )
    cfg = {
        **cfg,
        "epochs": 2,
        "patches_per_page": 8,
        "max_train_pages": 12,
        "batch_per_core": 4,
        "val_monitor_pages": 10,
        "smoke": False,
    }
    metrics = _run(cfg, data_dir, work, backend="cpu")
    hist = metrics["history"]
    assert len(hist) == 2, hist
    assert hist[0]["steps"] > 0 and hist[1]["steps"] > 0
    log(f"SELFTEST loss ep0 {hist[0]['loss']:.4f} -> ep1 {hist[1]['loss']:.4f}")
    assert hist[1]["loss"] < hist[0]["loss"], hist
    # val de lote fixo == page_prob (mesma pagina, mesmo modelo):
    _root, _p, tp, _vp, _rp = _split_package(cfg, data_dir)
    model = UNet(base=8, out_ch=1)
    model.load_state_dict(
        torch.load(f"{work}/last.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    page = tp[0]
    a = page_prob(model, page.img, "cpu")
    b = _page_prob_fixed(model, page.img, "cpu")
    maxdiff = float(np.abs(a - b).max())
    log(f"SELFTEST page_prob vs _page_prob_fixed: max_abs_diff {maxdiff:.2e}")
    assert maxdiff == 0.0, maxdiff
    rc = _export_and_grade(cfg, data_dir, work, smoke=False)
    m2 = json.load(open(f"{work}/metrics.json"))
    diff = m2["onnx"].get("max_abs_diff_torch_vs_ort", 9)
    log(
        f"SELFTEST export diff {diff:.2e} ruler "
        f"{json.dumps(m2.get('ruler_temporal', {}))[:200]}"
    )
    assert diff < 1e-3, diff
    assert m2.get("ruler_temporal"), "regua ausente"
    log("SELFTEST_TPU_BODY_OK")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
