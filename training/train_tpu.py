"""Treino do SegFormer 4 classes do Draw em TPU (Kaggle, torch_xla).

Variante T7: decoder do SegFormer com C=128 (config: 256). O decoder
concentrava ~59% dos FLOPs do detector medidos na auditoria de custo;
esta corrida produz o candidato de decoder leve da rota unificada.

Este script NÃO reimplementa o treino: ele reusa o corpo versionado
``training/train.py`` (build_dataset, combined_loss,
warmup_cosine_scheduler, evaluate, export_onnx, validate_onnx e o
gate de entrega). Só muda o encanamento que o runtime TPU exige:

- device XLA via torch_xla, com abort explícito se não responder
  (o main() do corpo cai pra CPU em silêncio; aqui isso é proibido);
- ``xmp.spawn`` nos 8 cores da TPU v3-8 do Kaggle, com
  DistributedSampler + ParallelLoader + xm.optimizer_step;
- export/validação em CPU no processo pai, depois do spawn;
- checkpoints por época (last/best) + history.jsonl + receipt.json
  de utilização em --out-dir desde a primeira época.

Dados (plano explícito): geração sintética DENTRO da sessão com
``tools/gen_synthetic.py`` (split por fonte, manifesto validado pelos
guards de training/train.py). Sem --hands-dir, as mãos são placeholders
sintéticos do gerador — limitação registrada no metrics.json; a rota
de dataset anexado (guzzbr/draw-segformer-v1-4class) foi conferida e
não está acessível nesta conta. Contagens padrão: 4000/400/400
(train/val/test), seed 42. Medido nesta VM: ~0,12 s/cena de geração.

Dimensionamento (medido no smoke da sessão, 2026-11-04): passo em
regime = 0,21-0,22 s com batch 2 @256 num core fp32. A 512 a conta é
~4x por amostra -> ~0,43 s/amostra/core; batch 8/core dá ~3,4 s/passo
e ~63 passos/época (global 64), ou ~3,6 min/época e ~1,8 h pras
30 épocas — dentro do teto de 9 h da sessão (doc do Kaggle: TPU
20 h/semana, 9 h por sessão). Batch/core maior não sobe amostras/s
(o gargalo é computação) e 128/core da regra geral da doc não cabe
em HBM/core a 512x512 com estados do otimizador; 8/core é o piso da
doc e o ponto certo aqui. LR: 2e-4 (config traz 6e-5 pra batch 8 em
GPU; batch global 64 = 8x, LR sobe ~3,3x, entre sqrt e linear, com
o warmup+cosine do corpo).

Uso (um comando, na sessão Kaggle com TPU, após o preparo):
    python training/train_tpu.py

Saída em --out-dir (padrão /kaggle/working/out_t7):
    checkpoints/last.pt, checkpoints/best.pt, history.jsonl,
    receipt.json, draw_segmentation_v1.onnx, metrics.json,
    classes.json. Exit 0 só com a validação ONNX aprovada (mesmo
    gate do corpo); senão exit 3 com checkpoints preservados.

``--selftest`` roda a prova local em CPU (sem TPU): gera 8/2/2,
monta o modelo C=128, dá 1 passo de treino e exporta/valida.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _log(*a):
    print(*a, flush=True)


def _default_workdir():
    kaggle = Path("/kaggle/working")
    return kaggle if kaggle.is_dir() else Path.cwd() / "tpu_run"


def _preflight():
    # O pai NAO importa torch_xla nem toca em device: o PJRT recusa o
    # xmp.spawn se o runtime XLA ja foi inicializado neste processo
    # (pjrt.run_multiprocess). A prova de device vive dentro de _mp_fn.
    import importlib

    for mod in ("cv2", "numpy", "torch", "transformers", "albumentations"):
        m = importlib.import_module(mod)
        _log("MOD", mod, getattr(m, "__version__", "?"))
    try:
        import onnx
        import onnxruntime
    except ImportError:
        _log(
            "ERRO: onnx/onnxruntime ausentes neste runtime. "
            "Preparo: pip install onnx onnxruntime"
        )
        raise SystemExit(2)
    _log("MOD onnx", onnx.__version__, "| onnxruntime", onnxruntime.__version__)


def _ensure_data(data_dir, seed, n_train, n_val, n_test):
    data_dir = Path(data_dir)
    if (data_dir / "train").is_dir():
        _log("dados ja presentes em", data_dir)
        return
    t0 = time.time()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "gen_synthetic.py"),
        "--out",
        str(data_dir),
        "--seed",
        str(seed),
        "--train",
        str(n_train),
        "--val",
        str(n_val),
        "--test",
        str(n_test),
    ]
    _log("gerando dados:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    _log("gen rc", r.returncode, f"{time.time() - t0:.1f}s")
    _log(r.stdout[-300:])
    if r.returncode != 0:
        _log(r.stderr[-1500:])
        raise SystemExit("geracao sintetica falhou")


def _build_model_t7(cfg, decoder):
    """build_model do corpo, com o override T7 quando decoder != 256."""
    import training.train as T

    if decoder == 256:
        return T.build_model(cfg)
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(
        cfg["model"]["backbone"],
        num_labels=cfg["model"]["num_classes"],
        decoder_hidden_size=decoder,
        ignore_mismatched_sizes=True,
    )


def _mp_fn(rank, run_cfg, data_dir, out_dir):
    """Treino distribuído: 1 processo por core XLA."""
    import torch
    import torch.nn.functional as F
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp  # noqa: F401
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    import training.train as T

    dev = xm.xla_device()
    assert float((torch.zeros(1, device=dev) + 1).cpu()) == 1.0
    import torch_xla.runtime as xr

    world = xr.world_size()
    if rank == 0:
        _log("DEV_OK", dev, "| world:", world)
    torch.manual_seed(run_cfg["seed"])
    cfg = T.load_config(str(REPO_ROOT / "training" / "config.json"))
    train_ds = T.build_dataset(data_dir, "train", cfg, augment=True)
    val_ds = T.build_dataset(data_dir, "val", cfg, augment=False)
    sampler = DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, seed=run_cfg["seed"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=run_cfg["batch_per_core"],
        sampler=sampler,
        num_workers=2,
        drop_last=False,
    )
    # Seed repetido aqui: a cabeça nova nasce do RNG do torch, e todas as
    # réplicas precisam nascer com os mesmos pesos.
    torch.manual_seed(run_cfg["seed"])
    model = _build_model_t7(cfg, run_cfg["decoder"]).to(dev)
    tr = cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=run_cfg["lr"], weight_decay=tr["weight_decay"]
    )
    sched = T.warmup_cosine_scheduler(
        opt,
        total_steps=run_cfg["epochs"] * len(train_loader),
        warmup_ratio=tr["warmup_ratio"],
    )
    cw = torch.tensor(
        cfg["loss"]["cross_entropy_class_weights"], dtype=torch.float32, device=dev
    )
    amp = torch.bfloat16 if run_cfg["precision"] == "bf16" else None
    out_dir = Path(out_dir)
    best_iou = -1.0
    t_start = time.time()

    for epoch in range(run_cfg["epochs"]):
        sampler.set_epoch(epoch)
        para = pl.ParallelLoader(train_loader, [dev])
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for images, masks in para.per_device_loader(dev):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="xla", dtype=amp, enabled=amp is not None):
                logits = model(pixel_values=images).logits
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = T.combined_loss(logits, masks, cw)
            loss.backward()
            xm.optimizer_step(opt)
            sched.step()
            tot += float(loss.detach())
            nb += 1
            if rank == 0 and nb <= 3:
                _log(f"PROBE_STEP epoca0 passo{nb} acumulado {time.time() - t0:.1f}s")
        dt = time.time() - t0
        mean_loss = xm.mesh_reduce("loss", tot / max(nb, 1), lambda v: sum(v) / len(v))
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world, rank=rank, shuffle=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=run_cfg["batch_per_core"],
            sampler=val_sampler,
            num_workers=2,
        )
        metrics = T.evaluate(model, val_loader, dev)
        iou = xm.mesh_reduce("iou", metrics["mean_iou"], lambda v: sum(v) / len(v))
        if rank == 0:
            steps = max(nb, 1)
            rec = {
                "epoch": epoch,
                "train_loss": round(mean_loss, 4),
                "val_mean_iou_shards": round(iou, 4),
                "epoch_s": round(dt, 1),
                "s_per_step": round(dt / steps, 3),
                "samples_s_per_core": round(steps * run_cfg["batch_per_core"] / dt, 2),
                "samples_s_aggregate": round(
                    steps * run_cfg["batch_per_core"] * world / dt, 2
                ),
                "elapsed_s": round(time.time() - t_start, 1),
            }
            _log("EPOCA", json.dumps(rec))
            with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if epoch == 1:
                receipt = dict(rec)
                receipt["projected_epoch_s"] = rec["epoch_s"]
                receipt["projected_total_min"] = round(
                    rec["epoch_s"] * run_cfg["epochs"] / 60, 1
                )
                receipt["world"] = world
                receipt["batch_global"] = run_cfg["batch_per_core"] * world
                with open(out_dir / "receipt.json", "w", encoding="utf-8") as f:
                    json.dump(receipt, f, indent=2)
                _log("RECEIPT", json.dumps(receipt))
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(sd, out_dir / "checkpoints" / "last.pt")
            if iou > best_iou:
                best_iou = iou
                torch.save(sd, out_dir / "checkpoints" / "best.pt")
                _log("BEST", iou)
        xm.mark_step()
    if rank == 0:
        _log("TREINO_TPU_DONE")


def _export_and_grade(run_cfg, data_dir, out_dir):
    """Export + avaliação oficial em CPU (processo pai, pós-spawn)."""
    import torch
    from torch.utils.data import DataLoader

    import training.train as T

    cfg = T.load_config(str(REPO_ROOT / "training" / "config.json"))
    out_dir = Path(out_dir)
    model = _build_model_t7(cfg, run_cfg["decoder"])
    model.load_state_dict(
        torch.load(
            out_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True
        )
    )
    model = model.to("cpu")
    val_ds = T.build_dataset(data_dir, "val", cfg, augment=False)
    test_ds = T.build_dataset(data_dir, "test", cfg, augment=False)
    onnx_path = out_dir / cfg["export"]["onnx"]
    try:
        T.export_onnx(model, str(onnx_path), cfg)
        rep = T.validate_onnx(
            model,
            str(onnx_path),
            DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0),
            "cpu",
            cfg,
        )
    except Exception as exc:  # registrar sem perder o treino (regra do corpo)
        rep = {"status": "failed", "error": str(exc)}
        _log("AVISO: exportacao ONNX falhou:", exc)
    final = T.evaluate(model, DataLoader(val_ds, batch_size=8, num_workers=0), "cpu")
    final["test"] = T.evaluate(
        model, DataLoader(test_ds, batch_size=8, num_workers=0), "cpu"
    )
    final["onnx_validation"] = rep
    final["delivery_status"] = T.onnx_delivery_status(rep)
    final["tpu_run"] = run_cfg
    final["data_sources"] = {
        "synthetic_root": str(data_dir),
        "generator": "tools/gen_synthetic.py (split por fonte)",
        "hands": "placeholder sintetico do gerador (sem --hands-dir)",
        "counts": {
            "train": run_cfg["n_train"],
            "val": run_cfg["n_val"],
            "test": run_cfg["n_test"],
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (out_dir / "classes.json").write_text(
        json.dumps(T.CLASSES, indent=2), encoding="utf-8"
    )
    _log("ONNX", onnx_path, onnx_path.stat().st_size if onnx_path.exists() else -1)
    _log(
        "DELIVERY",
        final["delivery_status"],
        "| val_iou",
        final["mean_iou"],
        "| test_iou",
        final["test"]["mean_iou"],
    )
    return 0 if final["delivery_status"] == "validated" else 3


def _selftest():
    """Prova local em CPU: override T7 + 1 passo + export/gate."""
    import tempfile

    import torch
    from torch.utils.data import DataLoader

    import training.train as T

    _preflight_sem_xla()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "synthetic"
        _ensure_data(data_dir, seed=42, n_train=8, n_val=2, n_test=2)
        cfg = T.load_config(str(REPO_ROOT / "training" / "config.json"))
        model = _build_model_t7(cfg, 128)
        fuse = model.decode_head.linear_fuse.weight
        _log("linear_fuse (C=128):", tuple(fuse.shape))
        assert fuse.shape[0] == 128, fuse.shape
        ds = T.build_dataset(str(data_dir), "train", cfg, augment=False)
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        images, masks = next(iter(loader))
        opt = torch.optim.AdamW(model.parameters(), lr=6e-5)
        cw = torch.tensor(
            cfg["loss"]["cross_entropy_class_weights"], dtype=torch.float32
        )
        logits = model(pixel_values=images).logits
        if logits.shape[-2:] != masks.shape[-2:]:
            import torch.nn.functional as F

            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = T.combined_loss(logits, masks, cw)
        loss.backward()
        opt.step()
        _log("passo CPU ok, loss", float(loss))
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        onnx_path = out_dir / cfg["export"]["onnx"]
        T.export_onnx(model.to("cpu"), str(onnx_path), cfg)
        rep = T.validate_onnx(
            model,
            str(onnx_path),
            DataLoader(
                T.build_dataset(str(data_dir), "val", cfg, augment=False),
                batch_size=2,
                num_workers=0,
            ),
            "cpu",
            cfg,
        )
        _log("ONNX_VALIDATION", json.dumps(rep)[:300])
        _log("DELIVERY", T.onnx_delivery_status(rep))
        assert rep.get("status") == "ok", rep
    _log("SELFTEST_TPU_BODY_OK")
    return 0


def _preflight_sem_xla():
    import importlib

    # albumentations so entra com augment=True; o selftest nao aumenta,
    # entao a ausencia dele aqui e aviso, nao bloqueio.
    for mod in (
        "cv2",
        "numpy",
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "albumentations",
    ):
        try:
            m = importlib.import_module(mod)
            _log("MOD", mod, getattr(m, "__version__", "?"))
        except ImportError:
            if mod != "albumentations":
                raise
            _log("MOD FALTA albumentations (ok no selftest: sem augment)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Treino T7 (SegFormer 4 classes, decoder C=128) em TPU Kaggle."
    )
    parser.add_argument("--decoder", type=int, default=128)
    parser.add_argument("--batch-per-core", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Prova local em CPU (sem TPU); não treina a corrida.",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    # Sessao Kaggle grava TPU_PROCESS_ADDRESSES=local como placeholder e o
    # libtpu le ao pe da letra: 1 endereco pra 8 cores e a init multiprocesso
    # morre. Sem a variavel, o runtime descobre o slice local sozinho.
    if os.environ.get("TPU_PROCESS_ADDRESSES") == "local":
        os.environ.pop("TPU_PROCESS_ADDRESSES")
        os.environ.pop("CLOUD_TPU_TASK_ID", None)
        _log("TPU_PROCESS_ADDRESSES=local removido (placeholder do Kaggle)")

    work = _default_workdir()
    data_dir = Path(args.data_dir) if args.data_dir else work / "data" / "synthetic"
    out_dir = Path(args.out_dir) if args.out_dir else work / "out_t7"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "decoder": args.decoder,
        "batch_per_core": args.batch_per_core,
        "epochs": args.epochs,
        "lr": args.lr,
        "precision": args.precision,
        "seed": args.seed,
        "n_train": args.train_count,
        "n_val": args.val_count,
        "n_test": args.test_count,
    }
    _log("RUN_CFG", json.dumps(run_cfg))
    _preflight()

    _ensure_data(data_dir, args.seed, args.train_count, args.val_count, args.test_count)

    import training.train as T

    _, manifest_path, split_validation = T.load_split_manifest(str(data_dir))
    _log(
        "split manifest:",
        manifest_path,
        "| familias verificadas:",
        split_validation["checked_source_families"],
    )
    counts = T.pixel_counts(T.pair_images_masks(str(Path(data_dir) / "train")))
    _log("pixels treino por classe:", counts.tolist())

    # Aquece o cache do HF e prova o override T7 no pai, antes do spawn.
    cfg = T.load_config(str(REPO_ROOT / "training" / "config.json"))
    warm = _build_model_t7(cfg, args.decoder)
    _log(
        "modelo T7 ok; linear_fuse:",
        tuple(warm.decode_head.linear_fuse.weight.shape),
        "| params:",
        sum(p.numel() for p in warm.parameters()),
    )
    del warm

    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(
        _mp_fn,
        args=(run_cfg, str(data_dir), str(out_dir)),
        nprocs=None,
        start_method="fork",
    )
    _log("spawn encerrado")
    return _export_and_grade(run_cfg, str(data_dir), out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
