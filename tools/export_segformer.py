#!/usr/bin/env python3
"""Reexporta um checkpoint SegFormer sem retreinar e compara resoluções.

Exemplo (seleção por validação, com métricas projetadas de volta a 512):

    python tools/export_segformer.py \
        --checkpoint out/checkpoints/best.pt \
        --data-dir dataset/synthetic --size 256 --out-dir out/export_256

O teste sintético só entra com ``--include-test``, depois da escolha da
resolução na validação. O ONNX usa a resolução pedida; as métricas de
segmentação são calculadas no canvas de referência para que IoU e a
tolerância de tinta em pixels sejam comparáveis entre exports.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import (  # noqa: E402
    build_dataset,
    build_model,
    evaluate,
    export_onnx,
    load_config,
    onnx_delivery_status,
    validate_onnx,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_state(torch, checkpoint, device):
    # Só tensores: sem fallback para o unpickler completo.
    return torch.load(checkpoint, map_location=device, weights_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/config.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--size", type=int, required=True, help="Lado da entrada/export, ex.: 256"
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=None,
        help="Canvas das métricas; padrão: resolução da config original",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Avalia teste somente após a resolução ter sido escolhida na validação",
    )
    args = parser.parse_args(argv)

    base_cfg = load_config(args.config)
    reference_size = args.reference_size or base_cfg["model"]["input"]["height"]
    for label, size in (("--size", args.size), ("--reference-size", reference_size)):
        if size <= 0 or size % 32:
            parser.error(f"{label} precisa ser positivo e múltiplo de 32 (SegFormer)")
    if not Path(args.checkpoint).is_file():
        parser.error(f"checkpoint inexistente: {args.checkpoint}")

    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["input"]["height"] = args.size
    cfg["model"]["input"]["width"] = args.size
    cfg["model"]["output"]["shape"] = [
        1,
        cfg["model"]["num_classes"],
        args.size,
        args.size,
    ]
    reference_cfg = copy.deepcopy(base_cfg)
    reference_cfg["model"]["input"]["height"] = reference_size
    reference_cfg["model"]["input"]["width"] = reference_size

    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or base_cfg["training"]["batch_size"]
    workers = base_cfg["training"]["num_workers"]

    def loaders(split):
        input_ds = build_dataset(args.data_dir, split, cfg, augment=False)
        reference_ds = build_dataset(args.data_dir, split, reference_cfg, augment=False)
        if len(input_ds) != len(reference_ds):
            raise ValueError(
                f"split {split}: loaders de entrada/referência divergentes"
            )
        common = dict(batch_size=batch_size, shuffle=False, num_workers=workers)
        return DataLoader(input_ds, **common), DataLoader(reference_ds, **common)

    val_loader, val_reference = loaders("val")
    test_loader, test_reference = loaders("test") if args.include_test else (None, None)
    model = build_model(cfg).to(device)
    model.load_state_dict(_load_state(torch, args.checkpoint, device))
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"draw_segmentation_{args.size}px.onnx"
    try:
        export_onnx(model, str(onnx_path), cfg)
        onnx_check = validate_onnx(model, onnx_path, val_loader, device, cfg)
    except Exception as exc:  # noqa: BLE001 - métricas continuam disponíveis
        onnx_check = {"status": "failed", "error": str(exc)}

    metrics = evaluate(model, val_loader, device, target_loader=val_reference)
    if test_loader is not None:
        metrics["test"] = evaluate(
            model, test_loader, device, target_loader=test_reference
        )
    metrics["onnx_validation"] = onnx_check
    metrics["delivery_status"] = onnx_delivery_status(onnx_check)
    metrics["resolution_protocol"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "weights_retrained": False,
        "input_size": args.size,
        "metric_reference_size": reference_size,
        "test_evaluated": args.include_test,
        "selection_rule": "escolher resolução na validação; teste apenas depois da escolha",
        "note": "predições são reamostradas para o canvas de referência antes de IoU e tolerância de tinta",
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    print(
        f"input={args.size} | métricas@{reference_size} | "
        f"val_mean_iou={metrics['mean_iou']:.4f} | ONNX={onnx_check.get('status')}"
    )
    if metrics["delivery_status"] != "validated":
        print("ENTREGA BLOQUEADA pela validação ONNX", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
