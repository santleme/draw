#!/usr/bin/env python3
"""Fusão de pesos (WiSE-FT / model merging) entre checkpoints da linhagem.

Interpola dois checkpoints de MESMA arquitetura e MESMA bacia (mesma
inicialização — ex.: base t13 ou R2, alvo t14, ambos nascidos do modelo A
com a cabeça de presença da R2):

    theta_merged = (1 - alpha) * theta_base + alpha * theta_target

Motivação (2026-11-09): a t14 curou o flicker da máscara na bancada selada
(3,58 px, portão <= 10) mas regrediu a geometria em fundo hostil (bg05 IoU
0,902 < 0,918; canto 20,49 > 18,9 px) — cabo de guerra ID x OOD clássico de
fine-tuning. A interpolação no espaço de pesos tenta ficar com a folga de
estabilidade da t14 e a geometria da base, a custo zero de treino.

Regras:
- Os dois checkpoints precisam ter EXATAMENTE as mesmas chaves, formas e
  dtypes (verificado tensor a tensor; qualquer divergência aborta).
- Tensores float são interpolados (acúmulo em float64, cast de volta);
  tensores não-float (ex.: ``num_batches_tracked``) são copiados do alvo.
- O ONNX exportado usa o MESMO caminho de exportação do treino
  (``training.train.export_onnx``) e a config resolvida da corrida do alvo
  (``training_config.json`` salvo ao lado do best.pt) — contrato idêntico
  ao da entrega: entrada ``image``, saídas ``logits`` + ``presence``.
- Autochecagem: IoU de argmax PyTorch x ONNX em entrada aleatória tem de
  ser >= 0,99, senão o candidato é marcado inválido e o script falha.

Uso:
    python tools/merge_checkpoints.py \
        --base /caminho/base_best.pt --base-name t13 \
        --target /caminho/t14_best.pt --target-name t14 \
        --config /caminho/training_config.json \
        --alphas 0.50,0.65,0.75 --out-dir /caminho/saida
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_state_dict(path: Path):
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(f"{path}: checkpoint não é um state_dict ({type(sd)})")
    return sd


def check_compatible(base: dict, target: dict) -> None:
    if set(base) != set(target):
        missing = sorted(set(target) - set(base))
        extra = sorted(set(base) - set(target))
        raise ValueError(f"chaves divergentes: faltam {missing[:5]} sobram {extra[:5]}")
    for k in target:
        if base[k].shape != target[k].shape:
            raise ValueError(f"shape divergente em {k}: {base[k].shape} x {target[k].shape}")
        if base[k].dtype != target[k].dtype:
            raise ValueError(f"dtype divergente em {k}: {base[k].dtype} x {target[k].dtype}")


def interpolate(base: dict, target: dict, alpha: float) -> dict:
    merged = {}
    for k, t in target.items():
        if t.is_floating_point():
            merged[k] = ((1.0 - alpha) * base[k].double() + alpha * t.double()).to(t.dtype)
        else:
            merged[k] = t.clone()
    return merged


def argmax_iou_check(model, onnx_path: Path, cfg: dict) -> float:
    """IoU de argmax entre o modelo PyTorch (caminho de exportação) e o ONNX."""
    import onnxruntime as ort
    import torch

    from training.train import make_onnx_export_model

    size = cfg["model"]["input"]["height"]
    export_model = make_onnx_export_model(model, size, include_presence=True)
    torch.manual_seed(0)
    dummy = torch.randn(2, 3, size, size)
    with torch.no_grad():
        logits_t, _ = export_model(dummy)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits_o = session.run(None, {"image": dummy.numpy()})[0]
    a = logits_t.numpy().argmax(axis=1)
    b = logits_o.argmax(axis=1)
    ious = []
    for cls in range(a.shape[1] if a.ndim == 4 else 4):
        pass
    for cls in range(4):
        ma, mb = a == cls, b == cls
        union = (ma | mb).sum()
        ious.append(float((ma & mb).sum() / union) if union else 1.0)
    return min(ious)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--base-name", required=True)
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--config", required=True, type=Path,
                    help="training_config.json resolvido da corrida do alvo")
    ap.add_argument("--alphas", default="0.50,0.65,0.75")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    import torch

    from training.train import build_model, export_onnx

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    base_sd = load_state_dict(args.base)
    target_sd = load_state_dict(args.target)
    check_compatible(base_sd, target_sd)
    print(f"[merge] compatível: {len(target_sd)} tensores, "
          f"base {args.base_name} sha {sha256_16(args.base)}, "
          f"alvo {args.target_name} sha {sha256_16(args.target)}")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    (cfg.get("training") or {})["init_checkpoint"] = None

    model = build_model(cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for alpha in alphas:
        tag = f"merge_{args.base_name}_a{alpha:.2f}"
        cand_dir = args.out_dir / tag
        cand_dir.mkdir(parents=True, exist_ok=True)
        merged = interpolate(base_sd, target_sd, alpha)
        model.load_state_dict(merged, strict=True)
        model.eval()
        pt_path = cand_dir / "best.pt"
        torch.save(merged, pt_path)
        onnx_path = cand_dir / "draw_segmentation_v1.onnx"
        export_onnx(model, str(onnx_path), cfg)
        iou = argmax_iou_check(model, onnx_path, cfg)
        info = {
            "tag": tag,
            "base": {"name": args.base_name, "path": str(args.base), "sha256_16": sha256_16(args.base)},
            "target": {"name": args.target_name, "path": str(args.target), "sha256_16": sha256_16(args.target)},
            "alpha_target": alpha,
            "formula": "(1-alpha)*base + alpha*target",
            "onnx_sha256_16": sha256_16(onnx_path),
            "argmax_iou_torch_vs_onnx_min": iou,
        }
        (cand_dir / "merge_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        status = "OK" if iou >= 0.99 else "INVALIDO"
        print(f"[merge] {tag}: alpha={alpha:.2f} onnx sha {info['onnx_sha256_16']} "
              f"argmax IoU min {iou:.5f} -> {status}")
        results.append(info)
        if iou < 0.99:
            raise SystemExit(f"autochecagem falhou em {tag}: argmax IoU {iou:.5f} < 0,99")
    (args.out_dir / f"merge_{args.base_name}_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"[merge] {len(results)} candidatos em {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
