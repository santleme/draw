"""Traduz frames.csv do Kaggle ou pasta/CSV de páginas em pool revisável local.

python tools/selflearn/import_external.py --root /dados/pool --dataset guzzbr/draw-real-capture-pool-v1 --out captures/external-pool
python tools/selflearn/import_external.py --root /dados/desenhos --dataset autor/lote --batch lote-1 --images-are-pages --out captures/external-pages

Depois: review_holdout.py --pool captures/external-pool --split train mark SESSAO GERACAO --label corrigido.png
Holdout: --split holdout, sempre --label feito do zero, nunca --accept-draft.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from draw.selflearn.external import translate  # noqa: E402
from draw.selflearn.session import sha256_file  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--batch")
    parser.add_argument("--split", choices=("auto", "train", "holdout"), default="auto")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--images-are-pages", action="store_true")
    parser.add_argument("--model")
    parser.add_argument(
        "--sheet-model",
        default=str(
            Path(__file__).resolve().parents[2] / "models/draw_seg_v3_dyn.onnx"
        ),
    )
    args = parser.parse_args(argv)
    from draw.page_clean import PageCleanInk

    model = PageCleanInk(model_path=args.model)
    prepare = None
    preparation_id = None
    if not args.images_are_pages:
        from draw import paper, config
        from draw.neural import NeuralSheetDetector

        preparation_id = {
            "sheet_model_sha256": sha256_file(args.sheet_model),
            "size": [config.PAPER_W, config.PAPER_H],
        }
        detector = NeuralSheetDetector(args.sheet_model)

        def prepare(image):
            # Fotos de autores/sessões diferentes não herdam memória de arestas.
            detector._edge_lines = detector._memory_mask = detector._memory_geometry = (
                None
            )
            corners, _confidence = detector.detect_sheet_with_confidence(image)
            return None if corners is None else paper.warp_sheet(image, corners)

    print(
        json.dumps(
            translate(
                args.root,
                args.out,
                args.dataset,
                manifest=args.manifest,
                batch=args.batch,
                split=args.split,
                holdout_fraction=args.holdout_fraction,
                model=model,
                prepare_page=prepare,
                images_are_pages=args.images_are_pages,
                preparation_id=preparation_id,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
