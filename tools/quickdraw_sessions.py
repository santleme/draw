"""Sessões de desenho (QuickDraw) pelo pipeline real do Draw.

Seleciona desenhos reconhecidos do corpus local em
~/draw-lab/ink-data/quickdraw, converte os traços vetoriais (polylines
0-255) em specs ``curve`` do simulador de escrita e exporta MP4 +
ground truth via tools/sim_source.py::export_video — pronto para
tools/eval_capture.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QDRAW = Path.home() / "draw-lab" / "ink-data" / "quickdraw"
INK_COLOR = (25, 25, 30)
THICK = 2


def load_drawing(category, pick, min_strokes=2, max_strokes=40, scan=8000):
    """pick-ésimo desenho reconhecido da categoria (ordem do arquivo)."""
    path = QDRAW / f"{category}.ndjson"
    if not path.is_file():
        raise FileNotFoundError(path)
    found = 0
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= scan:
                break
            rec = json.loads(line)
            if not rec.get("recognized"):
                continue
            strokes = [
                np.array(list(zip(s[0], s[1])), np.float32)
                for s in rec["drawing"]
                if len(s[0]) >= 2
            ]
            if not (min_strokes <= len(strokes) <= max_strokes):
                continue
            if found == pick:
                return strokes
            found += 1
    raise ValueError(f"{category}: só {found} candidatos (pick={pick})")


def place(strokes, box, rng, jitter=0.3):
    """Escala o desenho para a caixa (x, y, w, h) e vira specs curve."""
    from tools.sim_text import _smooth

    bx, by, bw, bh = box
    all_pts = np.concatenate(strokes)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    scale = min(bw / span[0], bh / span[1])
    offset = np.array([bx, by], np.float32) + (
        np.array([bw, bh], np.float32) - span * scale
    ) / 2
    specs = []
    for stroke in strokes:
        pts = np.rint((stroke - lo) * scale + offset).astype(np.float32)
        geo = _smooth(pts, rng, jitter)
        specs.append(("curve", INK_COLOR, THICK, geo))
    return specs


def make_planner(layout):
    """layout: lista de (strokes, box-fracao 0..1). Devolve o planner."""

    def planner(text, rng, w, h, height=22, **_kw):
        margin = 0.06
        specs = []
        for strokes, (fx, fy, fw, fh) in layout:
            box = (
                int(w * fx + w * margin * fw),
                int(h * fy + h * margin * fh),
                int(w * fw * (1 - 2 * margin)),
                int(h * fh * (1 - 2 * margin)),
            )
            specs.extend(place(strokes, box, rng))
        return specs

    return planner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="categoria:pick ou categoria:pick@fx,fy,fw,fh (repetível)",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--stroke-seconds", type=float, default=0.45)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--settle-seconds", type=float, default=7)
    parser.add_argument("--hand", action="store_true")
    args = parser.parse_args(argv)

    import tools.sim_source as ss

    layout = []
    for raw in args.spec:
        name, _, region = raw.partition("@")
        category, _, pick = name.partition(":")
        strokes = load_drawing(category, int(pick or 0))
        box = (
            tuple(float(v) for v in region.split(","))
            if region
            else (0.0, 0.0, 1.0, 1.0)
        )
        layout.append((strokes, box))

    original = ss.plan_text_strokes
    ss.plan_text_strokes = make_planner(layout)
    try:
        cap = ss.SyntheticCapture(
            args.label,
            seed=args.seed,
            fps=args.fps,
            stroke_seconds=args.stroke_seconds,
            warmup_seconds=args.warmup_seconds,
            settle_seconds=args.settle_seconds,
            hand=args.hand,
        )
    finally:
        ss.plan_text_strokes = original
    manifest = ss.export_video(cap, args.out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
