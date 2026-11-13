"""Overlay predito x GT para as sessões quickdraw.

Preto = acerto, vermelho = predito a mais, azul = perdido. O GT da
página é projetado na grade do canvas pela âncora detectada real.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
root = REPO / "captures" / "quickdraw-sessions"

from draw.paper import perspective_matrix

for canvas in sorted(root.glob("*.canvas.npy")):
    s = canvas.name[: -len(".canvas.npy")]
    pred = np.load(canvas, allow_pickle=False) > 0
    corners = np.load(root / f"{s}.canvas.corners.npy", allow_pickle=False)
    gt = np.load(root / f"{s}.gt.npz")
    page_ink = gt["page_ink_final"]
    H, W = pred.shape
    destination = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    matrix = perspective_matrix(corners, destination)
    page_matrix = matrix @ gt["page_to_scene"]
    gt_mask = (
        cv2.warpPerspective(page_ink, page_matrix, (W, H), flags=cv2.INTER_NEAREST)
        > 0
    )
    vis = np.full((H, W, 3), 255, np.uint8)
    tp, fp, fn = pred & gt_mask, pred & ~gt_mask, ~pred & gt_mask
    vis[tp] = (0, 0, 0)        # acerto: preto
    vis[fp] = (0, 0, 255)      # predito a mais: vermelho
    vis[fn] = (255, 0, 0)      # perdido: azul
    cv2.imwrite(str(root / f"{s}.overlay.png"), vis)
    print(s, "tp", int(tp.sum()), "fp", int(fp.sum()), "fn", int(fn.sum()))
