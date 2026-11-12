"""Testes do laboratório de replay (frente 102): sinais, seleção e
prancha — funções puras, sem ONNX."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from replay_lab import (  # noqa: E402
    build_panel,
    compute_signals,
    list_corpus_frames,
    quad_jump,
    select_worst,
)

QUAD = [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]]


def _rec(i, quad=QUAD, canvas=0, present=True):
    return {
        "frame": i,
        "quad": quad,
        "canvas_px": canvas,
        "present": present,
    }


def test_quad_jump_none_and_value():
    assert quad_jump(None, QUAD) is None
    moved = [[x + 30.0, y] for x, y in QUAD]
    assert quad_jump(QUAD, moved) == 30.0


def test_signals_growth_and_flip():
    recs = [
        _rec(0, canvas=100),
        _rec(1, canvas=350),
        _rec(2, canvas=300, present=False),
    ]
    compute_signals(recs)
    assert recs[1]["sig_canvas_growth_px"] == 250
    assert recs[2]["sig_canvas_growth_px"] == 0  # encolher não conta
    assert recs[2]["sig_presence_flip"] == 1
    assert recs[0]["sig_presence_flip"] == 0


def test_select_worst_picks_the_jump_and_dedupes():
    recs = [_rec(i, canvas=100) for i in range(12)]
    jumped = [[x + 200.0, y] for x, y in QUAD]
    recs[5]["quad"] = jumped
    recs[6]["quad"] = jumped
    compute_signals(recs)
    worst = select_worst(recs, top_k=3, min_gap=2)
    idxs = {w["index"] for w in worst}
    assert 5 in idxs
    assert not ({5, 6} <= idxs)  # vizinho suprimido


def test_build_panel_shape():
    frame = np.zeros((720, 1280, 3), np.uint8)
    warp = np.full((1200, 850, 3), 200, np.uint8)
    mask = np.zeros((1200, 850), np.uint8)
    mask[100:200, 100:300] = 1
    panel = build_panel(frame, QUAD, warp, mask)
    assert panel.shape == (720, 1280, 3)


def test_list_corpus_prefers_raw(tmp_path):
    (tmp_path / "raw").mkdir()
    for name in ("frame_000002.png", "frame_000001.png"):
        (tmp_path / "raw" / name).write_bytes(b"x")
    (tmp_path / "other.png").write_bytes(b"x")
    files = list_corpus_frames(tmp_path)
    assert [f.name for f in files] == [
        "frame_000001.png",
        "frame_000002.png",
    ]


def test_signals_tracker_jump_and_aspect():
    recs = [_rec(0), _rec(1)]
    wedge = [[0.0, 0.0], [900.0, 100.0], [950.0, 130.0], [50.0, 30.0]]
    recs[1]["quad"] = wedge
    compute_signals(recs)
    assert recs[1]["sig_tracker_jump_px"] > 400
    assert recs[1]["sig_quad_aspect"] > 5
    assert recs[0]["sig_quad_aspect"] < 2
