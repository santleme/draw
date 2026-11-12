"""Testes do simulador procedural (frente 103): funções puras e
o contrato do corpus — GT exato, bandas, impressão separada da
tinta, homografia salva. Sem ONNX, sem modelos."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "tools")
)

from sim_writer import (  # noqa: E402
    SHEET_H,
    SHEET_W,
    band_masks,
    bezier,
    build_strokes,
    corners_at,
    draw_ink,
    generate_scenario,
    homography,
    render_base,
    sub_polyline,
)


def test_bezier_endpoints():
    pts = bezier((0, 0), (0, 10), (10, 10), (10, 0), n=20)
    assert np.allclose(pts[0], (0, 0))
    assert np.allclose(pts[-1], (10, 0))
    assert len(pts) == 20


def test_sub_polyline_prefix_length():
    line = np.array([[0.0, 0.0], [100.0, 0.0]])
    sub = sub_polyline(line, 30.0)
    assert sub[-1, 0] == 30.0
    assert sub_polyline(line, 0.0) is None
    assert len(sub_polyline(line, 500.0)) == 2


def test_gt_final_equals_full_drawing():
    rng = np.random.default_rng(0)
    strokes, _, _, _ = build_strokes("cad", rng)
    total = sum(
        np.linalg.norm(np.diff(s.points, axis=0), axis=1).sum()
        for s in strokes
    )
    page = np.full((SHEET_H, SHEET_W, 3), 243, np.uint8)
    gt = np.zeros((SHEET_H, SHEET_W), np.uint8)
    _, gt_full = draw_ink(page, gt, strokes, total + 1)
    bands = band_masks(strokes)
    union = np.zeros_like(gt)
    for m in bands.values():
        union = np.maximum(union, m)
    assert (gt_full > 0).sum() == (union > 0).sum()
    assert np.array_equal(gt_full > 0, union > 0)


def test_gt_empty_before_any_reveal():
    rng = np.random.default_rng(0)
    strokes, _, _, _ = build_strokes("cursive", rng)
    page = np.full((SHEET_H, SHEET_W, 3), 243, np.uint8)
    gt = np.zeros((SHEET_H, SHEET_W), np.uint8)
    _, gt0 = draw_ink(page, gt, strokes, 0.0)
    assert not gt0.any()


def test_print_mask_disjoint_from_paper_and_covers_lines():
    rng = np.random.default_rng(0)
    _, pm_ruled = render_base("ruled", rng)
    _, pm_clean = render_base("clean", rng)
    assert pm_ruled.sum() > 0
    assert not pm_clean.any()


def test_thin_bands_are_the_9_cells():
    rng = np.random.default_rng(0)
    strokes, _, _, _ = build_strokes("thin", rng)
    bands = band_masks(strokes)
    assert len(bands) == 9
    for w in (1, 2, 3):
        for c in ("strong", "medium", "weak"):
            assert f"thin_{w}px_{c}" in bands


def test_homography_maps_sheet_corners():
    corners = corners_at(0, {})
    H = homography(corners)
    import cv2

    src = np.array(
        [[[0, 0], [SHEET_W, 0], [SHEET_W, SHEET_H], [0, SHEET_H]]],
        np.float32,
    )
    dst = cv2.perspectiveTransform(src, H.astype(np.float32))[0]
    assert np.allclose(dst, corners, atol=0.5)


def test_events_corners_push_and_rotation():
    events = {
        "push_frame": 10, "push_px": 25.0,
        "occlusion_frames": [14, 15, 16],
        "occlusion_center_sheet": [620.0, 390.0],
        "occlusion_radius_sheet": 95.0,
        "rotation_frame": 20, "rotation_deg": 2.0,
    }
    base = corners_at(0, events)
    pushed = corners_at(10, events)
    assert np.allclose(pushed - base, [25.0, 0.0])
    rotated = corners_at(20, events)
    assert not np.allclose(rotated, pushed)


def test_generate_scenario_corpus_contract(tmp_path):
    side = generate_scenario(tmp_path, "blank_keyboard")
    scen = tmp_path / "blank_keyboard"
    assert (scen / "frames").is_dir()
    assert len(list((scen / "frames").glob("*.png"))) == len(
        side["frames"])
    saved = json.loads((scen / "sidecar.json").read_text())
    assert saved["format"] == "sim_writer_v1"
    assert saved["sheet_space"] == [SHEET_W, SHEET_H]
    for fr in saved["frames"]:
        assert len(fr["H_sheet_to_camera"]) == 3
        assert fr["revealed_fraction"] == 0.0
