"""Contratos do gêmeo digital: escrita, fonte lazy, GT e laço real."""

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from tools import gen_synthetic as gs
from tools.eval_capture import (
    compare_masks,
    evaluate_canvas,
    load_canvas,
    pixel_metrics,
    run_pipeline,
)
from tools.sim_source import SyntheticCapture, export_video
from tools.sim_text import plan_text_strokes


def raster(specs, shape=(510, 360)):
    mask = np.zeros(shape, np.uint8)
    paper = np.full((*shape, 3), 255, np.uint8)
    for spec in specs:
        gs._apply_stroke(paper, mask, spec)
    return mask


def test_text_has_reproducible_resolved_curves_and_distinct_glyphs():
    def plan(text, seed=2):
        return plan_text_strokes(text, np.random.default_rng(seed), 360, 510)

    a, b = plan("Olá, Draw! 123"), plan("Olá, Draw! 123")
    assert len(a) == len(b)
    assert all(spec[:3] == other[:3] for spec, other in zip(a, b))
    assert all(np.array_equal(s[3], t[3]) for s, t in zip(a, b))
    assert all(s[0] == "curve" and s[3].dtype == np.int32 for s in a)
    assert not np.array_equal(raster(plan("a")), raster(plan("b")))
    assert not np.array_equal(raster(a), raster(plan("Olá, Draw! 123", seed=5)))
    assert np.array_equal(raster(plan("ação")), raster(plan("ac\u0327a\u0303o")))


def test_ligatures_pen_lifts_and_line_wrap():
    rng = np.random.default_rng
    linked = plan_text_strokes("ab", rng(1), 360, 510)
    separate = plan_text_strokes("ab", rng(1), 360, 510, ligatures=False)
    assert len(linked) == len(separate) + 1
    # i = haste + ponto; espaço e newline não ganham ligaduras.
    assert len(plan_text_strokes("i i\ni", rng(1), 360, 510)) == 6
    specs = plan_text_strokes("a" * 30, rng(1), 360, 510)
    assert max(s[3][:, 1].max() for s in specs) > 100
    assert raster(specs).sum() > 0


@pytest.mark.parametrize("text", ["🙂", "字", "a" * 1000])
def test_writer_rejects_unsupported_or_overflow(text):
    with pytest.raises(ValueError):
        plan_text_strokes(text, np.random.default_rng(1), 360, 510)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"fps": float("nan")},
        {"stroke_seconds": 0},
        {"settle_seconds": -1},
        {"height": float("inf")},
    ],
)
def test_source_rejects_invalid_timing_and_layout(kwargs):
    with pytest.raises(ValueError):
        SyntheticCapture("a", **kwargs)


def test_source_progressive_cumulative_deterministic_and_finite():
    options = dict(fps=4, stroke_seconds=1, warmup_seconds=0, settle_seconds=0)
    source, twin = SyntheticCapture("ab", **options), SyntheticCapture("ab", **options)
    assert np.all(source.quad >= 24) and np.all(source.quad <= gs.SIZE - 24)
    previous = np.zeros((512, 512), bool)
    areas = []
    for index in range(source.frame_count):
        ok, frame = source.read()
        other_ok, other = twin.read()
        assert ok and other_ok and np.array_equal(frame, other)
        assert frame.shape == (512, 512, 3) and frame.dtype == np.uint8
        current = source.last_scene["ink_full"] > 0
        assert np.all(current[previous])
        previous = current.copy()
        areas.append(int(current.sum()))
        assert source.get(cv2.CAP_PROP_POS_FRAMES) == index + 1
    assert areas[0] == 0 < areas[1] < areas[4]
    assert np.array_equal(source.last_scene["ink_full"], source.ink_full)
    assert source.read() == (False, None)
    assert source.isOpened()  # como VideoCapture depois de EOF
    assert source.get(cv2.CAP_PROP_FRAME_COUNT) == len(areas)
    assert source.get(cv2.CAP_PROP_FRAME_WIDTH) == 512
    assert source.get(cv2.CAP_PROP_FPS) == 4
    assert source.get(-100) == 0
    source.release()
    assert not source.isOpened() and source.read() == (False, None)
    twin.release()


def test_hand_occludes_visible_ink_but_not_ground_truth():
    source = SyntheticCapture("laços", fps=4, stroke_seconds=0.5, hand=True)
    covered = 0
    while source.read()[0]:
        scene = source.last_scene
        hidden = (scene["mask"] == gs.HAND) & (scene["ink_full"] > 0)
        covered += int(hidden.sum())
        assert not np.any(scene["mask"][hidden] == gs.INK)
    assert covered > 0
    assert not np.any(source.last_scene["mask"] == gs.HAND)
    assert np.array_equal(source.last_scene["ink_full"], source.ink_full)
    source.release()


def test_empty_text_is_a_valid_blank_sequence(tmp_path):
    source = SyntheticCapture("", fps=2)
    source.save_ground_truth(tmp_path / "blank.npz")
    while source.read()[0]:
        assert not source.last_scene["ink_full"].any()
    report = evaluate_canvas(None, tmp_path / "blank.npz")
    assert report["iou"] == 1 and report["strokes"] == []
    source.release()


def test_video_export_decodes_and_gt_matches_planned_geometry(tmp_path):
    source = SyntheticCapture("Oi!", fps=4, stroke_seconds=0.25, settle_seconds=0.5)
    path = tmp_path / "sim.mp4"
    manifest = export_video(source, path)
    assert not source.isOpened()
    reader = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        assert reader.isOpened() and reader.get(cv2.CAP_PROP_FPS) == 4
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            assert frame.shape == (512, 512, 3)
            decoded += 1
    finally:
        reader.release()
    assert decoded == manifest["frames"]
    assert json.loads(path.with_suffix(".json").read_text()) == manifest
    with np.load(path.with_suffix(".gt.npz"), allow_pickle=False) as bundle:
        assert np.array_equal(bundle["page_ink_full"], raster(source.specs))
        assert np.array_equal(bundle["ink_full"], source.ink_full)
    assert np.array_equal(np.load(tmp_path / "sim_ink_full.npy"), source.ink_full)
    with pytest.raises(ValueError, match="frame zero"):
        export_video(source, tmp_path / "again.mp4")


def test_exact_pixel_metrics_and_per_stroke_losses():
    a, b, c = [np.zeros((10, 10), np.uint8) for _ in range(3)]
    a[1:3, 1:3] = 255
    b[5:7, 5:7] = 255
    c[8, 8] = 255
    truth = a | b | c
    pred = a.copy()
    pred[5, 5] = 255
    pred[0, 9] = 255
    result = compare_masks(pred, truth, [a, b, c])
    assert result["tp"] == 5 and result["fp"] == 1 and result["fn"] == 4
    assert result["iou"] == 0.5
    assert result["precision"] == 5 / 6 and result["recall"] == 5 / 9
    assert [s["status"] for s in result["strokes"]] == ["captured", "partial", "lost"]
    assert pixel_metrics(a * 0, a * 0)["iou"] == 1
    with pytest.raises(ValueError):
        pixel_metrics(np.zeros((2, 2)), a)
    with pytest.raises(ValueError):
        compare_masks(pred, truth, stroke_recall=0)


def test_bundle_scene_page_and_detected_anchor_alignment(tmp_path):
    source = SyntheticCapture("abc")
    gt = tmp_path / "gt.npz"
    source.save_ground_truth(gt)
    for space, prediction, corners in (
        ("scene", source.ink_full, None),
        ("page", source.page_ink_full, None),
        ("page", source.page_ink_full, source.quad),
    ):
        report = evaluate_canvas(prediction, gt, canvas_space=space, corners=corners)
        assert report["iou"] == 1
        assert report["per_stroke_available"]
        assert report["stroke_counts"]["captured"] == len(source.specs)
    # Uma pose errada precisa penalizar a medição; nada alinha pelo conteúdo.
    shifted = evaluate_canvas(source.page_ink_full, gt, corners=source.quad + (20, 0))
    assert shifted["iou"] < 0.5
    missed = evaluate_canvas(None, gt)
    assert missed["recall"] == 0 and missed["stroke_counts"]["lost"] == len(
        source.specs
    )
    source.release()


def test_plain_npy_requires_pose_for_rectified_canvas(tmp_path):
    source = SyntheticCapture("a")
    gt = tmp_path / "ink_full.npy"
    np.save(gt, source.ink_full)
    report = evaluate_canvas(source.ink_full, gt, canvas_space="scene")
    assert report["iou"] == 1 and not report["per_stroke_available"]
    with pytest.raises(ValueError, match="exige --corners"):
        evaluate_canvas(source.page_ink_full, gt)
    source.release()


def test_load_png_ignores_provisional_gray_and_npy_is_positive_mask(tmp_path):
    image = np.array([[255, 0, 176]], np.uint8)
    cv2.imwrite(str(tmp_path / "canvas.png"), image)
    np.save(tmp_path / "mask.npy", np.array([[0, 1, 255]], np.uint8))
    assert load_canvas(tmp_path / "canvas.png").tolist() == [[False, True, False]]
    assert load_canvas(tmp_path / "mask.npy").tolist() == [[False, True, True]]


def test_real_loop_accepts_lazy_capture_without_gt_access(monkeypatch):
    from draw import config

    monkeypatch.setattr(config, "PAPER_W", 180)
    monkeypatch.setattr(config, "PAPER_H", 255)
    synthetic = SyntheticCapture("Draw", fps=6, stroke_seconds=0.2, settle_seconds=1)

    class CameraOnly:
        # Nenhum atributo de GT é acessível pelo consumidor.
        isOpened = synthetic.isOpened
        read = synthetic.read
        get = synthetic.get

    try:
        result = run_pipeline(CameraOnly(), legacy_ink=True)
        assert result["frames"] == synthetic.frame_count
        assert result["detections"] > 0
        assert result["scheduler"] == "synchronous_video_clock"
        assert result["mode"] == "legacy"
        assert result["mask"] is not None
        assert result["mask"].shape == (255, 180)
        assert result["corners"].shape == (4, 2)
    finally:
        synthetic.release()


def test_headless_uses_real_page_clean_worker(monkeypatch):
    from draw import config, page_clean

    monkeypatch.setattr(config, "PAPER_W", 180)
    monkeypatch.setattr(config, "PAPER_H", 255)
    calls = []

    class TestModel:
        # Somente a inferência ONNX é stub: worker, votos e laço são reais.
        threshold = 0.7

        def __init__(self, _path):
            pass

        def load(self):
            pass

        def ink_prob(self, image):
            calls.append(image.shape)
            return (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 100).astype(np.float32)

    monkeypatch.setattr(page_clean, "PageCleanInk", TestModel)
    synthetic = SyntheticCapture("Draw", fps=6, settle_seconds=2)
    try:
        result = run_pipeline(synthetic)
        assert calls and result["ink_passes"] == len(calls)
        assert result["mask"] is not None
        assert result["corners"].shape == (4, 2)
    finally:
        synthetic.release()


def test_clis_run_as_scripts_outside_repo(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "sim.mp4"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/sim_source.py"),
            "--text",
            "Oi",
            "--out",
            str(output),
            "--fps",
            "2",
            "--settle-seconds",
            "0",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    report = subprocess.run(
        [
            sys.executable,
            str(root / "tools/eval_capture.py"),
            "--canvas",
            str(tmp_path / "sim_ink_full.npy"),
            "--canvas-space",
            "scene",
            "--gt",
            str(tmp_path / "sim.gt.npz"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(report.stdout)["iou"] == 1
