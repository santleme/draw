"""Paridade após refino full-res e contratos do caminho de contornos nativo."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import neural, paper  # noqa: E402
from test_neural_onnx_contract import _Session, _logits_for_classes  # noqa: E402


_QUADS = (
    [[70, 40], [310, 48], [325, 340], [55, 330]],
    [[90, 55], [285, 55], [285, 330], [90, 330]],
    [[60, 90], [315, 65], [330, 310], [45, 320]],
)


def _mask(quad, jagged=False):
    mask = np.zeros((384, 384), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(quad, dtype=np.int32)], 255)
    if jagged:
        for start, end in zip(quad, np.roll(quad, -1, axis=0)):
            for i, t in enumerate(np.linspace(0.08, 0.92, 24)):
                point = np.rint((1 - t) * start + t * end).astype(int)
                cv2.circle(mask, tuple(point), 1, 255 if i % 2 else 0, -1)
    return mask


def _frame(shape, quad):
    frame = np.full((*shape, 3), 25, dtype=np.uint8)
    cv2.fillPoly(frame, [np.rint(quad).astype(np.int32)], (235, 235, 235))
    center = np.mean(quad, axis=0)
    for point in 0.65 * quad + 0.35 * center:
        cv2.circle(frame, tuple(np.rint(point).astype(int)), 18, (5, 5, 5), -1)
    cv2.line(
        frame,
        tuple(np.rint(center - [40, 0]).astype(int)),
        tuple(np.rint(center + [40, 0]).astype(int)),
        (5, 5, 5),
        9,
    )
    return frame


def _warp_ink(frame, quad):
    destination = np.float32([[0, 0], [599, 0], [599, 799], [0, 799]])
    matrix = cv2.getPerspectiveTransform(quad, destination)
    return (
        cv2.warpPerspective(frame[:, :, 0], matrix, (600, 800), flags=cv2.INTER_LINEAR)
        < 100
    )


@pytest.mark.parametrize("shape", [(720, 1280), (1080, 1920), (1280, 720), (710, 1289)])
@pytest.mark.parametrize("points", _QUADS)
@pytest.mark.parametrize("jagged", [False, True])
def test_native_scale_parity(shape, points, jagged):
    quad = np.asarray(points, dtype=np.float32)
    height, width = shape
    scale = np.float32([width / 384.0, height / 384.0])
    native = _mask(quad, jagged)
    full = cv2.resize(native, (width, height), interpolation=cv2.INTER_NEAREST)
    frame = _frame(shape, quad * scale)

    reference = neural.fit_quad(full, frame)
    actual = neural.fit_quad(native, frame, scale_x=scale[0], scale_y=scale[1])

    assert reference is not None and actual is not None
    assert actual.dtype == np.float32
    assert np.max(np.linalg.norm(actual - reference, axis=1)) <= 1.5
    reference_warp, native_warp = _warp_ink(frame, reference), _warp_ink(frame, actual)
    union = np.count_nonzero(reference_warp | native_warp)
    assert union > 1000  # Mede tinta, não duas folhas vazias preenchendo o warp.
    assert np.count_nonzero(reference_warp & native_warp) / union > 0.995


def test_only_quad_is_scaled_before_full_resolution_refinement(monkeypatch):
    quad = np.float32(_QUADS[0])
    native = _mask(quad)
    frame = np.zeros((720, 1280, 3), np.uint8)
    scale = np.float32([1280 / 384.0, 720 / 384.0])
    calls = []
    find_contours, approx = cv2.findContours, cv2.approxPolyDP

    def find(mask, *args):
        assert mask.shape == (384, 384) and mask.dtype == np.uint8
        calls.append("contours")
        return find_contours(mask, *args)

    def approximate(contour, *args):
        assert contour.max() < 384
        calls.append("approx")
        return approx(contour, *args)

    def refit(contour, candidate, shape, fallback, pixel_scale):
        assert contour.max() < 384
        assert shape == (384, 384)
        return quad, None, None

    def refine(image, corners):
        assert image is frame
        np.testing.assert_allclose(corners, quad * scale)
        calls.append("refine")
        return corners

    monkeypatch.setattr(cv2, "findContours", find)
    monkeypatch.setattr(cv2, "approxPolyDP", approximate)
    monkeypatch.setattr(neural, "_refit_corners", refit)
    monkeypatch.setattr(paper, "_refine_corners", refine)
    actual = neural.fit_quad(native, frame, scale_x=scale[0], scale_y=scale[1])
    np.testing.assert_array_equal(actual, quad * scale)
    assert calls[0] == "contours" and "approx" in calls and calls[-1] == "refine"


def _detector(mask, kind="semantic", presence=2.0):
    classes = (mask > 0).astype(np.uint8)
    logits = _logits_for_classes(classes)
    outputs = {"logits": logits}
    if kind != "semantic":
        outputs = {
            "mask_logits": logits[:, :2] if kind == "legacy" else logits,
            "presence_logit": np.array([presence], np.float32),
        }
    detector = neural.NeuralSheetDetector("unused.onnx")
    detector._session = _Session(outputs, input_shape=[1, 3, 384, 384])
    return detector


@pytest.mark.parametrize("kind", ["semantic", "hybrid", "legacy"])
@pytest.mark.parametrize("present", [True, False])
def test_app_path_never_upsamples_mask_and_debug_keeps_contract(
    monkeypatch, kind, present
):
    native = _mask(np.float32(_QUADS[0])) if present else np.zeros((384, 384), np.uint8)
    detector = _detector(native, kind, presence=2.0 if present else -2.0)
    frame = _frame((720, 1280), np.float32(_QUADS[0]) * [1280 / 384, 720 / 384])
    resize = cv2.resize
    nearest = []

    def record_resize(image, size, *args, **kwargs):
        if kwargs.get("interpolation") == cv2.INTER_NEAREST:
            nearest.append((image.shape, size))
        return resize(image, size, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", record_resize)
    corners, verdict = detector.detect_sheet_with_confidence(frame)
    assert (corners is not None) == present
    assert verdict == ("paper" if present else None)
    assert nearest == []
    if present:
        assert detector._memory_mask.shape == (96, 96)

    _, debug_mask = detector.predict(frame)
    _, native_mask = detector.predict(frame, native=True)
    _, debug_verdict, call_mask = detector(frame)
    assert debug_verdict == verdict
    assert native_mask.shape == (384, 384)
    np.testing.assert_array_equal(native_mask, native)
    expected = resize(native, (1280, 720), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(debug_mask, expected)
    np.testing.assert_array_equal(call_mask, expected)
    assert len(nearest) == 2


def test_min_area_fallback_is_scaled_without_subpixel_refinement(monkeypatch):
    native = _mask(np.float32(_QUADS[0]))
    contour = neural._largest_contour(native)
    expected = neural._min_area_quad(contour) * np.float32([1280 / 384, 720 / 384])
    monkeypatch.setattr(neural, "_approx_quad", lambda contour: None)

    def unexpected(*args):
        pytest.fail("a reserva de área mínima não deve reajustar nem refinar")

    monkeypatch.setattr(neural, "_refit_corners", unexpected)
    monkeypatch.setattr(paper, "_refine_corners", unexpected)
    actual = neural.fit_quad(
        native,
        np.zeros((720, 1280, 3), np.uint8),
        scale_x=1280 / 384,
        scale_y=720 / 384,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(384, 384), (512, 512)])
def test_empty_and_tiny_masks_remain_rejected(shape):
    mask = np.zeros(shape, np.uint8)
    for has_tiny_blob in (False, True):
        if has_tiny_blob:
            mask[10:12, 10:12] = 255
        assert (
            neural.fit_quad(mask, scale_x=1280 / shape[1], scale_y=720 / shape[0])
            is None
        )


def test_memory_is_native_and_resets_when_frame_scale_changes(monkeypatch):
    mask = _mask(np.float32(_QUADS[0]))
    detector = _detector(mask)
    scale = dict(scale_x=1280 / 384, scale_y=720 / 384)
    assert detector._fit(mask, **scale) is not None
    assert detector._memory_mask.shape == (96, 96)
    assert detector._edge_lines is not None
    assert detector._fallback_lines(mask) is detector._edge_lines
    original_fit = neural._fit_quad_impl
    fallbacks = []

    def fit(mask, frame, fallback, scale_x, scale_y):
        fallbacks.append(fallback)
        return original_fit(mask, frame, fallback, scale_x, scale_y)

    monkeypatch.setattr(neural, "_fit_quad_impl", fit)
    detector._fit(mask, **scale)
    assert fallbacks[-1] is not None
    detector._fit(mask, scale_x=1920 / 384, scale_y=1080 / 384)
    assert fallbacks[-1] is None


def test_subpixel_refinement_leaves_projected_corners_outside_frame_untouched(
    monkeypatch,
):
    frame = np.zeros((720, 1280, 3), np.uint8)
    quad = np.float32([[-15, 30], [800, 40], [810, 600], [-20, 610]])

    def refine(image, corners):
        assert image is frame and corners.shape == (2, 2)
        return corners + 0.25

    monkeypatch.setattr(paper, "_refine_corners", refine)
    actual = neural._refine_inside(frame, quad)
    np.testing.assert_array_equal(actual[[0, 3]], quad[[0, 3]])
    np.testing.assert_array_equal(actual[[1, 2]], quad[[1, 2]] + 0.25)


def test_shared_subpixel_helper_accepts_only_the_visible_corners():
    frame = np.zeros((720, 1280, 3), np.uint8)
    corners = np.float32([[30, 40], [1200, 650]])
    refined = paper._refine_corners(frame, corners)
    assert refined.shape == (2, 2) and refined.dtype == np.float32
    np.testing.assert_array_equal(refined, corners)


def test_native_memory_preserves_occlusion_guard_and_allows_real_rotation():
    quad = np.float32(_QUADS[1])
    scale = np.float32([1280 / 384, 720 / 384])
    kwargs = dict(scale_x=scale[0], scale_y=scale[1])
    detector = _detector(_mask(quad))
    reference = detector._fit(_mask(quad), **kwargs)
    saved_bottom = detector._edge_lines[2][0].copy()

    occluded = _mask(quad)
    cv2.circle(occluded, (185, 328), 35, 0, -1)
    actual = detector._fit(occluded, **kwargs)
    assert np.max(np.linalg.norm(actual - reference, axis=1)) <= 3.0
    np.testing.assert_allclose(detector._edge_lines[2][0], saved_bottom, atol=1e-4)

    sleeve = quad.copy()
    sleeve[2, 0] -= 36
    actual = detector._fit(_mask(sleeve), **kwargs)
    assert np.max(np.linalg.norm(actual - reference, axis=1)) <= 3.0

    angle = np.deg2rad(7)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    original = quad * scale
    center = original.mean(axis=0)
    rotated = (original - center) @ rotation.T + center
    actual = detector._fit(_mask(rotated / scale), **kwargs)
    assert np.max(np.linalg.norm(actual - rotated, axis=1)) <= 3.5


@pytest.mark.parametrize("size", [384, 512])
def test_real_onnx_keeps_four_class_output_and_native_resolution(tmp_path, size):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnx_fixtures

    path = onnx_fixtures.four_class_model(tmp_path / "sheet.onnx", size=size)
    detector = neural.NeuralSheetDetector(path)
    detector.load()
    assert detector.contract().size == (size, size)
    assert detector._get_session().get_outputs()[0].shape == [1, 4, size, size]
    frame = np.full((720, 1280, 3), 230, np.uint8)
    presence, mask = detector.predict(frame, native=True)
    assert mask.shape == (size, size) and mask.dtype == np.uint8
    assert presence in (1.0, -10.0)
