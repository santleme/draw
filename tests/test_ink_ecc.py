"""Direcao, escala e segurancas do refinamento residual na ancora."""

import time

import cv2
import numpy as np
import pytest

from draw import ink_layers as ink


def smooth_page():
    y, x = np.mgrid[:1200, :850].astype(np.float32)
    page = np.zeros_like(x)
    for i in range(7):
        center = 140 + 145 * i + 35 * np.sin(x / (75 + 13 * i) + i)
        stroke = np.exp(-0.5 * ((y - center) / 5.0) ** 2)
        envelope = np.clip((x - 65 - 9 * i) / 25, 0, 1)
        envelope *= np.clip((790 - 14 * i - x) / 25, 0, 1)
        np.maximum(page, 0.95 * stroke * envelope, out=page)
    return page


def test_ecc_direction_and_half_scale_on_smooth_probability_map():
    ref = smooth_page()
    corners = np.float32([[0, 0], [850, 0], [850, 1200], [0, 1200]])
    displaced = corners + np.float32([[12, -8], [16, 6], [10, 14], [-9, 12]])
    known = cv2.getPerspectiveTransform(corners, displaced)
    obs = cv2.warpPerspective(ref, known, (850, 1200), flags=cv2.INTER_LINEAR)
    aligned, align, info = ink._refine_ink_ecc(ref, obs)
    assert not info["ecc_rejected"]
    assert info["ecc_mode"] == "homography"
    assert np.sqrt(np.mean((aligned - ref) ** 2)) < np.sqrt(np.mean((obs - ref) ** 2))
    prior = cv2.dilate((ref > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    def containment(prob):
        seen = prob > 0.5
        return np.count_nonzero(seen & prior) / np.count_nonzero(seen)

    assert containment(aligned) > containment(obs)
    assert containment(aligned) > 0.90
    restored = cv2.perspectiveTransform(displaced[None], align)[0]
    assert np.max(np.linalg.norm(restored - corners, axis=1)) < 2.0
    assert info["ecc_ms"] > 0


def test_ecc_exact_parameters_mask_and_affine_fallback(monkeypatch):
    ref = smooth_page()
    obs = np.roll(ref, 14, axis=1)
    mask = np.zeros(ref.shape, np.uint8)
    mask[100:200, 200:300] = 255
    calls = []

    def ecc(template, image, warp, motion, criteria, input_mask, gauss):
        calls.append(motion)
        assert template.dtype == image.dtype == np.float32
        assert template.shape == image.shape == (600, 425)
        assert np.all(template[55:95, 105:145] == 0)
        assert np.all(image[55:95, 105:145] == 0)
        assert criteria == (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        assert input_mask is None and gauss == 5
        if motion == cv2.MOTION_HOMOGRAPHY:
            raise cv2.error("nao convergiu")
        warp[0, 2] = 7.0
        return 0.99, warp

    monkeypatch.setattr(cv2, "findTransformECC", ecc)
    monkeypatch.setattr(cv2, "GaussianBlur", lambda *a, **k: pytest.fail("blur externo"))
    aligned, matrix, info = ink._refine_ink_ecc(ref, obs, mask)
    np.testing.assert_allclose(matrix, [[1, 0, -14], [0, 1, 0], [0, 0, 1]])
    assert calls == [cv2.MOTION_HOMOGRAPHY, cv2.MOTION_AFFINE]
    masked = obs.copy()
    masked[mask > 0] = 0
    expected = cv2.warpPerspective(masked, matrix, (850, 1200), flags=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(aligned, expected)
    assert not info["ecc_rejected"]


@pytest.mark.parametrize("estimated", [
    [[1, 0, 28], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, float("nan")], [0, 1, 0], [0, 0, 1]],
])
def test_ecc_rejects_unsafe_homography_without_affine_retry(monkeypatch, estimated):
    def ecc(*args):
        assert args[3] == cv2.MOTION_HOMOGRAPHY
        return 0.99, np.array(estimated, dtype=np.float32)

    monkeypatch.setattr(cv2, "findTransformECC", ecc)
    ref = smooth_page()
    obs = np.roll(ref, 1, axis=1)
    aligned, matrix, info = ink._refine_ink_ecc(ref, obs)
    np.testing.assert_array_equal(matrix, np.eye(3))
    np.testing.assert_array_equal(aligned, obs)
    assert info["ecc_rejected"]


def test_ecc_low_ink_bypasses_optimizer_and_failed_fallback_is_safe(monkeypatch):
    obs = smooth_page()
    monkeypatch.setattr(cv2, "findTransformECC", lambda *a: pytest.fail("pouca tinta"))
    aligned, matrix, info = ink._refine_ink_ecc(np.zeros_like(obs), obs)
    np.testing.assert_array_equal(matrix, np.eye(3))
    assert aligned is obs and info["ecc_mode"] == "bypass"
    assert not info["ecc_rejected"]

    def fail(*args):
        raise cv2.error("nao convergiu")

    monkeypatch.setattr(cv2, "findTransformECC", fail)
    aligned, matrix, info = ink._refine_ink_ecc(obs, np.roll(obs, 1, axis=0))
    np.testing.assert_array_equal(matrix, np.eye(3))
    assert info["ecc_rejected"] and info["ecc_mode"] == "affine"


def test_ecc_corner_gate_uses_canonical_page_pixels_on_reduced_maps(monkeypatch):
    ref = cv2.resize(smooth_page(), (85, 120), interpolation=cv2.INTER_AREA)
    obs = np.roll(ref, 6, axis=1)

    def ecc(*args):
        return 0.99, np.float32([[1, 0, 3], [0, 1, 0], [0, 0, 1]])

    monkeypatch.setattr(cv2, "findTransformECC", ecc)
    aligned, matrix, info = ink._refine_ink_ecc(ref, obs)
    assert info["ecc_rejected"]
    np.testing.assert_array_equal(matrix, np.eye(3))
    np.testing.assert_array_equal(aligned, obs)


def test_visibility_follows_anchor_and_residual_warps(monkeypatch):
    prob = smooth_page()
    corners = np.float32([[0, 0], [850, 0], [850, 1200], [0, 1200]])
    current = corners + np.float32([8, 0])
    visible = np.ones(prob.shape, dtype=bool)
    visible[100:200, 300:400] = False
    geometry = ink._registration_matrix(corners, current, prob.shape)
    align = np.float64([[1, 0, -14], [0, 1, 5], [0, 0, 1]])
    anchor_visible = cv2.warpPerspective(
        visible.astype(np.uint8), geometry, (850, 1200),
        flags=cv2.INTER_NEAREST,
    ) > 0

    def refine(reference, observed, hand_mask=None):
        np.testing.assert_array_equal(hand_mask, ~anchor_visible)
        return (
            cv2.warpPerspective(observed, align, (850, 1200), flags=cv2.INTER_LINEAR),
            align,
            {"ecc_ms": 1.0, "ecc_mode": "homography", "ecc_rejected": False},
        )

    monkeypatch.setattr(ink, "_refine_ink_ecc", refine)
    _aligned, aligned_visible, _info = ink._register_ink_probability(
        prob, corners, current, prob, visible
    )
    expected = cv2.warpPerspective(
        anchor_visible.astype(np.uint8), align, (850, 1200),
        flags=cv2.INTER_NEAREST,
    ) > 0
    np.testing.assert_array_equal(aligned_visible, expected)
    # Um residuo nao pode reabrir a faixa que o registro recortou.
    assert not aligned_visible[10, 840]


@pytest.mark.lento
def test_worker_refines_after_anchor_registration_and_keeps_float_reference(monkeypatch):
    monkeypatch.setattr(ink, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(ink, "_IDLE_PROBE_SECONDS", 0.001)
    ref = smooth_page()
    corners = np.float32([[0, 0], [850, 0], [850, 1200], [0, 1200]])
    calls = []

    def refine(reference, observed, hand_mask=None):
        assert not worker._condition._is_owned(), "ECC nao pode bloquear submit/poll"
        calls.append((None if reference is None else reference.copy(), observed.copy()))
        return observed, np.eye(3), {"ecc_ms": 1.0, "ecc_rejected": False, "ecc_mode": "bypass"}

    class Model:
        threshold = 0.7

        def ink_prob(self, warped):
            return warped

    class Masker:
        def mask(self, warped):
            mask = np.zeros(warped.shape, np.uint8)
            mask[:20] = 255
            return mask

    monkeypatch.setattr(ink, "_refine_ink_ecc", refine)
    worker = ink._PageCleanWorker(Model(), Masker()).start()

    def submit(prob, generation, quad):
        before = worker.stats()["passes_total"]
        worker.submit(prob, generation, quad)
        deadline = time.monotonic() + 5
        while worker.stats()["passes_total"] == before:
            assert worker.error is None
            assert time.monotonic() < deadline
            time.sleep(0.001)

    try:
        generation = worker.new_generation()
        submit(ref, generation, corners)
        shifted = cv2.warpPerspective(ref, np.float32([[1, 0, -8], [0, 1, 0], [0, 0, 1]]), (850, 1200))
        submit(shifted, generation, corners + np.float32([8, 0]))
        np.testing.assert_allclose(calls[-1][1], ref, atol=1e-5)
        assert calls[-1][0].dtype == np.float32
        assert np.any((calls[-1][0] > 0) & (calls[-1][0] < 0.7))
        submit(ref, generation, corners)
        generation = worker.new_generation(preserve_identity=True)
        submit(ref, generation, corners)
        assert worker.telemetry_snapshot()["page_identity_events"][-1]["threshold"] == 0.65
        assert calls[-1][0] is not None
        generation = worker.new_generation()
        submit(ref, generation, corners)
        assert calls[-1][0] is None
        assert worker.telemetry_snapshot()["records"][-1]["ecc_ms"] == 1.0
    finally:
        worker.stop()
