import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import ink_layers as ink  # noqa: E402
from tools import replay_reregister_v2 as rr  # noqa: E402

SHAPE = (120, 85)


def content():
    prob = np.zeros(SHAPE, np.float32)
    prob[30:70, 30:60] = 0.95
    prob[65:85, 50:65] = 0.95
    return prob


def record(enabled=1, *, rejected=False, baseline_rejections=0, raw="new_page"):
    identities = rr.identity_measurements(
        [{"decision": raw, "ecc_rejected": rejected}], enabled
    )
    return {
        "ecc_refine_enabled": enabled,
        "DRAW_ECC_REFINE": str(enabled),
        "vacuous": False,
        "canvas_identical_after_move": False,
        "generations_created": 1,
        "false_page_turns": 1,
        "identity_decisions": [raw],
        "new_confirmed_px_outside_tol": 100,
        "lost_confirmed_px_outside_tol": 200,
        "diagnostics": {
            "baseline_rejections": baseline_rejections,
            "identity": identities,
            "canvas": {"new_px": 100, "lost_px": 200},
            "passes": [
                {
                    "ecc_rejected": rejected,
                    "roundtrip_corner_max_canonical_px": 0.0,
                    "same_page_anchor_unchanged": True,
                }
            ],
        },
    }


@pytest.mark.parametrize("value", [None, "", "0", "1", "true", True, False, -1, 2])
def test_requires_explicit_integer_ecc(value):
    with pytest.raises(ValueError, match="explicitly"):
        rr.require_ecc(value)


def test_cli_and_api_have_no_implicit_arm():
    with pytest.raises(SystemExit) as exc:
        rr.main(["--bench-root", "/nonexistent", "--out", "/unused"])
    assert exc.value.code == 2
    with pytest.raises(TypeError, match="ecc_refine_enabled"):
        rr.run_arm("/nonexistent")


@pytest.mark.parametrize("enabled", [0, 1])
def test_explicit_environment_is_isolated_and_restored(monkeypatch, enabled):
    monkeypatch.setenv("DRAW_ECC_REFINE", "yes")
    monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", "1")
    monkeypatch.delenv("DRAW_PAGE_TURN_EVENT", raising=False)
    with pytest.raises(RuntimeError):
        with rr.configured_ecc(enabled):
            assert os.environ["DRAW_ECC_REFINE"] == str(enabled)
            assert ink._ecc_refinement_enabled() is bool(enabled)
            assert os.environ["DRAW_IDENTITY_CHAMFER"] == "0"
            assert os.environ["DRAW_PAGE_TURN_EVENT"] == "0"
            raise RuntimeError("exit")
    assert os.environ["DRAW_ECC_REFINE"] == "yes"
    assert os.environ["DRAW_IDENTITY_CHAMFER"] == "1"
    assert "DRAW_PAGE_TURN_EVENT" not in os.environ


def test_preregistration_is_frozen_and_checks_scenario_drift(tmp_path, monkeypatch):
    assert rr.validate_protocol()["decisive_mode"] == "A_fixed_anchor"
    changed = tmp_path / "changed.json"
    changed.write_bytes(rr.PROTOCOL.read_bytes() + b" ")
    with pytest.raises(ValueError, match="new named"):
        rr.validate_protocol(changed)
    monkeypatch.setattr(rr.v1, "SETTLE_FRAMES", 1)
    with pytest.raises(ValueError, match="scenario drift"):
        rr.validate_protocol()


def _bound_page_clean(enabled):
    """v2 page_clean bound to a real worker in the replay's sync path."""
    clock = rr.v1.SceneClock()
    page_clean = rr.ReplayPageClean(ecc_refine_enabled=enabled)
    worker = rr.v1.make_ink_worker(page_clean, clock)
    return page_clean, worker, clock


def test_production_defaults_remain_off(monkeypatch):
    monkeypatch.delenv("DRAW_ECC_REFINE", raising=False)
    monkeypatch.delenv("DRAW_IDENTITY_CHAMFER", raising=False)
    assert not ink._ecc_refinement_enabled()
    assert not ink._identity_chamfer_enabled()
    page_clean, _worker, _clock = _bound_page_clean(0)
    quad = rr.v1.base_quad(SHAPE)
    with pytest.raises(ValueError, match="environment"):
        page_clean.observe(content(), quad, quad)


def test_fixed_anchor_contract_not_consecutive_or_true_quad(monkeypatch):
    quad = rr.v1.base_quad(SHAPE)
    moved = rr.v1.transform_quad(quad, (120, 30), -8, 1.1)
    prob = content()
    with rr.configured_ecc(0):
        bound = [_bound_page_clean(0) for _ in range(2)]
        observations = []
        for i, (page_clean, worker, _clock) in enumerate(bound):
            worker._anchor_corners = quad.copy()
            worker._generation_passes = 3
            page_clean.previous_quad = quad + i * 300
            observations.append(page_clean.observe(prob, quad + i * 100, moved))
            assert page_clean._geometry["roundtrip_corner_max_canonical_px"] < 1e-6
            np.testing.assert_array_equal(worker._anchor_corners, quad)
        np.testing.assert_array_equal(*observations)
        assert (
            bound[0][0]._geometry["consecutive_quad_displacement_canonical_px"]
            != (bound[1][0]._geometry["consecutive_quad_displacement_canonical_px"])
        )
        registered, _, _ = ink._register_ink_probability(
            observations[0], quad, moved, None
        )
        kernel = np.ones((7, 7), np.uint8)
        assert rr.v1._mismatch(prob > 0.7, registered > 0.7, np.eye(3), kernel) == (
            0,
            0,
        )


def test_diagnostic_quad_values_cannot_change_outcome():
    original = record(raw="same_page")
    original.update(canvas_identical_after_move=True, generations_created=0)
    changed = copy.deepcopy(original)
    original["diagnostics"]["passes"][0][
        "consecutive_quad_displacement_canonical_px"
    ] = 0
    changed["diagnostics"]["passes"][0][
        "consecutive_quad_displacement_canonical_px"
    ] = 1e9
    a, b = map(rr.finalize_episode, (original, changed))
    assert a["outcome"] == b["outcome"] == "passed"
    assert a["decisive"] == b["decisive"]


class ProbIsWarp:
    threshold = 0.7

    def ink_prob(self, warped):
        return warped


def _sync_worker():
    clock = rr.v1.SceneClock()
    return rr.v1.make_ink_worker(ProbIsWarp(), clock), clock


def _sync_pass(worker, clock, prob, generation, corners):
    clock.now += 10.0
    worker.submit(prob, generation, corners)
    assert worker.pump()
    assert worker.error is None


@pytest.mark.parametrize("enabled", [0, 1])
def test_real_ecc_rejection_never_votes_new_page(monkeypatch, enabled):
    prior = content()
    other = np.zeros_like(prior)
    other[80:115, 5:35] = 0.95
    quad = rr.v1.base_quad(SHAPE)
    worker, clock = _sync_worker()
    with rr.configured_ecc(0):
        generation = worker.new_generation()
        for _ in range(4):
            _sync_pass(worker, clock, prior, generation, quad)
    calls = []

    def fail(*args, **kwargs):
        calls.append(args[3])
        raise cv2.error("forced nonconvergence")

    monkeypatch.setattr(cv2, "findTransformECC", fail)
    generation = worker.new_generation(preserve_identity=True)
    with rr.configured_ecc(enabled):
        if not enabled:
            for count in range(1, 3):
                _sync_pass(worker, clock, other, generation, quad)
                assert worker._identity_events[-1]["decision"] == (
                    "deferred_switch_consensus"
                )
                assert worker._identity_switch_votes == count
                assert worker._identity_reference is not None
        _sync_pass(worker, clock, other, generation, quad)
    event = worker._identity_events[-1]
    assert event["decision"] == ("deferred_ecc_rejected" if enabled else "new_page")
    assert event["deferred"] is bool(enabled)
    assert event["ecc_refine_enabled"] == enabled
    assert (worker._identity_reference is not None) is bool(enabled)
    assert event["ecc_rejected"] is bool(enabled)
    assert calls == ([cv2.MOTION_HOMOGRAPHY, cv2.MOTION_AFFINE] if enabled else [])
    measured = rr.identity_measurements([event], enabled)[0]
    assert measured["decision"] == ("decision_deferred" if enabled else "new_page")
    result = rr.finalize_episode(record(enabled, rejected=bool(enabled)))
    summary = rr.summarize([result], enabled)
    assert summary["decisive_denominator"] == (0 if enabled else 1)
    assert summary["identity_votes"]["new_page"] == (0 if enabled else 1)
    assert summary["failed"] == (0 if enabled else 1)
    assert summary["diagnostic_rejected_episode_passes"] == enabled
    if enabled:
        assert result["passes_gate"] is None
        assert result["decisive"]["new_confirmed_px_outside_tol"] is None
        assert summary["new_confirmed_px_outside_tol"] == 0


def test_baseline_rejection_defers_clean_episode_and_all_votes():
    result = rr.finalize_episode(record(baseline_rejections=1, raw="same_page"))
    assert result["outcome"] == "decision_deferred"
    assert result["decisive"]["identity_votes"] == []
    assert rr.summarize([result], 1)["identity_votes"] == {
        "same_page": 0,
        "new_page": 0,
    }


@pytest.mark.parametrize("outcome", ["vacuous", "invalid"])
def test_nondecisions_do_not_enter_any_denominator(outcome):
    item = record()
    if outcome == "vacuous":
        item["vacuous"] = True
        item["diagnostics"]["passes"] = []
    else:
        item["diagnostics"]["passes"][0]["same_page_anchor_unchanged"] = False
    result = rr.finalize_episode(item)
    assert result["outcome"] == outcome
    summary = rr.summarize([result], 1)
    assert summary["decisive_denominator"] == summary["failed"] == 0
    assert summary["identity_votes"] == {"same_page": 0, "new_page": 0}


def arms():
    shared = {
        "protocol_sha256": "p",
        "protocol_id": "p",
        "runner_sha256": "r",
        "production_source_sha256": {"helper": "h"},
        "runtime": {"opencv": "x"},
        "input": {"input_fingerprint": "same-page"},
        "episodes": [{"page_id": "p1", "scenario": "move", "case": "c", "hand": "h"}],
    }
    return [
        dict(copy.deepcopy(shared), ecc_refine_enabled=i, groups={"state": i})
        for i in (0, 1)
    ]


def test_no_mixed_state_tables_or_mismatched_page_comparison():
    off, on = arms()
    result = rr.compare_arms(off, on)
    assert list(result["arms"]) == ["DRAW_ECC_REFINE=0", "DRAW_ECC_REFINE=1"]
    a, b = rr.finalize_episode(record(0)), rr.finalize_episode(record(1))
    with pytest.raises(ValueError, match="pool"):
        rr.summarize([a, b], 0)
    for mutate in (
        lambda a: a["input"].update(input_fingerprint="other-page"),
        lambda a: a.update(runner_sha256="changed"),
        lambda a: a["episodes"][0].update(hand="other"),
    ):
        changed = copy.deepcopy(on)
        mutate(changed)
        with pytest.raises(ValueError, match="unpaired"):
            rr.compare_arms(off, changed)
    with pytest.raises(ValueError, match="distinct"):
        rr.compare_arms(off, off)


def test_full_synthetic_contract_has_paired_scenarios_not_implicit_calibration(capsys):
    drawing, settled = rr.v1.synthetic_pages(SHAPE, stages=3)
    provenance = {"page_id": "unit-synthetic", "input_fingerprint": "unit-fixture"}
    protocol = rr.validate_protocol()
    with rr.configured_ecc(0):
        report = rr._run_page(drawing, settled, provenance, 0, protocol)
    assert report["groups"]["move_same_canvas"]["episodes"] == 20
    assert report["groups"]["control_static"]["episodes"] == 4
    assert report["groups"]["sway_negative"]["episodes"] == 8
    assert len(report["episodes"]) == 32
    for group in report["groups"].values():
        assert group["ecc_refine_enabled"] == 0
        assert sum(group[outcome] for outcome in rr.OUTCOMES) == group["episodes"]
        assert group["decisive_denominator"] == group["passed"] + group["failed"]
    assert report["diagnostics"]["ecc_rejected_passes"] == 0
    assert all(r["page_id"] == "unit-synthetic" for r in report["episodes"])


def test_bench_loader_validates_cache_without_inference(tmp_path):
    for name in ("cache", "warps", "model"):
        (tmp_path / name).mkdir()
    model = b"test-model"
    model_hash = hashlib.sha256(model).hexdigest()
    (tmp_path / "model/page_clean_v2_fixed.onnx").write_bytes(model)
    img = np.zeros((1216, 864, 3), np.uint8)
    key = rr.sha256(
        bytes.fromhex(model_hash) + repr(img.shape).encode() + img.tobytes()
    )
    cache_path = tmp_path / f"cache/{key}.npy"
    np.save(cache_path, np.zeros(img.shape[:2], np.float32))
    first = tmp_path / "warps/f01_warp.png"
    cv2.imwrite(str(first), img)
    for i in range(2, 31):
        (tmp_path / f"warps/f{i:02d}_warp.png").hardlink_to(first)
    index = {
        f"warp:f{i:02d}": {
            "key": key,
            "shape": list(img.shape),
            "model_sha256": model_hash,
        }
        for i in range(1, 31)
    }
    index_raw = json.dumps(index).encode()
    (tmp_path / "cache/cache_index.json").write_bytes(index_raw)
    protocol = {
        "baseline_input": {
            "cache_index_sha256": rr.sha256(index_raw),
            "model_sha256": model_hash,
            "package": "unit-bench.zip",
        }
    }
    package_path = tmp_path / "registered.zip"

    def register_package():
        with zipfile.ZipFile(
            package_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for directory in ("cache", "warps", "model"):
                for path in (tmp_path / directory).iterdir():
                    archive.write(path, f"unit-bench/{path.relative_to(tmp_path)}")
        protocol["baseline_input"]["package_sha256"] = rr.sha256(
            package_path.read_bytes()
        )

    register_package()
    drawing, settled, provenance = rr.load_bench(
        tmp_path, protocol, package_path=package_path
    )
    assert len(drawing) == 30 and len(settled) == 3
    assert provenance["cache_hits"] == 30 and provenance["cache_misses"] == 0
    assert len(provenance["entries"]) == 30
    np.save(cache_path, np.ones(img.shape[:2], np.float32))
    with pytest.raises(ValueError, match="package"):
        rr.load_bench(tmp_path, protocol, package_path=package_path)
    np.save(cache_path, np.zeros((10, 10), np.float32))
    register_package()
    with pytest.raises(ValueError, match="native"):
        rr.load_bench(tmp_path, protocol, package_path=package_path)
    cache_path.unlink()
    with pytest.raises(FileNotFoundError):
        rr.load_bench(tmp_path, protocol, package_path=package_path)
    package_path.write_bytes(package_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="package SHA-256"):
        rr.load_bench(tmp_path, protocol, package_path=package_path)


def _threaded_pass(real, prob, generation, corners):
    with real._condition:
        before = real._passes_total
        real.submit(prob, generation, corners)
        assert real._condition.wait_for(
            lambda: real._passes_total > before or real._stopped, timeout=5
        )
    assert real.error is None


def _assert_same_worker_state(real, sync, generation):
    np.testing.assert_array_equal(
        real.poll(generation, 0)[1], sync.poll(generation, 0)[1]
    )
    if sync._anchor_corners is None:
        assert real._anchor_corners is None
    else:
        np.testing.assert_array_equal(real._anchor_corners, sync._anchor_corners)
    assert real._identity_defer_count == sync._identity_defer_count
    assert real._identity_switch_votes == sync._identity_switch_votes
    assert (real._identity_reference is None) == (sync._identity_reference is None)
    np.testing.assert_array_equal(
        real._accumulator._window[-1], sync._accumulator._window[-1]
    )
    real_events = list(real._identity_events)
    sync_events = list(sync._identity_events)
    assert len(real_events) == len(sync_events)
    for actual, expected in zip(real_events, sync_events):
        for key in (
            "decision",
            "deferred",
            "quarantine_reason",
            "quarantine_frames",
            "ecc_rejected",
            "ecc_refine_enabled",
        ):
            assert actual[key] == expected[key]


@pytest.mark.parametrize("rejected_passes", [2, 4])
def test_v2_matches_production_quarantine_recovery_and_timeout(
    monkeypatch, rejected_passes
):
    monkeypatch.setattr(ink, "_PAGE_CLEAN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(ink, "_IDLE_PROBE_SECONDS", 0)
    monkeypatch.setattr(ink, "_PAGE_IDENTITY_MAX_DEFERRALS", 3)
    rejecting = False

    def refine(reference, observed, hand_mask=None):
        return (
            observed,
            np.eye(3),
            {
                "ecc_rejected": rejecting,
                "ecc_converged": False,
                "ecc_mode": "affine",
                "ecc_ms": 0.0,
            },
        )

    monkeypatch.setattr(ink, "_refine_ink_ecc", refine)
    prob = content()
    other = np.flipud(np.fliplr(prob)).copy()
    quad = rr.v1.base_quad(SHAPE)
    sync, _clock = _sync_worker()
    real = ink._PageCleanWorker(ProbIsWarp()).start()
    try:
        with rr.configured_ecc(1):
            generation = sync.new_generation()
            assert real.new_generation() == generation
            for i in range(4 + rejected_passes + 4):
                if i == 4:
                    generation = sync.new_generation(preserve_identity=True)
                    assert real.new_generation(preserve_identity=True) == generation
                rejecting = 4 <= i < 4 + rejected_passes
                current = other if rejecting else prob
                _sync_pass(sync, _clock, current, generation, quad)
                _threaded_pass(real, current, generation, quad)
                _assert_same_worker_state(real, sync, generation)
            assert sync._identity_events[-1]["decision"] == (
                "same_page" if rejected_passes == 2 else "quarantine_timeout"
            )
    finally:
        real.stop()


@pytest.mark.parametrize("enabled,interrupt", [(0, False), (1, False), (1, True)])
@pytest.mark.parametrize("blank", [False, True])
def test_v2_matches_production_switch_consensus(monkeypatch, enabled, interrupt, blank):
    monkeypatch.setattr(ink, "_PAGE_CLEAN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(ink, "_IDLE_PROBE_SECONDS", 0)
    rejecting = False

    def refine(reference, observed, hand_mask=None):
        return (
            observed,
            np.eye(3),
            {
                "ecc_rejected": rejecting,
                "ecc_converged": False,
                "ecc_mode": "affine",
                "ecc_ms": 0.0,
            },
        )

    monkeypatch.setattr(ink, "_refine_ink_ecc", refine)
    prob = content()
    other = np.zeros_like(prob) if blank else np.flipud(np.fliplr(prob)).copy()
    quad = rr.v1.base_quad(SHAPE)
    sync, clock = _sync_worker()
    real = ink._PageCleanWorker(ProbIsWarp()).start()
    try:
        with rr.configured_ecc(enabled):
            generation = sync.new_generation()
            assert real.new_generation() == generation
            recovery = (
                [False, False, True, False, False, False] if interrupt else [False] * 3
            )
            for i, rejected in enumerate([False] * 4 + recovery):
                if i == 4:
                    generation = sync.new_generation(preserve_identity=True)
                    assert real.new_generation(preserve_identity=True) == generation
                rejecting = rejected
                current = prob if i < 4 else other
                _sync_pass(sync, clock, current, generation, quad)
                _threaded_pass(real, current, generation, quad)
                _assert_same_worker_state(real, sync, generation)
                if i >= 4:
                    expected = sync._identity_events[-1]
                    if expected["decision"] == "deferred_switch_consensus":
                        assert not np.any(sync._accumulator._window[-1])
                        assert not sync._at_rest()
                        assert type(expected["ecc_refine_enabled"]) is int
            assert sync._identity_events[-1]["decision"] == "new_page"
            assert sync._identity_switch_votes == sync._identity_defer_count == 0
            assert sync._identity_reference is None
    finally:
        real.stop()


@pytest.mark.lento
@pytest.mark.parametrize("enabled", [0, 1])
def test_v2_sync_matches_threaded_passes_and_fixed_same_page_anchor(
    monkeypatch, enabled
):
    monkeypatch.setattr(ink, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink, "_IDLE_PROBE_SECONDS", 0.005)
    prob = content()
    quad = rr.v1.base_quad(SHAPE)
    page_clean, sync, clock = _bound_page_clean(enabled)
    observed = []
    original = page_clean.observe

    def capture(*args):
        value = original(*args)
        observed.append(value)
        return value

    page_clean.observe = capture
    real = ink._PageCleanWorker(ProbIsWarp()).start()
    try:
        with rr.configured_ecc(enabled):
            generation = sync.new_generation()
            assert real.new_generation() == generation
            for i in range(9):
                if i == 6:
                    generation = sync.new_generation(preserve_identity=True)
                    assert real.new_generation(preserve_identity=True) == generation
                corners = quad if i < 6 else quad + np.float32([3, -2])
                clock.now += 10.0
                sync.submit(rr.v1.SceneWarp(prob, quad, corners), generation, corners)
                assert sync.pump() and sync.error is None
                # The threaded worker gets the exact observation the
                # replay's sync pass consumed.
                before = real.stats()["passes_total"]
                real.submit(observed[-1], generation, corners)
                deadline = time.monotonic() + 5
                while real.stats()["passes_total"] == before:
                    assert time.monotonic() < deadline
                    time.sleep(0.001)
                _, canvas = real.poll(generation, 0)
                np.testing.assert_array_equal(canvas, sync.poll(generation, 0)[1])
            real_event = real.telemetry_snapshot()["page_identity_events"][-1]
            assert (
                real_event["decision"]
                == sync._identity_events[-1]["decision"]
                == "same_page"
            )
    finally:
        real.stop()
