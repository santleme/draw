import json

import cv2
import numpy as np
import pytest

from draw import ink_layers as layers
from draw.selflearn.session import SessionRecorder
from tests.test_replay_reregister import rr, _ProbIsWarp


def ink():
    prob = np.zeros((100, 120), np.float32)
    prob[15:45, 15:55] = 0.95
    prob[65:75, 65:85] = 0.95
    return prob


def confirmed(acc, prob):
    for _ in range(16):
        result = acc.add(prob, np.ones(prob.shape, bool))
    return result["confirmed"].copy()


@pytest.mark.parametrize("e3", ["0", "1"])
@pytest.mark.parametrize("fast", ["0", "1"])
def test_erasure_is_published_and_does_not_resurrect(monkeypatch, e3, fast):
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    monkeypatch.setenv("DRAW_INK_VOTE_E3", e3)
    monkeypatch.setenv("DRAW_INK_VOTE_FAST", fast)
    acc = layers._LayeredInkAccumulator()
    prob = ink()
    original = confirmed(acc, prob)
    erased = prob.copy()
    erased[65:75, 65:85] = 0
    visible = np.ones(prob.shape, bool)
    for _ in range(layers._REMOVAL_PASSES - 1):
        result = acc.add(erased, visible)
        np.testing.assert_array_equal(result["confirmed"], original)
    result = acc.add(erased, visible)
    assert result["confirmed_changed"] and result["removed_confirmed_px"] == 200
    for _ in range(20):
        result = acc.add(erased, visible)
        assert not result["confirmed"][65:75, 65:85].any()
        assert not result["provisional"][65:75, 65:85].any()
    assert np.all(result["confirmed"][15:45, 15:55] == 255)
    confirmed(acc, prob)
    assert np.all(acc._confirmed[65:75, 65:85] == 255)  # redraw still works


@pytest.mark.parametrize(
    "mode", ["occlusion", "no_visibility", "turn", "registration", "flicker"]
)
def test_negative_evidence_fails_closed(monkeypatch, mode):
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    acc = layers._LayeredInkAccumulator()
    prob = ink()
    original = confirmed(acc, prob)
    erased = prob.copy()
    erased[65:75, 65:85] = 0
    visible = np.ones(prob.shape, bool)
    if mode == "occlusion":
        visible[65:75, 65:85] = False
    elif mode == "no_visibility":
        visible = None
    elif mode == "turn":
        erased[:] = 0
    for i in range(20):
        current = prob if mode == "flicker" and i % 2 else erased
        result = acc.add(current, visible, removal_allowed=mode != "registration")
    np.testing.assert_array_equal(result["confirmed"], original)


def test_removal_off_is_byte_identical_and_monotone(monkeypatch):
    results = []
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv("DRAW_INK_REMOVAL", raising=False)
        else:
            monkeypatch.setenv("DRAW_INK_REMOVAL", value)
        acc = layers._LayeredInkAccumulator()
        original = confirmed(acc, ink())
        for _ in range(20):
            result = acc.add(np.zeros_like(ink()), np.ones(ink().shape, bool))
        np.testing.assert_array_equal(original, result["confirmed"])
        assert "removed_confirmed_px" not in result
        results.append(result["confirmed"])
    np.testing.assert_array_equal(*results)


def drive(sequence, monkeypatch, flag="1"):
    monkeypatch.setenv("DRAW_REG_CONTENT_GUARD", flag)
    monkeypatch.setenv("DRAW_ANCHOR_ARBITRATION", "0")
    monkeypatch.setenv("DRAW_ANCHOR_FOLLOW", "0")
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    clock = rr.SceneClock()
    worker = rr.make_ink_worker(_ProbIsWarp(), clock)
    gen = worker.new_generation()
    for prob, corners in sequence:
        clock.now += 1
        worker.submit(prob, gen, corners, np.ones(prob.shape, bool))
        assert worker.pump()
        assert worker.error is None
    return worker


def test_page_movement_does_not_reconfirm_duplicate_and_allows_new_ink(monkeypatch):
    prob = ink()
    base = rr.base_quad(prob.shape)
    moved = base + np.float32([80, 0])
    initial = [(prob, base)] * 8
    worker = drive(initial, monkeypatch)
    original = worker._accumulator._confirmed.copy()
    worker = drive(initial + [(prob, moved)] * 8, monkeypatch)
    np.testing.assert_array_equal(worker._accumulator._confirmed, original)
    records = worker.telemetry_snapshot()["records"]
    assert sum(bool(r.get("reg_content_blocked")) for r in records) == 2
    assert sum(bool(r.get("reg_content_identity_won")) for r in records) == 1
    more = prob.copy()
    more[80:90, 15:35] = 0.95
    worker = drive(initial + [(prob, moved)] * 8 + [(more, moved)] * 8, monkeypatch)
    assert (
        np.count_nonzero(worker._accumulator._confirmed)
        == np.count_nonzero(original) + 200
    )


def test_spurious_geometry_is_not_reanchored(monkeypatch):
    prob = ink()
    base = rr.base_quad(prob.shape)
    moved = base + np.float32([10, 0])
    matrix = layers._registration_matrix(base, moved, prob.shape)
    raw = cv2.warpPerspective(prob, np.linalg.inv(matrix), (120, 100))
    worker = drive([(prob, base)] * 8 + [(raw, moved)] * 8, monkeypatch)
    np.testing.assert_array_equal(worker._anchor_corners, base)
    assert not any(
        r.get("reg_content_identity_won")
        for r in worker.telemetry_snapshot()["records"]
    )


def test_content_guard_off_matches_unset(monkeypatch):
    prob = ink()
    base = rr.base_quad(prob.shape)
    seq = [(prob, base)] * 8 + [(prob, base + [20, 0])] * 8
    first = drive(seq, monkeypatch, "0")._accumulator._confirmed.copy()
    second = drive(seq, monkeypatch, "")._accumulator._confirmed.copy()
    np.testing.assert_array_equal(first, second)


def test_live_stream_preserves_more_than_worker_tail_and_bounds_records(tmp_path):
    recorder = SessionRecorder(tmp_path, session_id="test", acq_max_records=300)
    for sequence in range(1, 321):
        record = {
            "sequence": sequence,
            "anchor_arbitration_enabled": sequence % 2 == 0,
            "new_confirmed_px": 2,
            "removed_confirmed_px": 1,
        }
        recorder.note_ink_telemetry({"records": [record]})
        recorder.note_ink_telemetry({"records": [record]})
    path = recorder.finish()
    manifest = json.loads(path.read_text())
    records = [
        json.loads(s)
        for s in (path.parent / "ink_passes.jsonl").read_text().splitlines()
    ]
    assert len(records) == 300 and manifest["ink_passes"]["dropped"] == 20
    assert records[0]["sequence"] == 1
    assert manifest["ink_passes"]["sha256"]
    assert not recorder.errors


def test_live_telemetry_explicitly_records_disabled_flag(monkeypatch):
    prob = ink()
    worker = drive([(prob, rr.base_quad(prob.shape))] * 8, monkeypatch, "0")
    record = worker.telemetry_snapshot()["records"][-1]
    assert record["anchor_arbitration_enabled"] is False
    assert record["sequence"] == 8 and record["pass_gap_s"] == 1
    assert record["registration_is_identity"]
    assert "new_confirmed_px" in record and "removed_confirmed_px" in record


def test_worker_publishes_erasure_through_the_real_path(monkeypatch):
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    prob = ink()
    base = rr.base_quad(prob.shape)
    erased = prob.copy()
    erased[65:75, 65:85] = 0
    worker = drive([(prob, base)] * 16 + [(erased, base)] * 10, monkeypatch)
    assert not worker._accumulator._confirmed[65:75, 65:85].any()
    assert np.all(worker._accumulator._confirmed[15:45, 15:55])
    records = worker.telemetry_snapshot()["records"]
    assert sum(r["removed_confirmed_px"] for r in records) == 200
    assert not worker._at_rest() or worker._accumulator._removal_streak.max() == 0


def test_content_guard_defers_to_identity_and_arbitration(monkeypatch):
    prob = ink()
    base = rr.base_quad(prob.shape)
    worker = drive([(prob, base)] * 8, monkeypatch)
    before = worker._accumulator._confirmed.copy()
    registered = np.roll(prob, 25, axis=1)
    visible = np.ones(prob.shape, bool)
    for _ in range(5):
        _prob, _visible, info = worker._content_registration_guard(
            prob, registered, visible, visible, base + [80, 0], True
        )
        assert not info["reg_content_blocked"]
        assert not worker._content_guard_quads
    np.testing.assert_array_equal(before, worker._accumulator._confirmed)
    np.testing.assert_array_equal(base, worker._anchor_corners)


def test_stream_counts_missing_tail_even_after_the_cap(tmp_path):
    recorder = SessionRecorder(tmp_path, session_id="bounded", acq_max_records=1)
    recorder.note_ink_telemetry({"records": [{"sequence": 2}]})
    recorder.note_ink_telemetry({"records": [{"sequence": 10}]})
    manifest = json.loads(recorder.finish().read_text())
    assert manifest["ink_passes"]["records"] == 1
    assert manifest["ink_passes"]["dropped"] == 9


def test_local_erasure_does_not_reopen_e3_initial_confirmation(monkeypatch):
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    monkeypatch.setenv("DRAW_INK_VOTE_E3", "1")
    acc = layers._LayeredInkAccumulator()
    prob = ink()
    confirmed(acc, prob)
    noisy = prob.copy()
    noisy[85:88, 85:88] = 0.95  # 9 px: rejected by E3's new-component rule.
    visible = np.ones(prob.shape, bool)
    for _ in range(10):
        acc.add(noisy, visible)
    assert not acc._confirmed[85:88, 85:88].any()
    erased = noisy.copy()
    erased[65:75, 65:85] = 0
    for _ in range(16):
        acc.add(erased, visible)
    assert not acc._confirmed[65:75, 65:85].any()
    assert not acc._confirmed[85:88, 85:88].any()


@pytest.mark.parametrize("height,width", [(5, 10), (10, 20), (40, 80)])
@pytest.mark.parametrize("fast", ["0", "1"])
def test_e3_lateral_tracks_allow_erasure_then_redraw(monkeypatch, height, width, fast):
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    monkeypatch.setenv("DRAW_INK_VOTE_E3", "1")
    monkeypatch.setenv("DRAW_INK_VOTE_FAST", fast)
    acc = layers._LayeredInkAccumulator()
    initial = np.zeros((260, 400), np.float32)
    initial[10:180, 10:200] = 0.95
    confirmed(acc, initial)
    with_side = initial.copy()
    region = np.s_[200 : 200 + height, 250 : 250 + width]
    with_side[region] = 0.95
    confirmed(acc, with_side)
    assert np.all(acc._confirmed[region] == 255)
    confirmed(acc, initial)
    assert not acc._confirmed[region].any()
    confirmed(acc, with_side)
    assert np.all(acc._confirmed[region] == 255)
