"""Relógio determinístico: fronteiras por fase sem depender de hardware."""

import numpy as np
import pytest

from draw import neural
from test_neural_onnx_contract import _Session


def _outputs(contract):
    if contract == "semantic":
        return {"logits": np.zeros((1, 4, 8, 8), np.float32)}
    channels = 2 if contract == "legacy" else 4
    return {
        "mask_logits": np.zeros((1, channels, 8, 8), np.float32),
        "presence_logit": np.array([2.5], np.float32),
    }


@pytest.mark.parametrize("contract", ["legacy", "hybrid", "semantic"])
def test_timing_accumulates_each_phase_without_changing_outputs(monkeypatch, contract):
    monkeypatch.setenv("DRAW_TIMING", "1")
    detector = neural.NeuralSheetDetector("unused.onnx")
    detector._session = _Session(_outputs(contract))
    frame = np.zeros((16, 24, 3), np.uint8)
    ticks = iter((0.0, 0.001, 0.011, 0.013, 1.0, 1.003, 1.023, 1.027))
    events = []

    def clock():
        events.append("clock")
        return next(ticks)

    preprocess = neural._frame_to_tensor
    run = detector._session.run
    resize = neural.cv2.resize

    def timed_preprocess(*args):
        events.append("preprocess")
        return preprocess(*args)

    def timed_run(*args):
        events.append("session_run")
        return run(*args)

    def timed_resize(*args, **kwargs):
        events.append("resize")
        return resize(*args, **kwargs)

    monkeypatch.setattr(neural.time, "perf_counter", clock)
    monkeypatch.setattr(neural, "_frame_to_tensor", timed_preprocess)
    monkeypatch.setattr(detector._session, "run", timed_run)
    monkeypatch.setattr(neural.cv2, "resize", timed_resize)
    first_presence, first_mask = detector.predict(frame)
    second_presence, second_mask = detector.predict(frame)

    assert (
        events
        == [
            "clock",
            "preprocess",
            "resize",
            "clock",
            "session_run",
            "clock",
            "resize",
            "clock",
        ]
        * 2
    )
    for phase, total, maximum in (
        ("preprocess", 4.0, 3.0),
        ("session_run", 30.0, 20.0),
        ("postprocess", 6.0, 4.0),
    ):
        assert detector.get_timing_stats()[phase] == pytest.approx(
            {"count": 2, "sum_ms": total, "max_ms": maximum, "mean_ms": total / 2}
        )
    snapshot = detector.get_timing_stats()
    snapshot["session_run"]["count"] = -1
    assert detector.get_timing_stats()["session_run"]["count"] == 2
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_mask.shape == frame.shape[:2]
    assert first_mask.dtype == np.uint8
    assert detector.last_presence_logit == first_presence == second_presence

    monkeypatch.delenv("DRAW_TIMING")
    untimed = neural.NeuralSheetDetector("unused.onnx")
    untimed._session = _Session(_outputs(contract))
    presence, mask = untimed.predict(frame)
    assert presence == first_presence
    np.testing.assert_array_equal(mask, first_mask)


@pytest.mark.parametrize("setting", [None, "0", "true"])
def test_disabled_timing_never_reads_clock_or_logs(monkeypatch, capsys, setting):
    if setting is None:
        monkeypatch.delenv("DRAW_TIMING", raising=False)
    else:
        monkeypatch.setenv("DRAW_TIMING", setting)
    detector = neural.NeuralSheetDetector("unused.onnx")
    detector._session = _Session(_outputs("semantic"))

    def no_clock():
        pytest.fail("timing desativado não deve ler o relógio")

    monkeypatch.setattr(neural.time, "perf_counter", no_clock)
    detector.predict(np.zeros((16, 16, 3), np.uint8))

    assert all(
        stats == {"count": 0, "sum_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
        for stats in detector.get_timing_stats().values()
    )
    assert capsys.readouterr().out == ""


def test_timing_prints_only_at_periodic_boundary(monkeypatch, capsys):
    monkeypatch.setenv("DRAW_TIMING", "1")
    detector = neural.NeuralSheetDetector("unused.onnx")
    detector._session = _Session(_outputs("semantic"))
    ticks = iter(range(neural._TIMING_LOG_EVERY * 4))
    monkeypatch.setattr(neural.time, "perf_counter", lambda: next(ticks))
    frame = np.zeros((16, 16, 3), np.uint8)
    for _ in range(neural._TIMING_LOG_EVERY - 1):
        detector.predict(frame)
    assert capsys.readouterr().out == ""

    detector.predict(frame)
    message = capsys.readouterr().out
    assert f"DRAW_TIMING frames={neural._TIMING_LOG_EVERY}" in message
    for phase in ("preprocess", "session_run", "postprocess"):
        assert f"{phase}: mean=1000.000 ms max=1000.000 ms" in message


def test_failed_prediction_and_empty_frame_do_not_count(monkeypatch):
    monkeypatch.setenv("DRAW_TIMING", "1")
    detector = neural.NeuralSheetDetector("unused.onnx")
    detector._session = _Session({"logits": np.zeros((1, 3, 8, 8), np.float32)})
    with pytest.raises(ValueError, match="precisa ter forma"):
        detector.predict(np.zeros((16, 16, 3), np.uint8))
    assert detector(None) == (None, None, None)
    assert detector.last_presence_logit is None
    assert all(stats["count"] == 0 for stats in detector.get_timing_stats().values())
