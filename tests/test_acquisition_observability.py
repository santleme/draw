"""Observabilidade da aquisicao: quadros crus esparsos + evento por aquisicao.

Motivo (docs/FALHAS.md, fantasma de 2026-11-10): a sessao ao vivo
passou ~10 s lendo o mousepad como folha, e nada do que ficou salvo
cobria a fase sem folha nem dizia, por aquisicao, o logit de
presenca, a fonte do desfecho combinado e o estado do rastreador.

Contratos:

(a) uma sessao sintetica curta grava quadros crus esparsos no
    manifesto, inclusive no trecho forcado no_sheet, com corte de
    disco declarado;
(b) cada aquisicao gera exatamente um evento com todos os campos,
    e a fonte registrada bate com o desfecho do caminho combinado
    (neural x classica x veto x presenca-so) em casos montados a mao;
    fora de sessao gravada, nenhum evento novo.
"""

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import neural as neural_module  # noqa: E402
from draw.selflearn import session as session_module  # noqa: E402
from draw.selflearn.session import SessionRecorder, sha256_file  # noqa: E402

QUAD_A = np.array([[300, 150], [980, 150], [980, 620], [300, 620]], dtype=np.float32)
QUAD_B = np.array([[310, 160], [970, 160], [970, 610], [310, 610]], dtype=np.float32)

TRACE_FIELDS = {
    "neural_ran",
    "presence_logit",
    "veto_logit",
    "neural_corners",
    "neural_verdict",
    "classic_ran",
    "classic_verdict",
    "classic_corners",
    "source",
    "combined_corners",
    "combined_verdict",
}
EVENT_FIELDS = TRACE_FIELDS | {
    "acq_seq",
    "result_seq",
    "captured_at",
    "detected_frame_index",
    "detect_ms",
    "frame_index",
    "consumed_by_tracker",
    "tracker_before",
    "tracker_after",
    "tracker_event",
}
TRACKER_FIELDS = {
    "present",
    "corners",
    "verdict",
    "missed",
    "pending",
    "pending_since",
    "big_jump_seq",
    "last_big_jump",
    "frames_since_seen",
    "last_seen_at",
    "absent_calls",
}


def _frame(value):
    return np.full((12, 16, 3), value, dtype=np.uint8)


# -- (a) quadros crus no gravador --------------------------------------


def test_raw_frames_are_sparse_and_cover_no_sheet_and_transitions(tmp_path):
    recorder = SessionRecorder(
        tmp_path, session_id="sess-raw", raw_every_n_frames=5, raw_max_in_flight=64
    )
    plan = ["no_sheet"] * 7 + ["candidate"] * 5 + ["sheet"] * 10 + ["no_sheet"] * 6
    for index, state in enumerate(plan):
        recorder.note_raw_frame(_frame(index), index, state, capture_ms=1000.0 + index)
    manifest_path = recorder.finish()
    manifest = json.loads(manifest_path.read_text())
    raw = manifest["raw_frames"]

    saved = {entry["frame_index"]: entry for entry in raw["frames"]}
    # 1 a cada 5 passadas (0, 5, ...) + toda transicao de estado.
    assert set(saved) == {0, 5, 10, 15, 20, 25, 7, 12, 22}
    assert raw["saved"] == 9
    assert raw["noted_frames"] == len(plan)
    assert raw["truncated"] is None
    assert raw["dropped_backlog"] == 0
    assert raw["policy"]["every_n_frames"] == 5
    assert {saved[i]["reason"] for i in (7, 12, 22)} == {"transition"}
    assert saved[7]["previous_state"] == "no_sheet"
    assert saved[22]["state"] == "no_sheet"
    no_sheet = [e for e in raw["frames"] if e["state"] == "no_sheet"]
    assert [e["frame_index"] for e in no_sheet] == [0, 5, 22, 25]
    for index, entry in saved.items():
        path = manifest_path.parent / entry["file"]
        assert entry["file"] == f"raw/frame_{index:06d}.png"
        assert sha256_file(path) == entry["sha256"]
        assert entry["capture_ms"] == 1000.0 + index
        assert {"t_s", "t_epoch", "state", "bytes", "shape"} <= set(entry)
        assert np.array_equal(cv2.imread(str(path)), _frame(index))
    assert manifest["write_errors"] == []


def test_raw_frames_stop_at_the_declared_caps(tmp_path):
    by_count = SessionRecorder(
        tmp_path, session_id="cap-n", raw_every_n_frames=1, raw_max_frames=2
    )
    for index in range(5):
        by_count.note_raw_frame(_frame(index), index, "no_sheet")
    raw = json.loads(by_count.finish().read_text())["raw_frames"]
    assert [e["frame_index"] for e in raw["frames"]] == [0, 1]
    assert raw["truncated"]["reason"] == "max_frames"
    assert raw["truncated"]["frame_index"] == 2

    by_bytes = SessionRecorder(
        tmp_path, session_id="cap-b", raw_every_n_frames=1, raw_max_bytes=10
    )
    by_bytes.note_raw_frame(_frame(1), 0, "no_sheet")
    raw = json.loads(by_bytes.finish().read_text())["raw_frames"]
    assert raw["frames"] == []
    assert raw["truncated"]["reason"] == "max_bytes"
    assert not (tmp_path / "sessions" / "cap-b" / "raw").exists()


def test_acquisitions_land_in_the_session_file(tmp_path):
    recorder = SessionRecorder(tmp_path, session_id="acq", acq_max_records=2)
    for seq in (1, 2, 3):
        recorder.note_acquisition({"acq_seq": seq, "source": "neural"})
    manifest_path = recorder.finish()
    manifest = json.loads(manifest_path.read_text())
    lines = (manifest_path.parent / "acquisitions.jsonl").read_text().splitlines()
    assert [json.loads(line)["acq_seq"] for line in lines] == [1, 2]
    acq = manifest["acquisitions"]
    assert acq["records"] == 2 and acq["dropped"] == 1
    assert acq["truncated"]["acq_seq"] == 3
    assert acq["sha256"] == sha256_file(manifest_path.parent / "acquisitions.jsonl")


# -- (b) fonte do desfecho combinado, casos montados a mao ------------


class _FakeNeural:
    def __init__(self, corners, logit):
        self._corners = corners
        self.last_presence_logit = None
        self._logit = logit

    def detect_sheet_with_confidence(self, _frame):
        self.last_presence_logit = self._logit
        if self._corners is None:
            return None, None
        return self._corners, "paper"


_CASES = [
    # (id, neural (cantos, logit) ou None, classico, desfecho, fonte, veredito classico)
    ("neural", (QUAD_A, 4.0), (QUAD_B, "paper"), (QUAD_A, "paper"), "neural", None),
    ("veto", (None, -5.0), (QUAD_B, "paper"), (None, None), "veto", None),
    (
        "veto-limiar",
        (None, neural_module.PRESENCE_VETO_LOGIT),
        (QUAD_B, "paper"),
        (None, None),
        "veto",
        None,
    ),
    (
        "classica-paper",
        (None, -1.0),
        (QUAD_B, "paper"),
        (QUAD_B, "paper"),
        "classic",
        "corners",
    ),
    (
        "classica-blank",
        (None, -1.0),
        (QUAD_B, "blank_maybe"),
        (QUAD_B, "blank_maybe"),
        "classic",
        "blank_maybe",
    ),
    (
        "presenca-so",
        (None, 2.0),
        (None, None),
        (None, "paper"),
        "presence_only",
        None,
    ),
    ("nada", (None, -1.0), (None, None), (None, None), "none", None),
    ("so-classico", None, (QUAD_B, "paper"), (QUAD_B, "paper"), "classic", "corners"),
    ("so-classico-nada", None, (None, None), (None, None), "none", None),
]


@pytest.mark.parametrize(
    "neural, classic, expected, source, classic_verdict",
    [case[1:] for case in _CASES],
    ids=[case[0] for case in _CASES],
)
def test_traced_source_matches_the_combined_outcome(
    monkeypatch, neural, classic, expected, source, classic_verdict
):
    classic_calls = []

    def fake_classic(_frame):
        classic_calls.append(1)
        return classic

    monkeypatch.setattr(
        app_module.paper_module, "detect_sheet_with_confidence", fake_classic
    )

    def detector():
        return None if neural is None else _FakeNeural(*neural)

    frame = _frame(0)
    plain = app_module._detect_sheet_combined(frame, detector())
    corners, verdict, trace, _extras = app_module._detect_sheet_combined_traced(
        frame, detector()
    )
    # A decisao e a mesma do caminho sem rastro.
    for got in (plain, (corners, verdict)):
        if expected[0] is None:
            assert got[0] is None
        else:
            assert np.array_equal(got[0], expected[0])
        assert got[1] == expected[1]

    assert set(trace) == TRACE_FIELDS
    assert trace["source"] == source
    assert trace["combined_verdict"] == expected[1]
    assert trace["combined_corners"] == app_module._corners_json(expected[0])
    assert trace["classic_verdict"] == classic_verdict
    assert trace["classic_ran"] == (source in {"classic", "presence_only", "none"})
    assert len(classic_calls) == 2 * int(trace["classic_ran"])
    if neural is None:
        assert trace["neural_ran"] is False
        assert trace["presence_logit"] is None and trace["veto_logit"] is None
    else:
        assert trace["presence_logit"] == neural[1]
        assert trace["veto_logit"] == neural_module.PRESENCE_VETO_LOGIT
        assert trace["neural_corners"] == app_module._corners_json(neural[0])
    json.dumps(trace)


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condicao nao chegou")


@pytest.mark.lento
def test_worker_hands_each_trace_exactly_once():
    calls = []

    def detect(frame):
        calls.append(int(frame[0, 0, 0]))
        return None, None, {"source": "none", "value": calls[-1]}

    worker = app_module._SheetDetectionWorker(detect).start()
    try:
        worker.submit(_frame(1), 10.0, 100)
        _wait(lambda: worker.poll(0) is not None)
        worker.submit(_frame(2), 11.0, 101)
        _wait(lambda: worker.poll(1) is not None)
        # Dois resultados entre polls: o 1o foi superado (latest-wins).
        seq = worker.poll(0)[0]
        assert seq == 2
        traces = worker.take_traces(seq)
        assert [t["acq_seq"] for t in traces] == [1, 2]
        assert [t["result_seq"] for t in traces] == [1, 2]
        assert [t["captured_at"] for t in traces] == [10.0, 11.0]
        assert [t["value"] for t in traces] == [1, 2]
        assert [t["detected_frame_index"] for t in traces] == [100, 101]
        assert worker.take_traces(seq) == []
        assert worker.take_traces() == []
    finally:
        worker.stop()
    assert len(calls) == 2


@pytest.mark.lento
def test_two_element_detect_keeps_the_worker_contract():
    worker = app_module._SheetDetectionWorker(lambda _f: (QUAD_A, "paper")).start()
    try:
        worker.submit(_frame(1), 5.0)
        _wait(lambda: worker.poll(0) is not None)
        result = worker.poll(0)
        assert len(result) == 5 and result[2] == "paper" and result[3] == 5.0
        assert worker.take_traces() == []
    finally:
        worker.stop()


# -- (a)+(b) pelo laco real do app -------------------------------------


class _SpyGui:
    def imshow(self, _title, _frame):
        pass

    def waitKey(self, _delay):
        return -1

    def destroyAllWindows(self):
        pass


class _ScriptedCapture:
    def __init__(self, frames, period_s=0.006):
        self._frames = frames
        self._index = 0
        self._period = period_s

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        time.sleep(self._period)
        return True, frame.copy()

    def release(self):
        pass


class _FakeInk:
    model_path = "fake.onnx"
    threshold = 0.7

    def __init__(self, _path=None):
        pass

    def load(self):
        return True

    def ink_mask(self, _warped):
        cfg = app_module.ink_module.config
        return np.zeros((cfg.PAPER_H, cfg.PAPER_W), dtype=np.uint8)


_SHEET_VALUE = 200
_LOGIT_SHEET = 6.0
_LOGIT_DESK = -8.0


def _make_fake_neural(calls):
    class _Neural:
        last_presence_logit = None

        def __init__(self, _path):
            pass

        def load(self):
            return True

        def detect_sheet_with_confidence(self, frame):
            calls.append(1)
            if int(frame[0, 0, 0]) == _SHEET_VALUE:
                self.last_presence_logit = _LOGIT_SHEET
                return QUAD_A.copy(), "paper"
            self.last_presence_logit = _LOGIT_DESK
            return None, None

    return _Neural


def _run_app(monkeypatch, save_dir, plan):
    sheet = np.full((720, 1280, 3), _SHEET_VALUE, dtype=np.uint8)
    desk = np.full((720, 1280, 3), 40, dtype=np.uint8)
    frames = [sheet if kind == "sheet" else desk for kind in plan]
    calls = []
    gui = _SpyGui()
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    monkeypatch.setattr(
        app_module, "_open_capture", lambda _s: _ScriptedCapture(frames)
    )
    monkeypatch.setattr(app_module.page_clean_module, "PageCleanInk", _FakeInk)
    monkeypatch.setattr(
        app_module.neural_module, "NeuralSheetDetector", _make_fake_neural(calls)
    )
    monkeypatch.setattr(cv2, "imshow", gui.imshow)
    monkeypatch.setattr(cv2, "waitKey", gui.waitKey)
    monkeypatch.setattr(cv2, "destroyAllWindows", gui.destroyAllWindows)
    errors = []

    def target():
        try:
            app_module.run(source=0, save_dir=str(save_dir), neural_model="fake.onnx")
        except Exception as error:  # noqa: BLE001 - o teste le o erro
            errors.append(error)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=120.0)
    assert not thread.is_alive(), "run() nao terminou"
    assert errors == []
    return calls


def _telemetry_records(save_dir):
    path = Path(save_dir) / "telemetry.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.mark.lento
def test_recorded_session_logs_raw_frames_and_one_event_per_acquisition(
    monkeypatch, tmp_path
):
    plan = ["desk"] * 70 + ["sheet"] * 70 + ["desk"] * 150
    calls = _run_app(monkeypatch, tmp_path, plan)
    (session_dir,) = (tmp_path / "sessions").iterdir()
    manifest = json.loads((session_dir / "manifest.json").read_text())

    raw = manifest["raw_frames"]
    states = {entry["state"] for entry in raw["frames"]}
    assert {"no_sheet", "sheet"} <= states
    assert raw["frames"][0]["frame_index"] == 0
    assert raw["frames"][0]["state"] == "no_sheet"
    assert any(e["reason"] == "transition" for e in raw["frames"])
    assert raw["saved"] < raw["noted_frames"]
    for entry in raw["frames"]:
        assert sha256_file(session_dir / entry["file"]) == entry["sha256"]
        assert {"capture_ms", "acq_seq", "ink_passes_total"} <= set(entry)

    records = [
        json.loads(line)
        for line in (session_dir / "acquisitions.jsonl").read_text().splitlines()
    ]
    events = [
        r for r in _telemetry_records(tmp_path) if r.get("event") == "acquisition"
    ]
    assert len(calls) > 0
    assert len(records) == len(calls) == len(events)
    assert manifest["acquisitions"]["records"] == len(calls)
    assert sorted(r["acq_seq"] for r in records) == list(range(1, len(calls) + 1))
    assert sorted(e["acq_seq"] for e in events) == list(range(1, len(calls) + 1))
    consumed = 0
    for record in records:
        assert EVENT_FIELDS <= set(record)
        if record["presence_logit"] == _LOGIT_SHEET:
            assert record["source"] == "neural"
            assert record["combined_corners"] == app_module._corners_json(QUAD_A)
        else:
            assert record["presence_logit"] == _LOGIT_DESK
            assert record["source"] == "veto"
            assert record["combined_corners"] is None
            assert record["classic_ran"] is False
        assert 0 <= record["detected_frame_index"] < len(plan)
        if record["consumed_by_tracker"]:
            consumed += 1
            assert record["detected_frame_index"] <= record["frame_index"]
            assert set(record["tracker_before"]) == TRACKER_FIELDS
            assert set(record["tracker_after"]) == TRACKER_FIELDS
        else:
            assert record["tracker_before"] is None
    assert consumed > 0
    seen = [r for r in records if r["tracker_event"] == "seen"]
    assert seen and seen[0]["tracker_before"]["present"] is False
    assert any(r["tracker_event"] == "drop" for r in records)


@pytest.mark.lento
def test_unrecorded_session_keeps_the_current_telemetry(monkeypatch, tmp_path):
    def broken_recorder(*_args, **_kwargs):
        raise RuntimeError("gravador indisponivel de proposito")

    monkeypatch.setattr(session_module, "SessionRecorder", broken_recorder)
    calls = _run_app(monkeypatch, tmp_path, ["desk"] * 20 + ["sheet"] * 20)
    assert len(calls) > 0
    records = _telemetry_records(tmp_path)
    assert not any(r.get("event") == "acquisition" for r in records)
    assert not (tmp_path / "sessions").exists()
