"""Testes do relatorio de diagnostico por fonte, so com mocks.

Nenhum teste abre dispositivo real, depende de /dev/video* ou faz
acao no sistema. A fabrica de captura e injetada em probe_source()
e probe_all(), no mesmo padrao FakeCapture de test_capture.py.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import app as app_module  # noqa: E402
from draw import capture as capture_module  # noqa: E402
from draw import diagnose as diagnose_module  # noqa: E402


class FakeCapture:
    """VideoCapture falso com props negociadas e frames programaveis."""

    def __init__(self, source, backend=None, opened=True, frames=(),
                 props=None):
        self.source = source
        self.backend = backend
        self._opened = opened
        self._frames = list(frames)
        self._props = props or {}
        self.settings = {}
        self.released = False

    def isOpened(self):
        return self._opened and not self.released

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def get(self, prop):
        return self._props.get(prop, 0)

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def _frame():
    return np.zeros((2, 2, 3), dtype=np.uint8)


def _mjpg_props():
    return {
        cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FOURCC: float(
            cv2.VideoWriter_fourcc(*"MJPG")
        ),
    }


def _ok_factory(source, backend=None):
    return FakeCapture(
        source, backend, opened=True,
        frames=[_frame(), _frame(), _frame()],
        props=_mjpg_props(),
    )


def test_probe_three_frames_ok_records_backend_and_props():
    report = diagnose_module.probe_source(
        0, capture_factory=_ok_factory
    )
    assert report.opened is True
    assert report.frames_ok == 3
    assert report.error_kind is None
    assert report.ok is True
    assert report.backend_used == "CAP_V4L2"
    assert report.width == 1280
    assert report.height == 720
    assert report.fps == 30.0
    assert report.fourcc == "MJPG"
    assert report.latency_ms is not None
    assert report.latency_ms >= 0
    assert report.kind == "local"


def test_probe_passes_open_and_read_timeouts_to_backend():
    calls = []

    def factory(source, backend=None, params=None):
        calls.append((source, backend, params))
        return _ok_factory(source, backend)

    report = diagnose_module.probe_source(
        "http://camera.local/video",
        capture_factory=factory,
        open_timeout_ms=1400,
        read_timeout_ms=1800,
    )

    assert report.ok
    assert calls == [(
        "http://camera.local/video",
        cv2.CAP_ANY,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1400,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1800,
        ],
    )]


def test_probe_frame_deadline_starts_after_open():
    clock = [0.0]

    def factory(source, backend=None, params=None):
        clock[0] = 10.0
        return _ok_factory(source, backend)

    report = diagnose_module.probe_source(
        "recording.mp4",
        read_frames=1,
        deadline=0.1,
        capture_factory=factory,
        monotonic=lambda: clock[0],
    )

    assert report.ok


def test_probe_does_not_read_when_backend_cannot_confirm_timeout(monkeypatch):
    class UnboundedCapture(FakeCapture):
        def __init__(self):
            super().__init__("camera.mp4")
            self.read_calls = 0

        def read(self):
            self.read_calls += 1
            raise AssertionError("must not read without a timeout")

    capture = UnboundedCapture()
    monkeypatch.setattr(capture_module, "open_capture", lambda *args, **kwargs: capture)

    report = diagnose_module.probe_source("camera.mp4")

    assert report.error_kind == "timeout_unsupported"
    assert "timeout de leitura" in report.hint
    assert capture.read_calls == 0
    assert capture.released is True


def test_probe_does_not_retry_real_open_without_timeout_params(monkeypatch):
    calls = []

    def reject_timeout_params(source, backend=None, params=None):
        calls.append((source, backend, params))
        if params is not None:
            raise TypeError("backend rejected timeout params")
        return FakeCapture(source, backend)

    monkeypatch.setattr(
        diagnose_module.cv2, "VideoCapture", reject_timeout_params
    )

    report = diagnose_module.probe_source(0, read_timeout_ms=1000)

    assert report.error_kind == "timeout_unsupported"
    assert len(calls) == len(capture_module.backends_for_platform())
    assert all(call[2] is not None for call in calls)


def test_probe_opens_but_no_frames_reports_no_frames():
    report = diagnose_module.probe_source(
        0, deadline=0.05,
        capture_factory=lambda source, backend=None: FakeCapture(
            source, backend, opened=True, frames=[],
        ),
    )
    assert report.opened is True
    assert report.frames_ok == 0
    assert report.error_kind == "no_frames"
    assert report.ok is False
    assert "nenhum quadro" in report.hint


def test_probe_partial_frames_reports_no_frames():
    report = diagnose_module.probe_source(
        0, deadline=0.05,
        capture_factory=lambda source, backend=None: FakeCapture(
            source, backend, opened=True, frames=[_frame()],
        ),
    )
    assert report.frames_ok == 1
    assert report.error_kind == "no_frames"
    assert report.ok is False


def test_probe_read_exception_reports_failure_and_releases_capture():
    class BrokenReadCapture(FakeCapture):
        def read(self):
            raise RuntimeError("read failed")

    capture = BrokenReadCapture("recording.mp4")
    report = diagnose_module.probe_source(
        "recording.mp4",
        capture_factory=lambda source, backend=None: capture,
    )

    assert report.opened is True
    assert report.error_kind == "read_error"
    assert "read failed" in report.hint
    assert capture.released is True


@pytest.mark.parametrize("kind,hint_fragment", [
    ("not_found", "/dev/video7"),
    ("permission", "permissao"),
    ("busy", "Outro programa"),
])
def test_probe_each_open_error_kind_with_hint(monkeypatch, kind, hint_fragment):
    monkeypatch.setattr(
        capture_module, "diagnose_local",
        lambda index: (kind, f"hint de teste: {hint_fragment}"),
    )
    report = diagnose_module.probe_source(
        7,
        capture_factory=lambda source, backend=None: FakeCapture(
            source, backend, opened=False
        ),
    )
    assert report.opened is False
    assert report.error_kind == kind
    assert hint_fragment in report.hint


def test_probe_missing_local_node_reports_not_found():
    report = diagnose_module.probe_source(
        7,
        capture_factory=lambda source, backend=None: FakeCapture(
            source, backend, opened=False
        ),
    )
    assert report.error_kind == "not_found"
    assert "/dev/video7" in report.hint


def test_probe_network_error_kind_with_hint():
    report = diagnose_module.probe_source(
        "http://192.168.0.10:8080/video",
        capture_factory=lambda source, backend=None: FakeCapture(
            source, backend, opened=False
        ),
    )
    assert report.kind == "network"
    assert report.error_kind == "network"
    assert report.hint


def test_probe_opens_only_on_second_backend():
    def factory(source, backend=None):
        return FakeCapture(
            source, backend, opened=(backend == cv2.CAP_ANY),
            frames=[_frame(), _frame(), _frame()],
        )

    report = diagnose_module.probe_source(
        0, capture_factory=factory,
        backends=[cv2.CAP_V4L2, cv2.CAP_ANY],
    )
    assert report.opened is True
    assert report.backend_used == "CAP_ANY"
    assert report.frames_ok == 3


def test_probe_deadline_stops_without_hanging():
    started = time.monotonic()
    report = diagnose_module.probe_source(
        0, deadline=0.05,
        capture_factory=lambda source, backend=None: FakeCapture(
            source, backend, opened=True, frames=[],
        ),
    )
    elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert report.error_kind == "no_frames"
    assert report.frames_ok == 0


def test_probe_all_only_local_opened_plus_explicit_urls():
    def list_fn():
        return [
            {"index": 0, "node": "/dev/video0", "opened": True},
            {"index": 1, "node": "/dev/video1", "opened": False},
        ]

    reports = diagnose_module.probe_all(
        list_fn=list_fn, capture_factory=_ok_factory,
        include_urls=["http://192.168.0.10:8080/video"],
    )
    # Local aberta (0) + URL explicita; a fechada (1) nao e sondada.
    assert [str(report.source) for report in reports] == [
        "0", "http://192.168.0.10:8080/video",
    ]
    assert all(report.ok for report in reports)


def test_probe_all_empty_when_nothing_open():
    reports = diagnose_module.probe_all(
        list_fn=lambda: [
            {"index": 0, "node": "/dev/video0", "opened": False},
        ],
        capture_factory=_ok_factory,
    )
    assert reports == []


def _report(source="0", ok=True, kind="local"):
    return diagnose_module.SourceReport(
        source=source, kind=kind, opened=ok,
        backend_used="CAP_V4L2" if ok else None,
        width=1280 if ok else None, height=720 if ok else None,
        fps=30.0 if ok else None, fourcc="MJPG" if ok else None,
        frames_ok=3 if ok else 0, latency_ms=1 if ok else None,
        error_kind=None if ok else "not_found",
        hint="ok" if ok else "nao achou",
    )


def test_cli_test_source_exit_codes_and_json(monkeypatch, capsys):
    monkeypatch.setattr(
        diagnose_module, "probe_source",
        lambda source, **kwargs: _report(source),
    )
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--test-source", "0", "--report", "json"]
    )
    with pytest.raises(SystemExit) as excinfo:
        app_module.main()
    assert excinfo.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["source"] == "0"
    assert payload[0]["frames_ok"] == 3

    monkeypatch.setattr(
        diagnose_module, "probe_source",
        lambda source, **kwargs: _report(source, ok=False),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--test-source", "7"])
    with pytest.raises(SystemExit) as excinfo:
        app_module.main()
    assert excinfo.value.code == 1
    assert "FALHOU" in capsys.readouterr().out


def test_cli_test_all_no_source_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(
        diagnose_module, "probe_all", lambda **kwargs: []
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--test-all"])
    with pytest.raises(SystemExit) as excinfo:
        app_module.main()
    assert excinfo.value.code == 2


def test_cli_list_cameras_verbose_adds_backend_and_props(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        capture_module, "list_cameras",
        lambda: [
            {"index": 1, "node": "/dev/video1", "opened": True},
        ],
    )
    monkeypatch.setattr(
        diagnose_module, "probe_source",
        lambda source, **kwargs: _report(source),
    )
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--list-cameras", "--verbose"]
    )
    with pytest.raises(SystemExit) as excinfo:
        app_module.main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "1 (/dev/video1): ok" in out
    assert "CAP_V4L2" in out
    assert "1280x720" in out


def test_cli_list_cameras_default_format_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(
        capture_module, "list_cameras",
        lambda: [
            {"index": 1, "node": "/dev/video1", "opened": True},
        ],
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--list-cameras"])
    with pytest.raises(SystemExit) as excinfo:
        app_module.main()
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == "1 (/dev/video1): ok"
