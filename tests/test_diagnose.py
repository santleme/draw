"""Testes do relatorio de diagnostico por fonte, so com mocks.

Versao desacoplada para o upstream: nao depende de draw/capture.py
nem de draw/app.py. Nenhum teste abre dispositivo real, depende de
/dev/video* ou faz acao no sistema. A fabrica de captura e injetada
em probe_source() e probe_all().

Os testes de CLI (--test-source/--test-all/--list-cameras) do fork
nao estao aqui: eles mexem em draw/app.py e pertencem a ligacao
com a CLI, trabalho separado deste PR.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


@pytest.mark.parametrize("kind,hint_fragment", [
    ("not_found", "/dev/video7"),
    ("permission", "permissao"),
    ("busy", "Outro programa"),
])
def test_probe_each_open_error_kind_with_hint(monkeypatch, kind, hint_fragment):
    monkeypatch.setattr(
        diagnose_module, "diagnose_local",
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
