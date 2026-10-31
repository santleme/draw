"""Testes da camada de captura, so com mocks, sem camera fisica.

A fabrica de captura e injetada em cada teste: nenhum teste abre
um dispositivo real, le frame de hardware ou depende de /dev/video*.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import app as app_module  # noqa: E402
from draw import capture as capture_module  # noqa: E402


class FakeCapture:
    """VideoCapture falso: aberto ou nao, com frames programaveis."""

    def __init__(self, source, backend=None, opened=True, frames=()):
        self.source = source
        self.backend = backend
        self._opened = opened
        self._frames = list(frames)
        self.settings = {}
        self.released = False
        self.fps = 30.0

    def isOpened(self):
        return self._opened and not self.released

    def set(self, prop, value):
        self.settings[prop] = value
        return True

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self):
        self.released = True


def test_parse_source_index_path_url_and_file():
    assert capture_module.parse_source(None) == 0
    assert capture_module.parse_source(2) == 2
    assert capture_module.parse_source("1") == 1
    assert capture_module.parse_source(" 2 ") == 2
    assert capture_module.parse_source("/dev/video2") == 2
    assert capture_module.parse_source("http://192.168.0.10:8080/video") == (
        "http://192.168.0.10:8080/video"
    )
    assert capture_module.parse_source("rtsp://camera:554/stream") == (
        "rtsp://camera:554/stream"
    )
    assert capture_module.parse_source("video.mp4") == "video.mp4"
    assert app_module.parse_source("/dev/video1") == 1


def test_backends_order_by_platform():
    linux = capture_module.backends_for_platform("linux")
    assert linux[0] == cv2.CAP_V4L2
    assert cv2.CAP_ANY in linux
    windows = capture_module.backends_for_platform("win32")
    assert windows[0] == cv2.CAP_MSMF
    assert capture_module.backend_from_name("v4l2") == cv2.CAP_V4L2
    assert capture_module.backend_from_name(None) is None
    with pytest.raises(ValueError):
        capture_module.backend_from_name("backend-que-nao-existe")


def test_list_cameras_only_index_1_opens_and_releases():
    seen = []

    def factory(index, backend=None):
        capture = FakeCapture(index, backend, opened=(index == 1))
        seen.append(capture)
        return capture

    results = capture_module.list_cameras(
        max_index=2, capture_factory=factory, nodes=[]
    )
    assert [item["index"] for item in results] == [0, 1, 2]
    assert [item["opened"] for item in results] == [False, True, False]
    assert all(capture.released for capture in seen)
    assert "1" in capture_module.format_camera_list(results)


def test_list_cameras_includes_video_node_above_max_index():
    results = capture_module.list_cameras(
        max_index=0,
        capture_factory=lambda index, backend=None: FakeCapture(index, backend, opened=False),
        nodes=["/dev/video4"],
    )
    assert [item["index"] for item in results] == [0, 4]


def test_open_capture_tries_backends_and_applies_local_settings():
    attempts = []
    params_by_backend = []

    def factory(source, backend=None, params=None):
        attempts.append(backend)
        params_by_backend.append(params)
        return FakeCapture(source, backend, opened=(backend == cv2.CAP_ANY))

    capture = capture_module.open_capture(
        0, capture_factory=factory, backends=[cv2.CAP_V4L2, cv2.CAP_ANY]
    )
    assert capture.isOpened()
    assert attempts == [cv2.CAP_V4L2, cv2.CAP_V4L2, cv2.CAP_ANY]
    assert capture.settings[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert capture.settings[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    expected = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000]
    assert params_by_backend == [expected, None, expected]


def test_open_capture_custom_size_and_specific_backend():
    def factory(source, backend=None, params=None):
        return FakeCapture(source, backend, opened=True)

    capture = capture_module.open_capture(
        1, width=640, height=480, fps=30, backend=cv2.CAP_V4L2,
        capture_factory=factory,
    )
    assert capture.settings[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert capture.settings[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert capture.settings[cv2.CAP_PROP_FPS] == 30
    assert capture.backend == cv2.CAP_V4L2


def test_open_capture_url_does_not_apply_local_settings():
    attempts = []

    def factory(source, backend=None, params=None):
        attempts.append((source, backend, params))
        return FakeCapture(source, backend, opened=True)

    capture = capture_module.open_capture(
        "http://192.168.0.10:8080/video", read_timeout_ms=5000,
        capture_factory=factory,
    )
    assert capture.isOpened()
    assert cv2.CAP_PROP_FRAME_WIDTH not in capture.settings
    assert attempts == [
        (
            "http://192.168.0.10:8080/video",
            cv2.CAP_ANY,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ],
        )
    ]


def test_open_capture_local_falls_back_when_timeout_params_are_rejected():
    calls = []

    def factory(source, backend=None, params=None):
        calls.append((source, backend, params))
        if params is not None:
            raise TypeError("local backend rejects open params")
        return FakeCapture(source, backend, opened=True)

    capture = capture_module.open_capture(0, capture_factory=factory)

    assert capture.isOpened()
    assert calls == [
        (0, cv2.CAP_V4L2, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000]),
        (0, cv2.CAP_V4L2, None),
    ]


def test_open_capture_network_does_not_fallback_without_timeout_params():
    source = "http://camera.local/video"
    calls = []

    def factory(value, backend=None, params=None):
        calls.append((value, backend, params))
        raise TypeError("backend rejects open params")

    with pytest.raises(capture_module.CaptureError) as excinfo:
        capture_module.open_capture(source, capture_factory=factory)

    assert calls == [
        (source, cv2.CAP_ANY, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000])
    ]
    assert excinfo.value.kind == "network"
    assert "nao foi tentada novamente sem eles" in str(excinfo.value)


def test_open_capture_network_fails_closed_when_timeout_binding_is_missing(monkeypatch):
    monkeypatch.delattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC")
    calls = []

    with pytest.raises(capture_module.CaptureError) as excinfo:
        capture_module.open_capture(
            "http://camera.local/video",
            capture_factory=lambda *args: calls.append(args),
        )

    assert excinfo.value.kind == "timeout_unsupported"
    assert calls == []


def test_open_capture_local_fails_closed_when_read_timeout_binding_is_missing(monkeypatch):
    monkeypatch.delattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC")
    calls = []

    with pytest.raises(capture_module.CaptureError) as excinfo:
        capture_module.open_capture(
            0,
            read_timeout_ms=5000,
            capture_factory=lambda *args: calls.append(args),
        )

    assert excinfo.value.kind == "timeout_unsupported"
    assert calls == []


def test_open_capture_local_does_not_retry_without_requested_read_timeout():
    calls = []

    def factory(source, backend=None, params=None):
        calls.append((source, backend, params))
        raise TypeError("backend rejects timeout params")

    with pytest.raises(capture_module.CaptureError) as excinfo:
        capture_module.open_capture(
            0, read_timeout_ms=5000, capture_factory=factory
        )

    assert excinfo.value.kind == "timeout_unsupported"
    assert calls
    assert all(call[2] is not None for call in calls)


def test_open_capture_local_missing_node_reports_not_found(monkeypatch):
    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    monkeypatch.setattr(capture_module, "is_wsl", lambda: False)
    with pytest.raises(capture_module.CaptureError) as excinfo:
        capture_module.open_capture(
            7,
            capture_factory=lambda source, backend=None: FakeCapture(
                source, backend, opened=False
            ),
        )
    assert excinfo.value.kind == "not_found"
    assert "/dev/video7" in str(excinfo.value)


def test_open_capture_network_error_kind():
    with pytest.raises(capture_module.CaptureError) as excinfo:
        capture_module.open_capture(
            "http://192.168.0.10:8080/video",
            capture_factory=lambda source, backend=None: FakeCapture(
                source, backend, opened=False
            ),
        )
    assert excinfo.value.kind == "network"


def test_diagnose_local_permission_and_busy(monkeypatch):
    monkeypatch.setattr(capture_module.sys, "platform", "linux")
    kind, hint = capture_module.diagnose_local(
        0,
        exists=lambda path: True,
        access=lambda path, mode: False,
        wsl=False,
    )
    assert kind == "permission"
    assert "video" in hint
    kind, hint = capture_module.diagnose_local(
        0,
        exists=lambda path: True,
        access=lambda path, mode: True,
        wsl=False,
    )
    assert kind == "busy"
    assert "Outro programa" in hint


def test_reconnect_succeeds_on_second_attempt_and_fails_cleanly():
    calls = []

    def open_ok(source, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            raise capture_module.CaptureError(source, "busy", "ocupada")
        return FakeCapture(source, opened=True)

    reconnect = capture_module.ReconnectingCapture(
        0, open_func=open_ok, attempts=2, delay_s=0, sleep=lambda seconds: None
    )
    assert reconnect.try_reconnect() is True
    assert reconnect.reconnects == 1
    assert reconnect.isOpened()

    def open_never(source, **kwargs):
        raise capture_module.CaptureError(source, "not_found", "sumiu")

    never = capture_module.ReconnectingCapture(
        0, open_func=open_never, attempts=2, delay_s=0, sleep=lambda seconds: None
    )
    assert never.try_reconnect() is False
    assert never.last_error.kind == "not_found"

    off = capture_module.ReconnectingCapture(
        0, open_func=open_never, attempts=0, delay_s=0, sleep=lambda seconds: None
    )
    assert off.try_reconnect() is False


def test_latest_frame_capture_reconnects_after_read_exception(monkeypatch):
    class ReconnectAfterReadError:
        def __init__(self):
            self.reconnected = False
            self.reconnect_calls = 0
            self.released = False

        def read(self):
            if not self.reconnected:
                raise RuntimeError("camera disconnected")
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def try_reconnect(self):
            self.reconnect_calls += 1
            self.reconnected = True
            return True

        def release(self):
            self.released = True

    monkeypatch.setattr(app_module, "_CAPTURE_MAX_FAILURES", 1)
    capture = ReconnectAfterReadError()
    source = app_module._LatestFrameCapture(capture).start()
    try:
        item = source.wait_next(0, timeout=1.0)
        assert item is not None
        assert item[0].shape == (2, 2, 3)
        assert capture.reconnect_calls == 1
    finally:
        source.stop()
    assert capture.released is True


def test_latest_frame_capture_preserves_read_exception_when_reconnect_fails(
    monkeypatch,
):
    class LostCapture:
        def read(self):
            raise RuntimeError("camera disconnected")

        def try_reconnect(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(app_module, "_CAPTURE_MAX_FAILURES", 1)
    source = app_module._LatestFrameCapture(LostCapture()).start()
    try:
        assert source.wait_next(0, timeout=1.0) is None
        assert "camera disconnected" in str(source.last_error)
    finally:
        source.stop()


def test_app_list_cameras_cli_exits_with_status(monkeypatch, capsys):
    monkeypatch.setattr(
        capture_module,
        "list_cameras",
        lambda: [
            {"index": 0, "node": "/dev/video0", "opened": False},
            {"index": 1, "node": "/dev/video1", "opened": True},
        ],
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--list-cameras"])
    with pytest.raises(SystemExit) as excinfo:
        app_module.main()
    assert excinfo.value.code == 0
    assert "1 (/dev/video1): ok" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (1, 1),
        ("http://camera.local/video", "http://camera.local/video"),
        ("recording.mp4", "recording.mp4"),
    ],
)
def test_run_check_opens_exact_source_without_requiring_video_nodes(
    monkeypatch, capsys, source, expected
):
    capture = FakeCapture(expected, frames=[np.zeros((480, 640, 3), dtype=np.uint8)])
    opened = []

    def open_source(selected, **kwargs):
        opened.append((selected, kwargs))
        return capture

    monkeypatch.setattr(app_module, "_video_nodes", lambda: pytest.fail("must not enumerate"))
    monkeypatch.setattr(app_module, "_open_capture", open_source)

    assert app_module.run_check(source, open_timeout_ms=1700) == 0

    assert opened == [
        (expected, {"open_timeout_ms": 1700, "read_timeout_ms": 5000})
    ]
    assert capture.released
    output = capsys.readouterr().out
    assert f"Testando fonte: {expected!r}" in output
    assert "640x480, 30 FPS" in output


def test_run_check_cli_preserves_url_source(monkeypatch):
    source = "http://camera.local/video"
    checked = []
    monkeypatch.setattr(sys, "argv", ["main.py", "--check", "--source", source])
    monkeypatch.setattr(
        app_module, "run_check",
        lambda value, open_timeout_ms: checked.append((value, open_timeout_ms)) or 0,
    )

    with pytest.raises(SystemExit) as excinfo:
        app_module.main()

    assert excinfo.value.code == 0
    assert checked == [(source, 3000)]


def test_run_check_reports_open_failure(monkeypatch, capsys):
    source = "http://camera.local/video"

    def fail_open(selected, **kwargs):
        raise capture_module.CaptureError(selected, "network", "stream indisponivel")

    monkeypatch.setattr(app_module, "_open_capture", fail_open)

    assert app_module.run_check(source) == 1
    output = capsys.readouterr().out
    assert "FAIL:" in output
    assert source in output


def test_run_check_times_out_and_releases_capture(monkeypatch, capsys):
    capture = FakeCapture("camera.mp4")
    monkeypatch.setattr(app_module, "_open_capture", lambda source, **kwargs: capture)

    assert app_module.run_check(
        "camera.mp4", first_frame_timeout_s=0.02
    ) == 1

    assert capture.released
    assert "nao entregou um frame antes do prazo de 0.02 segundos" in capsys.readouterr().out


def test_run_check_releases_capture_when_read_raises(monkeypatch, capsys):
    class BrokenCapture(FakeCapture):
        def read(self):
            raise RuntimeError("read failed")

    capture = BrokenCapture("camera.mp4")
    monkeypatch.setattr(app_module, "_open_capture", lambda source, **kwargs: capture)

    assert app_module.run_check(
        "camera.mp4", first_frame_timeout_s=0.2
    ) == 1

    assert capture.released
    assert "read failed" in capsys.readouterr().out


def test_run_check_treats_fps_query_exception_as_unavailable(monkeypatch, capsys):
    class BrokenFpsCapture(FakeCapture):
        def get(self, prop):
            raise RuntimeError("fps query unsupported")

    capture = BrokenFpsCapture(
        "camera.mp4", frames=[np.zeros((480, 640, 3), dtype=np.uint8)]
    )
    monkeypatch.setattr(app_module, "_open_capture", lambda source, **kwargs: capture)

    assert app_module.run_check("camera.mp4", first_frame_timeout_s=0.02) == 0
    assert capture.released
    assert "FPS indisponivel" in capsys.readouterr().out
