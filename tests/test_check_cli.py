"""Testes do diagnostico de fonte de video (draw.app.run_check e parse_source)."""

import sys
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app  # noqa: E402


class _FakeCapture:
    def __init__(self, opened=True, frame=None):
        self._opened = opened
        self._frame = frame
        self.released = False
        self.props = {}

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if self._frame is None:
            return False, None
        return True, self._frame

    def release(self):
        self.released = True


def test_parse_source():
    assert app.parse_source(None) == 0
    assert app.parse_source(2) == 2
    assert app.parse_source("0") == 0
    assert app.parse_source(" 3 ") == 3
    assert (
        app.parse_source("http://192.168.0.10:8080/video")
        == "http://192.168.0.10:8080/video"
    )
    assert app.parse_source("sessao.drawcap") == "sessao.drawcap"


def test_run_check_honors_explicit_camera_index():
    seen_sources = []
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def fake_vc(src):
        seen_sources.append(src)
        return _FakeCapture(opened=True, frame=frame)

    with (
        mock.patch.object(
            app, "_video_nodes", return_value=["/dev/video0", "/dev/video2"]
        ),
        mock.patch("os.access", return_value=True),
        mock.patch("cv2.VideoCapture", side_effect=fake_vc),
    ):
        rc = app.run_check(2)
    assert rc == 0
    assert seen_sources == [2], f"esperado abrir indice 2, abriu {seen_sources}"


def test_run_check_fails_when_requested_index_missing():
    with mock.patch.object(app, "_video_nodes", return_value=["/dev/video0"]):
        rc = app.run_check(3)
    assert rc == 1


def test_run_check_supports_stream_or_file_source():
    seen_sources = []
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def fake_vc(src):
        seen_sources.append(src)
        return _FakeCapture(opened=True, frame=frame)

    with mock.patch("cv2.VideoCapture", side_effect=fake_vc):
        rc = app.run_check("http://10.0.0.2:8080/video")
    assert rc == 0
    assert seen_sources == ["http://10.0.0.2:8080/video"]


def main():
    test_parse_source()
    test_run_check_honors_explicit_camera_index()
    test_run_check_fails_when_requested_index_missing()
    test_run_check_supports_stream_or_file_source()
    print("TESTES DE CHECK/CLI OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
