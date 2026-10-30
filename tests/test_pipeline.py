"""Testes básicos do pipeline v0.1, sem webcam.

Estes testes usam só imagens sintéticas. O teste de ponta a ponta
com cena inclinada continua em tests/selftest.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draw import config, ink, paper  # noqa: E402
from template.generate_template import build_template  # noqa: E402


def test_blank_frame_has_no_markers():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert paper.detect_markers(frame) == {}


def test_warp_returns_none_without_all_markers():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    warped, markers = paper.warp_paper(frame)
    assert warped is None
    assert markers == {}


def test_template_has_four_markers():
    sheet = build_template()
    markers = paper.detect_markers(sheet)
    assert set(markers) == set(config.MARKER_IDS)


def test_white_sheet_has_no_ink():
    sheet = np.full((config.PAPER_H, config.PAPER_W, 3), 255, dtype=np.uint8)
    mask = ink.extract_ink(sheet)
    assert int((mask > 0).sum()) == 0


def test_render_canvas_shape_and_colors():
    mask = np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    canvas = ink.render_canvas(mask)
    assert canvas.shape == (config.PAPER_H, config.PAPER_W, 3)
    assert (canvas[15, 15] == np.array([0, 0, 0])).all()
    assert (canvas[0, 0] == np.array([255, 255, 255])).all()
