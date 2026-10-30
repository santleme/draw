"""Testes do extrator e serializador de traços, sem webcam."""

import json

import cv2
import numpy as np

from draw import config, strokes


def test_empty_mask_has_no_strokes():
    mask = np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
    assert strokes.extract_strokes(mask) == []


def test_straight_line_is_unchanged_by_smoothing():
    points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    smoothed = strokes.smooth_points(points, passes=1)
    assert smoothed == points


def test_mask_extracts_strokes_in_deterministic_order():
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, (80, 80), 30, 255, 3)
    cv2.line(mask, (40, 200), (260, 200), 255, 3)

    found = strokes.extract_strokes(mask)

    assert len(found) >= 2
    centroids = [stroke.centroid for stroke in found]
    assert centroids == sorted(centroids, key=lambda point: (point[1], point[0]))


def test_svg_and_json_roundtrip(tmp_path):
    found = [
        strokes.Stroke(points=[(10.0, 20.0), (30.0, 40.0)]),
        strokes.Stroke(points=[(50.0, 60.0), (70.0, 80.0)], closed=True),
    ]
    svg = strokes.strokes_to_svg(found)

    assert svg.startswith("<svg")
    assert svg.count("<path") == len(found)
    assert svg.strip().endswith("</svg>")
    svg_path, json_path = strokes.export_strokes(found, tmp_path / "out")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["width"] == config.PAPER_W
    assert data["height"] == config.PAPER_H
    assert len(data["strokes"]) == len(found)
    assert svg_path.read_text(encoding="utf-8") == svg
