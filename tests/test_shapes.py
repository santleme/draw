"""Testes da etapa 7 do Senado: reconhecimento de formas, sem hardware.

Primeira fatia: só linha, círculo, retângulo e unknown. Sem seta,
sem letra, sem diagrama, sem ML. Os testes usam geometria controlada;
eles não medem precisão em desenhos reais.
"""

import json
import math
from draw import config, shapes, strokes
from draw.shapes import classify_group, classify_strokes, group_strokes


def _line_stroke():
    return strokes.Stroke(points=[(100.0, 100.0), (200.0, 200.0), (300.0, 300.0)])


def _circle_stroke():
    pts = [
        (425.0 + 110.0 * math.cos(2.0 * math.pi * k / 64.0),
         500.0 + 110.0 * math.sin(2.0 * math.pi * k / 64.0))
        for k in range(64)
    ]
    return strokes.Stroke(points=pts, closed=True)


def _rectangle_stroke():
    pts = [
        (100.0, 100.0), (250.0, 100.0), (400.0, 100.0),
        (400.0, 200.0), (400.0, 300.0), (250.0, 300.0),
        (100.0, 300.0), (100.0, 200.0),
    ]
    return strokes.Stroke(points=pts, closed=True)


def _rectangle_segments():
    return [
        strokes.Stroke(points=[(100.0, 100.0), (250.0, 100.0), (400.0, 100.0)]),
        strokes.Stroke(points=[(400.0, 100.0), (400.0, 200.0), (400.0, 300.0)]),
        strokes.Stroke(points=[(400.0, 300.0), (250.0, 300.0), (100.0, 300.0)]),
        strokes.Stroke(points=[(100.0, 300.0), (100.0, 200.0), (100.0, 100.0)]),
    ]


def _x_segments():
    return [
        strokes.Stroke(points=[(400.0, 400.0), (500.0, 500.0)]),
        strokes.Stroke(points=[(500.0, 500.0), (600.0, 400.0)]),
        strokes.Stroke(points=[(400.0, 600.0), (500.0, 500.0)]),
        strokes.Stroke(points=[(500.0, 500.0), (600.0, 600.0)]),
    ]


def _scribble_stroke():
    pts = [
        (100.0 + 10.0 * k, 500.0 + 40.0 * math.sin(2.0 * math.pi * k / 8.0))
        for k in range(41)
    ]
    return strokes.Stroke(points=pts)


def _ellipse_stroke():
    pts = [
        (425.0 + 150.0 * math.cos(2.0 * math.pi * k / 64.0),
         500.0 + 80.0 * math.sin(2.0 * math.pi * k / 64.0))
        for k in range(64)
    ]
    return strokes.Stroke(points=pts, closed=True)


def test_line_is_classified():
    shape = classify_group([_line_stroke()], source_ids=[0])
    assert shape.kind == "line"
    assert shape.confidence >= 0.5
    assert shape.params["p1"] == (100.0, 100.0)
    assert shape.params["p2"] == (300.0, 300.0)
    assert shape.source_stroke_ids == [0]


def test_circle_is_classified_with_fit_params():
    shape = classify_group([_circle_stroke()], source_ids=[0])
    assert shape.kind == "circle"
    assert shape.confidence >= 0.5
    cx, cy = shape.params["center"]
    assert abs(cx - 425.0) < 5.0
    assert abs(cy - 500.0) < 5.0
    assert abs(shape.params["radius"] - 110.0) < 5.0


def test_rectangle_single_closed_stroke():
    shape = classify_group([_rectangle_stroke()], source_ids=[0])
    assert shape.kind == "rectangle"
    assert shape.confidence >= 0.5
    assert len(shape.params["corners"]) == 4
    assert all(abs(a - 90.0) <= 20.0 for a in shape.params["angles_deg"])


def test_rectangle_four_segments_group_into_one():
    segments = _rectangle_segments()
    groups = group_strokes(segments)
    assert len(groups) == 1
    assert len(groups[0]) == 4
    shape = classify_group(groups[0], source_ids=[0, 1, 2, 3])
    assert shape.kind == "rectangle"
    assert shape.source_stroke_ids == [0, 1, 2, 3]


def test_grouping_keeps_distant_shapes_separate():
    groups = group_strokes([_line_stroke(), _circle_stroke()])
    assert len(groups) == 2


def test_x_is_not_forced_into_a_shape():
    segments = _x_segments()
    groups = group_strokes(segments)
    assert len(groups) == 1
    shape = classify_group(groups[0], source_ids=[0, 1, 2, 3])
    assert shape.kind == "unknown"
    assert shape.confidence == 0.0


def test_scribble_is_unknown():
    shape = classify_group([_scribble_stroke()], source_ids=[0])
    assert shape.kind == "unknown"


def test_ellipse_is_unknown_not_forced_to_circle():
    # Elipse alongada é escopo futuro; nesta fatia não vira círculo.
    shape = classify_group([_ellipse_stroke()], source_ids=[0])
    assert shape.kind == "unknown"


def test_dense_short_cluster_is_unknown():
    cluster = [
        strokes.Stroke(points=[(200.0, 200.0), (230.0, 200.0)]),
        strokes.Stroke(points=[(200.0, 220.0), (230.0, 220.0)]),
        strokes.Stroke(points=[(215.0, 200.0), (215.0, 230.0)]),
    ]
    shape = classify_group(group_strokes(cluster)[0], source_ids=[0, 1, 2])
    assert shape.kind == "unknown"


def test_confusion_matrix_counts_unknown_separately():
    matrix = shapes.confusion_matrix(
        [("line", "line"), ("x", "unknown"), ("line", "unknown")]
    )
    assert matrix == {"line": {"line": 1, "unknown": 1}, "x": {"unknown": 1}}


def test_shapes_json_is_separate_from_strokes_v1(tmp_path):
    found_shapes = classify_strokes([_line_stroke(), _circle_stroke()])
    path = shapes.export_shapes(found_shapes, tmp_path / "shapes.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["coordinate_space"] == "warped_sheet_850x1200"
    assert data["width"] == config.PAPER_W
    assert data["height"] == config.PAPER_H
    assert [s["kind"] for s in data["shapes"]] == ["line", "circle"]
    # Coordenadas normalizadas em 0..1, sem tocar no strokes.json v1.
    for shape in data["shapes"]:
        for x, y in shape["points_normalized"]:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
    # O formato antigo continua intacto: pontos em px, sem escala.
    svg_path, json_path = strokes.export_strokes(
        [_line_stroke()], tmp_path / "out"
    )
    old = json.loads(json_path.read_text(encoding="utf-8"))
    assert old["version"] == 1
    assert "shapes" not in old
    assert svg_path.exists()
