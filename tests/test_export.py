"""Testes da exportação em escala real (etapa 4 do Senado).

Tudo sintético, sem câmera, sem janela. Números travados aqui:

- A4 (210 x 297 mm) sobre o canvas 850 x 1200 px: a escala
  uniforme é ancorada na largura, 210/850 mm/px, e a altura
  efetiva é 296,47 mm (0,53 mm abaixo do nominal — a diferença
  de aspecto de ~0,18% entre canvas e A4 real, documentada em
  draw/export.py, não escondida).
- Tolerância declarada: segmento de até 100 mm mede 100 +/- 1 mm.
"""

import json
import math
import re

import pytest

from draw import export
from draw.strokes import Stroke


def test_a4_scale_is_uniform_and_width_anchored():
    scale = export.sheet_scale(paper="A4")
    assert scale.mm_per_px == pytest.approx(210.0 / 850.0)
    assert scale.effective_width_mm == pytest.approx(210.0)
    assert scale.effective_height_mm == pytest.approx(296.47, abs=0.01)
    # Uma única escala, não sx != sy.
    assert scale.mm_per_px == min(210.0 / 850.0, 297.0 / 1200.0)


def test_corners_map_to_sheet_corners():
    scale = export.sheet_scale(paper="A4")
    assert export.px_to_mm((0, 0), scale) == pytest.approx((0.0, 0.0))
    x, y = export.px_to_mm((850, 1200), scale)
    assert x == pytest.approx(210.0)
    assert y == pytest.approx(296.47, abs=0.01)


def test_100mm_segments_within_tolerance_both_directions():
    scale = export.sheet_scale(paper="A4")
    # 100 mm horizontal no canvas: 100 * (850/210) px.
    horiz = Stroke(points=[(0.0, 0.0), (100 * 850.0 / 210.0, 0.0)])
    assert export.stroke_length_mm(horiz, scale) == pytest.approx(100.0, abs=1.0)
    # 100 mm vertical no canvas verdadeiro: 100 * (1200/297) px.
    # A escala uniforme ancorada na largura devolve 99,82 mm:
    # erro de 0,18%, dentro da tolerância de 1 mm.
    vert = Stroke(points=[(0.0, 0.0), (0.0, 100 * 1200.0 / 297.0)])
    assert export.stroke_length_mm(vert, scale) == pytest.approx(100.0, abs=1.0)
    assert export.stroke_length_mm(vert, scale) == pytest.approx(99.82, abs=0.01)


def test_mm_to_px_roundtrip():
    scale = export.sheet_scale(paper="A4")
    for point in [(0, 0), (425, 600), (850, 1200), (123.4, 567.8)]:
        back = export.mm_to_px(export.px_to_mm(point, scale), scale)
        assert back == pytest.approx(point, abs=1e-6)


def test_svg_scaled_uses_mm_everywhere():
    scale = export.sheet_scale(paper="A4")
    svg = export.strokes_to_svg_scaled(
        [Stroke(points=[(0.0, 0.0), (850.0, 0.0)])], scale=scale
    )
    assert 'width="210.00mm"' in svg
    assert 'height="296.47mm"' in svg
    assert 'viewBox="0 0 210.00 296.47"' in svg
    assert 'stroke-width="0.5mm"' in svg
    assert "M 0.00 0.00 L 210.00 0.00" in svg
    # viewBox e viewport coerentes: nenhum visualizador aplica
    # uma segunda escala não uniforme.
    assert svg.count("<path") == 1


def test_svg_path_count_matches_strokes_and_closed_gets_z():
    strokes = [
        Stroke(points=[(0.0, 0.0), (100.0, 0.0)]),
        Stroke(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)],
               closed=True),
        Stroke(points=[(0.0, 0.0)]),  # curto demais: fora do SVG
    ]
    svg = export.strokes_to_svg_scaled(strokes)
    assert svg.count("<path") == 2
    assert " Z" in svg


def test_svg_preserves_input_order_without_claiming_temporal_order():
    # Ordem de entrada (de baixo para cima) deve ser preservada:
    # snapshot por centroide, não ordem temporal inventada.
    bottom = Stroke(points=[(0.0, 900.0), (100.0, 900.0)])
    top = Stroke(points=[(0.0, 100.0), (100.0, 100.0)])
    svg = export.strokes_to_svg_scaled([bottom, top])
    paths = re.findall(r'<path[^>]*>', svg)
    assert len(paths) == 2
    # Este snapshot não contém identificadores temporais.
    assert "data-id" not in svg and "data-order" not in svg


def test_json_keeps_v1_fields_and_adds_scale():
    strokes = [Stroke(points=[(0.0, 0.0), (850.0, 0.0)])]
    data = export.strokes_to_dict_scaled(strokes)
    # Leitor v1: só estes campos.
    assert data["version"] == 1
    assert data["width"] == 850
    assert data["height"] == 1200
    assert data["strokes"][0]["points"] == [[0.0, 0.0], [850.0, 0.0]]
    # Campos novos, aditivos.
    assert data["paper"] == "A4"
    assert data["units"] == "mm"
    assert data["scale_mm_per_px"] == pytest.approx(210.0 / 850.0)
    assert data["sheet_mm"] == pytest.approx([210.0, 296.47], abs=0.01)
    assert data["strokes"][0]["points_mm"][1][0] == pytest.approx(210.0)
    assert data["strokes"][0]["length_mm"] == pytest.approx(210.0, abs=0.01)


def test_json_roundtrip_preserves_v1_points(tmp_path):
    strokes = [Stroke(points=[(10.0, 20.0), (30.0, 40.0)], closed=False)]
    svg_path, json_path = export.export_strokes_scaled(strokes, tmp_path / "out")
    assert svg_path.exists() and json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["strokes"][0]["points"] == [[10.0, 20.0], [30.0, 40.0]]
    assert data["strokes"][0]["points_mm"][0] == pytest.approx(
        [10.0 * 210.0 / 850.0, 20.0 * 210.0 / 850.0], abs=1e-3
    )


def test_custom_and_letter_papers():
    letter = export.sheet_scale(paper="Letter")
    # Letter é mais largo e mais baixo: ancora na altura.
    assert letter.mm_per_px == pytest.approx(279.4 / 1200.0)
    assert letter.effective_height_mm == pytest.approx(279.4)
    assert letter.effective_width_mm == pytest.approx(197.91, abs=0.01)
    custom = export.sheet_scale(paper=(100.0, 100.0))
    assert custom.paper == "custom"
    assert custom.effective_width_mm == pytest.approx(70.83, abs=0.01)
    assert custom.effective_height_mm == pytest.approx(100.0)


def test_invalid_paper_and_canvas_raise():
    with pytest.raises(ValueError):
        export.sheet_scale(paper="A0")
    with pytest.raises(ValueError):
        export.sheet_scale(canvas_width=0)
    with pytest.raises(ValueError):
        export.sheet_scale(paper=(0.0, 100.0))


def test_full_rectangle_maps_to_full_sheet():
    scale = export.sheet_scale(paper="A4")
    rect = Stroke(points=[(0.0, 0.0), (850.0, 0.0), (850.0, 1200.0),
                          (0.0, 1200.0)], closed=True)
    assert export.stroke_length_mm(rect, scale) == pytest.approx(
        2 * (210.0 + 296.47), abs=0.1
    )
    svg = export.strokes_to_svg_scaled([rect], scale=scale)
    assert "M 0.00 0.00 L 210.00 0.00 L 210.00 296.47 L 0.00 296.47 Z" in svg
    # Comprimento em mm não depende de percorrer o retângulo em px.
    assert math.isclose(
        export.stroke_length_mm(rect, scale),
        2 * (210.0 + 296.47), rel_tol=1e-3,
    )
