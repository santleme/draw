"""Testes do escritor DXF (etapa 6 do Senado).

Tudo sintético, sem câmera, sem janela, sem abrir CAD no CI:
o DXF gerado é validado por um parse de volta mínimo, par por
par (código de grupo, valor), escrito aqui mesmo. Não há
dependência nova: `ezdxf` não está instalado e não é usado.

Números travados aqui (etapa 4): A4 sobre o canvas 850x1200 px
usa escala uniforme ancorada na largura, 210/850 mm/px, folha
efetiva 210,00 x 296,47 mm. No DXF, y é invertido em relação
ao Draw: y_dxf = 296,47 - y_mm, porque o CAD mede y para cima
e o Draw mede y para baixo. Sem a inversão, o desenho sairia
espelhado verticalmente — o teste do "L" prova que não saiu.
"""

import pytest

from draw import export
from draw.strokes import Stroke


def parse_dxf(text):
    """Parse mínimo de DXF: devolve (header, entidades).

    header: dict nome da variável -> valor. entidades: lista de
    dicts com layer, closed (bool), vertices [(x, y), ...].
    """
    tokens = [line.strip() for line in text.splitlines() if line.strip()]
    assert len(tokens) % 2 == 0, "DXF deve ter pares código/valor"
    pairs = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]
    header = {}
    entities = []
    current = None
    pending_x = None
    for i, (code, value) in enumerate(pairs):
        if code == "9" and value.startswith("$"):
            ncode, nvalue = pairs[i + 1]
            header[value] = nvalue
        if value == "LWPOLYLINE" and code == "0":
            current = {"layer": None, "closed": False, "vertices": []}
            entities.append(current)
            pending_x = None
        elif current is not None:
            if code == "8":
                current["layer"] = value
            elif code == "70":
                current["closed"] = value == "1"
            elif code == "10":
                pending_x = float(value)
            elif code == "20" and pending_x is not None:
                current["vertices"].append((pending_x, float(value)))
                pending_x = None
    return header, entities


def test_dxf_header_units_and_version():
    dxf = export.strokes_to_dxf([Stroke(points=[(0.0, 0.0), (100.0, 0.0)])])
    header, _entities = parse_dxf(dxf)
    assert header["$ACADVER"] == "AC1015"  # R2000
    assert header["$INSUNITS"] == "4"  # milímetros
    assert header["$MEASUREMENT"] == "1"
    assert dxf.strip().endswith("0\nEOF")


def test_entity_count_layer_and_closed_flag():
    strokes = [
        Stroke(points=[(0.0, 0.0), (100.0, 0.0)]),
        Stroke(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)],
               closed=True),
        Stroke(points=[(0.0, 0.0)]),  # curto demais: fora do DXF
    ]
    _header, entities = parse_dxf(export.strokes_to_dxf(strokes))
    assert len(entities) == 2
    assert all(e["layer"] == "DRAW_STROKES" for e in entities)
    assert entities[0]["closed"] is False
    assert entities[1]["closed"] is True
    # Traço fechado: a flag fecha a polilinha, o ponto inicial
    # NÃO é repetido no fim.
    assert len(entities[1]["vertices"]) == 3
    assert entities[1]["vertices"][0] != entities[1]["vertices"][-1]
    # Nunca CIRCLE nem SPLINE: traço fechado não é círculo e
    # curva não vira spline no MVP.
    dxf = export.strokes_to_dxf(strokes)
    assert "CIRCLE" not in dxf and "SPLINE" not in dxf


def test_corners_and_coordinate_bounds():
    scale = export.sheet_scale(paper="A4")
    rect = Stroke(points=[(0.0, 0.0), (850.0, 0.0), (850.0, 1200.0),
                          (0.0, 1200.0)], closed=True)
    _header, entities = parse_dxf(export.strokes_to_dxf([rect], scale=scale))
    vertices = entities[0]["vertices"]
    # (0,0) px -> canto superior esquerdo do Draw vira, em CAD,
    # o canto superior esquerdo da folha: (0, 296.47).
    assert vertices[0] == pytest.approx((0.0, 296.47), abs=0.01)
    # (850,1200) px -> (210, 0): canto inferior direito.
    assert vertices[2] == pytest.approx((210.0, 0.0), abs=0.01)
    for x, y in vertices:
        assert 0.0 <= x <= 210.0 + 1e-3
        assert 0.0 <= y <= 296.47 + 1e-3


def test_l_shape_is_not_mirrored():
    # "L" no papel (y para baixo): haste vertical em cima, pé
    # embaixo, para a direita. Em CAD (y para cima) o pé tem
    # que continuar embaixo e para a direita — sem a inversão
    # de y, o pé iria para cima (espelho vertical).
    ell = Stroke(points=[(0.0, 0.0), (0.0, 1000.0), (500.0, 1000.0)])
    _header, entities = parse_dxf(export.strokes_to_dxf([ell]))
    (x0, y0), (x1, y1), (x2, y2) = entities[0]["vertices"]
    assert y0 > y1  # topo da haste acima do pé, em coordenadas CAD
    assert y1 == pytest.approx(y2)  # pé horizontal
    assert x2 > x1  # pé para a direita, não para a esquerda
    assert x0 == pytest.approx(x1)  # haste vertical


def test_dxf_file_roundtrip_and_empty():
    strokes = [Stroke(points=[(0.0, 0.0), (850.0, 0.0)])]
    dxf = export.strokes_to_dxf(strokes)
    _header, entities = parse_dxf(dxf)
    assert len(entities) == 1
    assert entities[0]["vertices"][1][0] == pytest.approx(210.0)
    _header, empty = parse_dxf(export.strokes_to_dxf([]))
    assert empty == []


def test_export_functions_write_dxf(tmp_path):
    strokes = [Stroke(points=[(0.0, 0.0), (100.0, 0.0)])]
    path = export.export_strokes_dxf(strokes, tmp_path / "out")
    assert path.suffix == ".dxf" and path.exists()
    _header, entities = parse_dxf(path.read_text(encoding="utf-8"))
    assert len(entities) == 1


def test_letter_paper_bounds():
    scale = export.sheet_scale(paper="Letter")
    rect = Stroke(points=[(0.0, 0.0), (850.0, 0.0), (850.0, 1200.0),
                          (0.0, 1200.0)], closed=True)
    _header, entities = parse_dxf(export.strokes_to_dxf([rect], scale=scale))
    for x, y in entities[0]["vertices"]:
        assert 0.0 <= x <= 197.91 + 0.01
        assert 0.0 <= y <= 279.4 + 0.01
