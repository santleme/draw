"""Exportação em escala real (etapa 4 do Senado).

O pipeline trabalha no espaço da folha retificada em pixels
(``config.PAPER_W`` x ``config.PAPER_H``, por padrão 850 x 1200 px).
Este módulo é a única camada que converte esses pixels para
milímetros. Ele não toca na detecção, na tinta nem na extração de
traços, e não muda o comportamento do snapshot atual (tecla ``v``),
que continua usando ``strokes.export_strokes`` em pixels.

Decisão de escala (documentada porque não há resposta exata)
------------------------------------------------------------
Um A4 real mede 210 x 297 mm, mas o canvas mede 850 x 1200 px.
As duas razões não são iguais::

    850 px / 210 mm = 4,0476 px/mm   (horizontal)
    1200 px / 297 mm = 4,0404 px/mm  (vertical)

Diferença de ~0,18%. Nenhuma escala uniforme torna as duas
dimensões exatas ao mesmo tempo, e uma escala não uniforme
distorceria o desenho no CAD. Por isso a escala aqui é uniforme
e ancorada na dimensão limitante (``min`` das duas razões): para
o A4, a largura ancora, então a folha efetiva mede 210,00 x
296,47 mm — 0,53 mm a menos que os 297 mm nominais na altura.

Consequência prática, dentro da tolerância declarada: qualquer
segmento de até 100 mm mede 100 +/- 1 mm após a exportação
(o pior caso real é ~0,18 mm, na direção vertical). A altura
total da folha é o único valor que fica 0,53 mm abaixo do
nominal, e isso está registrado em ``effective_mm``, não
escondido. Se a folha fotografada não for o papel declarado,
a escala está errada por definição — por isso o formato
carrega o papel usado (``paper``) e a escala (``scale_mm_per_px``).

Sobre o SVG: as coordenadas dos caminhos já saem em milímetros
e o ``viewBox`` usa as mesmas unidades, então
``width``/``height`` em mm e ``viewBox`` são coerentes e nenhum
visualizador aplica uma segunda escala não uniforme. Emitir
``height="297mm"`` com ``viewBox="0 0 850 1200"`` faria exatamente
isso (esticaria o desenho 0,18% na vertical) e por isso não é
feito aqui, mesmo sendo o esboço inicial do Senado.

Ordem dos traços: o SVG preserva a ordem da lista recebida. No
snapshot atual essa ordem é por centroide (ordem de leitura),
não temporal — o SVG não finge ordem temporal. Quando a etapa
de rastreamento temporal existir, os traços poderão carregar
``id``/``order`` (campos opcionais de ``Stroke``) e o SVG os
emite como ``data-id``/``data-order``; sem eles, os atributos
são omitidos.

DXF (etapa 6 do Senado): escritor mínimo, ASCII, versão R2000
(``AC1015``), escrito à mão, sem dependência nova. Usa a mesma
escala uniforme daqui (A4 efetivo 210,00 x 296,47 mm), um
``LWPOLYLINE`` por traço na layer ``DRAW_STROKES``, ``$INSUNITS
= 4`` (mm) e ``$MEASUREMENT = 1``. Duas diferenças em relação ao
SVG, ambas obrigatórias para CAD e ambas testadas:

- Inversão de y: o Draw usa y para baixo (origem no canto
  superior esquerdo da folha); o CAD usa y para cima. Sem
  inverter, o DXF sai espelhado verticalmente. Aqui
  ``y_dxf = altura_efetiva_mm - y_mm``.
- Sem entidade CIRCLE: um traço fechado não é um círculo. Sem
  ajuste de círculo, traço fechado vira ``LWPOLYLINE`` com flag
  de fechado (70 = 1), nunca CIRCLE. Curvas saem como polilinhas
  pelos pontos suavizados (facetação documentada, não spline).
  Traços com menos de 2 pontos são ignorados, como no SVG.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import config

# (largura_mm, altura_mm) por nome de papel.
PAPER_SIZES = {
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "A3": (297.0, 420.0),
    "Letter": (215.9, 279.4),
}

#: Tolerância declarada para segmentos de até 100 mm, em mm.
SCALE_TOLERANCE_MM_100 = 1.0


@dataclass(frozen=True)
class SheetScale:
    """Escala uniforme px -> mm para uma folha retificada."""

    paper: str
    paper_width_mm: float
    paper_height_mm: float
    canvas_width_px: int
    canvas_height_px: int
    mm_per_px: float
    effective_width_mm: float
    effective_height_mm: float

    def to_dict(self):
        return {
            "paper": self.paper,
            "paper_mm": [self.paper_width_mm, self.paper_height_mm],
            "canvas_px": [self.canvas_width_px, self.canvas_height_px],
            "scale_mm_per_px": self.mm_per_px,
            "effective_mm": [self.effective_width_mm, self.effective_height_mm],
            "units": "mm",
        }


def resolve_paper(paper):
    """Normaliza ``paper`` para (nome, largura_mm, altura_mm).

    Aceita um nome de ``PAPER_SIZES`` ou uma tupla
    ``(largura_mm, altura_mm)``.
    """
    if isinstance(paper, str):
        if paper not in PAPER_SIZES:
            raise ValueError(
                f"papel desconhecido: {paper!r} "
                f"(conhecidos: {sorted(PAPER_SIZES)})"
            )
        width_mm, height_mm = PAPER_SIZES[paper]
        return paper, width_mm, height_mm
    width_mm, height_mm = paper
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("dimensões do papel devem ser positivas")
    return "custom", float(width_mm), float(height_mm)


def sheet_scale(canvas_width=None, canvas_height=None, paper="A4"):
    """Devolve a escala uniforme para um canvas e um papel.

    A escala é a razão limitante (``min``), para a folha efetiva
    caber dentro do papel nominal sem distorção. Para o A4
    padrão (850 x 1200 px) isso ancora na largura:
    ``mm_per_px = 210 / 850`` e a altura efetiva é 296,47 mm.
    """
    canvas_width = config.PAPER_W if canvas_width is None else canvas_width
    canvas_height = config.PAPER_H if canvas_height is None else canvas_height
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("dimensões do canvas devem ser positivas")
    name, paper_width_mm, paper_height_mm = resolve_paper(paper)
    mm_per_px = min(
        paper_width_mm / canvas_width,
        paper_height_mm / canvas_height,
    )
    return SheetScale(
        paper=name,
        paper_width_mm=paper_width_mm,
        paper_height_mm=paper_height_mm,
        canvas_width_px=int(canvas_width),
        canvas_height_px=int(canvas_height),
        mm_per_px=mm_per_px,
        effective_width_mm=canvas_width * mm_per_px,
        effective_height_mm=canvas_height * mm_per_px,
    )


def px_to_mm(point, scale):
    """Converte um ponto do canvas (px) para milímetros."""
    x, y = point
    return (float(x) * scale.mm_per_px, float(y) * scale.mm_per_px)


def mm_to_px(point_mm, scale):
    """Converte um ponto em milímetros de volta para o canvas (px)."""
    x, y = point_mm
    return (float(x) / scale.mm_per_px, float(y) / scale.mm_per_px)


def stroke_length_mm(stroke, scale):
    """Comprimento de um traço em milímetros, na escala dada."""
    total = 0.0
    points = stroke.points
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        total += math.hypot(x2 - x1, y2 - y1)
    if getattr(stroke, "closed", False) and len(points) >= 2:
        (x1, y1), (x2, y2) = points[-1], points[0]
        total += math.hypot(x2 - x1, y2 - y1)
    return total * scale.mm_per_px


def _path_data_mm(stroke, scale):
    points_mm = [px_to_mm(p, scale) for p in stroke.points]
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points_mm)
    if getattr(stroke, "closed", False):
        d += " Z"
    return d


def strokes_to_svg_scaled(strokes, scale=None, paper="A4",
                          stroke_width_mm=0.5):
    """Serializa traços em SVG com coordenadas em milímetros.

    O padrão de ``stroke_width_mm`` (0,5 mm) preserva o peso
    visual do SVG antigo (2 px a ~0,247 mm/px = 0,49 mm).
    A ordem dos ``<path>`` é a ordem da lista recebida.
    """
    if scale is None:
        scale = sheet_scale(paper=paper)
    eff_w = scale.effective_width_mm
    eff_h = scale.effective_height_mm
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{eff_w:.2f}mm" height="{eff_h:.2f}mm" '
        f'viewBox="0 0 {eff_w:.2f} {eff_h:.2f}">',
        f"<!-- paper={scale.paper} scale_mm_per_px={scale.mm_per_px:.6f} "
        f"canvas_px={scale.canvas_width_px}x{scale.canvas_height_px} -->",
    ]
    for stroke in strokes:
        if len(stroke.points) < 2:
            continue
        attrs = ""
        stroke_id = getattr(stroke, "id", None)
        stroke_order = getattr(stroke, "order", None)
        if stroke_id is not None:
            attrs += f' data-id="{stroke_id}"'
        if stroke_order is not None:
            attrs += f' data-order="{stroke_order}"'
        lines.append(
            f'<path d="{_path_data_mm(stroke, scale)}" fill="none" '
            f'stroke="black" stroke-width="{stroke_width_mm}mm" '
            f'stroke-linecap="round" stroke-linejoin="round"{attrs}/>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def strokes_to_dict_scaled(strokes, scale=None, paper="A4"):
    """JSON com os campos v1 intactos mais escala/papel.

    ``width``/``height``/``points`` continuam em pixels, exatamente
    como no formato v1, para nenhum leitor atual quebrar. Os
    campos novos (``paper``, ``units``, ``scale_mm_per_px``,
    ``canvas_px``, ``sheet_mm``, ``points_mm`` por traço) são
    aditivos; um leitor v1 que só lê ``version``/``width``/
    ``height``/``strokes``/``points`` continua funcionando.
    """
    if scale is None:
        scale = sheet_scale(paper=paper)
    out_strokes = []
    for stroke in strokes:
        entry = stroke.to_dict()
        entry["points_mm"] = [
            [round(x, 3), round(y, 3)]
            for x, y in (px_to_mm(p, scale) for p in stroke.points)
        ]
        entry["length_mm"] = round(stroke_length_mm(stroke, scale), 3)
        out_strokes.append(entry)
    return {
        "version": 1,
        "width": scale.canvas_width_px,
        "height": scale.canvas_height_px,
        "paper": scale.paper,
        "units": "mm",
        "scale_mm_per_px": scale.mm_per_px,
        "canvas_px": [scale.canvas_width_px, scale.canvas_height_px],
        "sheet_mm": [round(scale.effective_width_mm, 2),
                     round(scale.effective_height_mm, 2)],
        "strokes": out_strokes,
    }


def export_strokes_scaled(strokes, base_path, scale=None, paper="A4",
                          stroke_width_mm=0.5):
    """Escreve base_path.svg (mm) e base_path.json (v1 + escala)."""
    if scale is None:
        scale = sheet_scale(paper=paper)
    base = Path(base_path)
    svg_path = base.with_suffix(".svg")
    json_path = base.with_suffix(".json")
    svg_path.write_text(
        strokes_to_svg_scaled(strokes, scale=scale,
                              stroke_width_mm=stroke_width_mm),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(strokes_to_dict_scaled(strokes, scale=scale), indent=2),
        encoding="utf-8",
    )
    return svg_path, json_path


def px_to_dxf(point, scale):
    """Converte um ponto do canvas (px) para coordenadas DXF (mm).

    O eixo y é invertido em relação ao SVG/JSON: o CAD mede y
    para cima a partir da borda inferior da folha, enquanto o
    Draw mede y para baixo a partir da borda superior.
    """
    x_mm, _y_mm = px_to_mm(point, scale)
    _x, y_px = point
    return (x_mm, scale.effective_height_mm - float(y_px) * scale.mm_per_px)


def strokes_to_dxf(strokes, scale=None, paper="A4"):
    """Serializa traços em DXF ASCII R2000, em milímetros.

    Um ``LWPOLYLINE`` por traço, na layer ``DRAW_STROKES``; a
    flag 70 é 1 para traço fechado e 0 para aberto, e o ponto
    inicial NÃO é repetido no fim de um traço fechado (a flag
    já fecha a polilinha). A ordem das entidades é a ordem da
    lista recebida: para uma sessão, passe os traços em ordem
    de sequência (``tracker.ordered``); para um snapshot, a
    ordem atual por centroide. Sem dependência nova: o texto
    é montado à mão, par por par (código de grupo, valor).
    """
    if scale is None:
        scale = sheet_scale(paper=paper)
    lines = [
        "0", "SECTION", "2", "HEADER",
        "9", "$ACADVER", "1", "AC1015",
        "9", "$INSUNITS", "70", "4",
        "9", "$MEASUREMENT", "70", "1",
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
    ]
    for stroke in strokes:
        if len(stroke.points) < 2:
            continue
        vertices = [px_to_dxf(p, scale) for p in stroke.points]
        closed = bool(getattr(stroke, "closed", False))
        lines += [
            "0", "LWPOLYLINE",
            "100", "AcDbEntity",
            "8", "DRAW_STROKES",
            "100", "AcDbPolyline",
            "90", str(len(vertices)),
            "70", "1" if closed else "0",
        ]
        for x, y in vertices:
            lines += ["10", f"{x:.3f}", "20", f"{y:.3f}"]
    lines += ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(lines) + "\n"


def export_strokes_dxf(strokes, base_path, scale=None, paper="A4"):
    """Escreve base_path.dxf (mm, R2000) e devolve o caminho."""
    if scale is None:
        scale = sheet_scale(paper=paper)
    base = Path(base_path)
    dxf_path = base.with_suffix(".dxf")
    dxf_path.write_text(
        strokes_to_dxf(strokes, scale=scale), encoding="utf-8"
    )
    return dxf_path
