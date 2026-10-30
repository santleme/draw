"""Reconhecimento de formas (etapa 7 do Senado), geometria pura.

Este módulo é a primeira fatia do reconhecimento: só linha, círculo,
retângulo e ``unknown``. Sem ML, sem câmera, sem UI, sem CAD. Ele só
lê traços (``draw/strokes.py``) e nunca os altera: em caso de dúvida
a saída é o traço cru, via ``unknown``, nunca uma forma forçada e
nunca um traço apagado ou deformado por uma classificação errada.

Como funciona, em duas etapas separadas (cada uma testável sozinha):

1. Agrupamento (``group_strokes``): segmentos que pertencem à mesma
   forma viram um grupo. Um X, por exemplo, chega como 4 segmentos
   que se encontram no centro; um retângulo desenhado em 4 traços
   chega como 4 segmentos que se encontram nos cantos. O agrupamento
   usa distância E ângulo, não só proximidade: dois traços só se
   ligam se as pontas mais próximas estiverem dentro da tolerância
   e a direção de chegada for continuação (ângulo pequeno) ou canto
   (ângulo perto de 90°). Ângulos ambíguos (entre 35° e 55°, ou
   retorno acima de 125°) não ligam — é melhor deixar dois grupos
   separados, ambos ``unknown``, do que fundir duas formas vizinhas.
2. Classificação (``classify_group``): cada grupo vira uma ``Shape``
   por regras geométricas, com confiança. Ordem de decisão: grupo
   fechado tenta retângulo, depois círculo; grupo aberto tenta
   linha; o resto é ``unknown``. Baixa confiança é ``unknown``
   legítimo: na métrica ele conta como erro, mas separado de
   rótulo forçado errado, que é o erro mais caro aqui.

Limites deliberados desta fatia (não são bugs, são escopo):

- Elipse não é classe: um laço fechado alongado (razão de aspecto
  longe de 1) vira ``unknown``. Ajuste de elipse fica para depois.
- Seta, letra, diagrama e OCR não existem aqui. Um aglomerado
  denso de traços curtos sem forma dominante vira ``unknown``;
  ``text_region`` fica para uma fatia posterior.
- Traço fechado não é círculo: sem ajuste de círculo com resíduo
  baixo, um laço fechado vira ``unknown``, nunca ``circle``.
- Um traço novo que surge colado a um traço ativo já pode ter
  sido absorvido pelo agrupamento; a classificação herda esse
  limite, não o esconde.

Calibração: todos os limiares abaixo foram calibrados na cena
sintética do selftest (círculo de raio 110 px, linha e X, espaço
da folha retificada 850 x 1200 px) e em polígonos desenhados por
código nos testes. Nenhum deles foi calibrado em foto real — o
dataset da etapa 2 existe justamente para isso, e os valores
devem ser revisitados quando houver casos reais.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

# --- Limiares nomeados (ver o docstring do módulo para a calibração) ---

# Agrupamento: pontas a até esta distância, em px da folha, podem se ligar.
GROUP_ENDPOINT_TOL_PX = 25.0
# Ângulos de ligação (graus): continuação [0, LINK_COLINEAR_MAX_DEG],
# canto [LINK_CORNER_MIN_DEG, LINK_CORNER_MAX_DEG]. Fora disso, não liga.
LINK_COLINEAR_MAX_DEG = 35.0
LINK_CORNER_MIN_DEG = 55.0
LINK_CORNER_MAX_DEG = 125.0
# Fechamento de laço: ponta final a até esta distância da ponta inicial.
CLOSE_TOL_PX = 25.0
# Linha: resíduo RMS perpendicular máximo, em px (ou 2% do comprimento).
LINE_MAX_RESIDUAL_PX = 3.0
LINE_MAX_RESIDUAL_FRAC = 0.02
# Linha: distância ponta-a-ponta mínima, como fração do comprimento.
LINE_MIN_ENDPOINT_FRAC = 0.90
# Círculo: circularidade mínima (4*pi*área/perímetro^2; círculo = 1).
CIRCLE_MIN_CIRCULARITY = 0.85
# Círculo: razão de aspecto (menor/maior) mínima — elipse alongada não é círculo.
CIRCLE_MIN_ASPECT = 0.80
# Círculo: resíduo RMS do ajuste máximo, em px (ou 5% do raio).
CIRCLE_MAX_FIT_RESIDUAL_PX = 4.0
CIRCLE_MAX_FIT_RESIDUAL_FRAC = 0.05
# Retângulo: epsilon do Douglas-Peucker, como fração do perímetro.
RECT_APPROX_EPSILON_FRAC = 0.02
# Retângulo: cada ângulo interno deve estar a até esta margem de 90°.
RECT_ANGLE_TOL_DEG = 20.0
# Retângulo: lados opostos devem ter razão dentro deste intervalo.
RECT_OPPOSITE_SIDE_MIN_RATIO = 0.75
# Retângulo: área do polígono / área da bbox mínima.
RECT_MIN_BBOX_FILL = 0.80
# Confiança mínima para um rótulo valer; abaixo disso, unknown.
CONFIDENCE_MIN = 0.50

COORDINATE_SPACE = "warped_sheet_850x1200"


@dataclass
class Shape:
    kind: str  # "line" | "circle" | "rectangle" | "unknown"
    points: list  # geometria normalizada da forma, em px da folha
    bbox: tuple  # (x, y, w, h), em px da folha
    confidence: float  # 0..1; unknown sempre carrega 0.0
    source_stroke_ids: list  # rastreabilidade até o Stroke de origem
    params: dict  # ex.: {"center": (x, y), "radius": r}

    def to_dict(self, width=None, height=None):
        width = config.PAPER_W if width is None else width
        height = config.PAPER_H if height is None else height
        return {
            "kind": self.kind,
            "points": [[float(x), float(y)] for x, y in self.points],
            "points_normalized": [
                [float(x) / width, float(y) / height] for x, y in self.points
            ],
            "bbox": [float(v) for v in self.bbox],
            "confidence": float(self.confidence),
            "source_stroke_ids": list(self.source_stroke_ids),
            "params": _jsonable_params(self.params),
        }


def _jsonable_params(params):
    out = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            out[key] = [
                [float(a), float(b)] if isinstance(v, (list, tuple)) else float(v)
                for v in value
            ] if value and isinstance(value[0], (list, tuple)) else [
                float(v) for v in value
            ]
        else:
            out[key] = float(value) if isinstance(value, (int, float)) else value
    return out


def _stroke_id(stroke, index):
    ident = getattr(stroke, "id", None)
    return ident if ident is not None else index


def _travel_direction(points, at_start):
    """Direção de percurso na ponta: para o início, do 2º para o 1º
    ponto; para o fim, do penúltimo para o último."""
    if len(points) < 2:
        return (0.0, 0.0)
    if at_start:
        (x1, y1), (x2, y2) = points[0], points[1]
    else:
        (x1, y1), (x2, y2) = points[-2], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm == 0:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


def _angle_between(d1, d2):
    dot = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1]))
    return math.degrees(math.acos(dot))


def _endpoints_link(stroke_a, stroke_b):
    """Duas pontas se ligam? Distância E ângulo, nunca só distância.

    Retorna (ponta_de_a, ponta_de_b) — "start"/"end" — ou None.
    """
    best = None
    for end_a, pt_a in (("start", stroke_a.points[0]), ("end", stroke_a.points[-1])):
        for end_b, pt_b in (("start", stroke_b.points[0]), ("end", stroke_b.points[-1])):
            dist = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
            if best is None or dist < best[0]:
                best = (dist, end_a, end_b)
    if best is None or best[0] > GROUP_ENDPOINT_TOL_PX:
        return None
    _, end_a, end_b = best
    # Direção de percurso AO CHEGAR na ponta de a e AO SAIR da ponta
    # de b (se b for ligado pelo início, o percurso sai de b em
    # direção ao seu segundo ponto; se pelo fim, o percurso chega
    # a b vindo do penúltimo — para o ângulo de junção usamos a
    # direção de percurso de cada traço na ponta, orientada para
    # fora da junção: em a, direção de chegada; em b, direção de
    # saída, que é a direção de percurso invertida quando a ponta
    # é o fim).
    dir_a = _travel_direction(stroke_a.points, at_start=(end_a == "start"))
    if end_a == "start":
        dir_a = (-dir_a[0], -dir_a[1])
    dir_b = _travel_direction(stroke_b.points, at_start=(end_b == "start"))
    if end_b == "end":
        dir_b = (-dir_b[0], -dir_b[1])
    angle = _angle_between(dir_a, dir_b)
    if angle <= LINK_COLINEAR_MAX_DEG:
        return (end_a, end_b)
    if LINK_CORNER_MIN_DEG <= angle <= LINK_CORNER_MAX_DEG:
        return (end_a, end_b)
    return None


def group_strokes(strokes):
    """Agrupa traços por ponta + ângulo. Retorna lista de grupos,
    cada grupo uma lista de Stroke, na ordem de entrada."""
    strokes = list(strokes)
    parent = list(range(len(strokes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(strokes)):
        if len(strokes[i].points) < 2:
            continue
        for j in range(i + 1, len(strokes)):
            if len(strokes[j].points) < 2:
                continue
            if _endpoints_link(strokes[i], strokes[j]) is not None:
                union(i, j)
    groups = {}
    for i, stroke in enumerate(strokes):
        if len(stroke.points) < 2:
            continue
        groups.setdefault(find(i), []).append(stroke)
    return list(groups.values())


def _order_group_points(group):
    """Ordena os pontos de um grupo em um único caminho, encadeando
    pelas pontas. Se algum traço não encadear, ele é anexado no fim
    (o classificador então trabalha com o conjunto, e a forma tende
    a unknown — melhor que inventar uma ordem)."""
    if not group:
        return []
    ordered = [list(group[0].points)]
    used = {0}
    path = list(group[0].points)
    # Estende pelo fim enquanto houver traço ligável.
    while len(used) < len(group):
        end = path[-1]
        best = None
        for i, stroke in enumerate(group):
            if i in used or len(stroke.points) < 2:
                continue
            for at_start, pt in (("start", stroke.points[0]), ("end", stroke.points[-1])):
                dist = math.hypot(end[0] - pt[0], end[1] - pt[1])
                if best is None or dist < best[0]:
                    best = (dist, i, at_start)
        if best is None or best[0] > GROUP_ENDPOINT_TOL_PX:
            break
        _, i, at_start = best
        pts = list(group[i].points)
        if at_start:
            # A ponta de ligação é o início: anexa sem repetir o ponto.
            path.extend(pts[1:])
        else:
            # Ligação pelo fim: inverte para o caminho continuar.
            path.extend(list(reversed(pts))[:-1])
        used.add(i)
    for i, stroke in enumerate(group):
        if i not in used:
            path.extend(stroke.points)
    # Remove duplicatas consecutivas (pontas compartilhadas).
    cleaned = [path[0]] if path else []
    for pt in path[1:]:
        if math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 1e-6:
            cleaned.append(pt)
    return cleaned


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0, y0, x1 - x0, y1 - y0)


def _polyline_length(points, closed=False):
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        total += math.hypot(x2 - x1, y2 - y1)
    if closed and len(points) > 2:
        total += math.hypot(
            points[-1][0] - points[0][0], points[-1][1] - points[0][1]
        )
    return total


def _polygon_area(points):
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        total += x1 * y2 - x2 * y1
    if len(points) > 2:
        total += points[-1][0] * points[0][1] - points[0][0] * points[-1][1]
    return abs(total) / 2.0


def _fit_line_residual(points):
    """Resíduo RMS perpendicular ao eixo principal (PCA)."""
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2:
        return 0.0
    center = arr.mean(axis=0)
    centered = arr - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[1]
    residuals = np.abs(centered @ normal)
    return float(np.sqrt(np.mean(residuals**2)))


def _fit_circle(points):
    """Ajuste algébrico de círculo (Kasa). Retorna (centro, raio, rms)."""
    arr = np.asarray(points, dtype=float)
    x, y = arr[:, 0], arr[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    d, e, f = sol
    cx, cy = -d / 2.0, -e / 2.0
    r2 = (d**2 + e**2) / 4.0 - f
    if r2 <= 0:
        return None
    radius = math.sqrt(r2)
    dists = np.hypot(x - cx, y - cy)
    rms = float(np.sqrt(np.mean((dists - radius) ** 2)))
    return (cx, cy), radius, rms


def _approx_vertices(points, closed):
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = _polyline_length(points, closed=closed)
    approx = cv2.approxPolyDP(
        contour, RECT_APPROX_EPSILON_FRAC * perimeter, closed
    )
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def _interior_angles(corners):
    angles = []
    n = len(corners)
    for i in range(n):
        prev = corners[i - 1]
        cur = corners[i]
        nxt = corners[(i + 1) % n]
        v1 = (prev[0] - cur[0], prev[1] - cur[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            angles.append(0.0)
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        angles.append(math.degrees(math.acos(dot)))
    return angles


def _classify_closed(points, bbox):
    """Grupo fechado: retângulo, depois círculo, senão unknown."""
    corners = _approx_vertices(points, closed=True)
    if len(corners) == 4:
        angles = _interior_angles(corners)
        sides = [
            math.hypot(
                corners[(i + 1) % 4][0] - corners[i][0],
                corners[(i + 1) % 4][1] - corners[i][1],
            )
            for i in range(4)
        ]
        opposite_ok = (
            sides[0] > 0
            and sides[1] > 0
            and RECT_OPPOSITE_SIDE_MIN_RATIO
            <= sides[0] / sides[2]
            <= 1.0 / RECT_OPPOSITE_SIDE_MIN_RATIO
            and RECT_OPPOSITE_SIDE_MIN_RATIO
            <= sides[1] / sides[3]
            <= 1.0 / RECT_OPPOSITE_SIDE_MIN_RATIO
        )
        area = _polygon_area(corners)
        bbox_area = bbox[2] * bbox[3]
        fill_ok = bbox_area > 0 and area / bbox_area >= RECT_MIN_BBOX_FILL
        if (
            all(abs(a - 90.0) <= RECT_ANGLE_TOL_DEG for a in angles)
            and opposite_ok
            and fill_ok
        ):
            max_dev = max(abs(a - 90.0) for a in angles)
            confidence = max(
                CONFIDENCE_MIN,
                min(0.99, 1.0 - max_dev / (2.0 * RECT_ANGLE_TOL_DEG)),
            )
            return ("rectangle", corners, confidence,
                    {"corners": corners, "angles_deg": angles})
    # Círculo: circularidade alta, aspecto perto de 1, ajuste bom.
    perimeter = _polyline_length(points, closed=True)
    area = _polygon_area(points)
    if perimeter > 0:
        circularity = 4.0 * math.pi * area / (perimeter**2)
        _, _, bw, bh = bbox
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
        fit = _fit_circle(points)
        if (
            circularity >= CIRCLE_MIN_CIRCULARITY
            and aspect >= CIRCLE_MIN_ASPECT
            and fit is not None
        ):
            (cx, cy), radius, rms = fit
            if rms <= max(CIRCLE_MAX_FIT_RESIDUAL_PX,
                          CIRCLE_MAX_FIT_RESIDUAL_FRAC * radius):
                confidence = max(
                    CONFIDENCE_MIN,
                    min(0.99, 0.50 + 0.49 * (circularity - CIRCLE_MIN_CIRCULARITY)
                        / (1.0 - CIRCLE_MIN_CIRCULARITY)),
                )
                sampled = [
                    (cx + radius * math.cos(2.0 * math.pi * k / 64.0),
                     cy + radius * math.sin(2.0 * math.pi * k / 64.0))
                    for k in range(64)
                ]
                return ("circle", sampled, confidence,
                        {"center": (cx, cy), "radius": radius,
                         "circularity": circularity, "fit_residual_rms": rms})
    return ("unknown", list(points), 0.0, {})


def _classify_open(points):
    """Grupo aberto: linha, senão unknown."""
    length = _polyline_length(points, closed=False)
    if length <= 0:
        return ("unknown", list(points), 0.0, {})
    residual = _fit_line_residual(points)
    endpoint_dist = math.hypot(
        points[-1][0] - points[0][0], points[-1][1] - points[0][1]
    )
    threshold = max(LINE_MAX_RESIDUAL_PX, LINE_MAX_RESIDUAL_FRAC * length)
    if residual <= threshold and endpoint_dist >= LINE_MIN_ENDPOINT_FRAC * length:
        confidence = max(
            CONFIDENCE_MIN, min(0.99, 1.0 - residual / (2.0 * threshold))
        )
        return ("line", [points[0], points[-1]], confidence,
                {"p1": points[0], "p2": points[-1],
                 "residual_rms": residual})
    return ("unknown", list(points), 0.0, {})


def classify_group(group, source_ids=None):
    """Classifica um grupo de traços. Nunca força um rótulo."""
    points = _order_group_points(group)
    if len(points) < 2:
        return Shape(kind="unknown", points=list(points), bbox=(0.0, 0.0, 0.0, 0.0),
                     confidence=0.0, source_stroke_ids=list(source_ids or []),
                     params={})
    bbox = _bbox(points)
    closed = any(getattr(s, "closed", False) for s in group) or (
        math.hypot(points[-1][0] - points[0][0],
                   points[-1][1] - points[0][1]) <= CLOSE_TOL_PX
        and len(points) > 2
    )
    if closed:
        kind, geom, confidence, params = _classify_closed(points, bbox)
    else:
        kind, geom, confidence, params = _classify_open(points)
    if confidence < CONFIDENCE_MIN:
        kind, confidence, params = "unknown", 0.0, {}
        geom = list(points)
    return Shape(kind=kind, points=geom, bbox=_bbox(geom) if geom else bbox,
                 confidence=confidence,
                 source_stroke_ids=list(source_ids or []), params=params)


def classify_strokes(strokes):
    """Agrupa e classifica traços. Retorna uma Shape por grupo, na
    ordem dos grupos (ordem de entrada dos traços)."""
    strokes = list(strokes)
    shapes = []
    for group in group_strokes(strokes):
        ids = [_stroke_id(s, strokes.index(s)) for s in group]
        shapes.append(classify_group(group, source_ids=ids))
    return shapes


def confusion_matrix(pairs):
    """Matriz de confusão a partir de pares (esperado, obtido).

    Retorna {esperado: {obtido: contagem}}. Não esconde ``unknown``:
    ele aparece como coluna própria, separado de rótulo errado.
    """
    matrix = {}
    for expected, obtained in pairs:
        matrix.setdefault(expected, {})
        matrix[expected][obtained] = matrix[expected].get(obtained, 0) + 1
    return matrix


def shapes_to_dict(shapes, width=None, height=None):
    """JSON separado do strokes.json v1: não toca no formato atual."""
    width = config.PAPER_W if width is None else width
    height = config.PAPER_H if height is None else height
    return {
        "version": 1,
        "coordinate_space": COORDINATE_SPACE,
        "width": width,
        "height": height,
        "shapes": [s.to_dict(width=width, height=height) for s in shapes],
    }


def export_shapes(shapes, path, width=None, height=None):
    """Escreve o JSON de formas no caminho dado. Retorna o caminho."""
    target = Path(path)
    target.write_text(
        json.dumps(shapes_to_dict(shapes, width=width, height=height), indent=2),
        encoding="utf-8",
    )
    return target
