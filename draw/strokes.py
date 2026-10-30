"""Traço vetorial (v0.2): tinta em pixels vira traços ordenados.

O pipeline v0.1 entrega uma máscara de tinta. Este módulo transforma
essa máscara em traços: caminhos ordenados de pontos, suavizados,
com comprimento e marca de fechamento, exportáveis em SVG e JSON.

Como funciona:

1. Afinamento (thinning) da máscara, para reduzir cada traço a um
   esqueleto de 1 pixel de largura. Sem afinamento, o contorno de um
   traço grosso vira um retângulo, não uma linha.
2. Rastreamento do esqueleto, pixel a pixel. Um traço termina num
   ponto final ou numa junção. Um X, por exemplo, vira 4 segmentos
   que se encontram no centro, e não um único caminho. Isso é
   intencional: junções dividem traços.
3. Suavização por média móvel, com as pontas preservadas.
4. Ordenação determinística: por centroide, de cima para baixo e
   da esquerda para a direita. Esta é a ordem de leitura, não a
   ordem temporal em que o traço foi desenhado. Rastrear a ordem
   temporal real, frame a frame, é trabalho futuro.

Limites conhecidos:

- Traços muito curtos, abaixo de `min_length`, são descartados
  como ruído.
- O thinning pode deixar corridas diagonais de dois pixels que
  parecem junções. O rastreamento de ciclos continua por essas
  corridas; os caminhos iniciados em pontas ainda param em junções.
- O afinamento depende de `cv2.ximgproc`, que vem com
  `opencv-contrib-python`, já listado em requirements.txt. Sem ele,
  o módulo usa contornos aproximados como fallback.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass
class Stroke:
    points: list
    closed: bool = False

    @property
    def length(self):
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for (x1, y1), (x2, y2) in zip(self.points, self.points[1:]):
            total += math.hypot(x2 - x1, y2 - y1)
        if self.closed:
            (x1, y1), (x2, y2) = self.points[-1], self.points[0]
            total += math.hypot(x2 - x1, y2 - y1)
        return total

    @property
    def centroid(self):
        if not self.points:
            return (0.0, 0.0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def to_dict(self):
        return {
            "points": [[float(x), float(y)] for x, y in self.points],
            "length": float(self.length),
            "closed": bool(self.closed),
        }


def smooth_points(points, passes=1):
    """Média móvel de janela 3, com pontas preservadas.

    Uma linha reta não muda: a média de pontos colineares continua
    na mesma reta. Um caminho fechado usa vizinhos circulares.
    """
    if len(points) < 3:
        return list(points)
    # Detecta caminho fechado pela proximidade entre ponta e início.
    closed = math.hypot(
        points[-1][0] - points[0][0], points[-1][1] - points[0][1]
    ) < 3.0
    current = [(float(x), float(y)) for x, y in points]
    for _ in range(passes):
        nxt = []
        n = len(current)
        for i, (x, y) in enumerate(current):
            if not closed and i in (0, n - 1):
                nxt.append((x, y))
                continue
            prev = current[i - 1]
            nxt_p = current[(i + 1) % n]
            nxt.append(
                ((prev[0] + x + nxt_p[0]) / 3.0,
                 (prev[1] + y + nxt_p[1]) / 3.0)
            )
        current = nxt
    return current


def _neighbors(pixel, skeleton_set):
    x, y = pixel
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            cand = (x + dx, y + dy)
            if cand in skeleton_set:
                out.append(cand)
    return out


def _trace_skeleton(skeleton):
    """Rastreia o esqueleto em segmentos ordenados.

    Retorna uma lista de caminhos, cada um uma lista de pontos (x, y).
    """
    pixels = set()
    ys, xs = np.where(skeleton > 0)
    for x, y in zip(xs, ys):
        pixels.add((int(x), int(y)))
    if not pixels:
        return []

    def degree(p):
        return len(_neighbors(p, pixels))

    visited_edges = set()
    segments = []

    def walk(start, stop_at_junction):
        path = [start]
        prev = None
        cur = start
        while True:
            nbrs = [
                n for n in _neighbors(cur, pixels)
                if n != prev and frozenset((cur, n)) not in visited_edges
            ]
            if not nbrs:
                break
            if prev is not None:
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                nbrs.sort(
                    key=lambda n: -((n[0] - cur[0]) * dx + (n[1] - cur[1]) * dy)
                )
            nxt = nbrs[0]
            visited_edges.add(frozenset((cur, nxt)))
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == start:
                break
            if stop_at_junction and degree(cur) >= 3:
                break
            if degree(cur) == 1:
                break
        return path

    # Primeiro, os segmentos que começam numa ponta.
    endpoints = sorted(p for p in pixels if degree(p) == 1)
    for start in endpoints:
        if any(
            frozenset((start, n)) in visited_edges
            for n in _neighbors(start, pixels)
        ):
            continue
        path = walk(start, stop_at_junction=True)
        if len(path) >= 2:
            segments.append(path)

    # Depois, os ciclos fechados, sem ponta, como um círculo.
    remaining = [
        p for p in sorted(pixels)
        if any(
            frozenset((p, n)) not in visited_edges
            for n in _neighbors(p, pixels)
        )
    ]
    for start in remaining:
        if not any(
            frozenset((start, n)) not in visited_edges
            for n in _neighbors(start, pixels)
        ):
            continue
        path = walk(start, stop_at_junction=False)
        if len(path) >= 2:
            segments.append(path)
    return segments


def _fallback_contours(ink_mask):
    contours, _ = cv2.findContours(
        ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    paths = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 2.0, True)
        points = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(points) >= 2:
            paths.append((points, True))
    return paths


def extract_strokes(ink_mask, min_length=15.0, smooth_passes=1):
    """Extrai traços de uma máscara de tinta (255 = tinta).

    Retorna uma lista de Stroke, ordenada por centroide, de cima
    para baixo e da esquerda para a direita.
    """
    if ink_mask is None or int((ink_mask > 0).sum()) == 0:
        return []
    if hasattr(cv2, "ximgproc"):
        binary = ((ink_mask > 0).astype(np.uint8)) * 255
        skeleton = cv2.ximgproc.thinning(binary)
        raw_paths = []
        for path in _trace_skeleton(skeleton):
            closed = len(path) > 2 and path[0] == path[-1]
            if closed:
                path = path[:-1]
            raw_paths.append(([(float(x), float(y)) for x, y in path], closed))
    else:
        raw_paths = _fallback_contours(ink_mask)

    strokes = []
    for points, closed in raw_paths:
        smoothed = smooth_points(points, passes=smooth_passes)
        stroke = Stroke(points=smoothed, closed=closed)
        if len(stroke.points) >= 2 and stroke.length >= min_length:
            strokes.append(stroke)
    strokes.sort(key=lambda s: (s.centroid[1], s.centroid[0]))
    return strokes


def strokes_to_svg(strokes, width=None, height=None):
    """Serializa traços em SVG, com fundo transparente."""
    width = config.PAPER_W if width is None else width
    height = config.PAPER_H if height is None else height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    for stroke in strokes:
        if len(stroke.points) < 2:
            continue
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in stroke.points)
        if stroke.closed:
            d += " Z"
        lines.append(
            f'<path d="{d}" fill="none" stroke="black" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def strokes_to_dict(strokes, width=None, height=None):
    return {
        "version": 1,
        "width": config.PAPER_W if width is None else width,
        "height": config.PAPER_H if height is None else height,
        "strokes": [s.to_dict() for s in strokes],
    }


def export_strokes(strokes, base_path, width=None, height=None):
    """Escreve base_path.svg e base_path.json. Retorna os dois caminhos."""
    base = Path(base_path)
    svg_path = base.with_suffix(".svg")
    json_path = base.with_suffix(".json")
    svg_path.write_text(
        strokes_to_svg(strokes, width=width, height=height), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(strokes_to_dict(strokes, width=width, height=height),
                   indent=2),
        encoding="utf-8",
    )
    return svg_path, json_path
