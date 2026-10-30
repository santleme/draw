"""Traços em ciclos e junções (etapa 3 do Senado).

Regressão da anomalia encontrada na etapa 2: o thinning deixa
corridas de 2 px em diagonais, e muitos pixels dessas corridas têm
grau 3 sem serem junção. O rastreador antigo quebrava em todo grau
>= 3, então um círculo virava centenas de fragmentos de 2 px, todos
descartados pelo min_length=15, e a cena sintética reportava só
2 traços. A correção (ciclos não quebram em grau >= 3; pontas sim)
é travada aqui com formas isoladas, sem webcam.
"""

import math
import cv2
import numpy as np
from draw import strokes


def _mask(draw_fn):
    img = np.zeros((300, 300), dtype=np.uint8)
    draw_fn(img)
    return img


def test_circle_is_one_closed_stroke():
    mask = _mask(
        lambda img: cv2.circle(img, (150, 150), 50, 255, 3)
    )
    found = strokes.extract_strokes(mask)
    assert len(found) == 1, [(s.length, s.closed) for s in found]
    assert found[0].closed
    # Circunferência 2*pi*50 = 314 px; o caminho do esqueleto fica
    # um pouco abaixo por atalhos diagonais nas corridas de 2 px.
    assert 200.0 <= found[0].length <= 350.0


def test_x_is_four_segments():
    mask = _mask(
        lambda img: (
            cv2.line(img, (50, 50), (250, 250), 255, 3),
            cv2.line(img, (250, 50), (50, 250), 255, 3),
        )
    )
    found = strokes.extract_strokes(mask)
    assert len(found) == 4, [(s.length, s.closed) for s in found]
    for stroke in found:
        assert not stroke.closed
        assert stroke.length >= 90.0


def test_straight_line_is_one_stroke():
    mask = _mask(
        lambda img: cv2.line(img, (10, 50), (200, 50), 255, 3)
    )
    found = strokes.extract_strokes(mask)
    assert len(found) == 1
    assert abs(found[0].length - 190.0) < 10.0
