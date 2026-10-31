"""Guarda combinada da normalizacao de luz.

Caminho pedido (fiação atual do app, draw/app.py): com a flag de luz
ligada, a detecção tenta "off" primeiro; se "off" já detectar, fica
com "off"; senão tenta "background". O resultado normalizado é
descartado, voltando para a recusa de "off", quando (a) o interior
do quad, medido no frame ORIGINAL, é majoritariamente saturado, ou
(b) o quad é de quadro inteiro. A tinta continua seguindo a fiação
atual do app: warp do frame original, normalização da folha
retificada uma única vez, com o modo vencedor devolvido aqui.

Nenhum limiar de paper/ink/light muda aqui. S < 60 é o mesmo
componente de "branco" do portão de textura de paper.py, e 0,5 é a
fração que separa "majoritariamente saturado" de "majoritariamente
não saturado": descartar quando a fração S < 60 no interior é menor
que 0,5 é o caso kraft, em que o papel colorido vira "papel" só
depois que a divisão por fundo achata a cor.
"""

import cv2
import numpy as np

from . import light as light_module
from . import paper as paper_module

_MIN_UNSATURATED_FRACTION = 0.5
_SATURATION_MAX = 60
_FULL_FRAME_AREA_RATIO = 0.90
_EDGE_FRACTION = 0.025
LIGHT_GUARD_MODES = ("off", "background")


def _interior_mask(frame, corners):
    center = corners.mean(axis=0)
    shrunk = center + (corners - center) * paper_module._TEXTURE_INTERIOR_SHRINK
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shrunk.astype(np.int32)], 255)
    eroded = cv2.erode(mask, np.ones((9, 9), np.uint8))
    return eroded > 0


def unsaturated_fraction(frame, corners):
    """Fração S < 60 no interior do quad, no frame original."""
    maps = paper_module._texture_maps(frame)
    selected = _interior_mask(frame, corners)
    if not np.any(selected):
        return 0.0
    return float(np.mean(maps["saturation"][selected] < _SATURATION_MAX))


def is_mostly_saturated(frame, corners):
    """True quando o interior é majoritariamente saturado (caso kraft)."""
    return unsaturated_fraction(frame, corners) < _MIN_UNSATURATED_FRACTION


def edge_margin_px(frame_shape):
    """Margem de borda em px, relativa ao tamanho do frame.

    2,5% do menor lado do frame. A escolha reproduz, na resolução de
    referência 1600x1200, a margem de 30 px usada antes (1200 * 0,025
    = 30), sem fixar 30 px absolutos: num frame 800x600 a margem é
    15 px, num 3200x2400 é 60 px. Um canto "perto da borda" é aquele
    cuja distância à borda mais próxima é menor que essa margem.
    """
    height, width = frame_shape[:2]
    return _EDGE_FRACTION * min(height, width)


def is_full_frame_quad(frame_shape, corners):
    """True quando o quad cobre o quadro inteiro.

    Área do polígono acima de 90% da área do frame E os 4 cantos
    perto das bordas (margem relativa, ver edge_margin_px). As duas
    condições juntas evitam descartar uma folha grande mas com
    margem visível, e evitam descartar um quad pequeno colado num
    canto só.
    """
    height, width = frame_shape[:2]
    frame_area = float(height * width)
    quad_area = float(cv2.contourArea(
        np.asarray(corners, dtype=np.float32).reshape(4, 1, 2)))
    if quad_area <= _FULL_FRAME_AREA_RATIO * frame_area:
        return False
    margin = edge_margin_px(frame_shape)
    for x, y in np.asarray(corners, dtype=np.float64).reshape(4, 2):
        nearest = min(x, width - 1 - x, y, height - 1 - y)
        if nearest >= margin:
            return False
    return True


def guard_passes(frame, corners):
    """True quando o resultado normalizado pode ser mantido."""
    if corners is None:
        return False
    if is_mostly_saturated(frame, corners):
        return False
    if is_full_frame_quad(frame.shape, corners):
        return False
    return True


def detect_sheet_with_light_guard(frame, light_normalize="off"):
    """Detecção protegida. Devolve (cantos, veredito, modo aplicado).

    Com "off", é exatamente a detecção atual, modo vencedor "off".
    Com a flag ligada, tenta "off" primeiro; se "off" detectar
    (cantos não None, veredito "paper" ou "blank_maybe"), fica com
    "off". Senão tenta "background": se passar na guarda, ele vira o
    modo vencedor; se a guarda descartar, volta para o
    resultado de "off", que é a recusa (None, None), com modo "off".
    O modo vencedor é o que a tinta deve usar, uma única vez, na
    folha retificada do frame original (ver draw/app.py).
    """
    if light_normalize not in LIGHT_GUARD_MODES:
        raise ValueError(
            f"modo de guarda de luz inválido: {light_normalize!r}; "
            f"use um de {LIGHT_GUARD_MODES}"
        )
    corners, verdict = paper_module.detect_sheet_with_confidence(frame)
    if light_normalize == "off" or corners is not None:
        return corners, verdict, "off"
    bg_frame = light_module.normalize_light(frame, "background")
    normalized_corners, normalized_verdict = (
        paper_module.detect_sheet_with_confidence(bg_frame)
    )
    if normalized_corners is None or not guard_passes(
        frame, normalized_corners
    ):
        return corners, verdict, "off"
    return normalized_corners, normalized_verdict, "background"
