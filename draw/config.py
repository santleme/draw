"""Configuração central do Draw.

A folha-modelo é um A4 com quatro marcadores ArUco, um em cada canto.
Todas as coordenadas aqui estão no espaço da folha retificada
(a "vista de cima" que o pipeline produz).
"""

import cv2

FORMATS = {
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "a3": (297.0, 420.0),
    "carta": (216.0, 279.0),
    "caderno": (200.0, 275.0),
}
PAPER_FORMAT = "a4"

# Grade A4 legada: nao derivar os 850 px da proporcao nominal.
PAPER_W = 850
PAPER_H = 1200

# Marcadores: id -> nome do canto.
MARKER_IDS = (0, 1, 2, 3)  # 0=sup. esq., 1=sup. dir., 2=inf. dir., 3=inf. esq.
MARKER_SIZE_PX = 120

# Centros dos marcadores no espaço da folha retificada.
MARKER_CENTERS = {
    0: (90, 90),
    1: (PAPER_W - 90, 90),
    2: (PAPER_W - 90, PAPER_H - 90),
    3: (90, PAPER_H - 90),
}

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50


def paper_grid(name):
    """Grade em retrato, com borda longa de 1200 px e A4 legado pinado."""
    if name not in FORMATS:
        raise ValueError(f"formato de folha desconhecido: {name}")
    width_mm, height_mm = FORMATS[name]
    return (850 if name == "a4" else round(1200 * width_mm / height_mm), 1200)


def mm_per_px(name=None):
    name = PAPER_FORMAT if name is None else name
    if name not in FORMATS:
        raise ValueError(f"formato de folha desconhecido: {name}")
    return FORMATS[name][1] / 1200


def apply_format(name):
    """Configura a grade antes de construir os consumidores da sessao."""
    global PAPER_FORMAT, PAPER_W, PAPER_H, MARKER_CENTERS
    width, height = paper_grid(name)
    PAPER_FORMAT, PAPER_W, PAPER_H = name, width, height
    MARKER_CENTERS = {
        0: (90, 90),
        1: (width - 90, 90),
        2: (width - 90, height - 90),
        3: (90, height - 90),
    }
    return width, height


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
