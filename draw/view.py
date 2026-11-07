"""Composicao da janela do app: camera e canvas lado a lado.

Este bloco morava em draw/app.py e saiu sem mudanca de logica.
_fit ajusta a altura de um quadro sem copiar quando ele ja cabe.
_blank_sheet cacheia a folha em branco renderizada. _compose_view
monta a vista unica (camera a esquerda, folha ao vivo a direita) e
_message_view desenha os quadros de aviso sobre a ultima vista.
"""

import cv2
import numpy as np

from . import ink as ink_module


def _fit(image, max_h=700):
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


_BLANK_SHEET = None


def _blank_sheet():
    global _BLANK_SHEET
    shape = (ink_module.config.PAPER_H, ink_module.config.PAPER_W)
    if _BLANK_SHEET is None or _BLANK_SHEET.shape[:2] != shape:
        _BLANK_SHEET = ink_module.render_canvas(
            np.zeros(shape, dtype=np.uint8)
        )
    return _BLANK_SHEET


def _compose_view(camera_img, canvas_img):
    """Janela unica: camera a esquerda, folha ao vivo a direita.

    A folha ao vivo e o canvas digital (fundo branco, tinta preta) na
    mesma janela e na mesma altura da camera: o traco no papel e o
    traco digital aparecem lado a lado, sem janelas separadas.
    """
    target_h = 640
    cam = _fit(camera_img, target_h)
    sheet = _fit(canvas_img, target_h)
    if sheet is canvas_img:
        # _fit devolve o PRÓPRIO array quando a altura já cabe; o
        # putText abaixo escreve nele. Sem esta cópia, compor a vista
        # com um canvas de 640 de altura pintava "folha ao vivo" no
        # canvas do chamador (e no _BLANK_SHEET, que é cache de
        # módulo). O caminho normal (canvas 850x1200) já sai do _fit
        # como array novo e não paga a cópia.
        sheet = sheet.copy()
    if cam.shape[0] != sheet.shape[0]:
        h = min(cam.shape[0], sheet.shape[0])
        cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
        sheet = cv2.resize(
            sheet, (int(sheet.shape[1] * h / sheet.shape[0]), h))
    cv2.putText(
        sheet,
        "folha ao vivo",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (140, 140, 140),
        1,
    )
    divider = np.full((cam.shape[0], 6, 3), (60, 60, 60), dtype=np.uint8)
    view = np.hstack([cam, divider, sheet])
    if view.shape[:2] != (640, 1600):
        # Tamanho final fixo: sem isto, a geometria da janela
        # depende do tamanho entregue pela camera (com 1289x710 o
        # resize antigo cortava a altura para 632; com 1280x720
        # ele nem rodava e a vista saia com 1596 de largura).
        view = cv2.resize(view, (1600, 640))
    return view


def _message_view(last_view, title, lines):
    """Aviso desenhado sobre a ultima vista, ou sobre fundo solido.

    A fonte Hershey do OpenCV nao tem acento: as mensagens chegam
    sem acentuacao, como o restante dos textos que o app desenha.
    """
    if last_view is not None:
        view = last_view.copy()
    else:
        view = np.full((640, 1280, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(
        view, title, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
    )
    for index, line in enumerate(lines):
        cv2.putText(
            view,
            line,
            (40, 140 + 40 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return view


def _dead_source_view(last_view):
    """Quadro de aviso da fonte morta, desenhado sobre a ultima vista."""
    return _message_view(
        last_view,
        "A fonte de video morreu",
        [
            "A camera parou de entregar quadros.",
            "Feche outros programas usando a camera e tente de novo.",
        ],
    )


def _error_view(last_view, title, lines):
    """Quadro de aviso de falha do pipeline, no formato da vista."""
    return _message_view(last_view, title, lines)
