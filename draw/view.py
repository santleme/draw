"""Composicao da janela do app: camera e canvas lado a lado em estilo Studio / Glass.

Monta a interface com:
- Barra superior Studio (logo DRAW, indicador de captura ao vivo e titulos editoriais).
- Barra vertical compacta de atalhos a esquerda da camera ([S], [C], [F], [Q]).
- Painel central com viewport dedicado para a visao da camera.
- Painel direito com viewport dedicado e paspatur para a folha digital, no mesmo padrao da camera.
- Rodape sutil com informacoes de formato e identidade.
- Quadros de aviso e erro em estilo card de vidro translucido.
"""

import cv2
import numpy as np

from . import ink as ink_module

# Cores do sistema de design (BGR para OpenCV)
# Baseadas em design.md:
# Light canvas: warm off-white #f7f7f4 -> (244, 247, 247)
# Ink: soft near-black #22231f -> (31, 35, 34)
# Muted text: #777870 -> (112, 120, 119)
# Accent: pale paper yellow #e7dfa1 -> (161, 223, 231)
# Studio Dark: #171815 -> (21, 24, 23)
# Header / Footer surface: #1f201d -> (29, 32, 31)
# Subtle borders: #32342e -> (46, 52, 50)
COLOR_BG_STUDIO = (21, 24, 23)
COLOR_SURFACE_BAR = (29, 32, 31)
COLOR_BORDER_SUBTLE = (46, 52, 50)
COLOR_BORDER_CARD = (60, 66, 64)
COLOR_TEXT_BRIGHT = (244, 247, 247)
COLOR_TEXT_SOFT = (210, 215, 212)
COLOR_TEXT_MUTED = (112, 120, 119)
COLOR_ACCENT_YELLOW = (161, 223, 231)
COLOR_ACCENT_GREEN = (120, 210, 140)
COLOR_ACCENT_RED = (90, 100, 230)

# Cores do painel direito (valores >= 60 para que ink detection nao confunda UI com tinta)
COLOR_BG_CANVAS_SLOT = (64, 68, 65)
COLOR_CANVAS_SLOT_CONTAINER = (70, 75, 72)
COLOR_CANVAS_BORDER = (85, 92, 88)
COLOR_PAPER_BAR = (68, 72, 70)
COLOR_PAPER_BORDER = (80, 85, 82)

_BLANK_SHEET = None


def _fit(image, max_h=700):
    """Ajusta a altura de um quadro sem copiar quando ele ja cabe."""
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


def _blank_sheet():
    global _BLANK_SHEET
    shape = (ink_module.config.PAPER_H, ink_module.config.PAPER_W)
    if _BLANK_SHEET is None or _BLANK_SHEET.shape[:2] != shape:
        _BLANK_SHEET = ink_module.render_canvas(np.zeros(shape, dtype=np.uint8))
    return _BLANK_SHEET


def _draw_pill(img, x, y, w, h, bg_color, border_color=None, radius=6, alpha=0.85):
    """Desenha um pill / retangulo arredondado translucido."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    ih, iw = img.shape[:2]
    if x < 0 or y < 0 or x + w > iw or y + h > ih or w <= 0 or h <= 0:
        return
    radius = min(radius, w // 2, h // 2)

    sub = img[y : y + h, x : x + w]
    overlay = sub.copy()

    cv2.rectangle(overlay, (radius, 0), (w - radius, h), bg_color, -1)
    cv2.rectangle(overlay, (0, radius), (w, h - radius), bg_color, -1)
    for cx, cy in [
        (radius, radius),
        (w - radius, radius),
        (radius, h - radius),
        (w - radius, h - radius),
    ]:
        cv2.circle(overlay, (cx, cy), radius, bg_color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)

    if border_color is not None:
        cv2.line(sub, (radius, 0), (w - radius, 0), border_color, 1, cv2.LINE_AA)
        cv2.line(
            sub, (radius, h - 1), (w - radius, h - 1), border_color, 1, cv2.LINE_AA
        )
        cv2.line(sub, (0, radius), (0, h - radius), border_color, 1, cv2.LINE_AA)
        cv2.line(
            sub, (w - 1, radius), (w - 1, h - radius), border_color, 1, cv2.LINE_AA
        )


def _draw_sheet_reticle(status, corners, color=COLOR_ACCENT_YELLOW, bracket_len=24):
    """Desenha cantoneiras em L e contorno na folha rastreada."""
    if corners is None or len(corners) != 4:
        return
    poly = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(status, [poly], True, color, 2, cv2.LINE_AA)

    for i, pt in enumerate(corners):
        x, y = int(pt[0]), int(pt[1])
        p_prev = corners[(i - 1) % 4]
        p_next = corners[(i + 1) % 4]

        v1 = p_prev - pt
        v2 = p_next - pt
        n1 = v1 / (np.linalg.norm(v1) + 1e-6)
        n2 = v2 / (np.linalg.norm(v2) + 1e-6)

        arm1 = (pt + n1 * bracket_len).astype(int)
        arm2 = (pt + n2 * bracket_len).astype(int)

        cv2.line(status, (x, y), tuple(arm1), color, 2, cv2.LINE_AA)
        cv2.line(status, (x, y), tuple(arm2), color, 2, cv2.LINE_AA)
        cv2.circle(status, (x, y), 3, COLOR_TEXT_BRIGHT, -1, cv2.LINE_AA)


def _draw_status_badge(status, text, status_color=None):
    """Desenha o badge de status da camera no canto inferior esquerdo."""
    if not text:
        return
    badge_w = max(160, int(len(text) * 8.5 + 46))
    badge_h = 26
    by = status.shape[0] - badge_h - 16
    bx = 16
    dot_color = status_color if status_color is not None else COLOR_ACCENT_GREEN
    _draw_pill(
        status,
        bx,
        by,
        badge_w,
        badge_h,
        (20, 22, 21),
        COLOR_BORDER_CARD,
        radius=13,
        alpha=0.88,
    )
    cv2.circle(status, (bx + 14, by + 13), 4, dot_color, -1, cv2.LINE_AA)
    cv2.putText(
        status,
        text,
        (bx + 26, by + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_BRIGHT,
        1,
        cv2.LINE_AA,
    )


def _compose_view(camera_img, canvas_img):
    """Janela unica: camera e folha digital em viewports dedicados simetricos.

    Tamanho final fixo (1600x640), garantindo contraste, legibilidade,
    enquadramento simetrico da folha no mesmo padrao da camera e sem mutacao.
    """
    target_w, target_h = 1600, 640
    header_h = 44
    footer_h = 36
    panel_h = target_h - header_h - footer_h - 8  # 552px
    panel_y = header_h + 4
    split_x = 1120

    studio = np.empty((target_h, target_w, 3), dtype=np.uint8)
    # Lado esquerdo: dark studio
    studio[:, :split_x] = COLOR_BG_STUDIO
    # Lado direito: studio mat para folha digital (valores >= 60 para ink detection)
    studio[:, split_x:] = COLOR_BG_CANVAS_SLOT

    # 1. Barra Superior (Header Studio)
    # Header esquerdo (dark)
    cv2.rectangle(studio, (0, 0), (split_x, header_h), COLOR_SURFACE_BAR, -1)
    cv2.line(
        studio, (0, header_h), (split_x, header_h), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA
    )

    # Header direito
    cv2.rectangle(studio, (split_x, 0), (target_w, header_h), COLOR_PAPER_BAR, -1)
    cv2.line(
        studio,
        (split_x, header_h),
        (target_w, header_h),
        COLOR_PAPER_BORDER,
        1,
        cv2.LINE_AA,
    )

    # Divisor vertical entre camera e canvas
    cv2.line(
        studio, (split_x, 0), (split_x, target_h), COLOR_BORDER_SUBTLE, 1, cv2.LINE_AA
    )

    # Logo DRAW e badge STUDIO
    cv2.putText(
        studio,
        "DRAW",
        (24, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        COLOR_TEXT_BRIGHT,
        1,
        cv2.LINE_AA,
    )
    cv2.line(studio, (108, 14), (108, 30), (55, 60, 58), 1, cv2.LINE_AA)
    cv2.putText(
        studio,
        "STUDIO",
        (122, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_ACCENT_YELLOW,
        1,
        cv2.LINE_AA,
    )

    # Centro: Indicador de captura ao vivo
    cv2.circle(studio, (680, 22), 4, COLOR_ACCENT_GREEN, -1, cv2.LINE_AA)
    cv2.putText(
        studio,
        "CAPTURA AO VIVO",
        (694, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_SOFT,
        1,
        cv2.LINE_AA,
    )

    # 2. Barra Inferior (Footer limpo)
    cv2.rectangle(
        studio, (0, target_h - footer_h), (split_x, target_h), COLOR_SURFACE_BAR, -1
    )
    cv2.line(
        studio,
        (0, target_h - footer_h),
        (split_x, target_h - footer_h),
        COLOR_BORDER_SUBTLE,
        1,
        cv2.LINE_AA,
    )

    # Footer direito
    cv2.rectangle(
        studio,
        (split_x, target_h - footer_h),
        (target_w, target_h),
        COLOR_PAPER_BAR,
        -1,
    )
    cv2.line(
        studio,
        (split_x, target_h - footer_h),
        (target_w, target_h - footer_h),
        COLOR_PAPER_BORDER,
        1,
        cv2.LINE_AA,
    )

    # 3. Teclas de atalho no rodape (esquerda, abaixo da camera)
    buttons = [
        ("[S]", "Salvar"),
        ("[C]", "Base"),
        ("[F]", "Frame"),
        ("[Q]", "Sair"),
    ]
    btn_h = 22
    btn_y = target_h - footer_h + (footer_h - btn_h) // 2
    btn_x = 12
    for key, label in buttons:
        bw = 94
        _draw_pill(
            studio,
            btn_x,
            btn_y,
            bw,
            btn_h,
            (35, 38, 36),
            (58, 64, 61),
            radius=5,
            alpha=1.0,
        )
        cv2.putText(
            studio,
            key,
            (btn_x + 9, btn_y + 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.38,
            COLOR_ACCENT_YELLOW,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            studio,
            label,
            (btn_x + 40, btn_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_TEXT_SOFT,
            1,
            cv2.LINE_AA,
        )
        btn_x += bw + 8

    # 4. Viewport da Camera (largura total do lado esquerdo)
    cam_area_x = 12
    cam_area_w = split_x - cam_area_x - 12
    cv2.rectangle(
        studio,
        (cam_area_x, panel_y),
        (cam_area_x + cam_area_w, panel_y + panel_h),
        (26, 28, 26),
        -1,
    )
    cv2.rectangle(
        studio,
        (cam_area_x, panel_y),
        (cam_area_x + cam_area_w, panel_y + panel_h),
        COLOR_BORDER_SUBTLE,
        1,
        cv2.LINE_AA,
    )

    cam_h, cam_w = camera_img.shape[:2]
    aspect_cam = cam_w / max(1, cam_h)
    inner_cam_h = panel_h - 8
    inner_cam_w = int(inner_cam_h * aspect_cam)
    if inner_cam_w > cam_area_w - 8:
        inner_cam_w = cam_area_w - 8
        inner_cam_h = int(inner_cam_w / aspect_cam)

    cam_resized = cv2.resize(camera_img, (inner_cam_w, inner_cam_h))
    cam_x = cam_area_x + (cam_area_w - inner_cam_w) // 2
    cam_y = panel_y + (panel_h - inner_cam_h) // 2
    studio[cam_y : cam_y + inner_cam_h, cam_x : cam_x + inner_cam_w] = cam_resized
    cv2.rectangle(
        studio,
        (cam_x, cam_y),
        (cam_x + inner_cam_w, cam_y + inner_cam_h),
        (50, 56, 54),
        1,
        cv2.LINE_AA,
    )

    _draw_pill(
        studio,
        cam_area_x + 12,
        panel_y + 12,
        90,
        22,
        (20, 22, 21),
        (60, 66, 64),
        radius=11,
        alpha=0.9,
    )
    cv2.circle(
        studio, (cam_area_x + 22, panel_y + 23), 3, COLOR_ACCENT_YELLOW, -1, cv2.LINE_AA
    )
    cv2.putText(
        studio,
        "CAMERA",
        (cam_area_x + 32, panel_y + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        COLOR_TEXT_BRIGHT,
        1,
        cv2.LINE_AA,
    )

    # 5. Viewport da Folha Digital (Container dedicado no mesmo padrão da câmera!)
    canvas_area_x = split_x + 12
    canvas_area_w = target_w - canvas_area_x - 12
    cv2.rectangle(
        studio,
        (canvas_area_x, panel_y),
        (canvas_area_x + canvas_area_w, panel_y + panel_h),
        COLOR_CANVAS_SLOT_CONTAINER,
        -1,
    )
    cv2.rectangle(
        studio,
        (canvas_area_x, panel_y),
        (canvas_area_x + canvas_area_w, panel_y + panel_h),
        COLOR_CANVAS_BORDER,
        1,
        cv2.LINE_AA,
    )

    canvas_h, canvas_w = canvas_img.shape[:2]
    aspect_sheet = canvas_w / max(1, canvas_h)
    inner_sheet_h = panel_h - 8
    inner_sheet_w = int(inner_sheet_h * aspect_sheet)
    if inner_sheet_w > canvas_area_w - 16:
        inner_sheet_w = canvas_area_w - 16
        inner_sheet_h = int(inner_sheet_w / aspect_sheet)

    sheet_resized = cv2.resize(canvas_img, (inner_sheet_w, inner_sheet_h))
    sheet_x = canvas_area_x + (canvas_area_w - inner_sheet_w) // 2
    sheet_y = panel_y + (panel_h - inner_sheet_h) // 2

    # Sombra sutil de apoio do papel
    cv2.rectangle(
        studio,
        (sheet_x - 2, sheet_y - 2),
        (sheet_x + inner_sheet_w + 2, sheet_y + inner_sheet_h + 2),
        (60, 64, 62),
        -1,
    )
    studio[sheet_y : sheet_y + inner_sheet_h, sheet_x : sheet_x + inner_sheet_w] = (
        sheet_resized
    )
    cv2.rectangle(
        studio,
        (sheet_x, sheet_y),
        (sheet_x + inner_sheet_w, sheet_y + inner_sheet_h),
        (180, 185, 182),
        1,
        cv2.LINE_AA,
    )

    return studio


def _message_view(last_view, title, lines):
    """Aviso desenhado em card de vidro sobre a ultima vista ou fundo solido."""
    if last_view is not None:
        view = last_view.copy()
    else:
        view = np.full((640, 1280, 3), COLOR_BG_STUDIO, dtype=np.uint8)

    vh, vw = view.shape[:2]
    avail_w = 1120 if vw >= 1600 else vw
    card_w = min(680, avail_w - 60)
    card_h = min(280, vh - 60)
    cx = (avail_w - card_w) // 2
    cy = (vh - card_h) // 2

    # Card de vidro escuro
    _draw_pill(
        view,
        cx,
        cy,
        card_w,
        card_h,
        (28, 30, 29),
        COLOR_BORDER_CARD,
        radius=14,
        alpha=0.95,
    )

    # Badge de Aviso
    _draw_pill(
        view,
        cx + 36,
        cy + 28,
        100,
        24,
        (38, 42, 40),
        (80, 88, 85),
        radius=12,
        alpha=1.0,
    )
    cv2.circle(view, (cx + 48, cy + 40), 4, COLOR_ACCENT_RED, -1, cv2.LINE_AA)
    cv2.putText(
        view,
        "AVISO",
        (cx + 58, cy + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_TEXT_BRIGHT,
        1,
        cv2.LINE_AA,
    )

    # Titulo
    cv2.putText(
        view,
        title,
        (cx + 36, cy + 86),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        COLOR_TEXT_BRIGHT,
        1,
        cv2.LINE_AA,
    )

    # Divisor sutil
    cv2.line(
        view,
        (cx + 36, cy + 104),
        (cx + card_w - 36, cy + 104),
        COLOR_BORDER_SUBTLE,
        1,
        cv2.LINE_AA,
    )

    # Linhas explicativas
    for idx, line in enumerate(lines):
        cv2.putText(
            view,
            line,
            (cx + 36, cy + 138 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT_SOFT,
            1,
            cv2.LINE_AA,
        )

    # Instrucao de saida
    _draw_pill(
        view,
        cx + card_w - 180,
        cy + card_h - 44,
        144,
        24,
        (36, 40, 38),
        (65, 72, 70),
        radius=6,
        alpha=1.0,
    )
    cv2.putText(
        view,
        "[Q / ESC] Fechar",
        (cx + card_w - 168, cy + card_h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        COLOR_ACCENT_YELLOW,
        1,
        cv2.LINE_AA,
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
