"""Contrato de _compose_view e dos quadros de aviso (draw/view.py).

_compose_view é a única função de desenho que recebe o canvas do
chamador. O putText do rótulo "folha ao vivo" escrevia no array
recebido quando ele já estava na altura alvo (o _fit devolve o
próprio array nesse caso): com um canvas de 640 de altura, compor a
vista pintava o canvas do chamador e o cache de módulo _BLANK_SHEET.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import config  # noqa: E402
from draw import view as view_module  # noqa: E402


def _camera(height=720, width=1280):
    return np.full((height, width, 3), 120, dtype=np.uint8)


def test_view_has_the_fixed_size():
    view = view_module._compose_view(_camera(), np.full((1200, 850, 3), 255, np.uint8))
    assert view.shape == (640, 1600, 3)
    assert view.dtype == np.uint8


def test_canvas_at_target_height_is_not_mutated():
    canvas = np.full((640, 850, 3), 255, dtype=np.uint8)
    before = canvas.copy()
    view_module._compose_view(_camera(), canvas)
    assert np.array_equal(canvas, before), "a vista escreveu no canvas do chamador"


def test_canvas_smaller_than_the_target_is_not_mutated_either():
    canvas = np.full((300, 200, 3), 255, dtype=np.uint8)
    before = canvas.copy()
    view_module._compose_view(_camera(), canvas)
    assert np.array_equal(canvas, before)


def test_blank_sheet_cache_survives_composition():
    view_module._BLANK_SHEET = None
    blank = view_module._blank_sheet()
    view_module._BLANK_SHEET = blank
    view_module._compose_view(_camera(), blank)
    assert np.array_equal(view_module._BLANK_SHEET, view_module._blank_sheet())
    assert view_module._blank_sheet().shape == (config.PAPER_H, config.PAPER_W, 3)


def test_message_view_paints_over_a_copy():
    last = np.full((640, 1600, 3), 200, dtype=np.uint8)
    before = last.copy()
    view = view_module._message_view(last, "falhou", ["linha"])
    assert not np.array_equal(view, before)
    assert np.array_equal(last, before)


def test_message_view_without_a_last_frame_uses_the_view_size():
    view = view_module._message_view(None, "falhou", ["linha"])
    assert view.shape == (640, 1280, 3)


def test_base_button_only_appears_in_baseline_mode():
    cam = _camera()
    canvas = np.full((1200, 850, 3), 255, np.uint8)
    default = view_module._compose_view(cam, canvas)
    baseline = view_module._compose_view(cam, canvas, show_base=True)
    footer = slice(600, 640)
    assert not np.array_equal(default[footer], baseline[footer])
    # E o modo padrao continua estavel depois de o baseline ter rodado:
    # os templates em cache nao se contaminam entre modos.
    again = view_module._compose_view(cam, canvas)
    assert np.array_equal(default, again)


def test_composed_view_does_not_share_the_cached_template():
    cam = _camera()
    canvas = np.full((1200, 850, 3), 255, np.uint8)
    first = view_module._compose_view(cam, canvas)
    snapshot = first.copy()
    first[:] = 0  # o chamador pode pintar sobre a vista devolvida
    second = view_module._compose_view(cam, canvas)
    assert np.array_equal(second, snapshot)
