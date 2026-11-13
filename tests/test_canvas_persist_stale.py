"""DRAW_CANVAS_PERSIST + DRAW_INK_SKIP_STALE (frente mão em massa).

O drop do tracker não prova troca de página — a geração da
trabalhadora já era guardada (clean_recovery_pending). Sem a flag,
o canvas confirmado sumia da tela na mesma hora; com ela, fica
congelado até o re-registro. E na janela de graça do tracker
(cantos retidos sem confirmação nova), as passadas de reuso/LK são
trabalho morto por construção — a flag as pula.
"""

import numpy as np

from draw import loop as loop_module

from tests.test_sheet_coverage import _FakeInkWorker, _SyncDetectionWorker, QUAD


def _make_loop(script):
    det = _SyncDetectionWorker(script)
    ink = _FakeInkWorker()
    loop = loop_module.SheetInkLoop(
        detection_worker=det,
        ink_worker=ink,
        clock=lambda: 0.0,
        log=lambda *_a: None,
    )
    return loop, det, ink


def _frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _seen_then_lost(miss_frames=80):
    """Vê a folha nos 6 primeiros chamados, depois perde tudo."""

    def script(i):
        if i < 6:
            return QUAD.copy(), "paper", None
        return None, None, None

    return script


def _drive(loop, steps):
    for _ in range(steps):
        loop.step(_frame(), loop.frame_number * 100.0)


def test_canvas_persist_flag_parsing(monkeypatch):
    monkeypatch.delenv("DRAW_CANVAS_PERSIST", raising=False)
    assert not loop_module._canvas_persist_enabled()
    monkeypatch.setenv("DRAW_CANVAS_PERSIST", "1")
    assert loop_module._canvas_persist_enabled()
    monkeypatch.setenv("DRAW_CANVAS_PERSIST", "lixo")
    assert not loop_module._canvas_persist_enabled()


def test_drop_clears_canvas_without_flag(monkeypatch):
    """Comportamento histórico: drop zera clean_mask na hora."""
    monkeypatch.delenv("DRAW_CANVAS_PERSIST", raising=False)
    loop, _det, _ink = _make_loop(_seen_then_lost())
    _drive(loop, 8)
    loop.clean_mask = np.ones((100, 100), np.uint8) * 255
    _drive(loop, 80)
    assert loop.generation_events["drop"] >= 1
    assert loop.clean_mask is None


def test_drop_keeps_canvas_with_flag(monkeypatch):
    """Com a flag: a queda congela o canvas — o usuário segue vendo
    o desenho enquanto a folha não é re-adquirida."""
    monkeypatch.setenv("DRAW_CANVAS_PERSIST", "1")
    loop, _det, _ink = _make_loop(_seen_then_lost())
    _drive(loop, 8)
    canvas = np.ones((100, 100), np.uint8) * 255
    loop.clean_mask = canvas.copy()
    _drive(loop, 80)
    assert loop.generation_events["drop"] >= 1
    assert loop.clean_mask is not None
    np.testing.assert_array_equal(loop.clean_mask, canvas)


def test_skip_stale_flag_parsing(monkeypatch):
    monkeypatch.delenv("DRAW_INK_SKIP_STALE", raising=False)
    assert not loop_module._ink_skip_stale_enabled()
    monkeypatch.setenv("DRAW_INK_SKIP_STALE", "1")
    assert loop_module._ink_skip_stale_enabled()


def test_stale_grace_passes_skipped_with_flag(monkeypatch):
    """Passadas sobre cantos de graça (> limiar sem confirmação nova)
    não sobem — é mesa/oclusão por construção."""
    monkeypatch.setenv("DRAW_INK_SKIP_STALE", "1")
    # Limiar baixo para abrir a janela "graca ativa, flag acordada"
    # dentro do _SHEET_LOST_MIN_FRAMES real da queda.
    monkeypatch.setattr(loop_module, "_STALE_PASS_MIN_FRAMES", 5)
    loop, _det, ink = _make_loop(_seen_then_lost())
    stale_seen = False
    for _ in range(60):
        loop.step(_frame(), loop.frame_number * 100.0)
        if loop.tracker._frames_since_seen > 5:
            stale_seen = True
            submits_at_stale = len(ink.submitted)
            for _ in range(6):
                loop.step(_frame(), loop.frame_number * 100.0)
            # graca ativa ou queda consumada: nenhuma submissao a mais
            assert len(ink.submitted) == submits_at_stale
            break
    assert stale_seen


def test_stale_passes_continue_without_flag(monkeypatch):
    """Sem a flag, o legado segue submetendo na janela de graça."""
    monkeypatch.delenv("DRAW_INK_SKIP_STALE", raising=False)
    loop, _det, ink = _make_loop(_seen_then_lost())
    submits_grew = False
    for _ in range(60):
        loop.step(_frame(), loop.frame_number * 100.0)
        if loop.tracker._frames_since_seen > 5 and loop.tracker.corners is not None:
            before = len(ink.submitted)
            loop.step(_frame(), loop.frame_number * 100.0)
            if len(ink.submitted) > before:
                submits_grew = True
            break
    assert submits_grew


def test_fresh_passes_unaffected_with_flag(monkeypatch):
    """Folha confirmada: _frames_since_seen = 0 — a flag não toca
    em nenhuma passada saudável."""
    monkeypatch.setenv("DRAW_INK_SKIP_STALE", "1")
    loop, _det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 40)
    assert len(ink.submitted) > 0
