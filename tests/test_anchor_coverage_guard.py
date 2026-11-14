"""Guarda de cobertura da âncora (DRAW_ANCHOR_COVERAGE_GUARD).

Medida na mesa real (2026-11-14): a folha sai de cena e a detecção
continua emitindo quads/veredito "paper" sobre a textura da mesa —
cada resultado renova o seen do rastreador, a âncora nunca cai e a
região vazia segue sendo projetada como tinta. A guarda mede a
fração do quad rastreado coberta pela máscara binária de folha
(≈0,99 com folha, ≈0,03 sem) e, abaixo do piso por algumas rodadas
seguidas, veta todo refresh; sem âncora, a mesma guarda vira portão
de aquisição exigindo folha de verdade no quad (fração alta + luma
de papel), o que barra o falso quad da mesa (luma ~110).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import loop as loop_module  # noqa: E402
from draw.sheet_track import (  # noqa: E402
    _SHEET_LOST_MIN_FRAMES,
    anchor_coverage_guard_enabled,
)

QUAD = np.array(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
    dtype=np.float32,
)


# ---------- funções puras ----------


def test_quad_coverage_fraction_full_empty_partial():
    mask = np.zeros((720, 1280), np.uint8)
    mask[100:400, 100:500] = 255
    assert loop_module._quad_coverage_fraction(mask, QUAD) > 0.95
    mask[:] = 0
    assert loop_module._quad_coverage_fraction(mask, QUAD) == 0.0
    mask[100:400, 100:200] = 255
    frac = loop_module._quad_coverage_fraction(mask, QUAD)
    assert 0.15 < frac < 0.35
    # Quad totalmente fora do frame: sem evidência, não há veto.
    off = QUAD + np.array([3000.0, 3000.0], np.float32)
    assert loop_module._quad_coverage_fraction(mask, off) == 1.0


def test_quad_luma_mean_bright_dark():
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert loop_module._quad_luma_mean(frame, QUAD) < 10
    frame[:] = 240
    assert loop_module._quad_luma_mean(frame, QUAD) > 230


def test_guard_flag_default_off(monkeypatch):
    monkeypatch.delenv("DRAW_ANCHOR_COVERAGE_GUARD", raising=False)
    assert not anchor_coverage_guard_enabled()
    monkeypatch.setenv("DRAW_ANCHOR_COVERAGE_GUARD", "1")
    assert anchor_coverage_guard_enabled()
    monkeypatch.setenv("DRAW_ANCHOR_COVERAGE_GUARD", "0")
    assert not anchor_coverage_guard_enabled()


# ---------- laço real ----------


def _sheet_mask_for(quad, shape=(720, 1280)):
    mask = np.zeros(shape, np.uint8)
    import cv2

    cv2.fillPoly(mask, [quad.astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


class _SyncDetectionWorker:
    """Trabalhadora síncrona com extras (7º elemento): o script
    devolve (cantos, veredito, extras) por submissão."""

    def __init__(self, script):
        self._script = script
        self._result = None
        self._seq = 0
        self._submits = 0
        self.error = None

    def submit(self, frame, captured_at, frame_index=None):
        corners, verdict, extras = self._script(self._submits)
        self._submits += 1
        self._seq += 1
        self._result = (self._seq, corners, verdict, captured_at, 1.0, None, extras)

    def poll(self, after_seq):
        if self._result is None or self._result[0] <= after_seq:
            return None
        return self._result

    def take_traces(self, upto_result_seq=None):
        return []


def _make_loop(script):
    return loop_module.SheetInkLoop(
        detection_worker=_SyncDetectionWorker(script),
        ink_source=lambda warped: np.zeros(warped.shape[:2], dtype=np.uint8),
        clock=lambda: 0.0,
        log=lambda *_a: None,
    )


def _drive(loop, steps, frame=None):
    if frame is None:
        frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
    for _ in range(steps):
        loop.step(frame, loop.frame_number * 100.0)


def test_healthy_anchor_never_vetoed(monkeypatch):
    monkeypatch.setenv("DRAW_ANCHOR_COVERAGE_GUARD", "1")
    mask = _sheet_mask_for(QUAD)

    def script(_i):
        return QUAD.copy(), "paper", {"sheet_mask": mask, "source": "neural"}

    loop = _make_loop(script)
    _drive(loop, 12)
    assert loop.tracker.corners is not None
    assert loop.anchor_guard_vetoes == 0


def test_empty_anchor_drops_and_stays_down(monkeypatch):
    monkeypatch.setenv("DRAW_ANCHOR_COVERAGE_GUARD", "1")
    good = _sheet_mask_for(QUAD)
    # A folha sai: a máscara passa a marcar outra região (mesa) e o
    # detector emite um quad escuro nela (luma da mesa, ~110 < 140).
    desk_quad = np.array(
        [[600.0, 500.0], [900.0, 500.0], [900.0, 680.0], [600.0, 680.0]],
        dtype=np.float32,
    )
    bad = _sheet_mask_for(desk_quad)
    calls = {"i": 0}

    def script(i):
        if i < 2:
            return QUAD.copy(), "paper", {"sheet_mask": good, "source": "neural"}
        calls["i"] += 1
        return desk_quad.copy(), "paper", {"sheet_mask": bad, "source": "neural"}

    loop = _make_loop(script)
    dark = np.full((720, 1280, 3), 110, dtype=np.uint8)
    _drive(loop, 4)
    assert loop.tracker.corners is not None
    # A âncora velha não cobre mais folha: risca sobe e o veto impede
    # qualquer renovação — o rastreador cai pelo caminho natural.
    _drive(loop, _SHEET_LOST_MIN_FRAMES + 40, frame=dark)
    assert loop.tracker.corners is None
    assert loop.anchor_guard_vetoes > 0
    assert calls["i"] > 0


def test_dark_quad_cannot_acquire_bright_can(monkeypatch):
    monkeypatch.setenv("DRAW_ANCHOR_COVERAGE_GUARD", "1")
    dark_quad = np.array(
        [[100.0, 500.0], [400.0, 500.0], [400.0, 680.0], [100.0, 680.0]],
        dtype=np.float32,
    )
    mask = _sheet_mask_for(dark_quad)
    calls = {"n": 0}

    def script(i):
        calls["n"] = i
        if i < 4:
            return dark_quad.copy(), "paper", {
                "sheet_mask": mask,
                "source": "neural",
            }
        return QUAD.copy(), "paper", {
            "sheet_mask": _sheet_mask_for(QUAD),
            "source": "neural",
        }

    loop = _make_loop(script)
    frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
    frame[500:680, 100:400] = 110  # região escura tipo mesa
    _drive(loop, 4, frame=frame)
    assert loop.tracker.corners is None
    assert loop.anchor_guard_vetoes >= 1
    _drive(loop, 4, frame=frame)
    # O quad claro e cheio de folha adquire normalmente.
    assert loop.tracker.corners is not None


def test_bad_quad_abstains_ink_votes(monkeypatch):
    """Quad rastreado medido não-folha: a passada se abstém
    inteira (visible=False) já na primeira medida ruim — não
    espera o streak da queda da âncora. Abstenção é reversível;
    voto de mesa confirmado, não."""
    monkeypatch.setenv("DRAW_ANCHOR_COVERAGE_GUARD", "1")
    good = _sheet_mask_for(QUAD)
    empty = np.zeros_like(good)

    def script(i):
        if i < 2:
            return QUAD.copy(), "paper", {"sheet_mask": good, "source": "neural"}
        # A folha some: detector não acha nada e a máscara vem
        # vazia — o quad rastreado continua ancorado na graça.
        return None, None, {"sheet_mask": empty, "source": "neural"}

    loop = _make_loop(script)
    seen_visible = []
    real_update = loop.ink_accumulator.update

    def spy(observed, visible=None, **kw):
        seen_visible.append(
            None if visible is None else bool(np.any(visible))
        )
        return real_update(observed, visible, **kw)

    loop.ink_accumulator.update = spy
    _drive(loop, 3)
    assert loop.tracker.corners is not None
    assert not loop._anchor_quad_bad
    # A cadência assentada pula detecções; dirigir bastante até a
    # próxima rodada medir a máscara vazia contra o quad ancorado.
    _drive(loop, 10)
    assert loop._anchor_quad_bad
    # As últimas passadas (quad ruim) chegaram ao acumulador sem
    # nenhum pixel visível — nenhum voto de mesa.
    assert seen_visible
    assert seen_visible[-1] is False
    assert not any(seen_visible[-2:])

    # A folha volta na mesma pose: o veredito limpa e o voto
    # flui de novo — a abstinência não virou trava permanente.
    def script_back(i):
        return QUAD.copy(), "paper", {"sheet_mask": good, "source": "neural"}

    loop.detection_worker._script = script_back
    _drive(loop, 4)
    assert not loop._anchor_quad_bad
    assert seen_visible[-1] is not False
