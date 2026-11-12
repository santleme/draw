"""Cadencia de deteccao com folha assentada (frente 93).

DRAW_SHEET_DETECT_SETTLED_EVERY=K (default desligado): com a folha
confirmada e deteccoes cruas estaveis (drift <= banda morta por 2
seguidas), so 1 em cada K oportunidades agendadas roda a deteccao
neural; nas oportunidades saltadas, a tinta passa reusando a
geometria mediana ja confirmada, na mesma cadencia de passadas de
hoje. Gatilho de mudanca de cena cancela o salto; folha em
movimento, missed do rastreador e caminho legado nunca saltam.
Sem a flag, contadores zerados e comportamento identico.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import loop as loop_module  # noqa: E402

QUAD = np.array(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
    dtype=np.float32,
)
QUAD_MOVED = QUAD + np.array([60.0, 0.0], dtype=np.float32)


class _SyncDetectionWorker:
    """Resultado da submissao colhido no step seguinte (atraso de
    1 quadro do replay). `script(i)` por indice de submissao."""

    def __init__(self, script):
        self._script = script
        self._result = None
        self._seq = 0
        self.submits = 0
        self.error = None

    def submit(self, frame, captured_at, frame_index=None):
        corners, verdict = self._script(self.submits)
        self.submits += 1
        self._seq += 1
        self._result = (self._seq, corners, verdict, captured_at, 1.0)

    def poll(self, after_seq):
        if self._result is None or self._result[0] <= after_seq:
            return None
        return self._result

    def take_traces(self, upto_result_seq=None):
        return []


class _FakeInkWorker:
    def __init__(self):
        self.error = None
        self.generation = 0
        self.submitted = []  # (generation, corners copy)

    def new_generation(self, preserve_identity=False):
        self.generation += 1
        return self.generation

    def submit(self, warped, generation, corners=None, visible=None):
        self.submitted.append(
            (generation, None if corners is None else corners.copy())
        )

    def poll_display(self, generation, version):
        return None

    def take_turn_event(self, generation):
        return None

    def take_identity_split(self, generation):
        return None

    def set_hand_presence(self, present, moved=False):
        pass

    def arm_identity_check(self, reason):
        pass


def _make_loop(script):
    det = _SyncDetectionWorker(script)
    ink = _FakeInkWorker()
    loop = loop_module.SheetInkLoop(
        detection_worker=det,
        ink_worker=ink,
        clock=lambda: 0.0,
        log=lambda *_a: None,
    )
    # O relogio dos testes e congelado; os carimbos de captura
    # (frame_capture_ms) e que movem os relogios do rastreador.
    return loop, det, ink


def _drive(loop, steps, frame=None):
    if frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(steps):
        loop.step(frame, loop.frame_number * 100.0)


def test_flag_parsing(monkeypatch):
    monkeypatch.delenv("DRAW_SHEET_DETECT_SETTLED_EVERY", raising=False)
    assert loop_module.detect_settled_every() == 0
    monkeypatch.setenv("DRAW_SHEET_DETECT_SETTLED_EVERY", "1")
    assert loop_module.detect_settled_every() == 0
    monkeypatch.setenv("DRAW_SHEET_DETECT_SETTLED_EVERY", "3")
    assert loop_module.detect_settled_every() == 3
    monkeypatch.setenv("DRAW_SHEET_DETECT_SETTLED_EVERY", "lixo")
    assert loop_module.detect_settled_every() == 0


def test_flag_off_no_skips_no_reuse(monkeypatch):
    monkeypatch.delenv("DRAW_SHEET_DETECT_SETTLED_EVERY", raising=False)
    loop, det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper"))
    _drive(loop, 60)
    assert loop.tracker.corners is not None
    assert loop.detect_settled_skipped == 0
    assert loop.ink_reuse_passes == 0
    # Toda passada de tinta veio de quadro com deteccao.
    assert len(ink.submitted) == det.submits or len(ink.submitted) == det.submits - 1


def test_flag_on_stable_sheet_skips_and_reuses(monkeypatch):
    monkeypatch.setenv("DRAW_SHEET_DETECT_SETTLED_EVERY", "3")
    loop, det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper"))
    _drive(loop, 90)
    assert loop.tracker.corners is not None
    assert loop.detect_settled_skipped >= 2
    assert loop.ink_reuse_passes == loop.detect_settled_skipped
    # Deteccoes caem para ~1/3 das oportunidades apos estabilizar,
    # mas a cadencia de passadas de tinta se mantem.
    assert det.submits < 12
    assert len(ink.submitted) >= 15
    # O quad do rastreador e o mesmo do braco sem flag.
    assert np.array_equal(loop.tracker.corners, QUAD)
    # Passadas de reuso usam a mesma geometria mediana das de
    # deteccao (historico intocado pelo reuso).
    quads = [c for _g, c in ink.submitted if c is not None]
    assert all(np.array_equal(c, QUAD) for c in quads)


def test_flag_on_moving_sheet_never_skips(monkeypatch):
    monkeypatch.setenv("DRAW_SHEET_DETECT_SETTLED_EVERY", "3")
    state = {"quad": QUAD}

    def script(_i):
        # Cada deteccao anda 60 px: estabilidade nunca se sustenta.
        state["quad"] = state["quad"] + np.array([60.0, 0.0], dtype=np.float32)
        return state["quad"].copy(), "paper"

    loop, det, ink = _make_loop(script)
    _drive(loop, 60)
    assert loop.detect_settled_skipped == 0
    assert loop.ink_reuse_passes == 0


def test_flag_on_scene_change_trigger_fires(monkeypatch):
    monkeypatch.setenv("DRAW_SHEET_DETECT_SETTLED_EVERY", "4")
    loop, det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper"))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _drive(loop, 60, frame)
    assert loop.detect_settled_skipped >= 1
    submits_before = det.submits
    # Cena muda de verdade (inversao completa): o proximo salto
    # tem que virar deteccao pelo gatilho, sem esperar o orcamento.
    changed = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _drive(loop, 12, changed)
    assert loop.detect_settled_triggers >= 1
    assert det.submits > submits_before


def test_flag_on_legacy_path_is_inert(monkeypatch):
    monkeypatch.setenv("DRAW_SHEET_DETECT_SETTLED_EVERY", "3")
    det = _SyncDetectionWorker(lambda _i: (QUAD.copy(), "paper"))
    loop = loop_module.SheetInkLoop(
        detection_worker=det,
        ink_source=lambda warped: np.zeros(warped.shape[:2], dtype=np.uint8),
        clock=lambda: 0.0,
        log=lambda *_a: None,
    )
    _drive(loop, 60)
    assert loop.detect_settled_skipped == 0
    assert loop.ink_reuse_passes == 0


def test_flag_on_no_sheet_never_skips(monkeypatch):
    monkeypatch.setenv("DRAW_SHEET_DETECT_SETTLED_EVERY", "3")
    loop, det, ink = _make_loop(lambda _i: (None, None))
    _drive(loop, 40)
    assert loop.tracker.corners is None
    assert loop.detect_settled_skipped == 0
    assert ink.submitted == []
