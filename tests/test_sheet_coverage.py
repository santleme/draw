"""Cobertura de folha no voto de tinta (DRAW_SHEET_COVERAGE).

A inundação medida no cenário E3 (bancada 2026-11-12): depois do
movimento da folha, o quad usado pelo warp admitia uma faixa de mesa
na borda e a UNet de tinta lia a textura como traço — ~166k px de
confirmado fantasma em uma sessão. Com a flag ligada, a detecção
publica a máscara de segmentação junto do resultado; o laço a
retifica com a mesma homografia do warp de tinta e, fora dela, a
trabalhadora só deixa votar sobre a tinta que a âncora já tem —
re-confirmar e retrair o que existe, nunca tinta nova na mesa.
Sem a flag, nada muda.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import loop as loop_module  # noqa: E402
from draw import sheet_track  # noqa: E402

QUAD = np.array(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
    dtype=np.float32,
)


def _coverage_mask(shape=(720, 1280)):
    """Folha falsa: só a metade esquerda do quad é papel; a direita
    do quad (x >= 300) já cai sobre mesa — é o vazamento que a
    cobertura tem de vedar dentro da região retificada."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[:, :300] = 255
    return mask


class _SyncDetectionWorker:
    """Publica o resultado de 7 elementos que a worker real monta
    com DRAW_SHEET_COVERAGE ligada (posição 5 = portão, 6 = extras)."""

    def __init__(self, script):
        self._script = script
        self._result = None
        self._seq = 0
        self.error = None

    def submit(self, frame, captured_at, frame_index=None):
        corners, verdict, extras = self._script(self._seq)
        self._seq += 1
        result = (self._seq, corners, verdict, captured_at, 1.0)
        if extras is not None:
            result = result + (None, extras)
        self._result = result

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
        self.submitted = []  # (generation, corners, visible, coverage)

    def new_generation(self, preserve_identity=False):
        self.generation += 1
        return self.generation

    def submit(self, warped, generation, corners=None, visible=None, coverage=None):
        self.submitted.append(
            (
                generation,
                None if corners is None else corners.copy(),
                None if visible is None else visible.copy(),
                None if coverage is None else coverage.copy(),
            )
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
    return loop, det, ink


def _drive(loop, steps, frame=None):
    if frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(steps):
        loop.step(frame, loop.frame_number * 100.0)


def test_flag_parsing(monkeypatch):
    monkeypatch.delenv("DRAW_SHEET_COVERAGE", raising=False)
    assert not sheet_track.sheet_coverage_enabled()
    monkeypatch.setenv("DRAW_SHEET_COVERAGE", "1")
    assert sheet_track.sheet_coverage_enabled()
    monkeypatch.setenv("DRAW_SHEET_COVERAGE", "lixo")
    assert not sheet_track.sheet_coverage_enabled()


def test_no_extras_no_coverage_gate(monkeypatch):
    """Sem extras no resultado, o submit segue sem cobertura."""
    monkeypatch.delenv("DRAW_SHEET_COVERAGE", raising=False)
    loop, _det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    assert ink.submitted, "a passada de tinta precisa rodar"
    assert all(
        visible is None and coverage is None
        for _g, _c, visible, coverage in ink.submitted
    )


def test_coverage_mask_gates_votes_off_sheet(monkeypatch):
    """A máscara da detecção veda o voto fora da folha retificada."""
    monkeypatch.setenv("DRAW_SHEET_COVERAGE", "1")
    mask = _coverage_mask()
    loop, _det, ink = _make_loop(
        lambda _i: (QUAD.copy(), "paper", {"coverage_mask": mask})
    )
    _drive(loop, 40)
    assert loop.tracker.corners is not None
    gated = [cov for _g, _c, _v, cov in ink.submitted if cov is not None]
    assert gated, "a cobertura precisa chegar ao submit"
    # Sem mão sintética o visible segue None — a cobertura viaja
    # em canal próprio e não mistura semântica de oclusão.
    assert all(v is None for _g, _c, v, _cov in ink.submitted)
    # A metade direita do quad cai sobre "mesa": nunca pode votar.
    # O warp da cena pelo quad retifica a fronteira da máscara
    # (x=300 em [100,500]) para ~metade da página; a margem além
    # dela tem de estar toda abstida (a erosão desconta a fronteira
    # medida, então a faixa profunda é estritamente vazia).
    page_h, page_w = gated[-1].shape
    assert not gated[-1][:, int(page_w * 0.60) :].any()
    # E a região de folha segue aberta para o voto.
    assert gated[-1][:, : int(page_w * 0.30)].all()


def test_result_without_mask_clears_coverage(monkeypatch):
    """Detecção sem máscara (clássico) limpa a cobertura anterior."""
    monkeypatch.setenv("DRAW_SHEET_COVERAGE", "1")
    state = {"with_mask": True}
    mask = _coverage_mask()

    def script(_i):
        if state["with_mask"]:
            return QUAD.copy(), "paper", {"coverage_mask": mask}
        # Com a flag ligada, a detecção real publica extras mesmo
        # sem máscara (clássico): o dict vazio é o sinal de limpeza.
        return QUAD.copy(), "paper", {}

    loop, _det, ink = _make_loop(script)
    _drive(loop, 30)
    assert loop._coverage_mask is not None
    state["with_mask"] = False
    _drive(loop, 10)
    assert loop._coverage_mask is None
    # Depois da limpeza, o submit volta a ir sem cobertura.
    last = ink.submitted[-1]
    assert last[3] is None


def _drive_clock(loop, steps, clock_state, dt=0.05):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(steps):
        loop.step(frame, loop.frame_number * 100.0)
        clock_state["t"] += dt


def test_follow_move_flag_parsing(monkeypatch):
    monkeypatch.delenv("DRAW_INK_FOLLOW_MOVE", raising=False)
    assert not loop_module.ink_follow_move_enabled()
    monkeypatch.setenv("DRAW_INK_FOLLOW_MOVE", "1")
    assert loop_module.ink_follow_move_enabled()
    monkeypatch.setenv("DRAW_INK_FOLLOW_MOVE", "lixo")
    assert not loop_module.ink_follow_move_enabled()


def _move_scenario(monkeypatch, flag):
    """Folha desliza sem mão: sessão aberta em QUAD, mão sai
    (freeze) e o quad confirmado salta +120 px em x."""
    if flag:
        monkeypatch.setenv("DRAW_INK_FOLLOW_MOVE", "1")
    else:
        monkeypatch.delenv("DRAW_INK_FOLLOW_MOVE", raising=False)
    clock_state = {"t": 0.0}
    moved = QUAD + np.array([120.0, 0.0], dtype=np.float32)
    state = {"move": False}

    def script(_i):
        return (moved if state["move"] else QUAD).copy(), "paper", None

    loop, _det, ink = _make_loop(script)
    loop._clock = lambda: clock_state["t"]
    _drive_clock(loop, 40, clock_state)
    assert loop.tracker.corners is not None
    # Mão sai da cena: o histórico congela na pose antiga.
    loop.clean_hand_present = False
    state["move"] = True
    _drive_clock(loop, 60, clock_state)
    return loop, ink, moved


def test_frozen_geometry_tracks_real_move(monkeypatch):
    """Com a flag, o quad congelado destrava num deslocamento real
    e a tinta volta a ser retificada na pose nova."""
    _loop, ink, moved = _move_scenario(monkeypatch, flag=True)
    last_corners = ink.submitted[-1][1]
    assert last_corners is not None
    assert np.max(np.abs(last_corners - moved)) < 5.0


def test_frozen_geometry_stays_without_flag(monkeypatch):
    """Sem a flag, o congelamento de sempre segue: a mediana fica
    na pose antiga mesmo com o rastreador já na posição nova."""
    loop, ink, moved = _move_scenario(monkeypatch, flag=False)
    assert np.max(np.linalg.norm(loop.tracker.corners - moved, axis=1)) < 40.0, (
        "o rastreador em si já acompanhou a folha"
    )
    last_corners = ink.submitted[-1][1]
    assert last_corners is not None
    assert np.max(np.abs(last_corners - QUAD)) < 5.0
