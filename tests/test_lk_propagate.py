"""Propagacao LK do quad (frente 105).

DRAW_SHEET_LK_PROPAGATE=1 (default desligado): entre updates de
deteccao, um propagador resemeado (grade 6x6 replantada a cada
passo, homografia composta por RANSAC) sustenta o quad e alimenta
a passada de tinta dos quadros sem deteccao, ate o teto de 4
passos; passo que morre arma redeteccao imediata; update de
deteccao com salto > 50 px contra a ultima saida e segurado pelo
portao anti-flip (frente 97) e so confirma com a deteccao seguinte
a <= 25 px do pendente. Sem a flag, contadores zerados, nenhuma
passada propagada e comportamento identico.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import lk_propagate as lk_module  # noqa: E402
from draw import loop as loop_module  # noqa: E402

QUAD = np.array(
    [[100.0, 100.0], [700.0, 100.0], [700.0, 500.0], [100.0, 500.0]],
    dtype=np.float32,
)


def _textured_frame(dx=0, dy=0, seed=7):
    """Folha texturizada (ruido) sobre fundo preto, em (100+dx, 100+dy)."""
    rng = np.random.default_rng(seed)
    tex = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[100 + dy : 500 + dy, 100 + dx : 700 + dx] = tex
    return frame


# ------------------------------------------------------------------
# Modulo de propagacao (unidade)
# ------------------------------------------------------------------
def test_flag_parsing(monkeypatch):
    monkeypatch.delenv("DRAW_SHEET_LK_PROPAGATE", raising=False)
    assert lk_module.lk_propagate_enabled() is False
    monkeypatch.setenv("DRAW_SHEET_LK_PROPAGATE", "1")
    assert lk_module.lk_propagate_enabled() is True
    monkeypatch.setenv("DRAW_SHEET_LK_PROPAGATE", "0")
    assert lk_module.lk_propagate_enabled() is False
    monkeypatch.setenv("DRAW_SHEET_LK_PROPAGATE", "lixo")
    assert lk_module.lk_propagate_enabled() is False


def _gray(frame):
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def test_propagator_tracks_translation():
    prop = lk_module.ReseededQuadPropagator()
    prop.anchor(QUAD)
    prev = _gray(_textured_frame(0))
    for step, dx in enumerate((3, 6, 9, 12), start=1):
        cur = _gray(_textured_frame(dx))
        out = prop.step(prev, cur)
        prev = cur
        assert out is not None
        assert prop.alive
        assert prop.steps_since_anchor == step
        # O canto acompanha a translacao com erro sub-pixel a ~1 px.
        assert abs(float(out[0][0]) - (100.0 + dx)) < 1.5
        assert abs(float(out[0][1]) - 100.0) < 1.5


def test_propagator_dies_on_scene_cut():
    prop = lk_module.ReseededQuadPropagator()
    prop.anchor(QUAD)
    prev = _gray(_textured_frame(0))
    blank = np.zeros((720, 1280), dtype=np.uint8)
    assert prop.step(prev, blank) is None
    assert not prop.alive
    assert prop.deaths == 1
    # Morta, nao ressuscita sozinha: so anchor() revive.
    assert prop.step(blank, prev) is None


def test_gate_holds_flip_and_confirms_sustained():
    gate = lk_module.LkUpdateGate()
    base = QUAD.astype(np.float64)
    flip = base + np.array([200.0, 0.0])
    assert gate.feed(base + 3.0, base, True) == ("accept", True)
    assert gate.feed(flip, base, True) == ("hold", False)
    assert gate.feed(flip + 5.0, base, True) == ("confirm", True)
    assert gate.held == 1 and gate.confirmed == 1


def test_gate_discards_isolated_flip():
    gate = lk_module.LkUpdateGate()
    base = QUAD.astype(np.float64)
    flip = base + np.array([200.0, 0.0])
    assert gate.feed(flip, base, True) == ("hold", False)
    # A deteccao seguinte voltou para perto da saida: pendente cai.
    assert gate.feed(base + 2.0, base, True) == ("discard_return", True)
    assert gate.discarded == 1


def test_gate_open_when_propagation_dead():
    gate = lk_module.LkUpdateGate()
    base = QUAD.astype(np.float64)
    far = base + np.array([300.0, 0.0])
    event, accepted = gate.feed(far, base, False)
    assert accepted is True
    assert gate.pending is None


# ------------------------------------------------------------------
# Laco real com trabalhadoras falsas (padrao das frentes 93/101)
# ------------------------------------------------------------------
class _SyncDetectionWorker:
    """Resultado da submissao colhido no step seguinte.

    script(frame_index) -> (cantos, veredito)."""

    def __init__(self, script):
        self._script = script
        self._result = None
        self._seq = 0
        self.submits = 0
        self.error = None

    def submit(self, frame, captured_at, frame_index=None):
        corners, verdict = self._script(frame_index)
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
        self.submitted = []

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
    return loop, det, ink


def _moving_script(frame_index):
    dx = 2 * int(frame_index or 0)
    return (QUAD + np.array([dx, 0], dtype=np.float32)).copy(), "paper"


def test_loop_flag_off_no_propagation(monkeypatch):
    monkeypatch.delenv("DRAW_SHEET_LK_PROPAGATE", raising=False)
    loop, det, ink = _make_loop(_moving_script)
    for i in range(40):
        loop.step(_textured_frame(2 * i), i * 100.0)
    assert loop.lk_propagated_passes == 0
    assert loop.lk_deaths == 0
    assert loop.lk_forced_updates == 0
    assert loop._lk_prev_gray is None
    # Passadas de tinta so nos quadros com resultado de deteccao.
    assert len(ink.submitted) <= det.submits


def test_loop_flag_on_propagates_between_detections(monkeypatch):
    monkeypatch.setenv("DRAW_SHEET_LK_PROPAGATE", "1")
    loop, det, ink = _make_loop(_moving_script)
    for i in range(40):
        loop.step(_textured_frame(2 * i), i * 100.0)
    assert loop.tracker.corners is not None
    assert loop.lk_propagated_passes >= 8
    assert loop.lk_deaths == 0
    # Ha mais passadas de tinta que deteccoes: a propagacao alimenta
    # os quadros intermediarios.
    assert len(ink.submitted) > det.submits
    # E o quad propagado segue o movimento (x do canto cresce alem
    # do quad da ultima deteccao consumida).
    quads = [c for _g, c in ink.submitted if c is not None]
    xs = [float(c[0][0]) for c in quads]
    assert max(xs) - min(xs) > 20.0


def test_loop_flag_on_death_forces_redetection(monkeypatch):
    monkeypatch.setenv("DRAW_SHEET_LK_PROPAGATE", "1")
    loop, det, ink = _make_loop(lambda _i: (QUAD.copy(), "paper"))
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(30):
        loop.step(_textured_frame(0), i * 100.0)
    submits_before_cut = det.submits
    for i in range(30, 60):
        loop.step(blank, i * 100.0)
    assert loop.lk_deaths >= 1
    assert loop.lk_redetect_requests >= 1
    assert loop.lk_forced_updates >= 1
    assert det.submits > submits_before_cut


def test_loop_gate_holds_detector_mode_jump(monkeypatch):
    monkeypatch.setenv("DRAW_SHEET_LK_PROPAGATE", "1")
    jumped = QUAD + np.array([200.0, 0.0], dtype=np.float32)
    state = {"quad": QUAD}

    def script(frame_index):
        if int(frame_index or 0) >= 60:
            state["quad"] = jumped
        return state["quad"].copy(), "paper"

    loop, det, ink = _make_loop(script)
    frame = _textured_frame(0)  # pixels NUNCA se mexem: e um flip
    for i in range(160):
        loop.step(frame, i * 100.0)
    # O salto do rastreador foi segurado pelo portao pelo menos uma
    # vez; sustentado, acabou confirmado e a ancora migrou.
    assert loop._lk_gate.held >= 1
    assert loop._lk_gate.confirmed >= 1
    assert loop._lk_prop.anchor_quad is not None
    assert abs(float(loop._lk_prop.anchor_quad[0][0]) - 300.0) < 1.0
