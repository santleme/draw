"""Portão de primeira confirmação anti-fantasma (frentes 85/88).

Cobre: a função pura do portão (logit E arestas, nos limiares
medidos), a receita de edge_in sobre a máscara, a extração do pacote
do portão pela trabalhadora de detecção (tupla de 5 sem rastro, de 6
com rastro) e a integração no laço real (`SheetInkLoop`): com a flag
ligada, só a PRIMEIRA confirmação é julgada — detecção neural fraca
é rebaixada na ronda, folha confirmada e reserva clássica passam
intocadas, e a reentrada após a queda volta a passar pelo portão.
Com a flag desligada, o comportamento é o de sempre.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import app as app_module  # noqa: E402
from draw import loop as loop_module  # noqa: E402
from draw import neural as neural_module  # noqa: E402
from draw import paper as paper_module  # noqa: E402
from draw.sheet_track import _SheetDetectionWorker, _gate_info_from_trace  # noqa: E402

QUAD = np.array(
    [[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]],
    dtype=np.float32,
)
WEAK = {"source": "neural", "presence_logit": 0.32, "edge_in": 0.001}
STRONG = {"source": "neural", "presence_logit": 5.0, "edge_in": 0.05}


# ---------- função pura ----------


def test_gate_passes_requires_both_legs_at_threshold():
    assert neural_module.first_confirm_passes(2.0, 0.005)
    assert neural_module.first_confirm_passes(9.8, 0.08)
    assert not neural_module.first_confirm_passes(1.99, 0.08)
    assert not neural_module.first_confirm_passes(9.8, 0.0049)
    # O fantasma da sessão: logit marginal E interior liso.
    assert not neural_module.first_confirm_passes(0.51, 0.004)
    assert not neural_module.first_confirm_passes(None, 0.08)
    assert not neural_module.first_confirm_passes(9.8, None)


def test_gate_flag_default_off(monkeypatch):
    monkeypatch.delenv("DRAW_FIRST_CONFIRM_GATE", raising=False)
    assert not neural_module.first_confirm_gate_enabled()
    monkeypatch.setenv("DRAW_FIRST_CONFIRM_GATE", "1")
    assert neural_module.first_confirm_gate_enabled()
    monkeypatch.setenv("DRAW_FIRST_CONFIRM_GATE", "0")
    assert not neural_module.first_confirm_gate_enabled()


def test_edge_fraction_smooth_blob_vs_textured_paper():
    frame = np.full((720, 1280, 3), 140, dtype=np.uint8)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[100:300, 400:1000] = 255
    # Mancha lisa: nenhuma aresta dentro da máscara.
    assert neural_module.mask_edge_fraction(frame, mask) == 0.0
    # Textura (xadrez fino) dentro da mesma máscara: arestas de sobra.
    yy, xx = np.mgrid[0:720, 0:1280]
    frame[((xx // 8) + (yy // 8)) % 2 == 0] = 230
    assert neural_module.mask_edge_fraction(frame, mask) >= 0.005
    # Máscara vazia: 0,0, sem exceção.
    assert neural_module.mask_edge_fraction(frame, np.zeros_like(mask)) == 0.0


# ---------- pacote do portão: rastro -> trabalhadora ----------


def test_gate_info_from_trace():
    assert _gate_info_from_trace(None) is None
    assert _gate_info_from_trace({}) is None
    trace = {"source": "neural", "presence_logit": 0.32, "edge_in": 0.001}
    assert _gate_info_from_trace(trace) == trace


def test_worker_result_arity_with_and_without_trace():
    def detect_plain(_frame):
        return QUAD.copy(), "paper"

    worker = _SheetDetectionWorker(detect_plain).start()
    try:
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.0, 0)
        result = _poll_until(worker, 0)
        assert len(result) == 5
    finally:
        worker.stop()

    def detect_traced(_frame):
        return QUAD.copy(), "paper", dict(WEAK)

    worker = _SheetDetectionWorker(detect_traced).start()
    try:
        worker.submit(np.zeros((8, 8, 3), np.uint8), 1.0, 0)
        result = _poll_until(worker, 0)
        assert len(result) == 6
        assert result[5] == WEAK
    finally:
        worker.stop()


def _poll_until(worker, after_seq):
    import time

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = worker.poll(after_seq)
        if result is not None:
            return result
        time.sleep(0.002)
    raise AssertionError("trabalhadora não publicou resultado")


# ---------- combinada com rastro: edge_in só com a flag ----------


class _FakeNeural:
    def __init__(self):
        self.last_presence_logit = 5.0
        self.called_with_mask = False

    def __call__(self, frame, *, return_mask=True):
        self.called_with_mask = return_mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[100:300, 400:1000] = 255
        return QUAD.copy(), "paper", mask if return_mask else None

    def detect_sheet_with_confidence(self, frame):
        corners, verdict, _ = self(frame, return_mask=False)
        return corners, verdict


def test_combined_traced_computes_edge_in_only_when_gate_on(monkeypatch):
    frame = np.full((720, 1280, 3), 140, dtype=np.uint8)
    yy, xx = np.mgrid[0:720, 0:1280]
    frame[((xx // 8) + (yy // 8)) % 2 == 0] = 230
    monkeypatch.setattr(
        paper_module, "detect_sheet_with_confidence", lambda _f: (None, None)
    )
    det = _FakeNeural()
    monkeypatch.delenv("DRAW_FIRST_CONFIRM_GATE", raising=False)
    _c, _v, trace = app_module._detect_sheet_combined_traced(frame, det)
    # Sem a flag, o rastro nem ganha o campo: forma idêntica à atual.
    assert "edge_in" not in trace
    assert det.called_with_mask is False
    monkeypatch.setenv("DRAW_FIRST_CONFIRM_GATE", "1")
    _c, _v, trace = app_module._detect_sheet_combined_traced(frame, det)
    assert det.called_with_mask is True
    assert trace["edge_in"] is not None and trace["edge_in"] >= 0.005
    assert trace["source"] == "neural"


# ---------- integração no laço ----------


class _SyncDetectionWorker:
    """Protocolo da trabalhadora, síncrono: o resultado da submissão
    é colhido no step seguinte (o atraso de 1 quadro do replay P1).
    `script` é chamado por índice de submissão e devolve
    (cantos, veredito, gate_info|None)."""

    def __init__(self, script):
        self._script = script
        self._result = None
        self._seq = 0
        self._submits = 0
        self.error = None

    def submit(self, frame, captured_at, frame_index=None):
        corners, verdict, gate = self._script(self._submits)
        self._submits += 1
        self._seq += 1
        result = (self._seq, corners, verdict, captured_at, 1.0)
        if gate is not None:
            result = result + (gate,)
        self._result = result

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


def _drive(loop, steps):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(steps):
        # 100 ms por quadro (10 fps), monotônico entre fases: o
        # carimbo sai do contador de quadros do próprio laço.
        loop.step(frame, loop.frame_number * 100.0)


def test_gate_off_weak_first_detection_confirms_as_always(monkeypatch):
    monkeypatch.delenv("DRAW_FIRST_CONFIRM_GATE", raising=False)
    loop = _make_loop(lambda _i: (QUAD.copy(), "paper", dict(WEAK)))
    _drive(loop, 3)
    assert loop.tracker.corners is not None
    assert loop.first_confirm_gate_blocked == 0


def test_gate_on_blocks_weak_then_strong_confirms(monkeypatch):
    monkeypatch.setenv("DRAW_FIRST_CONFIRM_GATE", "1")
    answers = [(QUAD.copy(), "paper", dict(WEAK)), (QUAD.copy(), "paper", dict(STRONG))]
    loop = _make_loop(lambda i: answers[min(i, len(answers) - 1)])
    _drive(loop, 2)  # 1º resultado colhido: o fraco
    assert loop.tracker.corners is None
    assert loop.first_confirm_gate_blocked == 1
    assert loop.last_first_confirm_gate["passed"] is False
    _drive(loop, 2)  # o forte entra sem atraso de desenho
    assert loop.tracker.corners is not None
    assert loop.first_confirm_gate_blocked == 1


def test_gate_on_and_semantics_each_leg_alone_blocks(monkeypatch):
    monkeypatch.setenv("DRAW_FIRST_CONFIRM_GATE", "1")
    for gate in (
        {"source": "neural", "presence_logit": 5.0, "edge_in": 0.001},
        {"source": "neural", "presence_logit": 0.5, "edge_in": 0.09},
    ):
        loop = _make_loop(lambda _i, g=gate: (QUAD.copy(), "paper", dict(g)))
        _drive(loop, 3)
        assert loop.tracker.corners is None
        assert loop.first_confirm_gate_blocked >= 1


def test_gate_on_confirmed_sheet_and_classic_are_untouched(monkeypatch):
    monkeypatch.setenv("DRAW_FIRST_CONFIRM_GATE", "1")
    classic_weak = {"source": "classic", "presence_logit": None, "edge_in": None}
    loop = _make_loop(lambda _i: (QUAD.copy(), "paper", dict(classic_weak)))
    _drive(loop, 3)
    # A reserva clássica não é julgada pelo portão (ele é da neural).
    assert loop.tracker.corners is not None
    assert loop.first_confirm_gate_blocked == 0

    # Folha confirmada: detecções seguintes não passam pelo portão,
    # nem fracas nem longe (o caminho do salto é do rastreador).
    state = {"gate": dict(STRONG)}
    loop = _make_loop(lambda _i: (QUAD.copy(), "paper", dict(state["gate"])))
    _drive(loop, 3)
    assert loop.tracker.corners is not None
    state["gate"] = dict(WEAK)
    _drive(loop, 3)
    assert loop.tracker.corners is not None
    assert loop.first_confirm_gate_blocked == 0


def test_gate_on_missing_gate_info_never_blocks(monkeypatch):
    monkeypatch.setenv("DRAW_FIRST_CONFIRM_GATE", "1")
    loop = _make_loop(lambda _i: (QUAD.copy(), "paper", None))
    _drive(loop, 3)
    assert loop.tracker.corners is not None
    assert loop.first_confirm_gate_blocked == 0


def test_gate_on_reentry_after_drop_is_gated_again(monkeypatch):
    monkeypatch.setenv("DRAW_FIRST_CONFIRM_GATE", "1")
    phase = {"kind": "strong"}

    def script(_i):
        if phase["kind"] == "strong":
            return QUAD.copy(), "paper", dict(STRONG)
        if phase["kind"] == "absent":
            return None, None, {"source": "veto", "presence_logit": -6.0, "edge_in": None}
        return QUAD.copy(), "paper", dict(WEAK)

    loop = _make_loop(script)
    _drive(loop, 3)
    assert loop.tracker.corners is not None
    phase["kind"] = "absent"
    _drive(loop, 40)  # >10 quadros e >0,5 s sem folha: a folha cai
    assert loop.tracker.corners is None
    phase["kind"] = "weak"
    _drive(loop, 12)  # além do backoff de ausência pós-queda
    # Reentrada fraca é fantasma de reentrada: o portão segura.
    assert loop.tracker.corners is None
    assert loop.first_confirm_gate_blocked >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
