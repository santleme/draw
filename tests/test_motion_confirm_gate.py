import numpy as np
import pytest

from draw import ink_layers as layers
from tests.test_replay_reregister import rr, _ProbIsWarp


def ink():
    prob = np.zeros((100, 120), np.float32)
    prob[15:45, 15:55] = 0.95
    prob[65:75, 65:85] = 0.95
    return prob


def confirmed(acc, prob):
    for _ in range(16):
        result = acc.add(prob, np.ones(prob.shape, bool))
    return result["confirmed"].copy()


@pytest.mark.parametrize("e3", ["0", "1"])
def test_gate_blocks_new_confirmation_and_preserves_canvas(monkeypatch, e3):
    """Registro instavel: a passada vota mas nada cristaliza.

    A copia deslocada do sim-hand nascia de quads degradados no
    meio do move confirmando o traco uma segunda vez. Com o portao,
    a evidencia segue entrando na janela (a provisoria pinta) mas
    nenhum pixel novo materializa confirmacao — e o que ja estava
    confirmado nao muda.
    """
    monkeypatch.setenv("DRAW_INK_VOTE_E3", e3)
    acc = layers._LayeredInkAccumulator()
    prob = ink()
    original = confirmed(acc, prob)
    moved = prob.copy()
    moved[80:90, 80:110] = 0.95  # tinta nova durante o movimento
    visible = np.ones(prob.shape, bool)
    for _ in range(16):
        result = acc.add(moved, visible, confirm_allowed=False)
        np.testing.assert_array_equal(result["confirmed"], original)
        assert result["new_confirmed_px"] == 0
    assert result["confirm_blocked_px"] == 300
    # A evidencia vive na provisoria — o usuario ve a tinta, ela so
    # nao cristalizou ainda.
    assert result["provisional"][80:90, 80:110].any()
    for _ in range(16):
        result = acc.add(moved, visible)
    assert np.all(result["confirmed"][80:90, 80:110] == 255)


def test_gate_does_not_close_on_empty_canvas(monkeypatch):
    """Sem traco para proteger, o portao nao fecha.

    A primeira tinta de uma geracao nova nao tem copia a duplicar —
    bloquea-la seria castigar quem comeca a escrever com a folha
    ainda assentando.
    """
    acc = layers._LayeredInkAccumulator()
    prob = ink()
    visible = np.ones(prob.shape, bool)
    for _ in range(16):
        result = acc.add(prob, visible, confirm_allowed=False)
    assert np.all(result["confirmed"][15:45, 15:55] == 255)
    assert result["confirm_blocked_px"] == 0


def test_confirm_allowed_default_is_byte_identical(monkeypatch):
    """Sem o argumento o voto corre como sempre correu."""
    first = layers._LayeredInkAccumulator()
    second = layers._LayeredInkAccumulator()
    prob = ink()
    visible = np.ones(prob.shape, bool)
    for _ in range(16):
        a = first.add(prob, visible)
        b = second.add(prob, visible, confirm_allowed=True)
        np.testing.assert_array_equal(a["confirmed"], b["confirmed"])
        np.testing.assert_array_equal(a["provisional"], b["provisional"])
        assert b["confirm_blocked_px"] == 0


def drive(sequence, monkeypatch, flag="1"):
    monkeypatch.setenv("DRAW_INK_MOTION_GATE", flag)
    monkeypatch.setenv("DRAW_REG_CONTENT_GUARD", "0")
    monkeypatch.setenv("DRAW_ANCHOR_ARBITRATION", "0")
    monkeypatch.setenv("DRAW_ANCHOR_FOLLOW", "0")
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    clock = rr.SceneClock()
    worker = rr.make_ink_worker(_ProbIsWarp(), clock)
    gen = worker.new_generation()
    for prob, corners in sequence:
        clock.now += 1
        worker.submit(prob, gen, corners, np.ones(prob.shape, bool))
        assert worker.pump()
        assert worker.error is None
    return worker


def test_moved_quad_suppresses_duplicate_settle_confirms_new_ink(monkeypatch):
    """Cenario do sim-hand em miniatura na trabalhadora real.

    Tinta confirmada assentada; a folha desliza (quad +80 px) com o
    mesmo traco chegando registrado na ancora — nada confirma
    enquanto o deslocamento segue acima do limiar. No retorno, a
    tinta legitima confirma normalmente.
    """
    prob = ink()
    base = rr.base_quad(prob.shape)
    moved = base + np.float32([80, 0])
    more = prob.copy()
    more[80:90, 15:35] = 0.95
    worker = drive(
        [(prob, base)] * 8 + [(more, moved)] * 8, monkeypatch
    )
    original = np.count_nonzero(
        (worker._accumulator._confirmed > 0)
    )
    records = worker.telemetry_snapshot()["records"]
    blocked = [r for r in records if r.get("motion_confirm_blocked")]
    assert blocked, "o portao tinha que registrar nas passadas moved"
    # Nenhum pixel novo materializou durante o deslize.
    assert np.count_nonzero(worker._accumulator._confirmed) == original
    # De volta assentado, a tinta nova confirma.
    worker = drive(
        [(prob, base)] * 8 + [(more, moved)] * 8 + [(more, base)] * 16,
        monkeypatch,
    )
    assert np.all(worker._accumulator._confirmed[80:90, 15:35] == 255)


def test_gate_off_does_not_record_or_block(monkeypatch):
    prob = ink()
    base = rr.base_quad(prob.shape)
    moved = base + np.float32([80, 0])
    worker = drive([(prob, base)] * 8 + [(prob, moved)] * 4, monkeypatch, "0")
    records = worker.telemetry_snapshot()["records"]
    assert not any(r.get("motion_confirm_blocked") for r in records)
