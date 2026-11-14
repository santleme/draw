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


def test_reset_preserves_confirmed_and_restarts_votes():
    """O canvas fica; a memoria posicional de voto recomeca.

    A ancora nova nao herda janela, provisoria, streak de remocao
    nem trilhas — tudo isso estava escrito sob o registro velho.
    O confirmado, que e o conteudo da folha, passa intacto.
    """
    acc = layers._LayeredInkAccumulator()
    prob = ink()
    original = confirmed(acc, prob)
    assert len(acc._window) > 0
    acc.reset_vote_memory()
    np.testing.assert_array_equal(acc._confirmed > 0, original > 0)
    assert len(acc._window) == 0
    assert acc._removal_streak is None
    assert not acc._provisional.any()
    # A tinta legitima reconfirma do zero — nao e resgate de voto velho.
    result = acc.add(prob, np.ones(prob.shape, bool))
    assert result["new_confirmed_px"] == 0
    for _ in range(16):
        acc.add(prob, np.ones(prob.shape, bool))
    np.testing.assert_array_equal(acc._confirmed > 0, original > 0)


def test_reset_before_first_add_is_safe():
    layers._LayeredInkAccumulator().reset_vote_memory()


def _big_ink():
    prob = np.zeros((260, 400), np.float32)
    prob[10:190, 10:200] = 0.95
    return prob


def test_fast_clean_removes_stale_grid_ink_in_half_the_streak(monkeypatch):
    """Detrito de grade sai com metade do streak dentro da janela.

    O pouso e veredito de conteudo: a copia defasada que fica na
    grade velha nao precisa das 6 passadas do protocolo cheio —
    a testemunha segue exigida, entao o fail-closed e identico.
    """
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    acc = layers._LayeredInkAccumulator()
    prob = _big_ink()
    confirmed(acc, prob)
    acc.note_reanchor()
    stale = prob.copy()
    stale[10:190, 10:110] = 0.0
    visible = np.ones(prob.shape, bool)
    for _ in range(layers._REMOVAL_FAST_PASSES - 1):
        acc.add(stale, visible)
        assert (acc._confirmed > 0)[10:190, 10:110].any()
    acc.add(stale, visible)
    assert not acc._confirmed[10:190, 10:110].any()
    assert np.all(acc._confirmed[10:190, 110:200] == 255)


def test_fast_clean_window_expires_back_to_full_protocol(monkeypatch):
    """Fora da janela o protocolo cheio volta a valer."""
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    acc = layers._LayeredInkAccumulator()
    prob = _big_ink()
    confirmed(acc, prob)
    acc.note_reanchor()
    visible = np.ones(prob.shape, bool)
    for _ in range(layers._REMOVAL_FAST_WINDOW):
        acc.add(prob, visible)  # gasta a janela sem ausencias
    stale = prob.copy()
    stale[10:190, 10:110] = 0.0
    for _ in range(layers._REMOVAL_FAST_PASSES):
        acc.add(stale, visible)
    # 3 ausencias nao bastam fora da janela — o protocolo cheio pede 6.
    assert (acc._confirmed > 0)[10:190, 10:110].any()
    for _ in range(layers._REMOVAL_PASSES - layers._REMOVAL_FAST_PASSES):
        acc.add(stale, visible)
    assert not acc._confirmed[10:190, 10:110].any()


def test_fast_clean_still_requires_witness(monkeypatch):
    """Sem testemunha a janela rapida nao remove nada — fail-closed."""
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    acc = layers._LayeredInkAccumulator()
    prob = _big_ink()
    original = confirmed(acc, prob)
    acc.note_reanchor()
    dark = np.zeros_like(prob)
    visible = np.ones(prob.shape, bool)
    for _ in range(2 * layers._REMOVAL_FAST_PASSES):
        acc.add(dark, visible)
    np.testing.assert_array_equal(acc._confirmed > 0, original > 0)


def test_fast_clean_unwitnessed_passes_do_not_consume_window(monkeypatch):
    """Oclusao/sem info nao gasta a janela — ela espera testemunha."""
    monkeypatch.setenv("DRAW_INK_REMOVAL", "1")
    acc = layers._LayeredInkAccumulator()
    prob = _big_ink()
    confirmed(acc, prob)
    acc.note_reanchor()
    occluded = np.zeros(prob.shape, bool)
    visible = np.ones(prob.shape, bool)
    for _ in range(5):
        acc.add(prob, occluded)  # nada testemunhado: janela intacta
    assert acc._removal_fast_left == layers._REMOVAL_FAST_WINDOW
    stale = prob.copy()
    stale[10:190, 10:110] = 0.0
    for _ in range(layers._REMOVAL_FAST_PASSES):
        acc.add(stale, visible)
    assert not acc._confirmed[10:190, 10:110].any()


def drive_stepped(sequence, monkeypatch, vote_reset="1"):
    monkeypatch.setenv("DRAW_ANCHOR_VOTE_RESET", vote_reset)
    monkeypatch.setenv("DRAW_ANCHOR_FOLLOW", "1")
    monkeypatch.setenv("DRAW_ANCHOR_ARBITRATION", "0")
    monkeypatch.setenv("DRAW_REG_CONTENT_GUARD", "0")
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    clock = rr.SceneClock()
    worker = rr.make_ink_worker(_ProbIsWarp(), clock)
    gen = worker.new_generation()
    for prob, corners in sequence:
        clock.now += 1
        worker.submit(prob, gen, corners, np.ones(prob.shape, bool))
        assert worker.pump()
        assert worker.error is None
        yield worker


@pytest.mark.parametrize("vote_reset", ["0", "1"])
def test_follow_reanchor_resets_window_written_in_old_grid(monkeypatch, vote_reset):
    """Na passada do re-ancora a janela reflete a grade vigente.

    Sem a flag a janela atravessa a troca cheia da evidencia
    registrada sob a ancora anterior — desalinhada pelo novo
    mapeamento, ela vota a copia de uma vez (os 10,7k px da seq 56
    do sim-hand). Com a flag a memoria posicional recomeca: a
    janela sai do re-ancora vazia e os votos daqui em diante sao
    todos da grade nova.
    """
    prob = ink()
    base = rr.base_quad(prob.shape)
    # Rotacao + deslize: muda o mapeamento da grade e passa do
    # limiar do FOLLOW (10° sozinhos dao media 7,8 px — abaixo de 8).
    moved = rr.transform_quad(base, shift=(30.0, 20.0), angle=12.0)
    sequence = [(prob, base)] * 16 + [(prob, moved)] * 8
    window_at_reanchor = None
    for worker in drive_stepped(sequence, monkeypatch, vote_reset):
        if worker._anchor_reanchors:
            window_at_reanchor = len(worker._accumulator._window)
            confirmed = worker._accumulator._confirmed.copy()
            break
    assert window_at_reanchor is not None, "o FOLLOW nao re-ancorou"
    assert np.count_nonzero(confirmed) > 0, "o canvas nao sobreviveu"
    if vote_reset == "1":
        # A passada do re-ancora votou na grade nova — a janela
        # so tem a evidencia desta passada.
        assert window_at_reanchor == 1
    else:
        assert window_at_reanchor > 1


def test_reset_vote_memory_keeps_retraction_totals_monotonic():
    """O total de retração não recua quando a janela zera.

    As ever-masks são posicionais — a grade nova as recomeca, mas
    o que já foi contado segue nos totais: o SessionTelemetry lê
    deltas e um retracted_px_total que cai vira incremento
    negativo na série.
    """
    acc = layers._LayeredInkAccumulator()
    prob = ink()
    confirmed(acc, prob)
    visible = np.ones(prob.shape, bool)
    # Faísca provisória: some da janela sem confirmar → retração.
    spark = prob.copy()
    spark[5:12, 90:110] = 0.95
    acc.add(spark, visible)
    for _ in range(layers._INK_VOTE_WINDOW + 1):
        result = acc.add(prob, visible)
    retracted = result["retracted_px_total"]
    provisional_ever = result["provisional_ever_px"]
    assert retracted > 0
    acc.reset_vote_memory()
    result = acc.add(prob, visible)
    assert result["retracted_px_total"] >= retracted
    assert result["provisional_ever_px"] >= provisional_ever


def test_incomplete_vote_window_blocks_total_rest(monkeypatch):
    """Pós-re-ancoragem o descanso espera a janela encher.

    O reset de voto esvazia a janela mas os contadores de
    quietude são da geração e seguem contados — sem a guarda, a
    trabalhadora dormia antes do 3º voto e a tinta fraca morria
    provisória.
    """
    monkeypatch.setenv("DRAW_ANCHOR_VOTE_RESET", "1")
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    clock = rr.SceneClock()
    worker = rr.make_ink_worker(_ProbIsWarp(), clock)
    gen = worker.new_generation()
    prob = ink()
    base = rr.base_quad(prob.shape)
    visible = np.ones(prob.shape, bool)
    for _ in range(6):
        clock.now += 1
        worker.submit(prob, gen, base, visible)
        assert worker.pump()
        assert worker.error is None
    # Condições de descanso total, montadas como a quiescência
    # real as deixa: sem mão, confirmada parada, geração velha.
    worker._hand_state = False
    worker._confirm_once = False
    worker._generation_passes = 99
    worker._quiet_passes = 99
    assert worker._at_rest()
    worker._accumulator.reset_vote_memory()
    assert not worker._at_rest()
    # A janela enche com as passadas seguintes e o descanso
    # volta a ser decidido pela quietude de sempre.
    for _ in range(layers._INK_VOTE_WINDOW):
        worker._accumulator.add(prob, visible)
    assert worker._at_rest()
