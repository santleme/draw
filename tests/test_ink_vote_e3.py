"""Regra de voto E3 (frente 89/90): commit do nucleo + trilhas.

O defeito (relatorios 86/89, draw-lab/live-audit-2026-11-10/): a
regra de voto atual confirma sobre janelas deslizantes
sobrepostas; na regua de 17 passadas do mesmo spread parado o
canvas cresce 2,90x porque a franja do flicker da v6 pisca 2x na
mesma posicao dentro de alguma janela e confirma. A candidata E3
da frente 89 — unica das 17 que passou nos portoes
pre-registrados — troca a regra atras de DRAW_INK_VOTE_E3=1:

- fase 1 (3 primeiras passadas): o canvas nasce uma vez do nucleo
  de consenso espacial (tinta a <=3 px em 3/3 passadas + vista
  >limiar 1x no pixel + soma >= 1,0); a uniao vista vira o
  "conteudo velho" da geracao;
- fase 2: so componente conexo NOVO (area >= 20, <50% sobre o
  velho dilatado e fora das trilhas abertas) abre trilha lateral,
  dividida por tamanho: >=100 px confirma com 3 consecutivas
  >limiar (soma >= 2,1); 20-99 px com 5 (soma >= 3,5); <20 px e
  ignorado.

Com a flag desligada (padrao), o acumulador roda a regra atual e
o resultado nao ganha nenhuma chave nova.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import ink_layers as ink_layers_module  # noqa: E402

SHAPE = (200, 200)
ENV = "DRAW_INK_VOTE_E3"


class _ProbIsWarp:
    """page_clean falso: o "warp" submetido ja e o mapa."""

    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def e3(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    return monkeypatch


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    return monkeypatch


def _acc():
    return ink_layers_module._LayeredInkAccumulator(threshold=0.7)


def _page(blocks=()):
    prob = np.zeros(SHAPE, dtype=np.float32)
    for (y0, y1, x0, x1), value in blocks:
        prob[y0:y1, x0:x1] = value
    return prob


# Bloco base da pagina estabelecida e blocos novos por tamanho.
BASE = ((20, 120, 20, 80), 0.95)  # 100x60 = 6.000 px
BIG = ((150, 190, 150, 190), 0.97)  # 40x40 = 1.600 px (>= 100)
SMALL = ((150, 155, 20, 30), 0.97)  # 5x10 = 50 px (20-99)
TINY = ((150, 154, 150, 154), 0.97)  # 4x4 = 16 px (< 20)
DOT = ((150, 158, 150, 158), 0.97)  # 8x8 = 64 px (trilha lenta)


def _mask(block):
    (y0, y1, x0, x1), _value = block
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _establish(acc, passes=3):
    result = None
    for _ in range(passes):
        result = acc.add(_page([BASE]))
    return result


def _fraction(confirmed, block):
    mask = _mask(block)
    return float(np.count_nonzero(confirmed & mask) / np.count_nonzero(mask))


def _latency(acc, block, max_passes=8):
    """Passadas do bloco novo ate >=90% dele estar confirmado."""
    page = _page([BASE, block])
    for k in range(1, max_passes + 1):
        result = acc.add(page)
        if _fraction(result["confirmed"] > 0, block) >= 0.9:
            return k
    return None


# --- flag -------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("0", False), ("2", False), ("1", True), (" On ", True)],
)
def test_flag_is_explicit_opt_in(monkeypatch, value, expected):
    monkeypatch.delenv(ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert ink_layers_module._ink_vote_e3_enabled() is expected


def test_constants_are_the_f89_values():
    m = ink_layers_module
    assert m._INK_VOTE_E3_PHASE1_PASSES == 3
    assert m._INK_VOTE_E3_CORE_MIN_SUM == 1.0
    assert m._INK_VOTE_E3_MIN_COMPONENT_PX == 20
    assert m._INK_VOTE_E3_BIG_COMPONENT_PX == 100
    assert (m._INK_VOTE_E3_BIG_WINDOW, m._INK_VOTE_E3_BIG_SCORE) == (3, 2.1)
    assert (m._INK_VOTE_E3_SMALL_WINDOW, m._INK_VOTE_E3_SMALL_SCORE) == (5, 3.5)


# --- desligada: regra atual, sem chave nova ----------------------------


def test_off_runs_the_current_rule_without_new_keys(off):
    acc = _acc()
    result = None
    for _ in range(2):
        result = acc.add(_page([BASE]))
    # Regra atual: tinta forte (0,95) confirma na 2a passada.
    assert result["confirmed_px"] == 6000
    assert not any(key.startswith("ink_vote_e3") for key in result)


def test_off_canvas_equals_default_rule_on_mixed_sequence(off):
    sequence = [
        _page([BASE]),
        _page([BASE]),
        _page([BASE, BIG]),
        _page([BASE, BIG]),
        _page([BASE]),
    ]
    acc_a, acc_b = _acc(), _acc()
    for prob in sequence:
        result_a = acc_a.add(prob)
        result_b = acc_b.add(prob.copy())
    assert np.array_equal(result_a["confirmed"], result_b["confirmed"])


# --- fase 1 ------------------------------------------------------------


def test_phase1_confirms_nothing_before_the_third_pass(e3):
    acc = _acc()
    first = acc.add(_page([BASE]))
    second = acc.add(_page([BASE]))
    assert first["confirmed_px"] == 0
    assert second["confirmed_px"] == 0
    # A provisoria mostra a tinta desde a 1a passada (exibicao).
    assert first["provisional_px"] == 6000


def test_phase1_core_is_the_block_exactly(e3):
    acc = _acc()
    result = _establish(acc)
    assert np.array_equal(result["confirmed"] > 0, _mask(BASE))
    assert result["ink_vote_e3"] is True
    assert result["ink_vote_e3_passes"] == 3


def test_phase1_spatial_consensus_confirms_single_hit_pixel(e3, monkeypatch):
    """Pixel visto forte 1x, com tinta a <=3 px nas 3 passadas.

    O pixel (30, 30) tem 0,97 na passada 1 e 0,30 nas outras duas
    (soma 1,57 >= 1,0), cercado pelo bloco nas 3: o consenso
    espacial da E3 confirma (vizinhanca 3/3 + 1 vista + soma).
    A regra atual nao confirma esse pixel (soma 1,57 < 1,9 e
    hits >= 0,5 em 1 passada so).
    """
    maps = []
    for k in range(3):
        prob = _page([BASE])
        prob[30, 30] = 0.97 if k == 0 else 0.30
        maps.append(prob)
    acc = _acc()
    for prob in maps:
        result = acc.add(prob)
    assert result["confirmed"][30, 30] == 255
    monkeypatch.delenv(ENV, raising=False)
    acc_off = _acc()
    for prob in maps:
        result_off = acc_off.add(prob)
    assert result_off["confirmed"][30, 30] == 0


def test_phase1_flicker_pixel_outside_consensus_stays_out(e3):
    """Pixel forte numa passada so, longe de tinta nas outras."""
    far = np.zeros(SHAPE, dtype=np.float32)
    far[180, 20] = 0.97
    acc = _acc()
    acc.add(_page([BASE]) + far)
    acc.add(_page([BASE]))
    result = acc.add(_page([BASE]))
    assert result["confirmed"][180, 20] == 0


def test_phase1_content_outside_core_is_old_forever(e3):
    """Custo 3 da F89: presente na fase 1 fora do nucleo = velho.

    O bloco aparece em posicoes diferentes nas 3 passadas da
    fase 1 (sem consenso espacial, sem nucleo) e depois assenta
    numa das posicoes: cai sobre a uniao velha dilatada e nunca
    confirma na geracao.
    """
    spots = [(150, 190, 20, 60), (150, 190, 80, 120), (150, 190, 140, 180)]
    acc = _acc()
    for spot in spots:
        acc.add(_page([(spot, 0.95)]))
    settled = _page([(spots[0], 0.95)])
    result = None
    for _ in range(6):
        result = acc.add(settled)
    assert _fraction(result["confirmed"] > 0, (spots[0], 0.95)) == 0.0
    assert result["ink_vote_e3_tracks_big"] == 0


# --- fase 2: trilhas por tamanho ---------------------------------------


def test_big_component_confirms_on_third_consecutive_pass(e3):
    acc = _acc()
    _establish(acc)
    assert _latency(acc, BIG) == 3


def test_small_component_uses_the_slow_track(e3):
    acc = _acc()
    _establish(acc)
    assert _latency(acc, SMALL) == 5


def test_tiny_component_is_ignored(e3):
    acc = _acc()
    _establish(acc)
    assert _latency(acc, TINY, max_passes=8) is None
    result = acc.add(_page([BASE, TINY]))
    assert _fraction(result["confirmed"] > 0, TINY) == 0.0


def test_small_mark_latency_is_five_vs_two_in_the_current_rule(e3, monkeypatch):
    """Regua extra da F89: ponto 8x8 sobre pagina estabelecida."""
    acc = _acc()
    _establish(acc)
    assert _latency(acc, DOT) == 5
    monkeypatch.delenv(ENV, raising=False)
    acc_off = _acc()
    _establish(acc_off)
    assert _latency(acc_off, DOT) == 2


def test_track_breaks_when_the_component_blinks(e3):
    """Persistencia consecutiva: 2 vistas, 1 ausencia, recomeca."""
    acc = _acc()
    _establish(acc)
    page_new = _page([BASE, BIG])
    acc.add(page_new)
    acc.add(page_new)
    acc.add(_page([BASE]))  # o traco some: a corrida quebra
    assert _latency(acc, BIG) == 3  # precisa de 3 novas consecutivas


def test_component_over_old_content_opens_no_track(e3):
    """Copia deslocada 2 px do bloco: sobreposicao com o velho."""
    acc = _acc()
    _establish(acc)
    shifted = ((22, 122, 22, 82), 0.97)
    for _ in range(5):
        result = acc.add(_page([BASE, shifted]))
    assert result["ink_vote_e3_tracks_big"] == 0
    assert result["ink_vote_e3_tracks_small"] == 0
    # Os 2 px de borda nova da copia nunca confirmam.
    edge = np.zeros(SHAPE, dtype=bool)
    edge[120:122, 22:82] = True
    assert not np.any((result["confirmed"] > 0) & edge)


def test_tracks_are_counted(e3):
    acc = _acc()
    _establish(acc)
    acc.add(_page([BASE, BIG, SMALL]))
    result = acc.add(_page([BASE, BIG, SMALL]))
    assert result["ink_vote_e3_tracks_big"] == 1
    assert result["ink_vote_e3_tracks_small"] == 1


# --- reset / geracao ----------------------------------------------------


def test_reset_starts_a_fresh_e3_generation(e3):
    acc = _acc()
    _establish(acc)
    canvas_a = acc.add(_page([BASE]))["confirmed"].copy()
    acc.reset()
    other = ((150, 190, 120, 180), 0.95)  # pagina B, disjunta de BASE
    result = None
    for _ in range(4):
        result = acc.add(_page([other]))
    confirmed_b = result["confirmed"] > 0
    assert np.array_equal(confirmed_b, _mask(other))
    assert not np.any(confirmed_b & (canvas_a > 0))
    assert result["ink_vote_e3_passes"] == 4


def test_mode_is_latched_per_generation(e3, monkeypatch):
    acc = _acc()
    acc.add(_page([BASE]))
    monkeypatch.delenv(ENV, raising=False)
    result = _establish(acc, passes=2)  # completa a fase 1 como E3
    assert result["confirmed_px"] == 6000
    assert result.get("ink_vote_e3") is True
    acc.reset()
    first = acc.add(_page([BASE]))
    second = acc.add(_page([BASE]))
    # Geracao nova sem a flag: regra atual (confirma na 2a).
    assert second["confirmed_px"] == 6000
    assert not any(key.startswith("ink_vote_e3") for key in first)


def test_shape_change_restarts_the_e3_state(e3):
    acc = _acc()
    _establish(acc)
    small_shape = np.zeros((50, 50), dtype=np.float32)
    small_shape[5:25, 5:25] = 0.95
    acc.add(small_shape)
    acc.add(small_shape)
    result = acc.add(small_shape)
    assert result["confirmed_px"] == 400
    assert result["ink_vote_e3_passes"] == 3


# --- trabalhadora real ---------------------------------------------------


def _drive(worker, generation, prob, corners):
    before = worker.stats()["passes_total"]
    worker.submit(prob, generation, corners)
    deadline = time.monotonic() + 5.0
    while worker.stats()["passes_total"] <= before:
        assert time.monotonic() < deadline, "a passada nao executou"
        time.sleep(0.001)
    assert worker.error is None


@pytest.fixture
def fast_worker(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.005)
    monkeypatch.delenv("DRAW_ECC_REFINE", raising=False)
    return monkeypatch


QUAD = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])


@pytest.mark.lento
def test_worker_with_e3_confirms_core_and_reports_telemetry(e3, fast_worker):
    worker = ink_layers_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        for _ in range(3):
            _drive(worker, generation, _page([BASE]), QUAD)
        confirmed = worker._accumulator._confirmed
        assert np.array_equal(confirmed > 0, _mask(BASE))
        records = worker.telemetry_snapshot()["records"]
        assert records[-1]["ink_vote_e3_passes"] == 3
        assert records[-1]["ink_vote_e3_tracks_big"] == 0
    finally:
        worker.stop()


@pytest.mark.lento
def test_worker_without_e3_has_no_e3_telemetry(off, fast_worker):
    worker = ink_layers_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        for _ in range(3):
            _drive(worker, generation, _page([BASE]), QUAD)
        records = worker.telemetry_snapshot()["records"]
        assert not any(
            key.startswith("ink_vote_e3") for record in records for key in record
        )
    finally:
        worker.stop()
