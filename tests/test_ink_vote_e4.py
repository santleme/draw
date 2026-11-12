"""Regra de voto E5-S (frentes 106/107, implementacao da 108):
E4j + subdivisao por nascimento, atras de DRAW_INK_VOTE_E4.

Os defeitos que ela fecha (relatorios 103/106/107,
draw-lab/live-audit-2026-11-10/): a E3 confunde "visto" com
"velho" — a uniao de tudo que foi visto na fase 1 vira barreira,
o residuo da fase 1 e velho pra sempre e um componente novo
majoritariamente sobre o velho dilatado e absorvido inteiro
(hachura progressiva: 1,000 -> 0,072 no simulador com GT exato).
A E5-S, porto do prototipo validado da F107 (classe E5 com
use_subdiv=True, use_weak=False, use_thinweak=False):

- "velho" = so a abertura da geracao (mascara da 1a passada
  congelada + mapa de probabilidade dela; evidencia >= 0,35 em
  >=50% do componente tambem e pre-existencia), sem expansao;
- fase 1 = nucleo de consenso da E3, mas o residuo dela e
  reclassificado como conteudo novo na fase 2;
- novidade/cobertura contra o confirmado e as regioes abertas:
  componente <50% coberto abre trilha pela forma/distancia
  (grande 3/2,1 regiao r6; pequena 5/3,5 regiao r6; lenta na
  banda 3-8 px do confirmado, 5/3,5 regiao r3; fina em forma de
  linha, voto tolerante a 1 px, 3/2,1 regiao r3);
- componente >=50% coberto e SUBDIVIDIDO: o residuo fora da
  cobertura abre as suas trilhas (regiao r3).

Precedencia declarada: DRAW_INK_VOTE_E4 + DRAW_INK_VOTE_E3
ligadas = roda so a E5-S (telemetria da E4). Com a flag
desligada (padrao), o acumulador roda a regra atual e o
resultado nao ganha nenhuma chave nova.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import ink_layers as ink_layers_module  # noqa: E402

SHAPE = (300, 400)
ENV = "DRAW_INK_VOTE_E4"
ENV_E3 = "DRAW_INK_VOTE_E3"


class _ProbIsWarp:
    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def e4(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.delenv(ENV_E3, raising=False)
    return monkeypatch


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(ENV_E3, raising=False)
    return monkeypatch


def _acc():
    return ink_layers_module._LayeredInkAccumulator(threshold=0.7)


def _page(blocks=()):
    prob = np.zeros(SHAPE, dtype=np.float32)
    for (y0, y1, x0, x1), value in blocks:
        prob[y0:y1, x0:x1] = value
    return prob


BASE = ((20, 120, 20, 80), 0.95)  # 100x60 = 6.000 px
BIG = ((150, 190, 150, 350), 0.97)  # 40x200 = 8.000 px (>= 100)
SMALL = ((150, 155, 20, 30), 0.97)  # 5x10 = 50 px (20-99)
TINY = ((150, 154, 150, 154), 0.97)  # 4x4 = 16 px (< 20)
DOT = ((250, 258, 300, 308), 0.97)  # 8x8 = 64 px (trilha pequena)


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
    assert ink_layers_module._ink_vote_e4_enabled() is expected


def test_constants_are_the_f106_f107_values():
    m = ink_layers_module
    assert m._INK_VOTE_E4_OPENING_EVIDENCE == 0.35
    assert m._INK_VOTE_E4_MIN_COMPONENT_PX == 20
    assert m._INK_VOTE_E4_BIG_COMPONENT_PX == 100
    assert (m._INK_VOTE_E4_BIG_WINDOW, m._INK_VOTE_E4_BIG_SCORE) == (3, 2.1)
    assert (m._INK_VOTE_E4_SMALL_WINDOW, m._INK_VOTE_E4_SMALL_SCORE) == (5, 3.5)
    assert (m._INK_VOTE_E4_SLOW_WINDOW, m._INK_VOTE_E4_SLOW_SCORE) == (5, 3.5)
    assert (m._INK_VOTE_E4_THIN_WINDOW, m._INK_VOTE_E4_THIN_SCORE) == (3, 2.1)
    assert m._INK_VOTE_E4_THIN_MIN_LENGTH == 50
    assert m._INK_VOTE_E4_THIN_MAX_AREA_RATIO == 3.5


# --- desligada: regra atual, sem chave nova ----------------------------


def test_off_runs_the_current_rule_without_new_keys(off):
    acc = _acc()
    result = None
    for _ in range(2):
        result = acc.add(_page([BASE]))
    assert result["confirmed_px"] == 6000
    assert not any(key.startswith("ink_vote_e4") for key in result)
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
    assert np.array_equal(result_a["provisional"], result_b["provisional"])


# --- fase 1 e abertura ---------------------------------------------------


def test_phase1_confirms_nothing_before_the_third_pass(e4):
    acc = _acc()
    first = acc.add(_page([BASE]))
    second = acc.add(_page([BASE]))
    assert first["confirmed_px"] == 0
    assert second["confirmed_px"] == 0
    assert first["provisional_px"] == 6000


def test_phase1_core_is_the_block_exactly_and_telemetry(e4):
    acc = _acc()
    result = _establish(acc)
    assert np.array_equal(result["confirmed"] > 0, _mask(BASE))
    assert result["ink_vote_e4"] is True
    assert result["ink_vote_e4_passes"] == 3
    assert result["ink_vote_e4_tracks_sub"] == 0


def test_phase1_residue_is_new_content_not_old_forever(e4):
    """A correcao central contra a E3: visto na fase 1 fora do
    nucleo confirma na fase 2 (na E3 seria "velho" pra sempre)."""
    spot = ((200, 240, 200, 240), 0.97)
    acc = _acc()
    acc.add(_page([BASE]))
    acc.add(_page([BASE, spot]))
    result = acc.add(_page([BASE, spot]))
    assert _fraction(result["confirmed"] > 0, spot) == 0.0  # fora do nucleo
    assert _latency(acc, spot) == 3  # trilha grande na fase 2


def test_opening_evidence_vetoes_preexisting_component(e4):
    """Evidencia de abertura >= 0,35 em >=50% = pre-existente:
    o bloco fraco da 1a passada que "fortalece" depois nunca
    abre trilha (e a abertura nao expande: outro bloco longe
    confirma normalmente)."""
    weak = ((200, 240, 200, 240), 0.4)
    strong = ((200, 240, 200, 240), 0.97)
    acc = _acc()
    acc.add(_page([BASE, weak]))
    acc.add(_page([BASE, strong]))
    acc.add(_page([BASE, strong]))
    result = None
    for _ in range(6):
        result = acc.add(_page([BASE, strong]))
    assert _fraction(result["confirmed"] > 0, strong) == 0.0
    assert result["ink_vote_e4_tracks_big"] == 0


def test_shifted_copy_of_confirmed_opens_no_track(e4):
    """Copia deslocada 2 px do confirmado: cobertura >= 50% e
    residuo < 20 px util — nenhuma trilha, nenhuma confirmacao
    da borda nova."""
    acc = _acc()
    _establish(acc)
    shifted = ((22, 122, 22, 82), 0.97)
    result = None
    for _ in range(5):
        result = acc.add(_page([BASE, shifted]))
    tracks = sum(
        result[f"ink_vote_e4_tracks_{k}"]
        for k in ("big", "small", "slow", "thin", "sub")
    )
    assert tracks == 0
    edge = np.zeros(SHAPE, dtype=bool)
    edge[120:122, 22:82] = True
    assert not np.any((result["confirmed"] > 0) & edge)


# --- fase 2: trilhas ------------------------------------------------------


def test_big_component_confirms_on_third_consecutive_pass(e4):
    acc = _acc()
    _establish(acc)
    assert _latency(acc, BIG) == 3
    assert acc._e4_tracks_big == 1


def test_small_component_uses_the_slow_small_track(e4):
    acc = _acc()
    _establish(acc)
    assert _latency(acc, SMALL) == 5


def test_tiny_component_is_ignored(e4):
    acc = _acc()
    _establish(acc)
    assert _latency(acc, TINY, max_passes=8) is None
    result = acc.add(_page([BASE, TINY]))
    assert _fraction(result["confirmed"] > 0, TINY) == 0.0


def test_small_mark_latency_is_five_vs_two_in_the_current_rule(e4, monkeypatch):
    acc = _acc()
    _establish(acc)
    assert _latency(acc, DOT) == 5
    monkeypatch.delenv(ENV, raising=False)
    acc_off = _acc()
    _establish(acc_off)
    assert _latency(acc_off, DOT) == 2


def test_slow_track_for_component_in_the_3_to_8px_band(e4):
    """Faixa colada no confirmado (5 px de gap): trilha lenta."""
    acc = _acc()
    _establish(acc)
    strip = ((125, 130, 20, 80), 0.97)
    result = None
    for _ in range(7):
        result = acc.add(_page([BASE, strip]))
    assert acc._e4_tracks_slow == 1
    assert _fraction(result["confirmed"] > 0, strip) == 1.0


def test_thin_track_tolerates_one_pixel_jitter(e4):
    """Linha de 1 px que alterna +-1 px por passada: o voto
    estrito no mesmo pixel nao fecharia; a trilha fina fecha."""
    acc = _acc()
    _establish(acc)
    line = np.zeros(SHAPE, dtype=bool)
    line[250, 50:350] = True
    latency = None
    for k in range(1, 9):
        page = _page([BASE])
        page[250 + (k % 2), 50:350] = 0.85
        result = acc.add(page)
        frac = float(
            np.count_nonzero((result["confirmed"] > 0) & line) / np.count_nonzero(line)
        )
        if frac >= 0.9:
            latency = k
            break
    assert latency == 3
    assert acc._e4_tracks_thin == 1


def test_subdivision_by_birth_opens_track_for_the_residue(e4):
    """A peca da F107: a barra cresce grudada na parte ja com
    trilha — o componente inteiro fica >=50% coberto, mas o
    residuo (a extensao) abre a SUA trilha e confirma. Sem
    subdivisao o componente seria pulado e a extensao perdida."""
    acc = _acc()
    _establish(acc)
    s1 = ((150, 160, 100, 200), 0.97)
    s2 = ((150, 160, 200, 300), 0.97)
    for _ in range(3):
        result = acc.add(_page([BASE, s1]))
    assert _fraction(result["confirmed"] > 0, s1) == 1.0
    latency = None
    for k in range(1, 9):
        result = acc.add(_page([BASE, s1, s2]))
        if _fraction(result["confirmed"] > 0, s2) >= 0.9:
            latency = k
            break
    assert latency == 3
    assert acc._e4_tracks_sub == 1


# --- precedencia, latch, reset, migracao ---------------------------------


def test_e4_wins_over_e3_when_both_flags_are_on(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(ENV_E3, "1")
    acc = _acc()
    result = _establish(acc)
    assert result.get("ink_vote_e4") is True
    assert not any(key.startswith("ink_vote_e3") for key in result)
    assert acc._e4_active is True
    assert acc._e3_active is False


def test_mode_is_latched_per_generation(e4, monkeypatch):
    acc = _acc()
    acc.add(_page([BASE]))
    monkeypatch.delenv(ENV, raising=False)
    result = _establish(acc, passes=2)
    assert result.get("ink_vote_e4") is True
    acc.reset()
    first = acc.add(_page([BASE]))
    second = acc.add(_page([BASE]))
    assert second["confirmed_px"] == 6000
    assert not any(key.startswith("ink_vote_e4") for key in first)


def test_reset_starts_a_fresh_e4_generation(e4):
    acc = _acc()
    _establish(acc)
    canvas_a = acc.add(_page([BASE]))["confirmed"].copy()
    acc.reset()
    other = ((150, 190, 120, 320), 0.95)
    result = None
    for _ in range(4):
        result = acc.add(_page([other]))
    confirmed_b = result["confirmed"] > 0
    assert np.array_equal(confirmed_b, _mask(other))
    assert not np.any(confirmed_b & (canvas_a > 0))
    assert result["ink_vote_e4_passes"] == 4


def test_shape_change_restarts_the_e4_state(e4):
    acc = _acc()
    _establish(acc)
    small_shape = np.zeros((50, 50), dtype=np.float32)
    small_shape[5:25, 5:25] = 0.95
    acc.add(small_shape)
    acc.add(small_shape)
    result = acc.add(small_shape)
    assert result["confirmed_px"] == 400
    assert result["ink_vote_e4_passes"] == 3


def test_migrate_keeps_e4_state_on_the_new_grid(e4):
    acc = _acc()
    _establish(acc)
    acc.add(_page([BASE, BIG]))  # abre a trilha grande
    identity = np.eye(3, dtype=np.float64)
    acc.migrate(identity)
    # Estado migrado segue coerente: a alimentacao pre-migracao
    # sobreviveu (faltam 2 das 3 da trilha grande, nao 3).
    assert _latency(acc, BIG) == 2
    assert np.array_equal(acc._confirmed > 0, _mask(BASE) | _mask(BIG))


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


QUAD = np.float32([[0, 0], [400, 0], [400, 300], [0, 300]])


@pytest.mark.lento
def test_worker_with_e4_reports_telemetry(e4, fast_worker):
    worker = ink_layers_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        for _ in range(3):
            _drive(worker, generation, _page([BASE]), QUAD)
        confirmed = worker._accumulator._confirmed
        assert np.array_equal(confirmed > 0, _mask(BASE))
        records = worker.telemetry_snapshot()["records"]
        assert records[-1]["ink_vote_e4_passes"] == 3
        assert records[-1]["ink_vote_e4_tracks_big"] == 0
        assert records[-1]["ink_vote_e4_tracks_sub"] == 0
    finally:
        worker.stop()


@pytest.mark.lento
def test_worker_without_e4_has_no_e4_telemetry(off, fast_worker):
    worker = ink_layers_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = worker.new_generation()
        for _ in range(3):
            _drive(worker, generation, _page([BASE]), QUAD)
        records = worker.telemetry_snapshot()["records"]
        assert not any(
            key.startswith("ink_vote_e4") for record in records for key in record
        )
    finally:
        worker.stop()
