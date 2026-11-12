"""Corte de latencia voto->canvas (frente 94): DRAW_INK_VOTE_FAST.

A medicao da F94 (draw-lab/live-audit-2026-11-10/94): a tinta
forte ja e visivel na 1a passada nos dois bracos (provisoria),
mas confirma na 2a (regra atual) ou 3a/5a (E3); a tinta fraca
canonica (0,55) nao e visivel antes de confirmar — na regra
atual surge confirmada na 3a passada; na E3 nunca surge
(componente <= limiar nao abre trilha). O corte entregue,
atras de DRAW_INK_VOTE_FAST=1 (padrao desligado, lido uma vez
por geracao junto com o modo E3), e so na camada provisoria
(exibicao): a evidencia de hit acumulada (>= 2 hits, soma
>= 1,0) pinta a tinta fraca na 2a passada nos dois bracos,
sem confirmar um pixel a mais — o canvas de poll() nao muda.

O corte de confirmacao (regra forte de 2 passadas na E3) foi
implementado, medido e REPROVADO na regua nesta frente
(crescimento 1,216-1,554x contra o portao de 1,15x): os
testes abaixo travam que, com a flag ligada, as latencias de
CONFIRMACAO da E3 sao exatamente as da E3 sem o corte.

Com as duas flags desligadas, comportamento e chaves do
resultado sao os de antes da F94.
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
E3_ENV = "DRAW_INK_VOTE_E3"
FAST_ENV = "DRAW_INK_VOTE_FAST"


class _ProbIsWarp:
    """page_clean falso: o "warp" submetido ja e o mapa."""

    threshold = 0.7

    def ink_prob(self, warped):
        return warped


@pytest.fixture
def fast_worker():
    workers = []

    def make():
        worker = ink_layers_module._PageCleanWorker(_ProbIsWarp()).start()
        workers.append(worker)
        return worker

    yield make
    for worker in workers:
        worker.stop()


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setenv(FAST_ENV, "1")
    monkeypatch.delenv(E3_ENV, raising=False)
    return monkeypatch


@pytest.fixture
def e3_fast(monkeypatch):
    monkeypatch.setenv(E3_ENV, "1")
    monkeypatch.setenv(FAST_ENV, "1")
    return monkeypatch


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(FAST_ENV, raising=False)
    monkeypatch.delenv(E3_ENV, raising=False)
    return monkeypatch


def _acc():
    return ink_layers_module._LayeredInkAccumulator(threshold=0.7)


def _page(blocks=()):
    prob = np.zeros(SHAPE, dtype=np.float32)
    for (y0, y1, x0, x1), value in blocks:
        prob[y0:y1, x0:x1] = value
    return prob


BASE = ((20, 120, 20, 80), 0.95)  # 100x60 = 6.000 px
BIG = ((150, 190, 150, 190), 0.97)  # 40x40 = 1.600 px (>= 100)
SMALL = ((150, 158, 150, 158), 0.97)  # 8x8 = 64 px (trilha lenta)
WEAK = ((150, 190, 150, 190), 0.55)  # lapis fraco canonico


def _mask(block):
    (y0, y1, x0, x1), _value = block
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _fraction(mask2d, block):
    mask = _mask(block)
    return float(np.count_nonzero(mask2d & mask) / np.count_nonzero(mask))


def _visible_fraction(result, block):
    visible = (result["confirmed"] > 0) | (result["provisional"] > 0)
    return _fraction(visible, block)


def _confirmed_fraction(result, block):
    return _fraction(result["confirmed"] > 0, block)


def _latencies(acc, block, max_passes=8):
    """(visivel, confirmado) do bloco novo ate >=90%, ou None."""
    page = _page([BASE, block])
    lat_v = lat_c = None
    for k in range(1, max_passes + 1):
        result = acc.add(page)
        if lat_v is None and _visible_fraction(result, block) >= 0.9:
            lat_v = k
        if lat_c is None and _confirmed_fraction(result, block) >= 0.9:
            lat_c = k
    return lat_v, lat_c


def _establish(acc, block=BASE, passes=4):
    result = None
    for _ in range(passes):
        result = acc.add(_page([block]))
    return result


# --- flag -------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("0", False), ("2", False), ("1", True), (" On ", True)],
)
def test_flag_is_explicit_opt_in(monkeypatch, value, expected):
    monkeypatch.delenv(FAST_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(FAST_ENV, value)
    assert ink_layers_module._ink_vote_fast_enabled() is expected


def test_constants_are_the_f94_values():
    m = ink_layers_module
    assert m._INK_VOTE_FAST_PROV_MIN_HITS == 2
    assert m._INK_VOTE_FAST_PROV_MIN_SCORE == 1.0


# --- desligada: nada muda ----------------------------------------------


def test_off_has_no_fast_key_and_current_latencies(off):
    acc = _acc()
    _establish(acc)
    assert _latencies(acc, BIG) == (1, 2)
    acc = _acc()
    _establish(acc)
    # Fraca canonica: invisivel ate confirmar na 3a (visivel ==
    # confirmado) — o comportamento que a F94 mediu e o corte muda.
    assert _latencies(acc, WEAK) == (3, 3)
    result = acc.add(_page([BASE]))
    assert not any(key.startswith("ink_vote_fast") for key in result)


def test_off_e3_latencies_unchanged(monkeypatch):
    monkeypatch.setenv(E3_ENV, "1")
    monkeypatch.delenv(FAST_ENV, raising=False)
    acc = _acc()
    _establish(acc, passes=3)
    assert _latencies(acc, BIG) == (1, 3)
    acc = _acc()
    _establish(acc, passes=3)
    assert _latencies(acc, SMALL) == (1, 5)
    acc = _acc()
    _establish(acc, passes=3)
    assert _latencies(acc, WEAK) == (None, None)


# --- braco default com o corte: so a provisoria anda --------------------


def test_fast_default_weak_visible_one_pass_earlier(fast):
    acc = _acc()
    _establish(acc)
    assert _latencies(acc, WEAK) == (2, 3)


def test_fast_default_confirmed_canvas_byte_identical(fast, monkeypatch):
    """O corte no braco default e so de exibicao: a confirmada de
    uma sequencia mista e byte-identica com e sem a flag."""
    sequence = [
        _page([BASE]),
        _page([BASE, WEAK]),
        _page([BASE, WEAK]),
        _page([BASE, WEAK, BIG]),
        _page([BASE, BIG]),
        _page([BASE]),
    ]
    acc_on = _acc()
    results_on = [acc_on.add(prob) for prob in sequence]
    monkeypatch.delenv(FAST_ENV, raising=False)
    acc_off = _acc()
    results_off = [acc_off.add(prob) for prob in sequence]
    for on, off_result in zip(results_on, results_off):
        assert np.array_equal(on["confirmed"], off_result["confirmed"])
    # E a provisoria da tinta fraca divergiu (o ponto do corte):
    assert not np.array_equal(
        results_on[2]["provisional"], results_off[2]["provisional"]
    )
    assert results_on[2]["ink_vote_fast"] is True
    assert "ink_vote_fast" not in results_off[2]


def test_fast_default_strong_latency_unchanged(fast):
    acc = _acc()
    _establish(acc)
    assert _latencies(acc, BIG) == (1, 2)


def test_fast_provisional_needs_two_hits(fast):
    """Uma passada fraca isolada nao pinta provisoria nova: o
    corte pede a evidencia acumulada de 2 hits, nao 1."""
    acc = _acc()
    _establish(acc)
    result = acc.add(_page([BASE, WEAK]))
    assert _visible_fraction(result, WEAK) == 0.0
    result = acc.add(_page([BASE]))
    assert _visible_fraction(result, WEAK) == 0.0


# --- braco E3 com o corte: confirmacao intocada, provisoria anda -------


def test_e3_fast_confirmation_latencies_are_e3s_own(e3_fast):
    """O corte nao confirma mais cedo na E3 (a variante de
    confirmacao rapida foi reprovada na regua — relatorio 94):
    fase 1 fecha na 3a, trilha grande na 3a, pequena na 5a."""
    acc = _acc()
    first = acc.add(_page([BASE]))
    second = acc.add(_page([BASE]))
    assert first["confirmed_px"] == 0
    assert second["confirmed_px"] == 0
    third = acc.add(_page([BASE]))
    assert np.array_equal(third["confirmed"] > 0, _mask(BASE))
    acc = _acc()
    _establish(acc, passes=3)
    assert _latencies(acc, BIG) == (1, 3)
    acc = _acc()
    _establish(acc, passes=3)
    assert _latencies(acc, SMALL) == (1, 5)


def test_e3_fast_confirmed_canvas_byte_identical(e3_fast, monkeypatch):
    sequence = [
        _page([BASE]),
        _page([BASE, WEAK]),
        _page([BASE, WEAK]),
        _page([BASE, WEAK, BIG]),
        _page([BASE, BIG]),
        _page([BASE]),
    ]
    acc_on = _acc()
    results_on = [acc_on.add(prob) for prob in sequence]
    monkeypatch.delenv(FAST_ENV, raising=False)
    acc_off = _acc()
    results_off = [acc_off.add(prob) for prob in sequence]
    for on, off_result in zip(results_on, results_off):
        assert np.array_equal(on["confirmed"], off_result["confirmed"])


def test_e3_fast_weak_visible_but_never_confirmed(e3_fast):
    """Na E3 a tinta fraca nao confirma nem com o corte (a
    confirmacao segue exigindo componente acima do limiar) —
    mas fica VISIVEL como provisoria da 2a passada em diante,
    em vez de nunca aparecer."""
    acc = _acc()
    _establish(acc, passes=3)
    assert _latencies(acc, WEAK) == (2, None)


def test_mode_is_latched_per_generation(fast, monkeypatch):
    acc = _acc()
    _establish(acc)
    acc.add(_page([BASE, WEAK]))
    monkeypatch.delenv(FAST_ENV, raising=False)
    result = acc.add(_page([BASE, WEAK]))
    # A geracao decidiu com a flag ligada: a provisoria rapida
    # segue valendo mesmo com a variavel ja desligada.
    assert result["ink_vote_fast"] is True
    assert _visible_fraction(result, WEAK) >= 0.9
    acc.reset()
    fresh = acc.add(_page([BASE]))
    assert "ink_vote_fast" not in fresh


# --- trabalhadora -------------------------------------------------------


def _drive(worker, generation, prob):
    before = worker.stats()["passes_total"]
    worker.submit(prob, generation)
    deadline = time.monotonic() + 30.0
    while worker.stats()["passes_total"] <= before:
        assert time.monotonic() < deadline
        time.sleep(0.002)
    assert worker.error is None


def test_worker_reports_fast_telemetry_only_when_on(e3_fast, fast_worker):
    worker = fast_worker()
    generation = worker.new_generation()
    for _ in range(3):
        _drive(worker, generation, _page([BASE]))
    record = worker.telemetry_snapshot()["records"][-1]
    assert record["ink_vote_fast"] is True
    # Fase 1 da E3, intocada pelo corte: o nucleo fecha na 3a.
    assert np.count_nonzero(worker._accumulator._confirmed) == 6000


def test_worker_without_fast_has_no_fast_telemetry(off, fast_worker):
    worker = fast_worker()
    generation = worker.new_generation()
    _drive(worker, generation, _page([BASE]))
    record = worker.telemetry_snapshot()["records"][-1]
    assert "ink_vote_fast" not in record
