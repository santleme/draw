"""Cadencia por atividade da tinta (frente 92): a sonda ociosa nao
governa enquanto a saida esta mudando.

O defeito medido (sessao viva 1794443693): com o sinal de presenca
dizendo "sem mao", _pass_interval devolvia _IDLE_PROBE_SECONDS em
93 de 102 intervalos — 1 passada/s — enquanto a confirmada crescia
em quase toda passada. A sonda pressupoe "sem mao, sem tinta
nova"; a saida desmentia o pressuposto e o escalonador nao ouvia a
saida. Atras de DRAW_INK_ACTIVE_CADENCE (default OFF):

1. Flag desligada: sem mao, tinta mudando ou nao, a cadencia e a
   da sonda — comportamento de sempre, byte a byte.
2. Flag ligada + sem mao + saida mudando nas ultimas
   _NO_HAND_REST_AFTER_STILL_PASSES passadas: cadencia ativa.
3. Flag ligada + saida assentada: a sonda volta, e o descanso
   total segue intacto (zero passadas).
4. Passada cuja geracao trocou entre a retirada e a execucao
   morre antes do modelo (o descarte tardio ja a jogava fora
   depois de pagar o modelo inteiro).
5. O registro de telemetria carrega o intervalo que governou a
   espera, o estado de mao na retirada e a quietude da saida.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink_layers_module  # noqa: E402

H, W = 40, 60


class _ShiftingInk:
    """Tinta falsa que imita o desenho real: cada passada ve tudo
    que ja foi desenhado mais uma celula nova (a tinta fica no
    papel), entao a confirmada cresce passada a passada — o regime
    da sessao medida. moving=False congela o mapa."""

    threshold = 0.7

    _CELL = 4
    _COLS = W // _CELL

    def __init__(self):
        self.calls = 0
        self.moving = True
        self._cells = 0

    def ink_prob(self, _warped):
        self.calls += 1
        if self.moving:
            self._cells += 1
        prob = np.zeros((H, W), dtype=np.float32)
        for index in range(self._cells):
            row, col = divmod(index, self._COLS)
            y, x = row * self._CELL, col * self._CELL
            prob[y : y + self._CELL, x : x + self._CELL] = 0.9
        return prob


def _feed(worker, generation, seconds):
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        worker.submit(warp, generation)
        time.sleep(0.005)


def _patched(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.30)
    monkeypatch.setattr(
        ink_layers_module, "_NO_HAND_REST_AFTER_STILL_PASSES", 3
    )


@pytest.mark.lento
def test_activity_cadence_off_keeps_probe_without_hand(monkeypatch):
    monkeypatch.delenv("DRAW_INK_ACTIVE_CADENCE", raising=False)
    _patched(monkeypatch)
    fake = _ShiftingInk()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        generation = worker.new_generation()
        worker.set_hand_presence(False)
        # A confirmacao da saida da mao roda; depois, mesmo com a
        # tinta mudando a cada passada, a cadencia e a da sonda.
        _feed(worker, generation, 0.25)
        before = fake.calls
        _feed(worker, generation, 0.65)
        probed = fake.calls - before
        assert probed <= 3, (
            f"flag desligada: {probed} passadas em 0,65 s sem mao; "
            "a sonda (0,30 s) admite no maximo 3"
        )
    finally:
        worker.stop()


@pytest.mark.lento
def test_activity_cadence_on_runs_active_while_output_changes(monkeypatch):
    monkeypatch.setenv("DRAW_INK_ACTIVE_CADENCE", "1")
    _patched(monkeypatch)
    fake = _ShiftingInk()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        generation = worker.new_generation()
        worker.set_hand_presence(False)
        _feed(worker, generation, 0.25)
        before = fake.calls
        _feed(worker, generation, 0.65)
        active = fake.calls - before
        assert active >= 6, (
            f"flag ligada, sem mao e com a saida mudando: {active} "
            "passadas em 0,65 s; a cadencia ativa (0,02 s) pede >= 6"
        )
    finally:
        worker.stop()


@pytest.mark.lento
def test_activity_cadence_on_settles_back_to_probe(monkeypatch):
    monkeypatch.setenv("DRAW_INK_ACTIVE_CADENCE", "1")
    _patched(monkeypatch)
    fake = _ShiftingInk()
    worker = app_module._PageCleanWorker(fake).start()
    try:
        generation = worker.new_generation()
        worker.set_hand_presence(False)
        _feed(worker, generation, 0.4)
        # A tinta para de mudar, mas a pagina NAO assenta para o
        # descanso total (a provisoria segue presente e o voto nao
        # fecha quietude de confirmada neste material): esgotada a
        # janela de atividade, a sonda ociosa volta a governar.
        fake.moving = False
        _feed(worker, generation, 0.5)
        before = fake.calls
        _feed(worker, generation, 0.65)
        probed = fake.calls - before
        assert probed <= 3, (
            f"saida assentada: {probed} passadas em 0,65 s; a sonda "
            "deveria ter voltado (maximo 3)"
        )
    finally:
        worker.stop()


def test_activity_cadence_rest_waits_for_provisional_quiet(monkeypatch):
    monkeypatch.setenv("DRAW_INK_ACTIVE_CADENCE", "1")
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.25)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 1.0)
    now = [2000.0]
    fake = _ShiftingInk()
    worker = app_module._PageCleanWorker(fake, clock=lambda: now[0])
    generation = worker.new_generation()
    worker.set_hand_presence(False)
    warp = np.zeros((H, W, 3), dtype=np.uint8)

    # Desenhando: a confirmada pode ate aquietar entre um traco e
    # outro, mas a provisoria muda a cada passada — nada de
    # descanso no meio do desenho.
    for _ in range(14):
        worker.submit(warp, generation)
        now[0] += 0.30
        worker.pump()
    assert fake.calls >= 8
    assert not worker.stats()["resting"], (
        "com a provisoria viva, o descanso total nao pode fechar"
    )

    # Congelou: as duas camadas aquietam, a janela de atividade
    # esgota e o descanso total volta a valer.
    fake.moving = False
    for _ in range(16):
        worker.submit(warp, generation)
        now[0] += 0.30
        worker.pump()
    assert worker.stats()["resting"], "pagina assentada deve descansar"
    calls = fake.calls
    worker.submit(warp, generation)
    now[0] += 2.0
    assert worker.pump() is False
    assert fake.calls == calls, "em descanso, submit nao vira passada"


def test_stale_generation_pass_skips_model(monkeypatch):
    monkeypatch.delenv("DRAW_INK_ACTIVE_CADENCE", raising=False)
    fake = _ShiftingInk()
    worker = app_module._PageCleanWorker(fake)
    generation = worker.new_generation()
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    worker.submit(warp, generation)
    with worker._condition:
        job = worker._take_pending_locked()
    worker.new_generation()
    assert worker._run_pass(job) is True
    assert fake.calls == 0, "passada de geracao morta nao paga o modelo"
    snapshot = worker.telemetry_snapshot()
    assert snapshot["passes_skipped_stale"] == 1
    assert snapshot["passes"] == 0


def test_telemetry_record_carries_interval_and_hand_state(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.25)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 1.0)
    now = [1000.0]
    fake = _ShiftingInk()
    worker = app_module._PageCleanWorker(fake, clock=lambda: now[0])
    generation = worker.new_generation()
    warp = np.zeros((H, W, 3), dtype=np.uint8)

    worker.submit(warp, generation)
    assert worker.pump() is True
    worker.submit(warp, generation)
    now[0] += 0.10
    assert worker.pump() is False, "a cadencia segura a segunda passada"
    now[0] += 0.20
    assert worker.pump() is True

    records = worker.telemetry_snapshot()["records"]
    assert len(records) == 2
    assert records[0]["interval_s"] is None, "primeira passada nao tem espera"
    assert records[0]["hand_state"] is None
    assert records[1]["interval_s"] == 0.25
    assert records[1]["hand_state"] is None
    assert records[1]["queue_wait_s"] == pytest.approx(0.30)
    assert "activity_quiet_passes" in records[1]
