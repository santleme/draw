"""Cadência ativa sem mão (frente 91, DRAW_INK_ACTIVE_CADENCE).

O contrato, com a mão reportada AUSENTE (hand_state False):

1. Flag desligada (padrão): sonda de _IDLE_PROBE_SECONDS, mesmo
   com a tinta mudando a cada passada — o comportamento de
   sempre, byte a byte.
2. Flag ligada e a última passada mudou a saída (provisória ou
   confirmada): cadência ativa de _PAGE_CLEAN_INTERVAL_SECONDS.
3. Flag ligada e a tinta para de mudar: volta a sondar devagar
   (e o descanso total, pela quietude da confirmada, segue
   podendo zerar as passadas — coberto pelos testes de descanso).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import ink_layers as ink_layers_module  # noqa: E402

H, W = 40, 200


class _ShiftingInk:
    """Tinta falsa: uma barra que CRESCE a cada passada (como um
    traço sendo desenhado: confirma sem parar, então a quietude
    da confirmada nunca fecha) ou parada (o mapa final, repetido).
    """

    threshold = 0.7

    def __init__(self):
        self.calls = 0
        self.moving = False
        self._frontier = 4

    def ink_prob(self, _warped):
        self.calls += 1
        if self.moving:
            self._frontier = min(W - 4, self._frontier + 3)
        prob = np.zeros((H, W), dtype=np.float32)
        prob[5:15, 2 : self._frontier] = 0.97
        return prob


def _feed(worker, generation, seconds):
    warp = np.zeros((H, W, 3), dtype=np.uint8)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        worker.submit(warp, generation)
        time.sleep(0.005)


def _worker(monkeypatch, fake):
    monkeypatch.setattr(
        ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.02
    )
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.30)
    worker = ink_layers_module._PageCleanWorker(fake).start()
    generation = worker.new_generation()
    worker.set_hand_presence(False)
    return worker, generation


@pytest.mark.lento
def test_active_cadence_off_keeps_idle_probe_with_moving_ink(monkeypatch):
    monkeypatch.delenv("DRAW_INK_ACTIVE_CADENCE", raising=False)
    fake = _ShiftingInk()
    fake.moving = True
    worker, generation = _worker(monkeypatch, fake)
    try:
        _feed(worker, generation, 0.35)  # aquece (confirm_once + 1a sonda)
        before = fake.calls
        _feed(worker, generation, 0.65)
        slow = fake.calls - before
        assert slow <= 3, f"flag desligada deveria sondar a 1 Hz: {slow}"
    finally:
        worker.stop()


@pytest.mark.lento
def test_active_cadence_on_speeds_up_while_ink_changes(monkeypatch):
    monkeypatch.setenv("DRAW_INK_ACTIVE_CADENCE", "1")
    fake = _ShiftingInk()
    fake.moving = True
    worker, generation = _worker(monkeypatch, fake)
    try:
        _feed(worker, generation, 0.35)
        before = fake.calls
        _feed(worker, generation, 0.65)
        fast = fake.calls - before
        assert fast >= 6, f"tinta mudando sem mao deveria acelerar: {fast}"

        # A tinta para: a cadência volta a cair (sonda ou descanso).
        fake.moving = False
        _feed(worker, generation, 0.40)
        before = fake.calls
        _feed(worker, generation, 0.65)
        slow = fake.calls - before
        assert slow <= 3, f"tinta parada deveria voltar a sondar: {slow}"
    finally:
        worker.stop()


def test_active_cadence_env_default_off():
    assert "DRAW_INK_ACTIVE_CADENCE" not in os.environ or (
        ink_layers_module._ink_active_cadence_enabled()
        == (os.environ["DRAW_INK_ACTIVE_CADENCE"].strip() == "1")
    )
