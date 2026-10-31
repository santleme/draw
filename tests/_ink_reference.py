"""Implementação de referência (float32 + carimbos densos) do InkAccumulator.

Mantida dentro de tests/ para que as regressões diferenciais
(test_accumulator_stamps.py e test_accumulator_integer_sparse.py)
rodem de forma autocontida em qualquer clone limpo ou ambiente de CI,
sem depender de cópias temporárias em /tmp ou ~/workspace/draw-lab.

Reproduz linha por linha o comportamento anterior a e0b8f36:
fração em float32 e limpeza densa dos planos de carimbo.
"""

from collections import deque
import time

import cv2
import numpy as np

from draw import config


class InkAccumulator:
    """Referência canônica em float32 para testes diferenciais."""

    def __init__(
        self,
        shape=(config.PAPER_H, config.PAPER_W),
        window=15,
        enter=0.45,
        exit=0.20,
        tolerance=1,
        min_visible=4,
    ):
        self._window = window
        self._enter = enter
        self._exit = exit
        self._min_visible = min_visible
        size = 2 * tolerance + 1
        self._kernel = np.ones((size, size), dtype=np.uint8)
        self._shape = shape
        self.reset()

    def reset(self):
        self._ink_votes = deque()
        self._vis_votes = deque()
        self._ink_counts = np.zeros(self._shape, dtype=np.int16)
        self._vis_counts = np.zeros(self._shape, dtype=np.int16)
        self.state = np.zeros(self._shape, dtype=bool)
        self._first_seen_ms = np.zeros(self._shape, dtype=np.uint32)
        self._last_update_ms = np.zeros(self._shape, dtype=np.uint32)

    def first_seen_ms(self):
        return self._first_seen_ms.copy()

    def last_update_ms(self):
        return self._last_update_ms.copy()

    def update(self, ink_mask, visible_mask=None, t_capture_ms=None):
        if t_capture_ms is None:
            t_ms = int(time.monotonic() * 1000)
        else:
            t_ms = int(t_capture_ms)
        ink = ink_mask > 0
        if visible_mask is None:
            visible = np.ones(ink.shape, dtype=bool)
        elif visible_mask.dtype == bool:
            visible = visible_mask
        else:
            visible = visible_mask > 0
        evidence = (
            cv2.dilate(ink.astype(np.uint8), self._kernel).astype(bool)
            & visible
        )
        if len(self._ink_votes) == self._window:
            self._ink_counts -= self._ink_votes.popleft()
            self._vis_counts -= self._vis_votes.popleft()
        self._ink_votes.append(evidence)
        self._vis_votes.append(visible)
        self._ink_counts += evidence
        self._vis_counts += visible
        judged = self._vis_counts >= self._min_visible
        fraction = np.zeros(self._shape, dtype=np.float32)
        fraction[judged] = self._ink_counts[judged] / np.maximum(
            self._vis_counts[judged], 1
        )
        enter = judged & ~self.state & (fraction >= self._enter)
        clear = judged & self.state & (fraction <= self._exit)
        self.state[enter] = True
        self.state[clear] = False

        t_val = np.uint32(t_ms & 0xFFFFFFFF)
        ink_visible = self.state & visible
        fresh = ink_visible & (self._first_seen_ms == 0)
        self._first_seen_ms[fresh] = t_val
        self._last_update_ms[fresh] = t_val
        renewed = ink_visible & evidence
        self._last_update_ms[renewed] = t_val
        gone = visible & ~self.state
        self._first_seen_ms[gone] = 0
        self._last_update_ms[gone] = 0
        return (self.state * 255).astype(np.uint8)
