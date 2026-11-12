"""Testes da drenagem da fila do driver de draw/capture.py (frente 95).

O defeito: o slot unico de _LatestFrameCapture resolve a fila do
lado do app, mas o read() desempilha FIFO a fila do DRIVER
(V4L2/usbipd, profundidade 4 medida na mesa do Lucas em
2026-10-31: ate 200-300 ms de quadro velho). Quando a thread de
captura atrasa — decode MJPG dentro do read(), contencao com o
laco — ela serve a rajada empilhada do mais velho pro mais novo,
um decode por vez, e o consumidor acorda no primeiro da rajada.

O conserto atras de DRAW_CAPTURE_DRAIN_LATEST=1: grab() continuo
drena a fila do driver sem decode; retrieve() (o decode) so roda
pro quadro que vai ser publicado, com o slot livre. Estes testes
usam uma fonte falsa que MODELA o driver: produtor em thread
propria, fila FIFO com profundidade, read() = desempilha + decode,
grab() = desempilha barato, retrieve() = decode do agarrado. O
modelo e o da documentacao do OpenCV (grab transfere sem
decodificar; retrieve decodifica o ultimo grab) — o que ele nao
prova e o comportamento do backend real via usbipd; isso so a
mesa diz (relatorio 95).

Uso: python tests/test_capture_drain.py
"""

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.capture import (  # noqa: E402
    _CAPTURE_MAX_FAILURES,
    _LatestFrameCapture,
    capture_drain_latest_enabled,
)

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"ok - {name}")
    else:
        print(f"FALHOU - {name} {detail}")
        FAILURES.append(name)


def _now_ms():
    return int(time.monotonic() * 1000)


def _encode_frame(seq):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[0, 0, 0] = seq & 0xFF
    frame[0, 1, 0] = (seq >> 8) & 0xFF
    frame[0, 2, 0] = (seq * 7) & 0xFF
    frame[4:, :, :] = seq % 256
    return frame


def _decode_seq(frame):
    seq = int(frame[0, 0, 0]) | (int(frame[0, 1, 0]) << 8)
    assert int(frame[0, 2, 0]) == (seq * 7) & 0xFF, "quadro corrompido"
    return seq


class _DriverQueueCapture:
    """Fonte falsa com a semantica do driver + OpenCV.

    Um produtor poe quadros numa FIFO de profundidade `depth`
    (cheia, o quadro novo e descartado pelo "driver", como no
    videobuf2 sem buffer livre). read() desempilha o mais velho e
    paga `decode_s`. grab() desempilha o mais velho pro slot
    interno pagando `grab_s` (sem decode); retrieve() paga
    `decode_s` e devolve o slot. `stall_every`/`stall_s` injetam
    um atraso de conclusao de servico a cada N chamadas (a chamada
    desempilha e so devolve `stall_s` depois — transferencia
    usbipd irregular, thread desescalonada): enquanto o servico
    atrasa, a camera segue produzindo e a fila do driver enche.
    O bloqueio DENTRO da chamada so acontece com a fila vazia,
    como no V4L2: e essa a semantica que a heuristica de frescor
    da drenagem le (grab imediato = havia fila; grab que bloqueou
    = quadro recem-chegado).
    """

    def __init__(
        self,
        period_s,
        depth=4,
        decode_s=0.012,
        grab_s=0.001,
        total=None,
        stall_every=0,
        stall_s=0.0,
    ):
        self._period = period_s
        self._depth = depth
        self._decode_s = decode_s
        self._grab_s = grab_s
        self._total = total
        self._stall_every = stall_every
        self._stall_s = stall_s
        self._cond = threading.Condition()
        self._queue = deque()
        self._slot = None
        self._released = False
        self.produced = 0
        self.produced_ms = {}
        self.device_dropped = 0
        self.read_calls = 0
        self.grab_calls = 0
        self.retrieve_calls = 0
        self._service_calls = 0
        self._producer = threading.Thread(target=self._produce, daemon=True)
        self._producer.start()

    def _produce(self):
        # Cadencia de sensor: um quadro por periodo, sem rajada de
        # compensacao — se a thread acorda tarde, o quadro sai
        # tarde e o proximo espera um periodo inteiro (uma camera
        # nao "recupera" quadros perdidos em rajada).
        while not self._released:
            if self._total is not None and self.produced >= self._total:
                time.sleep(0.01)
                continue
            time.sleep(self._period)
            with self._cond:
                if self._released:
                    return
                self.produced += 1
                seq = self.produced
                if len(self._queue) >= self._depth:
                    self.device_dropped += 1
                else:
                    self._queue.append(seq)
                    self.produced_ms[seq] = _now_ms()
                self._cond.notify_all()

    def _service_stall(self):
        # Chamado DEPOIS do desempilhar: a chamada conclui tarde.
        self._service_calls += 1
        if self._stall_every and self._service_calls % self._stall_every == 0:
            time.sleep(self._stall_s)

    def _take(self):
        with self._cond:
            while not self._queue and not self._released:
                self._cond.wait(timeout=0.05)
            if not self._queue:
                return None
            return self._queue.popleft()

    def read(self):
        self.read_calls += 1
        seq = self._take()
        if seq is None:
            return False, None
        time.sleep(self._decode_s)
        self._service_stall()
        return True, _encode_frame(seq)

    def grab(self):
        self.grab_calls += 1
        seq = self._take()
        if seq is None:
            return False
        time.sleep(self._grab_s)
        self._service_stall()
        self._slot = seq
        return True

    def retrieve(self):
        self.retrieve_calls += 1
        if self._slot is None:
            return False, None
        time.sleep(self._decode_s)
        return True, _encode_frame(self._slot)

    def release(self):
        self._released = True
        with self._cond:
            self._cond.notify_all()


class _ReadOnlyCapture:
    """Fonte sem grab/retrieve: a drenagem tem de cair pro read()."""

    def __init__(self, period_s):
        self._period = period_s
        self.produced = 0
        self.released = False

    def read(self):
        time.sleep(self._period)
        self.produced += 1
        return True, _encode_frame(self.produced)

    def release(self):
        self.released = True


class _GrabFailingCapture:
    def __init__(self):
        self.released = False

    def grab(self):
        return False

    def retrieve(self):
        return False, None

    def release(self):
        self.released = True


class _GrabFlakyCapture(_DriverQueueCapture):
    """grab() falha as primeiras chamadas e depois drena normal."""

    def __init__(self, failures, **kwargs):
        self._flaky_left = failures
        super().__init__(**kwargs)

    def grab(self):
        if self._flaky_left > 0:
            self._flaky_left -= 1
            return False
        return super().grab()


def _process_seconds(consumed_index):
    # O mesmo custo simulado do cenario de fila da frente original:
    # 127 ms a cada 5 quadros (deteccao) e 12 ms nos demais.
    return 0.127 if consumed_index % 5 == 4 else 0.012


def _consume(source, device, total_frames, process=_process_seconds):
    ages = []
    seqs = []
    last_seq = 0
    consumed = 0
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        item = source.wait_next(last_seq, timeout=10.0)
        if item is None:
            break
        frame, _capture_ms, last_seq = item
        seq = _decode_seq(frame)
        seqs.append(seq)
        ages.append(_now_ms() - device.produced_ms[seq])
        if process is not None:
            time.sleep(process(consumed))
        consumed += 1
        if seq >= total_frames and device.produced >= total_frames:
            # O ultimo produzido pode ainda nao ter sido o consumido
            # se quadros foram descartados; o fim e por tempo abaixo.
            pass
        if consumed >= total_frames:
            break
    return ages, seqs


def _stats(ages_ms):
    ages = np.array(ages_ms, dtype=float)
    return {
        "n": int(ages.size),
        "median": float(np.median(ages)),
        "p95": float(np.percentile(ages, 95)),
        "max": float(ages.max()),
    }


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("DRAW_CAPTURE_DRAIN_LATEST", raising=False)
    assert not capture_drain_latest_enabled()
    device = _DriverQueueCapture(period_s=0.01, total=30)
    source = _LatestFrameCapture(device).start()
    try:
        assert not source.drain_active
        item = source.wait_next(0, timeout=5.0)
        assert item is not None
        assert device.grab_calls == 0, "sem a flag, grab() nunca e chamado"
        assert device.read_calls > 0
        stats = source.stats()
        assert stats["capture_drain_active"] == 0
        assert stats["capture_drained"] == 0
    finally:
        source.stop()
        device.release()


def test_flag_on_falls_back_without_grab(monkeypatch):
    monkeypatch.setenv("DRAW_CAPTURE_DRAIN_LATEST", "1")
    assert capture_drain_latest_enabled()
    capture = _ReadOnlyCapture(period_s=0.005)
    source = _LatestFrameCapture(capture).start()
    try:
        assert not source.drain_active, "sem grab/retrieve cai pro read()"
        item = source.wait_next(0, timeout=5.0)
        assert item is not None
    finally:
        source.stop()


@pytest.mark.lento
def test_drain_latest_reduces_frame_age():
    total = 240
    results = {}
    for drain in (False, True):
        device = _DriverQueueCapture(
            period_s=1.0 / 30.0,
            depth=4,
            decode_s=0.012,
            grab_s=0.001,
            total=total,
            stall_every=45,
            stall_s=0.150,
        )
        source = _LatestFrameCapture(device, drain_latest=drain).start()
        # Consumidor roda ate a producao acabar e a fila escoar:
        # consome por um tempo fixo de bancada, como no cenario
        # original, e mede a idade dos quadros que pegou.
        ages, seqs = [], []
        last_seq = 0
        consumed = 0
        end_at = time.monotonic() + total / 30.0 + 2.0
        while time.monotonic() < end_at:
            item = source.wait_next(last_seq, timeout=0.5)
            if item is None:
                if device.produced >= total and consumed > 0:
                    break
                continue
            frame, _ms, last_seq = item
            seq = _decode_seq(frame)
            seqs.append(seq)
            ages.append(_now_ms() - device.produced_ms[seq])
            time.sleep(_process_seconds(consumed))
            consumed += 1
        results[drain] = {
            "ages": _stats(ages),
            "seqs": seqs,
            "drained": source.drained,
            "dropped": source.dropped,
            "grabbed": source.grabbed,
            "retrieves": device.retrieve_calls,
            "reads": device.read_calls,
        }
        source.stop()
        device.release()

    old, new = results[False], results[True]
    print()
    print(
        "read():     "
        f"n={old['ages']['n']} mediana={old['ages']['median']:.0f} ms "
        f"p95={old['ages']['p95']:.0f} ms max={old['ages']['max']:.0f} ms"
    )
    print(
        "drenagem:   "
        f"n={new['ages']['n']} mediana={new['ages']['median']:.0f} ms "
        f"p95={new['ages']['p95']:.0f} ms max={new['ages']['max']:.0f} ms "
        f"drenados={new['drained']} grabs={new['grabbed']} "
        f"retrieves={new['retrieves']}"
    )
    check(
        "caminho read() expoe a classe de idade da fila (max > 150 ms)",
        old["ages"]["max"] > 150,
        f"max={old['ages']['max']:.0f}",
    )
    check(
        "drenagem tira a idade da classe 200-300 ms (p95 < 100 ms)",
        new["ages"]["p95"] < 100,
        f"p95={new['ages']['p95']:.0f}",
    )
    check(
        "idade maxima da drenagem e menos da metade da do read()",
        new["ages"]["max"] < 0.5 * old["ages"]["max"],
        f"new={new['ages']['max']:.0f} old={old['ages']['max']:.0f}",
    )
    check(
        "drenagem conta os quadros aposentados no driver",
        new["drained"] > 0,
        f"drained={new['drained']}",
    )
    check(
        "drenagem decodifica ~so o que publica (retrieves <= consumidos + 1)",
        new["retrieves"] <= new["ages"]["n"] + 1,
        f"retrieves={new['retrieves']} n={new['ages']['n']}",
    )
    check(
        "sequencias consumidas sao estritamente crescentes nos dois modos",
        all(b > a for a, b in zip(old["seqs"], old["seqs"][1:], strict=False))
        and all(b > a for a, b in zip(new["seqs"], new["seqs"][1:], strict=False)),
    )


@pytest.mark.lento
def test_drain_slow_arrival_loses_nothing():
    # Chegada bem mais lenta que o consumo (5 fps contra consumo de
    # 2 ms): nenhum quadro e velho por construcao e nenhum pode
    # faltar — drenar aqui seria defeito, nao otimizacao.
    total = 12
    for drain in (False, True):
        device = _DriverQueueCapture(period_s=0.2, depth=4, total=total)
        source = _LatestFrameCapture(device, drain_latest=drain).start()
        ages, seqs = _consume(source, device, total, process=lambda _i: 0.002)
        source.stop()
        device.release()
        check(
            f"chegada lenta (drain={drain}): todos os quadros, em ordem",
            seqs == list(range(1, total + 1)),
            f"seqs={seqs}",
        )
        if drain:
            check(
                "chegada lenta: nada drenado, nada descartado no slot",
                source.drained == 0 and source.dropped == 0,
                f"drained={source.drained} dropped={source.dropped}",
            )


@pytest.mark.lento
def test_drain_grab_failure_limit_marks_dead():
    capture = _GrabFailingCapture()
    source = _LatestFrameCapture(capture, drain_latest=True).start()
    deadline = time.monotonic() + 5.0
    while not source.dead and time.monotonic() < deadline:
        time.sleep(0.01)
    check("grab falho seguido declara a fonte morta", source.dead)
    check(
        "contagem de falhas para no limite",
        source.failures == _CAPTURE_MAX_FAILURES,
        f"failures={source.failures}",
    )
    check("quem espera acorda com None", source.wait_next(0, timeout=1.0) is None)
    source.stop()


@pytest.mark.lento
def test_drain_transient_grab_failures_recover():
    device = _GrabFlakyCapture(failures=3, period_s=0.01, total=30)
    source = _LatestFrameCapture(device, drain_latest=True).start()
    item = source.wait_next(0, timeout=5.0)
    check("falha transitoria de grab nao impede o quadro", item is not None)
    check("fonte nao morreu", not source.dead)
    check("leitura boa zera as falhas", source.failures == 0)
    source.stop()
    device.release()


if __name__ == "__main__":

    class _MonkeyPatch:
        def __init__(self):
            self._saved = {}

        def setenv(self, name, value):
            import os

            self._saved[name] = os.environ.get(name)
            os.environ[name] = value

        def delenv(self, name, raising=True):
            import os

            self._saved[name] = os.environ.get(name)
            os.environ.pop(name, None)

        def undo(self):
            import os

            for name, value in self._saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    mp = _MonkeyPatch()
    test_flag_default_off(mp)
    mp.undo()
    mp = _MonkeyPatch()
    test_flag_on_falls_back_without_grab(mp)
    mp.undo()
    test_drain_latest_reduces_frame_age()
    test_drain_slow_arrival_loses_nothing()
    test_drain_grab_failure_limit_marks_dead()
    test_drain_transient_grab_failures_recover()
    print()
    if FAILURES:
        print(f"FALHARAM {len(FAILURES)} verificacoes: {FAILURES}")
        sys.exit(1)
    print("TODOS OS TESTES DE DRENAGEM OK.")
