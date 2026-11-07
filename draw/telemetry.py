"""Telemetria de sessão do app: um registro por segundo, local.

A sessão ao vivo de 2026-11-01 não deixou artefato nenhum: o app
não gravava nada, e o relato de "a fricção aumenta durante a
sessão" ficou sem como testar. Esta classe grava, uma vez por
segundo, o retrato que o laço principal entrega com update():
fps do laço, tempos por estágio, contadores da trabalhadora de
tinta e tamanho do canvas. Quem lê decide se a fricção cresce e
onde ela mora.

O desenho tira o custo do caminho crítico. update() só guarda o
retrato mais recente sob um cadeado. Uma thread própria acorda
no intervalo, monta o registro (com os deltas de contadores
desde o registro anterior) e escreve a linha JSON. O laço nunca
faz IO por causa da telemetria.

O arquivo é local e rotativo: passou de max_bytes, ele vira
<nome>.1 e um novo começa. Nada sai da máquina.
"""

import json
import threading
import time
from collections import deque
from pathlib import Path


class SessionTelemetry:
    """Escritor de telemetria a 1 Hz em JSONL rotativo."""

    def __init__(self, path, interval=1.0, max_bytes=5_000_000):
        self._path = Path(path)
        self._interval = float(interval)
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._latest = {}
        self._events = deque()
        self._previous_counters = {}
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="draw-telemetry", daemon=True
        )
        self._error = None

    def start(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def update(self, **fields):
        """Guarda o retrato mais recente. Barato: um dict sob cadeado."""
        with self._lock:
            self._latest.update(fields)

    def event(self, name, **fields):
        """Enfileira um evento pontual; a escrita continua fora do laco."""
        record = {**fields, "event": name, "t_epoch": round(time.time(), 3)}
        with self._lock:
            self._events.append(record)

    def _take_events(self):
        with self._lock:
            records = list(self._events)
            self._events.clear()
        return records

    def _snapshot(self):
        with self._lock:
            record = dict(self._latest)
        record["t_epoch"] = round(time.time(), 3)
        for counter in ("passes_total", "retracted_px_total"):
            if counter in record:
                previous = self._previous_counters.get(counter)
                record[f"{counter}_delta"] = (
                    record[counter] if previous is None
                    else record[counter] - previous
                )
                self._previous_counters[counter] = record[counter]
        return record

    def _rotate_if_needed(self):
        try:
            if self._path.exists() and (
                self._path.stat().st_size >= self._max_bytes
            ):
                backup = self._path.with_name(self._path.name + ".1")
                self._path.replace(backup)
        except OSError:
            pass

    def _run(self):
        while not self._stopped.wait(self._interval):
            if not self._write(self._take_events() + [self._snapshot()]):
                return
        self._write(self._take_events())

    def _write(self, records):
        if not records:
            return True
        try:
            self._rotate_if_needed()
            with self._path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as error:
            self._error = error
            return False
        return True

    @property
    def error(self):
        """O erro de IO que parou o escritor, ou None."""
        return self._error

    def close(self):
        self._stopped.set()
        self._thread.join(timeout=2.0)
