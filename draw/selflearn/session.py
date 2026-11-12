"""Coleta automatica de sessao (correcao 5 do loop v2).

A coleta mais simples que gera dado real rotulavel: uma sessao
com referencia + salvar no fim ja produz um par antes/depois sem
passar por modelo nenhum.

- No caminho padrao (page-clean), a referencia da geracao e o
  primeiro warp estavel dela; o par e (primeiro warp, warp final)
  mais o canvas final publicado.
- No caminho legado, apertar C abre geracao nova no gravador: a
  referencia e o warp do momento da tecla, como a pagina base.

O que grava, por sessao, em <raiz>/sessions/<session_id>/:

- gen_NNNN/reference_warp.png - primeiro warp da geracao
- gen_NNNN/final_warp.png      - ultimo warp da geracao
- gen_NNNN/final_canvas.png    - ultimo canvas da geracao
- raw/frame_NNNNNN.png          - quadro cru da camera (como chega,
  antes de warp/deteccao), esparso: 1 a cada _RAW_EVERY_N_FRAMES
  passadas do laco e em toda transicao de estado da folha,
  inclusive sem folha (no_sheet). NNNNNN e o indice da passada
- acquisitions.jsonl           - um registro por aquisicao de
  deteccao de folha (caminho combinado + rastreador antes/depois)
- manifest.json                - session_id, inicio/fim, sha256
  de cada arquivo gravado, sha256 dos arquivos de modelo
  (page-clean e folha), teclas com tempo relativo, telemetria
  de passadas do worker (U0), os contadores da sessao, a lista
  dos quadros crus (indice da passada, tempo, estado, sha256) e
  o resumo das aquisicoes

Teto de disco: os quadros crus param em _RAW_MAX_FRAMES arquivos
ou _RAW_MAX_BYTES bytes, o que vier antes, e o corte fica
registrado em manifest["raw_frames"]["truncated"]; as aquisicoes
param em _ACQ_MAX_RECORDS linhas, com o corte em
manifest["acquisitions"]["truncated"].

Privacidade: isto e LOCAL por desenho - a raiz padrao do app e
captures/, fora do git. Frame de mesa e dado pessoal: rodar a
coleta na mesa de outra pessoa exige o consentimento dela, e
nada desta pasta sobe para dataset nenhum sem a curadoria do
loop (divisao por sessao de split.py, holdout com rotulo
independente).

Custo no laco: as chamadas note_* so copiam o array (memcpy) e
guardam/substituem referencias; a escrita em disco acontece na
thread propria do gravador, fora do laco principal. note_raw_frame
so copia o quadro quando ele vai ser gravado, e descarta (contando)
quando ja ha _RAW_MAX_IN_FLIGHT quadros crus esperando a escrita.
"""

import hashlib
import json
import queue
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from ..safe_io import ensure_private_dir

_SCHEMA = 1
_DRAIN_SECONDS = 5.0

# Quadros crus esparsos (observabilidade da aquisicao, 2026-11-10):
# ~0,4 quadro/s a 12 fps; 600 quadros cobrem ~25 min de laco.
_RAW_EVERY_N_FRAMES = 30
_RAW_MAX_FRAMES = 600
_RAW_MAX_BYTES = 1 << 30
_RAW_MAX_IN_FLIGHT = 4
_ACQ_MAX_RECORDS = 200_000


def sha256_file(path):
    """sha256 de um arquivo, ou None se ele nao existe."""
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class SessionRecorder:
    """Gravador de sessoes: pares antes/depois por geracao.

    Uso no app: note_key() a cada tecla, note_warp() a cada warp
    submetido (o primeiro da geracao vira referencia, o ultimo
    vira final), note_canvas() a cada canvas novo, e finish() na
    saida com o retrato de telemetria do worker. start_generation
    so e necessario no caminho legado (tecla C); no padrao, a
    troca de geracao aparece sozinha no note_warp.
    """

    def __init__(
        self,
        root_dir,
        session_id=None,
        model_paths=None,
        clock=None,
        wall_clock=None,
        raw_every_n_frames=_RAW_EVERY_N_FRAMES,
        raw_max_frames=_RAW_MAX_FRAMES,
        raw_max_bytes=_RAW_MAX_BYTES,
        raw_max_in_flight=_RAW_MAX_IN_FLIGHT,
        acq_max_records=_ACQ_MAX_RECORDS,
    ):
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self.session_id = session_id or (
            f"{int(self._wall_clock())}-{uuid.uuid4().hex[:8]}"
        )
        self.session_dir = Path(root_dir) / "sessions" / self.session_id
        self._model_paths = dict(model_paths or {})
        self._started_wall = self._wall_clock()
        self._started_mono = self._clock()
        self._lock = threading.Lock()
        self._keys = []
        self._generations = {}
        self._finished = False
        self._errors = []
        self._raw_policy = {
            "every_n_frames": int(raw_every_n_frames),
            "on_state_transition": True,
            "max_frames": int(raw_max_frames),
            "max_bytes": int(raw_max_bytes),
            "max_in_flight": int(raw_max_in_flight),
        }
        self._raw_noted = 0
        self._raw_last_state = None
        self._raw_queued = 0
        self._raw_in_flight = 0
        self._raw_dropped_backlog = 0
        self._raw_bytes = 0
        self._raw_entries = []
        self._raw_truncated = None
        self._acq_max_records = int(acq_max_records)
        self._acq_queued = 0
        self._acq_dropped = 0
        self._ink_last_sequence = 0
        self._ink_records = 0
        self._ink_dropped = 0
        self._ink_handle = None
        self._acq_written = 0
        self._acq_truncated = None
        self._acq_handle = None
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._work, name="draw-session-recorder", daemon=True
        )
        self._thread.start()

    # -- entradas do laco (baratas: copia + troca de referencia) --

    def note_key(self, key):
        """Registra uma tecla com o tempo relativo ao inicio."""
        if isinstance(key, int):
            name = chr(key) if 32 <= key < 127 else f"key_{key}"
        else:
            name = str(key)
        with self._lock:
            self._keys.append(
                {
                    "t_s": round(self._clock() - self._started_mono, 3),
                    "key": name,
                }
            )

    def start_generation(self, generation):
        """Marca a geracao corrente (a referencia recomeca nela)."""
        with self._lock:
            self._generations.setdefault(int(generation), {})

    def note_warp(self, generation, warped_bgr):
        """Warp submetido: o 1o da geracao e referencia; vale o ultimo."""
        generation = int(generation)
        image = np.asarray(warped_bgr).copy()
        with self._lock:
            gen = self._generations.setdefault(generation, {})
            is_first = "reference" not in gen
            gen["final_warp"] = image
            if is_first:
                gen["reference"] = image
        if is_first:
            self._enqueue_image(generation, "reference_warp", image)

    def note_canvas(self, generation, canvas_bgr):
        """Canvas mais recente da geracao (o ultimo vale)."""
        with self._lock:
            gen = self._generations.setdefault(int(generation), {})
            gen["final_canvas"] = np.asarray(canvas_bgr).copy()

    def note_raw_frame(self, frame_bgr, frame_index, state, **context):
        """Quadro cru da passada frame_index; grava esparso.

        Chamado em toda passada do laco com o quadro como chegou
        da camera. Grava a primeira passada, 1 a cada
        every_n_frames e toda passada em que state muda (ex.:
        no_sheet -> candidate -> sheet). context (tempo de captura,
        aquisicao corrente, passadas da tinta) vai para a entrada
        do manifesto. Devolve True quando o quadro entrou na fila.
        """
        with self._lock:
            if self._finished or self._raw_truncated is not None:
                return False
            self._raw_noted += 1
            transition = (
                self._raw_last_state is not None and state != self._raw_last_state
            )
            previous_state = self._raw_last_state
            self._raw_last_state = state
            every = max(1, self._raw_policy["every_n_frames"])
            sampled = (self._raw_noted - 1) % every == 0
            if not (sampled or transition):
                return False
            t_s = round(self._clock() - self._started_mono, 3)
            if self._raw_queued >= self._raw_policy["max_frames"]:
                self._raw_truncated = {
                    "reason": "max_frames",
                    "frame_index": int(frame_index),
                    "t_s": t_s,
                }
                return False
            if self._raw_in_flight >= self._raw_policy["max_in_flight"]:
                self._raw_dropped_backlog += 1
                return False
            self._raw_queued += 1
            self._raw_in_flight += 1
        entry = {
            "frame_index": int(frame_index),
            "t_s": t_s,
            "t_epoch": round(self._wall_clock(), 3),
            "state": state,
            "reason": "transition" if transition else "sample",
            **context,
        }
        if transition:
            entry["previous_state"] = previous_state
        self._queue.put(("raw", entry, np.asarray(frame_bgr).copy()))
        return True

    def note_acquisition(self, record):
        """Registro de uma aquisicao de folha (dict JSON-serializavel)."""
        with self._lock:
            if self._finished:
                return False
            if self._acq_queued >= self._acq_max_records:
                self._acq_dropped += 1
                if self._acq_truncated is None:
                    self._acq_truncated = {
                        "reason": "max_records",
                        "t_s": round(self._clock() - self._started_mono, 3),
                        "acq_seq": record.get("acq_seq"),
                    }
                return False
            self._acq_queued += 1
            t_s = round(self._clock() - self._started_mono, 3)
        self._queue.put(("acq", {"t_s": t_s, **record}))
        return True

    def note_ink_telemetry(self, snapshot):
        """Persiste a régua por passada, sem perder o início no deque de 256."""
        for record in snapshot.get("records", []):
            sequence = record["sequence"]
            with self._lock:
                if self._finished or sequence <= self._ink_last_sequence:
                    continue
                if self._ink_records >= self._acq_max_records:
                    self._ink_dropped += sequence - self._ink_last_sequence
                    self._ink_last_sequence = sequence
                    continue
                self._ink_dropped += max(0, sequence - self._ink_last_sequence - 1)
                self._ink_last_sequence = sequence
                self._ink_records += 1
            self._queue.put(("ink", dict(record)))

    # -- fechamento --

    def finish(self, telemetry=None):
        """Grava finais + manifest.json e devolve o caminho dele.

        Idempotente: a segunda chamada devolve o mesmo caminho
        sem regravar. Erros de escrita da thread ficam em
        self.errors e o manifesto os registra.
        """
        with self._lock:
            if self._finished:
                return self.session_dir / "manifest.json"
            self._finished = True
            generations = {
                gen: dict(images) for gen, images in self._generations.items()
            }
            keys = list(self._keys)
        for generation, images in generations.items():
            if "final_warp" in images:
                self._enqueue_image(generation, "final_warp", images["final_warp"])
            if "final_canvas" in images:
                self._enqueue_image(generation, "final_canvas", images["final_canvas"])
        manifest = {
            "schema": _SCHEMA,
            "session_id": self.session_id,
            "started_at": self._started_wall,
            "ended_at": self._wall_clock(),
            "duration_s": round(self._clock() - self._started_mono, 3),
            "privacy": (
                "gravacao local; a mesa de outra pessoa exige o "
                "consentimento dela antes de coletar"
            ),
            "models": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in sorted(self._model_paths.items())
            },
            "keys": keys,
            "telemetry": telemetry,
            "generations": [],
            "raw_frames": None,
            "acquisitions": None,
        }
        self._queue.put(("manifest", manifest))
        self._queue.put(None)
        self._thread.join(timeout=_DRAIN_SECONDS)
        return self.session_dir / "manifest.json"

    @property
    def errors(self):
        with self._lock:
            return list(self._errors)

    # -- thread escritora --

    def _enqueue_image(self, generation, kind, image):
        self._queue.put(("image", generation, kind, image))

    def _generation_dir(self, generation):
        return self.session_dir / f"gen_{int(generation):04d}"

    def _work(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                if item[0] == "image":
                    _tag, generation, kind, image = item
                    folder = self._generation_dir(generation)
                    ensure_private_dir(self.session_dir)
                    folder.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(folder / f"{kind}.png"), image):
                        raise RuntimeError(f"imwrite devolveu False para {kind}")
                elif item[0] == "raw":
                    self._write_raw(item[1], item[2])
                elif item[0] == "acq":
                    self._write_acquisition(item[1])
                elif item[0] == "ink":
                    if self._ink_handle is None:
                        ensure_private_dir(self.session_dir)
                        self._ink_handle = open(
                            self.session_dir / "ink_passes.jsonl", "a", encoding="utf-8"
                        )
                    self._ink_handle.write(json.dumps(item[1], sort_keys=True) + "\n")
                    self._ink_handle.flush()
                elif item[0] == "manifest":
                    self._close_acquisitions()
                    if self._ink_handle is not None:
                        self._ink_handle.close()
                        self._ink_handle = None
                    self._write_manifest(item[1])
            except Exception as error:  # noqa: BLE001 - vai ao manifesto
                with self._lock:
                    self._errors.append(str(error))

    def _write_raw(self, entry, image):
        try:
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError(
                    f"imencode devolveu False no quadro cru {entry['frame_index']}"
                )
            data = encoded.tobytes()
            with self._lock:
                if self._raw_bytes + len(data) > self._raw_policy["max_bytes"]:
                    if self._raw_truncated is None:
                        self._raw_truncated = {
                            "reason": "max_bytes",
                            "frame_index": entry["frame_index"],
                            "t_s": entry["t_s"],
                        }
                    return
                self._raw_bytes += len(data)
            folder = self.session_dir / "raw"
            ensure_private_dir(self.session_dir)
            folder.mkdir(parents=True, exist_ok=True)
            name = f"frame_{entry['frame_index']:06d}.png"
            (folder / name).write_bytes(data)
            entry = {
                **entry,
                "file": f"raw/{name}",
                "sha256": _sha256_bytes(data),
                "bytes": len(data),
                "shape": list(image.shape),
            }
            with self._lock:
                self._raw_entries.append(entry)
        finally:
            with self._lock:
                self._raw_in_flight -= 1

    def _write_acquisition(self, record):
        if self._acq_handle is None:
            ensure_private_dir(self.session_dir)
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._acq_handle = open(
                self.session_dir / "acquisitions.jsonl", "a", encoding="utf-8"
            )
        self._acq_handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._acq_written += 1

    def _close_acquisitions(self):
        if self._acq_handle is not None:
            self._acq_handle.close()
            self._acq_handle = None

    def _write_manifest(self, manifest):
        ensure_private_dir(self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        generations = []
        for folder in sorted(self.session_dir.glob("gen_*")):
            entry = {"generation": int(folder.name.split("_")[1])}
            files = {}
            for name in ("reference_warp", "final_warp", "final_canvas"):
                path = folder / f"{name}.png"
                if path.exists():
                    entry[f"{name}_sha256"] = _sha256_bytes(path.read_bytes())
                    files[name] = f"{folder.name}/{name}.png"
            entry["files"] = files
            generations.append(entry)
        manifest["generations"] = generations
        ink_path = self.session_dir / "ink_passes.jsonl"
        manifest["ink_passes"] = {
            "file": "ink_passes.jsonl" if ink_path.exists() else None,
            "records": self._ink_records,
            "dropped": self._ink_dropped,
            "max_records": self._acq_max_records,
            "sha256": sha256_file(ink_path),
        }
        acq_path = self.session_dir / "acquisitions.jsonl"
        with self._lock:
            manifest["raw_frames"] = {
                "policy": dict(self._raw_policy),
                "noted_frames": self._raw_noted,
                "saved": len(self._raw_entries),
                "bytes": self._raw_bytes,
                "dropped_backlog": self._raw_dropped_backlog,
                "truncated": self._raw_truncated,
                "frames": sorted(
                    (dict(e) for e in self._raw_entries),
                    key=lambda e: e["frame_index"],
                ),
            }
            manifest["acquisitions"] = {
                "file": "acquisitions.jsonl" if acq_path.exists() else None,
                "records": self._acq_written,
                "max_records": self._acq_max_records,
                "dropped": self._acq_dropped,
                "truncated": self._acq_truncated,
                "sha256": sha256_file(acq_path),
            }
            manifest["write_errors"] = list(self._errors)
        (self.session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
