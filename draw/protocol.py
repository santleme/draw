"""Protocolo binário de captura (bridge/PROTOCOL.md) e adaptador de fonte.

Permite abrir tanto gravações `.drawcap` quanto streams TCP ao vivo do
ajudante Windows (`tcp://host[:porta]`) diretamente em `main.py --source`,
preservando o carimbo monotônico real de captura (`capture_ts_us`).
"""

import socket
import struct
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

MAGIC = b"DRAW"
VERSION = 1
HEADER = struct.Struct("<4sHHQqI")
HEADER_SIZE = HEADER.size  # 28 bytes
DEFAULT_PORT = 8765


class ProtocolError(Exception):
    """O fluxo violou bridge/PROTOCOL.md (magic, versão ou truncamento)."""


def pack_record(frame_id, capture_ts_us, payload):
    """Monta um registro completo (cabeçalho + payload) para escrever."""
    return (
        HEADER.pack(MAGIC, VERSION, 0, frame_id, capture_ts_us, len(payload))
        + payload
    )


def _read_exact(read, size):
    """Lê exatamente size bytes via read(n) -> bytes."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise ProtocolError(
                f"fluxo truncado: faltaram {remaining} de {size} bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_records(read):
    """Itera (frame_id, capture_ts_us, payload, bruto) sobre read(n)."""
    while True:
        header = _read_exact(read, HEADER_SIZE)
        if header is None:
            return
        magic, version, _reserved, frame_id, capture_ts_us, payload_len = (
            HEADER.unpack(header)
        )
        if magic != MAGIC:
            raise ProtocolError(
                f"magic invalido: {magic!r} (esperado {MAGIC!r}); "
                "fluxo dessincronizado ou arquivo de outro formato"
            )
        if version != VERSION:
            raise ProtocolError(
                f"versao {version} nao suportada (esperada {VERSION})"
            )
        payload = _read_exact(read, payload_len)
        if payload is None:
            raise ProtocolError(
                "fluxo truncado: cabecalho sem payload "
                f"(frame_id {frame_id}, {payload_len} bytes esperados)"
            )
        yield frame_id, capture_ts_us, payload, header + payload


def frames_from_file(path):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) de um arquivo .drawcap."""
    with open(path, "rb") as handle:
        for frame_id, capture_ts_us, payload, _raw in iter_records(
            handle.read
        ):
            yield frame_id, capture_ts_us, payload


def frames_from_socket(host, port=DEFAULT_PORT, timeout=None):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) do ajudante via TCP."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        for frame_id, capture_ts_us, payload, _raw in iter_records(sock.recv):
            yield frame_id, capture_ts_us, payload


def record_socket(host, port, out_path, max_frames=None, timeout=None):
    """Grava o fluxo do socket byte a byte num arquivo. Retorna n."""
    count = 0
    with socket.create_connection((host, port), timeout=timeout) as sock:
        with open(out_path, "wb") as handle:
            for _fid, _ts, _payload, raw in iter_records(sock.recv):
                handle.write(raw)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
    return count


def decode_jpeg(jpeg_bytes):
    """Decodifica um payload em imagem BGR (numpy) ou None se inválido."""
    if not jpeg_bytes:
        return None
    return cv2.imdecode(
        np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )


def is_helper_source(source):
    """True para arquivos `.drawcap` ou URLs `tcp://host[:porta]`."""
    if not isinstance(source, str):
        return False
    text = source.strip()
    lower = text.lower()
    return (
        lower.endswith(".drawcap")
        or lower.startswith("tcp://")
        or lower.startswith("drawcap://")
    )


def parse_tcp_endpoint(url):
    """Extrai (host, porta) de `tcp://host[:porta]` ou `drawcap://host[:porta]`."""
    parsed = urlparse(url.strip())
    if not parsed.hostname:
        raise ValueError(f"endereco TCP sem host: {url!r}")
    port = parsed.port if parsed.port is not None else DEFAULT_PORT
    return parsed.hostname, int(port)


class HelperCapture:
    """Adaptador compatível com VideoCapture sobre `.drawcap` ou `tcp://`.

    Além de `read() -> (ok, frame)`, expõe `last_capture_ts_ms` e
    `last_frame_id` do registro mais recente para alimentar o acumulador
    e o filtro de fusão com o relógio real de captura do emissor.
    """

    def __init__(self, source, timeout=5.0):
        self.source = str(source).strip()
        self._handle = None
        self._sock = None
        self._iter = None
        self._opened = False
        self.last_frame_id = None
        self.last_capture_ts_ms = None
        self._open(timeout)

    def _open(self, timeout):
        lower = self.source.lower()
        try:
            if lower.startswith("tcp://") or lower.startswith("drawcap://"):
                host, port = parse_tcp_endpoint(self.source)
                self._sock = socket.create_connection(
                    (host, port), timeout=timeout
                )
                self._iter = iter_records(self._sock.recv)
                self._opened = True
            else:
                path = Path(self.source)
                if not path.is_file():
                    self._opened = False
                    return
                self._handle = open(path, "rb")
                self._iter = iter_records(self._handle.read)
                self._opened = True
        except OSError:
            self.release()

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self._iter is None:
            return False, None
        try:
            frame_id, capture_ts_us, payload, _raw = next(self._iter)
        except (StopIteration, ProtocolError, OSError):
            return False, None
        frame = decode_jpeg(payload)
        if frame is None:
            return False, None
        self.last_frame_id = int(frame_id)
        self.last_capture_ts_ms = int(capture_ts_us // 1000)
        return True, frame

    def release(self):
        self._opened = False
        self._iter = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
