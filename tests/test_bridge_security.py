"""Fronteira de rede do bridge: bind loopback, token e framing hostil."""

import importlib.util
import socket
import struct
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import helper_receiver as rx

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "t" * 32


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "windows_helper", ROOT / "bridge" / "windows_helper.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = _load_helper()


def _jpeg(width=64, height=48):
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _reader(data):
    view = memoryview(data)
    pos = 0

    def read(n):
        nonlocal pos
        chunk = bytes(view[pos : pos + n])
        pos += len(chunk)
        return chunk

    return read


def test_helper_and_receiver_share_protocol_constants():
    for name in (
        "MAGIC",
        "VERSION",
        "DEFAULT_PORT",
        "HELLO_MAGIC",
        "HELLO_VERSION",
        "ACCEPT",
        "TOKEN_ENV",
        "TOKEN_MIN_LEN",
        "TOKEN_MAX_LEN",
    ):
        assert getattr(helper, name) == getattr(rx, name), name
    assert helper.HEADER.format == rx.HEADER.format
    assert helper.HELLO.format == rx.HELLO.format


def test_default_bind_is_loopback():
    assert helper.DEFAULT_HOST == "127.0.0.1"
    assert helper.resolve_bind_host() == "127.0.0.1"
    assert helper.resolve_bind_host("172.20.48.1") == "172.20.48.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "0:0::0"])
def test_wildcard_bind_requires_explicit_flag(host):
    with pytest.raises(SystemExit):
        helper.resolve_bind_host(host)


def test_wildcard_bind_with_flag_warns(capsys):
    assert helper.resolve_bind_host("0.0.0.0", allow_all_interfaces=True) == "0.0.0.0"
    assert "AVISO DE SEGURANCA" in capsys.readouterr().err


def _closed_without_reply(sock):
    try:
        return sock.recv(64) == b""
    except ConnectionResetError:
        return True


def _serve_once(token, frames, results):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def run():
        conn, _addr = server.accept()
        with conn:
            ok = helper.authenticate_client(conn, token.encode(), timeout=2)
            results.append(ok)
            if ok:
                for frame_id, payload in enumerate(frames):
                    conn.sendall(rx.pack_record(frame_id, frame_id * 10, payload))
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return server.getsockname()[1], thread


def test_helper_accepts_correct_token_and_streams():
    results = []
    payload = _jpeg()
    port, thread = _serve_once(TOKEN, [payload, payload], results)
    frames = list(rx.frames_from_socket("127.0.0.1", port, timeout=5, token=TOKEN))
    thread.join(5)
    assert results == [True]
    assert [f[0] for f in frames] == [0, 1]
    assert rx.decode_jpeg(frames[0][2]).shape == (48, 64, 3)


def test_helper_rejects_wrong_token_without_sending_frames():
    results = []
    port, thread = _serve_once(TOKEN, [_jpeg()], results)
    with pytest.raises(rx.AuthenticationError):
        list(rx.frames_from_socket("127.0.0.1", port, timeout=5, token="w" * 32))
    thread.join(5)
    assert results == [False]


def test_helper_rejects_connection_without_handshake():
    results = []
    port, thread = _serve_once(TOKEN, [_jpeg()], results)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        # Cliente legado: espera frames sem mandar HELLO.
        sock.sendall(rx.pack_record(0, 0, b"\xff\xd8\xff\xd9"))
        assert _closed_without_reply(sock)
    thread.join(5)
    assert results == [False]


def test_helper_rejects_oversized_token_length():
    results = []
    port, thread = _serve_once(TOKEN, [_jpeg()], results)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(rx.HELLO.pack(rx.HELLO_MAGIC, rx.HELLO_VERSION, 60000))
        assert _closed_without_reply(sock)
    thread.join(5)
    assert results == [False]


def test_receiver_requires_token(monkeypatch):
    monkeypatch.delenv(rx.TOKEN_ENV, raising=False)
    with pytest.raises(rx.AuthenticationError):
        rx.resolve_token()
    with pytest.raises(rx.AuthenticationError):
        rx.resolve_token("curto")
    monkeypatch.setenv(rx.TOKEN_ENV, TOKEN)
    assert rx.resolve_token() == TOKEN.encode()


def test_oversized_payload_len_rejected_before_reading():
    header = rx.HEADER.pack(rx.MAGIC, rx.VERSION, 0, 0, 0, 0xFFFFFFFF)
    reads = []

    def read(n):
        reads.append(n)
        return header if len(reads) == 1 else b"\x00" * n

    with pytest.raises(rx.ProtocolError, match="payload_len"):
        list(rx._iter_records(read))
    assert reads == [rx.HEADER_SIZE]


def test_zero_payload_len_rejected():
    data = rx.HEADER.pack(rx.MAGIC, rx.VERSION, 0, 0, 0, 0)
    with pytest.raises(rx.ProtocolError, match="payload_len"):
        list(rx._iter_records(_reader(data)))


def test_custom_max_payload_is_enforced(tmp_path):
    path = tmp_path / "s.drawcap"
    path.write_bytes(rx.pack_record(0, 0, _jpeg()))
    with pytest.raises(rx.ProtocolError):
        list(rx.frames_from_file(path, max_payload=16))


@pytest.mark.parametrize("cut", [1, 10, 27])
def test_truncated_header_is_protocol_error(tmp_path, cut):
    path = tmp_path / "s.drawcap"
    record = rx.pack_record(0, 0, _jpeg())
    path.write_bytes(record + record[:cut])
    frames = []
    with pytest.raises(rx.ProtocolError):
        for frame in rx.frames_from_file(path):
            frames.append(frame)
    assert len(frames) == 1


def test_truncated_payload_is_protocol_error(tmp_path):
    path = tmp_path / "s.drawcap"
    path.write_bytes(rx.pack_record(0, 0, _jpeg())[:-5])
    with pytest.raises(rx.ProtocolError, match="truncado"):
        list(rx.frames_from_file(path))


def test_bad_magic_and_version_are_protocol_errors():
    good = rx.pack_record(0, 0, _jpeg())
    with pytest.raises(rx.ProtocolError, match="magic"):
        list(rx._iter_records(_reader(b"XXXX" + good[4:])))
    bad_version = good[:4] + struct.pack("<H", 99) + good[6:]
    with pytest.raises(rx.ProtocolError, match="versao"):
        list(rx._iter_records(_reader(bad_version)))


def _sof_offset(jpeg):
    return jpeg.index(b"\xff\xc0")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b"",
        lambda b: b"\xff\xd8",
        lambda b: b[: len(b) // 2],
        lambda b: b[:-2],
        lambda b: b"\x00" + b[1:],
        lambda b: b[:20] + b"\x00" * 64 + b[84:],
        lambda b: b"\xff\xd8\xff\xd9",
        lambda b: b"\xff\xd8" + b"\xff\xe0\x00\x01" + b"\xff\xd9",
    ],
    ids=[
        "vazio",
        "so_soi",
        "metade",
        "sem_eoi",
        "sem_soi",
        "cabecalho_zerado",
        "soi_eoi",
        "segmento_curto",
    ],
)
def test_decode_jpeg_never_raises_on_corrupt_payload(mutate):
    assert rx.decode_jpeg(mutate(_jpeg())) is None


def test_decode_jpeg_rejects_huge_declared_dimensions():
    jpeg = bytearray(_jpeg())
    sof = _sof_offset(jpeg)
    # SOF0: FF C0 len(2) precision(1) height(2) width(2)
    struct.pack_into(">HH", jpeg, sof + 5, 60000, 60000)
    assert rx.decode_jpeg(bytes(jpeg)) is None


def test_decode_jpeg_accepts_valid_frame():
    assert rx.decode_jpeg(_jpeg(320, 240)).shape == (240, 320, 3)


def test_decode_jpeg_respects_frame_limits():
    assert rx.decode_jpeg(_jpeg(320, 240), max_side=100) is None
