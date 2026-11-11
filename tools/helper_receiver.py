"""Receptor do protocolo do ajudante Windows, lado WSL.

Implementa o contrato de bridge/PROTOCOL.md: um fluxo de registros
(cabeçalho binário de 28 bytes + payload JPEG) que chega por socket
TCP ou por arquivo gravado, sem o consumidor distinguir a fonte.
Cada frame sai como a tupla (frame_id, capture_ts_us, jpeg_bytes);
capture_ts_us é o relógio monotônico do ajudante, ancorado na
captura, e só serve para diferenças dentro da mesma sessão.

Toda conexão começa pelo handshake de token (PROTOCOL.md,
"Authentication handshake"): o token vem de DRAW_HELPER_TOKEN ou de
--token-file e nunca é impresso. O leitor limita payload_len antes
de alocar e decode_jpeg devolve None para JPEG malformado.

Uso (a partir da raiz do repositório):
  python tools/helper_receiver.py --selfcheck
  DRAW_HELPER_TOKEN=... python tools/helper_receiver.py --record \
      --host 192.168.0.10 --out sessao.drawcap [--max-frames 600]
  python tools/helper_receiver.py --dump sessao.drawcap
  DRAW_HELPER_TOKEN=... python tools/helper_receiver.py --dump \
      --host 192.168.0.10 --max-frames 5

Como biblioteca:
  from tools.helper_receiver import frames_from_file, frames_from_socket
  for frame_id, capture_ts_us, jpeg in frames_from_file("sessao.drawcap"):
      ...
"""

import argparse
import hmac
import os
import socket
import struct
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from draw.safe_io import decode_image, image_format  # noqa: E402

MAGIC = b"DRAW"
VERSION = 1
# magic, versao, reservado, frame_id, capture_ts_us, payload_len.
HEADER = struct.Struct("<4sHHQqI")
HEADER_SIZE = HEADER.size  # 28 bytes
DEFAULT_PORT = 8765

# Teto de payload_len: conferido ANTES de ler/alocar o payload. Um
# JPEG 4K de qualidade alta fica bem abaixo disso.
MAX_PAYLOAD_LEN = 16 << 20
MAX_FRAME_SIDE = 8192
MAX_FRAME_PIXELS = 40_000_000

# Handshake de autenticação (PROTOCOL.md): o cliente envia
# HELLO(magic, versão, len) + token; o ajudante responde ACCEPT ou fecha.
HELLO_MAGIC = b"DRAH"
HELLO_VERSION = 1
HELLO = struct.Struct("<4sHH")
ACCEPT = b"DROK"
TOKEN_ENV = "DRAW_HELPER_TOKEN"
TOKEN_MIN_LEN = 16
TOKEN_MAX_LEN = 256
HANDSHAKE_TIMEOUT_S = 5.0


class ProtocolError(Exception):
    """O fluxo violou bridge/PROTOCOL.md (magic, versão ou truncamento)."""


class AuthenticationError(ProtocolError):
    """Token ausente, inválido ou recusado pelo ajudante."""


def resolve_token(token=None, token_file=None):
    """Token como bytes: argumento, depois --token-file, depois o ambiente."""
    if token is None and token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    if token is None:
        token = os.environ.get(TOKEN_ENV, "")
    if isinstance(token, str):
        token = token.encode("utf-8")
    if not TOKEN_MIN_LEN <= len(token) <= TOKEN_MAX_LEN:
        raise AuthenticationError(
            f"token do ajudante ausente ou fora de {TOKEN_MIN_LEN}..{TOKEN_MAX_LEN} "
            f"bytes; defina {TOKEN_ENV} ou use --token-file"
        )
    return token


def client_handshake(sock, token):
    """Lado WSL: envia o token e exige ACCEPT antes de qualquer frame."""
    token = resolve_token(token)
    sock.sendall(HELLO.pack(HELLO_MAGIC, HELLO_VERSION, len(token)) + token)
    try:
        reply = _read_exact(sock.recv, len(ACCEPT))
    except (ProtocolError, OSError) as error:
        raise AuthenticationError(f"ajudante fechou o handshake: {error}") from error
    if reply != ACCEPT:
        raise AuthenticationError("ajudante recusou o token (handshake sem ACCEPT)")


def server_handshake(conn, token, timeout=HANDSHAKE_TIMEOUT_S):
    """Lado ajudante: True e ACCEPT enviado se o token confere.

    Espelha bridge/windows_helper.authenticate_client (usado no
    selfcheck e nos testes). Qualquer falha devolve False sem
    responder; quem chamou fecha a conexão.
    """
    expected = resolve_token(token)
    previous = conn.gettimeout()
    conn.settimeout(timeout)
    try:
        hello = _read_exact(conn.recv, HELLO.size)
        if hello is None:
            return False
        magic, version, length = HELLO.unpack(hello)
        if magic != HELLO_MAGIC or version != HELLO_VERSION:
            return False
        if not TOKEN_MIN_LEN <= length <= TOKEN_MAX_LEN:
            return False
        offered = _read_exact(conn.recv, length)
        if offered is None or not hmac.compare_digest(offered, expected):
            return False
        conn.sendall(ACCEPT)
        return True
    except (ProtocolError, OSError):
        return False
    finally:
        conn.settimeout(previous)


def _connect(host, port, token, timeout):
    token = resolve_token(token)
    sock = socket.create_connection(
        (host, port), timeout=timeout if timeout is not None else HANDSHAKE_TIMEOUT_S
    )
    try:
        client_handshake(sock, token)
        sock.settimeout(timeout)
    except BaseException:
        sock.close()
        raise
    return sock


def pack_record(frame_id, capture_ts_us, payload):
    """Monta um registro completo (cabeçalho + payload) para escrever."""
    return (
        HEADER.pack(MAGIC, VERSION, 0, frame_id, capture_ts_us, len(payload)) + payload
    )


def _read_exact(read, size):
    """Lê exatamente size bytes via read(n) -> bytes.

    Retorna None se a fonte acabar limpa antes do primeiro byte.
    Levanta ProtocolError se acabar no meio.
    """
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise ProtocolError(f"fluxo truncado: faltaram {remaining} de {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _iter_records(read, max_payload=MAX_PAYLOAD_LEN):
    """Itera registros sobre uma função read(n) -> bytes.

    Produz (frame_id, capture_ts_us, payload, bruto), onde bruto é o
    registro byte a byte como chegou (o que o gravador escreve).
    payload_len fora de 1..max_payload é erro de protocolo antes de
    qualquer leitura do payload.
    """
    while True:
        header = _read_exact(read, HEADER_SIZE)
        if header is None:
            return
        magic, version, _reserved, frame_id, capture_ts_us, payload_len = HEADER.unpack(
            header
        )
        if magic != MAGIC:
            raise ProtocolError(
                f"magic invalido: {magic!r} (esperado {MAGIC!r}); "
                "fluxo dessincronizado ou arquivo de outro formato"
            )
        if version != VERSION:
            raise ProtocolError(f"versao {version} nao suportada (esperada {VERSION})")
        if payload_len == 0 or payload_len > max_payload:
            raise ProtocolError(
                f"payload_len {payload_len} fora de 1..{max_payload} "
                f"(frame_id {frame_id}); fluxo corrompido ou hostil"
            )
        payload = _read_exact(read, payload_len)
        if payload is None:
            raise ProtocolError(
                "fluxo truncado: cabecalho sem payload "
                f"(frame_id {frame_id}, {payload_len} bytes esperados)"
            )
        yield frame_id, capture_ts_us, payload, header + payload


def frames_from_file(path, max_payload=MAX_PAYLOAD_LEN):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) de um arquivo gravado."""
    with open(path, "rb") as handle:
        for frame_id, capture_ts_us, payload, _raw in _iter_records(
            handle.read, max_payload
        ):
            yield frame_id, capture_ts_us, payload


def frames_from_socket(
    host, port=DEFAULT_PORT, timeout=None, token=None, max_payload=MAX_PAYLOAD_LEN
):
    """Itera (frame_id, capture_ts_us, jpeg_bytes) do ajudante via TCP.

    Autentica com o token (argumento ou DRAW_HELPER_TOKEN) antes de ler.
    """
    with _connect(host, port, token, timeout) as sock:
        for frame_id, capture_ts_us, payload, _raw in _iter_records(
            sock.recv, max_payload
        ):
            yield frame_id, capture_ts_us, payload


def record(
    host,
    port,
    out_path,
    max_frames=None,
    timeout=None,
    token=None,
    max_payload=MAX_PAYLOAD_LEN,
):
    """Grava o fluxo do socket byte a byte num arquivo. Retorna n."""
    count = 0
    with _connect(host, port, token, timeout) as sock:
        with open(out_path, "wb") as handle:
            for _fid, _ts, _payload, raw in _iter_records(sock.recv, max_payload):
                handle.write(raw)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
    return count


def is_complete_jpeg(payload):
    """True se o payload começa em FF D8 e termina em FF D9."""
    return (
        len(payload) >= 4
        and image_format(payload) == "jpeg"
        and bytes(payload[-2:]) == b"\xff\xd9"
    )


def decode_jpeg(jpeg_bytes, max_side=MAX_FRAME_SIDE, max_pixels=MAX_FRAME_PIXELS):
    """Decodifica um payload em imagem BGR (numpy) ou None se inválido.

    Payload truncado, sem cabeçalho SOF, com dimensões acima do teto
    ou que o decoder recuse devolve None; nunca levanta.
    """
    import cv2

    if not is_complete_jpeg(jpeg_bytes):
        return None
    return decode_image(
        bytes(jpeg_bytes),
        cv2.IMREAD_COLOR,
        max_bytes=MAX_PAYLOAD_LEN,
        max_side=max_side,
        max_pixels=max_pixels,
    )


def _synthetic_frames(n, base_id=1000, base_ts_us=5_000_000):
    """Frames JPEG sintéticos determinísticos para o selfcheck."""
    import cv2
    import numpy as np

    frames = []
    for i in range(n):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:, :, 0] = (i * 37) % 256
        image[:, :, 1] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
        image[:, :, 2] = (255 - i * 11) % 256
        cv2.putText(
            image,
            str(base_id + i),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("cv2.imencode falhou no selfcheck")
        frames.append((base_id + i, base_ts_us + i * 33_333, encoded.tobytes()))
    return frames


def _check_roundtrip(expected, received, label):
    """Confere ids, timestamps e bytes. Retorna (ok, mensagem)."""
    if len(received) != len(expected):
        return False, (f"{label}: n esperado {len(expected)}, recebido {len(received)}")
    for want, got in zip(expected, received):
        if want != got:
            return False, (
                f"{label}: divergencia no frame_id {want[0]}: "
                f"esperado ts={want[1]} {len(want[2])} bytes, "
                f"recebido ts={got[1]} {len(got[2])} bytes"
            )
    return True, (
        f"{label}: n={len(expected)}, ids, timestamps e bytes "
        "conferem em todos os frames"
    )


def _selfcheck():
    import secrets
    import tempfile

    n = 64
    expected = _synthetic_frames(n)
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selfcheck.drawcap"
        with open(path, "wb") as handle:
            for frame_id, capture_ts_us, payload in expected:
                handle.write(pack_record(frame_id, capture_ts_us, payload))

        ok, message = _check_roundtrip(
            expected, list(frames_from_file(path)), "arquivo (replay)"
        )
        print(("PASS " if ok else "FAIL ") + message)
        if not ok:
            failures.append(message)

        # Rabo truncado: os n frames bons saem, depois erro de protocolo.
        truncated = Path(tmp) / "truncado.drawcap"
        truncated.write_bytes(path.read_bytes() + b"DRA")
        seen = 0
        try:
            for _frame in frames_from_file(truncated):
                seen += 1
            failures.append(
                f"arquivo truncado: nenhum erro levantado apos os {seen} frames bons"
            )
            print(f"FAIL arquivo truncado: {seen} frames lidos sem erro")
        except ProtocolError:
            print(
                f"PASS arquivo truncado: erro de protocolo apos os "
                f"{seen} frames bons (n={n} esperado)"
            )
            if seen != n:
                failures.append(
                    f"arquivo truncado: {seen} frames antes do erro, esperados {n}"
                )

    # Caminho de socket: servidor loopback efêmero numa thread, com
    # handshake de token. A primeira conexão usa um token errado e
    # precisa ser recusada; a segunda autentica e recebe os frames.
    token = secrets.token_urlsafe(32)
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    port = server.getsockname()[1]

    def serve():
        for _attempt in range(2):
            conn, _addr = server.accept()
            with conn:
                if not server_handshake(conn, token):
                    continue
                for frame_id, capture_ts_us, payload in expected:
                    conn.sendall(pack_record(frame_id, capture_ts_us, payload))
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        list(frames_from_socket("127.0.0.1", port, timeout=10, token="x" * 32))
        failures.append("socket: token errado foi aceito")
        print("FAIL socket: token errado foi aceito")
    except AuthenticationError:
        print("PASS socket: token errado recusado no handshake")
    received = list(frames_from_socket("127.0.0.1", port, timeout=10, token=token))
    thread.join(timeout=10)
    ok, message = _check_roundtrip(expected, received, "socket (loopback)")
    print(("PASS " if ok else "FAIL ") + message)
    if not ok:
        failures.append(message)

    if failures:
        print(f"SELFCHECK FALHOU: {len(failures)} problema(s).")
        return 1
    print(f"SELFCHECK OK: {2 * n + n} frames conferidos no total.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Receptor do protocolo do ajudante (bridge/PROTOCOL.md)"
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="gera frames sinteticos, grava, rele por arquivo e por "
        "socket loopback e confere ids, timestamps e bytes",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="grava o fluxo do socket num arquivo (--host e --out)",
    )
    parser.add_argument(
        "--dump",
        nargs="?",
        const="",
        default=None,
        metavar="ARQUIVO",
        help="lista os frames de um arquivo gravado; sem arquivo e com "
        "--host, lista os do socket",
    )
    parser.add_argument("--host", default=None, help="IP do host Windows")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out", default=None, help="arquivo de saida")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--token-file",
        default=None,
        help=f"arquivo com o token do ajudante (padrao: ${TOKEN_ENV})",
    )
    args = parser.parse_args()

    if args.selfcheck:
        return _selfcheck()

    if args.record:
        if not args.host or not args.out:
            parser.error("--record exige --host e --out")
        token = resolve_token(token_file=args.token_file)
        count = record(
            args.host, args.port, args.out, max_frames=args.max_frames, token=token
        )
        print(f"Gravados {count} frames em {args.out}.")
        return 0

    if args.dump is not None:
        if args.dump:
            frames = frames_from_file(args.dump)
        elif args.host:
            frames = frames_from_socket(
                args.host, args.port, token=resolve_token(token_file=args.token_file)
            )
        else:
            parser.error("--dump sem arquivo exige --host")
        count = 0
        for frame_id, capture_ts_us, payload in frames:
            print(
                f"frame_id={frame_id} capture_ts_us={capture_ts_us} "
                f"jpeg={len(payload)} bytes"
            )
            count += 1
            if args.max_frames is not None and count >= args.max_frames:
                break
        print(f"Total: {count} frames.")
        return 0

    parser.error("nada a fazer: use --selfcheck, --record ou --dump")


if __name__ == "__main__":
    sys.exit(main())
