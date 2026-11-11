"""Leitura defensiva de material externo: imagens, JSON e JSONL.

Frames de stream, datasets, manifestos de sessão e receitas chegam de
fontes de procedência incerta (SECURITY_MODEL.md, "Entry points").
Estas funções colocam um teto no que é lido ANTES de alocar ou
decodificar e convertem entrada malformada em erro tratável:

- imagem: só PNG/JPEG, com tamanho em bytes, lado e número de pixels
  limitados a partir do cabeçalho (antes do decoder nativo); arquivo
  malformado devolve None, como cv2.imread, sem levantar.
- JSON/JSONL: tamanho do arquivo, número e tamanho de linhas
  limitados; JSON inválido vira InputLimitError com caminho e linha.

ensure_private_dir marca uma pasta de dados privados com um
.gitignore "*", para que capturas nunca entrem no git, qualquer que
seja a raiz escolhida pelo operador.
"""

import json
import struct
from pathlib import Path

MAX_IMAGE_BYTES = 64 << 20
MAX_IMAGE_SIDE = 16384
MAX_IMAGE_PIXELS = 64_000_000
MAX_JSON_BYTES = 16 << 20
MAX_JSONL_BYTES = 256 << 20
MAX_JSONL_LINES = 1_000_000
MAX_JSONL_LINE_BYTES = 1 << 20

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# SOF0..SOF15, menos DHT (C4), JPG (C8) e DAC (CC).
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_JPEG_STANDALONE = frozenset({0x01, *range(0xD0, 0xD9)})
_PRIVATE_GITIGNORE = (
    "# Dados privados (capturas, sessoes, exemplos): nunca versionar.\n*\n"
)


class InputLimitError(ValueError):
    """Entrada externa malformada ou acima dos limites de leitura."""


def _jpeg_dimensions(data):
    size = len(data)
    if size < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    while i + 1 < size:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        i += 2
        if marker in _JPEG_STANDALONE:
            continue
        if marker in (0xD9, 0xDA):
            return None
        if i + 2 > size:
            return None
        (length,) = struct.unpack(">H", data[i : i + 2])
        if length < 2 or i + length > size:
            return None
        if marker in _JPEG_SOF:
            if length < 7:
                return None
            height, width = struct.unpack(">HH", data[i + 3 : i + 7])
            return width, height
        i += length
    return None


def image_format(data):
    """'png', 'jpeg' ou None, pela assinatura dos primeiros bytes."""
    head = bytes(data[:8])
    if head.startswith(_PNG_SIGNATURE):
        return "png"
    if head.startswith(b"\xff\xd8"):
        return "jpeg"
    return None


def image_dimensions(data):
    """(largura, altura) lidas do cabeçalho PNG/JPEG, sem decodificar.

    None quando o formato não é PNG/JPEG ou o cabeçalho está truncado
    ou malformado.
    """
    kind = image_format(data)
    if kind == "png":
        if len(data) < 24 or bytes(data[12:16]) != b"IHDR":
            return None
        return struct.unpack(">II", bytes(data[16:24]))
    if kind == "jpeg":
        return _jpeg_dimensions(memoryview(data).tobytes())
    return None


def check_image_bytes(
    data,
    max_bytes=MAX_IMAGE_BYTES,
    max_side=MAX_IMAGE_SIDE,
    max_pixels=MAX_IMAGE_PIXELS,
):
    """Confere tamanho, formato e dimensões. Devolve (w, h) ou levanta."""
    if len(data) == 0:
        raise InputLimitError("imagem vazia")
    if len(data) > max_bytes:
        raise InputLimitError(f"imagem com {len(data)} bytes > limite {max_bytes}")
    dims = image_dimensions(data)
    if dims is None:
        raise InputLimitError("imagem sem cabecalho PNG/JPEG valido")
    width, height = dims
    if width <= 0 or height <= 0:
        raise InputLimitError(f"dimensoes invalidas: {width}x{height}")
    if width > max_side or height > max_side or width * height > max_pixels:
        raise InputLimitError(
            f"dimensoes {width}x{height} acima do limite "
            f"({max_side} px por lado, {max_pixels} px no total)"
        )
    return width, height


def decode_image(data, flags=None, **limits):
    """Decodifica bytes PNG/JPEG dentro dos limites; None se inválido."""
    import cv2
    import numpy as np

    try:
        check_image_bytes(data, **limits)
    except InputLimitError:
        return None
    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    except (cv2.error, ValueError):
        return None
    if image is None or image.size == 0:
        return None
    return image


def read_image(path, flags=None, max_bytes=MAX_IMAGE_BYTES, **limits):
    """cv2.imread com limites: None se ausente, grande demais ou inválido."""
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    return decode_image(data, flags, max_bytes=max_bytes, **limits)


def _read_bounded(path, max_bytes):
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise InputLimitError(f"{path}: ilegivel ({error})") from error
    if size > max_bytes:
        raise InputLimitError(f"{path}: {size} bytes > limite {max_bytes}")
    try:
        with open(path, "rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as error:
        raise InputLimitError(f"{path}: ilegivel ({error})") from error
    if len(data) > max_bytes:
        raise InputLimitError(f"{path}: cresceu acima do limite {max_bytes}")
    return data


def read_json(path, max_bytes=MAX_JSON_BYTES):
    """json.load com teto de tamanho; erro de sintaxe vira InputLimitError."""
    data = _read_bounded(path, max_bytes)
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise InputLimitError(f"{path}: JSON invalido ({error})") from error


def iter_jsonl(
    path,
    max_bytes=MAX_JSONL_BYTES,
    max_lines=MAX_JSONL_LINES,
    max_line_bytes=MAX_JSONL_LINE_BYTES,
):
    """Itera (numero_da_linha, objeto) de um JSONL, com tetos.

    Linhas vazias são puladas. Cada objeto precisa ser um dict.
    """
    data = _read_bounded(path, max_bytes)
    count = 0
    for number, raw in enumerate(data.split(b"\n"), 1):
        if not raw.strip():
            continue
        count += 1
        if count > max_lines:
            raise InputLimitError(f"{path}: mais de {max_lines} linhas")
        if len(raw) > max_line_bytes:
            raise InputLimitError(
                f"{path}:{number}: linha com {len(raw)} bytes > {max_line_bytes}"
            )
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise InputLimitError(
                f"{path}:{number}: JSON invalido ({error})"
            ) from error
        if not isinstance(row, dict):
            raise InputLimitError(f"{path}:{number}: linha nao e um objeto JSON")
        yield number, row


def ensure_private_dir(path):
    """Cria a pasta e garante um .gitignore "*" nela. Devolve o Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".gitignore"
    if not marker.exists():
        try:
            with open(marker, "x", encoding="utf-8") as handle:
                handle.write(_PRIVATE_GITIGNORE)
        except FileExistsError:
            pass
    return path
