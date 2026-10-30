"""Camada de captura do Draw, separada do pipeline de visao.

O pipeline (paper.py, ink.py) recebe apenas frames. Este modulo decide
de onde os frames vem: indice de webcam, caminho /dev/video*, arquivo
ou stream de rede. Ele nao abre janelas e nao depende de hardware real
para ser testado: a fabrica de captura e injetavel.

O que ja existia no upstream e continua valendo, sem ser duplicado
aqui: o diagnostico da fonte selecionada em `python main.py --check` e
o script `scripts/setup_camera_wsl.sh` (uvcvideo, usbipd, grupo video).
Para fontes locais, este modulo tambem diagnostica a porta especifica.
Ele acrescenta o que faltava:
listagem sem travar, tentativas por backend, erros com acao especifica
e reconexao quando a camera cai no meio do uso.
"""

import glob
import os
import re
import sys
import time
from pathlib import Path

import cv2

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_OPEN_TIMEOUT_MS = 3000
DEFAULT_MAX_PROBE_INDEX = 9
DEFAULT_RECONNECT_ATTEMPTS = 3
DEFAULT_RECONNECT_DELAY_S = 0.5

_VIDEO_NODE_RE = re.compile(r"^/dev/video(\d+)$")

_BACKEND_NAMES = {
    "any": "CAP_ANY",
    "v4l2": "CAP_V4L2",
    "msmf": "CAP_MSMF",
    "dshow": "CAP_DSHOW",
    "ffmpeg": "CAP_FFMPEG",
    "avfoundation": "CAP_AVFOUNDATION",
    "gstreamer": "CAP_GSTREAMER",
}


class CaptureError(RuntimeError):
    """Falha de abertura com o tipo do problema e a acao sugerida."""

    def __init__(self, source, kind, hint, attempts=()):
        self.source = source
        self.kind = kind
        self.attempts = tuple(attempts)
        super().__init__(hint)


def parse_source(value):
    """Indice, caminho /dev/videoN, arquivo ou URL, sem adivinhar.

    '0'/'1' viram indice; '/dev/video2' vira indice 2 (o mesmo
    dispositivo, visto pelo numero da porta, que e o que o diagnostico
    por porta do --check entende); qualquer outra coisa continua texto.
    """
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = _VIDEO_NODE_RE.match(text)
    if match:
        return int(match.group(1))
    return text


def is_wsl():
    """Detecta WSL pelo nome do kernel em /proc."""
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def video_nodes():
    """Caminhos /dev/video* numerados, ordenados, ou lista vazia."""
    return sorted(
        path
        for path in glob.glob("/dev/video*")
        if path[len("/dev/video"):].isdigit()
    )


def backend_from_name(name):
    """'v4l2' -> cv2.CAP_V4L2; nome desconhecido levanta ValueError."""
    if name is None:
        return None
    key = str(name).strip().lower()
    attr = _BACKEND_NAMES.get(key)
    if attr is None or not hasattr(cv2, attr):
        raise ValueError(
            f"Backend desconhecido: {name!r}. "
            f"Opcoes: {', '.join(sorted(_BACKEND_NAMES))}."
        )
    return getattr(cv2, attr)


def backends_for_platform(platform=None):
    """Backends na ordem de tentativa para o sistema operacional."""
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        wanted = ("CAP_MSMF", "CAP_DSHOW", "CAP_ANY")
    elif platform == "darwin":
        wanted = ("CAP_AVFOUNDATION", "CAP_ANY")
    else:
        wanted = ("CAP_V4L2", "CAP_ANY")
    backends = [getattr(cv2, name) for name in wanted if hasattr(cv2, name)]
    return backends or [cv2.CAP_ANY]


def diagnose_local(index, exists=os.path.exists, access=os.access, wsl=None):
    """Diagnostico por porta, o mesmo criterio do --check.

    Retorna (kind, hint): 'not_found' quando /dev/videoN nao existe,
    'permission' quando existe mas nao e legivel/gravavel, e 'busy'
    quando existe, tem permissao e mesmo assim nao abriu.
    """
    node = f"/dev/video{index}"
    if not sys.platform.startswith("linux"):
        return (
            "busy",
            "Nenhuma webcam local respondeu nesse indice. "
            "Confira se a webcam esta conectada e livre.",
        )
    if not exists(node):
        hint = f"A camera nao esta conectada a este sistema: {node} nao existe. "
        if is_wsl() if wsl is None else wsl:
            hint += (
                "Anexe a camera ao WSL com: bash scripts/setup_camera_wsl.sh. "
                "A alternativa e uma camera via rede: python main.py "
                "--source http://<ip-da-camera>:8080/video."
            )
        else:
            hint += "Confira se a webcam esta conectada a este computador."
        return "not_found", hint
    if not access(node, os.R_OK | os.W_OK):
        return (
            "permission",
            f"{node} existe, mas este processo nao tem permissao de "
            "leitura e escrita nele. A porta que falhou e a permissao. "
            "Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes). "
            'Correcao nesta sessao: sg video -c "python3 main.py".',
        )
    return (
        "busy",
        f"{node} existe e este processo tem permissao nele, mas a "
        "abertura falhou. Outro programa provavelmente esta usando a "
        "camera. Feche esse programa e tente de novo.",
    )


def apply_local_settings(capture, width=None, height=None, fps=None):
    """MJPG 720p por padrao para webcam local, ou o tamanho pedido.

    O formato cru padrao (YUYV em 1080p) estoura a banda do USB
    virtual do usbip no WSL e a entrega de quadros trava com timeout
    de select() no V4L2. Por isso o padrao e MJPG comprimido.
    """
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH if width is None else width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT if height is None else height)
    if fps is not None:
        capture.set(cv2.CAP_PROP_FPS, fps)


def _timeout_parameters(open_timeout_ms=None, read_timeout_ms=None):
    """Build the open-only timeout params supported by this cv2 binding."""
    params = []
    missing = []
    for name, value in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", open_timeout_ms),
        ("CAP_PROP_READ_TIMEOUT_MSEC", read_timeout_ms),
    ):
        if value is None:
            continue
        prop = getattr(cv2, name, None)
        if prop is None:
            missing.append(name)
        else:
            params.extend((prop, int(value)))
    return params, missing


def _release_quietly(capture):
    if capture is not None:
        try:
            capture.release()
        except Exception:
            pass


def open_capture(source, width=None, height=None, fps=None, backend=None,
                 open_timeout_ms=DEFAULT_OPEN_TIMEOUT_MS,
                 read_timeout_ms=None,
                 capture_factory=None, backends=None):
    """Abre uma fonte tentando os backends, sem esconder o erro.

    Para webcam local, tenta os backends do SO em ordem. Para URL ou
    arquivo, usa CAP_ANY, a nao ser que um backend seja pedido.
    Se nada abrir, levanta CaptureError com kind e hint acionavel.
    """
    parsed = parse_source(source)
    factory = capture_factory or cv2.VideoCapture
    if backend is not None:
        candidates = [backend]
    elif backends is not None:
        candidates = list(backends)
    elif isinstance(parsed, int):
        candidates = backends_for_platform()
    else:
        candidates = [cv2.CAP_ANY]

    local_source = isinstance(parsed, int)
    timeout_params, missing_timeout_params = _timeout_parameters(
        open_timeout_ms, read_timeout_ms
    )
    if not local_source and missing_timeout_params:
        missing = ", ".join(missing_timeout_params)
        raise CaptureError(
            parsed,
            "timeout_unsupported",
            f"Nao abri a fonte {parsed!r}: esta build do OpenCV nao expoe "
            f"{missing}. Nao foi tentada uma abertura sem timeout.",
        )

    attempts = []
    timeout_error = None
    for candidate in candidates:
        capture = None
        try:
            if timeout_params:
                try:
                    capture = factory(parsed, candidate, timeout_params)
                    opened = bool(capture.isOpened())
                except Exception as error:
                    timeout_error = error
                    _release_quietly(capture)
                    if not local_source:
                        attempts.append(candidate)
                        continue
                    capture = factory(parsed, candidate)
                    opened = bool(capture.isOpened())
                else:
                    if local_source and not opened:
                        _release_quietly(capture)
                        capture = factory(parsed, candidate)
                        opened = bool(capture.isOpened())
            else:
                capture = factory(parsed, candidate)
                opened = bool(capture.isOpened())
        except Exception as error:
            timeout_error = error if timeout_params else timeout_error
            attempts.append(candidate)
            continue

        if opened:
            if isinstance(parsed, int):
                apply_local_settings(capture, width=width, height=height, fps=fps)
            return capture
        attempts.append(candidate)
        _release_quietly(capture)

    if isinstance(parsed, int):
        kind, hint = diagnose_local(parsed)
    elif isinstance(parsed, str) and parsed.startswith(("http://", "https://", "rtsp://")):
        kind = "network"
        hint = (
            "Confira se a URL esta certa, se a camera esta transmitindo e "
            "se este computador alcanca o endereco dela na rede."
        )
    else:
        kind = "not_found"
        hint = "Confira se o arquivo existe e se o caminho esta certo."
    if not local_source:
        hint += (
            " A abertura foi tentada com parametros de timeout e nao foi "
            "tentada novamente sem eles; esta build ou backend do OpenCV "
            "pode nao suportar esses parametros."
        )
        if timeout_error is not None:
            hint += f" Detalhe: {timeout_error}"
    raise CaptureError(
        parsed, kind,
        f"Nao consegui abrir a fonte de video: {parsed!r}. {hint}",
        attempts=attempts,
    )


def list_cameras(max_index=DEFAULT_MAX_PROBE_INDEX, open_timeout_ms=1000,
                 capture_factory=None, nodes=None):
    """Enumera webcams locais sem travar indefinidamente.

    Tenta cada indice com parametro de timeout de abertura e solta a captura
    logo em seguida. Backends locais que rejeitam o parametro usam a
    abertura anterior sem parametros. Nao le frames aqui: abrir ja basta para listar,
    e ler poderia travar numa camera ocupada. Retorna uma lista de
    dicionarios com index, node e opened, incluindo indices vindos de
    /dev/video* que estejam acima de max_index.
    """
    factory = capture_factory or cv2.VideoCapture
    indices = set(range(max_index + 1))
    for node in video_nodes() if nodes is None else nodes:
        suffix = node[len("/dev/video"):]
        if suffix.isdigit():
            indices.add(int(suffix))
    backend = backends_for_platform()[0]
    timeout_params, _ = _timeout_parameters(open_timeout_ms=open_timeout_ms)
    results = []
    for index in sorted(indices):
        capture = None
        opened = False
        try:
            if timeout_params:
                try:
                    capture = factory(index, backend, timeout_params)
                    opened = bool(capture.isOpened())
                except Exception:
                    capture = factory(index, backend)
                if not opened and capture is not None:
                    _release_quietly(capture)
                    capture = factory(index, backend)
            else:
                capture = factory(index, backend)
            opened = bool(capture.isOpened())
        except Exception:
            opened = False
        finally:
            _release_quietly(capture)
        results.append({
            "index": index,
            "node": f"/dev/video{index}" if sys.platform.startswith("linux") else None,
            "opened": opened,
        })
    return results


def format_camera_list(results):
    """Texto da listagem para o terminal, uma camera por linha."""
    if not results:
        return "Nenhuma camera encontrada."
    lines = []
    for item in results:
        status = "ok" if item["opened"] else "nao abriu"
        node = f" ({item['node']})" if item.get("node") else ""
        lines.append(f"  {item['index']}{node}: {status}")
    return "\n".join(lines)


class ReconnectingCapture:
    """Envolve uma captura e tenta reabri-la quando ela cai.

    O loop do app continua chamando read() normalmente. Quando os
    quadros param, o app chama try_reconnect(): a captura antiga e
    solta e a mesma fonte e aberta de novo, ate attempts vezes, com
    delay_s entre tentativas. Se voltar, read() volta a entregar
    frames; se nao voltar, try_reconnect() retorna False e o app
    pode encerrar de forma limpa, avisando que foi desconexao.
    """

    def __init__(self, source, open_func=None, attempts=DEFAULT_RECONNECT_ATTEMPTS,
                 delay_s=DEFAULT_RECONNECT_DELAY_S, sleep=None, **open_kwargs):
        self.source = parse_source(source)
        self._open_func = open_func or open_capture
        self.attempts = attempts
        self.delay_s = delay_s
        self._sleep = sleep or time.sleep
        self._open_kwargs = open_kwargs
        self.capture = None
        self.last_error = None
        self.reconnects = 0

    def attach(self, capture):
        self.capture = capture
        return self

    def isOpened(self):
        return self.capture is not None and self.capture.isOpened()

    def read(self):
        if self.capture is None:
            return False, None
        return self.capture.read()

    def release(self):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

    def try_reconnect(self):
        """Tenta reabrir a fonte. Retorna True se voltou a abrir."""
        if self.attempts <= 0:
            return False
        for _ in range(self.attempts):
            self.release()
            try:
                candidate = self._open_func(self.source, **self._open_kwargs)
            except CaptureError as error:
                self.last_error = error
                candidate = None
            if candidate is not None and candidate.isOpened():
                self.capture = candidate
                self.reconnects += 1
                return True
            if self.delay_s:
                self._sleep(self.delay_s)
        return False
