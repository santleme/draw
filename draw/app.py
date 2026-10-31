"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a fonte e sai
  python main.py --markers                        # folha-modelo impressa
  python main.py --neural modelo.onnx             # detector neural primario
  python main.py --async-detect                   # detector em thread dedicada
  python main.py --hand-model modelo.task         # mascara de mao + cursor
  python main.py --session-baseline               # tinta da sessao (tecla C)
  python main.py --source 1                       # outra webcam local
  python main.py --source http://192.168.0.10:8080/video   # camera via rede
  python main.py --source tcp://192.168.0.10:8765          # helper Windows TCP
  python main.py --source sessao.drawcap                   # replay .drawcap

Teclas: s = salvar o canvas como PNG (+ carimbos .npz), f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames,
c = congelar a pagina base (so com --session-baseline), q/Esc = sair.
"""

import argparse
import glob
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from . import hand as hand_module
from . import ink as ink_module
try:
    from . import ink_model as ink_model_module
except ImportError:
    ink_model_module = None
from . import neural as neural_module
from .pipeline import (
    DrawPipeline,
    _INK_CORNER_HISTORY,
    _SHEET_BLEND_ALPHA,
    _SHEET_DETECT_INTERVAL,
    _SHEET_JUMP_PERSIST_SECONDS,
    _SHEET_LOST_MIN_FRAMES,
    _SHEET_LOST_SECONDS,
    _SHEET_NEAR_LIMIT_PX,
    _SheetTracker,
    _blend_corners,
    _corner_distance,
    _detect_sheet_combined,
)
from .protocol import HelperCapture, is_helper_source

_CAPTURE_MAX_FAILURES = 15
_READ_RETRY_SECONDS = 0.005
_PNG_DRAIN_SECONDS = 5.0
_DEAD_NOTICE_MS = 3000


class _LatestFrameCapture:
    """Captura em thread separada que guarda so o quadro mais novo.

    Uma thread le da captura sem parar e publica cada quadro com o
    instante de captura (ms monotonicos, tirado logo que o read()
    retorna), um numero de sequencia e a contagem de quadros
    descartados (publicados e substituidos antes de alguem consumir).
    Quem consome espera por sequencia nova numa Condition: nao gira
    em vazio e nunca reprocessa o mesmo quadro. Os quadros que o
    driver empilha enquanto o laco processa nao viram fila aqui: o
    consumidor pega sempre o mais novo, e e isso que tira o teto da
    latencia de captura.

    Falha de leitura tem um equivalente declarado do comportamento
    antigo. O laco antigo contava as falhas seguidas de read() e, com
    _CAPTURE_MAX_FAILURES, imprimia o aviso e saia; abaixo do limite,
    girava sem imshow e sem waitKey, com a janela congelada. Aqui a
    thread conta as falhas seguidas e, no limite, marca a fonte como
    morta e acorda quem espera; quem espera recebe None e o laco
    imprime o mesmo aviso e sai. Abaixo do limite, nenhum quadro novo
    chega e o consumidor fica bloqueado na espera: janela congelada e
    teclas sem resposta, como antes. Uma leitura boa zera a contagem.
    """

    def __init__(self, capture):
        self._capture = capture
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._pump, name="draw-capture", daemon=True
        )
        self._frame = None
        self._seq = 0
        self._taken_seq = 0
        self._capture_ms = 0
        self._dropped = 0
        self._failures = 0
        self._dead = False
        self._stopped = False

    def start(self):
        self._thread.start()
        return self

    def _pump(self):
        try:
            self._pump_loop()
        finally:
            with self._condition:
                if not self._stopped:
                    self._dead = True
                self._condition.notify_all()

    def _pump_loop(self):
        while True:
            with self._condition:
                if self._stopped:
                    return
            ok, frame = self._capture.read()
            if ok and frame is not None:
                source_ts = getattr(self._capture, "last_capture_ts_ms", None)
                capture_ms = (
                    int(source_ts)
                    if source_ts is not None
                    else int(time.monotonic() * 1000)
                )
                with self._condition:
                    if self._stopped:
                        return
                    if self._seq > self._taken_seq:
                        self._dropped += 1
                    self._seq += 1
                    self._frame = frame
                    self._capture_ms = capture_ms
                    self._failures = 0
                    self._condition.notify_all()
            else:
                with self._condition:
                    if self._stopped:
                        return
                    self._failures += 1
                    if self._failures >= _CAPTURE_MAX_FAILURES:
                        self._dead = True
                        self._condition.notify_all()
                        return
                time.sleep(_READ_RETRY_SECONDS)

    def wait_next(self, after_seq, timeout=None):
        """Espera um quadro com sequencia maior que after_seq.

        Devolve (frame, capture_ms, seq). Devolve None quando a
        fonte morreu, quando stop() foi chamado ou quando o timeout
        vence sem quadro novo; quem chama distingue por dead.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._seq > after_seq or self._dead or self._stopped
                ),
                timeout=timeout,
            )
            if self._seq > after_seq:
                self._taken_seq = self._seq
                return self._frame, self._capture_ms, self._seq
            return None

    @property
    def dead(self):
        with self._condition:
            return self._dead

    @property
    def dropped(self):
        with self._condition:
            return self._dropped

    @property
    def failures(self):
        with self._condition:
            return self._failures

    def stop(self):
        """Para a thread e libera a captura.

        O release() tenta destravar um read() pendurado no driver.
        Se mesmo assim a thread nao voltar, o join tem teto e a
        thread e daemon: a saida do app nao depende dela.
        """
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._capture.release()
        self._thread.join(timeout=2.0)


class _PngWriter:
    """Escritora de PNG em thread unica, para o imwrite sair do laco.

    Medido no laboratorio (draw-lab/bottleneck-2026-10-30/loop): o
    imwrite sincrono de um frame cru custa 50 a 71 ms (zlib) e o do
    canvas, 10 a 12 ms; dentro do laco, cada salvamento era um
    engasgo inteiro. save() copia a imagem e enfileira; a escritora
    grava em ordem e roda o callback de confirmacao depois da
    escrita, que e onde moram o contador e o print de "salvo".
    close() enfileira o fim da fila e espera a drenagem com teto de
    _PNG_DRAIN_SECONDS: a saida nao trava num disco patologico, e o
    que nao drenar a tempo fica por conta da thread daemon.
    """

    def __init__(self):
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._work, name="draw-png-writer", daemon=True
        )
        self._thread.start()

    def save(self, path, image, on_done=None):
        self._queue.put((Path(path), image.copy(), on_done))

    def _work(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, image, on_done = item
            cv2.imwrite(str(path), image)
            if on_done is not None:
                on_done(path)

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=_PNG_DRAIN_SECONDS)



def parse_source(value):
    """'0'/'1' viram indice de webcam local; qualquer outra coisa e URL/arquivo."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def is_wsl():
    """Detecta WSL pelo nome do kernel em /proc (ex.: '...-microsoft-standard-WSL2')."""
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _configure_local_capture(capture):
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def _local_open_hint(source):
    """Diagnostico por porta: nomeia a primeira condicao que impede a abertura."""
    node = f"/dev/video{source}"
    if not sys.platform.startswith("linux"):
        return (
            "Nenhuma webcam local respondeu nesse indice. "
            "Confira se a webcam esta conectada e livre."
        )
    if not os.path.exists(node):
        hint = f"A camera nao esta conectada a este sistema: {node} nao existe. "
        if is_wsl():
            hint += (
                "Anexe a camera ao WSL com: bash scripts/setup_camera_wsl.sh. "
                "A alternativa e uma camera via rede: python main.py "
                "--source tcp://<ip-do-windows>:8765 ou http://<ip>:8080/video."
            )
        else:
            hint += "Confira se a webcam esta conectada a este computador."
        return hint
    if not os.access(node, os.R_OK | os.W_OK):
        return (
            f"{node} existe, mas este processo nao tem permissao de "
            "leitura e escrita nele. A porta que falhou e a permissao. "
            "Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes). "
            'Correcao nesta sessao: sg video -c "python3 main.py".'
        )
    return (
        f"{node} existe e este processo tem permissao nele, mas a "
        "abertura falhou. Outro programa provavelmente esta usando a "
        "camera. Feche esse programa e tente de novo."
    )


def _open_capture(source):
    if is_helper_source(source):
        helper = HelperCapture(source)
        if helper.isOpened():
            return helper
        raise RuntimeError(
            f"Nao consegui abrir a fonte de protocolo Draw: {source!r}. "
            "Confira se o arquivo .drawcap existe ou se o helper Windows "
            "esta ouvindo na porta TCP informada."
        )
    capture = cv2.VideoCapture(source)
    if capture.isOpened():
        if isinstance(source, int):
            _configure_local_capture(capture)
        return capture
    hint = f"Nao consegui abrir a fonte de video: {source!r}. "
    if isinstance(source, int):
        hint += _local_open_hint(source)
    else:
        hint += (
            "Confira se a URL/caminho esta certo, se a camera esta "
            "transmitindo e se este computador alcanca o endereco na rede."
        )
    raise RuntimeError(hint)


def _video_nodes():
    return sorted(
        path
        for path in glob.glob("/dev/video*")
        if path[len("/dev/video") :].isdigit()
    )


def run_check(source=0):
    """Validacao da fonte de video porta por porta. Retorna 0 ou 1."""
    print(f"PASS: OpenCV disponivel. Versao {cv2.__version__}.")
    parsed = parse_source(source)

    if not isinstance(parsed, int):
        capture = (
            HelperCapture(parsed, timeout=3.0)
            if is_helper_source(parsed)
            else cv2.VideoCapture(parsed)
        )
        if not capture.isOpened():
            print(
                f"FAIL: nao consegui abrir a fonte {parsed!r}. "
                "Confira o caminho ou URL e se a transmissao esta ativa."
            )
            return 1
        frame = None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
                break
            time.sleep(0.05)
        capture.release()
        if frame is None:
            print(
                f"FAIL: {parsed!r} abriu, mas nenhum quadro chegou em "
                "3 segundos."
            )
            return 1
        extra = ""
        if isinstance(capture, HelperCapture) and capture.last_capture_ts_ms is not None:
            extra = (
                f" Protocolo Draw: frame_id={capture.last_frame_id}, "
                f"capture_ts_ms={capture.last_capture_ts_ms}."
            )
        print(
            f"PASS: quadro lido de {parsed!r}. "
            f"Formato (altura, largura, canais): {frame.shape}.{extra}"
        )
        return 0

    nodes = _video_nodes()
    if not nodes:
        print("FAIL: nenhum /dev/video* existe neste sistema.")
        if is_wsl():
            print(
                "  Correcao: anexe a camera com "
                "bash scripts/setup_camera_wsl.sh. A alternativa e uma "
                "camera via rede com --source."
            )
        else:
            print("  Correcao: conecte uma webcam a este computador.")
        return 1
    print(f"PASS: dispositivos de video encontrados: {', '.join(nodes)}.")

    all_ok = True
    node = f"/dev/video{parsed}"
    if node not in nodes:
        print(
            f"FAIL: {node} nao existe neste sistema "
            f"(disponiveis: {', '.join(nodes)})."
        )
        first_idx = nodes[0][len("/dev/video") :]
        print(
            f"  Correcao: use --source {first_idx} ou confira o indice "
            "da webcam desejada."
        )
        return 1

    access = os.access(node, os.R_OK | os.W_OK)
    group_note = ""
    try:
        import grp

        gid = os.stat(node).st_gid
        try:
            group_name = grp.getgrgid(gid).gr_name
        except KeyError:
            group_name = str(gid)
        in_group = gid in os.getgroups()
        group_note = (
            f" Grupo dono de {node}: {group_name}. "
            f"Este processo esta nesse grupo: {'sim' if in_group else 'nao'}."
        )
    except Exception:
        group_note = ""
    if access:
        print(f"PASS: este processo le e escreve em {node}.{group_note}")
    else:
        print(
            f"FAIL: este processo nao tem permissao de leitura e "
            f"escrita em {node}.{group_note}"
        )
        print(
            "  Correcao definitiva: sudo usermod -aG video $USER "
            "(vale nas proximas sessoes)."
        )
        print('  Correcao nesta sessao: sg video -c "python3 main.py".')
        all_ok = False

    capture = cv2.VideoCapture(parsed)
    if not capture.isOpened():
        print(
            f"FAIL: nao consegui abrir {node}. Outro programa "
            "provavelmente esta usando a camera."
        )
        return 1
    _configure_local_capture(capture)
    frame = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)
    capture.release()
    if frame is not None:
        print(
            f"PASS: quadro lido de {node}. "
            f"Formato (altura, largura, canais): {frame.shape}."
        )
    else:
        print(
            f"FAIL: {node} abriu, mas nenhum quadro chegou em 3 "
            "segundos. A camera pode estar ocupada ou travada."
        )
        all_ok = False

    return 0 if all_ok else 1


def _fit(image, max_h=700):
    h, w = image.shape[:2]
    if h <= max_h:
        return image.copy()
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


_BLANK_SHEET = None


def _blank_sheet():
    global _BLANK_SHEET
    if _BLANK_SHEET is None:
        _BLANK_SHEET = ink_module.render_canvas(
            np.zeros((config.PAPER_H, config.PAPER_W), dtype=np.uint8)
        )
    return _BLANK_SHEET


def _draw_cursor_overlay(sheet_img, cursor):
    """Desenha indicador de cursor apenas na cópia de visualização."""
    if cursor is None or not cursor.has_position or cursor.position_stale:
        return
    if cursor.x is None or cursor.y is None:
        return
    sh, sw = sheet_img.shape[:2]
    cx = int(round(cursor.x * sw / float(config.PAPER_W)))
    cy = int(round(cursor.y * sh / float(config.PAPER_H)))
    if not (0 <= cx < sw and 0 <= cy < sh):
        return
    if cursor.pen_down:
        cv2.circle(sheet_img, (cx, cy), 5, (20, 160, 40), -1)
    else:
        cv2.circle(sheet_img, (cx, cy), 6, (200, 120, 20), 2)


def _compose_view(camera_img, canvas_img, cursor=None):
    """Janela unica: camera a esquerda, folha ao vivo a direita."""
    target_h = 640
    cam = _fit(camera_img, target_h)
    sheet = _fit(canvas_img, target_h)
    if cam.shape[0] != sheet.shape[0]:
        h = min(cam.shape[0], sheet.shape[0])
        cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
        sheet = cv2.resize(
            sheet, (int(sheet.shape[1] * h / sheet.shape[0]), h)
        )
    _draw_cursor_overlay(sheet, cursor)
    cv2.putText(
        sheet,
        "folha ao vivo",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (140, 140, 140),
        1,
    )
    divider = np.full((cam.shape[0], 6, 3), (60, 60, 60), dtype=np.uint8)
    view = np.hstack([cam, divider, sheet])
    if view.shape[:2] != (640, 1600):
        view = cv2.resize(view, (1600, 640))
    return view


def _dead_source_view(last_view):
    """Quadro de aviso da fonte morta, desenhado sobre a ultima vista.

    Sem nenhum quadro exibido ainda, o aviso vai sobre um fundo
    solido no formato da vista composta. O texto usa a mesma
    mensagem do aviso no console, sem acentos, como o restante dos
    textos que o app desenha com a fonte Hershey.
    """
    if last_view is not None:
        view = last_view.copy()
    else:
        view = np.full((640, 1280, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(
        view,
        "A fonte de video morreu",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
    )
    cv2.putText(
        view,
        "A camera parou de entregar quadros.",
        (40, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        view,
        "Feche outros programas usando a camera e tente de novo.",
        (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return view


def _detect_sheet_combined(frame, neural_detector):
    """Detecção do modo sem marcadores: neural primário, clássico reserva.

    Sem detector neural, é o clássico puro, o comportamento de sempre.
    Com detector, ele decide primeiro; fechando o quadrilátero, vale o
    veredito dele. Não fechando, a reserva clássica só fala quando a
    neural não negou a folha com força: logit de presença até
    neural.PRESENCE_VETO_LOGIT devolve (None, None) direto, sem deixar
    o clássico declarar "paper" nem "blank_maybe" contra um "não há
    papel" forte do modelo. É o portão do incidente da caixa branca
    (2026-10-29): a v2 media logit ≈ −10,9 na caixa e o "Folha
    detectada" ao vivo vinha do clássico. A justificativa medida do
    limiar mora em draw.neural.PRESENCE_VETO_LOGIT. Logit negativo
    fraco (acima do limiar) não veta: a reserva segue valendo para a
    folha real que a neural subdispara num setup novo.

    Terceiro desfecho: se nenhum quadrilátero fechou (o ajuste neural
    foi recusado pela guarda de plausibilidade, ou não fechou, e o
    clássico também não achou candidato) mas o logit de presença é
    positivo, devolve (None, "paper"): presença sem posição. O
    rastreador lê isso como "a folha está lá, segura o quadrilátero
    anterior e renova o relógio de perda", em vez de contar o quadro
    como ausência (medido no replay do laboratório em 2026-10-30:
    sem esse canal, cada ajuste recusado vira contagem de perda e a
    folha cai durante os episódios de oclusão).
    """
    if neural_detector is not None:
        corners, verdict = neural_detector.detect_sheet_with_confidence(frame)
        if corners is not None:
            return corners, verdict
        logit = neural_detector.last_presence_logit
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return None, None
        corners, verdict = paper_module.detect_sheet_with_confidence(frame)
        if corners is None and logit is not None and logit > 0:
            return None, "paper"
        return corners, verdict
    return paper_module.detect_sheet_with_confidence(frame)



def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    neural_model=None,
    hand_model=None,
    ink_model=None,
    session_baseline=False,
    async_detect=False,
):
    neural_detector = None
    if neural_model is not None and not markers_mode:
        neural_detector = neural_module.NeuralSheetDetector(neural_model)
        neural_detector.load()
        print(
            f"Detector neural ativo: {neural_model}. "
            "O classico cobre os frames em que o neural nao ve folha."
        )

    ink_extractor = None
    if ink_model is not None and not markers_mode:
        if ink_model_module is None:
            print(
                "Modelo de tinta solicitado, mas o modulo "
                "draw/ink_model.py nao esta presente neste pacote. "
                "Seguindo com a heuristica."
            )
        else:
            ink_extractor = ink_model_module.InkModelExtractor(ink_model)
            if ink_extractor.load():
                print(
                    f"Modelo de tinta ativo: {ink_model}. "
                    "A heuristica segue como reserva por frame."
                )

    if session_baseline and not markers_mode:
        print(
            "Tinta da sessao ativa: com a folha na cena, aperte C "
            "para congelar a pagina base; so o traco feito depois "
            "dela entra no canvas."
        )

    hand_masker = None
    if hand_model is not None and not markers_mode:
        hand_masker = hand_module.HandMasker(hand_model or None)
        print(
            "Mascara de mao solicitada: mao sobre a folha vira regiao "
            "desconhecida no canvas acumulativo e alimenta fusao do cursor."
        )

    parsed_source = parse_source(source)
    capture = _open_capture(parsed_source)
    is_replay_file = isinstance(parsed_source, str) and (
        parsed_source.lower().endswith(".drawcap")
        or Path(parsed_source).is_file()
    )

    pipeline = DrawPipeline(
        markers_mode=markers_mode,
        neural_detector=neural_detector,
        hand_masker=hand_masker,
        ink_extractor=ink_extractor,
        session_baseline=session_baseline,
        async_detect=async_detect,
    )
    frame_source = _LatestFrameCapture(capture).start()
    png_writer = _PngWriter()
    canvas_buffer = np.empty(
        (config.PAPER_H, config.PAPER_W, 3), dtype=np.uint8
    )

    save_path = Path(save_dir)
    if markers_mode:
        print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    else:
        print("Draw v0 rodando. Mostre uma folha para a camera.")
    print(
        "Teclas: s = salvar canvas, f = capturar frame, "
        "g = captura automatica (desligada), c = pagina base, q/Esc = sair."
    )

    frames_path = save_path / "frames"
    frames_saved = 0
    auto_capture = False
    loop_frames = 0
    last_seq = 0
    last_view = None

    def save_raw_frame(img):
        nonlocal frames_saved
        frames_path.mkdir(parents=True, exist_ok=True)
        out = frames_path / f"frame_{int(time.time() * 1000)}.png"

        def _saved(path):
            nonlocal frames_saved
            frames_saved += 1
            print(f"Frame salvo em {path} (total: {frames_saved}).")

        png_writer.save(out, img, _saved)

    def handle_key(key, img, current_canvas):
        nonlocal auto_capture
        if key == ord("s"):
            def _saved(path):
                print(f"Canvas salvo em {path}")

            png_out, npz_out = pipeline.save_artifacts(
                save_path,
                canvas_override=current_canvas,
                png_writer=png_writer,
                on_png_saved=_saved,
            )
            if png_out is not None:
                if npz_out is not None:
                    print(f"Carimbos salvos em {npz_out}")
        elif key == ord("f"):
            save_raw_frame(img)
        elif key == ord("c") and pipeline.trigger_baseline():
            print(
                "Pagina base pedida: capturando os proximos quadros "
                "retificados. Maos fora da folha."
            )
        elif key == ord("g"):
            auto_capture = not auto_capture
            if auto_capture:
                print(
                    "Captura automatica ligada: um frame a cada "
                    f"10 quadros vai para {frames_path}."
                )
            else:
                print("Captura automatica desligada.")

    try:
        while True:
            item = frame_source.wait_next(last_seq)
            if item is None:
                if is_replay_file:
                    break
                print(
                    "A camera abriu, mas parou de entregar quadros. "
                    "Feche outros programas usando a camera e tente de novo."
                )
                last_view = _dead_source_view(last_view)
                cv2.imshow("Draw", last_view)
                cv2.waitKey(_DEAD_NOTICE_MS)
                break
            frame, frame_capture_ms, last_seq = item

            now_s = time.monotonic()

            loop_frames += 1
            if auto_capture and loop_frames % 10 == 0:
                save_raw_frame(frame)

            res = pipeline.process_frame(
                frame,
                t_capture_ms=frame_capture_ms,
                now_s=now_s,
                canvas_buffer=canvas_buffer,
            )

            status = frame.copy()
            if markers_mode:
                for x, y in res.markers.values():
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            elif res.corners is not None:
                outline = res.corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(status, [outline], True, (0, 255, 0), 2)
                for x, y in res.corners:
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)

            if not res.sheet_present:
                status_text = (
                    f"Mostre a folha-modelo ({len(res.markers)}/4 marcadores)"
                    if markers_mode
                    else "Mostre uma folha para a camera"
                )
                cv2.putText(
                    status,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                panel = res.canvas if res.canvas is not None else _blank_sheet()
                last_view = _compose_view(status, panel, res.cursor)
                cv2.imshow("Draw", last_view)
            else:
                if not markers_mode:
                    if res.verdict == "blank_maybe":
                        status_text = "Folha detectada (baixa confianca)"
                        status_color = (0, 255, 255)
                    else:
                        status_text = "Folha detectada"
                        status_color = (0, 255, 0)
                    if (
                        pipeline.session_ink is not None
                        and not pipeline.session_ink.ready
                    ):
                        if pipeline.baseline_armed:
                            status_text = (
                                "Folha detectada: capturando a pagina "
                                "base, maos fora da folha"
                            )
                        else:
                            status_text = (
                                "Folha detectada: aperte C para comecar "
                                "(pagina base)"
                            )
                        status_color = (0, 255, 255)
                    cv2.putText(
                        status,
                        status_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2,
                    )
                panel = res.canvas if res.canvas is not None else _blank_sheet()
                last_view = _compose_view(status, panel, res.cursor)
                cv2.imshow("Draw", last_view)

            key = cv2.waitKey(1) & 0xFF
            handle_key(key, frame, res.canvas)
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.close()
        frame_source.stop()
        png_writer.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Draw v0, papel para o digital"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="indice da webcam local (0, 1...), URL MJPEG/RTSP, "
        "tcp://host[:8765] do helper Windows ou arquivo .drawcap. "
        "Padrao: 0",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="usa a folha-modelo impressa com marcadores ArUco",
    )
    parser.add_argument(
        "--neural",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="detector neural (ONNX) como primario, com o classico de "
        "reserva. Sem caminho, usa a variavel de ambiente "
        "DRAW_NEURAL_MODEL",
    )
    parser.add_argument(
        "--async-detect",
        action="store_true",
        help="executa a deteccao da folha numa thread dedicada em "
        "segundo plano, desacoplando o custo do ONNX da cadencia de "
        "extracao de tinta e fusao do cursor",
    )
    parser.add_argument(
        "--ink-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo aprendido de tinta (ONNX) opcional; sem modulo "
        "presente, o app avisa uma vez e segue com a heuristica",
    )
    parser.add_argument(
        "--hand-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.task",
        help="mascara de mao (MediaPipe Hand Landmarker) para o "
        "canvas acumulativo e fusao do cursor. Sem caminho, usa "
        "DRAW_HAND_MODEL ou models/hand_landmarker.task",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="valida a fonte de captura e sai, sem abrir janela",
    )
    parser.add_argument(
        "--session-baseline",
        action="store_true",
        help="tinta da sessao: a pagina como esta quando voce aperta "
        "a tecla C vira fundo; so o traco feito depois entra. "
        "Incompativel com --markers",
    )
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    if args.check:
        sys.exit(run_check(parse_source(args.source)))
    neural_model = None
    if args.neural is not None:
        if args.markers:
            parser.error("--neural nao combina com --markers")
        neural_model = args.neural or os.environ.get("DRAW_NEURAL_MODEL")
        if not neural_model:
            parser.error(
                "--neural sem caminho: passe o arquivo .onnx ou defina "
                "a variavel DRAW_NEURAL_MODEL"
            )
    ink_model = None
    if args.ink_model is not None:
        if args.markers:
            parser.error("--ink-model nao combina com --markers")
        ink_model = args.ink_model or os.environ.get("DRAW_INK_MODEL")
        if not ink_model:
            parser.error(
                "--ink-model sem caminho: passe o arquivo .onnx ou "
                "defina a variavel DRAW_INK_MODEL"
            )
    if args.session_baseline and args.markers:
        parser.error("--session-baseline nao combina com --markers")
    if args.async_detect and args.markers:
        parser.error("--async-detect nao combina com --markers")
    hand_model = None
    if args.hand_model is not None:
        if args.markers:
            parser.error("--hand-model nao combina com --markers")
        hand_model = (
            args.hand_model or os.environ.get("DRAW_HAND_MODEL") or ""
        )
    try:
        run(
            source=args.source,
            save_dir=args.save_dir,
            markers_mode=args.markers,
            neural_model=neural_model,
            hand_model=hand_model,
            ink_model=ink_model,
            session_baseline=args.session_baseline,
            async_detect=args.async_detect,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
