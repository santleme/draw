"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a camera e sai
  python main.py --markers                        # folha-modelo impressa
  python main.py --neural modelo.onnx             # detector neural primario
  python main.py --ink-model modelo.onnx          # modelo aprendido de tinta
  python main.py --hand-model modelo.task         # mascara de mao no canvas
  python main.py --source 1                       # outra webcam local
  python main.py --source http://192.168.0.10:8080/video   # camera via rede

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames,
c = congelar a pagina base (so com --session-baseline), q/Esc = sair.
"""

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import hand as hand_module
from . import ink as ink_module
from . import capture as capture_module
from . import diagnose as diagnose_module
from . import light as light_module
from . import light_guard as light_guard_module
try:
    from . import ink_model as ink_model_module
except ImportError:
    # O extrator de tinta por modelo aprendido e opcional e o modulo
    # dele ainda nao faz parte do pacote publicado. Sem o modulo,
    # --ink-model avisa e o app segue so com a heuristica.
    ink_model_module = None
from . import neural as neural_module
from . import paper as paper_module


_SHEET_NEAR_LIMIT_PX = 40.0
_SHEET_BLEND_ALPHA = 0.3
# Confirmação de salto por TEMPO, não por contagem de detecções
# (medido em 2026-10-30 no laboratório,
# draw-lab/noite-2026-10-30/block6/): com a contagem de 3, um salto
# selvagem confirmava em ~0,1 s na cadência densa de "missed" (a
# detecção roda a cada quadro), e os vídeos de replay mediam 64
# passos oficiais acima de 100 px em 1.857 quadros. Exigir que o
# candidato persista 0,3 s de tempo de vídeo (o dobro para
# blank_maybe, como era o dobro na contagem) derruba os passos
# >100 px para 15 e o jitter médio da grade de 29,39 para 9,66 px,
# sem nenhuma queda nova e com o atraso de remoção intacto em
# 0,53 s. Ressalva: a medição é sobre gravações de tela com overlay
# queimado; reconfirmar em captura crua.
_SHEET_JUMP_PERSIST_SECONDS = 0.3
_SHEET_DETECT_INTERVAL = 5
# Perda da folha medida em tempo, nao em quadros: com a contagem em
# quadros, o mesmo limite valia ~1 s a 30 fps e ~3 s numa sessao lenta,
# e era esse o banner que ficava sobre a mesa vazia. O piso em quadros
# impede a queda entre duas deteccoes agendadas quando o loop roda
# devagar: a folha presente confirma de novo antes dos dois criterios.
_SHEET_LOST_SECONDS = 0.5
_SHEET_LOST_MIN_FRAMES = 10
# A tinta e lida numa geometria estabilizada: a mediana dos ultimos
# conjuntos de cantos do rastreador. O jitter de canto entre deteccoes
# (media de ~5 px, com saltos de dezenas de pixels quando a mao cobre
# um canto) desloca o traco de 1 a 3 px no espaco do papel e desfaz o
# voto por pixel do canvas acumulativo; a mediana de 5 segura a
# geometria no lugar por tempo bastante para a janela do acumulador.
_INK_CORNER_HISTORY = 5
# Limite de falhas seguidas de leitura que declara a fonte morta.
# Era a contagem do laco principal (15 read() falhos seguidos e o app
# avisava e saia); com a leitura numa thread propria, a contagem mora
# na thread e o desfecho visivel e o mesmo.
_CAPTURE_MAX_FAILURES = 15
# Pausa entre tentativas de leitura no caminho de falha. Sem ela, um
# driver que devolve False na hora faz a thread girar uma CPU inteira
# enquanto a camera nao volta; o laco antigo girava sem dormir.
_READ_RETRY_SECONDS = 0.005
# Teto da espera pela drenagem da fila de PNG na saida do app.
_PNG_DRAIN_SECONDS = 5.0
# Tempo que o aviso de fonte morta fica na janela antes da saida.
# Sem ele, a morte da fonte e um congelamento silencioso. O waitKey
# devolve antes se qualquer tecla for apertada.
_DEAD_NOTICE_MS = 3000


def _corner_distance(current, detected):
    return float(np.max(np.linalg.norm(current - detected, axis=1)))


def _blend_corners(current, detected):
    return (
        (1.0 - _SHEET_BLEND_ALPHA) * current
        + _SHEET_BLEND_ALPHA * detected
    ).astype(np.float32)


class _SheetTracker:
    """Estabilizacao temporal do quadrilatero da folha.

    Guarda o quadrilatero atual e decide, quadro a quadro, entre
    misturar uma deteccao vizinha, exigir persistencia para um salto
    de posicao e descartar a folha perdida. Deteccao perto mistura
    com _SHEET_BLEND_ALPHA; salto (deteccao a mais de
    _SHEET_NEAR_LIMIT_PX do quad atual) so confirma quando o
    candidato persiste _SHEET_JUMP_PERSIST_SECONDS de relogio com
    deteccoes seguintes pousando perto dele (o dobro para
    blank_maybe). A perda dispara quando os dois criterios de
    _SHEET_LOST_* passam: tempo sem confirmacao e quadros sem
    confirmacao.

    Presenca e posicao sao perguntas separadas no relogio de perda:
    enquanto deteccoes "paper" plausiveis continuam chegando, a
    folha esta la, mesmo com a posicao incerta. Por isso um salto
    ainda nao confirmado com veredito "paper" renova o relogio de
    perda sem mover o quadrilatero exibido, e um quadro com
    veredito "paper" sem quadrilatero (o modelo afirmou papel e
    nenhum ajuste fechou) tambem renova. Sem isso, a espera da
    confirmacao por tempo esfomeia o relogio e a folha cai no meio
    da janela da mao (medido no replay: 11 quadros sem folha e
    atraso de remocao de 0,30 s, contra 0,53 s com a renovacao).
    """

    def __init__(self):
        self.corners = None
        self.verdict = None
        self.missed = False
        self._pending = None
        self._pending_hits = 0
        self._pending_since = None
        self._frames_since_seen = 0
        self._last_seen_at = None

    def should_detect(self, frame_number):
        return (
            self.corners is None
            or self.missed
            or frame_number % _SHEET_DETECT_INTERVAL == 0
        )

    def _confirm(self, corners, verdict, now):
        self.corners = corners
        self.verdict = verdict
        self._pending = None
        self._pending_hits = 0
        self._pending_since = None
        self._frames_since_seen = 0
        self._last_seen_at = now
        self.missed = False

    def update(self, detected, verdict, ran_detection, now):
        """Alimenta o rastreador com o resultado do quadro.

        detected e o quadrilatero detectado ou None; ran_detection
        diz se a deteccao rodou neste quadro; now e um relogio
        monotonico em segundos. Retorna "seen" quando uma deteccao
        confirma a folha, "drop" quando a folha e descartada e None
        nos demais quadros (so para diagnostico).
        """
        event = None
        if (
            detected is None
            and ran_detection
            and verdict == "paper"
            and self.corners is not None
        ):
            # Evento so de presenca: quem detectou afirma papel mas
            # nenhum quadrilatero utilizavel fechou (o ajuste neural
            # foi recusado pela guarda de plausibilidade, ou nao
            # fechou). A folha esta la; a posicao e que nao veio.
            # Renova o relogio de perda, nao mexe no quadrilatero
            # exibido nem no candidato pendente.
            self._frames_since_seen = 0
            self._last_seen_at = now
            self.missed = True
            return None
        if detected is not None:
            if self.corners is None and verdict == "paper":
                self._confirm(detected.copy(), verdict, now)
                event = "seen"
            elif (
                self.corners is not None
                and _corner_distance(self.corners, detected)
                <= _SHEET_NEAR_LIMIT_PX
            ):
                self._confirm(
                    _blend_corners(self.corners, detected), verdict, now
                )
                event = "seen"
            else:
                # Salto de posicao ou candidato ainda sem trava.
                # Candidato blank_maybe (branco e liso, sem textura
                # de papel) exige o dobro do tempo de persistencia.
                confirm_seconds = (
                    2 * _SHEET_JUMP_PERSIST_SECONDS
                    if verdict == "blank_maybe"
                    else _SHEET_JUMP_PERSIST_SECONDS
                )
                if (
                    self._pending is not None
                    and _corner_distance(self._pending, detected)
                    <= _SHEET_NEAR_LIMIT_PX
                ):
                    self._pending = _blend_corners(self._pending, detected)
                    self._pending_hits += 1
                else:
                    self._pending = detected.copy()
                    self._pending_hits = 1
                    self._pending_since = now

                if (
                    self._pending_since is not None
                    and now - self._pending_since >= confirm_seconds
                ):
                    self._confirm(self._pending.copy(), verdict, now)
                    event = "seen"
                else:
                    if verdict == "paper":
                        # Presenca renovada, posicao ainda incerta:
                        # o relogio de perda nao corre contra uma
                        # folha que o detector continua vendo.
                        self._frames_since_seen = 0
                        self._last_seen_at = now
                    else:
                        self._frames_since_seen += 1
                    self.missed = True
        elif self.corners is not None:
            if ran_detection:
                self._pending = None
                self._pending_hits = 0
                self._pending_since = None
                self.missed = True
            self._frames_since_seen += 1

        if (
            self.corners is not None
            and self._frames_since_seen > _SHEET_LOST_MIN_FRAMES
            and self._last_seen_at is not None
            and now - self._last_seen_at > _SHEET_LOST_SECONDS
        ):
            self.corners = None
            self.verdict = None
            self._pending = None
            self._pending_hits = 0
            self._pending_since = None
            self._frames_since_seen = 0
            self.missed = False
            event = "drop"
        return event


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
        self._last_read_error = None
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
            read_error = None
            try:
                ok, frame = self._capture.read()
            except Exception as error:
                read_error = error
                ok, frame = False, None
            if ok and frame is not None:
                capture_ms = int(time.monotonic() * 1000)
                with self._condition:
                    if self._stopped:
                        return
                    if self._seq > self._taken_seq:
                        self._dropped += 1
                    self._seq += 1
                    self._frame = frame
                    self._capture_ms = capture_ms
                    self._failures = 0
                    self._last_read_error = None
                    self._condition.notify_all()
            else:
                with self._condition:
                    if self._stopped:
                        return
                    self._failures += 1
                    if read_error is not None:
                        self._last_read_error = read_error
                    reconnect_needed = (
                        self._failures >= _CAPTURE_MAX_FAILURES
                    )
                if reconnect_needed:
                    reconnect = getattr(
                        self._capture, "try_reconnect", None
                    )
                    if reconnect is not None and reconnect():
                        with self._condition:
                            if self._stopped:
                                return
                            self._failures = 0
                        continue
                    with self._condition:
                        if self._stopped:
                            return
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

    @property
    def last_error(self):
        with self._condition:
            read_error = self._last_read_error
        return read_error or getattr(self._capture, "last_error", None)

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
    """Delega para draw/capture.py, mantendo a CLI compatível."""
    return capture_module.parse_source(value)


def is_wsl():
    return capture_module.is_wsl()


def _configure_local_capture(capture, width=None, height=None, fps=None):
    capture_module.apply_local_settings(
        capture, width=width, height=height, fps=fps
    )


def _local_open_hint(source):
    return capture_module.diagnose_local(source)[1]


def _open_capture(source, **kwargs):
    return capture_module.open_capture(source, **kwargs)


def _video_nodes():
    return capture_module.video_nodes()


def _read_first_frame(capture, timeout_s):
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            ok, frame = capture.read()
        except Exception as error:
            return None, error
        if ok and frame is not None:
            return frame, None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, TimeoutError("nenhum frame recebido dentro do prazo")
        time.sleep(min(0.05, remaining))


def run_check(source=0, open_timeout_ms=3000, first_frame_timeout_s=5.0):
    """Valida a fonte informada, lê um frame e sempre libera a captura."""
    parsed = parse_source(source)
    print(f"PASS: OpenCV disponivel. Versao {cv2.__version__}.")
    print(f"Testando fonte: {parsed!r}.")
    capture = None
    try:
        read_timeout_ms = max(1, int(first_frame_timeout_s * 1000))
        capture = _open_capture(
            parsed,
            open_timeout_ms=open_timeout_ms,
            read_timeout_ms=read_timeout_ms,
        )
        if not capture.isOpened():
            print(f"FAIL: a fonte {parsed!r} nao abriu.")
            return 1
        print(f"PASS: fonte {parsed!r} aberta.")
        frame, read_error = _read_first_frame(capture, first_frame_timeout_s)
        if read_error is not None:
            if isinstance(read_error, TimeoutError):
                print(
                    f"FAIL: a fonte {parsed!r} abriu, mas nao entregou "
                    f"um frame antes do prazo de {first_frame_timeout_s:g} "
                    "segundos. O backend pode ultrapassar o prazo se "
                    "capture.read() bloquear."
                )
            else:
                print(f"FAIL: erro ao ler frame da fonte {parsed!r}: {read_error}")
            return 1
        height_px, width_px = frame.shape[:2]
        try:
            fps_value = float(capture.get(cv2.CAP_PROP_FPS))
        except Exception:
            fps_value = 0.0
        fps_text = f"{fps_value:g} FPS" if fps_value > 0 else "FPS indisponivel"
        print(
            f"PASS: primeiro frame recebido: {width_px}x{height_px}, "
            f"{fps_text}."
        )
        return 0
    except capture_module.CaptureError as error:
        print(f"FAIL: {error}")
        return 1
    except Exception as error:
        print(f"FAIL: erro ao testar a fonte {parsed!r}: {error}")
        return 1
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


def _fit(image, max_h=700):
    h, w = image.shape[:2]
    if h <= max_h:
        return image
    scale = max_h / h
    return cv2.resize(image, (int(w * scale), max_h))


_BLANK_SHEET = None


def _blank_sheet():
    global _BLANK_SHEET
    if _BLANK_SHEET is None:
        _BLANK_SHEET = ink_module.render_canvas(
            np.zeros(
                (ink_module.config.PAPER_H, ink_module.config.PAPER_W),
                dtype=np.uint8,
            )
        )
    return _BLANK_SHEET


def _compose_view(camera_img, canvas_img):
    """Janela unica: camera a esquerda, folha ao vivo a direita.

    A folha ao vivo e o canvas digital (fundo branco, tinta preta) na
    mesma janela e na mesma altura da camera: o traco no papel e o
    traco digital aparecem lado a lado, sem janelas separadas.
    """
    target_h = 640
    cam = _fit(camera_img, target_h)
    sheet = _fit(canvas_img, target_h)
    if cam.shape[0] != sheet.shape[0]:
        h = min(cam.shape[0], sheet.shape[0])
        cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
        sheet = cv2.resize(
            sheet, (int(sheet.shape[1] * h / sheet.shape[0]), h))
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
        # Tamanho final fixo: sem isto, a geometria da janela
        # depende do tamanho entregue pela camera (com 1289x710 o
        # resize antigo cortava a altura para 632; com 1280x720
        # ele nem rodava e a vista saia com 1596 de largura).
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


def _detect_sheet_combined(frame, neural_detector, classic_detect=None):
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
    if classic_detect is None:
        classic_detect = paper_module.detect_sheet_with_confidence
    if neural_detector is not None:
        corners, verdict = neural_detector.detect_sheet_with_confidence(frame)
        if corners is not None:
            return corners, verdict
        logit = neural_detector.last_presence_logit
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return None, None
        corners, verdict = classic_detect(frame)
        if corners is None and logit is not None and logit > 0:
            return None, "paper"
        return corners, verdict
    return classic_detect(frame)


def detect_sheet_with_light_guard(frame, light_normalize="off"):
    """Usa luz normalizada só quando a guarda confirma o candidato."""
    return light_guard_module.detect_sheet_with_light_guard(
        frame, light_normalize
    )


def _detect_sheet_for_run(frame, neural_detector, light_normalize,
                          winning_light):
    """Detecta um quadro e guarda o modo que deve ser usado na tinta.

    O detector neural recebe o frame original, portanto uma detecção
    neural confirmada mantém a tinta sem normalização. Quando ele não
    fecha os cantos, a reserva clássica pode selecionar o modo aceito
    pela guarda de luz.
    """
    winning_light["mode"] = "off"

    def classic_with_light_guard(candidate_frame):
        corners, verdict, winning = detect_sheet_with_light_guard(
            candidate_frame, light_normalize
        )
        winning_light["mode"] = winning
        return corners, verdict

    return _detect_sheet_combined(
        frame, neural_detector, classic_with_light_guard
    )


def _ink_input(warped, light_normalize="off"):
    return light_module.normalize_light(warped, light_normalize)


def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    neural_model=None,
    hand_model=None,
    ink_model=None,
    session_baseline=False,
    width=None,
    height=None,
    fps=None,
    backend=None,
    reconnect_attempts=3,
    open_timeout_ms=3000,
    light_normalize="off",
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

    session_ink = None
    if session_baseline and not markers_mode:
        session_ink = ink_module.SessionBaselineInk()
        print(
            "Tinta da sessao ativa: com a folha na cena, aperte C "
            "para congelar a pagina base; so o traco feito depois "
            "dela entra no canvas."
        )

    def ink_source(warped_img):
        if ink_extractor is not None:
            mask = ink_extractor.ink_mask(warped_img)
            if mask is not None:
                return mask
        return ink_module.extract_ink(warped_img)

    hand_masker = None
    if hand_model is not None and not markers_mode:
        hand_masker = hand_module.HandMasker(hand_model or None)
        print(
            "Mascara de mao solicitada: mao sobre a folha vira regiao "
            "desconhecida no canvas acumulativo."
        )

    winning_light = {"mode": light_normalize}

    def detect_sheet(frame):
        return _detect_sheet_for_run(
            frame, neural_detector, light_normalize, winning_light
        )

    parsed_source = parse_source(source)
    capture_kwargs = {
        "width": width,
        "height": height,
        "fps": fps,
        "backend": backend,
        "open_timeout_ms": open_timeout_ms,
    }
    capture = _open_capture(parsed_source, **capture_kwargs)
    if reconnect_attempts > 0:
        capture = capture_module.ReconnectingCapture(
            parsed_source,
            attempts=reconnect_attempts,
            **capture_kwargs,
        ).attach(capture)
    frame_source = _LatestFrameCapture(capture).start()
    png_writer = _PngWriter()
    save_path = Path(save_dir)
    if markers_mode:
        print("Draw v0 rodando. Mostre a folha-modelo para a camera.")
    else:
        print("Draw v0 rodando. Mostre uma folha para a camera.")
    print(
        "Teclas: s = salvar canvas, f = capturar frame, "
        "g = captura automatica (desligada), q/Esc = sair."
    )

    frames_path = save_path / "frames"
    frames_saved = 0
    auto_capture = False
    restart_reference = False
    baseline_armed = False
    # Geometria da tinta CONGELADA na tecla C: os warps da sessao
    # usam estes cantos, nao os do rastreador ao vivo. Medido nos
    # frames reais de 2026-10-30: sobre o caderno aberto, o ajuste
    # do detector oscila ate 314 px entre frames de uma cena
    # parada; com a geometria viva, o warp teleporta e nenhuma
    # referencia sobrevive. O rastreador segue vivo para o
    # contorno verde e os banners; a tinta vive na geometria do C.
    session_corners = None
    freeze_pending = False
    loop_frames = 0

    def save_raw_frame(img):
        frames_path.mkdir(parents=True, exist_ok=True)
        out = frames_path / f"frame_{int(time.time() * 1000)}.png"

        def _saved(path):
            nonlocal frames_saved
            frames_saved += 1
            print(f"Frame salvo em {path} (total: {frames_saved}).")

        png_writer.save(out, img, _saved)

    def handle_capture_key(key, img):
        nonlocal auto_capture, restart_reference
        if key == ord("f"):
            save_raw_frame(img)
        elif key == ord("c") and session_ink is not None:
            restart_reference = True
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

    tracker = _SheetTracker()
    ink_accumulator = ink_module.InkAccumulator()
    corner_history = deque(maxlen=_INK_CORNER_HISTORY)
    ink_session_open = False
    accumulated_canvas = None
    # Buffers donos dos canvas renderizados no laco. O acumulado
    # mora em canvas_buffer de um frame de deteccao ao outro, e
    # view_canvas_buffer serve aos canvas transitorios de exibicao.
    # Sao buffers separados porque o acumulado e retido entre
    # frames: nenhum outro uso pode escrever nele.
    canvas_buffer = np.empty(
        (ink_module.config.PAPER_H, ink_module.config.PAPER_W, 3),
        dtype=np.uint8,
    )
    view_canvas_buffer = np.empty_like(canvas_buffer)
    frame_number = 0
    last_seq = 0
    last_view = None
    while True:
        item = frame_source.wait_next(last_seq)
        if item is None:
            # A thread de captura declarou a fonte morta (falhas
            # seguidas de leitura ou excecao no read) e wait_next
            # acordou sem quadro. O aviso vai para o console e para
            # a janela, sobre a ultima vista, e fica _DEAD_NOTICE_MS
            # ou ate uma tecla: so o console seria um congelamento
            # silencioso para quem esta olhando a janela.
            message = (
                "A camera abriu, mas parou de entregar quadros. "
                "Feche outros programas usando a camera e tente de novo."
            )
            if frame_source.last_error is not None:
                message += f" Erro mais recente: {frame_source.last_error}"
            print(message)
            cv2.imshow("Draw", _dead_source_view(last_view))
            cv2.waitKey(_DEAD_NOTICE_MS)
            break
        frame, frame_capture_ms, last_seq = item
        # O carimbo de captura nasce na thread de captura, logo que
        # o read() retorna la: a camera nao expoe carimbo de
        # hardware neste caminho, entao esse e o instante mais perto
        # da captura que o app conhece. E ele que carimba os pixels
        # no acumulador, como fazia o carimbo tirado no laco.

        loop_frames += 1
        if auto_capture and loop_frames % 10 == 0:
            save_raw_frame(frame)

        if markers_mode:
            warped, markers = paper_module.warp_paper(frame)
            sheet_present = warped is not None
        else:
            markers = {}
            if tracker.should_detect(frame_number):
                detected_corners, detected_verdict = detect_sheet(frame)
                ran_detection = True
            else:
                detected_corners, detected_verdict = None, None
                ran_detection = False
            tracker_event = tracker.update(
                detected_corners,
                detected_verdict,
                ran_detection,
                time.monotonic(),
            )
            if tracker_event == "drop":
                ink_session_open = False
                session_corners = None
                freeze_pending = False

            frame_number += 1
            # Presenca e o estado do rastreador: warp_sheet com
            # cantos nunca devolve None, entao o warp por quadro
            # era so um teste de presenca caro, e o resultado era
            # descartado. O unico consumo dos pixels fora do bloco
            # de tinta e o fallback de exibicao, que retifica sob
            # demanda.
            sheet_present = tracker.corners is not None
            warped = None

            if ran_detection and tracker.corners is not None:
                if not ink_session_open:
                    # Folha nova na cena: canvas e geometria recomecam.
                    ink_accumulator.reset()
                    if session_ink is not None:
                        session_ink.reset()
                        baseline_armed = False
                        session_corners = None
                        freeze_pending = False
                    corner_history.clear()
                    ink_session_open = True
                if restart_reference:
                    # Tecla C: novo ponto zero da sessao. O canvas
                    # acumulado zera junto, porque a pagina base nova
                    # redefine o que conta como traco desta sessao.
                    # A geometria congela nos cantos deste momento.
                    restart_reference = False
                    ink_accumulator.reset()
                    session_ink.reset()
                    baseline_armed = True
                    freeze_pending = True
                corner_history.append(tracker.corners.copy())
                ink_corners = np.median(
                    np.stack(corner_history), axis=0
                ).astype(np.float32)
                if freeze_pending:
                    session_corners = ink_corners.copy()
                    freeze_pending = False
                warp_corners = (
                    session_corners
                    if session_ink is not None
                    and session_corners is not None
                    else ink_corners
                )
                ink_warped = _ink_input(
                    paper_module.warp_sheet(frame, warp_corners),
                    winning_light["mode"],
                )
                visible = None
                if hand_masker is not None:
                    camera_mask = hand_masker.mask(frame)
                    if camera_mask is not None:
                        hand_warped = paper_module.warp_sheet(
                            camera_mask, warp_corners
                        )
                        visible = hand_warped == 0
                if session_ink is not None:
                    if not session_ink.ready:
                        if baseline_armed:
                            session_ink.add_sample(ink_warped, visible)
                        observed = np.zeros(
                            ink_warped.shape[:2], dtype=np.uint8)
                    else:
                        observed = session_ink.observe(
                            ink_warped, visible)
                else:
                    observed = ink_source(ink_warped)
                accumulated_canvas = ink_module.render_canvas(
                    ink_accumulator.update(
                        observed, visible,
                        t_capture_ms=frame_capture_ms
                    ),
                    out=canvas_buffer,
                )

        status = frame.copy()
        if markers_mode:
            for (x, y) in markers.values():
                cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
        elif tracker.corners is not None:
            outline = tracker.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(status, [outline], True, (0, 255, 0), 2)
            for x, y in tracker.corners:
                cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
        if not sheet_present:
            if markers_mode:
                status_text = (
                    f"Mostre a folha-modelo ({len(markers)}/4 marcadores)"
                )
            else:
                status_text = "Mostre uma folha para a camera"
            cv2.putText(
                status,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            panel = (
                accumulated_canvas
                if accumulated_canvas is not None
                else _blank_sheet()
            )
            last_view = _compose_view(status, panel)
            cv2.imshow("Draw", last_view)
        else:
            if not markers_mode:
                if tracker.verdict == "blank_maybe":
                    status_text = "Folha detectada (baixa confianca)"
                    status_color = (0, 255, 255)
                else:
                    status_text = "Folha detectada"
                    status_color = (0, 255, 0)
                if session_ink is not None and not session_ink.ready:
                    if baseline_armed:
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
            if markers_mode:
                ink_mask = ink_module.extract_ink(
                    warped, clear_markers=True
                )
                canvas = ink_module.render_canvas(
                    ink_mask, out=view_canvas_buffer
                )
            elif accumulated_canvas is not None:
                canvas = accumulated_canvas
            else:
                ink_mask = ink_source(
                    _ink_input(
                        paper_module.warp_sheet(frame, tracker.corners),
                        winning_light["mode"],
                    )
                )
                canvas = ink_module.render_canvas(
                    ink_mask, out=view_canvas_buffer
                )
            last_view = _compose_view(status, canvas)
            cv2.imshow("Draw", last_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_path.mkdir(parents=True, exist_ok=True)
                out = save_path / f"canvas_{int(time.time())}.png"
                png_writer.save(
                    out,
                    canvas,
                    lambda path: print(f"Canvas salvo em {path}"),
                )
            handle_capture_key(key, frame)
            if key in (ord("q"), 27):
                break
            continue

        key = cv2.waitKey(1) & 0xFF
        handle_capture_key(key, frame)
        if key in (ord("q"), 27):
            break

    frame_source.stop()
    png_writer.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Draw v0, papel para o digital")
    parser.add_argument(
        "--source",
        default="0",
        help="indice da webcam local (0, 1...) ou URL do stream "
        "(ex.: http://192.168.0.10:8080/video). Padrao: 0",
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
        "--ink-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo aprendido de tinta (ONNX) como fonte da mascara "
        "no modo sem marcadores, no lugar da heuristica. Sem caminho, "
        "usa a variavel de ambiente DRAW_INK_MODEL. Sem o arquivo ou "
        "com erro de carga, o app avisa uma vez e segue com a "
        "heuristica",
    )
    parser.add_argument(
        "--hand-model",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.task",
        help="mascara de mao (MediaPipe Hand Landmarker) para o "
        "canvas acumulativo: a mao vira regiao desconhecida e nao "
        "ganha nem perde tinta. Sem caminho, usa a variavel de "
        "ambiente DRAW_HAND_MODEL ou models/hand_landmarker.task. "
        "Sem o pacote mediapipe ou sem o modelo, o app roda sem a "
        "mascara e avisa uma vez",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="testa a fonte selecionada com --source e sai",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="lista webcams locais que abrem e sai",
    )
    parser.add_argument(
        "--test-source",
        default=None,
        metavar="FONTE",
        help="faz um diagnóstico detalhado da fonte e sai",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="diagnostica todas as webcams locais e sai",
    )
    parser.add_argument(
        "--include-urls",
        nargs="*",
        default=[],
        help="com --test-all, diagnostica também estas URLs",
    )
    parser.add_argument(
        "--report",
        choices=("text", "json"),
        default="text",
        help="formato do diagnóstico de --test-source/--test-all",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="com --list-cameras, mostra detalhes da câmera",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--backend",
        default=None,
        help="backend OpenCV (any, v4l2, msmf, dshow, ffmpeg, avfoundation, gstreamer)",
    )
    parser.add_argument(
        "--reconnect-attempts",
        type=int,
        default=3,
        help="tentativas de reabrir a câmera quando os quadros param",
    )
    parser.add_argument(
        "--open-timeout-ms",
        type=int,
        default=3000,
        help="limite para abrir a fonte, em milissegundos",
    )
    parser.add_argument(
        "--light-normalize",
        choices=list(light_guard_module.LIGHT_GUARD_MODES),
        default="off",
        help="normalização de fundo protegida: off (padrão) ou background",
    )
    parser.add_argument(
        "--session-baseline",
        action="store_true",
        help="tinta da sessao: a pagina como esta quando voce aperta "
        "a tecla C vira fundo (pauta, margem impressa e desenho "
        "antigo somem do canvas); so o traco feito depois da pagina "
        "base entra. Apertar C de novo congela uma base nova. "
        "Incompativel com --markers",
    )
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    if args.test_source is not None or args.test_all or args.include_urls:
        reports = []
        if args.test_source is not None:
            reports.append(
                diagnose_module.probe_source(
                    args.test_source, open_timeout_ms=args.open_timeout_ms
                )
            )
        if args.test_all:
            reports.extend(
                diagnose_module.probe_all(
                    include_urls=args.include_urls,
                    open_timeout_ms=args.open_timeout_ms,
                )
            )
        elif args.include_urls:
            reports.extend(
                diagnose_module.probe_all(
                    include_urls=args.include_urls,
                    list_fn=lambda: [],
                    open_timeout_ms=args.open_timeout_ms,
                )
            )
        if args.report == "json":
            import json

            print(json.dumps(
                [report.to_dict() for report in reports],
                indent=2,
                default=str,
            ))
        else:
            for report in reports:
                print(diagnose_module.format_report(report))
        if not reports:
            sys.exit(2)
        sys.exit(0 if all(report.ok for report in reports) else 1)
    if args.list_cameras:
        results = capture_module.list_cameras()
        if args.verbose:
            for item in results:
                detail = ""
                if item["opened"]:
                    report = diagnose_module.probe_source(
                        item["index"], read_frames=1,
                        open_timeout_ms=args.open_timeout_ms,
                    )
                    props = []
                    if report.width and report.height:
                        props.append(f"{report.width}x{report.height}")
                    if report.fps:
                        props.append(f"{report.fps:g}fps")
                    if report.fourcc:
                        props.append(str(report.fourcc))
                    if report.backend_used:
                        props.append(str(report.backend_used))
                    if props:
                        detail = f" [{', '.join(props)}]"
                node = f" ({item['node']})" if item.get("node") else ""
                status = "ok" if item["opened"] else "não abriu"
                print(f"  {item['index']}{node}: {status}{detail}")
        else:
            print(capture_module.format_camera_list(results))
        sys.exit(0 if any(item["opened"] for item in results) else 1)
    if args.check:
        sys.exit(run_check(args.source, args.open_timeout_ms))
    if args.reconnect_attempts < 0:
        parser.error("--reconnect-attempts precisa ser >= 0")
    if args.open_timeout_ms <= 0:
        parser.error("--open-timeout-ms precisa ser > 0")
    try:
        backend = capture_module.backend_from_name(args.backend)
    except ValueError as error:
        parser.error(str(error))
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
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=backend,
            reconnect_attempts=args.reconnect_attempts,
            open_timeout_ms=args.open_timeout_ms,
            light_normalize=args.light_normalize,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
