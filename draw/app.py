"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a fonte e sai
  python main.py --list-cameras                   # lista webcams que abrem
  python main.py --markers                        # folha-modelo impressa
  python main.py --neural modelo.onnx             # detector neural primario
  python main.py --hand-model modelo.task         # mascara de mao no canvas
  python main.py --source 1                       # outra webcam local
  python main.py --source /dev/video2             # porta V4L2 especifica
  python main.py --source http://192.168.0.10:8080/video   # camera via rede

A fonte via rede (MJPEG/RTSP) e o caminho recomendado no WSL. O WSL nao
enxerga webcams USB do Windows por padrao, mas enxerga streams de rede.
Assim todo o processamento (e o ML futuro) fica no Linux. A camera so
envia os frames. Ela pode ser um celular, outro computador ou uma camera IP.

Teclas: s = salvar o canvas como PNG, f = capturar um frame cru,
g = ligar ou desligar a captura automatica de frames, q/Esc = sair.
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import capture as capture_module
from . import hand as hand_module
from . import ink as ink_module
from . import neural as neural_module
from . import paper as paper_module


_SHEET_NEAR_LIMIT_PX = 40.0
_SHEET_BLEND_ALPHA = 0.3
_SHEET_JUMP_CONFIRMATIONS = 3
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
    misturar uma deteccao vizinha, exigir confirmacoes para um salto
    de posicao e descartar a folha perdida. Deteccao perto mistura
    com _SHEET_BLEND_ALPHA; salto precisa de
    _SHEET_JUMP_CONFIRMATIONS deteccoes consistentes seguidas (o
    dobro para o candidato blank_maybe). A perda dispara quando os
    dois criterios de _SHEET_LOST_* passam: tempo sem confirmacao e
    quadros sem confirmacao.
    """

    def __init__(self):
        self.corners = None
        self.verdict = None
        self.missed = False
        self._pending = None
        self._pending_hits = 0
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
                # de papel) exige o dobro de confirmacoes seguidas.
                confirmations_needed = (
                    2 * _SHEET_JUMP_CONFIRMATIONS
                    if verdict == "blank_maybe"
                    else _SHEET_JUMP_CONFIRMATIONS
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

                if self._pending_hits >= confirmations_needed:
                    self._confirm(self._pending.copy(), verdict, now)
                    event = "seen"
                else:
                    self._frames_since_seen += 1
                    self.missed = True
        elif self.corners is not None:
            if ran_detection:
                self._pending = None
                self._pending_hits = 0
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
            self._frames_since_seen = 0
            self.missed = False
            event = "drop"
        return event


def parse_source(value):
    """Delega para draw/capture.py, mantendo o mesmo comportamento."""
    return capture_module.parse_source(value)


def is_wsl():
    """Detecta WSL pelo nome do kernel em /proc."""
    return capture_module.is_wsl()


def _configure_local_capture(capture, width=None, height=None, fps=None):
    """MJPG 720p por padrao; ver draw/capture.apply_local_settings."""
    capture_module.apply_local_settings(
        capture, width=width, height=height, fps=fps
    )


def _local_open_hint(source):
    """Diagnostico por porta, o mesmo criterio do --check."""
    return capture_module.diagnose_local(source)[1]


def _open_capture(source, **kwargs):
    """Abre via draw/capture.py, com backends, timeout e erro especifico."""
    return capture_module.open_capture(source, **kwargs)


def _video_nodes():
    return capture_module.video_nodes()


def _read_first_frame(capture, timeout_s):
    """Tenta obter um frame até o prazo; capture.read() pode bloquear."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            ok, frame = capture.read()
        except Exception as error:
            return None, error
        if time.monotonic() > deadline:
            return None, TimeoutError("nenhum frame recebido dentro do prazo")
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

        height, width = frame.shape[:2]
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        except Exception:
            fps = 0.0
        fps_text = f"{fps:g} FPS" if fps > 0 else "FPS indisponivel"
        print(f"PASS: primeiro frame recebido: {width}x{height}, {fps_text}.")
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


def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    width=None,
    height=None,
    fps=None,
    backend=None,
    reconnect_attempts=3,
    open_timeout_ms=3000,
    neural_model=None,
    hand_model=None,
):
    neural_detector = None
    if neural_model is not None and not markers_mode:
        neural_detector = neural_module.NeuralSheetDetector(neural_model)
        neural_detector.load()
        print(
            f"Detector neural ativo: {neural_model}. "
            "O classico cobre os frames em que o neural nao ve folha."
        )

    hand_masker = None
    if hand_model is not None and not markers_mode:
        hand_masker = hand_module.HandMasker(hand_model or None)
        print(
            "Mascara de mao solicitada: mao sobre a folha vira regiao "
            "desconhecida no canvas acumulativo."
        )

    def detect_sheet(frame):
        if neural_detector is not None:
            corners, verdict = (
                neural_detector.detect_sheet_with_confidence(frame)
            )
            if corners is not None:
                return corners, verdict
        return paper_module.detect_sheet_with_confidence(frame)

    parsed = parse_source(source)
    capture = capture_module.ReconnectingCapture(
        parsed, attempts=reconnect_attempts, width=width, height=height,
        fps=fps, backend=backend, open_timeout_ms=open_timeout_ms,
    ).attach(_open_capture(
        parsed, width=width, height=height, fps=fps, backend=backend,
        open_timeout_ms=open_timeout_ms,
    ))
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
    loop_frames = 0

    def save_raw_frame(img):
        nonlocal frames_saved
        frames_path.mkdir(parents=True, exist_ok=True)
        out = frames_path / f"frame_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(out), img)
        frames_saved += 1
        print(f"Frame salvo em {out} (total: {frames_saved}).")

    def handle_capture_key(key, img):
        nonlocal auto_capture
        if key == ord("f"):
            save_raw_frame(img)
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
    frame_number = 0
    falhas_seguidas = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            falhas_seguidas += 1
            if falhas_seguidas >= 15:
                if capture.try_reconnect():
                    print("Camera reconectada, continuando.")
                    falhas_seguidas = 0
                    continue
                print(
                    "A camera abriu, mas parou de entregar quadros e "
                    "nao voltou apos a reconexao. Feche outros programas "
                    "usando a camera, reconecte-a e tente de novo."
                )
                break
            continue
        falhas_seguidas = 0

        loop_frames += 1
        if auto_capture and loop_frames % 10 == 0:
            save_raw_frame(frame)

        if markers_mode:
            warped, markers = paper_module.warp_paper(frame)
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

            frame_number += 1
            warped = (
                paper_module.warp_sheet(frame, tracker.corners)
                if tracker.corners is not None
                else None
            )

            if ran_detection and tracker.corners is not None:
                if not ink_session_open:
                    # Folha nova na cena: canvas e geometria recomecam.
                    ink_accumulator.reset()
                    corner_history.clear()
                    ink_session_open = True
                corner_history.append(tracker.corners.copy())
                ink_corners = np.median(
                    np.stack(corner_history), axis=0
                ).astype(np.float32)
                ink_warped = paper_module.warp_sheet(frame, ink_corners)
                observed = ink_module.extract_ink(ink_warped)
                visible = None
                if hand_masker is not None:
                    camera_mask = hand_masker.mask(frame)
                    if camera_mask is not None:
                        hand_warped = paper_module.warp_sheet(
                            camera_mask, ink_corners
                        )
                        visible = hand_warped == 0
                accumulated_canvas = ink_module.render_canvas(
                    ink_accumulator.update(observed, visible)
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
        if warped is None:
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
            cv2.imshow("Draw - camera", _fit(status, 700))
        else:
            if not markers_mode:
                if tracker.verdict == "blank_maybe":
                    status_text = "Folha detectada (baixa confianca)"
                    status_color = (0, 255, 255)
                else:
                    status_text = "Folha detectada"
                    status_color = (0, 255, 0)
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
                canvas = ink_module.render_canvas(ink_mask)
            elif accumulated_canvas is not None:
                canvas = accumulated_canvas
            else:
                ink_mask = ink_module.extract_ink(warped)
                canvas = ink_module.render_canvas(ink_mask)
            cv2.imshow("Draw - camera", _fit(status, 700))
            cv2.imshow("Draw - papel (vista de cima)", _fit(warped, 700))
            cv2.imshow("Draw - canvas digital", _fit(canvas, 700))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_path.mkdir(parents=True, exist_ok=True)
                out = save_path / f"canvas_{int(time.time())}.png"
                cv2.imwrite(str(out), canvas)
                print(f"Canvas salvo em {out}")
            handle_capture_key(key, frame)
            if key in (ord("q"), 27):
                break
            continue

        key = cv2.waitKey(1) & 0xFF
        handle_capture_key(key, frame)
        if key in (ord("q"), 27):
            break

    capture.release()
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
        help="valida a fonte escolhida com --source e sai, sem abrir janela",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="lista as webcams locais que abrem e sai, sem abrir janela",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="largura pedida para webcam local (padrao: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="altura pedida para webcam local (padrao: 720)",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="FPS pedido para webcam local (padrao: o da camera)",
    )
    parser.add_argument(
        "--backend", default=None,
        help="backend OpenCV: any, v4l2, msmf, dshow, ffmpeg, "
        "avfoundation, gstreamer (padrao: o do SO, para webcam local)",
    )
    parser.add_argument(
        "--reconnect-attempts", type=int, default=3,
        help="tentativas de reabrir a camera quando os quadros param",
    )
    parser.add_argument(
        "--open-timeout-ms", type=int, default=3000,
        help="timeout de abertura da fonte, em milissegundos",
    )
    parser.add_argument("--save-dir", default="captures")
    args = parser.parse_args()
    if args.list_cameras:
        results = capture_module.list_cameras()
        print(capture_module.format_camera_list(results))
        sys.exit(0 if any(item["opened"] for item in results) else 1)
    if args.check:
        parsed = parse_source(args.source)
        sys.exit(run_check(parsed, open_timeout_ms=args.open_timeout_ms))
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
    hand_model = None
    if args.hand_model is not None:
        if args.markers:
            parser.error("--hand-model nao combina com --markers")
        hand_model = (
            args.hand_model or os.environ.get("DRAW_HAND_MODEL") or ""
        )
    try:
        backend = capture_module.backend_from_name(args.backend)
    except ValueError as error:
        parser.error(str(error))
    try:
        run(
            source=args.source,
            save_dir=args.save_dir,
            markers_mode=args.markers,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=backend,
            reconnect_attempts=args.reconnect_attempts,
            open_timeout_ms=args.open_timeout_ms,
            neural_model=neural_model,
            hand_model=hand_model,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
