"""Loop principal: camera -> folha retificada -> canvas digital ao vivo.

Uso:
  python main.py                                  # webcam local (indice 0)
  python main.py --check                          # valida a camera e sai
  python main.py --markers                        # folha-modelo impressa
  python main.py --legacy-ink                     # pilha antiga de tinta
  python main.py --neural modelo.onnx             # outro detector de folha
  python main.py --page-clean modelo.onnx         # outro modelo de tinta
  python main.py --source 1                       # outra webcam local
  python main.py --source http://192.168.0.10:8080/video   # camera via rede

Sem nenhuma flag, o caminho e o do reinicio de 2026-10-31: folha
pelo detector neural v3 (models/draw_seg_v3_dyn.onnx, com o classico
de reserva) e tinta pelo modelo pagina->limpa
(models/page_clean_unet.onnx), em duas camadas na trabalhadora:
a provisoria aparece na tela na primeira passada e a confirmada,
que e o canvas dos consumidores, entra pelo voto entre passadas.
A pilha antiga de tinta (heuristica, --ink-model,
--session-baseline com a tecla C, --hand-model) so roda atras de
--legacy-ink; passar qualquer uma dessas flags liga o legado com
aviso.

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
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import hand as hand_module
from . import config
from . import ink as ink_module

try:
    from . import ink_model as ink_model_module
except ImportError:
    # O extrator de tinta por modelo aprendido e opcional e o modulo
    # dele ainda nao faz parte do pacote publicado. Sem o modulo,
    # --ink-model avisa e o app segue so com a heuristica.
    ink_model_module = None
from . import manifest as manifest_module
from .safe_io import ensure_private_dir
from . import neural as neural_module
from . import page_clean as page_clean_module
from .selflearn import session as session_module
from . import paper as paper_module
from . import session_report as session_report_module
from . import telemetry as telemetry_module
from .view import _blank_sheet, _compose_view, _dead_source_view, _error_view
from .loop import (
    INK_WORKER_FAILED,
    SheetInkLoop,
    _INK_CORNER_HISTORY as _INK_CORNER_HISTORY,
    _acquisition_record as _acquisition_record,
)

from . import capture as capture_module
from .sheet_track import (
    _SHEET_NEAR_LIMIT_PX as _SHEET_NEAR_LIMIT_PX,
    _SHEET_BLEND_ALPHA as _SHEET_BLEND_ALPHA,
    _SHEET_FILTER_WINDOW as _SHEET_FILTER_WINDOW,
    _SHEET_DEADBAND_PX as _SHEET_DEADBAND_PX,
    _SHEET_BIG_JUMP_PX as _SHEET_BIG_JUMP_PX,
    _SHEET_BIG_JUMP_PERSIST_SECONDS as _SHEET_BIG_JUMP_PERSIST_SECONDS,
    _SHEET_JUMP_PERSIST_SECONDS as _SHEET_JUMP_PERSIST_SECONDS,
    _SHEET_DETECT_INTERVAL as _SHEET_DETECT_INTERVAL,
    _SHEET_ABSENT_BACKOFF_CALLS as _SHEET_ABSENT_BACKOFF_CALLS,
    _SHEET_ABSENT_BACKOFF_INTERVAL as _SHEET_ABSENT_BACKOFF_INTERVAL,
    _SHEET_LOST_SECONDS as _SHEET_LOST_SECONDS,
    _SHEET_LOST_MIN_FRAMES as _SHEET_LOST_MIN_FRAMES,
    _corner_distance as _corner_distance,
    _blend_corners as _blend_corners,
    _SheetTracker as _SheetTracker,
    _corners_json as _corners_json,
    _LK_MAX_STEP_PX as _LK_MAX_STEP_PX,
    _LK_MAX_FB_ERROR_PX as _LK_MAX_FB_ERROR_PX,
    _lk_track_corners as _lk_track_corners,
    _DETECTION_STOP_TIMEOUT_SECONDS as _DETECTION_STOP_TIMEOUT_SECONDS,
    _SheetDetectionWorker as _SheetDetectionWorker,
)
from .capture import (
    _CAPTURE_MAX_FAILURES as _CAPTURE_MAX_FAILURES,
    _READ_RETRY_SECONDS as _READ_RETRY_SECONDS,
    _PNG_DRAIN_SECONDS as _PNG_DRAIN_SECONDS,
    _LatestFrameCapture as _LatestFrameCapture,
    _CAPTURE_DRAIN_ENV as _CAPTURE_DRAIN_ENV,
    capture_drain_latest_enabled as capture_drain_latest_enabled,
    _PngWriter as _PngWriter,
    parse_source as parse_source,
    is_wsl as is_wsl,
    _configure_local_capture as _configure_local_capture,
    _EXPOSURE_SETTLED_MEAN as _EXPOSURE_SETTLED_MEAN,
    _EXPOSURE_BLOWN_MEAN as _EXPOSURE_BLOWN_MEAN,
    _EXPOSURE_PROBE_SECONDS as _EXPOSURE_PROBE_SECONDS,
    _EXPOSURE_MAX_ATTEMPTS as _EXPOSURE_MAX_ATTEMPTS,
    _LAST_OPEN_PROBE as _LAST_OPEN_PROBE,
    _record_open_probe as _record_open_probe,
    _probe_exposure as _probe_exposure,
    _supervise_local_open as _supervise_local_open,
    _local_open_hint as _local_open_hint,
    _open_capture as _open_capture,
    _video_nodes as _video_nodes,
)
from .hand_presence import (
    _HAND_MOVE_CENTROID_PX as _HAND_MOVE_CENTROID_PX,
    _HAND_MOVE_AREA_FRACTION as _HAND_MOVE_AREA_FRACTION,
    _MediaPipeHandPresence as _MediaPipeHandPresence,
    _make_hand_presence as _make_hand_presence,
    _HandPresenceWorker as _HandPresenceWorker,
)
from .ink_layers import (
    _IDLE_PROBE_SECONDS as _IDLE_PROBE_SECONDS,
    _NO_HAND_REST_AFTER_STILL_PASSES as _NO_HAND_REST_AFTER_STILL_PASSES,
    _REST_MIN_GENERATION_PASSES as _REST_MIN_GENERATION_PASSES,
    _REST_QUIET_PASSES as _REST_QUIET_PASSES,
    _HAND_REST_STILL_SECONDS as _HAND_REST_STILL_SECONDS,
    _PAGE_CLEAN_INTERVAL_SECONDS as _PAGE_CLEAN_INTERVAL_SECONDS,
    _PAGE_CLEAN_STOP_TIMEOUT_SECONDS as _PAGE_CLEAN_STOP_TIMEOUT_SECONDS,
    _INK_TURN_MIN_CANVAS_PX as _INK_TURN_MIN_CANVAS_PX,
    _INK_TURN_MAX_OVERLAP as _INK_TURN_MAX_OVERLAP,
    _INK_TURN_PASSES as _INK_TURN_PASSES,
    _ENV_TURN_EVENT_VAR as _ENV_TURN_EVENT_VAR,
    _page_turn_event_enabled as _page_turn_event_enabled,
    _INK_VOTE_WINDOW as _INK_VOTE_WINDOW,
    _INK_VOTE_SCORE as _INK_VOTE_SCORE,
    _INK_VOTE_SCORE_STRONG as _INK_VOTE_SCORE_STRONG,
    _INK_VOTE_HIT as _INK_VOTE_HIT,
    _INK_VOTE_MIN_HITS as _INK_VOTE_MIN_HITS,
    _INK_FRONTIER_RADIUS as _INK_FRONTIER_RADIUS,
    _INK_FRONTIER_KERNEL as _INK_FRONTIER_KERNEL,
    _REGISTRATION_IDENTITY_TOL as _REGISTRATION_IDENTITY_TOL,
    _corners_page_matrix as _corners_page_matrix,
    _registration_matrix as _registration_matrix,
    _registration_is_identity as _registration_is_identity,
    _LayeredInkAccumulator as _LayeredInkAccumulator,
    _PageCleanWorker as _PageCleanWorker,
    _render_layered_canvas as _render_layered_canvas,
)


# Tempo que o aviso de fonte morta fica na janela antes da saida.
# Sem ele, a morte da fonte e um congelamento silencioso. O waitKey
# devolve antes se qualquer tecla for apertada.
_DEAD_NOTICE_MS = 3000


def run_check(source=0):
    """Validacao da camera, porta por porta. Retorna 0 ou 1.

    O `source` importa: a versao anterior abria sempre o
    /dev/video0, qualquer que fosse o indice pedido, e com uma URL
    caia no 0 tambem. Agora o indice pedido e o testado, e uma URL
    e aberta como URL.
    """
    print(f"PASS: OpenCV disponivel. Versao {cv2.__version__}.")
    all_ok = _check_models()

    if not isinstance(source, int):
        return 0 if _check_remote_source(source) and all_ok else 1

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

    node = f"/dev/video{source}"
    if node not in nodes:
        print(
            f"FAIL: {node} nao existe. Os dispositivos deste sistema "
            f"sao: {', '.join(nodes)}."
        )
        print(
            "  Correcao: use --source com um dos indices acima, ou "
            "conecte a camera pedida."
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

    capture = cv2.VideoCapture(source)
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


def _check_remote_source(url):
    """Abre uma fonte de rede e confere que um quadro chega. 0/1."""
    print(f"PASS: fonte de rede informada: {url}.")
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        print(
            "FAIL: nao consegui abrir a fonte de rede. Confira a URL, "
            "se a camera esta transmitindo e se este computador "
            "alcanca o endereco."
        )
        return 0
    frame = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        time.sleep(0.1)
    capture.release()
    if frame is None:
        print("FAIL: a fonte abriu, mas nenhum quadro chegou em 5 segundos.")
        return 0
    print(
        "PASS: quadro lido da fonte de rede. "
        f"Formato (altura, largura, canais): {frame.shape}."
    )
    return 1


def _check_models():
    """Confere os modelos contra models/MANIFEST.json. True se tudo bate.

    Sem manifesto (o caso de um clone novo), imprime o aviso e
    devolve True: nada a verificar nao e falha. Um .onnx/.task em
    models/ sem entrada no manifesto e FAIL no modo estrito (padrao)
    e AVISO com DRAW_MODEL_STRICT=0.
    """
    manifest_path, models = manifest_module.load_manifest()
    ok = True
    strict = manifest_module.strict_enabled()
    for extra in manifest_module.unregistered_model_files():
        label = "FAIL" if strict else "AVISO"
        print(
            f"{label}: {extra.name} esta em models/ e nao tem entrada no "
            "manifesto; a carga dele e recusada no modo estrito."
        )
        ok = ok and not strict
    if not models:
        print(
            f"AVISO: {manifest_path} sem modelos registrados; "
            "nenhum binario foi conferido."
        )
        return ok
    for name in sorted(models):
        path = Path(__file__).resolve().parent.parent / "models" / name
        if not path.exists():
            print(
                f"AVISO: {name} esta no manifesto e nao esta em models/. "
                "Baixe os modelos oficiais com: "
                "python tools/fetch_models.py"
            )
            continue
        try:
            entry = manifest_module.verify_model(path)
        except RuntimeError as error:
            print(f"FAIL: {error}")
            ok = False
            continue
        if entry is None:
            print(f"AVISO: {name} nao tem entrada no manifesto.")
            continue
        print(
            f"PASS: {name} confere com o manifesto "
            f"({entry.get('approved_by', 'sem régua registrada')})."
        )
    return ok


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
    corners, verdict, _trace = _detect_sheet_combined_traced(frame, neural_detector)
    return corners, verdict


_CLASSIC_TRACE_VERDICT = {"paper": "corners", "blank_maybe": "blank_maybe"}


def _detect_sheet_combined_traced(frame, neural_detector):
    """_detect_sheet_combined com o rastro da aquisicao.

    Mesma decisao, linha a linha; o terceiro elemento so descreve
    o que aconteceu: presence_logit, cantos/veredito neurais, se o
    classico rodou e o que disse (corners, blank_maybe ou None) e a
    fonte do desfecho combinado: "neural" (quad neural fechou),
    "veto" (logit ate PRESENCE_VETO_LOGIT), "classic" (o classico
    devolveu quad, paper ou blank_maybe), "presence_only" (nenhum
    quad, logit positivo: (None, "paper")) ou "none".
    """
    trace = {
        "neural_ran": neural_detector is not None,
        "presence_logit": None,
        "veto_logit": None,
        "neural_corners": None,
        "neural_verdict": None,
        "classic_ran": False,
        "classic_verdict": None,
        "classic_corners": None,
    }

    def _done(corners, verdict, source):
        trace["source"] = source
        trace["combined_corners"] = _corners_json(corners)
        trace["combined_verdict"] = verdict
        return corners, verdict, trace

    def _classic():
        corners, verdict = paper_module.detect_sheet_with_confidence(frame)
        trace["classic_ran"] = True
        trace["classic_verdict"] = (
            _CLASSIC_TRACE_VERDICT.get(verdict, verdict)
            if corners is not None
            else None
        )
        trace["classic_corners"] = _corners_json(corners)
        return corners, verdict

    if neural_detector is not None:
        trace["veto_logit"] = float(neural_module.PRESENCE_VETO_LOGIT)
        if neural_module.first_confirm_gate_enabled():
            # Portão de primeira confirmação (frente 88): com a flag
            # ligada, a detecção pede a máscara — é dela que sai o
            # edge_in que o laço (draw.loop) julga antes de o
            # rastreador confirmar a primeira folha. A decisão
            # combinada é a mesma; só o rastro ganha o campo.
            corners, verdict, mask = neural_detector(frame, return_mask=True)
            if corners is not None and mask is not None:
                trace["edge_in"] = float(neural_module.mask_edge_fraction(frame, mask))
        else:
            corners, verdict = neural_detector.detect_sheet_with_confidence(frame)
        logit = neural_detector.last_presence_logit
        trace["presence_logit"] = None if logit is None else float(logit)
        trace["neural_corners"] = _corners_json(corners)
        trace["neural_verdict"] = verdict
        if corners is not None:
            return _done(corners, verdict, "neural")
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return _done(None, None, "veto")
        corners, verdict = _classic()
        if corners is None and logit is not None and logit > 0:
            return _done(None, "paper", "presence_only")
        return _done(corners, verdict, "classic" if corners is not None else "none")
    corners, verdict = _classic()
    return _done(corners, verdict, "classic" if corners is not None else "none")


def _emit_acquisition(telemetry, session_recorder, record):
    telemetry.event("acquisition", **record)
    if session_recorder is not None:
        session_recorder.note_acquisition(record)


# Teto do pool paralelo do OpenCV. Sem teto, o pool abre com as
# CPUs logicas do host (16 no Ryzen) e vira o terceiro disputante
# dos nucleos fisicos, junto dos pools ORT do detector e da tinta
# (oversubscription medida na mesa do Lucas, 2026-11-02: CPU
# desenhando 550,8% com os pools 4+8+16; teto combinado 2+4 no
# ORT cortou para 289,8%, passada igual). DRAW_OPENCV_THREADS
# troca o valor; invalido volta ao padrao.
_OPENCV_THREAD_CAP = 4


_OPENCV_THREADS_ENV = "DRAW_OPENCV_THREADS"


def _apply_opencv_thread_cap():
    """Limita o pool do OpenCV ao teto combinado. Devolve o teto."""
    cap = _OPENCV_THREAD_CAP
    raw = os.environ.get(_OPENCV_THREADS_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            cap = value
    cap = max(1, min(cap, os.cpu_count() or cap))
    cv2.setNumThreads(cap)
    return cap


def run(
    source=0,
    save_dir="captures",
    markers_mode=False,
    neural_model=None,
    hand_model=None,
    ink_model=None,
    session_baseline=False,
    page_clean_model=None,
    legacy_ink=False,
):
    _apply_opencv_thread_cap()
    # A pilha antiga de tinta so existe atras do legado. Pedir
    # qualquer peca dela (modelo de tinta, pagina base na tecla C,
    # mascara de mao) liga o legado com aviso, para nenhuma flag
    # ficar silenciosamente sem efeito no caminho novo.
    if not legacy_ink and (
        ink_model is not None or session_baseline or hand_model is not None
    ):
        legacy_ink = True
        print(
            "Flags da pilha antiga de tinta detectadas: rodando com "
            "--legacy-ink (heuristica/sessao). O caminho padrao novo "
            "e o modelo pagina->limpa."
        )

    if neural_model is None and not markers_mode:
        # Padrao do reinicio, tambem para quem chama run() direto:
        # a variavel de ambiente ou o modelo v3 no lugar de sempre.
        # main() resolve o mesmo padrao para a linha de comando.
        default_sheet = (
            Path(__file__).resolve().parent.parent / "models" / "draw_seg_v3_dyn.onnx"
        )
        if default_sheet.exists():
            neural_model = str(default_sheet)
        else:
            neural_model = os.environ.get("DRAW_NEURAL_MODEL")

    neural_detector = None
    if neural_model is not None and not markers_mode:
        neural_detector = neural_module.NeuralSheetDetector(neural_model)
        neural_detector.load()
        print(
            f"Detector neural ativo: {neural_model}. "
            "O classico cobre os frames em que o neural nao ve folha."
        )
    elif not markers_mode:
        print(
            "Aviso: sem detector neural de folha (modelo nao "
            "encontrado); usando so o detector classico, que nao "
            "enxerga a folha em varias mesas. Baixe os modelos "
            "oficiais com: python tools/fetch_models.py"
        )

    page_clean = None
    if not markers_mode and not legacy_ink:
        candidate = page_clean_module.PageCleanInk(page_clean_model)
        try:
            candidate.load()
            page_clean = candidate
            print(
                f"Tinta pagina->limpa ativa: {candidate.model_path} "
                f"(limiar {candidate.threshold}). Canvas por uniao "
                "simples das mascaras."
            )
        except (RuntimeError, FileNotFoundError) as error:
            print(
                f"Aviso: modelo pagina->limpa nao carregou ({error}). "
                "Seguindo com a pilha antiga de tinta. Baixe os "
                "modelos oficiais com: python tools/fetch_models.py"
            )
            legacy_ink = True

    ink_worker = None
    if page_clean is not None:
        ecc_hand_masker = (
            hand_module.HandMasker(hand_model)
            if not markers_mode and hand_module.hand_model_available(hand_model)
            else None
        )
        ink_worker = _PageCleanWorker(page_clean, ecc_hand_masker).start()
        print(
            "Tinta em thread propria: o laco principal nao espera "
            "o modelo (passadas com cadencia de ~250 ms)."
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
        if page_clean is not None:
            return page_clean.ink_mask(warped_img)
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

    clean_hand_presence = None
    hand_presence_worker = None
    if page_clean is not None and not markers_mode:
        clean_hand_presence = _make_hand_presence(hand_model)
        if clean_hand_presence is not None:
            print(
                "Porta de ociosidade pela mao ativa no caminho "
                "pagina->limpa: sem mao, a tinta sonda ate a saida "
                "confirmada assentar, descansa sem passadas e a "
                "geometria da tinta congela."
            )
            # Amostragem fora do laco (T4): o laco submete e colhe.
            hand_presence_worker = _HandPresenceWorker(clean_hand_presence).start()

    # Rastro por aquisicao: ligado so em sessao gravada (abaixo, com
    # o gravador); fora dela, a deteccao e a de sempre.
    trace_acquisitions = False

    def detect_sheet(frame):
        # Com o portão de primeira confirmação ligado, a forma com
        # rastro é obrigatória mesmo fora de sessão gravada: é o
        # rastro que carrega logit/edge_in até o laço (a trabalhadora
        # extrai dele o gate_info do resultado).
        if trace_acquisitions or neural_module.first_confirm_gate_enabled():
            return _detect_sheet_combined_traced(frame, neural_detector)
        return _detect_sheet_combined(frame, neural_detector)

    capture = _open_capture(parse_source(source))
    frame_source = _LatestFrameCapture(capture).start()
    detection_worker = None
    if not markers_mode:
        detection_worker = _SheetDetectionWorker(detect_sheet).start()
    png_writer = _PngWriter()
    # Capturas, telemetria e relatórios são privados: a pasta ganha
    # um .gitignore "*" qualquer que seja o --save-dir.
    save_path = ensure_private_dir(save_dir)
    telemetry = telemetry_module.SessionTelemetry(save_path / "telemetry.jsonl").start()
    format_metadata = {
        "paper_format": config.PAPER_FORMAT,
        "page_grid": [config.PAPER_W, config.PAPER_H],
        "mm_per_px": config.mm_per_px(),
    }
    telemetry.update(**format_metadata)
    telemetry.event("session_open", **format_metadata)
    print(
        f"Formato da sessao: {config.PAPER_FORMAT}, "
        f"grade {config.PAPER_W}x{config.PAPER_H}, "
        f"{config.mm_per_px():.6f} mm/px."
    )
    print(
        "Telemetria de sessao ligada: 1 registro/s em "
        f"{save_path / 'telemetry.jsonl'} (arquivo local e rotativo)."
    )

    # Relatorio de sessao: um JSON por corrida com cabecalho
    # (modelos com sha, abertura da camera, porta da mao) e os
    # agregados da sessao, para uma corrida que falha deixar dito
    # o que rodou — a telemetria bruta sozinha nao diz.
    def _model_header(path):
        if path is None:
            return None
        try:
            return {
                "arquivo": str(path),
                "sha256": manifest_module.sha256_of(path),
            }
        except OSError:
            return {"arquivo": str(path), "sha256": None}

    session_report = session_report_module.SessionReport(
        save_path / "session-reports",
        {
            "inicio_epoch": time.time(),
            **format_metadata,
            "fonte": str(source),
            "modo_marcadores": bool(markers_mode),
            "modelo_folha": _model_header(
                neural_model if neural_detector is not None else None
            ),
            "modelo_tinta": _model_header(
                page_clean.model_path if page_clean is not None else None
            ),
            "tinta_modo": ("page-clean" if page_clean is not None else "heuristica"),
            "mao_disponivel": clean_hand_presence is not None,
            "camera_abertura": capture_module._LAST_OPEN_PROBE,
        },
    ).start()
    print(
        "Relatorio de sessao ligado: resumo desta corrida em "
        f"{session_report.path} (atualizado a cada 15 s e no fim)."
    )
    # Coleta automatica de sessao (loop v2, correcao 5): cada
    # geracao grava referencia + warp final + canvas final +
    # manifesto em save_path/sessions/<id>/, local, para o par
    # antes/depois virar dado do loop sem passar por modelo. A
    # escrita e da thread do gravador; o laco so copia arrays.
    # No modo de marcadores nao ha geracao de folha: sem coleta.
    session_recorder = None
    if not markers_mode:
        model_paths = {}
        if page_clean is not None:
            model_paths["page_clean"] = page_clean.model_path
        if neural_model is not None:
            model_paths["sheet"] = neural_model
        try:
            session_recorder = session_module.SessionRecorder(
                save_path, model_paths=model_paths
            )
        except Exception as error:  # noqa: BLE001 - coleta nao derruba o app
            print(
                "Aviso: a coleta de sessao nao iniciou "
                f"({error}); o app segue sem gravar a sessao."
            )
    # Sessao gravada e diagnosticavel: quadros crus esparsos e um
    # evento por aquisicao de folha (observabilidade, 2026-11-10).
    trace_acquisitions = session_recorder is not None
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
    tel_window_start = time.monotonic()
    tel_window_frames = 0
    tel_loop_fps = 0.0
    tel_detect_ms = None
    tel_sheet_present = False

    def save_raw_frame(img):
        frames_path.mkdir(parents=True, exist_ok=True)
        out = frames_path / f"frame_{int(time.time() * 1000)}.png"

        def _saved(path):
            nonlocal frames_saved
            frames_saved += 1
            print(f"Frame salvo em {path} (total: {frames_saved}).")

        png_writer.save(out, img, _saved)

    def handle_capture_key(key, img):
        nonlocal auto_capture
        if session_recorder is not None:
            session_recorder.note_key(key)
        if key == ord("f"):
            save_raw_frame(img)
        elif key == ord("c") and session_ink is not None:
            loop.restart_reference = True
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

    format_suggester = paper_module.FormatSuggester()

    def save_page_snapshot(prefix, mask):
        save_path.mkdir(parents=True, exist_ok=True)
        snapshot_path = save_path / f"{prefix}_{int(time.time())}.png"
        png_writer.save(
            snapshot_path,
            ink_module.render_canvas(mask),
            lambda path: print(f"Pagina anterior salva em {path}"),
        )

    loop = SheetInkLoop(
        detection_worker=detection_worker,
        ink_worker=ink_worker,
        hand_presence_worker=hand_presence_worker,
        hand_masker=hand_masker,
        session_ink=session_ink,
        ink_source=ink_source,
        session_recorder=session_recorder,
        trace_acquisitions=trace_acquisitions,
        format_suggester=format_suggester,
        on_format_suggestion=lambda suggestion: _report_format_suggestion(
            telemetry, suggestion
        ),
        on_acquisition=lambda record: _emit_acquisition(
            telemetry, session_recorder, record
        ),
        save_page_snapshot=save_page_snapshot,
    )
    tracker = loop.tracker
    last_recorded_ink_pass = 0
    # Buffer dos canvas transitorios de exibicao (o acumulado mora
    # no buffer do laco: nenhum outro uso pode escrever nele).
    view_canvas_buffer = np.empty(
        (ink_module.config.PAPER_H, ink_module.config.PAPER_W, 3),
        dtype=np.uint8,
    )
    last_seq = 0
    last_view = None

    try:
        while True:
            item = frame_source.wait_next(last_seq)
            if item is None:
                # A thread de captura declarou a fonte morta (falhas
                # seguidas de leitura ou excecao no read) e wait_next
                # acordou sem quadro. O aviso vai para o console e para
                # a janela, sobre a ultima vista, e fica _DEAD_NOTICE_MS
                # ou ate uma tecla: so o console seria um congelamento
                # silencioso para quem esta olhando a janela.
                print(
                    "A camera abriu, mas parou de entregar quadros. "
                    "Feche outros programas usando a camera e tente de novo."
                )
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
            tel_window_frames += 1
            tel_now = time.monotonic()
            if tel_now - tel_window_start >= 1.0:
                tel_loop_fps = round(
                    tel_window_frames / (tel_now - tel_window_start), 2
                )
                tel_window_start = tel_now
                tel_window_frames = 0
            tel_fields = {
                "loop_fps": tel_loop_fps,
                "detect_ms": tel_detect_ms,
                "sheet_present": tel_sheet_present,
                "state": "sheet" if tel_sheet_present else "no_sheet",
            }
            tel_fields.update(frame_source.stats())
            if ink_worker is not None:
                tel_fields.update(ink_worker.stats())
                if (
                    session_recorder is not None
                    and tel_fields["passes_total"] != last_recorded_ink_pass
                ):
                    snapshot = ink_worker.telemetry_snapshot()
                    session_recorder.note_ink_telemetry(snapshot)
                    last_recorded_ink_pass = snapshot["passes"]
            if detection_worker is not None:
                tel_fields.update(detection_worker.stats())
            telemetry.update(**tel_fields)
            session_report.update(**tel_fields)
            if auto_capture and loop_frames % 10 == 0:
                save_raw_frame(frame)

            if markers_mode:
                warped, markers = paper_module.warp_paper(frame)
                sheet_present = warped is not None
                tel_sheet_present = sheet_present
                if sheet_present:
                    suggestion = format_suggester.observe(
                        [markers[i] for i in config.MARKER_IDS],
                        frame.shape,
                        sheet_boundary=False,
                    )
                    if suggestion is not None:
                        _report_format_suggestion(telemetry, suggestion)
            else:
                markers = {}
                warped = None
                loop_outcome = loop.step(
                    frame,
                    frame_capture_ms,
                    ink_passes_total=tel_fields.get("passes_total"),
                )
                tel_detect_ms = loop.detect_ms
                sheet_present = loop.sheet_present
                tel_sheet_present = sheet_present
                if loop_outcome == INK_WORKER_FAILED:
                    cv2.imshow(
                        "Draw",
                        _error_view(
                            last_view,
                            "A tinta pagina->limpa falhou",
                            [
                                "O canvas parou de atualizar.",
                                "Veja a mensagem no terminal e reinicie o app.",
                            ],
                        ),
                    )
                    cv2.waitKey(_DEAD_NOTICE_MS)
                    break

            status = frame.copy()
            if markers_mode:
                for x, y in markers.values():
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            elif tracker.corners is not None:
                view_corners = (
                    loop.display_corners
                    if loop.display_corners is not None
                    else tracker.corners
                )
                outline = view_corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(status, [outline], True, (0, 255, 0), 2)
                for x, y in view_corners:
                    cv2.circle(status, (int(x), int(y)), 8, (0, 255, 0), -1)
            if not sheet_present:
                if markers_mode:
                    status_text = f"Mostre a folha-modelo ({len(markers)}/4 marcadores)"
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
                    loop.accumulated_canvas
                    if loop.accumulated_canvas is not None
                    else _blank_sheet()
                )
                last_view = _compose_view(
                    status, panel, show_base=session_ink is not None
                )
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
                        if loop.baseline_armed:
                            status_text = (
                                "Folha detectada: capturando a pagina "
                                "base, maos fora da folha"
                            )
                        else:
                            status_text = (
                                "Folha detectada: aperte C para comecar (pagina base)"
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
                    ink_mask = ink_module.extract_ink(warped, clear_markers=True)
                    canvas = ink_module.render_canvas(ink_mask, out=view_canvas_buffer)
                elif loop.accumulated_canvas is not None:
                    canvas = loop.accumulated_canvas
                elif page_clean is not None:
                    # Tinta assincrona: ate a trabalhadora publicar a
                    # primeira mascara desta folha, o painel e a folha
                    # em branco. Rodar o modelo aqui, sincrono, era o
                    # congelamento que a trabalhadora existe para tirar.
                    canvas = _blank_sheet()
                else:
                    ink_mask = ink_source(
                        paper_module.warp_sheet(
                            frame,
                            loop.display_corners
                            if loop.display_corners is not None
                            else tracker.corners,
                        )
                    )
                    canvas = ink_module.render_canvas(ink_mask, out=view_canvas_buffer)
                last_view = _compose_view(
                    status, canvas, show_base=session_ink is not None
                )
                cv2.imshow("Draw", last_view)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    save_path.mkdir(parents=True, exist_ok=True)
                    out = save_path / f"canvas_{int(time.time())}.png"
                    if ink_worker is not None and loop.clean_mask is not None:
                        # O snapshot e um consumidor de canvas: ele
                        # salva a camada confirmada, nunca o painel
                        # com a tinta provisoria.
                        save_image = ink_module.render_canvas(loop.clean_mask)
                    else:
                        save_image = canvas
                    png_writer.save(
                        out,
                        save_image,
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

    finally:
        # Epilogo sempre roda: captura liberada, trabalhadora de
        # tinta parada, fila de PNG drenada e janela destruida
        # mesmo quando o laco morre por excecao. Sem o finally, um
        # erro no meio do laco vazava a camera aberta e a thread
        # daemon ficava lendo o dispositivo ate o processo sair.
        frame_source.stop()
        if detection_worker is not None:
            detection_worker.stop()
            if trace_acquisitions:
                # Aquisicoes que nenhum quadro do rastreador consumiu
                # (latest-wins ou fim de sessao) tambem viram evento.
                for trace in detection_worker.take_traces():
                    _emit_acquisition(
                        telemetry,
                        session_recorder,
                        _acquisition_record(trace, None, False, None, None, None),
                    )
        if loop.hand_presence_worker is not None:
            loop.hand_presence_worker.stop()
        if ink_worker is not None:
            ink_worker.stop()
        if session_recorder is not None:
            # O manifesto fecha a sessao do loop: pares
            # antes/depois, shas dos modelos, teclas e o retrato
            # de telemetria da trabalhadora (U0).
            try:
                worker_telemetry = (
                    ink_worker.telemetry_snapshot() if ink_worker is not None else None
                )
                if worker_telemetry is not None:
                    session_recorder.note_ink_telemetry(worker_telemetry)
                manifest_path = session_recorder.finish(telemetry=worker_telemetry)
                print(f"Sessao gravada para o loop em {manifest_path}")
            except Exception as error:  # noqa: BLE001 - aviso, nao queda
                print(f"Aviso: nao consegui fechar a gravacao da sessao ({error}).")
        telemetry.close()
        session_report.close()
        png_writer.close()
        cv2.destroyAllWindows()


def _report_format_suggestion(telemetry, suggestion):
    telemetry.event("format_suggestion", **suggestion)
    ratio = suggestion["ratio"]
    ratio_text = "indisponivel" if ratio is None else f"{ratio:.4f}"
    label = suggestion["suggestion"]
    if label == "serie_a":
        label = (
            "serie A (proporcao 0,71) — tamanho nao estimado; "
            "A4 assumido se nada for escolhido"
        )
    elif label is None:
        label = f"abstencao ({suggestion['reason']})"
    print(f"Sugestao de formato: proporcao {ratio_text}; {label}.")


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
        "reserva. Sem a flag, o padrao e models/draw_seg_v3_dyn.onnx "
        "se o arquivo existir, senao a variavel de ambiente "
        "DRAW_NEURAL_MODEL",
    )
    parser.add_argument(
        "--page-clean",
        nargs="?",
        const="",
        default=None,
        metavar="MODELO.onnx",
        help="modelo pagina->limpa (ONNX) como fonte da tinta no "
        "caminho padrao. Sem caminho, usa a variavel de ambiente "
        "DRAW_PAGE_CLEAN_MODEL ou models/page_clean_unet.onnx",
    )
    parser.add_argument(
        "--legacy-ink",
        action="store_true",
        help="pilha antiga de tinta (heuristica, --ink-model, "
        "--session-baseline, --hand-model) no lugar do modelo "
        "pagina->limpa",
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
        help="valida a camera local porta por porta e sai, sem abrir janela",
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
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="carrega modelos divergentes ou fora de models/MANIFEST.json "
        "com aviso, em vez de recusar (equivale a DRAW_MODEL_STRICT=0)",
    )
    parser.add_argument(
        "--format",
        choices=list(config.FORMATS),
        default=None,
        help="formato da sessao em retrato (flag > DRAW_FORMAT > a4)",
    )
    args = parser.parse_args()
    if args.allow_unverified:
        manifest_module.allow_unverified()
        print(
            "AVISO: --allow-unverified ativo; modelos divergentes ou fora do "
            "manifesto serao carregados apenas com aviso."
        )
    try:
        config.apply_format(args.format or os.environ.get("DRAW_FORMAT", "a4"))
    except ValueError as error:
        parser.error(str(error))
    if args.check:
        parsed = parse_source(args.source)
        sys.exit(run_check(parsed if isinstance(parsed, int) else 0))
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
    elif not args.markers:
        # Padrao do reinicio: a folha e neural sempre que o modelo
        # v3 estiver no lugar de sempre. Sem o arquivo, neural_model
        # fica None e run() avisa que caiu no classico puro.
        default_sheet = (
            Path(__file__).resolve().parent.parent / "models" / "draw_seg_v3_dyn.onnx"
        )
        if default_sheet.exists():
            neural_model = str(default_sheet)
        else:
            neural_model = os.environ.get("DRAW_NEURAL_MODEL")
    page_clean_model = None
    if args.page_clean is not None:
        if args.markers:
            parser.error("--page-clean nao combina com --markers")
        page_clean_model = (
            args.page_clean
            or os.environ.get("DRAW_PAGE_CLEAN_MODEL")
            or str(page_clean_module.DEFAULT_MODEL_PATH)
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
        hand_model = args.hand_model or os.environ.get("DRAW_HAND_MODEL") or ""
    try:
        run(
            source=args.source,
            save_dir=args.save_dir,
            markers_mode=args.markers,
            neural_model=neural_model,
            hand_model=hand_model,
            ink_model=ink_model,
            session_baseline=args.session_baseline,
            page_clean_model=page_clean_model,
            legacy_ink=args.legacy_ink,
        )
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Erro: {error}")
        sys.exit(2)


if __name__ == "__main__":
    main()
