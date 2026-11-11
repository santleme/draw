"""Orquestracao do laco sem marcadores: folha, geracao e tinta.

E o miolo do laco de `draw.app.run` (deteccao -> rastreador ->
sessao de tinta -> trabalhadora -> canvas), extraido sem mudar a
ordem das operacoes, as flags nem os padroes. `run()` monta este
objeto com as dependencias reais (threads de deteccao, de tinta e
de presenca de mao, relogio monotonico, warp e LK do OpenCV) e
chama `step()` a cada quadro; a exibicao, as teclas, a telemetria
por segundo e o epilogo continuam em `run()`.

Os replays de `tools/` montam o MESMO objeto com dependencias
simuladas (camera de cena, detector falso com latencia de cena,
relogio deterministico e a trabalhadora real em `pump()`, sem
thread), para que toda mudanca neste laco seja medida por eles
sem espelho a manter.
"""

import os
import time
from collections import deque

import cv2
import numpy as np

from . import ink as ink_module
from . import paper as paper_module
from .ink_layers import _render_layered_canvas
from .sheet_track import _SheetTracker, _lk_track_corners


# A tinta e lida numa geometria estabilizada: a mediana dos ultimos
# conjuntos de cantos do rastreador. O jitter de canto entre deteccoes
# (media de ~5 px, com saltos de dezenas de pixels quando a mao cobre
# um canto) desloca o traco de 1 a 3 px no espaco do papel e desfaz o
# voto por pixel do canvas acumulativo; a mediana de 5 segura a
# geometria no lugar por tempo bastante para a janela do acumulador.
_INK_CORNER_HISTORY = 5


# Desfecho de step() quando a trabalhadora de tinta morreu: run()
# mostra o aviso na janela e sai do laco.
INK_WORKER_FAILED = "ink_worker_failed"


def _acquisition_record(trace, frame_index, consumed, before, after, tracker_event):
    """Evento por aquisicao: rastro do combinado + rastreador.

    detected_frame_index (do rastro) e a passada cujo quadro o
    detector examinou; frame_index e a passada em que o rastreador
    consumiu o resultado (com deteccao atrasada, posterior).
    """
    return {
        **trace,
        "frame_index": frame_index,
        "consumed_by_tracker": consumed,
        "tracker_before": before,
        "tracker_after": after,
        "tracker_event": tracker_event,
    }


def _bgr_to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _ignore(*_args, **_kwargs):
    return None


class SheetInkLoop:
    """Estado e passo do laco sem marcadores, dependencias injetadas.

    detection_worker: poll/submit/error/take_traces de
    `_SheetDetectionWorker`. ink_worker: `_PageCleanWorker` (caminho
    pagina->limpa) ou None (pilha antiga: session_ink, ink_source e
    o InkAccumulator). hand_presence_worker: `_HandPresenceWorker`
    ou None. clock: relogio monotonico em segundos (o mesmo das
    trabalhadoras). warp_sheet, to_gray e track_display: o warp da
    folha e o LK do quad de exibicao; render_display: o painel em
    camadas (confirmada + provisoria). Callbacks: on_format_suggestion
    (sugestao de formato), on_acquisition (evento por aquisicao),
    save_page_snapshot(prefixo, mascara) (pagina anterior na virada
    ou na divisao) e log (console).
    """

    def __init__(
        self,
        *,
        detection_worker,
        ink_worker=None,
        hand_presence_worker=None,
        hand_masker=None,
        session_ink=None,
        ink_source=None,
        session_recorder=None,
        trace_acquisitions=False,
        format_suggester=None,
        on_format_suggestion=_ignore,
        on_acquisition=_ignore,
        save_page_snapshot=_ignore,
        clock=time.monotonic,
        warp_sheet=paper_module.warp_sheet,
        to_gray=_bgr_to_gray,
        track_display=_lk_track_corners,
        render_display=_render_layered_canvas,
        log=print,
    ):
        self.detection_worker = detection_worker
        self.ink_worker = ink_worker
        self.hand_presence_worker = hand_presence_worker
        self.hand_masker = hand_masker
        self.session_ink = session_ink
        self._ink_source = ink_source
        self.session_recorder = session_recorder
        # Rastro por aquisicao: ligado so em sessao gravada.
        self.trace_acquisitions = trace_acquisitions
        self.format_suggester = (
            paper_module.FormatSuggester()
            if format_suggester is None
            else format_suggester
        )
        self._on_format_suggestion = on_format_suggestion
        self._on_acquisition = on_acquisition
        self._save_page_snapshot = save_page_snapshot
        self._clock = clock
        self._warp_sheet = warp_sheet
        self._to_gray = to_gray
        self._track_display = track_display
        self._render_display = render_display
        self._log = log

        self.last_acq_seq = None
        # Geracao do gravador no caminho legado (sem worker, a geracao
        # e local: abre na folha nova e na tecla C).
        self.legacy_generation = 0
        self.restart_reference = False
        self.baseline_armed = False
        # Geometria da tinta CONGELADA na tecla C: os warps da sessao
        # usam estes cantos, nao os do rastreador ao vivo. Medido nos
        # frames reais de 2026-10-30: sobre o caderno aberto, o ajuste
        # do detector oscila ate 314 px entre frames de uma cena
        # parada; com a geometria viva, o warp teleporta e nenhuma
        # referencia sobrevive. O rastreador segue vivo para o
        # contorno verde e os banners; a tinta vive na geometria do C.
        self.session_corners = None
        self.freeze_pending = False
        self.detect_ms = None
        self.sheet_present = False
        self.clean_hand_present = None
        self.hand_presence_seq = 0

        self.tracker = _SheetTracker()
        self.ink_accumulator = ink_module.InkAccumulator()
        self.corner_history = deque(maxlen=_INK_CORNER_HISTORY)
        self.ink_session_open = False
        self.accumulated_canvas = None
        # Canvas do caminho novo: a mascara acumulada e a uniao simples
        # das mascaras do modelo pagina->limpa, sem acumulador temporal.
        # None enquanto nenhuma folha foi lida nesta sessao.
        self.clean_mask = None
        # Geracao da tinta assincrona: acompanha a geracao da
        # trabalhadora; folha nova sobe a geracao. Ao cair do tracker,
        # a identidade fica pendente ate a folha voltar e o conteudo
        # decidir se e re-registro ou pagina nova. (A versao aplicada do poll()
        # saiu na orquestracao em camadas: o laco le o canvas pelo
        # poll_display(), e a clean_applied_version ficou morta.)
        self.clean_generation = 0
        self.clean_display_version = 0
        self.clean_provisional_mask = None
        self.clean_recovery_pending = False
        # Vigia de identidade (frente 28): seq do ultimo salto grande
        # de pose ja sinalizado a trabalhadora.
        self.last_big_jump_seq = 0
        # Buffer dono do canvas acumulado: mora aqui de um frame de
        # deteccao ao outro, e nenhum outro uso pode escrever nele.
        self.canvas_buffer = np.empty(
            (ink_module.config.PAPER_H, ink_module.config.PAPER_W, 3),
            dtype=np.uint8,
        )
        self.frame_number = 0
        self.last_det_seq = 0
        self.display_corners = None
        self.display_gray = None
        # Observabilidade (sem efeito no laco): ultimo evento do
        # rastreador e contagem das aberturas de geracao por causa.
        self.last_tracker_event = None
        self.generation_events = {"drop": 0, "reopen": 0, "turn": 0, "split": 0}

    def step(self, frame, frame_capture_ms, ink_passes_total=None):
        """Um quadro do laco sem marcadores.

        frame_capture_ms e o carimbo de captura (relogio do clock, em
        ms). ink_passes_total so alimenta o quadro cru do gravador.
        Devolve None, ou INK_WORKER_FAILED quando a trabalhadora de
        tinta morreu (o aviso ja saiu no log).
        """
        det_result = self.detection_worker.poll(self.last_det_seq)
        if det_result is not None:
            (
                self.last_det_seq,
                detected_corners,
                detected_verdict,
                det_captured_at,
                self.detect_ms,
            ) = det_result
            ran_detection = True
        else:
            detected_corners, detected_verdict = None, None
            det_captured_at = None
            ran_detection = False
        if self.tracker.should_detect(self.frame_number):
            self.detection_worker.submit(
                frame, frame_capture_ms / 1000.0, self.frame_number
            )
        if self.detection_worker.error is not None:
            raise self.detection_worker.error
        tracker_before = (
            self.tracker.diagnostic_state()
            if self.trace_acquisitions and ran_detection
            else None
        )
        tracker_event = self.tracker.update(
            detected_corners,
            detected_verdict,
            ran_detection,
            det_captured_at if ran_detection else self._clock(),
        )
        if self.trace_acquisitions:
            if ran_detection:
                tracker_after = self.tracker.diagnostic_state()
                for trace in self.detection_worker.take_traces(self.last_det_seq):
                    consumed = trace["result_seq"] == self.last_det_seq
                    if consumed:
                        self.last_acq_seq = trace["acq_seq"]
                    self._on_acquisition(
                        _acquisition_record(
                            trace,
                            self.frame_number,
                            consumed,
                            tracker_before if consumed else None,
                            tracker_after if consumed else None,
                            tracker_event if consumed else None,
                        )
                    )
            self.session_recorder.note_raw_frame(
                frame,
                self.frame_number,
                "sheet"
                if self.tracker.corners is not None
                else "candidate"
                if self.tracker.has_pending
                else "no_sheet",
                capture_ms=frame_capture_ms,
                acq_seq=self.last_acq_seq,
                ink_passes_total=ink_passes_total,
            )
        self.last_tracker_event = tracker_event
        if tracker_event == "drop":
            self.generation_events["drop"] += 1
            self.ink_session_open = False
            self.session_corners = None
            self.freeze_pending = False
            self.clean_mask = None
            if self.ink_worker is not None:
                # A queda do tracker nao prova que a pagina
                # mudou. Guarda a geracao ate a folha voltar;
                # a primeira mascara decide pela tinta se e
                # re-registro ou pagina nova.
                self.clean_recovery_pending = True
        if self.tracker.big_jump_seq != self.last_big_jump_seq:
            self.last_big_jump_seq = self.tracker.big_jump_seq
            if (
                self.ink_worker is not None
                and self.ink_session_open
                and os.environ.get("DRAW_IDENTITY_WATCH", "").strip() == "1"
                and (self.tracker.last_big_jump or {}).get("episode_discontinuous")
            ):
                # Salto grande de pose com a folha presente
                # E episodio descontínuo (frentes 28/32): so
                # ai o juiz de identidade e armado no meio
                # da geracao. Reposicionamento continuo da
                # mesma folha (candidato fixo, papel o tempo
                # todo, sem perda) NAO arma: neste material
                # (spread enrolado), o conteudo deformado e
                # indistinguivel de pagina nova pro juiz, e
                # partir um desenho por reposicionamento e o
                # erro mais caro. A troca em repouso segue
                # com a vigia de conteudo.
                self.ink_worker.arm_identity_check("pose_jump")

        self.frame_number += 1
        # Presenca e o estado do rastreador: warp_sheet com
        # cantos nunca devolve None, entao o warp por quadro
        # era so um teste de presenca caro, e o resultado era
        # descartado. O unico consumo dos pixels fora do bloco
        # de tinta e o fallback de exibicao, que retifica sob
        # demanda.
        self.sheet_present = self.tracker.corners is not None
        if ran_detection and self.sheet_present and detected_corners is not None:
            suggestion = self.format_suggester.observe(detected_corners, frame.shape)
            if suggestion is not None:
                self._on_format_suggestion(suggestion)

        # Presenca de mao assincrona (T4): colhe o
        # resultado mais novo da trabalhadora (o estado
        # clean_hand_present que o congelamento de
        # geometria le abaixo) e alimenta a porta da
        # tinta. O atraso e de ate um ciclo de amostragem
        # frente ao desenho sincrono antigo — irrelevante
        # na cadencia de 250 ms da tinta e na janela de
        # quietude de 1 s da mao.
        if self.hand_presence_worker is not None:
            if self.hand_presence_worker.error is not None:
                # Presenca morta degrada para "sem sinal":
                # a cadencia trata None como ativo e o app
                # segue como antes da porta existir.
                self._log(
                    "Aviso: a presenca de mao falhou e foi "
                    "desligada: "
                    f"{self.hand_presence_worker.error}. A tinta "
                    "segue na cadencia normal."
                )
                self.hand_presence_worker = None
                self.clean_hand_present = None
                if self.ink_worker is not None:
                    self.ink_worker.set_hand_presence(None)
            hp_result = (
                self.hand_presence_worker.poll(self.hand_presence_seq)
                if self.hand_presence_worker is not None
                else None
            )
            if hp_result is not None:
                (
                    self.hand_presence_seq,
                    self.clean_hand_present,
                    clean_hand_moved,
                ) = hp_result
                if self.ink_worker is not None:
                    self.ink_worker.set_hand_presence(
                        self.clean_hand_present,
                        moved=clean_hand_moved,
                    )
        # Quad de EXIBICAO (T9): em quadro com resultado de
        # deteccao, o exibido e o confirmado; nos demais,
        # propaga por LK sobre o cinza anterior. Falha de
        # propagacao volta ao confirmado. So exibicao: a
        # tinta nunca le este quad.
        if self.tracker.corners is None:
            self.display_corners = None
            self.display_gray = None
        else:
            gray_now = self._to_gray(frame)
            if (
                ran_detection
                or self.display_corners is None
                or self.display_gray is None
            ):
                self.display_corners = self.tracker.corners.copy()
            else:
                tracked_display = self._track_display(
                    self.display_gray, gray_now, self.display_corners
                )
                self.display_corners = (
                    tracked_display
                    if tracked_display is not None
                    else self.tracker.corners.copy()
                )
            self.display_gray = gray_now
        if ran_detection and self.tracker.corners is not None:
            if self.hand_presence_worker is not None:
                # Submete com o carimbo de captura: e ele
                # que alimenta o modo VIDEO do landmarker.
                self.hand_presence_worker.submit(frame, frame_capture_ms)
            if not self.ink_session_open:
                # Folha nova na cena: canvas e geometria recomecam.
                self.ink_accumulator.reset()
                self.clean_mask = None
                # accumulated_canvas tambem zera: sem isto, o
                # painel continuava exibindo o canvas da folha
                # ANTERIOR ate a trabalhadora publicar a
                # primeira mascara da geracao nova (ate ~250 ms
                # de cadencia mais a passada do modelo), e a
                # tecla s salvava esse canvas errado. Medido no
                # harness headless: 22 quadros com a tinta velha
                # depois de a folha voltar.
                self.accumulated_canvas = None
                if self.ink_worker is not None:
                    self.generation_events["reopen"] += 1
                    self.clean_generation = self.ink_worker.new_generation(
                        preserve_identity=self.clean_recovery_pending
                    )
                    self.clean_recovery_pending = False
                    self.clean_display_version = 0
                    self.clean_provisional_mask = None
                if self.session_ink is not None:
                    self.session_ink.reset()
                    self.baseline_armed = False
                    self.session_corners = None
                    self.freeze_pending = False
                self.corner_history.clear()
                self.ink_session_open = True
                if self.session_recorder is not None and self.ink_worker is None:
                    self.legacy_generation += 1
                    self.session_recorder.start_generation(self.legacy_generation)
            if self.restart_reference:
                # Tecla C: novo ponto zero da sessao. O canvas
                # acumulado zera junto, porque a pagina base nova
                # redefine o que conta como traco desta sessao.
                # A geometria congela nos cantos deste momento.
                self.restart_reference = False
                self.ink_accumulator.reset()
                self.session_ink.reset()
                self.baseline_armed = True
                self.freeze_pending = True
                if self.session_recorder is not None:
                    # Tecla C: a geracao do gravador recomeca
                    # aqui; a referencia dela e o warp deste
                    # momento (a pagina base do legado).
                    self.legacy_generation += 1
                    self.session_recorder.start_generation(self.legacy_generation)
            # Sem mao, a geometria da tinta congela na
            # ultima mediana: a pagina esta assentada e a
            # mediana viva seguiria andando com o balanco
            # do detector, deslocando a tinta ja publicada.
            if self.clean_hand_present is not False or (not self.corner_history):
                self.corner_history.append(self.tracker.corners.copy())
            ink_corners = np.median(np.stack(self.corner_history), axis=0).astype(
                np.float32
            )
            if self.freeze_pending:
                self.session_corners = ink_corners.copy()
                self.freeze_pending = False
            warp_corners = (
                self.session_corners
                if self.session_ink is not None and self.session_corners is not None
                else ink_corners
            )
            ink_warped = self._warp_sheet(frame, warp_corners)
            # Visibilidade na grade da pagina: onde a mao
            # esta, a passada nao ve a folha — nenhum dos
            # dois caminhos pode votar com evidencia
            # fabricada sob oclusao. O mesmo mapa serve a
            # trabalhadora neural e ao caminho classico.
            visible = None
            if self.hand_masker is not None:
                camera_mask = self.hand_masker.mask(frame)
                if camera_mask is not None:
                    hand_warped = self._warp_sheet(camera_mask, warp_corners)
                    visible = hand_warped == 0
            if self.ink_worker is not None:
                # Caminho novo: o warp vai para a trabalhadora
                # de tinta e o laco segue sem esperar o modelo.
                # Sem referencia e sem pagina base: as camadas
                # moram na trabalhadora e entram no painel
                # pelo poll_display() logo abaixo, a provisoria
                # na primeira passada e a confirmada quando o
                # voto fecha. Os cantos vao junto: sao o quad
                # que retificou ESTE warp, e a trabalhadora
                # registra a evidencia de cada passada na
                # geometria ancora da geracao — sem isso, o
                # quad que anda entre passadas re-projeta a
                # tinta assentada e cada alinhamento confirma
                # a sua copia (a reimpressao da regua de
                # 2026-11-05 em draw/ink_layers.py).
                self.ink_worker.submit(
                    ink_warped,
                    self.clean_generation,
                    warp_corners,
                    visible,
                )
                if self.session_recorder is not None:
                    self.session_recorder.note_warp(self.clean_generation, ink_warped)
            else:
                if self.session_ink is not None:
                    if not self.session_ink.ready:
                        if self.baseline_armed:
                            self.session_ink.add_sample(ink_warped, visible)
                        observed = np.zeros(ink_warped.shape[:2], dtype=np.uint8)
                    else:
                        observed = self.session_ink.observe(ink_warped, visible)
                else:
                    observed = self._ink_source(ink_warped)
                self.accumulated_canvas = ink_module.render_canvas(
                    self.ink_accumulator.update(
                        observed, visible, t_capture_ms=frame_capture_ms
                    ),
                    out=self.canvas_buffer,
                )
                if self.session_recorder is not None:
                    self.session_recorder.note_warp(self.legacy_generation, ink_warped)
                    self.session_recorder.note_canvas(
                        self.legacy_generation, self.accumulated_canvas
                    )

        return self.collect_ink()

    def collect_ink(self):
        """Colhe a tinta assincrona: painel, virada, divisao e erro.

        E o fim de step(); quem dirige a trabalhadora em pump() logo
        depois de step() chama de novo para o consumidor ver a passada
        no mesmo quadro. Devolve None ou INK_WORKER_FAILED.
        """
        if self.ink_worker is None:
            return None
        # Colhe a tinta assincrona pelo canal de
        # exibicao: a confirmada vira o canvas
        # (clean_mask, o que snapshot e consumidores
        # leem) e a provisoria entra so no painel, em
        # cinza. Sem versao nova, o painel exibido e o
        # mesmo de antes, sem custo de modelo neste laco.
        got = self.ink_worker.poll_display(
            self.clean_generation, self.clean_display_version
        )
        if got is not None:
            (
                self.clean_display_version,
                self.clean_mask,
                self.clean_provisional_mask,
            ) = got
            self.accumulated_canvas = self._render_display(
                self.clean_mask,
                self.clean_provisional_mask,
                out=self.canvas_buffer,
            )
            if self.session_recorder is not None:
                self.session_recorder.note_canvas(
                    self.clean_generation, self.accumulated_canvas
                )
        # Virada de pagina (U3): a trabalhadora detectou
        # conteudo descorrelacionado do canvas por 2
        # passadas. A pagina anterior vira snapshot em
        # disco (nao se perde) e a geracao nova comeca em
        # branco, com passada imediata (U4 via
        # new_generation).
        turn_canvas = self.ink_worker.take_turn_event(self.clean_generation)
        if turn_canvas is not None:
            self.generation_events["turn"] += 1
            if self.clean_mask is not None:
                self._save_page_snapshot("canvas_turn", turn_canvas)
            self.clean_generation = self.ink_worker.new_generation()
            self.clean_display_version = 0
            self.clean_provisional_mask = None
            self.clean_mask = None
            self.accumulated_canvas = None
        # Divisao por identidade (frente 28): o juiz,
        # armado no meio da geracao pela vigia (salto de
        # pose ou ruptura de conteudo), decidiu pagina
        # nova com a folha presente o tempo todo. A
        # pagina anterior sai como snapshot (nao se
        # perde) e a geracao nova comeca, como na
        # virada U3 — sem isto, virar a pagina somava
        # as duas no mesmo canvas.
        identity_split = self.ink_worker.take_identity_split(self.clean_generation)
        if identity_split is not None:
            self.generation_events["split"] += 1
            split_canvas, _split_reason = identity_split
            if self.clean_mask is not None:
                self._save_page_snapshot("canvas_page", split_canvas)
            self.clean_generation = self.ink_worker.new_generation()
            self.clean_display_version = 0
            self.clean_provisional_mask = None
            self.clean_mask = None
            self.accumulated_canvas = None
        # Trabalhadora morta por excecao: sem isto o canvas
        # congelava no ultimo quadro e nem o console nem a
        # janela diziam nada (o erro morria na thread).
        worker_error = self.ink_worker.error
        if worker_error is not None:
            self._log(
                "A tinta pagina->limpa falhou e a thread "
                f"parou: {worker_error}. O canvas nao avanca."
            )
            return INK_WORKER_FAILED
        return None
