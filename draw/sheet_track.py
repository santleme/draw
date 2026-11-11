"""Folha: estabilizacao dos cantos, tracking optico e deteccao.

O rastreador confirma o quad da folha com filtro de mediana,
deadband e persistencia de salto (o detector cru anda dezenas de
pixels por frame numa cena parada); entre deteccoes, os cantos de
EXIBICAO sao propagados por Lucas-Kanade — a geometria da tinta
segue lendo os cantos confirmados, nunca os propagados. A
trabalhadora de deteccao roda a deteccao em thread propria,
latest-wins, e alimenta o rastreador com o carimbo de captura.

Extraido de draw/app.py na modularizacao de 2026-11-05; o app
re-exporta todos os nomes deste modulo.
"""

import threading
import time
from collections import deque

import cv2
import numpy as np


_SHEET_NEAR_LIMIT_PX = 40.0


_SHEET_BLEND_ALPHA = 0.3


# Estabilizacao contra o jitter do detector (auditoria de
# 2026-11-02, medido em laboratorio pelo agente): nos 5 frames
# reais de uma cena parada da mesa do Lucas, os cantos crus do
# detector andam em media 55 px por frame (maximo 156 px) e os
# cantos exibidos andavam em media 9 px (maximo 101 px) — a folha
# parada "se mexia" na tela. Tres defesas, nesta ordem:
# (1) a deteccao que alimenta o rastreador e a mediana das ultimas
# _SHEET_FILTER_WINDOW deteccoes, que apaga o salto isolado de um
# frame so; (2) deteccao filtrada a ate _SHEET_DEADBAND_PX do quad
# atual confirma presenca SEM mover o quad — abaixo disso e ruido
# de medida, nao movimento, e a folha parada fica bit-estavel;
# (3) o candidato a salto fica FIXO na primeira deteccao que o
# abriu: antes ele era misturado a cada deteccao nova e "andava"
# ate confirmar um salto que nenhuma posicao sustentou pelo tempo
# de persistencia.
_SHEET_FILTER_WINDOW = 3


_SHEET_DEADBAND_PX = 2.0


# Salto GRANDE (candidato a mais de _SHEET_BIG_JUMP_PX do quad
# atual) exige persistencia maior: e a assinatura do detector
# trocando de modo num ajuste biestavel (a folha cortada pela
# borda do quadro projeta cantos a ~100 px frame a frame), e um
# modo errado raramente se sustenta por 1 s em captura real, com
# frames novos a cada instante. Mover a folha de verdade por mais
# de 100 px paga ~0,7 s a mais de confirmacao; trocar de pagina
# nao passa por aqui (a folha some e volta, e folha nova confirma
# pelo caminho de primeira deteccao).
_SHEET_BIG_JUMP_PX = 100.0


_SHEET_BIG_JUMP_PERSIST_SECONDS = 1.0


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


# Backoff sem folha (pesquisa de performance de 2026-11-03, T8):
# sem folha na cena, should_detect pedia deteccao a CADA quadro —
# ~0,5 nucleo so para confirmar que continua nao havendo folha
# (regime S1 medido: processo a ~255% com a mesa vazia). Depois de
# _SHEET_ABSENT_BACKOFF_CALLS chamadas seguidas sem folha (a ~12
# fps, ~2 s), a cadencia cai para 1 a cada
# _SHEET_ABSENT_BACKOFF_INTERVAL chamadas. A graca apos a queda e
# parte do desenho: o contador recomeca na confirmacao e na
# queda, entao a folha que some e volta em seguida (virada de
# pagina) e a primeira deteccao de uma sessao seguem na cadencia
# cheia; quem paga o backoff e so a ausencia sustentada, com
# +ate ~3 quadros (~250 ms) na primeira deteccao.
_SHEET_ABSENT_BACKOFF_CALLS = 24


_SHEET_ABSENT_BACKOFF_INTERVAL = 4


# Perda da folha medida em tempo, nao em quadros: com a contagem em
# quadros, o mesmo limite valia ~1 s a 30 fps e ~3 s numa sessao lenta,
# e era esse o banner que ficava sobre a mesa vazia. O piso em quadros
# impede a queda entre duas deteccoes agendadas quando o loop roda
# devagar: a folha presente confirma de novo antes dos dois criterios.
_SHEET_LOST_SECONDS = 0.5


_SHEET_LOST_MIN_FRAMES = 10


def _corners_json(corners):
    """Cantos como lista JSON (2 casas), ou None. So diagnostico."""
    if corners is None:
        return None
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in points]


def _corner_distance(current, detected):
    return float(np.max(np.linalg.norm(current - detected, axis=1)))


def _blend_corners(current, detected):
    return (
        (1.0 - _SHEET_BLEND_ALPHA) * current + _SHEET_BLEND_ALPHA * detected
    ).astype(np.float32)


class _SheetTracker:
    """Estabilizacao temporal do quadrilatero da folha.

    Guarda o quadrilatero atual e decide, quadro a quadro, entre
    misturar uma deteccao vizinha, exigir persistencia para um salto
    de posicao e descartar a folha perdida. A deteccao crua passa
    antes pelo filtro de mediana e pela banda morta de
    _SHEET_DEADBAND_PX (ver as constantes). Deteccao perto mistura
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
        # Salto grande confirmado (fato geometrico, so leitura):
        # seq monotonica + retrato do ultimo salto, para o app
        # armar o juiz de identidade quando a pose salta com a
        # folha presente (frente 28, 2026-11-10). O tracker nao
        # decide nada com isto — so reporta.
        self.big_jump_seq = 0
        self.last_big_jump = None
        # Episodio do salto em curso foi DESCONTINUO? (candidato
        # substituido no meio, veredito nao-paper com candidato
        # ativo, ou candidato morto por perda de deteccao). Um
        # reposicionamento continuo da mesma folha nao marca nada
        # disto — e a distincao que a vigia v3 usa para armar (ou
        # nao) o juiz de identidade num salto de pose.
        self._episode_discontinuous = False
        self._pending_sustain_count = 0
        self._pending = None
        self._pending_since = None
        self._frames_since_seen = 0
        self._last_seen_at = None
        self._recent = deque(maxlen=_SHEET_FILTER_WINDOW)
        self._absent_calls = 0

    @property
    def has_pending(self):
        """Ha candidato a salto/primeira folha esperando confirmacao."""
        return self._pending is not None

    def diagnostic_state(self):
        """Retrato JSON do estado interno (leitura pura, so diagnostico)."""
        return {
            "present": self.corners is not None,
            "corners": _corners_json(self.corners),
            "verdict": self.verdict,
            "missed": bool(self.missed),
            "pending": _corners_json(self._pending),
            "pending_since": self._pending_since,
            "big_jump_seq": int(self.big_jump_seq),
            "last_big_jump": self.last_big_jump,
            "frames_since_seen": int(self._frames_since_seen),
            "last_seen_at": self._last_seen_at,
            "absent_calls": int(self._absent_calls),
        }

    def should_detect(self, frame_number):
        if self.corners is not None:
            self._absent_calls = 0
            return self.missed or frame_number % _SHEET_DETECT_INTERVAL == 0
        self._absent_calls += 1
        if self._absent_calls <= _SHEET_ABSENT_BACKOFF_CALLS:
            return True
        return self._absent_calls % _SHEET_ABSENT_BACKOFF_INTERVAL == 0

    def _confirm(self, corners, verdict, now):
        if self.corners is not None:
            jump_distance = _corner_distance(self.corners, corners)
            if jump_distance > _SHEET_BIG_JUMP_PX:
                self.big_jump_seq += 1
                self.last_big_jump = {
                    "seq": self.big_jump_seq,
                    "distance_px": float(jump_distance),
                    "at": now,
                    "episode_discontinuous": bool(self._episode_discontinuous),
                }
        # Toda confirmacao encerra o episodio do candidato: o
        # marcador de descontinuidade nao vaza pro proximo salto.
        self._episode_discontinuous = False
        # Confirmacao que SALTA (primeira folha ou candidato a
        # salto) recomeca a janela do filtro na posicao nova: sem
        # isso, a mediana das proximas deteccoes misturaria a
        # posicao velha com a nova e arrastaria o quad confirmado.
        if self.corners is None or (
            _corner_distance(self.corners, corners) > _SHEET_NEAR_LIMIT_PX
        ):
            self._recent.clear()
            self._recent.append(np.asarray(corners, dtype=np.float32).copy())
        self.corners = corners
        self.verdict = verdict
        self._pending = None
        self._pending_since = None
        self._pending_sustain_count = 0
        self._frames_since_seen = 0
        self._last_seen_at = now
        self.missed = False

    def _filtered(self, detected):
        """Mediana das ultimas deteccoes: apaga o salto de 1 frame."""
        self._recent.append(np.asarray(detected, dtype=np.float32).copy())
        return np.median(np.stack(self._recent), axis=0).astype(np.float32)

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
            detected = self._filtered(detected)
            if self.corners is None and verdict == "paper":
                self._confirm(detected.copy(), verdict, now)
                event = "seen"
            elif (
                self.corners is not None
                and _corner_distance(self.corners, detected) <= _SHEET_DEADBAND_PX
            ):
                # Dentro da banda morta: presenca confirmada, quad
                # NAO se move. Folha parada fica bit-estavel em vez
                # de nadar 30% do ruido a cada deteccao.
                self._confirm(self.corners.copy(), verdict, now)
                event = "seen"
            elif (
                self.corners is not None
                and _corner_distance(self.corners, detected) <= _SHEET_NEAR_LIMIT_PX
            ):
                self._confirm(_blend_corners(self.corners, detected), verdict, now)
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
                    self.corners is not None
                    and self._pending is not None
                    and _corner_distance(self.corners, self._pending)
                    > _SHEET_BIG_JUMP_PX
                ):
                    confirm_seconds = max(
                        confirm_seconds, _SHEET_BIG_JUMP_PERSIST_SECONDS
                    )
                if self._pending is not None and verdict != "paper":
                    # Candidato ativo julgado nao-papel no meio do
                    # episodio: a transicao nao foi uma folha so
                    # andando em cena.
                    self._episode_discontinuous = True
                if (
                    self._pending is not None
                    and _corner_distance(self._pending, detected)
                    <= _SHEET_NEAR_LIMIT_PX
                ):
                    # Candidato FIXO: a deteccao nova sustenta o
                    # candidato que ja existe, sem move-lo. Misturar
                    # o candidato a cada deteccao deixava ele andar
                    # atras dos modos errantes do detector e
                    # confirmar um salto que nenhuma posicao fixa
                    # sustentou pelo tempo de persistencia.
                    self._pending_sustain_count += 1
                else:
                    if self._pending is not None and self._pending_sustain_count >= 2:
                        # Candidato ESTABELECIDO substituido no meio
                        # do episodio: transicao descontinua. (A
                        # troca nas primeiras deteccoes e o transi-
                        # ente do filtro temporal varrendo da pose
                        # velha pra nova — nao marca.)
                        self._episode_discontinuous = True
                    self._pending = detected.copy()
                    self._pending_since = now
                    self._pending_sustain_count = 0

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
                if self._pending is not None:
                    # Candidato morto por perda de deteccao no meio
                    # do episodio: transicao descontinua.
                    self._episode_discontinuous = True
                self._pending = None
                self._pending_since = None
                self._pending_sustain_count = 0
                self.missed = True
            self._frames_since_seen += 1

        # A perda so e avaliada em quadro com resultado de deteccao.
        # Desde que a deteccao virou thread propria (carimbo de
        # captura no resultado), avaliar nos quadros intermediarios
        # do laco mistura os dois relogios: _last_seen_at guarda o
        # carimbo de CAPTURA do ultimo quadro visto, enquanto o
        # `now` intermediario e o instante corrente — a janela de
        # _SHEET_LOST_SECONDS encolhe pela idade do pipeline
        # (submissao->resultado). Medido em laboratorio pelo agente
        # (bancada com o worker e o rastreador reais, detector
        # perfeito, folha parada): com deteccao de 450 ms, a folha
        # confirmada caia 32 vezes em 15 s, reconfirmando em seguida
        # — a folha "some e volta" sem nada ter mudado na mesa. Nos
        # quadros com resultado, os dois carimbos sao de captura e a
        # idade do pipeline cancela: a perda mede o tempo de cena
        # desde o ultimo quadro examinado que viu a folha, e a folha
        # removida de verdade segue caindo ~_SHEET_LOST_SECONDS
        # depois (mais a latencia de uma deteccao).
        if (
            ran_detection
            and self.corners is not None
            and self._frames_since_seen > _SHEET_LOST_MIN_FRAMES
            and self._last_seen_at is not None
            and now - self._last_seen_at > _SHEET_LOST_SECONDS
        ):
            self.corners = None
            self.verdict = None
            self._pending = None
            self._pending_since = None
            self._pending_sustain_count = 0
            self._episode_discontinuous = False
            self._frames_since_seen = 0
            self._recent.clear()
            self.missed = False
            # Graca do backoff: a ausencia recomeca a contar da
            # queda — folha que volta em seguida (virada) e
            # detectada na cadencia cheia.
            self._absent_calls = 0
            event = "drop"
        return event


# Tracking optico dos cantos para EXIBICAO (pesquisa de
# performance de 2026-11-03, T9): entre deteccoes, o quad exibido
# ficava congelado no ultimo quad confirmado enquanto a camera e
# a folha se movem de verdade — a folha "nadava" contra o
# contorno. Propagar os 4 cantos por Lucas-Kanade com checagem
# ida-e-volta custa ~nada (4 pontos) e mantem o contorno colado.
# Validacao offline em clips SmartDoc contra o detector: erro
# mediano de canto de 2,0-2,8 px apos 7-11 passos de propagacao,
# abaixo do passo quadro-a-quadro do proprio detector (~5,4 px);
# a checagem ida-e-volta rejeitou 1 de 5 segmentos (movimento
# rapido), e a rejeicao cai para o quad confirmado — o
# comportamento anterior. CONTRATO: o resultado alimenta so o
# desenho do contorno e o warp de exibicao legado; a geometria
# da tinta segue lendo tracker.corners (confirmado), nunca o
# propagado — mover o caderno nao move o desenho.
_LK_MAX_STEP_PX = 80.0


_LK_MAX_FB_ERROR_PX = 1.5


def _lk_track_corners(prev_gray, next_gray, corners):
    """Cantos propagados por LK, ou None se a propagacao nao e
    confiavel (ponto perdido, ida-e-volta inconsistente ou salto
    maior que _LK_MAX_STEP_PX)."""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    # Janela 31x31 e piramide de 2 niveis: com 21x21 e 3 niveis, a
    # volta da checagem divergia em janelas de textura repetitiva
    # mesmo com a ida perfeita (medido no teste sintetico denso e
    # na taxa de rejeicao da validacao em video real).
    tracked, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        points,
        None,
        winSize=(31, 31),
        maxLevel=2,
        criteria=criteria,
    )
    if tracked is None or not np.all(status == 1):
        return None
    back, status_back, _err_b = cv2.calcOpticalFlowPyrLK(
        next_gray,
        prev_gray,
        tracked,
        None,
        winSize=(31, 31),
        maxLevel=2,
        criteria=criteria,
    )
    if back is None or not np.all(status_back == 1):
        return None
    fb_error = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
    if float(fb_error.max()) > _LK_MAX_FB_ERROR_PX:
        return None
    step = np.linalg.norm((tracked - points).reshape(-1, 2), axis=1)
    if float(step.max()) > _LK_MAX_STEP_PX:
        return None
    return tracked.reshape(-1, 2)


# Teto da espera pela thread de deteccao na saida do app. Uma
# deteccao nesta VM pode passar de 1 s; no Ryzen ela mede 36,7 ms.
_DETECTION_STOP_TIMEOUT_SECONDS = 5.0


class _SheetDetectionWorker:
    """Deteccao de folha em thread propria, latest-wins.

    O laco entrega o quadro com submit() e segue: a fila tem
    tamanho 1 e um quadro nao processado e substituido pelo
    proximo, nunca acumulado. A thread roda a deteccao e publica
    (sequencia, cantos, veredito, carimbo do quadro, duracao). O
    laco colhe com poll() sem bloquear, em qualquer estado do
    rastreador, inclusive no missed: os 36,7 ms medidos da
    deteccao no Ryzen saiam do laco de exibicao, e no missed
    saiam de todo quadro.

    O rastreador e alimentado com o carimbo de captura do quadro
    detectado, nao com o instante de chegada do resultado: os
    relogios de persistencia de salto (0,3 s) e de perda (0,5 s)
    medem o tempo da cena, e um resultado que chega atrasado nao
    pode inflar nem encolher esses relogios.

    Rastro por aquisicao (observabilidade, 2026-11-10): se detect
    devolve um terceiro elemento (dict de diagnostico), o worker o
    guarda por aquisicao com acq_seq, result_seq (None se a
    deteccao terminou depois do stop), carimbo de captura,
    detected_frame_index (o frame_index passado a submit: a passada
    cujo quadro foi examinado) e duracao; take_traces() entrega cada rastro exatamente uma vez.
    Com detect de dois elementos, nada disso existe.
    """

    def __init__(self, detect):
        self._detect = detect
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._work, name="draw-sheet-detection", daemon=True
        )
        self._pending = None
        self._result = None
        self._result_seq = 0
        self._stopped = False
        self._error = None
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._last_detect_ms = None
        self._acq_seq = 0
        self._traces = []

    def start(self):
        self._thread.start()
        return self

    def submit(self, frame, captured_at, frame_index=None):
        with self._condition:
            if self._stopped:
                return
            if self._pending is not None:
                self._dropped += 1
            self._pending = (frame, captured_at, frame_index)
            self._submitted += 1
            self._condition.notify_all()

    def poll(self, after_seq):
        """Resultado mais novo que after_seq, ou None. Nunca bloqueia."""
        with self._condition:
            if self._result is None or self._result[0] <= after_seq:
                return None
            return self._result

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopped
                )
                if self._stopped:
                    return
                frame, captured_at, frame_index = self._pending
                self._pending = None
            started = time.monotonic()
            try:
                outcome = self._detect(frame)
            except Exception as error:  # noqa: BLE001 - vai para o laco
                with self._condition:
                    self._error = error
                    self._stopped = True
                    self._condition.notify_all()
                return
            corners, verdict = outcome[0], outcome[1]
            trace = outcome[2] if len(outcome) > 2 else None
            detect_ms = round((time.monotonic() - started) * 1000.0, 1)
            with self._condition:
                if trace is not None:
                    self._acq_seq += 1
                    self._traces.append(
                        {
                            **trace,
                            "acq_seq": self._acq_seq,
                            "result_seq": (
                                None if self._stopped else self._result_seq + 1
                            ),
                            "captured_at": captured_at,
                            "detected_frame_index": frame_index,
                            "detect_ms": detect_ms,
                        }
                    )
                if self._stopped:
                    return
                self._completed += 1
                self._last_detect_ms = detect_ms
                self._result_seq += 1
                self._result = (
                    self._result_seq,
                    corners,
                    verdict,
                    captured_at,
                    detect_ms,
                )
                self._condition.notify_all()

    def take_traces(self, upto_result_seq=None):
        """Retira os rastros ate o resultado upto_result_seq.

        None retira todos, inclusive os de deteccao nao publicada
        (fim de sessao). Cada rastro sai uma unica vez.
        """
        with self._condition:
            if upto_result_seq is None:
                taken, self._traces = self._traces, []
                return taken
            taken, kept = [], []
            for trace in self._traces:
                seq = trace["result_seq"]
                if seq is not None and seq <= upto_result_seq:
                    taken.append(trace)
                else:
                    kept.append(trace)
            self._traces = kept
            return taken

    @property
    def error(self):
        """A excecao que matou a thread, ou None."""
        with self._condition:
            return self._error

    def stats(self):
        with self._condition:
            return {
                "detect_submitted": self._submitted,
                "detect_completed": self._completed,
                "detect_dropped": self._dropped,
                "detect_ms": self._last_detect_ms,
            }

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=_DETECTION_STOP_TIMEOUT_SECONDS)
