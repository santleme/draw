"""Tinta em camadas: acumulador com voto e a trabalhadora page-clean.

O acumulador guarda a tinta de uma geracao em duas camadas — a
provisoria, que a primeira passada viu, e a confirmada, que o voto
entre passadas fechou (com a porta de fronteira contra o ombro do
tremor). A trabalhadora roda as passadas em thread propria com a
cadencia de _PAGE_CLEAN_INTERVAL_SECONDS, a porta de ociosidade
sem mao, o descanso total de pagina assentada e o evento de
virada atras da flag DRAW_PAGE_TURN_EVENT.

Extraido de draw/app.py na modularizacao de 2026-11-05; o app
re-exporta todos os nomes deste modulo. As constantes de cadencia
e descanso sao lidas deste modulo em tempo de execucao: quem
precisa muda-las (os testes) patcheia aqui, onde a leitura mora.
"""

import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from . import ink as ink_module


# Porta de ociosidade sem mao: no maximo uma sonda a cada
# _IDLE_PROBE_SECONDS. A porta e a presenca da mao, nao um
# detector de mudanca por diff de warp: medido nesta webcam, o
# churn entre quadros parados (1,5-2,0) supera o sinal do lapis,
# entao diff cego dispara a toa ou engole o traco fraco. Escrever
# exige mao no papel; sem mao, a pagina esta assentada e so a
# sonda periodica confere.
_IDLE_PROBE_SECONDS = 1.0


# Sem sinal de mao (clone sem MediaPipe), a porta de ociosidade
# nunca fecha e a trabalhadora rodaria a 250 ms para sempre —
# CPU queimada a toa num app que o dono acha "lento". O descanso,
# nesse caso, vem da tinta parada: depois de
# _NO_HAND_REST_AFTER_STILL_PASSES passadas seguidas sem nenhuma
# mudanca no acumulador (nada confirmou, nada provisoriamente
# mudou), a cadencia cai para a da sonda ate algo mudar. O sinal
# e o do proprio acumulador, nao diff de quadro — o diff cego
# nesta webcam ja foi medido e descartado (churn de exposicao
# empata com o traco). Com sinal de mao presente, nada muda.
_NO_HAND_REST_AFTER_STILL_PASSES = 4


# Descanso total (especificacao do Lucas, 2026-11-01 e de novo
# 2026-11-03): folha assentada e saida confirmada quieta = ZERO
# passadas, nao sonda de 1 Hz para sempre. A sonda de 1 Hz
# seguia alimentando o voto sobre warps que tremem, e cada
# alinhamento de fase confirmava uma camada nova sobre o
# desenho parado (regua de replay, modo hold: salto de +9.999
# px na passada 89). O descanso entra quando a geracao ja rodou
# uma janela de voto inteira (_REST_MIN_GENERATION_PASSES) e as
# ultimas _REST_QUIET_PASSES passadas nao confirmaram nada novo
# — o portao e sobre a SAIDA CONFIRMADA do acumulador, nunca
# sobre diff de quadro (licao de 2026-11-01: o churn da camera
# supera o sinal do lapis). Acorda com evento que anuncia tinta
# possivel: a mao volta ou se mexe (set_hand_presence), a folha
# troca (new_generation) ou a geracao vira. Sem sinal de mao
# (None) o descanso total NAO fecha: nao existe evento que
# anuncie tinta nova, e a sonda de 1 Hz segue sendo o mecanismo
# de acordar desse clone — com a porta de fronteira, ela nao
# acumula mais.
_REST_MIN_GENERATION_PASSES = 3


_REST_QUIET_PASSES = 2


# Descanso total tambem COM a mao presente (decisao do Lucas,
# 2026-11-03): ele segurando o caderno parado, a cadencia de
# 250 ms seguia eterna. A extensao usa a mesma maquinaria de
# quietude da saida confirmada (_REST_*), mais a mao parada por
# _HAND_REST_STILL_SECONDS. Tinta nova com a mao ja no quadro
# nao se anuncia pela tinta — quem acorda e a mao se mexendo,
# medida na assinatura da mascara que o adaptador ja computa
# (area e centroide), nunca por diff do quadro da pagina (o
# churn da camera supera o sinal do lapis, licao de 2026-11-01).
# Nao ha sonda periodica de tinta neste caminho: sonda e
# passada, e o ponto do descanso e zero passadas.
_HAND_REST_STILL_SECONDS = 1.0


# Cadencia maxima da tinta: uma passada completa a cada ~250 ms.
# A passada custa 168 ms no Ryzen com 8 threads (perfil em
# draw-lab/page-clean-v0/PERF_BASELINE_RYZEN.md), entao quem manda
# no ritmo do canvas e a cadencia, nao o modelo — e o laco
# principal nao paga nenhum dos dois.
_PAGE_CLEAN_INTERVAL_SECONDS = 0.25


# Teto da espera pela trabalhadora de tinta na saida do app.
_PAGE_CLEAN_STOP_TIMEOUT_SECONDS = 5.0


# Virada (U3): canvas com pelo menos _INK_TURN_MIN_CANVAS_PX e a
# mascara da passada cobrindo <= _INK_TURN_MAX_OVERLAP dele por
# _INK_TURN_PASSES passadas seguidas = pagina trocada (conteudo
# descorrelacionado). Apagar um trecho a mao nao derruba a
# cobertura da pagina inteira a <=30%: o apagado continua no canvas
# e o resto do desenho segue coberto pela mascara.
_INK_TURN_MIN_CANVAS_PX = 2000


_INK_TURN_MAX_OVERLAP = 0.30


_INK_TURN_PASSES = 2


# HOTFIX 2026-11-01 (draw-lab/page-turn-falsefire-2026-11-01): o
# gatilho de cobertura acima foi REPROVADO no teste ao vivo. Na
# mesa real, o ajuste da folha do caderno aberto oscila (o proprio
# codigo registra oscilacao de ate 314 px nos cantos): o warp
# balanca, a saida do modelo desencontra do canvas acumulado e a
# cobertura desaba SEM pagina nenhuma virar — o evento disparava
# a cada 2-5 s, salvando snapshot e zerando o canvas sem parar.
# A prova offline usou warps estaticos com tremor de +-1-3 px,
# mansos demais pra expor isso; o teste negativo era borracha,
# nao balanco de geometria. Decisao: o disparo automatico por
# cobertura fica DESLIGADO por padrao, atras da flag
# DRAW_PAGE_TURN_EVENT=1, ate existir um gatilho provado contra
# serie de balanco de geometria realista. O codigo do evento e o
# snapshot ficam intactos (versao estrita futura); com a flag
# desligada, nenhum colapso de cobertura zera o canvas — a troca
# de pagina segue pelo caminho pre-existente de drop/reaparicao
# da folha, que abre geracao nova do mesmo jeito.
_ENV_TURN_EVENT_VAR = "DRAW_PAGE_TURN_EVENT"


def _page_turn_event_enabled():
    """True so com DRAW_PAGE_TURN_EVENT=1 (ver o hotfix acima)."""
    return os.environ.get(_ENV_TURN_EVENT_VAR, "").strip() == "1"


# Confirmacao entre passadas (regras medidas no diagnostico de
# 2026-11-01, draw-lab/stall-diagnosis-2026-11-01): sobre a soma
# das probabilidades das ultimas _INK_VOTE_WINDOW passadas, um
# pixel confirma quando (a) a soma chega a _INK_VOTE_SCORE_STRONG
# (tinta forte confirma na 2a passada) ou (b) ele aparece com
# prob >= _INK_VOTE_HIT nas 3 passadas da janela e a soma chega a
# _INK_VOTE_SCORE (o lapis fraco estavel confirma na 3a). Copia
# deslocada de tremor e FP de mobiliario aparecem de forma
# intermitente por posicao e nao fecham nem (a) nem (b).
_INK_VOTE_WINDOW = 3


_INK_VOTE_SCORE = 1.4


_INK_VOTE_SCORE_STRONG = 1.9


_INK_VOTE_HIT = 0.5


_INK_VOTE_MIN_HITS = 3


# Porta de fronteira da confirmacao (medida na regua de replay,
# draw-lab/replay-ruler-2026-11-01, modo hold sobre o codigo sem
# ela): com a pagina desenhada parada, o canvas confirmado
# crescia em degraus a cada alinhamento de fase entre o ciclo do
# warp e a janela do voto — um salto de +9.999 px na passada 89
# e 6.144 px confirmados que passada nenhuma viu acima do
# limiar. O mecanismo: o warp treme, copias deslocadas do mesmo
# traco somam probabilidade SUB-limiar (0,5-0,7) no ombro do
# traco e o voto fecha sobre a soma, camada sobre camada, sem
# nenhuma passada ter chamado aquilo de tinta.
# A porta: pixel dentro da faixa do ombro — dilatacao de
# _INK_FRONTIER_RADIUS px sobre (tinta confirmada ∪ tinta que
# a janela viu acima do limiar) — so confirma se alguma
# passada da janela o viu acima do limiar: evidencia nova
# SUBSTITUI a posicao da antiga, nao soma um ombro novo ao
# redor dela. O raio e o ombro medido nos mapas reais do
# cache da regua: a banda 0,5-0,7 mora a p50 = 1 px e
# p90 ≈ 3 px do nucleo > 0,7. Longe do ombro o voto segue
# intacto: e ele que descobre o lapis fraco consistente
# (0,55 x 3) em regiao nova. Varredura offline sobre os warps
# f28-f30 em ciclo, 300 passadas: fora-da-referencia cai de
# 8.567 px (sem porta) para 419 px (raio 3 ancorado), e o
# crescimento total satura no teto da uniao das passadas.
_INK_FRONTIER_RADIUS = 3


_INK_FRONTIER_KERNEL = np.ones(
    (2 * _INK_FRONTIER_RADIUS + 1, 2 * _INK_FRONTIER_RADIUS + 1),
    dtype=np.uint8,
)


# Registro da evidencia na geometria ancora da geracao (medido
# na regua desta frente, 2026-11-05, draw-lab/parallel-2026-11-05/
# repassada): o acumulador vive numa grade fixa da pagina, mas a
# geometria que retifica cada passada anda — a cadeia rastreador
# + mediana do app sustenta deslocamentos de varios px por
# dezenas de passadas quando um vies persistente (a mao sobre um
# canto) puxa o quad, e ate o tremor de +-1 px entre passadas
# basta. Sem registro, a evidencia entra na grade na posicao em
# que AQUELE quad a projetou: a janela do voto e de 0,75 s e nao
# distingue "tinta nova" de "a mesma tinta re-projetada 2 px ao
# lado" — cada alinhamento confirma a sua copia e a confirmada
# degenera para a uniao de todas as alinhagens, a tinta assentada
# reimpressa camada sobre camada. Na regua (pagina real assentada
# do cache de 2026-11-01, 300 passadas, nenhum conteudo novo):
# geometria congelada confirma 20.137 px e para; tremor de +-1 px
# confirma 45.528 px; deriva lenta de 3 px, 42.930 px; vies de
# canto sustentado, 71.548 px — contra unioes de 47.472/42.711/
# 76.302 px das mascaras das alinhagens: o voto, sob geometria
# movel, e a uniao antiga com outro nome.
# O registro: a primeira passada da geracao fixa a geometria
# ancora; o mapa de cada passada seguinte e re-projetado para a
# grade da ancora (M = A @ C^-1, homografias camera->pagina da
# ancora e da passada) antes do voto. Mover o caderno vira so
# re-registro: o conteudo confirmado nao anda nem duplica
# (registrar uma vez, adaptar por substituicao). O custo e um
# warpPerspective por passada sobre o mapa float, ~ms contra os
# ~168 ms da passada do modelo; geometria igual a da ancora (o
# caso assentado, com o quad congelado) pula o warp.
_REGISTRATION_IDENTITY_TOL = 1e-4


# Ao retomar apos um drop do tracker, a primeira mascara e registrada
# na grade antiga e precisa reencontrar a maior parte da tinta existente.
# Evidencia nova pode acrescentar tracos, mas uma pagina diferente nao
# deve herdar o canvas anterior.
_PAGE_IDENTITY_MIN_RECALL = 0.65
# Guarda dual da identidade: alem do recall da tinta antiga, a
# tinta da passada tem de cair sobre o canvas anterior (mesma
# tolerancia dilatada). O limiar e menor que o do recall porque a
# tinta nova desenhada durante o drop e legitima: medido na regua
# (2026-11-06), mesma pagina com +75% de tinta nova da
# containment 0,65 e perderia o canvas; com 0,5 aceita ~1,8x de
# crescimento. Pagina diferente mede 0,23-0,25 (sintetica), margem
# de 2x contra falso positivo.
_PAGE_IDENTITY_MIN_CONTAINMENT = 0.5
# O canvas confirmado inclui o halo da fronteira, que uma passada
# sozinha nunca cobre no limiar do voto: a comparacao dilata os
# dois lados pela mesma tolerancia do comparador da regua.
_IDENTITY_DILATION_KERNEL = 2 * _INK_FRONTIER_RADIUS + 1


# Oclusao na decisao de identidade: o recall so julga a tinta
# anterior que a passada consegue VER (a mao sobre a pagina nao e
# evidencia de pagina nova). Abaixo deste minimo de tinta anterior
# visivel, a decisao e ADIADA para a proxima passada em vez de
# sair no chute com a pagina coberta.
_IDENTITY_MIN_VISIBLE_FRACTION = 0.25
_IDENTITY_MIN_VISIBLE_PX = 32

# Julgamento de virada sob oclusao: a cobertura e medida sobre a
# tinta confirmada VISIVEL; com menos que isto de tinta visivel, a
# passada se abstem — nem avanca nem zera a sequencia de virada.
_OCCLUSION_TURN_MIN_VISIBLE_PX = 200


def _corners_page_matrix(corners, width, height):
    """Homografia camera->pagina de um quad, na grade do mapa.

    Mesma convencao de draw.paper.perspective_matrix (os cantos
    na ordem do rastreador, casando posicionalmente com o destino
    (0,0), (W,0), (W,H), (0,H)), com o destino no tamanho do
    mapa da passada: e nessa grade que ancora e passada
    precisam conversar.
    """
    destination = np.array(
        [(0, 0), (width, 0), (width, height), (0, height)],
        dtype=np.float32,
    )
    source = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(source, destination)


def _registration_matrix(anchor_corners, corners, shape):
    """M que leva o mapa desta passada para a grade da ancora.

    Um ponto fisico da pagina tem UMA coordenada na ancora:
    p_ancora = A @ C^-1 @ p_passada, com A e C as homografias
    camera->pagina da ancora e desta passada.
    """
    height, width = shape
    anchor = _corners_page_matrix(anchor_corners, width, height)
    current = _corners_page_matrix(corners, width, height)
    return anchor @ np.linalg.inv(current)


def _registration_is_identity(matrix):
    """True se a matriz nao move pixel nenhum (tol sub-pixel)."""
    return bool(
        np.max(np.abs(matrix - np.eye(3, dtype=np.float64)))
        < _REGISTRATION_IDENTITY_TOL
    )


def _refine_ink_ecc(reference, observed, hand_mask=None):
    """Refina somente o residuo na grade da ancora; devolve mapa, H e custo."""
    started = time.perf_counter()
    identity = np.eye(3, dtype=np.float64)
    info = {"ecc_rejected": False, "ecc_mode": "bypass"}

    def finish(aligned, matrix):
        info["ecc_ms"] = (time.perf_counter() - started) * 1000.0
        return aligned, matrix, info

    if reference is None or reference.shape != observed.shape:
        return finish(observed, identity)
    ref = np.asarray(reference, dtype=np.float32)
    obs = np.asarray(observed, dtype=np.float32)
    if hand_mask is not None:
        ref = ref.copy()
        obs = obs.copy()
        ref[hand_mask > 0] = 0.0
        obs[hand_mask > 0] = 0.0
    if np.array_equal(ref, obs):
        return finish(obs, identity)
    # Massa equivalente a 64 pixels e suporte de 64 pixels: apenas
    # seguranca numerica do otimizador, nao um limiar de identidade.
    if (not np.isfinite(ref).all() or not np.isfinite(obs).all()
            or float(ref.sum()) < 64.0
            or np.count_nonzero(ref > 0.1) < 64):
        return finish(obs, identity)
    height, width = obs.shape
    half_size = (width // 2, height // 2)
    if min(half_size) < 2:
        return finish(obs, identity)
    ref_half = cv2.resize(ref, half_size, interpolation=cv2.INTER_AREA)
    obs_half = cv2.resize(obs, half_size, interpolation=cv2.INTER_AREA)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    info["ecc_mode"] = "homography"
    try:
        _, estimated = cv2.findTransformECC(
            ref_half, obs_half, np.eye(3, dtype=np.float32),
            cv2.MOTION_HOMOGRAPHY, criteria, None, 5,
        )
    except cv2.error:
        info["ecc_mode"] = "affine"
        try:
            _, affine = cv2.findTransformECC(
                ref_half, obs_half, np.eye(2, 3, dtype=np.float32),
                cv2.MOTION_AFFINE, criteria, None, 5,
            )
            estimated = np.vstack((affine, [0.0, 0.0, 1.0]))
        except cv2.error:
            info["ecc_rejected"] = True
            return finish(obs, identity)
    scale = np.diag([half_size[0] / width, half_size[1] / height, 1.0])
    try:
        full = np.linalg.inv(scale) @ estimated @ scale
        align = np.linalg.inv(full)
        if not np.isfinite(align).all() or abs(align[2, 2]) < 1e-12:
            raise ValueError("homografia nao normalizavel")
        align /= align[2, 2]
        corners = np.array(
            [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
            dtype=np.float64,
        )
        projected = corners @ align.T
        if (not np.isfinite(align).all() or np.linalg.det(align) <= 1e-6
                or np.any(projected[:, 2] <= 1e-6)):
            raise ValueError("homografia degenerada")
        projected = projected[:, :2] / projected[:, 2:]
        # A margem e medida na grade titular 850x1200, inclusive
        # quando ferramentas/testes usam uma pagina reduzida.
        displacement = np.linalg.norm(
            (projected - corners[:, :2]) * [850.0 / width, 1200.0 / height],
            axis=1,
        )
        if not np.isfinite(displacement).all() or displacement.max() > 55.0:
            raise ValueError("residuo fora da margem")
    except (np.linalg.LinAlgError, ValueError):
        info["ecc_rejected"] = True
        return finish(obs, identity)
    aligned = cv2.warpPerspective(
        obs, align, (width, height), flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )
    return finish(aligned, align)


def _register_ink_probability(prob, anchor, corners, reference, visible=None):
    """Registro existente na ancora, seguido do refinamento residual ECC."""
    info = {"ecc_ms": 0.0, "ecc_rejected": False, "ecc_mode": "bypass"}
    if visible is not None:
        prob = prob.copy()
        prob[~visible] = 0.0
    if anchor is None or corners is None:
        return prob, visible, info
    matrix = _registration_matrix(anchor, corners, prob.shape)
    if not _registration_is_identity(matrix):
        size = (prob.shape[1], prob.shape[0])
        prob = cv2.warpPerspective(
            prob, matrix, size, flags=cv2.INTER_LINEAR, borderValue=0.0,
        )
        if visible is not None:
            visible = cv2.warpPerspective(
                visible.astype(np.uint8), matrix, size,
                flags=cv2.INTER_NEAREST, borderValue=0,
            ) > 0
    hand_mask = None if visible is None else ~visible
    prob, align, info = _refine_ink_ecc(reference, prob, hand_mask)
    if visible is not None and not _registration_is_identity(align):
        visible = cv2.warpPerspective(
            visible.astype(np.uint8), align, (prob.shape[1], prob.shape[0]),
            flags=cv2.INTER_NEAREST, borderValue=0,
        ) > 0
    return prob, visible, info


class _LayeredInkAccumulator:
    """Tinta de uma geracao em duas camadas: provisoria e confirmada.

    Cada passada entrega o mapa de probabilidade do modelo. A
    camada provisoria e o que as passadas da janela viram acima do
    limiar e ainda nao confirmou: ela existe para a tela mostrar a
    tinta nova na primeira passada, e so para a tela. A camada
    confirmada e o canvas: um pixel so entra nela pelas regras de
    voto acima com a porta de fronteira (evidencia sub-limiar nao
    expande a borda da tinta ja confirmada), e dela nao sai
    dentro da geracao.

    Pixel provisorio que sai da janela sem confirmar e retratado:
    some da camada provisoria e entra na contagem de retratados.
    E essa contagem que mede o risco declarado da publicacao
    especulativa, o page-clean pegar mobiliario: a regua de replay
    le quanta tinta provisoria some depois.

    So a trabalhadora de tinta toca nesta classe.
    """

    def __init__(self, threshold=0.7):
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self._window = deque(maxlen=_INK_VOTE_WINDOW)
        self._confirmed = None
        self._provisional = None
        self._provisional_ever = None
        self._retracted_ever = None
        self._low_overlap_streak = 0
        self._turn_fired = False
        self._score_buf = None
        self._hits_buf = None
        self._seen_buf = None

    def _window_stats(self):
        """(soma, hits, visto-como-tinta) da janela, sem empilhar.

        O caminho antigo fazia np.stack da janela (3 mapas float32
        de pagina inteira, ~12 MB) e somava/contava/maximizava
        sobre a pilha a cada passada — o maior transitorio do
        acumulador (pesquisa de performance de 2026-11-03, T10).
        Aqui as tres reducoes saem por acumulacao em buffers
        reusados: a soma adiciona os mesmos mapas na mesma ordem
        (resultado float identico), hits e visto sao contagens
        inteiras/booleanas exatas. Os buffers sao devolvidos para
        leitura imediata; a proxima chamada os sobrescreve.
        """
        window = list(self._window)
        shape = window[0].shape
        if self._score_buf is None or self._score_buf.shape != shape:
            self._score_buf = np.zeros(shape, dtype=np.float32)
            self._hits_buf = np.zeros(shape, dtype=np.int32)
            self._seen_buf = np.zeros(shape, dtype=bool)
        score = self._score_buf
        np.copyto(score, window[0])
        for item in window[1:]:
            np.add(score, item, out=score)
        hits = self._hits_buf
        hits[...] = window[0] >= _INK_VOTE_HIT
        for item in window[1:]:
            np.add(hits, item >= _INK_VOTE_HIT, out=hits)
        seen = self._seen_buf
        seen[...] = window[0] > self.threshold
        for item in window[1:]:
            np.logical_or(seen, item > self.threshold, out=seen)
        return score, hits, seen

    def add(self, prob, visible=None):
        prob = np.asarray(prob, dtype=np.float32)
        if visible is not None:
            visible = np.asarray(visible, dtype=bool)
            if visible.shape != prob.shape:
                visible = None
        if self._confirmed is None or self._confirmed.shape != prob.shape:
            self._confirmed = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional = np.zeros(prob.shape, dtype=np.uint8)
            self._provisional_ever = np.zeros(prob.shape, dtype=bool)
            self._retracted_ever = np.zeros(prob.shape, dtype=bool)
            self._window.clear()
            self._low_overlap_streak = 0
            self._turn_fired = False
        confirmed_before = self._confirmed > 0
        canvas_before = int(np.count_nonzero(self._confirmed))
        # Oclusao: pixel que a passada nao pode ver (a mao sobre a
        # pagina) nao entrega evidencia, nem a favor nem contra. A
        # probabilidade dele entra zerada na janela — soma zero no
        # escore, nao conta hit nem "visto como tinta" —, o
        # julgamento de virada mede a cobertura so sobre a tinta
        # confirmada visivel (ou se abstem quando quase tudo esta
        # coberto: mao segurando a pagina nao e pagina trocada) e a
        # camada provisoria congela onde nao ha visao (ver abaixo).
        evidence = (
            prob
            if visible is None
            else np.where(visible, prob, 0.0).astype(np.float32)
        )
        mask_now = evidence > self.threshold
        turn = False
        if (
            canvas_before >= _INK_TURN_MIN_CANVAS_PX
            and not self._turn_fired
        ):
            if visible is None:
                judge_turn = True
                turn_low = (
                    int(np.count_nonzero(mask_now & confirmed_before))
                    <= _INK_TURN_MAX_OVERLAP * canvas_before
                )
            else:
                visible_confirmed = confirmed_before & visible
                visible_px = int(np.count_nonzero(visible_confirmed))
                judge_turn = (
                    visible_px >= _OCCLUSION_TURN_MIN_VISIBLE_PX
                    and visible_px >= 0.5 * canvas_before
                )
                turn_low = judge_turn and (
                    int(np.count_nonzero(mask_now & visible_confirmed))
                    <= _INK_TURN_MAX_OVERLAP * visible_px
                )
            if judge_turn:
                if turn_low:
                    self._low_overlap_streak += 1
                    if self._low_overlap_streak >= _INK_TURN_PASSES:
                        turn = True
                        self._turn_fired = True
                else:
                    self._low_overlap_streak = 0
        self._window.append(evidence.copy())
        score, hits, seen_as_ink = self._window_stats()
        confirmed_rule = (score >= _INK_VOTE_SCORE_STRONG) | (
            (hits >= _INK_VOTE_MIN_HITS) & (score >= _INK_VOTE_SCORE)
        )
        if np.any(confirmed_rule):
            # Porta de fronteira (ver _INK_FRONTIER_RADIUS):
            # dentro da faixa do ombro da tinta (confirmada ou
            # vista acima do limiar nesta janela), so confirma
            # quem alguma passada viu como tinta; a soma
            # sub-limiar das copias deslocadas nao engrossa o
            # traco nem poe camada nova sobre ele.
            anchor = seen_as_ink | confirmed_before
            near_anchor = (
                cv2.dilate(
                    anchor.astype(np.uint8),
                    _INK_FRONTIER_KERNEL,
                )
                > 0
            )
            confirmed_rule &= seen_as_ink | ~near_anchor
        new_confirmed = confirmed_rule & ~confirmed_before
        new_confirmed_px = int(np.count_nonzero(new_confirmed))
        if new_confirmed_px:
            self._confirmed[new_confirmed] = 255
        confirmed_now = self._confirmed > 0
        provisional_now = seen_as_ink & ~confirmed_now
        previous_provisional = self._provisional > 0
        if visible is not None:
            # Sem visao nao ha retracao nem estreia: o provisorio
            # coberto fica como estava. Retrair e afirmar que a
            # tinta sumiu — e quem esta sobre ela e a mao, nao o
            # apagador; sem este congelamento, toda oclusao virava
            # retracao na regua e a tinta piscava ao reaparecer.
            provisional_now = provisional_now | (
                previous_provisional & ~visible
            )
        retracted = (
            previous_provisional & ~provisional_now & ~confirmed_now
        )
        if np.any(retracted):
            self._retracted_ever |= retracted
        self._provisional_ever |= provisional_now
        provisional_changed = bool(
            np.any(provisional_now != previous_provisional)
        )
        self._provisional = np.where(
            provisional_now, 255, 0
        ).astype(np.uint8)
        return {
            "confirmed": self._confirmed.copy(),
            "provisional": self._provisional.copy(),
            "confirmed_changed": new_confirmed_px > 0,
            "provisional_changed": provisional_changed,
            "new_confirmed_px": new_confirmed_px,
            "confirmed_px": int(np.count_nonzero(self._confirmed)),
            "provisional_px": int(np.count_nonzero(self._provisional)),
            "provisional_ever_px": int(
                np.count_nonzero(self._provisional_ever)
            ),
            "retracted_px_total": int(
                np.count_nonzero(self._retracted_ever)
            ),
            "turn": turn,
        }


class _PageCleanWorker:
    """Tinta pagina->limpa em thread propria, latest-wins.

    O laco principal entrega o warp mais recente com submit() e
    segue: a fila tem tamanho 1 e um warp nao processado e
    substituido pelo proximo, nunca acumulado. A trabalhadora roda
    passadas completas do modelo com pelo menos
    _PAGE_CLEAN_INTERVAL_SECONDS entre inicios — e nenhuma
    passada em descanso total (folha assentada, saida confirmada
    quieta e mao fora ou parada; _at_rest) — e alimenta o
    _LayeredInkAccumulator da geracao: a tinta nova sai na hora
    como camada provisoria para a exibicao (poll_display) e so
    vira canvas pelos votos de confirmacao (poll). Publicar cedo
    sem voto era a uniao antiga, que integrava o tremor e o
    mobiliario para sempre; confirmar antes de mostrar era o
    atraso de segundos da cadeia assentada. As duas camadas
    separam as duas perguntas.

    Geracao e o contrato de sessao da folha: folha nova ou folha
    que sai chama new_generation(), que descarta pendente e
    publicado da geracao anterior; a uniao da trabalhadora zera na
    primeira passada da geracao nova. Resultado de geracao velha
    nunca entra no canvas novo, mesmo com uma passada em voo na
    troca. Dentro da geracao, a evidencia de cada passada e
    registrada na geometria da primeira (a ancora) antes do voto:
    o quad que retifica anda entre passadas, e sem o registro cada
    alinhamento confirmava a sua copia da mesma tinta — a
    reimpressao medida na regua de 2026-11-05 (ver
    _REGISTRATION_IDENTITY_TOL). Blocos sujos por diff de pixels NAO entram aqui: medido
    nesta webcam, o churn entre quadros consecutivos parados
    (media de |diff| por bloco ~1,5-2,0 niveis; maximo pos-blur
    4-6) empata ou supera o sinal de um traco novo no bloco
    (lapis: maximo pos-blur 6-7), entao pular bloco por diff
    arriscaria engolir justamente o lapis. A economia vem da
    assincronia e da cadencia, nao de pular bloco.
    """

    def __init__(self, page_clean, hand_masker=None):
        self._page_clean = page_clean
        self._hand_masker = hand_masker
        self._ecc_reference = None
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._work, name="draw-page-clean", daemon=True
        )
        self._pending = None
        self._generation = 0
        self._published = None
        self._version = 0
        self._stopped = False
        self._error = None
        # Estado privado da trabalhadora (so ela toca):
        self._worker_generation = -1
        # Geometria ancora da geracao: os cantos da primeira
        # passada com cantos. Toda evidencia posterior e
        # registrada nela antes do voto (ver _registration_matrix).
        # None enquanto a geracao nao teve passada com cantos —
        # chamadores que nao informam cantos (ferramentas, testes
        # antigos) seguem com a evidencia na propria geometria.
        self._anchor_corners = None
        self._identity_reference = None
        self._identity_reference_corners = None
        self._identity_recovery_requested = False
        self._accumulator = _LayeredInkAccumulator(
            threshold=getattr(page_clean, "threshold", 0.7)
        )
        self._last_pass_start = None
        self._passes_total = 0
        self._last_pass_ms = None
        self._provisional_px = 0
        self._confirmed_px = 0
        self._retracted_px_total = 0
        self._published_display = None
        self._display_version = 0
        self._hand_state = None
        self._confirm_once = False
        self._still_passes = 0
        # Descanso total (so a trabalhadora e o new_generation
        # tocam): passadas desta geracao e passadas seguidas sem
        # confirmar nada novo.
        self._generation_passes = 0
        self._quiet_passes = 0
        # Relogio de quietude da mao (carimbo monotonico do
        # ultimo movimento/aparicao reportado): o descanso com a
        # mao presente exige a mao parada alem de
        # _HAND_REST_STILL_SECONDS. None = nenhum movimento
        # reportado ainda — nao conta como parada.
        self._hand_motion_at = None
        # Telemetria por etapa (U0 do loop v2): contadores e os
        # carimbos das ultimas passadas, para medir onde o tempo
        # submit->publicacao mora sem mudar comportamento nenhum.
        # Quem le usa telemetry_snapshot(); quem escreve e o submit
        # (laco) e a trabalhadora, sempre sob a Condition.
        self._telemetry_submits = 0
        self._telemetry_replaced = 0
        self._telemetry_passes = 0
        self._telemetry_publishes = 0
        self._telemetry_records = deque(maxlen=256)
        self._identity_events = deque(maxlen=128)
        # Evento de virada (desligado por padrao): pendente
        # (geracao, canvas confirmado anterior).
        self._turn_event = None

    def start(self):
        self._thread.start()
        return self

    def new_generation(self, preserve_identity=False):
        """Abre geracao; opcionalmente identifica a pagina anterior.

        preserve_identity guarda uma copia do canvas confirmado para
        que a primeira passada possa decidir, por conteudo registrado,
        se uma folha que voltou e a mesma pagina em outra pose.
        """
        with self._condition:
            self._identity_recovery_requested = bool(preserve_identity)
            if preserve_identity:
                reference = self._accumulator._confirmed
                self._identity_reference = (
                    None if reference is None else reference.copy()
                )
                self._identity_reference_corners = (
                    None if self._anchor_corners is None
                    else self._anchor_corners.copy()
                )
            else:
                self._identity_reference = None
                self._identity_reference_corners = None
            self._generation += 1
            self._still_passes = 0
            self._generation_passes = 0
            self._quiet_passes = 0
            self._pending = None
            self._published = None
            self._published_display = None
            self._turn_event = None
            self._condition.notify_all()
            return self._generation

    def submit(self, warped, generation, corners=None, visible=None):
        """Entrega o warp mais recente (latest-wins).

        corners e o quad que retificou ESTE warp (os mesmos
        cantos que o chamador passou ao warp_sheet): a
        trabalhadora registra a evidencia da passada na
        geometria ancora da geracao antes do voto. Cantos fora
        do formato (4, 2) sao tratados como ausentes — um quad
        malformado nao pode derrubar a thread da tinta; sem
        cantos, a passada entra na propria geometria.

        visible e a mascara de visibilidade DESTE warp (True onde
        a camera ve a pagina; False sob a mao), na mesma grade do
        warp. Onde ela e False, a passada se abstem: a evidencia
        nao vota, a virada nao julga e a decisao de identidade so
        olha a tinta anterior visivel. Formato divergente do warp
        e tratado como ausente, pela mesma razao dos cantos.
        """
        if corners is not None:
            try:
                corners = np.asarray(corners, dtype=np.float32)
            except (TypeError, ValueError):
                corners = None
            if corners is not None and corners.shape != (4, 2):
                corners = None
        if visible is not None:
            try:
                visible = np.asarray(visible, dtype=bool)
            except (TypeError, ValueError):
                visible = None
            if visible is not None and visible.shape != warped.shape[:2]:
                visible = None
        with self._condition:
            if self._stopped or generation != self._generation:
                return
            if self._pending is not None:
                # Latest-wins: um warp pendente substituido nunca
                # vira passada; a telemetria conta a substituicao
                # porque ela e atraso real de atualizacao.
                self._telemetry_replaced += 1
            self._telemetry_submits += 1
            self._pending = (
                warped, generation, time.monotonic(), corners, visible
            )
            self._condition.notify_all()

    def set_hand_presence(self, present, moved=False):
        """Alimenta a porta de ociosidade: True, False ou None.

        A transicao para False arma exatamente uma passada de
        confirmacao: a mao saiu, a pagina assentou, e essa passada
        fecha o voto do que foi escrito. Depois dela, sem mao, a
        trabalhadora sonda a no maximo 1 Hz enquanto a saida
        confirmada ainda muda — e entra em descanso total (zero
        passadas) quando ela para de mudar; ver _at_rest.

        moved=True e o sinal de atividade da mascara (ver
        _MediaPipeHandPresence.sample): reseta o relogio de
        quietude da mao e acorda o descanso com a mao presente.
        Transicao de presenca tambem conta como movimento: a mao
        chegando ou saindo nunca e mao parada.
        """
        with self._condition:
            if present is False and self._hand_state is not False:
                self._confirm_once = True
            if present is not None:
                self._still_passes = 0
            if present is not None and (
                moved or present != self._hand_state
            ):
                self._hand_motion_at = time.monotonic()
            self._hand_state = present
            self._condition.notify_all()

    def _hand_still(self):
        """Mao presente e sem movimento alem da janela de quietude."""
        return (
            self._hand_motion_at is not None
            and time.monotonic() - self._hand_motion_at
            >= _HAND_REST_STILL_SECONDS
        )

    def _at_rest(self):
        """Descanso total: nenhuma passada ate um evento acordar.

        Sem mao: sinal presente e False, passada de confirmacao
        da saida da mao ja gasta, janela de voto da geracao
        completa e as ultimas passadas sem confirmar nada novo.
        Com mao (decisao do Lucas, 2026-11-03): a mesma quietude
        da saida confirmada, mais a mao parada alem de
        _HAND_REST_STILL_SECONDS — acorda com o movimento da
        propria mao, reportado pela assinatura da mascara.
        Ver _REST_MIN_GENERATION_PASSES/_REST_QUIET_PASSES.
        """
        if (
            self._generation_passes < _REST_MIN_GENERATION_PASSES
            or self._quiet_passes < _REST_QUIET_PASSES
        ):
            return False
        if self._hand_state is False:
            return not self._confirm_once
        if self._hand_state is True:
            return self._hand_still()
        return False

    def _pass_interval(self):
        if self._at_rest():
            return None
        if self._hand_state is False and not self._confirm_once:
            return _IDLE_PROBE_SECONDS
        if (
            self._hand_state is None
            and self._still_passes >= _NO_HAND_REST_AFTER_STILL_PASSES
        ):
            return _IDLE_PROBE_SECONDS
        return _PAGE_CLEAN_INTERVAL_SECONDS

    def telemetry_snapshot(self):
        """Copia da telemetria por etapa (somente leitura).

        Contadores desde a criacao da trabalhadora e, por passada
        recente, os carimbos monotonicos e as duracoes derivadas:
        queue_wait_s (submit->inicio da passada; inclui a espera
        da cadencia de _PAGE_CLEAN_INTERVAL_SECONDS), pass_s
        (inicio->fim do modelo) e submit_to_publish_s (submit->
        fim da passada que decide a publicacao). Na reconciliacao
        com a orquestracao em camadas, passes conta cada passada
        do modelo e publishes conta so as passadas que publicam
        canvas confirmado novo no poll(). Eventos de identidade de
        pagina ficam numa lista separada para sobreviverem ao buffer
        curto de passadas. Nenhum campo muda comportamento.
        """
        with self._condition:
            return {
                "submits": self._telemetry_submits,
                "submits_replaced": self._telemetry_replaced,
                "passes": self._telemetry_passes,
                "publishes": self._telemetry_publishes,
                "records": [dict(r) for r in self._telemetry_records],
                "page_identity_events": [
                    dict(event) for event in self._identity_events
                ],
            }

    def poll(self, generation, after_version):
        """(versao, copia do canvas confirmado) se ha publicado
        novo da geracao depois de after_version; senao None.

        Este e o canal dos consumidores de canvas (snapshot,
        cursor, Paint, ponte): so tinta confirmada passa por aqui.
        A camada provisoria nunca sai por este poll.
        """
        with self._condition:
            published = self._published
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            return published[1], published[2].copy()

    def poll_display(self, generation, after_version):
        """(versao, confirmado, provisorio) para a tela, ou None.

        As camadas vao separadas para a exibicao pintar a
        provisoria em estilo distinto. Este canal e so de
        exibicao: quem consome canvas usa poll().
        """
        with self._condition:
            published = self._published_display
            if (
                published is None
                or published[0] != generation
                or published[1] <= after_version
            ):
                return None
            return (
                published[1],
                published[2].copy(),
                published[3].copy(),
            )

    def take_turn_event(self, generation):
        """Canvas anterior se a virada (U3) disparou nesta geracao.

        Devolve a copia do canvas da pagina que acabou e limpa o
        evento; None se nao ha evento pendente pra esta geracao.
        Quem chama (o laco) salva o snapshot e abre geracao nova.
        So dispara com DRAW_PAGE_TURN_EVENT=1 (ver o portao acima).
        """
        with self._condition:
            if (
                self._turn_event is None
                or self._turn_event[0] != generation
            ):
                return None
            _gen, canvas = self._turn_event
            self._turn_event = None
            return canvas

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._stopped
                )
                if self._stopped:
                    return
                # Cadencia: o intervalo corre com o pendente na
                # mao; um submit durante a espera o substitui
                # (latest-wins) e e ele que roda ao vencer. Se a
                # geracao trocou, new_generation limpou o pendente
                # e esta volta nao roda nada.
                while self._last_pass_start is not None:
                    interval = self._pass_interval()
                    if interval is None:
                        # Descanso total: espera sem prazo por um
                        # evento que acorde (a mao voltar, a mao
                        # presente se mexer, geracao nova, stop).
                        # Submit NAO acorda: ele so troca o
                        # pendente, que e reavaliado quando um
                        # evento de verdade chegar. Sem isto a
                        # sonda de 1 Hz rodava para sempre sobre a
                        # pagina parada e o voto acumulava camadas.
                        self._condition.wait()
                        if self._stopped:
                            return
                        if self._pending is None:
                            break
                        continue
                    remaining = interval - (
                        time.monotonic() - self._last_pass_start
                    )
                    if remaining <= 0 or self._pending is None:
                        break
                    self._condition.wait(timeout=remaining)
                    if self._stopped:
                        return
                if self._pending is None:
                    continue
                warped, generation, submitted_at, corners, visible = (
                    self._pending
                )
                self._pending = None
                self._last_pass_start = time.monotonic()
                pass_started_at = self._last_pass_start
                if self._confirm_once:
                    self._confirm_once = False
            # Uma excecao do modelo (shape errado, sessao morta, ONNX
            # sem memoria) matava esta thread em silencio: o canvas
            # congelava no ultimo quadro e nada era impresso. O erro
            # fica guardado, e o laco principal o reporta.
            pass_started = time.monotonic()
            identity_decision = None
            identity_recall = None
            identity_containment = None
            identity_reference_px = None
            identity_visible_reference_px = None
            identity_deferred = False
            identity_geometry_registered = False
            ecc_info = {"ecc_ms": 0.0, "ecc_rejected": False, "ecc_mode": "bypass"}
            try:
                if visible is None and self._hand_masker is not None:
                    hand_mask = self._hand_masker.mask(warped)
                    if hand_mask is not None:
                        visible = hand_mask == 0
                ink_prob = getattr(self._page_clean, "ink_prob", None)
                if ink_prob is not None:
                    prob = ink_prob(warped)
                else:
                    prob = (
                        self._page_clean.ink_mask(warped) > 0
                    ).astype(np.float32)
                pass_ended_at = time.monotonic()
                with self._condition:
                    if self._stopped:
                        return
                    if generation != self._generation:
                        continue
                    first_pass = generation != self._worker_generation
                    recovering = (
                        self._identity_reference is not None
                        and (first_pass or self._identity_recovery_requested)
                    )
                    anchor = (
                        self._identity_reference_corners if recovering
                        else corners if first_pass else self._anchor_corners
                    )
                    ecc_reference = (
                        self._ecc_reference
                        if ink_prob is not None and (not first_pass or recovering)
                        else None
                    )
                # ECC nao segura o lock que submit/poll/new_generation usam.
                registered_prob, registered_visible, ecc_info = _register_ink_probability(
                    prob, anchor, corners, ecc_reference, visible
                )
            except Exception as error:  # noqa: BLE001 - vai para o laco
                with self._condition:
                    self._error = error
                    self._stopped = True
                    self._condition.notify_all()
                return
            with self._condition:
                if self._stopped:
                    return
                if generation != self._generation:
                    continue
                first_pass = generation != self._worker_generation
                if first_pass:
                    self._worker_generation = generation
                    self._generation_passes = 0
                    self._quiet_passes = 0
                # A decisao de identidade roda na primeira passada e,
                # enquanto estiver adiada por oclusao, nas seguintes
                # — a referencia so e consumida quando a decisao sai.
                identity_pending = (
                    self._identity_recovery_requested
                    and self._identity_reference is not None
                )
                if first_pass or identity_pending:
                    identity_reference = self._identity_reference
                    identity_corners = self._identity_reference_corners
                    identity_recovery_requested = (
                        self._identity_recovery_requested
                    )
                    keep_page = False
                    if identity_reference is not None:
                        candidate = registered_prob
                        identity_visible = registered_visible
                        if identity_corners is not None and corners is not None:
                            identity_geometry_registered = True
                        prior = identity_reference > 0
                        prior_px = int(np.count_nonzero(prior))
                        identity_reference_px = prior_px
                        if prior_px == 0:
                            seen_any = (
                                candidate > self._accumulator.threshold
                            )
                            if identity_visible is not None:
                                seen_any = seen_any & identity_visible
                            keep_page = not np.any(seen_any)
                            identity_recall = 1.0 if keep_page else 0.0
                        else:
                            seen = candidate > self._accumulator.threshold
                            kernel = np.ones(
                                (_IDENTITY_DILATION_KERNEL,) * 2, np.uint8
                            )
                            seen_d = cv2.dilate(
                                seen.astype(np.uint8), kernel
                            ) > 0
                            if identity_visible is not None:
                                prior_visible = prior & identity_visible
                                prior_visible_px = int(
                                    np.count_nonzero(prior_visible)
                                )
                                identity_visible_reference_px = (
                                    prior_visible_px
                                )
                                if prior_visible_px < max(
                                    _IDENTITY_MIN_VISIBLE_PX,
                                    int(
                                        _IDENTITY_MIN_VISIBLE_FRACTION
                                        * prior_px
                                    ),
                                ):
                                    # Tinta anterior quase toda
                                    # coberta: decidir agora seria
                                    # condenar a pagina pela mao.
                                    # Adia; a referencia fica para a
                                    # proxima passada.
                                    identity_deferred = True
                                else:
                                    prior_d = cv2.dilate(
                                        prior_visible.astype(np.uint8),
                                        kernel,
                                    ) > 0
                                    identity_recall = (
                                        np.count_nonzero(
                                            prior_visible & seen_d
                                        )
                                        / prior_visible_px
                                    )
                                    seen_px = int(np.count_nonzero(seen))
                                    identity_containment = (
                                        0.0
                                        if seen_px == 0
                                        else (
                                            np.count_nonzero(seen & prior_d)
                                            / seen_px
                                        )
                                    )
                                    keep_page = (
                                        identity_recall
                                        >= _PAGE_IDENTITY_MIN_RECALL
                                        and identity_containment
                                        >= _PAGE_IDENTITY_MIN_CONTAINMENT
                                    )
                            else:
                                prior_d = cv2.dilate(
                                    prior.astype(np.uint8), kernel
                                ) > 0
                                identity_recall = (
                                    np.count_nonzero(prior & seen_d) / prior_px
                                )
                                seen_px = int(np.count_nonzero(seen))
                                identity_containment = (
                                    0.0
                                    if seen_px == 0
                                    else (
                                        np.count_nonzero(seen & prior_d)
                                        / seen_px
                                    )
                                )
                                keep_page = (
                                    identity_recall >= _PAGE_IDENTITY_MIN_RECALL
                                    and identity_containment
                                    >= _PAGE_IDENTITY_MIN_CONTAINMENT
                                )
                    if identity_recovery_requested:
                        if identity_deferred:
                            identity_decision = "deferred_occlusion"
                        else:
                            identity_decision = (
                                "same_page" if keep_page else "new_page"
                            )
                        self._identity_events.append({
                            "generation": generation,
                            "decision": identity_decision,
                            "recall": identity_recall,
                            "containment": identity_containment,
                            "reference_px": identity_reference_px,
                            "visible_reference_px": (
                                identity_visible_reference_px
                            ),
                            "geometry_registered": (
                                identity_geometry_registered
                            ),
                            "threshold": _PAGE_IDENTITY_MIN_RECALL,
                            "containment_threshold": (
                                _PAGE_IDENTITY_MIN_CONTAINMENT
                            ),
                            **ecc_info,
                        })
                    if identity_deferred:
                        # A referencia NAO e consumida: a proxima
                        # passada decide quando houver tinta
                        # anterior visivel o bastante. A evidencia
                        # desta entra no acumulador com a oclusao
                        # abstida; se a decisao final for pagina
                        # nova, o reset dela a descarta.
                        pass
                    else:
                        if not keep_page:
                            self._accumulator.reset()
                            self._anchor_corners = corners
                            self._ecc_reference = None
                        # Keep the old anchor when identity matched: all future
                        # observations remain registered into the same canvas.
                        self._identity_reference = None
                        self._identity_reference_corners = None
                        self._identity_recovery_requested = False
                if ((first_pass or identity_pending)
                        and not identity_deferred and not keep_page):
                    if visible is not None:
                        prob = np.where(visible, prob, 0.0).astype(np.float32)
                else:
                    prob = registered_prob
                    visible = registered_visible
                if ink_prob is not None:
                    if (self._ecc_reference is None
                            or self._ecc_reference.shape != prob.shape):
                        self._ecc_reference = prob.copy()
                    else:
                        np.maximum(self._ecc_reference, prob, out=self._ecc_reference)
                result = self._accumulator.add(prob, visible)
                if (
                    result["confirmed_changed"]
                    or result["provisional_changed"]
                    or result["turn"]
                ):
                    self._still_passes = 0
                else:
                    self._still_passes += 1
                # Contadores do descanso total: so a saida
                # CONFIRMADA interessa (provisoria tremula com o
                # warp e nao pode segurar a trabalhadora acordada).
                self._generation_passes += 1
                if result["confirmed_changed"] or result["turn"]:
                    self._quiet_passes = 0
                else:
                    self._quiet_passes += 1
                self._passes_total += 1
                self._last_pass_ms = (
                    time.monotonic() - pass_started
                ) * 1000.0
                self._provisional_px = result["provisional_px"]
                self._confirmed_px = result["confirmed_px"]
                self._retracted_px_total = result["retracted_px_total"]
                # Virada: o acumulador detecta a cobertura sobre o
                # canvas confirmado; o evento so e guardado atras
                # de _page_turn_event_enabled(), desligado por
                # padrao (hotfix 2026-11-01).
                if (
                    result["turn"]
                    and _page_turn_event_enabled()
                    and self._turn_event is None
                ):
                    self._turn_event = (
                        generation,
                        result["confirmed"],
                    )
                if first_pass or result["confirmed_changed"]:
                    self._version += 1
                    self._published = (
                        generation,
                        self._version,
                        result["confirmed"],
                    )
                if (
                    first_pass
                    or result["confirmed_changed"]
                    or result["provisional_changed"]
                ):
                    self._display_version += 1
                    self._published_display = (
                        generation,
                        self._display_version,
                        result["confirmed"],
                        result["provisional"],
                    )
                # Telemetria por etapa (loop v2) sobre o caminho em
                # camadas: um registro por passada; publishes conta
                # so as passadas que publicam canvas confirmado
                # novo, e o campo de mascara da passada vira
                # confirmed_px (a uniao nao existe mais).
                published_at = time.monotonic()
                self._telemetry_passes += 1
                if first_pass or result["confirmed_changed"]:
                    self._telemetry_publishes += 1
                self._telemetry_records.append({
                    **ecc_info,
                    "generation": generation,
                    "version": self._version,
                    "page_identity_decision": identity_decision,
                    "page_identity_recall": identity_recall,
                    "page_identity_containment": identity_containment,
                    "page_identity_reference_px": identity_reference_px,
                    "page_identity_visible_reference_px": (
                        identity_visible_reference_px
                    ),
                    "page_identity_geometry_registered": (
                        identity_geometry_registered
                    ),
                    "submitted_at": submitted_at,
                    "pass_started_at": pass_started_at,
                    "pass_ended_at": pass_ended_at,
                    "published_at": published_at,
                    "queue_wait_s": pass_started_at - submitted_at,
                    "pass_s": pass_ended_at - pass_started_at,
                    "submit_to_publish_s": published_at - submitted_at,
                    "mask_px": int(np.count_nonzero(
                        prob > self._accumulator.threshold)),
                    "confirmed_px": result["confirmed_px"],
                })
                self._condition.notify_all()

    def stats(self):
        """Contadores para a telemetria: passadas e custo da ultima."""
        with self._condition:
            return {
                "passes_total": self._passes_total,
                "pass_ms": (
                    None
                    if self._last_pass_ms is None
                    else round(self._last_pass_ms, 1)
                ),
                "canvas_px": self._confirmed_px,
                "provisional_px": self._provisional_px,
                "retracted_px_total": self._retracted_px_total,
                "worker_mode": (
                    "rest"
                    if self._at_rest()
                    else "active"
                    if self._hand_state
                    else "idle"
                    if self._hand_state is False
                    else "unknown"
                ),
                "resting": self._at_rest(),
                "hand_present": self._hand_state,
            }

    @property
    def error(self):
        """A excecao que matou a trabalhadora, ou None."""
        with self._condition:
            return self._error

    def stop(self):
        # O teto cobre uma passada em voo: a trabalhadora so ve o
        # stop entre passadas, e sair com ela dentro do ONNX
        # derruba o processo na finalizacao (visto na bancada
        # desta VM, passada de segundos). No Ryzen a passada e
        # 168 ms e o join volta muito antes do teto.
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=_PAGE_CLEAN_STOP_TIMEOUT_SECONDS)


def _render_layered_canvas(confirmed, provisional, out=None):
    """Canvas de exibicao: confirmada em preto, provisoria em cinza.

    A provisoria e tinta que a primeira passada viu e o voto
    ainda nao confirmou. Ela aparece mais clara para quem desenha
    saber que pode sumir, e nunca entra no canvas que poll()
    entrega aos consumidores.
    """
    canvas = ink_module.render_canvas(confirmed, out=out)
    if provisional is not None:
        light = (provisional > 0) & (confirmed == 0)
        if np.any(light):
            canvas[light] = (176, 176, 176)
    return canvas
