"""Propagacao do quad por Lucas-Kanade resemeado (frente 105).

Forma viavel desenhada pela frente 100 (relatorio 100, secao 7) da
proposta de arquitetura do Lucas: entre updates de deteccao, a
homografia da folha e sustentada por fluxo optico — mas NAO pelos
cantos (o canto superior-direito nao e rastreavel, FM-TR-CORNER) e
NAO por uma nuvem fixa (atrito de ~0,56 de sobrevivencia por passo
esvazia a nuvem em 4-5 passos, FM-ATTRITION). O propagador RESEMEIA
uma grade 6x6 dentro do quad estimado corrente a cada passo: os
pontos sao fungiveis, so morre o passo inteiro (<12 sobreviventes
com FB <= 1,0 px ou <10 inliers do RANSAC), e a morte e binaria e
detectavel no proprio passo — e esse sinal que o laco usa para
redetectar na hora, em vez de esperar o teto periodico.

Politica de updates (a parte que decide, frente 100 secao 5/7):
- teto periodico: no maximo LK_MAX_PROP_STEPS passos propagados sem
  um update de deteccao (a faixa medida viavel e N=3-5);
- redeteccao imediata quando o passo morre (FM-DISTURBANCE-DEATH e
  FM-BLIND-INTERVAL: update so periodico deixava corridas paradas
  inteiras sem quad);
- nos updates, portao anti-flip da frente 97: inovacao (deteccao x
  ultima saida) <= 50 px/canto aceita na hora (reset duro); maior
  que isso, com a propagacao viva, a deteccao fica SEGURADA um
  ciclo e so entra se a deteccao seguinte pousar a <= 25 px dela
  (flip isolado nunca confirma; troca de modo sustentada entra com
  1 update de atraso). Com a propagacao morta, a deteccao manda sem
  portao — e reposicionamento, nao flip.

Fronteiras: o detector continua dono de aquisicao, presenca,
identidade e virada de pagina. A propagacao nunca cria folha nem
decide geracao; sem ancora (nenhuma deteccao aceita), nada se
propaga — o fantasma da frente 85 nao ganha vida propria aqui.
EKF por cima foi medido na frente 100 com ganho ~0 e nao entra.

Tudo atras de DRAW_SHEET_LK_PROPAGATE (default desligado): com a
flag desligada, nenhum codigo deste modulo executa no laco.
"""

import os

import cv2
import numpy as np

_ENV_LK_PROPAGATE_VAR = "DRAW_SHEET_LK_PROPAGATE"

# Parametros do passo LK (frente 100, medidos): janela 21x21, 3
# niveis de piramide, 30 iteracoes; checagem forward-backward com
# limiar 1,0 px — quem limpa e o RANSAC (3,0 px); o FB decide quantos
# pontos restam.
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
_LK_FB_MAX_PX = 1.0
_LK_MIN_SURVIVORS = 12
_LK_MIN_INLIERS = 10
_LK_RANSAC_PX = 3.0

# Teto periodico de passos propagados entre updates (faixa viavel
# medida na frente 100: N=3-5; 4 e o meio da faixa).
LK_MAX_PROP_STEPS = 4

# Portao anti-flip nos updates (frente 97): salto de canto > 50 px
# contra a ultima saida e segurado; confirma se a deteccao seguinte
# ficar a <= 25 px do pendente.
LK_GATE_HOLD_PX = 50.0
LK_GATE_CONFIRM_PX = 25.0


def lk_propagate_enabled():
    """True so com DRAW_SHEET_LK_PROPAGATE=1 (default desligado)."""
    return os.environ.get(_ENV_LK_PROPAGATE_VAR, "0").strip() == "1"


def max_corner_distance(a, b):
    """Maior distancia de canto entre dois quads (px)."""
    return float(
        np.max(
            np.linalg.norm(
                np.asarray(a, dtype=np.float64).reshape(-1, 2)
                - np.asarray(b, dtype=np.float64).reshape(-1, 2),
                axis=1,
            )
        )
    )


def cloud_seed(quad):
    """Grade bilinear 6x6 dentro do quad (margem de 4% nas bordas)."""
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    us = np.linspace(0.04, 0.96, 6)
    pts = []
    for v in us:
        for u in us:
            pts.append(
                (1 - u) * (1 - v) * q[0]
                + u * (1 - v) * q[1]
                + u * v * q[2]
                + (1 - u) * v * q[3]
            )
    return np.array(pts, dtype=np.float32)


def _lk_step_points(prev_gray, curr_gray, pts):
    """Um passo LK com checagem FB. Devolve (novos, mascara, erro_fb)."""
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None, **LK_PARAMS
    )
    if nxt is None or st is None:
        return None, np.zeros(len(pts), dtype=bool), np.full(len(pts), np.nan)
    back, stb, _err_b = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, nxt, None, **LK_PARAMS
    )
    if back is None or stb is None:
        return None, np.zeros(len(pts), dtype=bool), np.full(len(pts), np.nan)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    ok = (st.reshape(-1) == 1) & (stb.reshape(-1) == 1) & (fb <= _LK_FB_MAX_PX)
    return nxt, ok, fb


def _quad_is_plausible(q, anchor_area):
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(q).all():
        return False
    crosses = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        u, v = b - a, c - b
        crosses.append(float(u[0] * v[1] - u[1] * v[0]))
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        return False
    area = abs(cv2.contourArea(q.astype(np.float32)))
    return 0.4 * anchor_area <= area <= 2.5 * anchor_area


class ReseededQuadPropagator:
    """Cadeia de propagacao resemeada a partir de uma ancora.

    anchor(quad) (re)planta a cadeia numa deteccao aceita;
    step(prev_gray, curr_gray) avanca um quadro e devolve o quad
    propagado, ou None quando o passo morre (a cadeia fica morta
    ate o proximo anchor). A homografia total compoe os passos:
    H_total = H_passo @ H_total, e o quad vigente e sempre a
    ancora transformada por H_total — a deriva de um passo
    enviesado nao se re-semeia como verdade, so se compoe.
    """

    def __init__(self):
        self.anchor_quad = None
        self.anchor_area = 0.0
        self.quad = None
        self.alive = False
        self.steps_since_anchor = 0
        self.deaths = 0
        self.last_loss = None
        self.last_inliers = 0

    def reset(self):
        self.anchor_quad = None
        self.anchor_area = 0.0
        self.quad = None
        self.alive = False
        self.steps_since_anchor = 0
        self.last_loss = None
        self.last_inliers = 0

    def anchor(self, quad):
        self.anchor_quad = np.asarray(quad, dtype=np.float64).reshape(-1, 2).copy()
        self.anchor_area = abs(cv2.contourArea(self.anchor_quad.astype(np.float32)))
        self._h_total = np.eye(3)
        self.quad = self.anchor_quad.copy()
        self.alive = True
        self.steps_since_anchor = 0
        self.last_loss = 0.0
        self.last_inliers = 0

    def _die(self):
        self.alive = False
        self.deaths += 1
        return None

    def step(self, prev_gray, curr_gray):
        if not self.alive or prev_gray is None or curr_gray is None:
            return None
        seed = cloud_seed(self.quad)
        pts = seed.reshape(-1, 1, 2)
        nxt, ok, _fb = _lk_step_points(prev_gray, curr_gray, pts)
        if nxt is None:
            return self._die()
        self.last_loss = 1.0 - float(ok.mean())
        # Perda de pontos acima de ~0,6 num passo ainda vivo e o
        # sinal precoce da frente 100 (secao 7.2): o laco trata
        # last_loss como pedido de update, sem matar a cadeia.
        if int(ok.sum()) < _LK_MIN_SURVIVORS:
            return self._die()
        h_step, inl = cv2.findHomography(
            seed[ok].astype(np.float32),
            nxt[ok].reshape(-1, 2).astype(np.float32),
            cv2.RANSAC,
            _LK_RANSAC_PX,
        )
        if h_step is None or inl is None or int(inl.sum()) < _LK_MIN_INLIERS:
            return self._die()
        self.last_inliers = int(inl.sum())
        q_new = (
            cv2.perspectiveTransform(
                self.quad.astype(np.float32).reshape(-1, 1, 2), h_step
            )
            .reshape(-1, 2)
            .astype(np.float64)
        )
        area_new = abs(cv2.contourArea(q_new.astype(np.float32)))
        area_cur = abs(cv2.contourArea(self.quad.astype(np.float32)))
        if not _quad_is_plausible(q_new, self.anchor_area) or not (
            0.5 * area_cur <= area_new <= 2.0 * area_cur
        ):
            return self._die()
        self._h_total = h_step @ self._h_total
        self.quad = (
            cv2.perspectiveTransform(
                self.anchor_quad.astype(np.float32).reshape(-1, 1, 2),
                self._h_total,
            )
            .reshape(-1, 2)
            .astype(np.float64)
        )
        self.steps_since_anchor += 1
        return self.quad.copy()


class LkUpdateGate:
    """Portao anti-flip dos updates de deteccao (frente 97).

    feed(deteccao, ultima_saida, propagacao_viva) devolve
    (evento, aceita). Sem saida anterior ou com a propagacao morta,
    aceita sem julgar. Inovacao <= LK_GATE_HOLD_PX aceita. Acima
    disso, segura (pendente) e so confirma quando a deteccao
    seguinte pousa a <= LK_GATE_CONFIRM_PX do pendente; deteccao
    que volta para perto da saida descarta o pendente e aceita;
    deteccao que salta para um terceiro lugar substitui o pendente
    e segue segurada.
    """

    def __init__(self):
        self.pending = None
        self.held = 0
        self.confirmed = 0
        self.discarded = 0

    def reset(self):
        self.pending = None

    def feed(self, detected, last_output, propagation_alive):
        detected = np.asarray(detected, dtype=np.float64).reshape(-1, 2)
        if last_output is None or not propagation_alive:
            if self.pending is not None:
                self.pending = None
                self.discarded += 1
                return "discard_dead", True
            return "accept_open", True
        if self.pending is not None:
            if max_corner_distance(self.pending, detected) <= LK_GATE_CONFIRM_PX:
                self.pending = None
                self.confirmed += 1
                return "confirm", True
            if max_corner_distance(last_output, detected) <= LK_GATE_HOLD_PX:
                self.pending = None
                self.discarded += 1
                return "discard_return", True
            self.pending = detected.copy()
            self.held += 1
            return "hold_replace", False
        if max_corner_distance(last_output, detected) <= LK_GATE_HOLD_PX:
            return "accept", True
        self.pending = detected.copy()
        self.held += 1
        return "hold", False
