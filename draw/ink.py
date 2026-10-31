"""Extração da tinta: separar o que foi desenhado do papel branco.

A separação tem duas camadas, medidas no replay do vídeo de desenho
(~/workspace/draw-lab/stroke-fix-2026-10-29/REPORT.md). A primeira é o
limiar adaptativo de sempre sobre a folha retificada, robusto a
variações de iluminação ao longo da página. A segunda é um veto de
fundo separável: um pixel só é tinta se estiver escuro o bastante
contra a mediana da sua linha inteira e da sua coluna inteira. A
pauta impressa ocupa a linha toda, então a mediana da linha já é o
brilho da pauta e ela não passa no veto; a lombada do caderno e as
bordas da folha ocupam a coluna toda e caem pelo mesmo caminho. O
traço de caneta fica dezenas de níveis abaixo das duas medianas e
passa. Um portão de cor (saturação ou preto absoluto) foi medido e
reprovado neste material: o traço real é cinza de baixa saturação,
igual ao papel nesses canais, e o portão apagava o desenho.

A tinta do canvas não é mais lida de um frame só: o InkAccumulator
junta as observações de uma janela de frames no espaço do papel. Um
pixel entra no canvas quando a fração de tinta entre as amostras
visíveis da janela passa de um limiar de entrada e sai quando cai
abaixo de um limiar de saída menor (histerese). Pixel coberto pela
mão é desconhecido: não conta como tinta nem como papel limpo, o
que preserva o que já foi desenhado embaixo da mão e também
preserva a borracha (apagou e ficou visível como papel limpo por
amostras bastantes, sai do canvas).

No modo com marcadores, as áreas dos marcadores são apagadas da
máscara para não virarem "tinta" no canvas. No modo sem marcadores
não há marcador nenhum para apagar: apagar as mesmas áreas
apagaria desenho de verdade perto dos cantos da página.
"""

from collections import deque
from fractions import Fraction
import math
import time

import cv2
import numpy as np

from . import config

_VETO_DELTA = 6.0
_MORPH_OPEN_KERNEL = np.ones((2, 2), dtype=np.uint8)
_ERODE_3X3_KERNEL = np.ones((3, 3), dtype=np.uint8)


def _marker_mask(shape):
    """Máscara (255 = ignorar) cobrindo as áreas dos marcadores."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    radius = config.MARKER_SIZE_PX // 2 + 15
    for center in config.MARKER_CENTERS.values():
        cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask


def _twice_median_rows_u8(arr_u8):
    """Retorna 2 * mediana de cada linha de um array uint8 2D, em int16.

    Exato bit a bit contra 2 * np.median(arr.astype(float32), axis=1),
    sem alocar cópia float32 de 4 MiB da página inteira.
    """
    w = arr_u8.shape[1]
    mid = w // 2
    if w % 2 == 1:
        part = np.partition(arr_u8, mid, axis=1)
        return part[:, mid].astype(np.int16) * 2
    part = np.partition(arr_u8, (mid - 1, mid), axis=1)
    return part[:, mid - 1].astype(np.int16) + part[:, mid].astype(np.int16)


def _separable_veto(gray):
    """True onde o pixel é escuro contra a linha e a coluna inteiras.

    A mediana de uma linha da folha retificada é o brilho típico
    daquela linha (papel, ou pauta onde a linha é pautada); a da
    coluna idem (papel, ou lombada e bordas onde a coluna as cruza).
    Estruturas que ocupam a linha ou a coluna inteira têm delta
    perto de zero contra a própria mediana e são vetadas; o traço,
    local e bem mais escuro, passa.

    Para gray uint8 e _VETO_DELTA semisseiro/inteiro, a desigualdade
    gray <= mediana - delta equivale exatamente a
    2*gray <= 2*mediana - 2*delta, ou gray < floor((2*med - 2*delta)/2) + 1
    limitado a [0, 255] em uint8 (quando o limite é negativo, clip em 0
    torna gray < 0 sempre False). Zero pixels divergentes da versão float32.
    """
    if gray.dtype != np.uint8:
        gray_f = gray.astype(np.float32)
        bg_row = np.median(gray_f, axis=1, keepdims=True)
        bg_col = np.median(gray_f, axis=0, keepdims=True)
        return (gray_f <= bg_row - _VETO_DELTA) & (
            gray_f <= bg_col - _VETO_DELTA
        )
    delta2 = int(round(2.0 * _VETO_DELTA))
    r2 = _twice_median_rows_u8(gray)
    c2 = _twice_median_rows_u8(np.ascontiguousarray(gray.T))
    r_u8 = np.clip(((r2 - delta2) >> 1) + 1, 0, 255).astype(np.uint8)[:, None]
    c_u8 = np.clip(((c2 - delta2) >> 1) + 1, 0, 255).astype(np.uint8)[None, :]
    return gray < np.minimum(r_u8, c_u8)


def _apply_separable_veto_sparse(ink, gray):
    """Zera em `ink` os pixels reprovados pelo veto separável.

    Calcula as medianas apenas nas linhas e colunas que já contêm
    algum candidato em `ink` após o limiar adaptativo: linhas e
    colunas sem tinta não precisam de partição. Resultado idêntico
    pixel a pixel a `ink[~_separable_veto(gray)] = 0`.
    """
    rows = np.flatnonzero(ink.any(axis=1))
    if rows.size == 0:
        return
    cols = np.flatnonzero(ink.any(axis=0))
    delta2 = int(round(2.0 * _VETO_DELTA))
    r2 = _twice_median_rows_u8(gray[rows])
    c2 = _twice_median_rows_u8(np.ascontiguousarray(gray[:, cols].T))
    r_u8 = np.clip(((r2 - delta2) >> 1) + 1, 0, 255).astype(np.uint8)[:, None]
    c_u8 = np.clip(((c2 - delta2) >> 1) + 1, 0, 255).astype(np.uint8)[None, :]
    thresh = np.minimum(r_u8, c_u8)
    sub_ink = ink[np.ix_(rows, cols)]
    sub_ink[gray[np.ix_(rows, cols)] >= thresh] = 0
    ink[np.ix_(rows, cols)] = sub_ink


def extract_ink(warped, clear_markers=False):
    """Retorna a máscara de tinta (255 = tinta) da folha retificada.

    clear_markers apaga as áreas nominais dos marcadores ArUco dos
    cantos. Só faz sentido no modo com marcadores, onde a folha-modelo
    tem marcadores impressos de verdade; o chamador desse modo é quem
    pede a limpeza. Sem o pedido, nenhum pixel é apagado por posição.

    O veto de fundo separável vale só no caminho sem marcadores, que
    é onde mora o papel pautado de caderno. A folha-modelo impressa
    não tem pauta: nela o veto não derruba artefato nenhum e só custa
    recall (medido no selftest: 96,03% sem o veto, 93,84% com ele).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=12,
    )
    ink = cv2.bitwise_not(binary)  # tinta escura vira branco (255)
    if clear_markers:
        ink[_marker_mask(warped.shape) > 0] = 0
    # Remove ruído pequeno (poeira, textura do papel).
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, _MORPH_OPEN_KERNEL)
    if not clear_markers:
        _apply_separable_veto_sparse(ink, gray)
    return ink


# Amostras da pagina base e delta da tinta da sessao. A pagina base
# e a mediana dos primeiros quadros retificados depois que o
# usuario manda comecar (tecla C no app): pauta, margem impressa,
# lombada e o que ja estava desenhado entram nela e somem da
# observacao. So conta como tinta o que escurece DEPOIS da base
# fechada. O delta de 12 niveis fica logo abaixo do traco real
# medido contra o papel (~13 niveis, auditoria da tinta) e acima
# da cintilacao de luz; o resto quem limpa e o voto temporal do
# InkAccumulator sobre esta mascara. A observacao normaliza a
# iluminacao do quadro pela razao das medianas contra a base
# (ganho limitado), para uma mudanca global de luz nao virar
# tinta na pagina inteira.
_SESSION_BASE_SAMPLES = 12
_SESSION_INK_DELTA = 12.0

# Limiar do realinhamento a referencia, medido no laboratorio
# (draw-lab/session-align/REPORT.md e as rodadas de replay de
# 2026-10-30). A resposta da correlacao de fase NAO serve de
# confianca em producao: ela mede a fracao de energia espectral no
# pico, e tinta nova no quadro a derruba (0,95 sem traco novo para
# 0,65 com 8 tracos, na pagina real), embora a estimativa de
# deslocamento continue solida (erro ~0,2 px em todos esses
# casos). Acima de 10 px a estimativa nao e aplicada: pode ser
# movimento real ou wrap da correlacao, e nos dois casos chutar
# um warp de 40+ px e pior que nao corrigir. Nenhum numero daqui
# bloqueia a sessao; portoes foram removidos (ver observe).
_ALIGN_MAX_SHIFT_PX = 10.0
_BASE_MIN_VISIBLE = 1
_MIN_GAIN_VISIBLE_SUBSAMPLES = 256
_ALIGN_WINDOW = None


def align_to_reference(reference_gray, current_gray):
    """Realinha o quadro atual a referencia da sessao.

    Os dois argumentos sao cinza float32 no espaco do papel.
    Retorna (alinhado, (sx, sy), response, ok): alinhado e o
    quadro atual com o deslocamento estimado removido (o proprio
    current_gray, sem copia, quando ok e False); (sx, sy) e o
    deslocamento estimado do conteudo do atual contra a
    referencia, em px; response e a resposta da correlacao de
    fase (diagnostico; nao entra no ok, ver o comentario dos
    limiares); ok diz apenas que o deslocamento estimado e
    plausivel como jitter (<= _ALIGN_MAX_SHIFT_PX). A confianca
    de verdade e a consistencia da estimativa entre quadros,
    julgada por quem chama (ver observe).

    Medido no laboratorio: erro de estimativa <= 0,04 px ate 5 px
    de deslocamento e falsa tinta da regra da sessao de 21.889 px
    (2,15 por cento da pagina, sem correcao a 5 px) para 93 px
    (0,009 por cento), a 13,5 ms por chamada. O ECC affine entrega
    o mesmo neste modo de falha (deslocamento puro) custando
    243,6 ms, por isso fica fora do laco.
    """
    global _ALIGN_WINDOW
    if _ALIGN_WINDOW is None or _ALIGN_WINDOW.shape != reference_gray.shape:
        h, w = reference_gray.shape
        _ALIGN_WINDOW = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (sx, sy), response = cv2.phaseCorrelate(
        reference_gray, current_gray, _ALIGN_WINDOW)
    shift = float(np.hypot(sx, sy))
    if shift > _ALIGN_MAX_SHIFT_PX:
        return current_gray, (float(sx), float(sy)), float(response), False
    # Sem WARP_INVERSE_MAP, o warpAffine inverte a matriz: o
    # deslocamento do conteudo e (b1, b2). Para remover o
    # deslocamento estimado (sx, sy), b = (-sx, -sy).
    m = np.array([[1.0, 0.0, -sx], [0.0, 1.0, -sy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        current_gray, m,
        (current_gray.shape[1], current_gray.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned, (float(sx), float(sy)), float(response), True


class SessionBaselineInk:
    """Tinta da sessao: o que escureceu em relacao a pagina base.

    Diferente de extract_ink, que julga cada quadro sozinho contra
    limiares absolutos e por isso entrega a estrutura do caderno
    (pauta, margem, lombada) como tinta, esta fonte compara o
    quadro com a pagina como ela estava quando a sessao abriu.
    Desenho preexistente vira fundo por construcao: a pergunta que
    ela responde e "o que foi desenhado nesta sessao".
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._samples = []
        self._sample_vis = []
        self._baseline_min = None
        self.reference = None
        self._ref_median = None

    @property
    def ready(self):
        return self._baseline_min is not None

    def add_sample(self, warped, visible=None):
        """Alimenta a base; a mediana fecha na decima segunda amostra.

        Com mascara de visibilidade, a mediana de cada pixel usa
        SO as amostras em que ele estava visivel: mao ou braco
        parados sobre a pagina durante o C nao entram na
        referencia (medido no replay de 2026-10-30: com o braco
        em 10 de 12 amostras, a mediana comum gravava o braco
        como estrutura da pagina e a sessao morria em segundos
        quando ele saia). Pixel nunca visivel cai para a mediana
        comum, que e o que existe.
        """
        if self.ready:
            return
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        self._samples.append(gray)
        self._sample_vis.append(
            None if visible is None else visible.astype(bool))
        if len(self._samples) >= _SESSION_BASE_SAMPLES:
            stack = np.stack(self._samples).astype(np.float32)
            base_raw = np.median(stack, axis=0).astype(np.float32)
            if any(v is not None for v in self._sample_vis):
                vis = np.stack([
                    np.ones(stack.shape[1:], dtype=bool)
                    if v is None else v
                    for v in self._sample_vis
                ])
                count = vis.sum(axis=0)
                ordered = np.sort(
                    np.where(vis, stack, np.float32(1e6)), axis=0)
                pick = np.clip((count - 1) // 2, 0, stack.shape[0] - 1)
                picked = np.take_along_axis(
                    ordered, pick[None].astype(np.int64), axis=0)[0]
                base_raw = np.where(
                    count >= _BASE_MIN_VISIBLE, picked, base_raw
                ).astype(np.float32)
            base = cv2.GaussianBlur(base_raw, (3, 3), 0)
            # A referencia cheia fica guardada sem blur: e contra
            # ela que o realinhamento mede o deslocamento do
            # quadro atual quando o caderno se mexe.
            self.reference = base_raw
            self._ref_median = float(np.median(base[::2, ::2]))
            # Minimo local 3x3: a estrutura escura da base ganha um
            # halo de 1 px que absorve o jitter residual do warp,
            # para a pauta nao virar tinta quando o quadro treme.
            self._baseline_min = cv2.erode(base, _ERODE_3X3_KERNEL)
            self._samples = []
            self._sample_vis = []

    def observe(self, warped, visible=None):
        """Mascara (255 = tinta) do que escureceu contra a base.

        Antes do delta, o quadro e realinhado a referencia por
        correlacao de fase: sem isso, um deslocamento de 2 px do
        caderno transforma a pauta em falsa tinta (medido: 1.362 px
        sem correcao, 0 com). Estimativa alem de
        _ALIGN_MAX_SHIFT_PX nao e aplicada (a correlacao da o wrap
        e o quadro entra como esta), mas NADA aqui bloqueia a
        sessao: nao existe portao, contador de falhas nem banner
        de alinhamento. Portoes de alinhamento foram removidos em
        2026-10-30 depois de matarem sessoes boas ao vivo e em
        replay (residuo estrutural nao separa "a folha mexeu" de
        "algo cobriu a folha": mao, antebraco, sombra e flexao
        divergem tanto quanto movimento real). Se a folha mexe de
        verdade, o canvas degrada e quem decide recomecar e o
        usuario, com a tecla C. Quem filtra o lixo transitorio do
        delta e o acumulador com histerese e veto de visibilidade.
        """
        shape = warped.shape[:2]
        if not self.ready:
            return np.zeros(shape, dtype=np.uint8)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        aligned, _est, _response, _ok = align_to_reference(
            self.reference, gray)
        sub_aligned = aligned[::2, ::2]
        if visible is not None:
            vis_bool = (
                visible if visible.dtype == bool else (visible > 0)
            )
            sub_vis = vis_bool[::2, ::2]
            if int(sub_vis.sum()) >= _MIN_GAIN_VISIBLE_SUBSAMPLES:
                cur_median = float(np.median(sub_aligned[sub_vis]))
            else:
                cur_median = self._ref_median or 0.0
        else:
            cur_median = float(np.median(sub_aligned))
            vis_bool = None
        gain = 1.0
        if cur_median > 1.0 and self._ref_median is not None:
            gain = float(np.clip(
                self._ref_median / cur_median, 0.85, 1.18))
        scaled = aligned * gain if gain != 1.0 else aligned
        blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
        delta = self._baseline_min - blurred
        ink = delta >= _SESSION_INK_DELTA
        if vis_bool is not None:
            # Pixel coberto nao e tinta: a mao pousada sobre a
            # pagina escurece contra a base, mas nao desenha. O
            # acumulador tambem veta por visibilidade; zerar aqui
            # mantem a saida do observe limpa para qualquer
            # consumidor.
            ink &= vis_bool
        return ink.view(np.uint8) * np.uint8(255)


def render_canvas(ink_mask, out=None):
    """Renderiza canvas branco com tinta preta, reutilizando ``out`` se dado."""
    if out is None:
        canvas = np.full((config.PAPER_H, config.PAPER_W, 3), 255, dtype=np.uint8)
    else:
        expected_shape = (config.PAPER_H, config.PAPER_W, 3)
        if out.shape != expected_shape or out.dtype != np.uint8:
            raise ValueError(
                f"out de render_canvas precisa ser {expected_shape} uint8, "
                f"veio {out.shape} {out.dtype}"
            )
        canvas = out
        canvas.fill(255)
    canvas[ink_mask > 0] = (0, 0, 0)
    return canvas


class InkAccumulator:
    """Canvas acumulativo no espaço do papel, com histerese temporal.

    Cada update recebe a observação de tinta de um frame (a máscara
    de extract_ink) e a máscara do que está visível nesse frame
    (False sob a mão; None quando não há máscara de mão, caso em que
    tudo conta como visível). A evidência de tinta é a observação
    dilatada de tolerance pixels, para absorver o jitter residual do
    warp entre frames vizinhos; evidência só conta onde o pixel está
    visível. Sobre a janela das últimas window amostras, cada pixel
    tem uma fração de tinta: votos de tinta dividido por votos
    visíveis. O pixel entra no canvas quando a fração passa de
    enter e sai quando cai a exit ou abaixo, julgado apenas onde a
    janela já viu o pixel ao menos min_visible vezes. Pixel sem
    julgamento suficiente congela no estado atual: é o que faz a
    região sob a mão não ganhar nem perder tinta.

    Os parâmetros foram medidos no replay do vídeo de desenho: a
    presença por pixel do traço real fica perto de 0,55 e a do
    artefato de pauta abaixo de 0,3, então enter=0,45 separa os
    dois; exit=0,20 segura o traço através da cintilação sem segurar
    o artefato, cuja fração desaba quando a luz muda.

    Além do canvas, o acumulador guarda dois carimbos por pixel,
    first_seen_ms e last_update_ms (uint32, 850×1200 cada, 0 =
    sem carimbo): quando o pixel entrou no canvas e quando a
    evidência dele se renovou pela última vez. É a ordem temporal
    dos traços que o raster sozinho perde, e é o que a vetorização
    consome para ordenar as polilinhas. A época dos carimbos é o
    relógio monotônico em milissegundos (time.monotonic), e o
    instante que vale é o de CAPTURA do frame, passado por quem
    chama em update(t_capture_ms=...): carimbo de processamento
    mediria a fila do pipeline, não o desenho. A manutenção dos
    carimbos só acontece em pixel visível no frame corrente:
    pixel sob a mão congela os dois carimbos junto com o estado.
    """

    def __init__(
        self,
        shape=(config.PAPER_H, config.PAPER_W),
        window=15,
        enter=0.45,
        exit=0.20,
        tolerance=1,
        min_visible=4,
    ):
        self._window = window
        self._enter = enter
        self._exit = exit
        # Julgamento da histerese em inteiro. enter e exit viram
        # fracoes exatas (0,45 = 9/20; 0,20 = 1/5) e a comparacao
        # ink/vis >= enter vira ink*escala >= coef*vis, sem a
        # divisao float32 de pagina cheia. Limiar sem fracao exata
        # de denominador pequeno falha alto na construcao em vez de
        # derivar em silencio. Equivalencia com a forma float
        # varrida par a par em draw-lab/bottleneck-2026-10-30/
        # ink-optim/check_predicates.py.
        enter_frac = Fraction(float(enter)).limit_denominator(100)
        exit_frac = Fraction(float(exit)).limit_denominator(100)
        if float(enter_frac) != float(enter) or float(
            exit_frac
        ) != float(exit):
            raise ValueError(
                "enter/exit precisam de fracao exata de denominador "
                "<= 100 para o julgamento inteiro"
            )
        self._count_scale = (
            enter_frac.denominator * exit_frac.denominator
        )
        self._enter_coef = enter_frac.numerator * exit_frac.denominator
        self._exit_coef = exit_frac.numerator * enter_frac.denominator
        self._min_visible = min_visible
        # Para window <= 254 (padrão: 15), contagens cabem em uint8 e a
        # histerese inteira inteira vira duas buscas SIMD via cv2.LUT
        # (~6x mais rápido e sem 16 MiB de temporários int32 por frame),
        # preservando equivalência matemática exata par a par.
        self._use_u8_lut = int(window) <= 254
        if self._use_u8_lut:
            lut_en = np.full(256, int(window) + 1, dtype=np.uint8)
            lut_ex = np.zeros(256, dtype=np.uint8)
            for v in range(int(window) + 1):
                if v >= min_visible:
                    lut_en[v] = min(255, math.ceil(v * enter_frac))
                    # Compara (ink + 1) <= lut_ex[v] para cobrir ink=0
                    # sem underflow em uint8 (0 representa nunca sair).
                    lut_ex[v] = min(255, math.floor(v * exit_frac) + 1)
            self._enter_lut = lut_en
            self._exit_lut = lut_ex
        else:
            self._enter_lut = None
            self._exit_lut = None
        size = 2 * tolerance + 1
        self._kernel = np.ones((size, size), dtype=np.uint8)
        self._shape = shape
        self.reset()

    def reset(self):
        """Esquece a janela e esvazia o canvas (folha nova, sessão nova)."""
        self._ink_votes = deque()
        self._vis_votes = deque()
        count_dtype = np.uint8 if self._use_u8_lut else np.int16
        self._ink_counts = np.zeros(self._shape, dtype=count_dtype)
        self._vis_counts = np.zeros(self._shape, dtype=count_dtype)
        self.state = np.zeros(self._shape, dtype=bool)
        self._first_seen_ms = np.zeros(self._shape, dtype=np.uint32)
        self._last_update_ms = np.zeros(self._shape, dtype=np.uint32)

    def first_seen_ms(self):
        """Cópia do plano de primeira vista (uint32, ms monotônico)."""
        return self._first_seen_ms.copy()

    def last_update_ms(self):
        """Cópia do plano de última atualização (uint32, ms monotônico)."""
        return self._last_update_ms.copy()

    def update(self, ink_mask, visible_mask=None, t_capture_ms=None):
        """Alimenta uma observação e retorna o canvas (255 = tinta).

        t_capture_ms é o instante de captura do frame, em ms do
        relógio monotônico. Se None, usa o instante de processamento
        (monotônico de agora) e os carimbos passam a medir
        processamento, não captura: o app deve passar o de captura.
        """
        if t_capture_ms is None:
            t_ms = int(time.monotonic() * 1000)
        else:
            t_ms = int(t_capture_ms)
        ink = ink_mask > 0
        if visible_mask is None:
            visible = np.ones(ink.shape, dtype=bool)
        elif visible_mask.dtype == bool:
            visible = visible_mask
        else:
            visible = visible_mask > 0
        evidence = (
            cv2.dilate(ink.astype(np.uint8), self._kernel).astype(bool)
            & visible
        )
        if len(self._ink_votes) == self._window:
            self._ink_counts -= self._ink_votes.popleft()
            self._vis_counts -= self._vis_votes.popleft()
        self._ink_votes.append(evidence)
        self._vis_votes.append(visible)
        self._ink_counts += evidence
        self._vis_counts += visible
        if self._use_u8_lut:
            th_en = cv2.LUT(self._vis_counts, self._enter_lut)
            th_ex = cv2.LUT(self._vis_counts, self._exit_lut)
            enter = ~self.state & (self._ink_counts >= th_en)
            clear = self.state & (self._ink_counts + np.uint8(1) <= th_ex)
        else:
            judged = self._vis_counts >= self._min_visible
            vis = np.maximum(self._vis_counts, 1).astype(np.int32)
            scaled = self._ink_counts.astype(np.int32) * self._count_scale
            enter = judged & ~self.state & (scaled >= vis * self._enter_coef)
            clear = judged & self.state & (scaled <= vis * self._exit_coef)
        self.state[enter] = True
        self.state[clear] = False
        self._update_stamps(evidence, visible, t_ms)
        return self.state.view(np.uint8) * np.uint8(255)

    def _update_stamps(self, evidence, visible, t_ms):
        """Mantém first_seen/last_update sobre o estado já atualizado.

        Semântica, só em pixel visível neste frame: pixel de tinta
        sem carimbo (entrou agora, ou entrou coberto e só agora
        ficou visível) ganha first_seen = last_update = t; pixel de
        tinta cuja evidência (a mesma dilatada que alimenta os
        votos) se renova neste frame tem last_update = t; pixel fora
        do canvas zera os dois. Pixel não visível não entra em
        nenhum dos três casos: sob a mão, os carimbos congelam.
        """
        t_val = np.uint32(t_ms & 0xFFFFFFFF)
        ink_visible = self.state & visible
        fresh = ink_visible & (self._first_seen_ms == 0)
        self._first_seen_ms[fresh] = t_val
        self._last_update_ms[fresh] = t_val
        renewed = ink_visible & evidence
        self._last_update_ms[renewed] = t_val
        # Quem saiu zera os dois, mas so onde ha carimbo para
        # zerar: a pagina inteira fora do canvas nao precisa das
        # duas escritas de zero. Restringir por first_seen basta,
        # porque last_update != 0 implica first_seen != 0: os
        # dois nascem juntos no fresh, e o renewed so escreve em
        # pixel de tinta visivel, que ou ja tinha first_seen ou
        # e fresh deste mesmo frame.
        gone = visible & ~self.state & (self._first_seen_ms != 0)
        self._first_seen_ms[gone] = 0
        self._last_update_ms[gone] = 0
