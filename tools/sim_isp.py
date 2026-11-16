"""Gêmeo óptico do sensor e deformações físicas da cena (ISP twin).

Camada entre a cena composta e o frame entregue, cobrindo os efeitos
que o script de colagem simples não tem — os quatro fenômenos do
sim-to-real gap da mesa de desenho:

* acoplamento de auto-exposição (AE) e balanço de branco (AWB) da
  webcam: o firmware reage à luminância/cor média da cena com atraso
  — mão ou folha grande entrando derruba a exposição por vários
  frames até convergir de novo (``IspTwin``);
* motion blur direcional proporcional à velocidade real de mão,
  borracha e folha (``motion_kernel``/``blur_along``);
* penumbra: a mão projeta sombra suave deslocada na direção da luz
  sobre folha e mesa (``cast_shadow``);
* reflexo especular da tinta: caneta brilha numa faixa da folha no
  ângulo da luminária — o traço fica cinza claro ali (``ink_sheen``);
* curvatura da folha: papel real não é plano — malha deformável por
  remap denso, com foreshortening e sombreamento da área curvada
  (``scene_to_page_map``/``warp_paper_mapped``);
* compressão JPEG controlada, somando-se aos artefatos do mp4v.

Tudo é opt-in no SyntheticCapture: sem os flags novos, o caminho é o
fotométrico histórico (``_sequence_photometric`` com ganho 1), byte a
byte. Nenhum efeito toca o ground truth além do que a câmera veria —
sombras e brilho mudam pixels, não classes; a curvatura muda a
geometria e o GT exporta o mapa denso correspondente.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from tools import gen_synthetic as gs

SIZE = gs.SIZE

_SCENE_GRID = np.meshgrid(
    np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32)
)


# ------------------------------------------------------------------- blur


def motion_kernel(vx: float, vy: float, gain: float = 1.0) -> np.ndarray | None:
    """Kernel linear de motion blur na direção da velocidade (px/frame).

    O comprimento cresce com a velocidade (teto 15 px — exposição de
    webcam na mesa real fica na casa de 1/30–1/60 s). Velocidade
    insignificante devolve None: a chamada é no-op.
    """
    speed = math.hypot(vx, vy) * gain
    if not math.isfinite(speed) or speed <= 0:
        return None
    length = int(round(min(15.0, speed)))
    if length < 2:
        return None
    side = 2 * length + 1
    cx = cy = float(length)
    ux, uy = vx / math.hypot(vx, vy), vy / math.hypot(vx, vy)
    kernel = np.zeros((side, side), np.float32)
    cv2.line(
        kernel,
        (int(round(cx - ux * length)), int(round(cy - uy * length))),
        (int(round(cx + ux * length)), int(round(cy + uy * length))),
        1.0,
    )
    total = float(kernel.sum())
    return kernel / total if total > 0 else None


def blur_along(img: np.ndarray, vx: float, vy: float, gain: float = 1.0):
    """Aplica motion blur direcional; devolve o mesmo array se |v| < 2.

    Borda constante zero: fora do sprite/folha não há sinal —
    BORDER_REFLECT espalharia a cobertura para além do objeto e
    criaria franjas de rótulo ambíguo onde nada se moveu.
    """
    kernel = motion_kernel(vx, vy, gain)
    if kernel is None:
        return img
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)


# ------------------------------------------------------------ penumbra


def cast_shadow(
    rgb: np.ndarray,
    coverage: np.ndarray,
    cx: int,
    cy: int,
    offset: tuple[float, float],
    blur_sigma: float,
    strength: float,
) -> None:
    """Escurece a cena sob a sombra projetada de um sprite (in-place).

    ``coverage`` é o mapa de cobertura do sprite (mesma convenção de
    ``_transform_hand``); ``cx, cy`` é onde ele será colado. A sombra é
    a cobertura deslocada por ``offset`` na cena, esfumaçada por
    ``blur_sigma`` (penumbra, não recorte duro) e aplicada como
    escurecimento multiplicativo — cai sobre folha, tinta e mesa, como
    na mesa real.
    """
    field = np.zeros((SIZE, SIZE), np.float32)
    hh, hw = coverage.shape[:2]
    x0, y0 = cx - hw // 2, cy - hh // 2
    sx0, sy0 = max(0, -x0), max(0, -y0)
    ex0, ey0 = max(0, x0 + hw - SIZE), max(0, y0 + hh - SIZE)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(SIZE, x0 + hw), min(SIZE, y0 + hh)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    field[dy0:dy1, dx0:dx1] = coverage[sy0 : hh - ey0, sx0 : hw - ex0]
    shift = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    shifted = cv2.warpAffine(
        field, shift, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0.0
    )
    shifted = cv2.GaussianBlur(shifted, (0, 0), blur_sigma)
    factor = np.clip(1.0 - strength * shifted[..., None], 0.0, 1.0)
    rgb[:] = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# ---------------------------------------------------------- especular


def ink_sheen(
    rgb_w: np.ndarray,
    ink_w: np.ndarray,
    center: tuple[float, float],
    direction: tuple[float, float],
    width: float,
    strength: float,
) -> None:
    """Reflexo especular da tinta na faixa da luminária (in-place).

    A faixa de brilho é uma gaussiana alongada na direção da luz
    incidente centrada em ``center``: dentro dela o traço preto vira
    cinza claro — o regime em que extratores de tinta perdem o traço.
    ``strength`` controla o quanto a tinta clareia no pico (0 desliga).
    """
    if strength <= 0:
        return
    gx, gy = _SCENE_GRID
    dx, dy = direction
    dist = (gx - center[0]) * dx + (gy - center[1]) * dy
    band = np.exp(-((dist / max(1.0, width)) ** 2)).astype(np.float32)
    lift = (band * strength)[ink_w > 0][..., None]
    region = rgb_w[ink_w > 0].astype(np.float32)
    rgb_w[ink_w > 0] = np.clip(region * (1 - lift) + 225.0 * lift, 0, 255).astype(
        np.uint8
    )


# ---------------------------------------------------- curvatura da folha


def scene_to_page_map(
    page_to_scene_inv: np.ndarray,
    pw: int,
    ph: int,
    curl: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mapa denso cena→página para ``cv2.remap``.

    Sem ``curl``, é exatamente a homografia inversa avaliada na grade
    da cena (comportamento de ``_warp_paper``). Com ``curl`` — dict com
    ``axis`` (vetor unitário em espaço de página), ``bend_start`` e
    ``theta_max`` (rad) — aplica a deformação inversa da curvatura
    cilíndrica: a folha encurvada projetada encurta em direção à dobra
    (foreshortening), e pontos além da borda projetada ficam inválidos
    (mapa em -1 → cobertura zero, a folha realmente não está ali).

    Modelo: para coordenada s = p·axis além de ``bend_start``, o arco
    projeta s' = bend_start + R·sin((s − bend_start)/R), com
    R = (s_edge − bend_start)/theta_max. O inverso analítico é
    s = bend_start + R·asin((s' − bend_start)/R).
    """
    gx, gy = _SCENE_GRID
    h_inv = page_to_scene_inv.astype(np.float64)
    den = h_inv[2, 0] * gx + h_inv[2, 1] * gy + h_inv[2, 2]
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    qx = (h_inv[0, 0] * gx + h_inv[0, 1] * gy + h_inv[0, 2]) / den
    qy = (h_inv[1, 0] * gx + h_inv[1, 1] * gy + h_inv[1, 2]) / den
    if curl is None:
        return qx.astype(np.float32), qy.astype(np.float32)
    ux, uy = curl["axis"]
    s = qx * ux + qy * uy
    s_b = float(curl["bend_start"])
    s_edge = float(curl["edge"])
    theta_max = float(curl["theta_max"])
    radius = (s_edge - s_b) / theta_max
    beyond = s > s_b
    arg = np.clip((s - s_b) / radius, -1.0, 1.0)
    s_src = np.where(beyond, s_b + radius * np.arcsin(arg), s)
    delta = s_src - s
    qx = qx + delta * ux
    qy = qy + delta * uy
    # Além da borda projetada da dobra não há papel: o mapa sai da
    # página e as coberturas caem a zero (constante de borda 0).
    invalid = beyond & (((s - s_b) / radius) > math.sin(theta_max))
    qx = np.where(invalid, -1.0, qx)
    qy = np.where(invalid, -1.0, qy)
    return qx.astype(np.float32), qy.astype(np.float32)


def curl_shade(pw: int, ph: int, curl: dict, depth: float = 0.45) -> np.ndarray:
    """Sombreamento multiplicativo da dobra, em espaço de página.

    A área curvada escurece com o seno do ângulo local da tangente —
    a face que vira para longe da luz perde luminância (o "vale" do
    papel levantado). Devolve campo float32 (ph, pw) em [1−depth, 1].
    """
    gy, gx = np.meshgrid(
        np.arange(ph, dtype=np.float32),
        np.arange(pw, dtype=np.float32),
        indexing="ij",
    )
    ux, uy = curl["axis"]
    s = gx * ux + gy * uy
    s_b = float(curl["bend_start"])
    radius = (float(curl["edge"]) - s_b) / float(curl["theta_max"])
    tilt = np.clip((s - s_b) / max(radius, 1e-6), 0.0, 1.0)
    return 1.0 - depth * tilt


def warp_paper_mapped(
    paper: np.ndarray,
    ink: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``_warp_paper`` por mapa denso (curvatura) — mesmo contrato de saída.

    RGB e coberturas em INTER_LINEAR (fração real de camada por pixel),
    máscaras duras em INTER_NEAREST; borda constante zero — fora do
    mapa não há papel, como na homografia.
    """
    h, w = paper.shape[:2]
    rgb_w = cv2.remap(
        paper, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    paper_field = np.full((h, w), 255, dtype=np.uint8)
    paper_mask = cv2.remap(
        paper_field,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    ink_w = cv2.remap(
        ink, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
    )
    paper_cov = cv2.remap(
        paper_field.astype(np.float32) / 255.0,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    ink_cov = cv2.remap(
        ink.astype(np.float32) / 255.0,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rgb_w, paper_mask, ink_w, paper_cov, ink_cov


# ----------------------------------------------------------- ISP twin


class IspTwin:
    """Firmware da webcam: AE e AWB reativos à cena, ruído e JPEG.

    A auto-exposição mede a luminância média da cena COMPOSTA (mão,
    folha e mesa já dentro do quadro) e aproxima um ganho que mira um
    alvo de meio-tom, com constante de tempo de alguns frames — a
    folha branca ocupando metade do quadro ou a mão entrando derruba a
    exposição até convergir, exatamente o regime em que o traço perde
    contraste na mesa real. O AWB faz o mesmo para os canais
    (estimador gray-world com deriva lenta). O ruído de sensor mantém
    o σ do caminho histórico, escalando com o ganho aplicado (ISO
    sobe, ruído sobe junto).

    ``process`` opera em frames BGR (ordem do OpenCV): os pesos de
    luminância e os índices do AWB seguem essa convenção.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        fps: float,
        *,
        ae: bool = True,
        awb: bool = True,
        jpeg_quality: int | None = None,
        ae_target: float = 120.0,
        ae_tau: float = 0.35,
        awb_tau: float = 1.2,
        noise_sigma: float = 5.0,
    ):
        if jpeg_quality is not None and not 10 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality deve estar em [10, 100]")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps deve ser positivo e finito")
        for name, value in (
            ("ae_target", ae_target),
            ("ae_tau", ae_tau),
            ("awb_tau", awb_tau),
            ("noise_sigma", noise_sigma),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve ser positivo e finito")
        self._rng = rng
        self._alpha_ae = 1.0 - math.exp(-1.0 / (float(fps) * ae_tau))
        self._alpha_awb = 1.0 - math.exp(-1.0 / (float(fps) * awb_tau))
        self.ae = bool(ae)
        self.awb = bool(awb)
        self.jpeg_quality = jpeg_quality
        self.ae_target = float(ae_target)
        self.noise_sigma = float(noise_sigma)
        self.gain = 1.0
        self.wb = np.ones(3, np.float32)

    def process(self, rgb: np.ndarray) -> np.ndarray:
        frame = rgb.astype(np.float32)
        if self.ae:
            lum = float(
                (
                    0.114 * frame[..., 0]
                    + 0.587 * frame[..., 1]
                    + 0.299 * frame[..., 2]
                ).mean()
            )
            desired = float(np.clip(self.ae_target / max(lum, 1.0), 0.5, 2.0))
            self.gain += (desired - self.gain) * self._alpha_ae
        if self.awb:
            means = frame.reshape(-1, 3).mean(axis=0)
            desired_wb = np.clip(
                float(means.mean()) / np.maximum(means, 1.0), 0.8, 1.25
            )
            self.wb += (desired_wb.astype(np.float32) - self.wb) * self._alpha_awb
            self.wb *= 1.0 + self._rng.normal(0.0, 0.003, 3).astype(np.float32)
            self.wb = np.clip(self.wb, 0.75, 1.33)
        img = frame * self.wb[None, None, :]
        img = img * self.gain
        # Ruído de sensor escala com o ganho — ISO virtual.
        sigma = self.noise_sigma * max(1.0, float(self.gain))
        img += self._rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        out = np.clip(img, 0, 255).astype(np.uint8)
        if self.jpeg_quality is not None:
            ok, encoded = cv2.imencode(
                ".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)]
            )
            if ok:
                out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return out
