"""Régua de re-registro: mover o caderno = mesmo canvas? (F-07/F-09/F-11)

Mede, sem corrigir nada, se o laço do app distingue "a folha andou"
de "a folha trocou". Até 87f52e4 os dois eram o mesmo evento: `drop`
do `_SheetTracker` e a reabertura chamavam `new_generation()`. Na
main o drop só deixa a identidade pendente e a reabertura chama
`new_generation(preserve_identity=True)`: a primeira passada mantém o
canvas se o recall da tinta registrada for ≥
`_PAGE_IDENTITY_MIN_RECALL`. O gatilho de cobertura (`_INK_TURN_*`)
está desligado por padrão desde o hotfix de 2026-11-01.

O replay roda em contagem de passadas, sem ONNX e sem threads, e
dirige o CÓDIGO REAL do laço — `draw.loop.SheetInkLoop`, o mesmo
objeto que `draw.app.run` monta — com as dependências simuladas:

- uma câmera de cena: a página (mapa de probabilidade na grade da
  página) está num quad VERDADEIRO na câmera; cada quadro declara o
  quad que o detector devolve (com ruído, balançando, ou nada);
- um detector falso com latência de cena (`SceneDetectionWorker`,
  protocolo de `_SheetDetectionWorker`: `--detect-ms`, padrão
  36,7 ms medidos no Ryzen, um quadro em voo e fila latest-wins de
  tamanho 1) e uma presença de mão falsa (`SceneHandPresence`);
- um relógio de cena determinístico, o mesmo para o laço e para a
  trabalhadora;
- a `_PageCleanWorker` real executada por `pump()` (sem thread) logo
  depois de cada quadro do laço, com um page_clean falso que
  re-warpa o mapa da página pelo quad que o laço usou.

Drop, reabertura, mediana de `_INK_CORNER_HISTORY` cantos,
congelamento sem mão, evento de virada atrás da flag, registro na
âncora e acumulador são os do app: mudar o laço muda a régua, sem
espelho a manter.

Cada passada re-projeta o mapa da página pelo quad que o app usou
(`cv2.warpPerspective` do mapa: o modelo não é re-rodado no quadro
re-warpado — aproximação declarada). Com o laboratório
(`~/workspace/draw-lab`) presente, a página é a sessão real f01–f30
do cache de probabilidades; sem ele, uma página sintética
determinística (`"source": "synthetic"` no JSON).

Cenários:

1. move_same_canvas — desenho assentado, depois o caderno anda
   (translação/rotação/escala) e assenta de novo. Portão: canvas
   confirmado idêntico ao de antes a menos de re-warp (alinhado por
   homografia estimada, ECC), zero `new_generation()`, zero pixel
   confirmado novo fora da tolerância de fronteira.
2. drop_and_return — a folha some por N quadros e volta na mesma
   pose. Só reporta (sem veto): o canvas confirmado no fim é o de
   antes (a menos de re-warp) ou foi reconstruído do zero.
3. sway_negative — página parada, o ajuste do detector balança até
   ~314 px de canto. Portão: zero página nova (configuração padrão);
   a flag DRAW_PAGE_TURN_EVENT=1 é reportada à parte.

Tudo nas configurações de sinal de mão: sem_sinal (None), ausente
(False), presente_parada e presente_mexendo (True). Um run sem
nenhuma passada depois da perturbação passa por vacuidade (a
trabalhadora em descanso não viu nada) e é marcado `vacuous`; o
erro da geometria da tinta contra a página no fim
(`ink_geometry_error_px`) diz se o próximo traço cairia no lugar.
Uso:

  python tools/replay_reregister.py [--out resultado.json] [--strict]
      [--detect-ms 36.7]
"""

import argparse
import contextlib
import copy
import importlib.util
import json
import os
import sys
import threading
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import ink_layers as _ink  # noqa: E402
from draw.ink_layers import _LayeredInkAccumulator, _PageCleanWorker  # noqa: E402
from draw.loop import SheetInkLoop  # noqa: E402

HOME = Path.home()
LAB = HOME / "workspace/draw-lab"
WARPS = LAB / "real-negatives-2026-10-31/proc/warps"
RULER = LAB / "replay-ruler-2026-11-01"
THRESHOLD = 0.7
PASSES_PER_WARP = 3

# Relógio de cena: só alimenta os relógios do rastreador (0,3 s de
# salto, 0,5 s de perda) e a cadência da trabalhadora. Nada é
# reportado em segundos; a unidade da régua é a passada.
FPS = 30
DT = 1.0 / FPS
CAMERA_W, CAMERA_H = 1280, 720
DETECTOR_NOISE_PX = 1.0
# Duração de uma detecção de folha no Ryzen (draw/sheet_track.py).
DETECT_MS = 36.7
# Lógica do laço: a de draw/loop.py, dirigida de verdade. Desde
# 87f52e4 é a recuperação de identidade (drop guarda a geração, a
# reabertura decide pela tinta); a lógica anterior ("legacy", drop →
# new_generation()) não existe mais no app e saiu da régua junto com
# o espelho que a reproduzia.
APP_LOGIC = "main"
SETTLE_FRAMES = 5 * FPS
SWAY_FRAMES = 60 * FPS
SWAY_AMPLITUDE_PX = 314.0
SYNTHETIC_SHAPE = (600, 424)
# Sinal de mão: (estado, quando a mão "se mexe"). presente_parada
# só se mexe enquanto move o caderno; presente_mexendo é quem
# desenha (movimento a cada amostra: a trabalhadora nunca descansa).
HAND_CONFIGS = {
    "sem_sinal": (None, "never"),
    "ausente": (False, "never"),
    "presente_parada": (True, "move"),
    "presente_mexendo": (True, "always"),
}
GEOMETRY_TOL_PX = 5.0
BOUNDARY_TOL_PX = _ink._INK_FRONTIER_RADIUS
# Crescimento residual de confirmacao apos a recuperacao de
# identidade: medido 1-13 px em ~12,6 mil na baseline sintetica
# (2026-11-06); a duplicacao pre-correcao era 3,5-8,1 mil px.
RECOVERY_GROWTH_TOL_PX = 25

MOVES = {
    # Abaixo de _SHEET_NEAR_LIMIT_PX: caminho de mistura do rastreador.
    "translate_small": {"shift": (30.0, 10.0), "frames": 15},
    # Acima de _SHEET_BIG_JUMP_PX: caminho de salto com persistência.
    "translate_large": {"shift": (180.0, -40.0), "frames": 30},
    "rotate_12deg": {"angle": 12.0, "frames": 30},
    "scale_0_85": {"scale": 0.85, "frames": 30},
    "combined": {
        "shift": (120.0, 30.0),
        "angle": -8.0,
        "scale": 1.1,
        "frames": 45,
    },
}
DROP_FRAMES = {"short_6f": 6, "medium_30f": 30, "long_150f": 150}
DROP_MODES = ("detector_miss", "page_out")
SWAY_PROFILES = ("smooth", "bimodal")


# --------------------------------------------------------------------
# Páginas
# --------------------------------------------------------------------


def _load_prob_cache():
    spec = importlib.util.spec_from_file_location("prob_cache", RULER / "prob_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ProbCache()


def real_pages():
    """Sessão real f01–f30 (um mapa por estágio); assentada cicla f28–f30."""
    cache = _load_prob_cache()
    probs = []
    for i in range(1, 31):
        fid = f"f{i:02d}"
        img = cv2.imread(str(WARPS / f"{fid}_warp.png"))
        if img is None:
            raise RuntimeError(f"warp nao encontrado: {fid}")
        probs.append(np.asarray(cache.get(img, name=f"warp:{fid}"), np.float32))
    return [[prob] for prob in probs], probs[-3:]


def synthetic_pages(shape=SYNTHETIC_SHAPE, stages=10, seed=7):
    """Desenho sintético em estágios, com ombro sub-limiar real.

    Caneta (núcleo ~0,95) e lápis (núcleo ~0,6, só confirma pelo
    voto de hits); blur σ≈1 põe a banda 0,5–0,7 a 1–3 px do núcleo,
    o ombro medido nos mapas reais. Cada estágio tem 3 variantes
    (ganho, ruído, tremor sub-pixel) — a variação passada a passada
    que os warps reais têm.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    ink = np.zeros(shape, np.float32)
    sequence = []
    settled = []
    for stage in range(stages):
        for _ in range(3):
            n = int(rng.integers(5, 10))
            start = rng.uniform((0.1 * w, 0.1 * h), (0.9 * w, 0.9 * h))
            steps = rng.normal(0.0, 0.06 * min(h, w), size=(n, 2))
            pts = np.clip(
                start + np.cumsum(steps, axis=0),
                (0.05 * w, 0.05 * h),
                (0.95 * w, 0.95 * h),
            )
            strength = 0.85 if rng.random() < 0.8 else 0.55
            layer = np.zeros(shape, np.float32)
            cv2.polylines(
                layer,
                [pts.astype(np.int32).reshape(-1, 1, 2)],
                False,
                1.0,
                int(rng.integers(2, 4)),
                lineType=cv2.LINE_AA,
            )
            ink = np.maximum(ink, layer * strength)
        base = np.clip(cv2.GaussianBlur(ink, (0, 0), 1.0) * 1.15, 0, 0.98)
        variants = []
        for _ in range(3):
            gain = rng.uniform(0.92, 1.04)
            noise = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
            dx, dy = rng.uniform(-0.4, 0.4, size=2)
            moved = cv2.warpAffine(
                base,
                np.float32([[1, 0, dx], [0, 1, dy]]),
                (w, h),
                flags=cv2.INTER_LINEAR,
            )
            variants.append(
                np.clip(moved * gain + noise * (moved > 0.05), 0.0, 1.0).astype(
                    np.float32
                )
            )
        sequence.append(variants)
        settled = variants
    return sequence, settled


# --------------------------------------------------------------------
# Geometria da câmera
# --------------------------------------------------------------------


def base_quad(shape):
    """Quad verdadeiro da página na câmera (TL, TR, BR, BL)."""
    h, w = shape
    height = 466.0
    width = height * w / h
    cx, cy = 560.0, 370.0
    quad = np.array(
        [
            (cx - width / 2 + 12, cy - height / 2),
            (cx + width / 2 - 12, cy - height / 2),
            (cx + width / 2, cy + height / 2),
            (cx - width / 2, cy + height / 2),
        ],
        np.float32,
    )
    return quad


def transform_quad(quad, shift=(0.0, 0.0), angle=0.0, scale=1.0):
    center = quad.mean(axis=0)
    rad = np.deg2rad(angle)
    rot = np.array(
        [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]],
        np.float32,
    )
    out = (quad - center) @ rot.T * scale + center
    return (out + np.asarray(shift, np.float32)).astype(np.float32)


def _page_matrix(quad, shape):
    h, w = shape
    return _ink._corners_page_matrix(quad, w, h)


_ONES_FRAME = np.full((CAMERA_H, CAMERA_W), 255, np.uint8)


def render_pass(page, true_quad, warp_corners):
    """O que o modelo devolveria retificando com warp_corners.

    page vive na grade da página; a câmera a vê em true_quad; o app
    retifica com warp_corners. Fora do quadro da câmera não há
    imagem (prob 0).
    """
    shape = page.shape
    matrix = _page_matrix(warp_corners, shape) @ np.linalg.inv(
        _page_matrix(true_quad, shape)
    )
    if np.max(np.abs(matrix - np.eye(3))) < 1e-9:
        prob = page.copy()
    else:
        prob = cv2.warpPerspective(
            page,
            matrix,
            (shape[1], shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
    visible = cv2.warpPerspective(
        _ONES_FRAME,
        _page_matrix(warp_corners, shape),
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    prob[visible == 0] = 0.0
    return prob


# --------------------------------------------------------------------
# Dependências simuladas do laço real (draw/loop.py)
# --------------------------------------------------------------------


class SceneClock:
    """Relógio de cena: o mesmo instante para o laço e as trabalhadoras."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SceneWarp:
    """O "warp" que a câmera de cena entrega ao laço.

    Carrega a página, o quad verdadeiro e o quad com que o laço
    retificou; o mapa só é renderizado quando a passada da
    trabalhadora roda (latest-wins: warps substituídos nunca custam).
    """

    __slots__ = ("page", "true_quad", "corners")

    def __init__(self, page, true_quad, corners):
        self.page = page
        self.true_quad = true_quad
        self.corners = corners


class ScenePageClean:
    """page_clean falso: o mapa de probabilidade é o da cena re-warpado."""

    threshold = THRESHOLD

    def __init__(self):
        self.worker = None

    def bind(self, worker):
        self.worker = worker

    def ink_prob(self, warped):
        return self.observe(warped.page, warped.true_quad, warped.corners)

    def observe(self, page, true_quad, corners):
        return render_pass(page, true_quad, corners)


class CountingAccumulator(_LayeredInkAccumulator):
    """O acumulador real; só conta os sinais de virada e guarda o resultado."""

    def __init__(self, threshold=0.7):
        super().__init__(threshold)
        self.turn_signals = 0
        self.last_result = None

    def add(self, prob, visible=None):
        result = super().add(prob, visible)
        if result["turn"]:
            self.turn_signals += 1
        self.last_result = result
        return result


class ReplayInkWorker(_PageCleanWorker):
    """A trabalhadora real, dirigida por pump() (sem thread).

    Só acrescenta cópia profunda (os cenários partem de uma cópia do
    app assentado); nenhum comportamento é sobrescrito — o teste
    tests/test_replay_reregister.py cobra isso.
    """

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for key, value in self.__dict__.items():
            if key in ("_condition", "_thread"):
                continue
            setattr(clone, key, copy.deepcopy(value, memo))
        clone._condition = threading.Condition()
        clone._thread = threading.Thread(
            target=clone._work, name="draw-page-clean", daemon=True
        )
        return clone


def make_ink_worker(page_clean, clock):
    """`_PageCleanWorker` real em modo síncrono, com histórico completo."""
    worker = ReplayInkWorker(page_clean, clock=clock, history=(None, None))
    worker._accumulator = CountingAccumulator(
        threshold=getattr(page_clean, "threshold", 0.7)
    )
    bind = getattr(page_clean, "bind", None)
    if bind is not None:
        bind(worker)
    return worker


class SceneDetectionWorker:
    """Detector de folha falso com a latência de cena.

    O protocolo é o de `_SheetDetectionWorker` (submit/poll/error/
    take_traces), que o laço real consome: uma detecção em voo por
    vez, que termina `latency` segundos depois de começar; um quadro
    submetido com a detecção ocupada espera numa fila de tamanho 1 e
    é substituído pelo próximo (latest-wins). A "detecção" devolve o
    quad que a cena declara para o quadro (`Frame.detected`).
    """

    def __init__(self, latency, clock):
        self.latency = latency
        self.clock = clock
        self.inflight = None
        self.pending = None
        self.result = None
        self.seq = 0
        self.submitted = 0
        self.completed = 0
        self.replaced = 0
        self.error = None

    @staticmethod
    def _detect(frame):
        corners = (
            None
            if frame.detected is None
            else np.asarray(frame.detected, np.float32).copy()
        )
        return corners, frame.verdict

    def _start(self, job, start):
        frame, captured_at = job
        corners, verdict = self._detect(frame)
        self.inflight = (corners, verdict, captured_at, start + self.latency)

    def _advance(self, now):
        while self.inflight is not None and self.inflight[3] <= now + 1e-9:
            corners, verdict, captured_at, done_at = self.inflight
            self.inflight = None
            self.seq += 1
            self.completed += 1
            self.result = (
                self.seq,
                corners,
                verdict,
                captured_at,
                round(self.latency * 1000.0, 1),
            )
            if self.pending is not None:
                job, self.pending = self.pending, None
                self._start(job, max(done_at, job[1]))

    def poll(self, after_seq):
        self._advance(self.clock())
        if self.result is None or self.result[0] <= after_seq:
            return None
        return self.result

    def submit(self, frame, captured_at, frame_index=None):
        now = self.clock()
        self._advance(now)
        self.submitted += 1
        if self.inflight is None:
            self._start((frame, captured_at), now)
            return
        if self.pending is not None:
            self.replaced += 1
        self.pending = (frame, captured_at)

    def take_traces(self, upto_result_seq=None):
        return []


class SceneHandPresence:
    """Presença de mão falsa: o sinal que a cena declara para o quadro.

    Protocolo de `_HandPresenceWorker` (submit/poll/error): o
    resultado da amostra submetida num quadro é colhido pelo laço no
    poll do quadro seguinte, como a amostragem assíncrona real.
    """

    def __init__(self):
        self.seq = 0
        self.result = None
        self.error = None

    def submit(self, frame, _captured_ms):
        present, moved = frame.hand
        self.seq += 1
        self.result = (self.seq, present, moved)

    def poll(self, after_seq):
        if self.result is None or self.result[0] <= after_seq:
            return None
        return self.result

    def stop(self):
        pass


def _no_gray(_frame):
    return None


def _no_display_track(*_args):
    return None


def _no_display_render(*_args, **_kwargs):
    # O painel da janela fica fora da régua: a grade da página da
    # cena não é a do formato do app (render_canvas exige a dele).
    return None


def _log_stderr(*args):
    print(*args, file=sys.stderr)


class Frame:
    """O que a câmera de cena entrega num quadro."""

    shape = (CAMERA_H, CAMERA_W, 3)

    def __init__(self, page, true_quad, detected, verdict="paper", hand=(None, False)):
        self.page = page
        self.true_quad = true_quad
        self.detected = detected
        self.verdict = verdict
        self.hand = hand


class ReplayApp:
    """O laço real de draw/app.py (`SheetInkLoop`) em relógio de cena.

    Dependências injetadas: câmera de cena (os quadros que step()
    recebe), detector falso com latência de cena, presença de mão
    falsa, relógio determinístico e a trabalhadora de tinta real
    executada por pump() logo depois de cada quadro do laço. A
    decisão de geometria, sessão e geração é a do app, não um
    espelho dela.
    """

    def __init__(
        self,
        hand_state,
        turn_event_enabled=False,
        detect_ms=None,
        page_clean_factory=ScenePageClean,
    ):
        self.hand_state = hand_state
        self.turn_event_enabled = bool(turn_event_enabled)
        self.clock = SceneClock()
        self.worker = make_ink_worker(page_clean_factory(), self.clock)
        self.detector = SceneDetectionWorker(
            (DETECT_MS if detect_ms is None else detect_ms) / 1000.0, self.clock
        )
        self.last_warp_corners = None
        self._page_shape_hint = None
        self.loop = SheetInkLoop(
            detection_worker=self.detector,
            ink_worker=self.worker,
            hand_presence_worker=None if hand_state is None else SceneHandPresence(),
            clock=self.clock,
            warp_sheet=self._warp_sheet,
            to_gray=_no_gray,
            track_display=_no_display_track,
            render_display=_no_display_render,
            log=_log_stderr,
        )
        self.frame_index = 0

    def _warp_sheet(self, frame, corners):
        self.last_warp_corners = np.asarray(corners, np.float32).copy()
        page = (
            frame.page
            if frame.page is not None
            else np.zeros_like(self._page_shape_hint)
        )
        return SceneWarp(page, frame.true_quad, self.last_warp_corners)

    # Leituras do estado real (para os relatórios).
    @property
    def clean_generation(self):
        return self.loop.clean_generation

    @property
    def events(self):
        return dict(self.loop.generation_events)

    def canvas(self):
        """O canvas dos consumidores (clean_mask do laço)."""
        return self.loop.clean_mask

    def passes_total(self):
        return self.worker._passes_total

    def anchor(self):
        return self.worker._anchor_corners

    def identity_events(self):
        return list(self.worker._identity_events)

    def ecc_records(self):
        return list(self.worker._telemetry_records)

    def coverage_turn_signals(self):
        return self.worker._accumulator.turn_signals

    def pump_ink(self):
        return self.worker.pump()

    def step(self, frame):
        self.clock.now = self.frame_index * DT
        if frame.page is not None:
            self._page_shape_hint = frame.page
        with _scoped_env(_ink._ENV_TURN_EVENT_VAR, "1" if self.turn_event_enabled else "0"):
            outcome = self.loop.step(frame, self.clock.now * 1000.0)
            ran = self.pump_ink()
        if outcome is not None or self.worker.error is not None:
            raise RuntimeError("a trabalhadora de tinta falhou") from self.worker.error
        self.frame_index += 1
        return ran


@contextlib.contextmanager
def _scoped_env(name, value):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _hand(config, moving):
    """Sinal de mão do quadro para a configuração do run."""
    state, policy = config
    if state is not True:
        return (state, False)
    return (True, policy == "always" or (policy == "move" and moving))


def _noisy(quad, rng, sigma=DETECTOR_NOISE_PX):
    return (quad + rng.normal(0.0, sigma, size=quad.shape)).astype(np.float32)


def _rng(*names):
    return np.random.default_rng(zlib.crc32("/".join(names).encode()))


# --------------------------------------------------------------------
# Comparação a menos de re-warp
# --------------------------------------------------------------------


def _ecc_refine(before, after, init):
    """Homografia before→after refinada por ECC, ou None."""
    h, w = before.shape
    factor = max(1.0, max(h, w) / 600.0)
    size = (int(round(w / factor)), int(round(h / factor)))
    scale = np.diag([1.0 / factor, 1.0 / factor, 1.0])

    def prep(mask):
        img = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(img, (0, 0), 3.0)

    template, moving = prep(after), prep(before)
    small = scale @ init @ np.linalg.inv(scale)
    warp = np.linalg.inv(small).astype(np.float32)
    warp /= warp[2, 2]
    try:
        _cc, warp = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            None,
            5,
        )
    except cv2.error:
        return None
    forward = np.linalg.inv(scale) @ np.linalg.inv(warp) @ scale
    return forward / forward[2, 2]


def _mismatch(before, after, matrix, kernel):
    h, w = after.shape
    b8 = before.astype(np.uint8) * 255
    a8 = after.astype(np.uint8) * 255
    warped = (
        cv2.warpPerspective(b8, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        > 127
    )
    new_px = after & ~(cv2.dilate(warped.astype(np.uint8), kernel) > 0)
    back = (
        cv2.warpPerspective(
            cv2.dilate(a8, kernel),
            matrix,
            (before.shape[1], before.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=0,
        )
        > 127
    )
    lost_px = before & ~back
    return int(np.count_nonzero(new_px)), int(np.count_nonzero(lost_px))


def compare_canvases(before, after, candidates):
    """Canvas depois contra antes, alinhados pela melhor homografia.

    candidates são inicializações nomeadas (identidade e as
    geometrias que o replay conhece); cada uma é refinada por ECC
    sobre os próprios canvases, e vale a que deixa menos pixels
    fora da tolerância de fronteira — o benefício da dúvida é do
    sistema ("a menos de re-warp").
    """
    b = np.zeros((1, 1), bool) if before is None else before > 0
    a = np.zeros((1, 1), bool) if after is None else after > 0
    if before is not None and after is None:
        a = np.zeros_like(b)
    if after is not None and before is None:
        b = np.zeros_like(a)
    kernel = np.ones((2 * BOUNDARY_TOL_PX + 1,) * 2, np.uint8)
    out = {
        "confirmed_px_before": int(np.count_nonzero(b)),
        "confirmed_px_after": int(np.count_nonzero(a)),
    }
    out["confirmed_px_delta"] = out["confirmed_px_after"] - out["confirmed_px_before"]
    if not b.any() or not a.any():
        out.update(
            {
                "new_confirmed_px_outside_tol": int(np.count_nonzero(a)),
                "lost_confirmed_px_outside_tol": int(np.count_nonzero(b)),
                "alignment": {"method": "none (canvas vazio)"},
            }
        )
    else:
        best = None
        for name, init in candidates:
            options = [(f"{name}", init)]
            refined = _ecc_refine(b, a, init)
            if refined is not None:
                options.append((f"{name}+ecc", refined))
            for label, matrix in options:
                new_px, lost_px = _mismatch(b, a, matrix, kernel)
                key = new_px + lost_px
                if best is None or key < best[0]:
                    best = (key, label, matrix, new_px, lost_px)
        _key, label, matrix, new_px, lost_px = best
        h, w = b.shape
        frame_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(frame_pts, matrix)
        out.update(
            {
                "new_confirmed_px_outside_tol": new_px,
                "lost_confirmed_px_outside_tol": lost_px,
                "alignment": {
                    "method": label,
                    "max_corner_shift_px": round(
                        float(
                            np.max(
                                np.linalg.norm(
                                    (moved - frame_pts).reshape(-1, 2), axis=1
                                )
                            )
                        ),
                        2,
                    ),
                },
            }
        )
    out["canvas_identical_after_move"] = (
        out["new_confirmed_px_outside_tol"] == 0
        and out["lost_confirmed_px_outside_tol"] == 0
        and (out["confirmed_px_before"] > 0) == (out["confirmed_px_after"] > 0)
    )
    return out


# --------------------------------------------------------------------
# Cenários
# --------------------------------------------------------------------


def settle_drawing(
    drawing, settled, hand_config, turn_event_enabled, app_factory=ReplayApp
):
    """Desenha a sessão sob geometria parada e deixa assentar.

    Na sessão de desenho a mão está presente e se mexendo (ou sem
    sinal); ao assentar, o sinal vira o da configuração do run.
    """
    shape = settled[0].shape
    quad = base_quad(shape)
    rng = _rng("drawing", str(hand_config))
    app = app_factory(hand_config[0], turn_event_enabled)
    drawing_hand = (None, False) if hand_config[0] is None else (True, True)
    target = PASSES_PER_WARP * len(drawing)
    guard = 0
    while app.passes_total() < target:
        passes = app.passes_total()
        stage = drawing[min(passes // PASSES_PER_WARP, len(drawing) - 1)]
        app.step(
            Frame(
                stage[passes % len(stage)],
                quad,
                _noisy(quad, rng),
                hand=drawing_hand,
            )
        )
        guard += 1
        if guard > 100 * target:
            raise RuntimeError("a sessao de desenho nao avancou")
    for i in range(SETTLE_FRAMES):
        app.step(
            Frame(
                settled[i % 3],
                quad,
                _noisy(quad, rng),
                hand=_hand(hand_config, False),
            )
        )
    return app, quad


def identity_chamfer_distribution(events):
    """Distribuicoes separadas por estado ECC, nunca por decisao inferida."""
    groups = {}
    for enabled in (0, 1):
        records = [e for e in events if e["ecc_refine_enabled"] == enabled]
        metrics = {}
        keys = _ink._identity_chamfer_fields(None, None, None, {})
        for key in keys:
            if not key.startswith("chamfer_"):
                continue
            values = [e[key] for e in records if e[key] is not None]
            metrics[key] = {
                "values": values,
                "count": len(values),
                "missing": len(records) - len(values),
                **{
                    label: float(np.percentile(values, q)) if values else None
                    for label, q in (
                        ("min", 0),
                        ("p25", 25),
                        ("median", 50),
                        ("p75", 75),
                        ("max", 100),
                    )
                },
            }
        groups[str(enabled)] = {"events": len(records), "metrics": metrics}
    return groups


class Episode:
    """Mede um trecho do replay a partir de um app assentado."""

    def __init__(self, app):
        self.app = app
        self.before = None if app.canvas() is None else app.canvas().copy()
        self.anchor_before = (
            None if app.anchor() is None else app.anchor().copy()
        )
        self.warp_before = app.last_warp_corners.copy()
        self.events_before = dict(app.events)
        self.passes_before = app.passes_total()
        self.turns_before = app.coverage_turn_signals()
        self.identity_before = len(app.identity_events())
        self.det_before = (app.detector.completed, app.detector.replaced)
        self.generation_before = app.clean_generation
        self.frames = 0
        self.min_px = 0 if self.before is None else int(np.count_nonzero(self.before))
        self.passes_during = 0
        self.max_excursion = 0.0

    def step(self, frame):
        self.app.step(frame)
        self.frames += 1
        if self.app.last_warp_corners is not None:
            self.max_excursion = max(
                self.max_excursion,
                float(
                    np.max(
                        np.linalg.norm(
                            self.app.last_warp_corners - frame.true_quad, axis=1
                        )
                    )
                ),
            )
        canvas = self.app.canvas()
        px = 0 if canvas is None else int(np.count_nonzero(canvas))
        self.min_px = min(self.min_px, px)

    def mark(self):
        """Fim da perturbação (o resto é assentamento)."""
        self.passes_during = self.app.passes_total() - self.passes_before

    def report(self, true_before, true_after, identity_true_class=None):
        app = self.app
        shape = (self.before if self.before is not None else app.canvas()).shape
        events = {k: app.events[k] - self.events_before[k] for k in app.events}
        anchor_after = app.anchor()
        candidates = [("identity", np.eye(3))]
        if self.anchor_before is not None and anchor_after is not None:
            pa1 = _page_matrix(self.anchor_before, shape)
            pa2 = _page_matrix(anchor_after, shape)
            # Onde o registro na âncora põe a evidência depois do
            # movimento (a geometria nova lida como tremor).
            candidates.append(
                (
                    "registration_displacement",
                    (
                        pa2
                        @ np.linalg.inv(_page_matrix(app.last_warp_corners, shape))
                        @ _page_matrix(self.warp_before, shape)
                        @ np.linalg.inv(pa1)
                    ),
                )
            )
            # O conteúdo físico da página entre as duas âncoras.
            candidates.append(
                (
                    "page_motion",
                    (
                        pa2
                        @ np.linalg.inv(_page_matrix(true_after, shape))
                        @ _page_matrix(true_before, shape)
                        @ np.linalg.inv(pa1)
                    ),
                )
            )
        comparison = compare_canvases(self.before, app.canvas(), candidates)
        generations = app.clean_generation - self.generation_before
        passes = app.passes_total() - self.passes_before
        geometry_error = float(
            np.max(np.linalg.norm(app.last_warp_corners - true_after, axis=1))
        )
        identity = app.identity_events()[self.identity_before :]
        ecc_records = app.ecc_records()[self.passes_before :]
        ecc_ms = [record["ecc_ms"] for record in ecc_records]
        same_page = sum(1 for e in identity if e["decision"] == "same_page")
        chamfer = {}
        if _ink._identity_chamfer_enabled():
            chamfer = {
                "identity_true_class": identity_true_class,
                "identity_events": identity,
                "identity_chamfer_by_ecc": identity_chamfer_distribution(identity),
            }
        return {
            **comparison,
            **chamfer,
            "ecc_records": ecc_records,
            "ecc_ms_median": float(np.median(ecc_ms)) if ecc_ms else None,
            "ecc_ms_p95": float(np.percentile(ecc_ms, 95)) if ecc_ms else None,
            "generations_created": generations,
            # Reabertura que a identidade reconheceu como a mesma
            # página não é virada: a geração sobe, o canvas fica.
            "false_page_turns": (
                events["reopen"] - same_page + events["turn"] + events["split"]
            ),
            "identity_decisions": [e["decision"] for e in identity],
            "identity_recalls": [e["recall"] for e in identity],
            "identity_containments": [e.get("containment") for e in identity],
            "drops": events["drop"],
            "reopens": events["reopen"],
            "coverage_turns_applied": events["turn"],
            "coverage_turn_signals": (
                app.coverage_turn_signals() - self.turns_before
            ),
            "canvas_zeroed": self.min_px == 0,
            "min_confirmed_px_during": self.min_px,
            "passes_during_perturbation": self.passes_during,
            "passes_total": passes,
            "frames_total": self.frames,
            "detections_completed": (app.detector.completed - self.det_before[0]),
            "detections_replaced": app.detector.replaced - self.det_before[1],
            "vacuous": passes == 0,
            "ink_geometry_error_px": round(geometry_error, 2),
            "ink_geometry_follows_page": geometry_error <= GEOMETRY_TOL_PX,
            "max_ink_geometry_excursion_px": round(self.max_excursion, 1),
        }


def _settle(ep, settled, quad, rng, hand_config, frames=SETTLE_FRAMES):
    for i in range(frames):
        ep.step(
            Frame(
                settled[i % 3],
                quad,
                _noisy(quad, rng),
                hand=_hand(hand_config, False),
            )
        )


def run_control(base, settled, hand_config, episode_factory=Episode):
    app, quad = copy.deepcopy(base)
    rng = _rng("control", str(hand_config))
    ep = episode_factory(app)
    ep.mark()
    _settle(ep, settled, quad, rng, hand_config)
    out = ep.report(quad, quad)
    out["passes_gate"] = (
        out["canvas_identical_after_move"]
        and out["generations_created"] == 0
        and out["new_confirmed_px_outside_tol"] == 0
    )
    return out


def run_move(base, settled, hand_config, spec, episode_factory=Episode):
    app, quad = copy.deepcopy(base)
    rng = _rng("move", str(spec), str(hand_config))
    ep = episode_factory(app)
    frames = spec["frames"]
    target = transform_quad(
        quad,
        spec.get("shift", (0.0, 0.0)),
        spec.get("angle", 0.0),
        spec.get("scale", 1.0),
    )
    for i in range(1, frames + 1):
        t = i / frames
        ease = t * t * (3.0 - 2.0 * t)
        true_quad = transform_quad(
            quad,
            np.asarray(spec.get("shift", (0.0, 0.0))) * ease,
            spec.get("angle", 0.0) * ease,
            1.0 + (spec.get("scale", 1.0) - 1.0) * ease,
        )
        ep.step(
            Frame(
                settled[i % 3],
                true_quad,
                _noisy(true_quad, rng),
                hand=_hand(hand_config, True),
            )
        )
    ep.mark()
    _settle(ep, settled, target, rng, hand_config)
    out = ep.report(quad, target, identity_true_class="same_page")
    out["passes_gate"] = (
        out["canvas_identical_after_move"]
        and out["generations_created"] == 0
        and out["new_confirmed_px_outside_tol"] == 0
    )
    return out


def run_drop(base, settled, hand_config, lost_frames, mode, return_pages=None):
    """Folha perdida por N quadros e de volta.

    return_pages=None: a MESMA página volta (F-09). Com outra
    página: a identidade tem de decidir new_page e o canvas não
    pode ser herdado (F-11, guarda da correção do recall).
    """
    returned = settled if return_pages is None else return_pages
    app, quad = copy.deepcopy(base)
    rng = _rng(
        "drop",
        mode,
        str(lost_frames),
        str(hand_config),
        "turn" if return_pages is not None else "same",
    )
    ep = Episode(app)
    for i in range(lost_frames):
        page = settled[i % 3] if mode == "detector_miss" else None
        ep.step(
            Frame(
                page,
                quad,
                None,
                verdict=None,
                hand=_hand(hand_config, False),
            )
        )
    ep.mark()
    _settle(ep, returned, quad, rng, hand_config)
    out = ep.report(
        quad,
        quad,
        identity_true_class="same_page" if return_pages is None else "new_page",
    )
    out["return_kind"] = "turn" if return_pages is not None else "same"
    # Sobreviver = o acumulador não foi zerado: nenhuma reabertura
    # como página nova (zera quando a
    # identidade decide new_page). canvas_zeroed diz se o consumidor
    # ficou sem canvas durante a perda (o drop zera clean_mask nas
    # duas lógicas); canvas_identical_after_move compara o fim.
    out["canvas_survived"] = out["false_page_turns"] == 0
    if return_pages is None:
        out["passes_gate"] = (
            out["canvas_survived"]
            and out["lost_confirmed_px_outside_tol"] == 0
            and out["new_confirmed_px_outside_tol"] <= RECOVERY_GROWTH_TOL_PX
        )
    else:
        terminal_decisions = [
            decision for decision in out["identity_decisions"]
            if decision != "deferred_switch_consensus"
        ]
        # Sem drop nao ha identidade a decidir: o cenario e trivial.
        out["passes_gate"] = out["drops"] == 0 or (
            terminal_decisions == ["new_page"]
            and not out["canvas_survived"]
            and out["generations_created"] >= 1
        )
    return out


def sway_offsets(profile, frames, rng):
    """Deslocamento por canto (frames, 4, 2), pico = 314 px."""
    t = np.arange(frames) * DT
    if profile == "smooth":
        offsets = np.zeros((frames, 4, 2), np.float32)
        for corner in range(4):
            for axis in range(2):
                for _ in range(2):
                    period = rng.uniform(2.0, 5.0)
                    phase = rng.uniform(0, 2 * np.pi)
                    offsets[:, corner, axis] += np.sin(2 * np.pi * t / period + phase)
    else:
        # Biestável: o ajuste alterna entre a folha e o caderno
        # aberto (um lado inteiro do quad salta para a outra
        # página), com permanência de 1–4 s em cada modo.
        offsets = np.zeros((frames, 4, 2), np.float32)
        side = np.array([[-1, 0], [0, 0], [0, 0], [-1, 0]], np.float32)
        i, mode = 0, 0
        while i < frames:
            dwell = int(rng.uniform(1.0, 4.0) * FPS)
            if mode:
                offsets[i : i + dwell] = side
            i += dwell
            mode ^= 1
    peak = np.max(np.linalg.norm(offsets, axis=2))
    return offsets * (SWAY_AMPLITUDE_PX / max(peak, 1e-6))


def run_sway(base, settled, hand_config, profile, episode_factory=Episode):
    app, quad = copy.deepcopy(base)
    rng = _rng("sway", profile, str(hand_config))
    offsets = sway_offsets(profile, SWAY_FRAMES, rng)
    ep = episode_factory(app)
    for i in range(SWAY_FRAMES):
        ep.step(
            Frame(
                settled[i % 3],
                quad,
                _noisy(quad + offsets[i], rng),
                hand=_hand(hand_config, False),
            )
        )
    ep.mark()
    _settle(ep, settled, quad, rng, hand_config)
    out = ep.report(quad, quad)
    out["max_corner_sway_px"] = round(float(np.max(np.linalg.norm(offsets, axis=2))), 1)
    out["passes_gate"] = out["false_page_turns"] == 0
    return out


# --------------------------------------------------------------------


def _gate(runs, key="passes_gate"):
    flat = [r for r in runs if key in r]
    failed = sum(1 for r in flat if not r[key])
    return {
        "runs": len(flat),
        "failed": failed,
        "passed_vacuously": sum(1 for r in flat if r[key] and r["vacuous"]),
        "passed_with_stale_ink_geometry": sum(
            1 for r in flat if r[key] and not r["ink_geometry_follows_page"]
        ),
        "verdict": "APROVADO" if failed == 0 else "REPROVADO",
    }


def run_all(drawing, settled, source):
    results = {
        "inputs": (
            (
                "warps reais f01-f30 da mesa, probabilidades do cache da "
                "regua de 2026-11-01 (modelo v2); pagina assentada cicla "
                "f28-f30"
            )
            if source == "real"
            else (
                "pagina sintetica deterministica (10 estagios x 3 tracos, "
                "caneta e lapis, ombro sub-limiar por blur); draw-lab "
                "ausente nesta maquina"
            )
        )
        + (
            f", {PASSES_PER_WARP} passadas por warp no desenho, "
            "rastreador + geracao do app + registro + acumulador reais, "
            "unidade = passada"
        ),
        "source": source,
        "unit": "passada",
        "scene": {
            "camera_px": [CAMERA_W, CAMERA_H],
            "nominal_fps_for_tracker_clocks": FPS,
            "detector_noise_px": DETECTOR_NOISE_PX,
            "detect_latency_ms": DETECT_MS,
            "app_logic": APP_LOGIC,
            "detection_worker": "latest-wins, fila 1, um quadro em voo",
            "boundary_tol_px": BOUNDARY_TOL_PX,
            "settle_frames": SETTLE_FRAMES,
            "page_grid": list(settled[0].shape),
            "approximation": (
                "a passada re-warpa o mapa de probabilidade; o modelo "
                "nao e re-rodado no quadro re-warpado"
            ),
        },
        "settled_drawing": {},
        "control_static": {},
        "move_same_canvas": {name: {} for name in MOVES},
        "drop_and_return": {
            f"{mode}_{name}": {} for mode in DROP_MODES for name in DROP_FRAMES
        },
        "drop_and_turn": {
            f"{mode}_{name}": {} for mode in DROP_MODES for name in DROP_FRAMES
        },
        "sway_negative": {
            f"{profile}_{flag}": {}
            for profile in SWAY_PROFILES
            for flag in ("default", "turn_event_enabled")
        },
    }
    # Página de troca (conteúdo diferente, mesma grade): na fonte
    # sintética, um seed independente; na real, o espelho vertical
    # da assentada como proxy de conteúdo distinto (declarado).
    if source == "real":
        settled_turn = [np.flipud(page).copy() for page in settled]
    else:
        _turn_drawing, settled_turn = synthetic_pages(
            shape=settled[0].shape, stages=2, seed=101
        )
    results["inputs"] += "; pagina de troca: " + (
        "espelho vertical da assentada (proxy de conteudo "
        "distinto; a sessao real f01-f30 e uma pagina so)"
        if source == "real"
        else "seed sintetico independente (101)"
    )
    for hand_name, hand_config in HAND_CONFIGS.items():
        bases = {}
        for flag in (False, True):
            bases[flag] = settle_drawing(drawing, settled, hand_config, flag)
        app = bases[False][0]
        results["settled_drawing"][hand_name] = {
            "confirmed_px": int(np.count_nonzero(app.canvas())),
            "passes_total": app.passes_total(),
            "generations": app.clean_generation,
        }
        results["control_static"][hand_name] = run_control(
            bases[False], settled, hand_config
        )
        for name, spec in MOVES.items():
            results["move_same_canvas"][name][hand_name] = run_move(
                bases[False], settled, hand_config, spec
            )
        for mode in DROP_MODES:
            for name, frames in DROP_FRAMES.items():
                results["drop_and_return"][f"{mode}_{name}"][hand_name] = run_drop(
                    bases[False], settled, hand_config, frames, mode
                )
                results["drop_and_turn"][f"{mode}_{name}"][hand_name] = run_drop(
                    bases[False],
                    settled,
                    hand_config,
                    frames,
                    mode,
                    return_pages=settled_turn,
                )
        for profile in SWAY_PROFILES:
            for flag, label in ((False, "default"), (True, "turn_event_enabled")):
                results["sway_negative"][f"{profile}_{label}"][hand_name] = run_sway(
                    bases[flag], settled, hand_config, profile
                )

    def runs(section, only=None):
        return [
            r
            for key, by_hand in results[section].items()
            if only is None or key.endswith(only)
            for r in by_hand.values()
        ]

    drops = runs("drop_and_return")
    results["summary"] = {
        "control_static": _gate(list(results["control_static"].values())),
        "move_same_canvas": _gate(runs("move_same_canvas")),
        "drop_and_return_same_page": _gate(runs("drop_and_return")),
        "drop_and_turn": _gate(runs("drop_and_turn")),
        "drop_and_return": {
            "runs": len(drops),
            "canvas_survived": sum(1 for r in drops if r["canvas_survived"]),
            "generation_reset": sum(1 for r in drops if r["generations_created"] > 0),
            "identity_same_page": sum(
                r["identity_decisions"].count("same_page") for r in drops
            ),
            "identity_new_page": sum(
                r["identity_decisions"].count("new_page") for r in drops
            ),
            "verdict": "SO_REPORTE",
        },
        "sway_negative": _gate(runs("sway_negative", "_default")),
        "sway_negative_turn_event_enabled": _gate(
            runs("sway_negative", "_turn_event_enabled")
        ),
    }
    if _ink._identity_chamfer_enabled():
        episodes = (
            runs("move_same_canvas") + runs("drop_and_return") + runs("drop_and_turn")
        )
        results["identity_chamfer"] = {
            "replay_visibility": (
                "not modeled: identity uses full prior, no occlusion deferral; "
                "ECC rejection defers only the shadow_decision"
            ),
            "by_true_class": {
                label: {
                    "episodes": sum(
                        r["identity_true_class"] == label for r in episodes
                    ),
                    "by_ecc": identity_chamfer_distribution(
                        [
                            event
                            for r in episodes
                            if r["identity_true_class"] == label
                            for event in r["identity_events"]
                        ]
                    ),
                }
                for label in ("same_page", "new_page")
            },
        }
    return results


def main():
    global DETECT_MS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="forca a pagina sintetica mesmo com o draw-lab presente",
    )
    parser.add_argument(
        "--detect-ms",
        type=float,
        default=DETECT_MS,
        help="duracao simulada de uma deteccao de folha (ms)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="sai com 1 se um portao (moveu/balanco padrao) reprovar",
    )
    args = parser.parse_args()
    DETECT_MS = args.detect_ms
    if not args.synthetic and (RULER / "prob_cache.py").exists():
        drawing, settled = real_pages()
        source = "real"
    else:
        drawing, settled = synthetic_pages()
        source = "synthetic"
    results = run_all(drawing, settled, source)
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summary = results["summary"]
    failed = any(
        summary[key]["verdict"] == "REPROVADO"
        for key in (
            "control_static",
            "move_same_canvas",
            "drop_and_return_same_page",
            "drop_and_turn",
            "sway_negative",
        )
    )
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
