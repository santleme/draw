"""Régua de re-registro: mover o caderno = mesmo canvas? (F-07/F-09/F-11)

Mede, sem corrigir nada, se o laço do app distingue "a folha andou"
de "a folha trocou". Até 87f52e4 os dois eram o mesmo evento: `drop`
do `_SheetTracker` e a reabertura chamavam `new_generation()` em
`draw/app.py` (`--app-logic legacy`). Na main (`--app-logic main`,
padrão) o drop só deixa a identidade pendente e a reabertura chama
`new_generation(preserve_identity=True)`: a primeira passada mantém o
canvas se o recall da tinta registrada for ≥
`_PAGE_IDENTITY_MIN_RECALL`. O gatilho de cobertura (`_INK_TURN_*`)
está desligado por padrão desde o hotfix de 2026-11-01.

O replay roda em contagem de passadas, sem ONNX e sem threads, no
mesmo padrão de `tools/replay_orchestration.py`, mas dirige a
orquestração inteira e não só o acumulador:

- uma câmera simulada: a página (mapa de probabilidade na grade da
  página) está num quad VERDADEIRO na câmera; o detector devolve
  esse quad com ruído (ou o quad balançando, ou nada);
- o `_SheetTracker` real, com relógio de cena e a trabalhadora de
  detecção espelhada sem thread: latência configurável
  (`--detect-ms`, padrão 36,7 ms medidos no Ryzen), um quadro em
  voo e fila latest-wins de tamanho 1, como `_SheetDetectionWorker`;
- a lógica de geração do laço do app, espelhada linha a linha: drop
  (identidade pendente, ou `new_generation()` em legacy), reabertura
  → `new_generation(preserve_identity=...)`, mediana de
  `_INK_CORNER_HISTORY` cantos, congelamento sem mão, evento de
  virada atrás da flag;
- a trabalhadora de tinta espelhada sem thread: cadência,
  descanso total, registro na âncora (`_registration_matrix`) e o
  `_LayeredInkAccumulator` reais.

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
import copy
import importlib.util
import json
import sys
import zlib
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw.app import (  # noqa: E402
    _INK_CORNER_HISTORY,
    _LayeredInkAccumulator,
    _SheetTracker,
)
from draw import ink_layers as _ink  # noqa: E402

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
# Lógica do laço espelhada: "main" é a recuperação de identidade
# (87f52e4: drop guarda a geração, a reabertura decide pela tinta);
# "legacy" é a anterior (drop → new_generation()).
APP_LOGICS = ("main", "legacy")
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
        "shift": (120.0, 30.0), "angle": -8.0, "scale": 1.1,
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
    spec = importlib.util.spec_from_file_location(
        "prob_cache", RULER / "prob_cache.py"
    )
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
        probs.append(
            np.asarray(cache.get(img, name=f"warp:{fid}"), np.float32)
        )
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
                layer, [pts.astype(np.int32).reshape(-1, 1, 2)], False,
                1.0, int(rng.integers(2, 4)), lineType=cv2.LINE_AA,
            )
            ink = np.maximum(ink, layer * strength)
        base = np.clip(cv2.GaussianBlur(ink, (0, 0), 1.0) * 1.15, 0, 0.98)
        variants = []
        for _ in range(3):
            gain = rng.uniform(0.92, 1.04)
            noise = rng.normal(0.0, 0.03, size=shape).astype(np.float32)
            dx, dy = rng.uniform(-0.4, 0.4, size=2)
            moved = cv2.warpAffine(
                base, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                flags=cv2.INTER_LINEAR,
            )
            variants.append(np.clip(
                moved * gain + noise * (moved > 0.05), 0.0, 1.0
            ).astype(np.float32))
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
    quad = np.array([
        (cx - width / 2 + 12, cy - height / 2),
        (cx + width / 2 - 12, cy - height / 2),
        (cx + width / 2, cy + height / 2),
        (cx - width / 2, cy + height / 2),
    ], np.float32)
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
            page, matrix, (shape[1], shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=0.0,
        )
    visible = cv2.warpPerspective(
        _ONES_FRAME, _page_matrix(warp_corners, shape),
        (shape[1], shape[0]), flags=cv2.INTER_NEAREST, borderValue=0,
    )
    prob[visible == 0] = 0.0
    return prob


# --------------------------------------------------------------------
# Trabalhadora de tinta espelhada (sem thread)
# --------------------------------------------------------------------


class ReplayWorker:
    """Espelho de _PageCleanWorker._work com relógio de cena.

    Mesma cadência (_pass_interval), mesmo descanso total
    (_at_rest), mesma âncora e registro, mesmo acumulador. A passada
    é instantânea; o mapa da passada é renderizado só quando a
    passada roda, a partir do quadro submetido (latest-wins).
    """

    def __init__(self, turn_event_enabled=False):
        self.turn_event_enabled = turn_event_enabled
        self.acc = _LayeredInkAccumulator(threshold=THRESHOLD)
        self.generation = 0
        self.worker_generation = -1
        self.anchor = None
        self.pending = None
        self.last_pass_start = None
        self.published = None
        self.turn_event = None
        self.hand_state = None
        self.confirm_once = False
        self.still_passes = 0
        self.generation_passes = 0
        self.quiet_passes = 0
        self.hand_motion_at = None
        self.passes_total = 0
        self.coverage_turn_signals = 0
        self.publish_seq = 0
        self.identity_reference = None
        self.identity_reference_corners = None
        self.identity_recovery_requested = False
        self.identity_events = []
        self.ecc_reference = None
        self.ecc_records = []
        self._identity_candidate = None
        self._identity_ecc = None

    def new_generation(self, preserve_identity=False):
        self.identity_recovery_requested = bool(preserve_identity)
        if preserve_identity:
            reference = self.acc._confirmed
            self.identity_reference = (
                None if reference is None else reference.copy()
            )
            self.identity_reference_corners = (
                None if self.anchor is None else self.anchor.copy()
            )
        else:
            self.identity_reference = None
            self.identity_reference_corners = None
        self.generation += 1
        self.still_passes = 0
        self.generation_passes = 0
        self.quiet_passes = 0
        self.pending = None
        self.published = None
        self.turn_event = None
        return self.generation

    def submit(self, job, generation):
        if generation == self.generation:
            self.pending = (job, generation)

    def set_hand_presence(self, present, moved, now):
        if present is False and self.hand_state is not False:
            self.confirm_once = True
        if present is not None:
            self.still_passes = 0
        if present is not None and (moved or present != self.hand_state):
            self.hand_motion_at = now
        self.hand_state = present

    def _at_rest(self, now):
        if (
            self.generation_passes < _ink._REST_MIN_GENERATION_PASSES
            or self.quiet_passes < _ink._REST_QUIET_PASSES
        ):
            return False
        if self.hand_state is False:
            return not self.confirm_once
        if self.hand_state is True:
            return (
                self.hand_motion_at is not None
                and now - self.hand_motion_at
                >= _ink._HAND_REST_STILL_SECONDS
            )
        return False

    def _interval(self, now):
        if self._at_rest(now):
            return None
        if self.hand_state is False and not self.confirm_once:
            return _ink._IDLE_PROBE_SECONDS
        if (
            self.hand_state is None
            and self.still_passes >= _ink._NO_HAND_REST_AFTER_STILL_PASSES
        ):
            return _ink._IDLE_PROBE_SECONDS
        return _ink._PAGE_CLEAN_INTERVAL_SECONDS

    def tick(self, now):
        """Roda a passada se a cadência e o descanso deixarem."""
        if self.pending is None:
            return None
        if self.last_pass_start is not None:
            interval = self._interval(now)
            if interval is None or now - self.last_pass_start < interval - 1e-9:
                return None
        (page, true_quad, corners), generation = self.pending
        self.pending = None
        self.last_pass_start = now
        if self.confirm_once:
            self.confirm_once = False
        prob = render_pass(page, true_quad, corners)
        first_pass = generation != self.worker_generation
        if first_pass:
            self.worker_generation = generation
            self.generation_passes = 0
            self.quiet_passes = 0
            if not self._identity_keeps_page(prob, corners, generation):
                self.acc.reset()
                self.anchor = corners
                self.ecc_reference = None
        if first_pass and self._identity_candidate is not None:
            prob = self._identity_candidate
            ecc_info = self._identity_ecc
        else:
            if first_pass and self._identity_ecc is not None:
                ecc_info = self._identity_ecc
            else:
                prob, _visible, ecc_info = _ink._register_ink_probability(
                    prob, self.anchor, corners, self.ecc_reference
                )
        self.ecc_records.append(ecc_info)
        if self.ecc_reference is None or self.ecc_reference.shape != prob.shape:
            self.ecc_reference = prob.copy()
        else:
            np.maximum(self.ecc_reference, prob, out=self.ecc_reference)
        result = self.acc.add(prob)
        if (
            result["confirmed_changed"]
            or result["provisional_changed"]
            or result["turn"]
        ):
            self.still_passes = 0
        else:
            self.still_passes += 1
        self.generation_passes += 1
        if result["confirmed_changed"] or result["turn"]:
            self.quiet_passes = 0
        else:
            self.quiet_passes += 1
        self.passes_total += 1
        if result["turn"]:
            self.coverage_turn_signals += 1
            if self.turn_event_enabled and self.turn_event is None:
                self.turn_event = (generation, result["confirmed"])
        if first_pass or result["confirmed_changed"]:
            self.published = (generation, result["confirmed"])
            self.publish_seq += 1
        return result

    def _identity_keeps_page(self, prob, corners, generation):
        """Decisão de identidade da primeira passada (ink_layers)."""
        reference = self.identity_reference
        reference_corners = self.identity_reference_corners
        requested = self.identity_recovery_requested
        self.identity_reference = None
        self.identity_reference_corners = None
        self.identity_recovery_requested = False
        keep_page = False
        recall = None
        containment = None
        self._identity_candidate = None
        self._identity_ecc = None
        if reference is not None:
            candidate, _visible, self._identity_ecc = _ink._register_ink_probability(
                prob, reference_corners, corners, self.ecc_reference
            )
            prior = reference > 0
            prior_px = int(np.count_nonzero(prior))
            seen = candidate > self.acc.threshold
            if prior_px == 0:
                keep_page = not np.any(seen)
                recall = 1.0 if keep_page else 0.0
                containment = 1.0 if keep_page else 0.0
            else:
                kernel = np.ones(
                    (_ink._IDENTITY_DILATION_KERNEL,) * 2, np.uint8
                )
                seen_d = cv2.dilate(
                    seen.astype(np.uint8), kernel
                ) > 0
                prior_d = cv2.dilate(
                    prior.astype(np.uint8), kernel
                ) > 0
                recall = np.count_nonzero(prior & seen_d) / prior_px
                seen_px = int(np.count_nonzero(seen))
                containment = (
                    0.0
                    if seen_px == 0
                    else np.count_nonzero(seen & prior_d) / seen_px
                )
                keep_page = (
                    recall >= _ink._PAGE_IDENTITY_MIN_RECALL
                    and containment >= _ink._PAGE_IDENTITY_MIN_CONTAINMENT
                )
            if keep_page:
                self._identity_candidate = candidate
        if requested:
            self.identity_events.append({
                "generation": generation,
                "decision": "same_page" if keep_page else "new_page",
                "recall": None if recall is None else round(float(recall), 4),
                "containment": (
                    None if containment is None
                    else round(float(containment), 4)
                ),
                **(self._identity_ecc or {}),
            })
        return keep_page

    def take_turn_event(self, generation):
        if self.turn_event is None or self.turn_event[0] != generation:
            return None
        event, self.turn_event = self.turn_event, None
        return event[1]


# --------------------------------------------------------------------
# Laço do app espelhado (folha + geração + tinta)
# --------------------------------------------------------------------


class ReplayDetector:
    """`_SheetDetectionWorker` sem thread, em relógio de cena.

    Uma detecção em voo por vez, que termina `latency` segundos
    depois de começar; um quadro submetido com a trabalhadora
    ocupada espera numa fila de tamanho 1 e é substituído pelo
    próximo (latest-wins). poll() devolve só o resultado mais novo,
    com o carimbo de captura do quadro detectado.
    """

    def __init__(self, latency):
        self.latency = latency
        self.inflight = None
        self.pending = None
        self.result = None
        self.seq = 0
        self.submitted = 0
        self.completed = 0
        self.replaced = 0

    def _advance(self, now):
        while self.inflight is not None and self.inflight[3] <= now + 1e-9:
            corners, verdict, captured_at, done_at = self.inflight
            self.inflight = None
            self.seq += 1
            self.completed += 1
            self.result = (self.seq, corners, verdict, captured_at)
            if self.pending is not None:
                corners, verdict, captured_at = self.pending
                self.pending = None
                self.inflight = (
                    corners, verdict, captured_at,
                    max(done_at, captured_at) + self.latency,
                )

    def poll(self, now, after_seq):
        self._advance(now)
        if self.result is None or self.result[0] <= after_seq:
            return None
        return self.result

    def submit(self, corners, verdict, now):
        self._advance(now)
        self.submitted += 1
        if self.inflight is None:
            self.inflight = (corners, verdict, now, now + self.latency)
            return
        if self.pending is not None:
            self.replaced += 1
        self.pending = (corners, verdict, now)


class Frame:
    """O que a cena entrega num quadro."""

    def __init__(self, page, true_quad, detected, verdict="paper",
                 hand=(None, False)):
        self.page = page
        self.true_quad = true_quad
        self.detected = detected
        self.verdict = verdict
        self.hand = hand


class ReplayApp:
    """Espelho do caminho page-clean do laço de draw/app.py::run.

    Só o que decide geometria, sessão e geração: o resto do laço
    (exibição, gravação, telemetria) não toca o canvas confirmado.
    """

    def __init__(self, hand_state, turn_event_enabled=False,
                 detect_ms=None, app_logic=None):
        self.hand_state = hand_state
        self.app_logic = APP_LOGIC if app_logic is None else app_logic
        if self.app_logic not in APP_LOGICS:
            raise ValueError(f"app_logic desconhecida: {self.app_logic}")
        self.recovery_pending = False
        self.mask_cleared = False
        self.mask_cleared_at_seq = 0
        self.tracker = _SheetTracker()
        self.detector = ReplayDetector(
            (DETECT_MS if detect_ms is None else detect_ms) / 1000.0
        )
        self.last_det_seq = 0
        self.worker = ReplayWorker(turn_event_enabled)
        self.corner_history = deque(maxlen=_INK_CORNER_HISTORY)
        self.ink_session_open = False
        self.clean_generation = 0
        self.frame_number = 0
        self.frame_index = 0
        self.events = {"drop": 0, "reopen": 0, "turn": 0}
        self.last_warp_corners = None

    def _new_generation(self, reason, preserve_identity=False):
        self.events[reason] += 1
        self.clean_generation = self.worker.new_generation(
            preserve_identity=preserve_identity
        )

    def canvas(self):
        """O canvas dos consumidores (clean_mask do app).

        O drop zera clean_mask no app nas duas lógicas; ele só volta
        quando a trabalhadora publica de novo.
        """
        published = self.worker.published
        if published is None or published[0] != self.clean_generation:
            return None
        if self.mask_cleared:
            if self.worker.publish_seq == self.mask_cleared_at_seq:
                return None
            self.mask_cleared = False
        return published[1]

    def step(self, frame):
        now = self.frame_index * DT
        det_result = self.detector.poll(now, self.last_det_seq)
        if det_result is not None:
            self.last_det_seq, detected, verdict, captured_at = det_result
            ran_detection = True
        else:
            detected, verdict, captured_at = None, None, None
            ran_detection = False
        if self.tracker.should_detect(self.frame_number):
            corners = (
                None if frame.detected is None
                else np.asarray(frame.detected, np.float32).copy()
            )
            self.detector.submit(corners, frame.verdict, now)
        event = self.tracker.update(
            detected, verdict, ran_detection,
            captured_at if ran_detection else now,
        )
        if event == "drop":
            self.ink_session_open = False
            self.mask_cleared = True
            self.mask_cleared_at_seq = self.worker.publish_seq
            if self.app_logic == "legacy":
                self._new_generation("drop")
            else:
                # 87f52e4: a queda não prova página nova; a primeira
                # passada da reabertura decide pela tinta.
                self.events["drop"] += 1
                self.recovery_pending = True
        self.frame_number += 1
        hand_present = None
        if self.hand_state is not None:
            hand_present, moved = frame.hand
            if ran_detection and self.tracker.corners is not None:
                self.worker.set_hand_presence(hand_present, moved, now)
            hand_present = self.worker.hand_state
        if ran_detection and self.tracker.corners is not None:
            if not self.ink_session_open:
                self._new_generation(
                    "reopen", preserve_identity=self.recovery_pending
                )
                self.recovery_pending = False
                self.corner_history.clear()
                self.ink_session_open = True
            if hand_present is not False or not self.corner_history:
                self.corner_history.append(self.tracker.corners.copy())
            ink_corners = np.median(
                np.stack(self.corner_history), axis=0
            ).astype(np.float32)
            self.last_warp_corners = ink_corners
            page = (
                frame.page if frame.page is not None
                else np.zeros_like(self._page_shape_hint)
            )
            self.worker.submit(
                (page, frame.true_quad, ink_corners), self.clean_generation
            )
        if frame.page is not None:
            self._page_shape_hint = frame.page
        result = self.worker.tick(now)
        if self.worker.take_turn_event(self.clean_generation) is not None:
            self._new_generation("turn")
        self.frame_index += 1
        return result


def _hand(config, moving):
    """Sinal de mão do quadro para a configuração do run."""
    state, policy = config
    if state is not True:
        return (state, False)
    return (True, policy == "always" or (policy == "move" and moving))


def _noisy(quad, rng, sigma=DETECTOR_NOISE_PX):
    return (quad + rng.normal(0.0, sigma, size=quad.shape)).astype(
        np.float32
    )


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
        img = cv2.resize(
            mask.astype(np.float32), size, interpolation=cv2.INTER_AREA
        )
        return cv2.GaussianBlur(img, (0, 0), 3.0)

    template, moving = prep(after), prep(before)
    small = scale @ init @ np.linalg.inv(scale)
    warp = np.linalg.inv(small).astype(np.float32)
    warp /= warp[2, 2]
    try:
        _cc, warp = cv2.findTransformECC(
            template, moving, warp, cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            None, 5,
        )
    except cv2.error:
        return None
    forward = np.linalg.inv(scale) @ np.linalg.inv(warp) @ scale
    return forward / forward[2, 2]


def _mismatch(before, after, matrix, kernel):
    h, w = after.shape
    b8 = before.astype(np.uint8) * 255
    a8 = after.astype(np.uint8) * 255
    warped = cv2.warpPerspective(
        b8, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    ) > 127
    new_px = after & ~(cv2.dilate(warped.astype(np.uint8), kernel) > 0)
    back = cv2.warpPerspective(
        cv2.dilate(a8, kernel), matrix, (before.shape[1], before.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0,
    ) > 127
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
    out["confirmed_px_delta"] = (
        out["confirmed_px_after"] - out["confirmed_px_before"]
    )
    if not b.any() or not a.any():
        out.update({
            "new_confirmed_px_outside_tol": int(np.count_nonzero(a)),
            "lost_confirmed_px_outside_tol": int(np.count_nonzero(b)),
            "alignment": {"method": "none (canvas vazio)"},
        })
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
        frame_pts = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]
        ).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(frame_pts, matrix)
        out.update({
            "new_confirmed_px_outside_tol": new_px,
            "lost_confirmed_px_outside_tol": lost_px,
            "alignment": {
                "method": label,
                "max_corner_shift_px": round(float(np.max(
                    np.linalg.norm((moved - frame_pts).reshape(-1, 2), axis=1)
                )), 2),
            },
        })
    out["canvas_identical_after_move"] = (
        out["new_confirmed_px_outside_tol"] == 0
        and out["lost_confirmed_px_outside_tol"] == 0
        and (out["confirmed_px_before"] > 0) == (out["confirmed_px_after"] > 0)
    )
    return out


# --------------------------------------------------------------------
# Cenários
# --------------------------------------------------------------------


def settle_drawing(drawing, settled, hand_config, turn_event_enabled):
    """Desenha a sessão sob geometria parada e deixa assentar.

    Na sessão de desenho a mão está presente e se mexendo (ou sem
    sinal); ao assentar, o sinal vira o da configuração do run.
    """
    shape = settled[0].shape
    quad = base_quad(shape)
    rng = _rng("drawing", str(hand_config))
    app = ReplayApp(hand_config[0], turn_event_enabled)
    drawing_hand = (None, False) if hand_config[0] is None else (True, True)
    target = PASSES_PER_WARP * len(drawing)
    guard = 0
    while app.worker.passes_total < target:
        passes = app.worker.passes_total
        stage = drawing[min(passes // PASSES_PER_WARP, len(drawing) - 1)]
        app.step(Frame(
            stage[passes % len(stage)], quad, _noisy(quad, rng),
            hand=drawing_hand,
        ))
        guard += 1
        if guard > 100 * target:
            raise RuntimeError("a sessao de desenho nao avancou")
    for i in range(SETTLE_FRAMES):
        app.step(Frame(
            settled[i % 3], quad, _noisy(quad, rng),
            hand=_hand(hand_config, False),
        ))
    return app, quad


class Episode:
    """Mede um trecho do replay a partir de um app assentado."""

    def __init__(self, app):
        self.app = app
        self.before = None if app.canvas() is None else app.canvas().copy()
        self.anchor_before = (
            None if app.worker.anchor is None else app.worker.anchor.copy()
        )
        self.warp_before = app.last_warp_corners.copy()
        self.events_before = dict(app.events)
        self.passes_before = app.worker.passes_total
        self.turns_before = app.worker.coverage_turn_signals
        self.identity_before = len(app.worker.identity_events)
        self.det_before = (
            app.detector.completed, app.detector.replaced
        )
        self.generation_before = app.clean_generation
        self.frames = 0
        self.min_px = (
            0 if self.before is None else int(np.count_nonzero(self.before))
        )
        self.passes_during = 0
        self.max_excursion = 0.0

    def step(self, frame):
        self.app.step(frame)
        self.frames += 1
        if self.app.last_warp_corners is not None:
            self.max_excursion = max(self.max_excursion, float(np.max(
                np.linalg.norm(
                    self.app.last_warp_corners - frame.true_quad, axis=1
                )
            )))
        canvas = self.app.canvas()
        px = 0 if canvas is None else int(np.count_nonzero(canvas))
        self.min_px = min(self.min_px, px)

    def mark(self):
        """Fim da perturbação (o resto é assentamento)."""
        self.passes_during = self.app.worker.passes_total - self.passes_before

    def report(self, true_before, true_after):
        app = self.app
        shape = (self.before if self.before is not None else app.canvas()).shape
        events = {
            k: app.events[k] - self.events_before[k] for k in app.events
        }
        anchor_after = app.worker.anchor
        candidates = [("identity", np.eye(3))]
        if self.anchor_before is not None and anchor_after is not None:
            pa1 = _page_matrix(self.anchor_before, shape)
            pa2 = _page_matrix(anchor_after, shape)
            # Onde o registro na âncora põe a evidência depois do
            # movimento (a geometria nova lida como tremor).
            candidates.append(("registration_displacement", (
                pa2 @ np.linalg.inv(_page_matrix(app.last_warp_corners, shape))
                @ _page_matrix(self.warp_before, shape) @ np.linalg.inv(pa1)
            )))
            # O conteúdo físico da página entre as duas âncoras.
            candidates.append(("page_motion", (
                pa2 @ np.linalg.inv(_page_matrix(true_after, shape))
                @ _page_matrix(true_before, shape) @ np.linalg.inv(pa1)
            )))
        comparison = compare_canvases(self.before, app.canvas(), candidates)
        generations = app.clean_generation - self.generation_before
        passes = app.worker.passes_total - self.passes_before
        geometry_error = float(np.max(np.linalg.norm(
            app.last_warp_corners - true_after, axis=1
        )))
        identity = app.worker.identity_events[self.identity_before:]
        ecc_records = app.worker.ecc_records[self.passes_before:]
        ecc_ms = [record["ecc_ms"] for record in ecc_records]
        same_page = sum(1 for e in identity if e["decision"] == "same_page")
        return {
            **comparison,
            "ecc_records": ecc_records,
            "ecc_ms_median": float(np.median(ecc_ms)) if ecc_ms else None,
            "ecc_ms_p95": float(np.percentile(ecc_ms, 95)) if ecc_ms else None,
            "generations_created": generations,
            # Reabertura que a identidade reconheceu como a mesma
            # página não é virada: a geração sobe, o canvas fica.
            "false_page_turns": events["reopen"] - same_page + events["turn"],
            "identity_decisions": [e["decision"] for e in identity],
            "identity_recalls": [e["recall"] for e in identity],
            "identity_containments": [
                e.get("containment") for e in identity
            ],
            "drops": events["drop"],
            "reopens": events["reopen"],
            "coverage_turns_applied": events["turn"],
            "coverage_turn_signals": (
                app.worker.coverage_turn_signals - self.turns_before
            ),
            "canvas_zeroed": self.min_px == 0,
            "min_confirmed_px_during": self.min_px,
            "passes_during_perturbation": self.passes_during,
            "passes_total": passes,
            "frames_total": self.frames,
            "detections_completed": (
                app.detector.completed - self.det_before[0]
            ),
            "detections_replaced": app.detector.replaced - self.det_before[1],
            "vacuous": passes == 0,
            "ink_geometry_error_px": round(geometry_error, 2),
            "ink_geometry_follows_page": geometry_error <= GEOMETRY_TOL_PX,
            "max_ink_geometry_excursion_px": round(self.max_excursion, 1),
        }


def _settle(ep, settled, quad, rng, hand_config, frames=SETTLE_FRAMES):
    for i in range(frames):
        ep.step(Frame(
            settled[i % 3], quad, _noisy(quad, rng),
            hand=_hand(hand_config, False),
        ))


def run_control(base, settled, hand_config):
    app, quad = copy.deepcopy(base)
    rng = _rng("control", str(hand_config))
    ep = Episode(app)
    ep.mark()
    _settle(ep, settled, quad, rng, hand_config)
    out = ep.report(quad, quad)
    out["passes_gate"] = (
        out["canvas_identical_after_move"]
        and out["generations_created"] == 0
        and out["new_confirmed_px_outside_tol"] == 0
    )
    return out


def run_move(base, settled, hand_config, spec):
    app, quad = copy.deepcopy(base)
    rng = _rng("move", str(spec), str(hand_config))
    ep = Episode(app)
    frames = spec["frames"]
    target = transform_quad(
        quad, spec.get("shift", (0.0, 0.0)), spec.get("angle", 0.0),
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
        ep.step(Frame(
            settled[i % 3], true_quad, _noisy(true_quad, rng),
            hand=_hand(hand_config, True),
        ))
    ep.mark()
    _settle(ep, settled, target, rng, hand_config)
    out = ep.report(quad, target)
    out["passes_gate"] = (
        out["canvas_identical_after_move"]
        and out["generations_created"] == 0
        and out["new_confirmed_px_outside_tol"] == 0
    )
    return out


def run_drop(base, settled, hand_config, lost_frames, mode,
             return_pages=None):
    """Folha perdida por N quadros e de volta.

    return_pages=None: a MESMA página volta (F-09). Com outra
    página: a identidade tem de decidir new_page e o canvas não
    pode ser herdado (F-11, guarda da correção do recall).
    """
    returned = settled if return_pages is None else return_pages
    app, quad = copy.deepcopy(base)
    rng = _rng(
        "drop", mode, str(lost_frames), str(hand_config),
        "turn" if return_pages is not None else "same",
    )
    ep = Episode(app)
    for i in range(lost_frames):
        page = settled[i % 3] if mode == "detector_miss" else None
        ep.step(Frame(
            page, quad, None, verdict=None,
            hand=_hand(hand_config, False),
        ))
    ep.mark()
    _settle(ep, returned, quad, rng, hand_config)
    out = ep.report(quad, quad)
    out["return_kind"] = "turn" if return_pages is not None else "same"
    # Sobreviver = o acumulador não foi zerado: nenhuma reabertura
    # como página nova (legacy sempre zera; main zera quando a
    # identidade decide new_page). canvas_zeroed diz se o consumidor
    # ficou sem canvas durante a perda (o drop zera clean_mask nas
    # duas lógicas); canvas_identical_after_move compara o fim.
    out["canvas_survived"] = out["false_page_turns"] == 0
    if return_pages is None:
        out["passes_gate"] = (
            out["canvas_survived"]
            and out["lost_confirmed_px_outside_tol"] == 0
            and out["new_confirmed_px_outside_tol"]
            <= RECOVERY_GROWTH_TOL_PX
        )
    else:
        # Sem drop nao ha identidade a decidir: o cenario e trivial.
        out["passes_gate"] = out["drops"] == 0 or (
            out["identity_decisions"] == ["new_page"]
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
                    offsets[:, corner, axis] += np.sin(
                        2 * np.pi * t / period + phase
                    )
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
                offsets[i:i + dwell] = side
            i += dwell
            mode ^= 1
    peak = np.max(np.linalg.norm(offsets, axis=2))
    return offsets * (SWAY_AMPLITUDE_PX / max(peak, 1e-6))


def run_sway(base, settled, hand_config, profile):
    app, quad = copy.deepcopy(base)
    rng = _rng("sway", profile, str(hand_config))
    offsets = sway_offsets(profile, SWAY_FRAMES, rng)
    ep = Episode(app)
    for i in range(SWAY_FRAMES):
        ep.step(Frame(
            settled[i % 3], quad, _noisy(quad + offsets[i], rng),
            hand=_hand(hand_config, False),
        ))
    ep.mark()
    _settle(ep, settled, quad, rng, hand_config)
    out = ep.report(quad, quad)
    out["max_corner_sway_px"] = round(
        float(np.max(np.linalg.norm(offsets, axis=2))), 1
    )
    out["passes_gate"] = out["false_page_turns"] == 0
    return out


# --------------------------------------------------------------------


def _gate(runs, key="passes_gate"):
    flat = [r for r in runs if key in r]
    failed = sum(1 for r in flat if not r[key])
    return {
        "runs": len(flat),
        "failed": failed,
        "passed_vacuously": sum(
            1 for r in flat if r[key] and r["vacuous"]
        ),
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
            ) if source == "real" else (
                "pagina sintetica deterministica (10 estagios x 3 tracos, "
                "caneta e lapis, ombro sub-limiar por blur); draw-lab "
                "ausente nesta maquina"
            )
        ) + (
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
    results["inputs"] += (
        "; pagina de troca: "
        + (
            "espelho vertical da assentada (proxy de conteudo "
            "distinto; a sessao real f01-f30 e uma pagina so)"
            if source == "real" else "seed sintetico independente (101)"
        )
    )
    for hand_name, hand_config in HAND_CONFIGS.items():
        bases = {}
        for flag in (False, True):
            bases[flag] = settle_drawing(drawing, settled, hand_config, flag)
        app = bases[False][0]
        results["settled_drawing"][hand_name] = {
            "confirmed_px": int(np.count_nonzero(app.canvas())),
            "passes_total": app.worker.passes_total,
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
                results["drop_and_return"][f"{mode}_{name}"][hand_name] = (
                    run_drop(bases[False], settled, hand_config, frames, mode)
                )
                results["drop_and_turn"][f"{mode}_{name}"][hand_name] = (
                    run_drop(
                        bases[False], settled, hand_config, frames, mode,
                        return_pages=settled_turn,
                    )
                )
        for profile in SWAY_PROFILES:
            for flag, label in ((False, "default"), (True, "turn_event_enabled")):
                results["sway_negative"][f"{profile}_{label}"][hand_name] = (
                    run_sway(bases[flag], settled, hand_config, profile)
                )

    def runs(section, only=None):
        return [
            r for key, by_hand in results[section].items()
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
            "generation_reset": sum(
                1 for r in drops if r["generations_created"] > 0
            ),
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
    return results


def main():
    global DETECT_MS, APP_LOGIC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--synthetic", action="store_true",
        help="forca a pagina sintetica mesmo com o draw-lab presente",
    )
    parser.add_argument(
        "--detect-ms", type=float, default=DETECT_MS,
        help="duracao simulada de uma deteccao de folha (ms)",
    )
    parser.add_argument(
        "--app-logic", choices=APP_LOGICS, default=APP_LOGIC,
        help="main = recuperacao de identidade; legacy = drop zera",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="sai com 1 se um portao (moveu/balanco padrao) reprovar",
    )
    args = parser.parse_args()
    DETECT_MS = args.detect_ms
    APP_LOGIC = args.app_logic
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
            "control_static", "move_same_canvas", "drop_and_return_same_page",
            "drop_and_turn", "sway_negative",
        )
    )
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
