"""Pipeline sem interface gráfica: quadro -> folha -> tinta -> acumulador -> cursor.

Separa toda a lógica de visão, estabilização temporal, linha de base da
sessão, acumulador com carimbos e fusão da ponta da caneta da janela
OpenCV (`draw/app.py`), eliminando o warp duplicado por quadro, mantendo
cadência uniforme 1/1 para tinta/mão/cursor independentemente de
`_SHEET_DETECT_INTERVAL` e oferecendo trabalhador assíncrono opcional
para detectores neurais pesados (`--async-detect`).
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from . import config
from . import fusion as fusion_module
from . import hand as hand_module
from . import ink as ink_module
from . import neural as neural_module
from . import paper as paper_module

_SHEET_NEAR_LIMIT_PX = 40.0
_SHEET_BLEND_ALPHA = 0.3
_SHEET_JUMP_PERSIST_SECONDS = 0.3
_SHEET_DETECT_INTERVAL = 5
_SHEET_LOST_SECONDS = 0.5
_SHEET_LOST_MIN_FRAMES = 10
_INK_CORNER_HISTORY = 5
_FRESH_DELTA_MIN_PIXELS = 8


def _corner_distance(current, detected):
    return float(np.max(np.linalg.norm(current - detected, axis=1)))


def _blend_corners(current, detected):
    return (
        (1.0 - _SHEET_BLEND_ALPHA) * current
        + _SHEET_BLEND_ALPHA * detected
    ).astype(np.float32)


class _SheetTracker:
    """Estabilização temporal do quadrilátero da folha."""

    def __init__(self):
        self.corners = None
        self.verdict = None
        self.missed = False
        self._pending = None
        self._pending_hits = 0
        self._pending_since = None
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
        self._pending_since = None
        self._frames_since_seen = 0
        self._last_seen_at = now
        self.missed = False

    def update(self, detected, verdict, ran_detection, now):
        """Alimenta o rastreador com o resultado do quadro."""
        event = None
        if (
            detected is None
            and ran_detection
            and verdict == "paper"
            and self.corners is not None
        ):
            self._frames_since_seen = 0
            self._last_seen_at = now
            self.missed = True
            return None
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
                confirm_seconds = (
                    2 * _SHEET_JUMP_PERSIST_SECONDS
                    if verdict == "blank_maybe"
                    else _SHEET_JUMP_PERSIST_SECONDS
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
                    self._pending_since = now

                if (
                    self._pending_since is not None
                    and now - self._pending_since >= confirm_seconds
                ):
                    self._confirm(self._pending.copy(), verdict, now)
                    event = "seen"
                else:
                    if verdict == "paper":
                        self._frames_since_seen = 0
                        self._last_seen_at = now
                    else:
                        self._frames_since_seen += 1
                    self.missed = True
        elif self.corners is not None:
            if ran_detection:
                self._pending = None
                self._pending_hits = 0
                self._pending_since = None
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
            self._pending_since = None
            self._frames_since_seen = 0
            self.missed = False
            event = "drop"
        return event


def _detect_sheet_combined(frame, neural_detector):
    """Detecção sem marcadores: neural primário, portão de veto e reserva clássica."""
    if neural_detector is not None:
        corners, verdict = neural_detector.detect_sheet_with_confidence(frame)
        if corners is not None:
            return corners, verdict
        logit = neural_detector.last_presence_logit
        if logit is not None and logit <= neural_module.PRESENCE_VETO_LOGIT:
            return None, None
        corners, verdict = paper_module.detect_sheet_with_confidence(frame)
        if corners is None and logit is not None and logit > 0:
            return None, "paper"
        return corners, verdict
    return paper_module.detect_sheet_with_confidence(frame)


class _AsyncDetectorWorker:
    """Executa `detect_fn(frame)` em segundo plano mantendo apenas o quadro mais novo."""

    def __init__(self, detect_fn):
        self._detect_fn = detect_fn
        self._cond = threading.Condition()
        self._pending = None
        self._busy = False
        self._result = None
        self._stopped = False
        self._thread = threading.Thread(
            target=self._loop, name="draw-sheet-detector", daemon=True
        )
        self._thread.start()

    @property
    def busy(self):
        with self._cond:
            return self._busy or (self._pending is not None)

    def submit(self, frame, submitted_at):
        with self._cond:
            if self._stopped:
                return
            self._pending = (frame.copy(), float(submitted_at))
            self._cond.notify()

    def poll(self):
        with self._cond:
            res = self._result
            self._result = None
            return res

    def close(self):
        with self._cond:
            self._stopped = True
            self._pending = None
            self._cond.notify_all()
        self._thread.join(timeout=2.0)

    def _loop(self):
        while True:
            with self._cond:
                while not self._stopped and self._pending is None:
                    self._cond.wait()
                if self._stopped:
                    return
                frame, submitted_at = self._pending
                self._pending = None
                self._busy = True
            try:
                corners, verdict = self._detect_fn(frame)
            except Exception:
                corners, verdict = None, None
            with self._cond:
                self._busy = False
                if not self._stopped:
                    self._result = (corners, verdict, submitted_at)


def _fresh_ink_centroid(observed_u8, consolidated_bool):
    """Centroide da tinta observada neste quadro que ainda não consolidou."""
    fresh = (observed_u8 > 0) & (~consolidated_bool)
    moments = cv2.moments(fresh.view(np.uint8), binaryImage=True)
    m00 = moments["m00"]
    if m00 < _FRESH_DELTA_MIN_PIXELS:
        return None
    return (float(moments["m10"] / m00), float(moments["m01"] / m00))


def _select_sheet_landmark(hands_landmarks, warp_corners, delta_centroid=None):
    """Projeta o marco 8 (ponta do indicador) para o papel e aplica o portão da folha."""
    if not hands_landmarks:
        return None
    tips_cam = np.array(
        [pts[hand_module.INDEX_FINGER_TIP] for pts in hands_landmarks],
        dtype=np.float32,
    )
    tips_paper = paper_module.warp_points(tips_cam, warp_corners)
    valid = [
        (float(xy[0]), float(xy[1]))
        for xy in tips_paper
        if fusion_module.landmark_on_sheet(xy)
    ]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    ref_x, ref_y = (
        delta_centroid
        if delta_centroid is not None
        else (0.5 * config.PAPER_W, 0.5 * config.PAPER_H)
    )
    return min(
        valid, key=lambda p: (p[0] - ref_x) ** 2 + (p[1] - ref_y) ** 2
    )


@dataclass
class FrameResult:
    """Resultado completo de `DrawPipeline.process_frame`."""

    sheet_present: bool
    corners: np.ndarray | None
    verdict: str | None
    markers: dict
    canvas: np.ndarray | None
    observed_ink: np.ndarray | None
    cursor: fusion_module.CursorState
    ran_detection: bool
    tracker_event: str | None


class DrawPipeline:
    """Pipeline unificado do Draw, testável sem câmera e sem GUI."""

    def __init__(
        self,
        markers_mode=False,
        neural_detector=None,
        hand_masker=None,
        ink_extractor=None,
        session_baseline=False,
        async_detect=False,
        detect_fn=None,
    ):
        self.markers_mode = bool(markers_mode)
        self.neural_detector = neural_detector if not markers_mode else None
        self.hand_masker = hand_masker if not markers_mode else None
        self.ink_extractor = ink_extractor if not markers_mode else None
        self.session_ink = (
            ink_module.SessionBaselineInk()
            if (session_baseline and not markers_mode)
            else None
        )
        self._detect_fn = detect_fn or (
            lambda frame: _detect_sheet_combined(frame, self.neural_detector)
        )
        self._async_worker = (
            _AsyncDetectorWorker(self._detect_fn)
            if (async_detect and not markers_mode)
            else None
        )

        self.tracker = _SheetTracker()
        self.ink_accumulator = ink_module.InkAccumulator()
        self.pen_fusion = fusion_module.PenTipFusion()
        self.corner_history = deque(maxlen=_INK_CORNER_HISTORY)
        self.ink_session_open = False
        self.accumulated_canvas = None
        self.baseline_armed = False
        self._baseline_stride_count = 0
        self.session_corners = None
        self._restart_reference = False
        self._freeze_pending = False
        self.frame_number = 0
        self.last_cursor = fusion_module.CursorState(
            None, None, False, False, True
        )

    def close(self):
        if self._async_worker is not None:
            self._async_worker.close()
            self._async_worker = None

    def trigger_baseline(self):
        """Equivalente programático da tecla C (`--session-baseline`)."""
        if self.session_ink is not None:
            self._restart_reference = True
            return True
        return False

    def _ink_source(self, warped_img):
        if self.ink_extractor is not None:
            mask = self.ink_extractor.ink_mask(warped_img)
            if mask is not None:
                return mask
        return ink_module.extract_ink(warped_img)

    def process_frame(
        self, frame, t_capture_ms=None, now_s=None, canvas_buffer=None
    ):
        """Processa um quadro BGR e devolve `FrameResult`."""
        if now_s is None:
            now_s = time.monotonic()
        if t_capture_ms is None:
            t_capture_ms = int(now_s * 1000)
        t_capture_s = float(t_capture_ms) / 1000.0

        if self.markers_mode:
            warped, markers = paper_module.warp_paper(frame)
            if warped is None:
                return FrameResult(
                    sheet_present=False,
                    corners=None,
                    verdict=None,
                    markers=markers,
                    canvas=self.accumulated_canvas,
                    observed_ink=None,
                    cursor=self.last_cursor,
                    ran_detection=True,
                    tracker_event=None,
                )
            ink_mask = ink_module.extract_ink(warped, clear_markers=True)
            canvas = ink_module.render_canvas(ink_mask, out=canvas_buffer)
            self.accumulated_canvas = canvas
            return FrameResult(
                sheet_present=True,
                corners=None,
                verdict="paper",
                markers=markers,
                canvas=canvas,
                observed_ink=ink_mask,
                cursor=self.last_cursor,
                ran_detection=True,
                tracker_event="seen",
            )

        # 1. Estágio de detecção da folha (síncrono ou assíncrono).
        detected_corners, detected_verdict = None, None
        ran_detection = False
        tracker_now = now_s
        if self._async_worker is not None:
            polled = self._async_worker.poll()
            if polled is not None:
                detected_corners, detected_verdict, _submitted_at = polled
                ran_detection = True
            if self.tracker.should_detect(self.frame_number):
                if self.tracker.corners is None and not self._async_worker.busy:
                    # Primeira aquisição sem folha: submete já e coleta quando pronto.
                    self._async_worker.submit(frame, now_s)
                else:
                    self._async_worker.submit(frame, now_s)
        else:
            if self.tracker.should_detect(self.frame_number):
                detected_corners, detected_verdict = self._detect_fn(frame)
                ran_detection = True

        tracker_event = self.tracker.update(
            detected_corners, detected_verdict, ran_detection, tracker_now
        )
        if tracker_event == "drop":
            self.ink_session_open = False
            self.session_corners = None
            self._freeze_pending = False

        self.frame_number += 1

        if self.tracker.corners is None:
            return FrameResult(
                sheet_present=False,
                corners=None,
                verdict=None,
                markers={},
                canvas=self.accumulated_canvas,
                observed_ink=None,
                cursor=self.last_cursor,
                ran_detection=ran_detection,
                tracker_event=tracker_event,
            )

        # 2. Estágio de desenho: roda em TODOS os quadros com folha presente
        # (cadência uniforme 1/1 e um único warp da folha por quadro).
        if not self.ink_session_open:
            self.ink_accumulator.reset()
            self.pen_fusion = fusion_module.PenTipFusion()
            if self.session_ink is not None:
                self.session_ink.reset()
                self.baseline_armed = False
                self._baseline_stride_count = 0
                self.session_corners = None
                self._freeze_pending = False
            self.corner_history.clear()
            self.ink_session_open = True

        if ran_detection or not self.corner_history:
            self.corner_history.append(self.tracker.corners.copy())

        ink_corners = np.median(
            np.stack(self.corner_history), axis=0
        ).astype(np.float32)

        if self._restart_reference and self.session_ink is not None:
            self._restart_reference = False
            self.ink_accumulator.reset()
            self.session_ink.reset()
            self.baseline_armed = True
            self._baseline_stride_count = 0
            self.session_corners = ink_corners.copy()
            self._freeze_pending = False
        elif self._freeze_pending:
            self.session_corners = ink_corners.copy()
            self._freeze_pending = False

        warp_corners = (
            self.session_corners
            if self.session_ink is not None
            and self.session_corners is not None
            else ink_corners
        )

        homography = paper_module.perspective_matrix(warp_corners)
        ink_warped = cv2.warpPerspective(
            frame, homography, (config.PAPER_W, config.PAPER_H)
        )

        visible = None
        hands_landmarks = []
        if self.hand_masker is not None:
            camera_mask, hands_landmarks = self.hand_masker.detect(frame)
            if camera_mask is not None:
                hand_warped = cv2.warpPerspective(
                    camera_mask,
                    homography,
                    (config.PAPER_W, config.PAPER_H),
                    flags=cv2.INTER_NEAREST,
                )
                visible = hand_warped == 0

        if self.session_ink is not None:
            if not self.session_ink.ready:
                if self.baseline_armed:
                    # Espacamento das amostras da base (condicao 5
                    # da revisao do PR #9, aplicada na reconciliacao):
                    # em producao as 12 amostras eram coletadas pelo
                    # bloco de tinta, que rodava na cadencia de
                    # deteccao (1 a cada _SHEET_DETECT_INTERVAL
                    # quadros), ou seja ~2 s de cena a 30 fps. Na
                    # cadencia 1/1 deste pipeline, coletar em quadros
                    # consecutivos encolhe a janela para ~0,4 s, com
                    # amostras muito mais correlacionadas entre si
                    # (menos diversidade de micro-movimento e de mao
                    # passando). O corpo do PR declara o
                    # encolhimento como ganho, mas nenhuma medicao
                    # compara as duas janelas sobre a mesma cena;
                    # sem ela, vale a semantica de producao: uma
                    # amostra a cada _SHEET_DETECT_INTERVAL quadros.
                    if (
                        self._baseline_stride_count
                        % _SHEET_DETECT_INTERVAL
                        == 0
                    ):
                        self.session_ink.add_sample(ink_warped, visible)
                    self._baseline_stride_count += 1
                observed = np.zeros(ink_warped.shape[:2], dtype=np.uint8)
            else:
                observed = self.session_ink.observe(ink_warped, visible)
        else:
            observed = self._ink_source(ink_warped)

        # Delta fresco ANTES da histerese do acumulador alimenta PenTipFusion.
        delta_xy = _fresh_ink_centroid(observed, self.ink_accumulator.state)
        landmark_xy = _select_sheet_landmark(
            hands_landmarks, warp_corners, delta_xy
        )
        self.last_cursor = self.pen_fusion.update(
            t_capture_s,
            landmark_xy=landmark_xy,
            frame_ink_delta_centroid=delta_xy,
        )

        consolidated = self.ink_accumulator.update(
            observed, visible, t_capture_ms=t_capture_ms
        )
        self.accumulated_canvas = ink_module.render_canvas(
            consolidated, out=canvas_buffer
        )

        return FrameResult(
            sheet_present=True,
            corners=self.tracker.corners.copy(),
            verdict=self.tracker.verdict,
            markers={},
            canvas=self.accumulated_canvas,
            observed_ink=observed,
            cursor=self.last_cursor,
            ran_detection=ran_detection,
            tracker_event=tracker_event,
        )

    def save_artifacts(
        self, save_dir, canvas_override=None, ts=None,
        png_writer=None, on_png_saved=None,
    ):
        """Salva `canvas_<ts>.png` e, quando há carimbos, `canvas_<ts>_stamps.npz`."""
        canvas = (
            canvas_override
            if canvas_override is not None
            else self.accumulated_canvas
        )
        if canvas is None:
            return None, None
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time()) if ts is None else int(ts)
        png_path = save_path / f"canvas_{stamp}.png"
        if png_writer is None:
            cv2.imwrite(str(png_path), canvas)
        else:
            png_writer.save(png_path, canvas, on_png_saved)

        npz_path = None
        if not self.markers_mode and bool(self.ink_accumulator.state.any()):
            npz_path = save_path / f"canvas_{stamp}_stamps.npz"
            np.savez_compressed(
                npz_path,
                mask=self.ink_accumulator.state.view(np.uint8) * np.uint8(255),
                first_seen_ms=self.ink_accumulator.first_seen_ms(),
                last_update_ms=self.ink_accumulator.last_update_ms(),
            )
        return png_path, npz_path
