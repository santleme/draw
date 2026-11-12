"""Câmera sintética lazy e exportador MP4 + ground truth da escrita.

Usa as primitivas de compose_scene_sequence, mas mantém a pose fixa e
revela também frações de traço. Não chama a sequência eager (que escolhe
seu próprio plano aleatório) nem modifica o gerador de dados de treino.

Cenário opcional escrever→apagar→redesenhar→mover: ``erase_stroke`` ou
``erase_rect`` remove tinta da página (borracha visível varrendo a
região), ``redraw_text`` escreve de novo sobre a área limpa e ``move``
desliza/gira a folha. O GT declara cada fase: traços marcados com
``stroke_phase`` 0/1 e ``stroke_erased``, ``page_ink_final`` com o
estado real do fim e ``events`` com os limites de frame de cada fase.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import gen_synthetic as gs
from tools.sim_text import plan_text_strokes


class SyntheticCapture:
    """Subset de VideoCapture: BGR uint8, EOF finito, sem fila de frames.

    ``realtime`` limita a entrega ao FPS no relógio de parede. Desligado
    permite exportação/avaliação offline rápida; a cena usa sempre tempo
    de frame determinístico. Não usa GT para decidir o que foi capturado.
    """

    def __init__(
        self,
        text: str,
        *,
        seed: int = 42,
        fps: float = 15,
        stroke_seconds: float = 0.2,
        warmup_seconds: float = 1,
        settle_seconds: float = 3,
        height: float = 22,
        hand: bool = False,
        realtime: bool = False,
        erase_stroke: int | None = None,
        erase_rect: tuple[int, int, int, int] | None = None,
        redraw_text: str | None = None,
        erase_seconds: float = 1.0,
        hold_seconds: float = 4.0,
        move: tuple[float, float, float] | None = None,
        move_seconds: float = 0.8,
        lift: tuple[float, float] | None = None,
        return_pose: tuple[float, float, float] | None = None,
        absent_seconds: float = 2.0,
        lift_seconds: float = 0.5,
        return_seconds: float = 0.5,
    ):
        if not math.isfinite(fps) or not 0 < fps <= 120:
            raise ValueError("fps deve estar entre 0 e 120")
        if not math.isfinite(stroke_seconds) or stroke_seconds <= 0:
            raise ValueError("stroke_seconds deve ser positivo e finito")
        if any(not math.isfinite(v) or v < 0 for v in (warmup_seconds, settle_seconds)):
            raise ValueError("warmup/settle devem ser finitos e não negativos")
        if erase_stroke is not None and erase_rect is not None:
            raise ValueError("erase_stroke e erase_rect são alternativas, não combinam")
        if redraw_text is not None and erase_stroke is None and erase_rect is None:
            raise ValueError(
                "redraw_text exige uma região apagada (erase_stroke ou erase_rect)"
            )
        for name, value in (
            ("erase_seconds", erase_seconds),
            ("hold_seconds", hold_seconds),
            ("move_seconds", move_seconds),
            ("absent_seconds", absent_seconds),
            ("lift_seconds", lift_seconds),
            ("return_seconds", return_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} deve ser finito e não negativo")
        self.fps = float(fps)
        self.seed, self.text = seed, text
        self.realtime = realtime
        self._started = None
        self._opened = True
        self._position = 0
        self._completed = 0
        self._rng = gs._scene_rng(seed, "simulator", 0)
        self._background = gs._background(self._rng, "table_wood")
        self._paper, self.pw, self.ph = gs._paper_base(self._rng, "white", size_idx=0)
        self._paper_pristine = self._paper.copy()
        self.specs = plan_text_strokes(text, self._rng, self.pw, self.ph, height=height)
        self.quad = gs._sample_normal_quad(self._rng, self.pw, self.ph)
        # O gerador de treino admite folha cortada. A câmera inicial do
        # simulador mantém a página inteira visível para permitir aquisição
        # sem entregar sua geometria ao detector. Pequena perturbação dos
        # cantos cria perspectiva, além da rotação do gerador original.
        self.quad += self._rng.uniform(-4, 4, (4, 2)).astype(np.float32)
        center = self.quad.mean(axis=0)
        scale = min(1.0, (gs.SIZE - 48) / float(np.ptp(self.quad, axis=0).max()))
        self.quad = (self.quad - center) * scale + center
        self.quad += np.maximum(24 - self.quad.min(axis=0), 0)
        self.quad -= np.maximum(self.quad.max(axis=0) - (gs.SIZE - 24), 0)
        self._quad_now = self.quad.copy()
        self._ink = np.zeros((self.ph, self.pw), np.uint8)
        self._page_corners = np.float32(
            [[0, 0], [self.pw, 0], [self.pw, self.ph], [0, self.ph]]
        )
        self.page_to_scene = cv2.getPerspectiveTransform(self._page_corners, self.quad)
        self._page_to_scene_init = self.page_to_scene.copy()
        self.warmup_frames = max(1, math.ceil(warmup_seconds * fps))
        self.frames_per_stroke = max(1, math.ceil(stroke_seconds * fps))
        self.writing_frames = len(self.specs) * self.frames_per_stroke

        # --- cenário apagar → redesenhar → mover -------------------------
        self.erase_rect = self._resolve_erase_rect(erase_stroke, erase_rect)
        self.stroke_erased = np.zeros(len(self.specs), dtype=bool)
        if self.erase_rect is not None:
            x, y, w, h = self.erase_rect
            for index, spec in enumerate(self.specs):
                if _polyline_rect_overlap(spec[3], (x, y, w, h)):
                    self.stroke_erased[index] = True
        self.redraw_specs = []
        if redraw_text:
            specs = plan_text_strokes(
                redraw_text, self._rng, self.pw, self.ph, height=height
            )
            self.redraw_specs = _translate_specs(
                specs, self.erase_rect, self.pw, self.ph
            )
        self.specs_all = [*self.specs, *self.redraw_specs]
        self.stroke_phase = np.array(
            [0] * len(self.specs) + [1] * len(self.redraw_specs), dtype=np.int32
        )
        move = tuple(move) if move is not None else None
        if move is not None and (
            len(move) != 3 or not all(math.isfinite(v) for v in move)
        ):
            raise ValueError("move espera (dx, dy, graus) finitos no espaço da cena")

        # lift & return: a folha sai de cena por inteiro (lift_out),
        # some por absent_seconds (só mesa — a tinta vai junto, fica
        # na folha) e volta deslizando na pose `return_pose` relativa
        # à pose de quando saiu. É o gesto mais comum da mesa real —
        # pegar a folha, virar e repousar — e o que mais estressa a
        # vigia de identidade: mesma página ou página nova?
        lift_off = None
        quad_at_lift = None
        if lift is not None:
            if len(lift) != 2 or not all(math.isfinite(v) for v in lift):
                raise ValueError("lift espera (dx, dy) finitos na cena")
            direction = np.float32(lift)
            norm = float(np.linalg.norm(direction))
            if norm < 1.0:
                raise ValueError("lift precisa de direção não trivial")
            # A pose de saída é a pose APÓS as fases de geometria que
            # vieram antes na linha do tempo (move): _quad_now em
            # __init__ ainda é a pose inicial — o move só aplica na
            # reprodução — então a composição é calculada aqui.
            quad_at_lift = self._quad_now.copy()
            if move is not None:
                mdx, mdy, mdeg = move
                quad_at_lift = quad_at_lift + np.float32([mdx, mdy])
                _c = quad_at_lift.mean(axis=0)
                _rad = math.radians(mdeg)
                _rot = np.float32(
                    [
                        [math.cos(_rad), -math.sin(_rad)],
                        [math.sin(_rad), math.cos(_rad)],
                    ]
                )
                quad_at_lift = (quad_at_lift - _c) @ _rot.T + _c
            # Deslocamento maior que o lado do frame garante saída
            # completa para qualquer quad.
            lift_off = quad_at_lift + direction / norm * (gs.SIZE + 400)
        if return_pose is not None:
            if lift is None:
                raise ValueError("return_pose exige lift")
            if len(return_pose) != 3 or not all(math.isfinite(v) for v in return_pose):
                raise ValueError("return_pose espera (dx, dy, graus) finitos na cena")

        # Linha do tempo em fases; sem cenário é exatamente
        # warmup → escrita → settle, como sempre foi.
        hold_frames = max(0, math.ceil(hold_seconds * fps))
        self.phases = []
        start = 0

        def push(kind, frames, **data):
            nonlocal start
            if frames <= 0:
                return
            self.phases.append(
                {"kind": kind, "start": start, "end": start + frames, **data}
            )
            start += frames

        push("warmup", self.warmup_frames)
        push("write", self.writing_frames, specs=self.specs, applied=0)
        if self.erase_rect is not None:
            push("hold", hold_frames)
            push("erase", max(1, math.ceil(erase_seconds * fps)), rect=self.erase_rect)
            push("hold", hold_frames)
        if self.redraw_specs:
            push(
                "write",
                len(self.redraw_specs) * self.frames_per_stroke,
                specs=self.redraw_specs,
                applied=0,
            )
            push("hold", hold_frames)
        if move is not None:
            push(
                "move",
                max(1, math.ceil(move_seconds * fps)),
                delta=move,
                quad_from=self._quad_now.copy(),
            )
        if lift_off is not None:
            push(
                "lift_out",
                max(1, math.ceil(lift_seconds * fps)),
                quad_from=quad_at_lift,
                quad_to=lift_off,
            )
            push("absent", max(1, math.ceil(absent_seconds * fps)))
            if return_pose is None:
                quad_return = quad_at_lift
            else:
                rdx, rdy, rdeg = return_pose
                target = quad_at_lift + np.float32([rdx, rdy])
                center = target.mean(axis=0)
                rad = math.radians(rdeg)
                rot = np.float32(
                    [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]]
                )
                quad_return = (target - center) @ rot.T + center
            push(
                "return",
                max(1, math.ceil(return_seconds * fps)),
                quad_from=lift_off,
                quad_to=quad_return,
            )
        push("settle", max(1, math.ceil(settle_seconds * fps)))
        self.frame_count = start
        self._phase_ends = np.array([p["end"] for p in self.phases])

        self.page_ink_full = np.zeros_like(self._ink)
        final_paper = self._paper.copy()
        for spec in self.specs_all:
            gs._apply_stroke(final_paper, self.page_ink_full, spec)
        self.ink_full = cv2.warpPerspective(
            self.page_ink_full,
            self.page_to_scene,
            (gs.SIZE, gs.SIZE),
            flags=cv2.INTER_NEAREST,
        )
        self._hand = None
        if hand:
            source = gs.synthetic_hand_sources(1)[0]
            self._hand = gs._transform_hand(self._rng, source, 0.6)
        self._eraser = None
        if self.erase_rect is not None:
            self._eraser = _eraser_sprite()
        self.last_scene = None

    def _resolve_erase_rect(self, erase_stroke, erase_rect):
        if erase_rect is not None:
            x, y, w, h = (int(v) for v in erase_rect)
            if w < 1 or h < 1 or x < 0 or y < 0 or x + w > self.pw or y + h > self.ph:
                raise ValueError("erase_rect fora da folha")
            return (x, y, w, h)
        if erase_stroke is None:
            return None
        if -len(self.specs) <= erase_stroke < 0:
            erase_stroke += len(self.specs)
        if not 0 <= erase_stroke < len(self.specs):
            raise ValueError(
                f"erase_stroke {erase_stroke} fora de -{len(self.specs)}..{len(self.specs) - 1}"
            )
        points = self.specs[erase_stroke][3]
        pad = max(10, self.specs[erase_stroke][2] * 4)
        x0 = int(max(0, points[:, 0].min() - pad))
        y0 = int(max(0, points[:, 1].min() - pad))
        x1 = int(min(self.pw, points[:, 0].max() + pad))
        y1 = int(min(self.ph, points[:, 1].max() + pad))
        return (x0, y0, x1 - x0, y1 - y0)

    def isOpened(self):
        return self._opened

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
            cv2.CAP_PROP_FRAME_WIDTH: gs.SIZE,
            cv2.CAP_PROP_FRAME_HEIGHT: gs.SIZE,
            cv2.CAP_PROP_POS_FRAMES: self._position,
            cv2.CAP_PROP_POS_MSEC: self._position * 1000 / self.fps,
        }.get(prop, 0.0)

    def release(self):
        self._opened = False
        self.last_scene = None

    def _current_phase(self):
        index = int(np.searchsorted(self._phase_ends, self._position, side="right"))
        return self.phases[index]

    def _step_write(self, phase, local):
        """Revela os traços da fase como hoje: fração por frame, ponta na caneta."""
        specs = phase["specs"]
        completed = min(len(specs), max(0, local + 1) // self.frames_per_stroke)
        while phase["applied"] < completed:
            gs._apply_stroke(self._paper, self._ink, specs[phase["applied"]])
            self._completed += 1
            phase["applied"] += 1
        paper, ink = self._paper.copy(), self._ink.copy()
        pen = None
        remainder = max(0, local + 1) % self.frames_per_stroke
        if completed < len(specs) and remainder:
            kind, color, thick, points = specs[completed]
            count = max(2, math.ceil(len(points) * remainder / self.frames_per_stroke))
            gs._apply_stroke(paper, ink, (kind, color, thick, points[:count]))
            pen = points[count - 1]
        elif completed:
            pen = specs[completed - 1][3][-1]
        return paper, ink, pen

    def _step_erase(self, phase, local):
        """Varredura de borracha: devolve o papel original sob a faixa já
        limpa, apaga a tinta correspondente e deixa a borracha no front."""
        x, y, w, h = phase["rect"]
        t = min(1.0, (local + 1) / (phase["end"] - phase["start"]))
        front = x + int(round(w * t))
        if front > x:
            self._paper[y : y + h, x:front] = self._paper_pristine[y : y + h, x:front]
            self._ink[y : y + h, x:front] = 0
        eraser_page = (min(front, x + w - 1), y + h // 2)
        return eraser_page

    def _step_move(self, phase, local):
        t = min(1.0, (local + 1) / (phase["end"] - phase["start"]))
        dx, dy, deg = phase["delta"]
        quad = phase["quad_from"] + np.float32([dx * t, dy * t])
        center = quad.mean(axis=0)
        rad = math.radians(deg * t)
        rotation = np.float32(
            [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]]
        )
        self._quad_now = (quad - center) @ rotation.T + center

    def _step_quad_lerp(self, phase, local):
        """Interpola o quad entre quad_from e quad_to — saída da folha
        (lift_out) e o retorno dela deslizando para a pose nova."""
        t = min(1.0, (local + 1) / (phase["end"] - phase["start"]))
        self._quad_now = phase["quad_from"] + t * (
            phase["quad_to"] - phase["quad_from"]
        )

    def read(self):
        if not self._opened or self._position >= self.frame_count:
            return False, None
        if self.realtime:
            if self._started is None:
                self._started = time.monotonic()
            delay = self._started + self._position / self.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        phase = self._current_phase()
        local = self._position - phase["start"]
        paper, ink = self._paper.copy(), self._ink.copy()
        pen = None
        eraser_page = None
        if phase["kind"] == "write":
            paper, ink, pen = self._step_write(phase, local)
        elif phase["kind"] == "erase":
            eraser_page = self._step_erase(phase, local)
            paper, ink = self._paper.copy(), self._ink.copy()
        elif phase["kind"] == "move":
            self._step_move(phase, local)
        elif phase["kind"] in ("lift_out", "return"):
            self._step_quad_lerp(phase, local)
        self.page_to_scene = cv2.getPerspectiveTransform(
            self._page_corners, self._quad_now
        )
        quad = self._quad_now
        rgb_w, paper_mask, ink_w, paper_cov, ink_cov = gs._warp_paper(paper, ink, quad)
        rgb = self._background.copy()
        on_paper = paper_mask > 0
        rgb[on_paper] = rgb_w[on_paper]
        mask = np.zeros_like(paper_mask)
        mask[on_paper] = gs.PAPER
        mask[(ink_w > 0) & on_paper] = gs.INK
        hand_cov = np.zeros_like(paper_cov)
        if eraser_page is not None and self._eraser is not None:
            location = cv2.perspectiveTransform(
                np.float32(eraser_page).reshape(1, 1, 2), self.page_to_scene
            )[0, 0]
            eraser_rgb, eraser_mask, eraser_cov = self._eraser
            cx, cy = (int(v) for v in location)
            gs._paste_hand(rgb, mask, eraser_rgb, eraser_mask, cx, cy)
            gs._paste_coverage(hand_cov, eraser_cov, cx, cy)
            if self._hand is not None:
                hand_rgb, hand_mask, coverage = self._hand
                gs._paste_hand(rgb, mask, hand_rgb, hand_mask, cx, cy - 60)
                gs._paste_coverage(hand_cov, coverage, cx, cy - 60)
        elif self._hand is not None and pen is not None:
            location = cv2.perspectiveTransform(
                np.float32(pen).reshape(1, 1, 2), self.page_to_scene
            )[0, 0]
            hand_rgb, hand_mask, coverage = self._hand
            cx, cy = (int(v) for v in location)
            gs._paste_hand(rgb, mask, hand_rgb, hand_mask, cx, cy)
            gs._paste_coverage(hand_cov, coverage, cx, cy)
        mask, _ = gs._ambiguity_ignore(mask, paper_cov, ink_cov, hand_cov)
        # O ground truth é geométrico, anterior ao ruído e à oclusão.
        rgb = gs._sequence_photometric(self._rng, rgb, 1.0)
        self.last_scene = {"rgb": rgb, "mask": mask, "ink_full": ink_w}
        self._position += 1
        return True, rgb

    def save_ground_truth(self, path):
        """Bundle sem pickle: máscaras, pose e geometria individual dos traços.

        ``page_ink_full`` é o plano completo (todos os traços, antes de
        apagar); ``page_ink_final`` é o estado real da folha no fim da
        captura — igual ao plano em cenário sem borracha. ``events``
        descreve a linha do tempo para a auditoria por fase.
        """
        offsets = np.cumsum([0, *[len(spec[3]) for spec in self.specs_all]])
        points = (
            np.concatenate([spec[3] for spec in self.specs_all])
            if self.specs_all
            else np.empty((0, 2), np.int32)
        )
        erased = np.zeros(len(self.specs_all), dtype=bool)
        erased[: len(self.stroke_erased)] = self.stroke_erased
        events = [
            {
                key: value
                for key, value in phase.items()
                if key in ("kind", "start", "end", "rect", "delta")
            }
            for phase in self.phases
        ]
        np.savez_compressed(
            path,
            version=np.array(1),
            ink_full=self.ink_full,
            page_ink_full=self.page_ink_full,
            page_ink_final=self._ink,
            quad=self.quad,
            quad_final=self._quad_now,
            page_to_scene=self._page_to_scene_init,
            page_to_scene_final=self.page_to_scene,
            points=points,
            offsets=offsets,
            thickness=np.array([spec[2] for spec in self.specs_all], dtype=np.int32),
            stroke_phase=self.stroke_phase,
            stroke_erased=erased,
            erase_rect=np.array(
                self.erase_rect if self.erase_rect is not None else (0, 0, 0, 0),
                dtype=np.int32,
            ),
            events=np.array(json.dumps(events, ensure_ascii=False)),
            text=np.array(self.text),
            seed=np.array(self.seed),
        )


def _polyline_rect_overlap(points, rect) -> bool:
    """Algum ponto do traço dentro do retângulo de página (x, y, w, h)."""
    x, y, w, h = rect
    inside = (
        (points[:, 0] >= x)
        & (points[:, 0] < x + w)
        & (points[:, 1] >= y)
        & (points[:, 1] < y + h)
    )
    return bool(inside.any())


def _translate_specs(specs, rect, pw, ph):
    """Move os specs planejados na margem para o centro do retângulo.

    O redesenho cai sobre a área limpa, como na mesa. O deslocamento é
    clampado para manter o texto dentro da folha.
    """
    if rect is None:
        raise ValueError("redesenhar exige retângulo alvo")
    all_points = np.concatenate([spec[3] for spec in specs])
    x, y, w, h = rect
    target = np.array([x + w / 2, y + h / 2])
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    center = (lo + hi) / 2
    delta = target - center
    delta[0] = np.clip(delta[0], 2 - lo[0], pw - 2 - hi[0])
    delta[1] = np.clip(delta[1], 2 - lo[1], ph - 2 - hi[1])
    return [
        (kind, color, thick, np.round(points + delta).astype(np.int32))
        for kind, color, thick, points in specs
    ]


def _eraser_sprite():
    """Borracha sintética: bloco claro tom borracha, alpha em elipse.

    Borracha real é clara — escura demais, ela vira tinta falsa no
    modelo de página limpa em vez de oclusão estrangeira. Mesmo
    contrato de ``_transform_hand``: (rgb, máscara, cobertura), colada
    pelas primitivas de mão — ela também oclui a cena.
    """
    h, w = 56, 88
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, 255, -1)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (205, 195, 225)
    cv2.ellipse(
        rgb,
        (w // 2, h // 2 - 6),
        (w // 2 - 10, h // 2 - 10),
        0,
        0,
        360,
        (215, 208, 235),
        -1,
    )
    cv2.ellipse(
        rgb, (w // 2, h // 2), (w // 2 - 3, h // 2 - 3), 0, 0, 360, (185, 170, 205), 2
    )
    rgb[mask == 0] = 0
    cov = (mask > 0).astype(np.float32)
    return rgb, mask, cov


def export_video(capture: SyntheticCapture, output: str | Path) -> dict:
    """Grava a fonte do início ao EOF, mais GT e manifesto ao lado do MP4."""
    output = Path(output)
    if capture.get(cv2.CAP_PROP_POS_FRAMES) != 0 or not capture.isOpened():
        raise ValueError("exportação precisa de uma câmera nova, aberta no frame zero")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        capture.fps,
        (gs.SIZE, gs.SIZE),
    )
    try:
        if not writer.isOpened():
            raise RuntimeError(f"não foi possível abrir VideoWriter para {output}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    gt_path = output.with_suffix(".gt.npz")
    ink_path = output.with_name(output.stem + "_ink_full.npy")
    capture.save_ground_truth(gt_path)
    np.save(ink_path, capture.ink_full, allow_pickle=False)
    manifest = {
        "version": 1,
        "video": output.name,
        "gt": gt_path.name,
        "ink_full": ink_path.name,
        "text": capture.text,
        "seed": capture.seed,
        "fps": capture.fps,
        "frames": capture.frame_count,
        "strokes": len(capture.specs_all),
        "strokes_erased": int(capture.stroke_erased.sum()),
        "strokes_redraw": len(capture.redraw_specs),
        "hand_placeholder": capture._hand is not None,
        "pose": "fixed" if len(capture.phases) == 3 else "scenario",
        "warmup_frames": capture.warmup_frames,
        "frames_per_stroke": capture.frames_per_stroke,
        "phases": [
            {k: v for k, v in phase.items() if k in ("kind", "start", "end")}
            for phase in capture.phases
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--stroke-seconds", type=float, default=0.2)
    parser.add_argument("--warmup-seconds", type=float, default=1)
    parser.add_argument("--settle-seconds", type=float, default=3)
    parser.add_argument("--height", type=float, default=22)
    parser.add_argument("--hand", action="store_true", help="mão placeholder sintética")
    parser.add_argument(
        "--erase-stroke",
        type=int,
        help="apaga o traço de índice N (bbox com folga) após a escrita",
    )
    parser.add_argument(
        "--erase-rect",
        type=str,
        help="apaga o retângulo x,y,w,h em pixels de página após a escrita",
    )
    parser.add_argument(
        "--redraw-text",
        type=str,
        help="escreve este texto sobre a área apagada (exige apagar)",
    )
    parser.add_argument("--erase-seconds", type=float, default=1.0)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=4.0,
        help="parada de cena entre fases do cenário",
    )
    parser.add_argument(
        "--move",
        type=str,
        help="desliza/gira a folha no fim: dx,dy,graus no espaço da cena",
    )
    parser.add_argument("--move-seconds", type=float, default=0.8)
    parser.add_argument(
        "--lift",
        type=str,
        help="tira a folha de cena (lift & return): dx,dy da direção de saída",
    )
    parser.add_argument(
        "--return-pose",
        type=str,
        help="pose do retorno vs a pose da saída: dx,dy,graus (default: mesma pose)",
    )
    parser.add_argument("--absent-seconds", type=float, default=2.0)
    parser.add_argument("--lift-seconds", type=float, default=0.5)
    parser.add_argument("--return-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        erase_rect = None
        if args.erase_rect:
            erase_rect = tuple(int(v) for v in args.erase_rect.split(","))
            if len(erase_rect) != 4:
                raise ValueError("--erase-rect espera x,y,w,h")
        move = None
        if args.move:
            move = tuple(float(v) for v in args.move.split(","))
        lift = None
        if args.lift:
            lift = tuple(float(v) for v in args.lift.split(","))
        return_pose = None
        if args.return_pose:
            return_pose = tuple(float(v) for v in args.return_pose.split(","))
        capture = SyntheticCapture(
            args.text,
            seed=args.seed,
            fps=args.fps,
            stroke_seconds=args.stroke_seconds,
            warmup_seconds=args.warmup_seconds,
            settle_seconds=args.settle_seconds,
            height=args.height,
            hand=args.hand,
            erase_stroke=args.erase_stroke,
            erase_rect=erase_rect,
            redraw_text=args.redraw_text,
            erase_seconds=args.erase_seconds,
            hold_seconds=args.hold_seconds,
            move=move,
            move_seconds=args.move_seconds,
            lift=lift,
            return_pose=return_pose,
            absent_seconds=args.absent_seconds,
            lift_seconds=args.lift_seconds,
            return_seconds=args.return_seconds,
        )
        manifest = export_video(capture, args.out)
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        parser.exit(2, f"erro: {error}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
