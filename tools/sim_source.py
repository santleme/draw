"""Câmera sintética lazy e exportador MP4 + ground truth da escrita.

Usa as primitivas de compose_scene_sequence, mas mantém a pose fixa e
revela também frações de traço. Não chama a sequência eager (que escolhe
seu próprio plano aleatório) nem modifica o gerador de dados de treino.
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
    ):
        if not math.isfinite(fps) or not 0 < fps <= 120:
            raise ValueError("fps deve estar entre 0 e 120")
        if not math.isfinite(stroke_seconds) or stroke_seconds <= 0:
            raise ValueError("stroke_seconds deve ser positivo e finito")
        if any(not math.isfinite(v) or v < 0 for v in (warmup_seconds, settle_seconds)):
            raise ValueError("warmup/settle devem ser finitos e não negativos")
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
        self._ink = np.zeros((self.ph, self.pw), np.uint8)
        page_corners = np.float32(
            [[0, 0], [self.pw, 0], [self.pw, self.ph], [0, self.ph]]
        )
        self.page_to_scene = cv2.getPerspectiveTransform(page_corners, self.quad)
        self.warmup_frames = max(1, math.ceil(warmup_seconds * fps))
        self.frames_per_stroke = max(1, math.ceil(stroke_seconds * fps))
        self.writing_frames = len(self.specs) * self.frames_per_stroke
        self.frame_count = (
            self.warmup_frames
            + self.writing_frames
            + max(1, math.ceil(settle_seconds * fps))
        )
        self.page_ink_full = np.zeros_like(self._ink)
        final_paper = self._paper.copy()
        for spec in self.specs:
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
        self.last_scene = None

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

    def read(self):
        if not self._opened or self._position >= self.frame_count:
            return False, None
        if self.realtime:
            if self._started is None:
                self._started = time.monotonic()
            delay = self._started + self._position / self.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        progress = max(0, self._position - self.warmup_frames + 1)
        completed = min(len(self.specs), progress // self.frames_per_stroke)
        while self._completed < completed:
            gs._apply_stroke(self._paper, self._ink, self.specs[self._completed])
            self._completed += 1
        paper, ink = self._paper.copy(), self._ink.copy()
        pen = None
        remainder = progress % self.frames_per_stroke
        if completed < len(self.specs) and remainder:
            kind, color, thick, points = self.specs[completed]
            count = max(2, math.ceil(len(points) * remainder / self.frames_per_stroke))
            gs._apply_stroke(paper, ink, (kind, color, thick, points[:count]))
            pen = points[count - 1]
        elif 0 < progress <= self.writing_frames and completed:
            pen = self.specs[completed - 1][3][-1]
        rgb_w, paper_mask, ink_w, paper_cov, ink_cov = gs._warp_paper(
            paper, ink, self.quad
        )
        rgb = self._background.copy()
        on_paper = paper_mask > 0
        rgb[on_paper] = rgb_w[on_paper]
        mask = np.zeros_like(paper_mask)
        mask[on_paper] = gs.PAPER
        mask[(ink_w > 0) & on_paper] = gs.INK
        hand_cov = np.zeros_like(paper_cov)
        if self._hand is not None and pen is not None:
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
        """Bundle sem pickle: máscaras, pose e geometria individual dos traços."""
        offsets = np.cumsum([0, *[len(spec[3]) for spec in self.specs]])
        points = (
            np.concatenate([spec[3] for spec in self.specs])
            if self.specs
            else np.empty((0, 2), np.int32)
        )
        np.savez_compressed(
            path,
            version=np.array(1),
            ink_full=self.ink_full,
            page_ink_full=self.page_ink_full,
            quad=self.quad,
            page_to_scene=self.page_to_scene,
            points=points,
            offsets=offsets,
            thickness=np.array([spec[2] for spec in self.specs], dtype=np.int32),
            text=np.array(self.text),
            seed=np.array(self.seed),
        )


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
        "strokes": len(capture.specs),
        "hand_placeholder": capture._hand is not None,
        "pose": "fixed",
        "warmup_frames": capture.warmup_frames,
        "frames_per_stroke": capture.frames_per_stroke,
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
    args = parser.parse_args(argv)
    try:
        capture = SyntheticCapture(
            args.text,
            seed=args.seed,
            fps=args.fps,
            stroke_seconds=args.stroke_seconds,
            warmup_seconds=args.warmup_seconds,
            settle_seconds=args.settle_seconds,
            height=args.height,
            hand=args.hand,
        )
        manifest = export_video(capture, args.out)
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        parser.exit(2, f"erro: {error}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
