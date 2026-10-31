#!/usr/bin/env python3
"""Rank captured Draw frames for human review; this does not create labels.

The fast default pass uses OpenCV image-quality, paper-candidate, and adjacent
perceptual-duplicate features. Optional OpenCLIP scores add semantic candidates.
All scores are triage signals, not ground truth or calibrated probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SESSIONS = [
    ROOT / "captures/training/session_20261029_130013_453419",
    ROOT / "captures/batches/batch_01/training/session_20261029_151126_089557",
]

CLIP_CATEGORIES = {
    "paper_with_user_marks": [
        "a camera frame of a sheet of paper with a person's hand-drawn marks",
        "a sheet of paper with a simple drawing made by hand, not printed text",
        "papel com desenho ou traços feitos à mão por uma pessoa",
    ],
    "hand_over_paper": [
        "a person's hand covering or touching a sheet of paper",
        "a hand holding a pen while drawing on paper",
    ],
    "hand_elsewhere": [
        "a person's hand on a keyboard beside a sheet of paper, not covering the paper",
        "a hand on a desk or keyboard, away from the sheet of paper",
    ],
    "blank_paper": [
        "a blank white sheet of paper on a desk, with no writing or drawing",
        "folha de papel em branco sobre uma mesa",
    ],
    "printed_paper": [
        "pre-printed text or graphics on a sheet of paper, no new hand-drawn marks",
        "a printed document on paper, with typed text rather than user handwriting",
    ],
    "pen_without_hand": [
        "a pen lying on a desk beside paper, with no hand visible",
        "caneta sobre a mesa perto de uma folha, sem mão visível",
    ],
    "no_paper_background": [
        "an empty desk or table with no sheet of paper",
        "mesa azul vazia, sem folha de papel",
    ],
    "unusable_quality": [
        "a very blurry, dark, overexposed, or obstructed camera frame",
        "imagem muito borrada, escura, estourada ou obstruída",
    ],
}


def load_metadata(session: Path) -> tuple[dict[str, dict[str, str]], dict[int, dict[str, str]]]:
    frames: dict[str, dict[str, str]] = {}
    frame_index: dict[int, dict[str, str]] = {}
    path = session / "frames.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                frames[row["filename"]] = row
                frame_index[int(row["frame_index"])] = row
    takes: dict[int, dict[str, str]] = {}
    path = session / "takes.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                takes[int(row["start_frame"])] = row
    return frames, takes


def take_for(index: int, takes: dict[int, dict[str, str]]) -> dict[str, str] | None:
    # There are only about 100 takes; this linear scan is negligible.
    for row in takes.values():
        if int(row["start_frame"]) <= index <= int(row["end_frame"]):
            return row
    return None


def perceptual_hash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    values = dct.flatten()[1:]
    median = float(np.median(values))
    bits = values > median
    return sum(int(bit) << i for i, bit in enumerate(bits))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def image_features(path: Path, include_sheet_detector: bool = False) -> dict[str, object]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"decode_ok": False}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape
    mean = float(np.mean(gray))
    dark_fraction = float(np.mean(gray < 12))
    clipped_fraction = float(np.mean(gray > 245))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # White/low-saturation pixels are only a rough paper proposal; a keyboard,
    # glare, or a light background can trigger it. Keep the ratio for review.
    paper_pixels = (gray > 145) & (hsv[:, :, 1] < 105)
    paper_ratio = float(np.mean(paper_pixels))
    edges = cv2.Canny(gray, 65, 145)
    paper_edge_ratio = float(np.mean((edges > 0) & paper_pixels))
    detected_sheet_ratio = None
    sheet_verdict = "not_run"
    if include_sheet_detector:
        from draw.paper import detect_sheet_with_confidence
        sheet_corners, sheet_verdict = detect_sheet_with_confidence(image)
        if sheet_corners is not None:
            detected_sheet_ratio = float(abs(cv2.contourArea(sheet_corners)) / (width * height))
    return {
        "decode_ok": True,
        "width": width,
        "height": height,
        "mean_brightness": round(mean, 3),
        "dark_fraction": round(dark_fraction, 5),
        "clipped_fraction": round(clipped_fraction, 5),
        "sharpness_laplacian": round(sharpness, 3),
        "paper_candidate_ratio": round(paper_ratio, 5),
        "paper_edge_ratio": round(paper_edge_ratio, 5),
        "existing_sheet_detector": sheet_verdict or "none",
        "detected_sheet_area_ratio": round(detected_sheet_ratio, 5) if detected_sheet_ratio is not None else "",
        "phash": perceptual_hash(gray),
    }


def load_clip(model_name: str, pretrained: str):
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "OpenCLIP opcional ausente. Instale em ambiente isolado: "
            "pip install torch open_clip_torch"
        ) from exc
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device="cpu"
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    with torch.no_grad():
        all_prompts = [p for prompts in CLIP_CATEGORIES.values() for p in prompts]
        text = model.encode_text(tokenizer(all_prompts))
        text = text / text.norm(dim=-1, keepdim=True)
    offsets: dict[str, list[int]] = {}
    cursor = 0
    for category, prompts in CLIP_CATEGORIES.items():
        offsets[category] = list(range(cursor, cursor + len(prompts)))
        cursor += len(prompts)
    return model, preprocess, text, offsets, torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", type=Path, help="session folder; repeatable")
    parser.add_argument("--output", type=Path, default=ROOT / "captures/triage/frames.csv")
    parser.add_argument("--takes-output", type=Path, default=ROOT / "captures/triage/takes.csv")
    parser.add_argument("--limit", type=int, help="process only first N frames per session (smoke/benchmark)")
    parser.add_argument("--every", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--clip", action="store_true", help="add optional OpenCLIP semantic candidate scores")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--phash-distance", type=int, default=3)
    parser.add_argument("--sheet-detector", action="store_true", help="also run the slower existing quadrilateral detector")
    args = parser.parse_args()
    if args.every < 1 or args.clip_batch_size < 1:
        parser.error("--every and --clip-batch-size must be positive")

    sessions = [p.resolve() for p in (args.session or DEFAULT_SESSIONS)]
    clip_state = load_clip(args.clip_model, args.clip_pretrained) if args.clip else None
    output_rows: list[dict[str, object]] = []
    take_acc: dict[tuple[str, str, str], dict[str, object]] = {}
    start_time = time.monotonic()

    for session in sessions:
        if not session.is_dir():
            print(f"session missing: {session}", file=sys.stderr)
            continue
        frame_meta, take_starts = load_metadata(session)
        paths = sorted(session.glob("frame_*.jpg"))
        if args.every > 1:
            paths = paths[::args.every]
        if args.limit is not None:
            paths = paths[:args.limit]
        print(f"Scanning {session.name}: {len(paths)} frames", file=sys.stderr)
        previous: dict[str, object] | None = None

        # OpenCLIP batches avoid repeated model overhead, but keep a modest
        # batch in memory for this host's limited RAM.
        for offset in range(0, len(paths), args.clip_batch_size if clip_state else 1):
            batch = paths[offset: offset + (args.clip_batch_size if clip_state else 1)]
            features_batch = [image_features(path, args.sheet_detector) for path in batch]
            clip_scores_batch: list[dict[str, float] | None] = [None] * len(batch)
            if clip_state:
                model, preprocess, text, prompt_offsets, torch = clip_state
                from PIL import Image
                tensors = []
                for path in batch:
                    with Image.open(path) as image:
                        tensors.append(preprocess(image.convert("RGB")))
                image_tensor = torch.stack(tensors)
                with torch.no_grad():
                    image_features_tensor = model.encode_image(image_tensor)
                    image_features_tensor = image_features_tensor / image_features_tensor.norm(dim=-1, keepdim=True)
                    similarities = (image_features_tensor @ text.T).cpu().numpy()
                for i, scores in enumerate(similarities):
                    clip_scores_batch[i] = {
                        category: round(float(np.mean(scores[indices])), 6)
                        for category, indices in prompt_offsets.items()
                    }

            for path, features, clip_scores in zip(batch, features_batch, clip_scores_batch):
                filename = path.name
                meta = frame_meta.get(filename, {})
                try:
                    frame_index = int(meta.get("frame_index", path.stem.split("_")[-1]))
                except ValueError:
                    frame_index = -1
                take = take_for(frame_index, take_starts) if take_starts and frame_index > 0 else None
                take_id = str(take.get("take", "")) if take else ""
                batch_id = str(take.get("batch", "")) if take else ""
                phash = features.get("phash")
                if previous and (previous.get("take_id") != take_id or previous.get("batch_id") != batch_id):
                    previous = None
                duplicate_distance = hamming(int(phash), int(previous["phash"])) if previous and features.get("decode_ok") and previous.get("decode_ok") else None
                adjacent_duplicate = duplicate_distance is not None and duplicate_distance <= args.phash_distance

                if not features.get("decode_ok"):
                    category = "decode_error"
                    quality_flag = "decode_error"
                    productive_candidate = "review"
                else:
                    sharpness = float(features["sharpness_laplacian"])
                    brightness = float(features["mean_brightness"])
                    paper_ratio = float(features["paper_candidate_ratio"])
                    quality_flag = "unranked"
                    if clip_scores:
                        category = max(clip_scores, key=clip_scores.get)
                    else:
                        category = "paper_candidate" if paper_ratio >= 0.06 else "background_candidate"
                    semantic_useful = bool(clip_scores and category in {
                        "paper_with_user_marks", "hand_over_paper", "hand_elsewhere",
                        "blank_paper", "printed_paper",
                    })
                    if adjacent_duplicate:
                        productive_candidate = "lower_priority_near_duplicate"
                    elif semantic_useful or paper_ratio >= 0.06:
                        productive_candidate = "higher_priority_candidate"
                    else:
                        productive_candidate = "lower_priority_candidate"

                row: dict[str, object] = {
                    "session_id": session.name,
                    "frame_index": frame_index,
                    "timestamp_ns": meta.get("timestamp_ns", ""),
                    "batch_id": batch_id,
                    "take_id": take_id,
                    "take_status": take.get("status", "") if take else "",
                    "filename": filename,
                    "source_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    **{k: v for k, v in features.items() if k != "phash"},
                    "adjacent_phash_distance": duplicate_distance,
                    "adjacent_near_duplicate_candidate": adjacent_duplicate,
                    "quality_candidate": quality_flag,
                    "semantic_candidate": category,
                    "triage_candidate": productive_candidate,
                    "review_status": "unreviewed_candidate",
                }
                if clip_scores:
                    row.update({f"clip_{k}": v for k, v in clip_scores.items()})
                    ranked_scores = sorted(clip_scores.values(), reverse=True)
                    row["clip_top_score"] = ranked_scores[0]
                    row["clip_margin_top2"] = round(ranked_scores[0] - ranked_scores[1], 6)
                output_rows.append(row)

                if take_id:
                    acc = take_acc.setdefault((session.name, batch_id, take_id), {"frames": 0, "decode_ok": 0, "paper": 0, "quality": 0, "duplicates": 0, "categories": {}})
                    acc["frames"] += 1
                    acc["decode_ok"] += int(bool(features.get("decode_ok")))
                    acc["paper"] += int(bool(features.get("decode_ok")) and float(features.get("paper_candidate_ratio", 0)) >= 0.06)
                    acc["duplicates"] += int(bool(adjacent_duplicate))
                    cats = acc["categories"]
                    cats[category] = cats.get(category, 0) + 1
                if features.get("decode_ok"):
                    previous = {"phash": phash, "decode_ok": True, "take_id": take_id, "batch_id": batch_id}
                else:
                    previous = None

            done = min(offset + len(batch), len(paths))
            if done % 500 < len(batch):
                print(f"  {done}/{len(paths)}", file=sys.stderr)

    # Rank sharpness within each session because camera resolution and focus
    # differ. This identifies relative review candidates, not an absolute
    # image-quality pass/fail threshold.
    sharpness_by_session: dict[str, list[float]] = {}
    for row in output_rows:
        if row.get("decode_ok"):
            sharpness_by_session.setdefault(str(row["session_id"]), []).append(float(row["sharpness_laplacian"]))
    sharpness_floor = {
        session: float(np.percentile(values, 10))
        for session, values in sharpness_by_session.items() if values
    }
    take_quality: dict[tuple[str, str, str], int] = {}
    for row in output_rows:
        if not row.get("decode_ok"):
            row["quality_candidate"] = "decode_error"
            continue
        brightness = float(row["mean_brightness"])
        sharpness = float(row["sharpness_laplacian"])
        quality_ok = (
            sharpness >= sharpness_floor.get(str(row["session_id"]), 0)
            and 35 <= brightness <= 235
        )
        row["quality_candidate"] = "quality_candidate" if quality_ok else "review_quality"
        if row.get("batch_id") and row.get("take_id"):
            key = (str(row["session_id"]), str(row["batch_id"]), str(row["take_id"]))
            take_quality[key] = take_quality.get(key, 0) + int(quality_ok)
    for key, acc in take_acc.items():
        acc["quality"] = take_quality.get(key, 0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in output_rows for k in row))
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    take_rows = []
    for (session_id, batch_id, take_id), acc in sorted(take_acc.items()):
        frame_count = int(acc["frames"])
        cats = acc["categories"]
        paper_fraction = acc["paper"] / frame_count if frame_count else 0
        quality_fraction = acc["quality"] / frame_count if frame_count else 0
        clip_count = sum(value for name, value in cats.items() if name.startswith("paper_") or name in {"printed_paper", "hand_over_paper", "hand_elsewhere", "pen_without_hand"})
        clip_fraction = clip_count / frame_count if frame_count else 0
        if clip_state is not None:
            if clip_fraction >= 0.5:
                take_candidate = "likely_productive_candidate"
            elif clip_fraction >= 0.15:
                take_candidate = "mixed_review"
            else:
                take_candidate = "lower_priority_candidate"
        elif paper_fraction >= 0.5:
            take_candidate = "likely_productive_candidate"
        elif paper_fraction >= 0.15:
            take_candidate = "mixed_review"
        else:
            take_candidate = "lower_priority_candidate"
        take_rows.append({
            "session_id": session_id,
            "batch_id": batch_id,
            "take_id": take_id,
            "frames_scanned": frame_count,
            "decode_ok": acc["decode_ok"],
            "paper_candidate_frames": acc["paper"],
            "quality_candidate_frames": acc["quality"],
            "adjacent_duplicate_candidates": acc["duplicates"],
            "paper_candidate_fraction": round(acc["paper"] / frame_count, 5) if frame_count else 0,
            "semantic_useful_candidate_fraction": round(clip_fraction, 5) if clip_count else "",
            "quality_candidate_fraction": round(acc["quality"] / frame_count, 5) if frame_count else 0,
            "semantic_candidate_counts": json.dumps(cats, sort_keys=True),
            "take_triage_candidate": take_candidate,
            "review_status": "unreviewed_candidate",
        })
    with args.takes_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(take_rows[0]) if take_rows else ["session_id", "take_id"])
        writer.writeheader()
        writer.writerows(take_rows)

    elapsed = time.monotonic() - start_time
    print(f"Frames indexed: {len(output_rows)}; elapsed: {elapsed:.1f}s; output: {args.output}; takes: {args.takes_output}")
    if output_rows and elapsed > 0:
        rate = len(output_rows) / elapsed
        print(f"Observed rate: {rate:.2f} frames/s; estimated full 15,845-frame scan: {15845 / rate / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
