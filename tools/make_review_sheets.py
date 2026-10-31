#!/usr/bin/env python3
"""Split the primary frame-review queue into take-preserving visual batches."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "captures/triage/review_queue.csv"
DEFAULT_OUTPUT = ROOT / "captures/triage/review_batches"
LABEL_FIELDS = (
    "session_id", "frame_index", "batch_id", "take_id", "review_group_id",
    "source_path", "model_suggestion", "decision", "visible_tags", "confidence", "notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def split_groups(rows: list[dict[str, str]], parts: int) -> list[list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["session_id"], row["review_group_id"])].append(row)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    shards: list[list[dict[str, str]]] = [[] for _ in range(parts)]
    for _, group in ordered:
        target = min(range(parts), key=lambda index: (len(shards[index]), index))
        shards[target].extend(group)
    for shard in shards:
        shard.sort(key=lambda row: (row["session_id"], row["review_group_id"], int(row["frame_index"])))
    return shards


def contact_sheet(rows: list[dict[str, str]], output: Path, part: int, start: int) -> None:
    columns, thumb_w, thumb_h, caption_h = 4, 320, 180, 42
    cell_h = thumb_h + caption_h
    count = (len(rows) + columns - 1) // columns
    canvas = np.full((count * cell_h, columns * thumb_w, 3), 248, dtype=np.uint8)
    for offset, row in enumerate(rows):
        path = Path(row["source_path"])
        if not path.is_absolute():
            path = ROOT / path
        image = cv2.imread(str(path)) if path.is_file() else None
        x, y = (offset % columns) * thumb_w, (offset // columns) * cell_h
        if image is None:
            cv2.rectangle(canvas, (x, y), (x + thumb_w - 1, y + thumb_h - 1), (210, 210, 210), -1)
            cv2.putText(canvas, "MISSING / DECODE ERROR", (x + 8, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1, cv2.LINE_AA)
        else:
            scale = min(thumb_w / image.shape[1], thumb_h / image.shape[0])
            resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            ox, oy = x + (thumb_w - resized.shape[1]) // 2, y + (thumb_h - resized.shape[0]) // 2
            canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        cv2.rectangle(canvas, (x, y + thumb_h), (x + thumb_w - 1, y + cell_h - 1), (232, 232, 232), -1)
        label = f"P{part} #{start + offset + 1} | {row['session_id'][-6:]} F{row['frame_index']}"
        group = row.get("review_group_id", "")
        suggestion = row.get("semantic_candidate", "")
        cv2.putText(canvas, label[:48], (x + 6, y + thumb_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{group[:25]} | suggestion: {suggestion[:18]}", (x + 6, y + thumb_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parts", type=int, default=3)
    parser.add_argument("--frames-per-sheet", type=int, default=16)
    args = parser.parse_args()
    if args.parts < 1 or args.frames_per_sheet < 1:
        parser.error("--parts and --frames-per-sheet must be positive")

    rows = read_csv(args.input)
    shards = split_groups(rows, args.parts)
    for part, shard in enumerate(shards, start=1):
        part_dir = args.output_dir / f"part_{part:02d}"
        label_rows = [{
            "session_id": row["session_id"],
            "frame_index": row["frame_index"],
            "batch_id": row.get("batch_id", ""),
            "take_id": row.get("take_id", ""),
            "review_group_id": row.get("review_group_id", ""),
            "source_path": row["source_path"],
            "model_suggestion": row.get("semantic_candidate", ""),
            "decision": "",
            "visible_tags": "",
            "confidence": "",
            "notes": "",
        } for row in shard]
        write_csv(part_dir / "labels_template.csv", label_rows, LABEL_FIELDS)
        for start in range(0, len(shard), args.frames_per_sheet):
            sheet_rows = shard[start:start + args.frames_per_sheet]
            contact_sheet(sheet_rows, part_dir / f"sheet_{start // args.frames_per_sheet + 1:03d}.jpg", part, start)
        print(f"part {part}: {len(shard)} frames, {len({r['review_group_id'] for r in shard})} groups, {part_dir}")
    print(f"source queue: {len(rows)}; unique source frames: {len({(r['session_id'], r['frame_index']) for r in rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
