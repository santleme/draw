#!/usr/bin/env python3
"""Apply explicit, human-reviewed gross-error corrections to mask proposals.

Review here means contact-sheet triage at thumbnail scale. It does not verify
fine ink strokes or every pixel boundary, so outputs remain pseudo-labels.
Source photographs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from propose_semantic_masks import image_sha256, save_overlay

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "captures/triage/mask_proposals/full/proposal_manifest.csv"
DEFAULT_OUTPUT = ROOT / "captures/triage/mask_proposals/reviewed"

SESSION_OLD = "session_20261029_130013_453419"
SESSION_NEW = "session_20261029_151126_089557"

# Reviewers found no visible hand in these earphone/headset frames. The
# skin-colour detector had assigned earcup plastic to class 2 (hand).
HEADSET_NO_HAND_FRAMES = {
    2311, 2313, 2329, 2344, 2359, 2360, 2365, 2369, 2370, 2395,
    2402, 2411, 2426, 2427, 2430, 2442, 2457, 2458, 2490,
}

# Contact-sheet reviewers confirmed these two newer-session frames show blue desk and a
# bright triangular desk region, not a visible sheet. Class-1 pixels are false.
NO_VISIBLE_PAPER_FRAMES = {389, 390}
NO_VISIBLE_HAND_FRAMES = {389, 390}


def correct(manifest_path: Path, output_dir: Path) -> tuple[int, Counter[str]]:
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(ROOT):
        raise ValueError("output directory must stay inside the repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for required in ("session_id", "frame_index", "source_path", "source_sha256", "mask_path", "overlay_path"):
        if required not in fields:
            raise ValueError(f"manifest missing column {required}")
    for extra in ("review_status", "review_notes", "mask_sha256", "fraction_class_255"):
        if extra not in fields:
            fields.append(extra)

    corrections: Counter[str] = Counter()
    seen: set[tuple[str, int]] = set()
    try:
        for row in rows:
            session = row["session_id"]
            index = int(row["frame_index"])
            key = (session, index)
            if key in seen:
                raise ValueError(f"duplicate source id: {key}")
            seen.add(key)

            source = ROOT / row["source_path"]
            source_img = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if source_img is None or image_sha256(source) != row["source_sha256"]:
                raise ValueError(f"source missing or changed: {source}")
            original_mask_path = ROOT / row["mask_path"]
            mask = cv2.imread(str(original_mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2 or mask.shape != source_img.shape[:2]:
                raise ValueError(f"invalid original proposal: {original_mask_path}")

            notes: list[str] = []
            if session == SESSION_OLD and index in HEADSET_NO_HAND_FRAMES:
                count = int(np.count_nonzero(mask == 2))
                mask[mask == 2] = 0
                notes.append(f"set {count} headset-colored pixels from hand(2) to background(0); no hand visible")
                corrections["headset_hand_to_background"] += 1
            if session == SESSION_NEW and index in NO_VISIBLE_PAPER_FRAMES:
                count = int(np.count_nonzero(mask == 1))
                mask[mask == 1] = 0
                notes.append(f"set {count} false desk-wedge pixels from paper(1) to background(0); no sheet visible")
                corrections["false_paper_to_background"] += 1
            if session == SESSION_NEW and index in NO_VISIBLE_HAND_FRAMES:
                count = int(np.count_nonzero(mask == 2))
                mask[mask == 2] = 0
                notes.append(f"set {count} wood-trim pixels from hand(2) to background(0); no hand visible")
                corrections["false_hand_to_background"] += 1

            stem = original_mask_path.stem.removesuffix("_mask")
            mask_path = output_dir / f"{stem}_mask.png"
            overlay_path = output_dir / f"{stem}_overlay.jpg"
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"could not write {mask_path}")
            save_overlay(source_img, mask, overlay_path)

            values, counts = np.unique(mask, return_counts=True)
            pixel_counts = {int(v): int(n) for v, n in zip(values, counts)}
            row["mask_path"] = str(mask_path.relative_to(ROOT))
            row["overlay_path"] = str(overlay_path.relative_to(ROOT))
            row["mask_sha256"] = image_sha256(mask_path)
            row["pixel_counts"] = ";".join(f"{v}:{pixel_counts.get(v, 0)}" for v in (0, 1, 2, 3, 255))
            total = mask.size
            for class_id in (0, 1, 2, 3, 255):
                row[f"fraction_class_{class_id}"] = f"{pixel_counts.get(class_id, 0) / total:.8f}"
            row["ignore_fraction"] = f"{pixel_counts.get(255, 0) / total:.8f}"
            row["review_status"] = "visual_triage_reviewed_thumbnail"
            row["review_notes"] = " | ".join(notes) if notes else "No gross error noted at thumbnail scale; fine ink boundaries remain unverified."
            if np.count_nonzero(mask != 255) == 0:
                row["annotation_status"] = "non_contributory_all_unknown"
                row["proposal_status"] = "unknown_no_paper_polygon"
                row["contributes_supervised_pixels"] = "False"
            else:
                row["annotation_status"] = "unverified_pseudo_label_after_visual_triage"
                row["proposal_status"] = "visual_triage_reviewed_fine_labels_unverified"
                row["contributes_supervised_pixels"] = "True"

        with (output_dir / "proposal_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return len(rows), corrections
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count, corrections = correct(args.manifest, args.output_dir)
    print(f"reviewed {count} proposals; contact-sheet corrections: {dict(corrections)}")
    print("Status remains pseudo-label: review did not verify fine ink pixels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
