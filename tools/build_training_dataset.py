#!/usr/bin/env python3
"""Package productive-frame mask proposals with a strict integrity preflight.

The output is a *pseudo-label candidate* dataset. Passing structural checks
does not make heuristic masks ground truth or semantically verified labels.
Unknown pixels (255) are retained for the trainer's ignore-index path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "captures/triage/classified_all_candidates.csv"
DEFAULT_OUTPUT = ROOT / "captures/triage/pseudo_training_candidate"
ALLOWED_VALUES = {0, 1, 2, 3, 255}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_selected(selection_csv: Path) -> list[dict[str, str]]:
    with selection_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"session_id", "frame_index", "source_path", "decision"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"selection CSV missing columns: {', '.join(sorted(missing))}")
        selected = [row for row in reader if row.get("decision") == "keep_productive"]

    ids: set[tuple[str, int]] = set()
    for row in selected:
        key = (row["session_id"], int(row["frame_index"]))
        if key in ids:
            raise ValueError(f"duplicate selected source ID: {key[0]} frame {key[1]}")
        ids.add(key)
    if not selected:
        raise ValueError("selection CSV contains no keep_productive rows")
    return selected


def read_proposal_manifests(paths: Iterable[Path]) -> dict[tuple[str, int], dict[str, str]]:
    proposals: dict[tuple[str, int], dict[str, str]] = {}
    for manifest_path in paths:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"session_id", "frame_index", "source_path", "mask_path", "width", "height"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{manifest_path} missing columns: {', '.join(sorted(missing))}")
            for row in reader:
                key = (row["session_id"], int(row["frame_index"]))
                if key in proposals:
                    raise ValueError(
                        f"multiple mask proposals for {key[0]} frame {key[1]}; pass only one manifest per frame"
                    )
                row["_manifest_path"] = str(manifest_path.resolve())
                proposals[key] = row
    return proposals


def build(selection_csv: Path, manifest_paths: list[Path], output_dir: Path) -> dict[str, object]:
    selection_csv = selection_csv.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(ROOT):
        raise ValueError("--output-dir must stay inside the repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")

    rows = read_selected(selection_csv)
    proposals = read_proposal_manifests(path.resolve() for path in manifest_paths)
    selected_ids = {(r["session_id"], int(r["frame_index"])) for r in rows}
    missing_ids = sorted(selected_ids - proposals.keys())
    if missing_ids:
        sample = ", ".join(f"{session}/{frame}" for session, frame in missing_ids[:8])
        raise ValueError(f"missing mask proposals for {len(missing_ids)} selected frames: {sample}")

    # Extra proposals are harmless but should not silently enter the package.
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "train" / "images"
    masks_dir = output_dir / "train" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    sample_rows: list[dict[str, object]] = []
    aggregate: Counter[int] = Counter()
    source_hashes: dict[str, tuple[str, int]] = {}

    try:
        for row in rows:
            session = row["session_id"]
            frame_index = int(row["frame_index"])
            key = (session, frame_index)
            proposal = proposals[key]
            source = resolve_repo_path(row["source_path"])
            mask_path = resolve_repo_path(proposal["mask_path"])
            if not source.is_file():
                raise FileNotFoundError(f"missing selected source image: {source}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"missing mask proposal for {session} frame {frame_index}: {mask_path}")

            # Require the proposal manifest to point to this exact source, not merely
            # another file with a matching frame number.
            proposed_source = resolve_repo_path(proposal["source_path"])
            if proposed_source != source:
                raise ValueError(f"source mapping mismatch for {session} frame {frame_index}: {proposed_source} != {source}")
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"could not decode source image: {source}")
            if mask is None:
                raise ValueError(f"could not decode proposal mask: {mask_path}")
            if mask.ndim != 2:
                raise ValueError(f"mask must be single-channel: {mask_path} has shape {mask.shape}")
            height, width = image.shape[:2]
            if mask.shape != (height, width):
                raise ValueError(f"image/mask dimensions differ for {session} frame {frame_index}: {(width, height)} vs {(mask.shape[1], mask.shape[0])}")
            if (width, height) != (int(proposal["width"]), int(proposal["height"])):
                raise ValueError(f"proposal manifest dimensions mismatch for {session} frame {frame_index}")
            values, counts = np.unique(mask, return_counts=True)
            invalid = sorted(set(int(value) for value in values) - ALLOWED_VALUES)
            if invalid:
                raise ValueError(f"invalid mask values {invalid} in {mask_path}; allowed: {sorted(ALLOWED_VALUES)}")
            pixels = {int(value): int(count) for value, count in zip(values, counts)}
            supervised = sum(count for value, count in pixels.items() if value != 255)
            contributes = (proposal.get("contributes_supervised_pixels") or "").strip().lower()
            annotation_status = (proposal.get("annotation_status") or "").strip()
            proposal_status = (proposal.get("proposal_status") or "").strip()
            if supervised == 0:
                if not (
                    contributes == "false"
                    and annotation_status == "non_contributory_all_unknown"
                    and proposal_status == "unknown_no_paper_polygon"
                    and pixels.get(255, 0) == width * height
                ):
                    raise ValueError(
                        f"all-255 mask for {session} frame {frame_index} lacks explicit "
                        "non-contributory/all-unknown manifest status"
                    )
            elif contributes == "false":
                raise ValueError(
                    f"manifest marks {session} frame {frame_index} non-contributory, but mask has supervised pixels"
                )
            elif contributes != "true":
                raise ValueError(
                    f"manifest must explicitly set contributes_supervised_pixels for {session} frame {frame_index}"
                )
            if annotation_status == "non_contributory_all_unknown" and supervised != 0:
                raise ValueError(
                    f"manifest marks {session} frame {frame_index} all-unknown, but mask has supervised pixels"
                )

            source_digest = sha256(source)
            expected_digest = (row.get("source_sha256") or "").strip()
            if expected_digest and expected_digest != source_digest:
                raise ValueError(f"selection source checksum mismatch for {session} frame {frame_index}")
            manifest_digest = (proposal.get("source_sha256") or "").strip()
            if manifest_digest and manifest_digest != source_digest:
                raise ValueError(f"proposal source checksum mismatch for {session} frame {frame_index}")
            # Distinct source IDs may contain identical images, but flag the fact in
            # the manifest; do not discard them automatically.
            duplicate_of = source_hashes.get(source_digest)
            source_hashes.setdefault(source_digest, key)

            stem = f"{session}_frame_{frame_index:08d}"
            image_out = images_dir / f"{stem}{source.suffix.lower()}"
            mask_out = masks_dir / f"{stem}.png"
            shutil.copyfile(source, image_out)
            shutil.copyfile(mask_path, mask_out)
            aggregate.update(pixels)
            sample_rows.append({
                "session_id": session,
                "frame_index": frame_index,
                "take_id": row.get("take_id", ""),
                "timestamp_ns": row.get("timestamp_ns", ""),
                "split": "train",
                "dataset_role": "real_train_pseudo_label_candidate",
                "label_status": "unverified_mask_proposal",
                "review_status": proposal.get("review_status", "not_reviewed"),
                "review_notes": proposal.get("review_notes", ""),
                "annotation_status": annotation_status,
                "contributes_supervised_pixels": contributes == "true",
                "contribution_status": (
                    "non_contributory_all_unknown" if supervised == 0 else "contributes_supervised_pixels"
                ),
                "source_path": str(source.relative_to(ROOT)),
                "source_sha256": source_digest,
                "image_path": str(image_out.relative_to(output_dir)),
                "image_sha256": sha256(image_out),
                "mask_proposal_path": str(mask_path.relative_to(ROOT)) if mask_path.is_relative_to(ROOT) else str(mask_path),
                "mask_path": str(mask_out.relative_to(output_dir)),
                "mask_sha256": sha256(mask_out),
                "proposal_manifest": proposal["_manifest_path"],
                "width": width,
                "height": height,
                "supervised_pixels": supervised,
                "ignored_pixels": pixels.get(255, 0),
                "ignored_fraction": round(pixels.get(255, 0) / (width * height), 8),
                "duplicate_source_hash_of": f"{duplicate_of[0]}:{duplicate_of[1]}" if duplicate_of else "",
                "class_0_pixels": pixels.get(0, 0),
                "class_1_pixels": pixels.get(1, 0),
                "class_2_pixels": pixels.get(2, 0),
                "class_3_pixels": pixels.get(3, 0),
                "visible_tags_hint_only": row.get("visible_tags", ""),
            })

        # This dataset tranche must contain ink supervision to be four-class useful.
        if aggregate[3] == 0:
            raise ValueError("package has no class-3 pixels; cannot represent four-class ink supervision")

        manifest_csv = output_dir / "samples.csv"
        with manifest_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)
        metadata = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_kind": "real_frame_pseudo_label_candidate",
            "semantic_review_status": "unverified; not ground truth",
            "trainer_contract": {"split": "train", "mask_values": [0, 1, 2, 3, 255], "ignore_index": 255},
            "source_selection_csv": str(selection_csv.relative_to(ROOT)),
            "selected_frame_count": len(rows),
            "packaged_frame_count": len(sample_rows),
            "sessions": sorted({r["session_id"] for r in sample_rows}),
            "class_pixel_counts": {str(value): aggregate[value] for value in (0, 1, 2, 3, 255)},
            "frames_with_class_3": sum(1 for r in sample_rows if r["class_3_pixels"] > 0),
            "frames_with_ignored_pixels": sum(1 for r in sample_rows if r["ignored_pixels"] > 0),
            "supervised_frames": sum(1 for r in sample_rows if r["contributes_supervised_pixels"]),
            "non_contributory_all_unknown_frames": sum(
                1 for r in sample_rows if r["contribution_status"] == "non_contributory_all_unknown"
            ),
            "frames_contact_sheet_reviewed": sum(
                1 for r in sample_rows if r["review_status"] == "visual_triage_reviewed_thumbnail"
            ),
            "frames_with_explicit_review_corrections": sum(
                1 for r in sample_rows if "set " in r["review_notes"]
            ),
            "identical_source_hash_groups": len(sample_rows) - len(source_hashes),
            "visual_review_scope": "all source/mask pairs reviewed on contact sheets; gross errors corrected; fine class-3 boundaries remain unverified",
            "warning": "Structural preflight passed; visual triage does not make pseudo masks pixel-accurate ground truth.",
        }
        (output_dir / "dataset_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata
    except Exception:
        # Do not leave a partial package looking complete.
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--proposal-manifest", type=Path, action="append", required=True,
                        help="CSV emitted by tools/propose_semantic_masks.py; may be repeated for disjoint batches")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = build(args.selection, args.proposal_manifest, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Packaged {result['packaged_frame_count']} pseudo-labeled candidates to {args.output_dir}")
    print("Class pixels:", result["class_pixel_counts"])
    print("WARNING: structurally trainable format does not mean semantically verified labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
