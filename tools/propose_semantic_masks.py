#!/usr/bin/env python3
"""Create conservative, explicitly unverified semantic-mask proposals.

The default run selects a deterministic 20-frame pilot. Pass --all-selected
for every keep_productive frame. Outputs are unverified pseudo-labels, never
ground truth. Pixels that cannot be assigned conservatively are 255 (unknown),
not background; all-unknown frames remain in the manifest as non-contributory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "captures/triage/classified_all_candidates.csv"
DEFAULT_OUTPUT = ROOT / "captures/triage/mask_proposals/pilot"
CLASSES = {0: "background", 1: "paper", 2: "hand", 3: "ink", 255: "unknown"}
TAG_COVERAGE = ("user_added_ink", "hand_on_paper", "printed_text", "occlusion", "blur_or_bad_quality")
PALETTE = {
    0: (25, 25, 25),
    1: (190, 190, 190),
    2: (40, 80, 230),
    3: (220, 100, 30),
    255: (200, 0, 210),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if row.get("decision") == "keep_productive"]


def choose_pilot(rows: list[dict[str, str]], count: int = 20) -> list[dict[str, str]]:
    """Choose a stable, tag-stratified pilot with both sessions represented."""
    sessions = sorted({row["session_id"] for row in rows})
    if len(sessions) < 2:
        raise ValueError("pilot requires productive frames from both source sessions")
    target_per_session = count // len(sessions)
    chosen: dict[tuple[str, str], dict[str, str]] = {}
    by_session = {
        session: sorted(
            (row for row in rows if row["session_id"] == session),
            key=lambda row: int(row["frame_index"]),
        )
        for session in sessions
    }

    # Cover each requested visual risk in each session where possible.
    for session in sessions:
        session_rows = by_session[session]
        for tag in TAG_COVERAGE:
            candidate = next(
                (row for row in session_rows if tag in row.get("visible_tags", "").split(";")),
                None,
            )
            if candidate is not None:
                chosen[(candidate["session_id"], candidate["frame_index"])] = candidate

    for session in sessions:
        session_rows = by_session[session]
        target = target_per_session + (1 if session == sessions[-1] and count % len(sessions) else 0)
        selected = [row for row in session_rows if (session, row["frame_index"]) in chosen]
        if len(selected) >= target:
            continue
        needed = target - len(selected)
        # Spread fillers through the session rather than clustering at frame 0.
        candidates = [row for row in session_rows if (session, row["frame_index"]) not in chosen]
        if not candidates:
            continue
        positions = np.linspace(0, len(candidates) - 1, num=min(needed, len(candidates)), dtype=int)
        for index in positions:
            row = candidates[int(index)]
            chosen[(session, row["frame_index"])] = row

    selected = list(chosen.values())
    selected.sort(key=lambda row: (row["session_id"], int(row["frame_index"])))
    if len(selected) < count:
        raise ValueError(f"only selected {len(selected)} pilot frames; required {count}")
    # If tag coverage exceeds 20, retain deterministic order while keeping
    # session balance and the first representative for each tag.
    if len(selected) > count:
        required = set()
        for session in sessions:
            for tag in TAG_COVERAGE:
                match = next((row for row in selected if row["session_id"] == session and tag in row.get("visible_tags", "").split(";")), None)
                if match:
                    required.add((match["session_id"], match["frame_index"]))
        required_rows = [row for row in selected if (row["session_id"], row["frame_index"]) in required]
        remainder = [row for row in selected if (row["session_id"], row["frame_index"]) not in required]
        remainder.sort(key=lambda row: hashlib.sha256(f"{row['session_id']}:{row['frame_index']}".encode()).hexdigest())
        selected = required_rows + remainder[: max(0, count - len(required_rows))]
        selected.sort(key=lambda row: (row["session_id"], int(row["frame_index"])))
    return selected


def _paper_polygon(frame: np.ndarray, likely_skin: np.ndarray) -> tuple[np.ndarray | None, str, str]:
    """Return a detected sheet polygon after excluding likely skin pixels."""
    try:
        from draw.paper import detect_sheet_with_confidence

        corners, verdict = detect_sheet_with_confidence(frame)
        if corners is not None:
            return np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2), "sheet_detector", str(verdict)
    except (ImportError, cv2.error, ValueError):
        pass

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    # The pale sheet is usually the largest low-saturation bright component.
    threshold = max(110, int(np.percentile(value, 72)))
    candidate = ((value >= threshold) & (saturation <= 105) & ~likely_skin).astype(np.uint8) * 255
    # A smaller close joins paper texture gaps without bridging from a page
    # across the removed skin/arm region into nearby bright background.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
    candidate[likely_skin] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "none", "no_candidate"
    h, w = frame.shape[:2]
    min_area = h * w * 0.07
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return None, "none", "candidate_too_small"
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for epsilon in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
        approximation = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            quad = approximation
            break
    # The sheet is approximately rectangular. Prefer its four outer corners
    # to a convex hull around incidental bright regions; skin was excluded
    # before this geometric step and is removed again from the raster mask.
    polygon = quad if quad is not None else hull
    return polygon, "skin_suppressed_bright_quad" if quad is not None else "skin_suppressed_bright_hull", "medium" if quad is not None else "low"


def _skin_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce hand core, uncertain boundary, and broad skin-suppression mask."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    skin_hue = (hue <= 25) | (hue >= 170)
    core_candidate = (
        (y >= 45) & (cr >= 133) & (cr <= 180) & (cb >= 75) & (cb <= 138)
        & (sat >= 25) & skin_hue
    ).astype(np.uint8) * 255
    # Broaden the skin range for polygon suppression. These pixels are not
    # automatically classed as hand; only their eroded core is class 2.
    likely_skin = (
        (y >= 35) & (cr >= 128) & (cr <= 185) & (cb >= 68) & (cb <= 138)
        & (sat >= 30) & skin_hue
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_candidate = cv2.morphologyEx(core_candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    likely_skin = cv2.morphologyEx(likely_skin, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(likely_skin, 8)
    min_area = max(80, int(frame.shape[0] * frame.shape[1] * 0.00015))
    kept_skin = np.zeros_like(likely_skin)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            kept_skin[labels == label] = 255
    strict_core = cv2.bitwise_and(core_candidate, kept_skin)
    strict_core = cv2.morphologyEx(strict_core, cv2.MORPH_CLOSE, kernel)
    core = cv2.erode(strict_core, np.ones((5, 5), np.uint8), iterations=1)
    likely = kept_skin > 0
    edge = likely & (core == 0)
    return core > 0, edge, likely


def propose_mask(frame: np.ndarray, tags: set[str]) -> tuple[np.ndarray, dict[str, str]]:
    h, w = frame.shape[:2]
    mask = np.full((h, w), 0, dtype=np.uint8)
    hand_core, hand_edge, likely_skin = _skin_masks(frame)
    polygon, paper_method, paper_confidence = _paper_polygon(frame, likely_skin)
    if polygon is None:
        mask.fill(255)
        # Retain only high-confidence hand-core pixels when the image review
        # already confirms a visible hand. The rest stays unknown because no
        # sheet/background boundary was established.
        hand_confirmed = bool({"hand_on_paper", "hand_elsewhere"} & tags)
        if hand_confirmed:
            mask[hand_core] = 2
        return mask, {
            "paper_method": paper_method,
            "paper_confidence": paper_confidence,
            "proposal_status": "partial_hand_only_no_paper_polygon" if hand_confirmed else "unknown_no_paper_polygon",
            "ink_method": "not_run",
        }

    paper_fill = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(paper_fill, [polygon], 255)
    # Never retain likely skin/arm as paper, even if the geometric proposal
    # straddles an occlusion or the page contour touches the arm.
    paper_fill[likely_skin] = 0
    # A geometric polygon can include desk or an arm when the sheet is partly
    # occluded. Only call pixels paper when their appearance also matches the
    # pale, low-saturation sheet; keep the rest of the polygon unknown.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    value_floor = max(110, int(np.percentile(value, 72)))
    paper_appearance = (value >= value_floor) & (saturation <= 95)
    paper_candidate = cv2.bitwise_and(
        paper_fill,
        paper_appearance.astype(np.uint8) * 255,
    )
    paper_core = cv2.erode(paper_candidate, np.ones((5, 5), np.uint8), iterations=1) > 0
    paper_region = paper_fill > 0
    paper_edge = (paper_candidate > 0) & ~paper_core
    mask[paper_region] = 255
    mask[paper_core] = 1
    mask[paper_edge] = 255

    mask[hand_edge] = 255
    mask[hand_core] = 2

    ink_method = "none_image_tag_absent"
    if "user_added_ink" in tags:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        darkness = cv2.subtract(background, gray)
        ink_candidate = (darkness >= 16) & paper_core & ~likely_skin
        ink_candidate = cv2.morphologyEx(
            ink_candidate.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        ) > 0
        if "printed_text" in tags:
            # The dark-pixel proposal cannot separate handwriting from print.
            # Keep the area explicitly unknown rather than calling print ink.
            mask[ink_candidate] = 255
            ink_method = "dark_stroke_unknown_print_conflict"
        else:
            ink_ring = cv2.dilate(ink_candidate.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
            ink_edge = ink_ring & ~ink_candidate & paper_core & ~likely_skin
            mask[ink_edge] = 255
            mask[ink_candidate] = 3
            ink_method = "local_dark_stroke_proposal"

    # Visible hand pixels override both paper and ink, per the project rule.
    mask[likely_skin & ~hand_core] = 255
    mask[hand_core] = 2
    return mask, {
        "paper_method": paper_method,
        "paper_confidence": paper_confidence,
        "paper_value_floor": str(value_floor),
        "paper_saturation_ceiling": "95",
        "proposal_status": "draft_requires_visual_review",
        "ink_method": ink_method,
    }


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_overlay(frame: np.ndarray, mask: np.ndarray, path: Path) -> None:
    color = np.zeros_like(frame)
    for class_id, bgr in PALETTE.items():
        color[mask == class_id] = bgr
    overlay = cv2.addWeighted(frame, 0.63, color, 0.37, 0)
    unknown = mask == 255
    if np.any(unknown):
        overlay[unknown] = (overlay[unknown].astype(np.uint16) * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
    cv2.imwrite(str(path), overlay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all-selected", action="store_true",
        help="process every keep_productive frame instead of the 20-frame pilot",
    )
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error("--output-dir must stay inside the repository")
    rows = read_rows(args.input)
    if not args.all_selected:
        rows = choose_pilot(rows)
    ids = [(row["session_id"], row["frame_index"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("selected input contains duplicate session/frame IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    aggregate: Counter[int] = Counter()

    for row in rows:
        source = Path(row["source_path"])
        if not source.is_absolute():
            source = ROOT / source
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"could not decode source image: {source}")
        tags = {tag.strip() for tag in row.get("visible_tags", "").split(";") if tag.strip()}
        mask, methods = propose_mask(frame, tags)
        stem = f"{row['session_id']}_frame_{int(row['frame_index']):08d}"
        mask_path = args.output_dir / f"{stem}_mask.png"
        overlay_path = args.output_dir / f"{stem}_overlay.jpg"
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"failed to write mask: {mask_path}")
        save_overlay(frame, mask, overlay_path)
        values, pixel_counts = np.unique(mask, return_counts=True)
        counts = {int(value): int(count) for value, count in zip(values, pixel_counts)}
        aggregate.update(counts)
        pixel_total = int(mask.size)
        fractions = {class_id: counts.get(class_id, 0) / pixel_total for class_id in (0, 1, 2, 3, 255)}
        problems = []
        if "no_paper_polygon" in methods.get("proposal_status", ""):
            problems.append("paper_not_localized")
        if methods.get("paper_confidence") in {"low", "medium"}:
            problems.append("paper_geometry_low_confidence")
        if fractions[255] >= 0.5:
            problems.append("at_least_half_pixels_ignored")
        if "user_added_ink" in tags and fractions[3] == 0:
            problems.append("ink_tag_without_ink_pixels")
        if {"hand_on_paper", "hand_elsewhere"} & tags and fractions[2] == 0:
            problems.append("hand_tag_without_hand_pixels")
        all_unknown = counts.get(255, 0) == pixel_total
        proposal_status = "non_contributory_all_unknown" if all_unknown else "unverified"
        manifest.append({
            "session_id": row["session_id"],
            "frame_index": row["frame_index"],
            "review_group_id": row.get("review_group_id", ""),
            "source_path": row["source_path"],
            "source_sha256": image_sha256(source),
            "source_manifest_sha256": row.get("source_sha256", ""),
            "mask_path": str(mask_path.relative_to(ROOT)),
            "overlay_path": str(overlay_path.relative_to(ROOT)),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "tags_used_as_proposal_hints_only": ";".join(sorted(tags)),
            "annotation_status": proposal_status,
            "contributes_supervised_pixels": not all_unknown,
            "problem_flags": ";".join(problems),
            **methods,
            "pixel_counts": ";".join(f"{class_id}:{counts.get(class_id, 0)}" for class_id in (0, 1, 2, 3, 255)),
            **{f"fraction_class_{class_id}": f"{fractions[class_id]:.8f}" for class_id in (0, 1, 2, 3)},
            "ignore_fraction": f"{fractions[255]:.8f}",
        })

    manifest_path = args.output_dir / "proposal_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    non_contributory = sum(row["annotation_status"] == "non_contributory_all_unknown" for row in manifest)
    print(f"Unverified proposal frames: {len(manifest)}; non-contributory all-unknown: {non_contributory}; output: {args.output_dir}")
    print("Aggregate pixels:", dict(aggregate))
    print("WARNING: pseudo-labels are unverified experiments, not ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
