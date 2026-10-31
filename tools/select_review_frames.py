#!/usr/bin/env python3
"""Build a compact, auditable review queue from classify_frames.py output.

This creates review candidates and dispositions only. It never deletes or edits
source frames, annotations, or remote data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WINDOW_NS = 2_000_000_000
EVENT_WINDOW_NS = 500_000_000
CORE_CATEGORIES = ("paper_with_user_marks", "hand_over_paper", "pen_without_hand")
SECONDARY_CATEGORIES = ("printed_paper", "blank_paper", "hand_elsewhere", "unusable_quality")


def p_hash(path: Path) -> int | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    values = dct.flatten()[1:]
    bits = values > float(np.median(values))
    return sum(int(bit) << i for i, bit in enumerate(bits))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hamming(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return (a ^ b).bit_count()


def source_key(row: dict[str, str]) -> tuple[str, str]:
    return row["session_id"], row["frame_index"]


def read_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    return rows


def make_groups(rows: list[dict[str, str]]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    session_start: dict[str, int] = {}
    for row in rows:
        if row.get("timestamp_ns"):
            value = int(row["timestamp_ns"])
            session_start[row["session_id"]] = min(session_start.get(row["session_id"], value), value)

    for row in rows:
        if row.get("take_id"):
            key = ("take", row["session_id"], row.get("batch_id", ""), row["take_id"])
            row["review_group_type"] = "take"
            row["review_group_id"] = f"batch_{row.get('batch_id') or 'na'}_take_{row['take_id']}"
        else:
            start = session_start.get(row["session_id"])
            timestamp = int(row["timestamp_ns"]) if row.get("timestamp_ns") else 0
            segment = max(0, (timestamp - start) // WINDOW_NS) if start is not None else max(0, (int(row["frame_index"]) - 1) // 30)
            key = ("review_segment", row["session_id"], str(segment))
            row["review_group_type"] = "review_segment"
            row["review_group_id"] = f"segment_{segment + 1:03d}"
        groups[key].append(row)

    for group in groups.values():
        group.sort(key=lambda row: (int(row.get("timestamp_ns") or 0), int(row["frame_index"])))
    return groups


def add_reason(row: dict[str, object], reason: str, priority: int, score: float = 0.0) -> None:
    reasons = row.setdefault("selection_reasons", set())
    reasons.add(reason)
    row["selection_priority"] = min(int(row.get("selection_priority", priority)), priority)
    row["selection_score"] = max(float(row.get("selection_score", 0.0)), score)


def category_candidate(group: list[dict[str, str]], category: str) -> dict[str, str] | None:
    column = f"clip_{category}"
    candidates = [row for row in group if row.get("semantic_candidate") == category and row.get(column)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row[column]), float(row.get("clip_margin_top2") or 0), float(row.get("sharpness_laplacian") or 0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "captures/triage/frames.csv")
    parser.add_argument("--manifest-output", type=Path, default=ROOT / "captures/triage/selection_manifest.csv")
    parser.add_argument("--review-output", type=Path, default=ROOT / "captures/triage/review_queue.csv")
    parser.add_argument("--overflow-output", type=Path, default=ROOT / "captures/triage/overflow_review_queue.csv")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "captures/triage/selection_summary.csv")
    parser.add_argument("--max-review-queue", type=int, default=650)
    parser.add_argument("--limit", type=int, help="process first N rows only for a quick local check")
    parser.add_argument("--event-phash-distance", type=int, default=8)
    parser.add_argument("--protect-frame", action="append", default=[], metavar="SESSION:INDEX:REASON", help="force a known review example into the queue; repeatable")
    args = parser.parse_args()
    if args.max_review_queue < 1:
        parser.error("--max-review-queue must be positive")

    rows = read_rows(args.input, args.limit)
    groups = make_groups(rows)
    phashes: dict[tuple[str, str], int | None] = {}
    checksums: dict[tuple[str, str], str] = {}
    missing: list[tuple[str, str]] = []
    for row in rows:
        key = source_key(row)
        raw_path = Path(row["source_path"])
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        if path.is_file():
            phashes[key] = p_hash(path)
            checksums[key] = sha256(path)
        else:
            phashes[key] = None
            checksums[key] = ""
            missing.append(key)

    selected: dict[tuple[str, str], dict[str, object]] = {}
    candidates: dict[tuple[str, str], dict[str, object]] = {}
    suppressed_reasons: dict[tuple[str, str], str] = {}
    for group_key, group in groups.items():
        first, middle, last = group[0], group[len(group) // 2], group[-1]
        for row, reason in ((first, "anchor:first"), (middle, "anchor:middle"), (last, "anchor:last")):
            key = source_key(row)
            item = selected.setdefault(key, {"row": row, "selection_priority": 0, "selection_score": 0.0, "selection_reasons": set()})
            add_reason(item, reason, 0)

        # Add one winning frame for each core category not already represented
        # by the anchors. Predictions stay candidates; a low CLIP margin never
        # blocks inclusion.
        represented = {row.get("semantic_candidate") for row in (first, middle, last)}
        for category in CORE_CATEGORIES:
            if category in represented:
                continue
            candidate = category_candidate(group, category)
            if candidate is None:
                continue
            key = source_key(candidate)
            existing = selected.get(key) or candidates.setdefault(
                key,
                {"row": candidate, "selection_priority": 1, "selection_score": float(candidate.get(f"clip_{category}") or 0), "selection_reasons": set()},
            )
            add_reason(existing, f"semantic:{category}", 1, float(candidate.get(f"clip_{category}") or 0))

        # Compare frames to the previous retained representative, not only to
        # their immediate neighbor. Choose the strongest non-anchor change in
        # each take/window and replace its routine midpoint when possible.
        retained = set(selected)
        retained.update(key for key, item in candidates.items() if int(item["selection_priority"]) <= 1)
        previous_retained: dict[str, str] | None = None
        changes: list[tuple[int, dict[str, str], dict[str, str]]] = []
        for row in group:
            key = source_key(row)
            if key in retained:
                previous_retained = row
                continue
            if previous_retained is None:
                continue
            distance = hamming(phashes.get(key), phashes.get(source_key(previous_retained)))
            if distance is not None:
                changes.append((distance, row, previous_retained))
        if changes:
            distance, event, previous_event = max(
                changes,
                key=lambda item: (
                    item[0],
                    1 if item[1].get("semantic_candidate") in CORE_CATEGORIES else 0,
                    float(item[1].get(f"clip_{item[1].get('semantic_candidate')}") or 0),
                ),
            )
            if distance >= args.event_phash_distance:
                event_key = source_key(event)
                middle_key = source_key(middle)
                middle_item = selected.get(middle_key)
                if middle_item and middle_key not in {source_key(first), source_key(last)} and middle_item["selection_reasons"] == {"anchor:middle"}:
                    del selected[middle_key]
                    candidates.setdefault(middle_key, {"row": middle, "selection_priority": 3, "selection_score": 0.0, "selection_reasons": set()})
                    add_reason(candidates[middle_key], "routine_anchor:replaced_by_visual_event", 3)
                    event_item = candidates.pop(event_key, None) or selected.get(event_key) or {
                        "row": event, "selection_priority": 0, "selection_score": float(distance), "selection_reasons": set()
                    }
                    add_reason(event_item, f"event:change_from_last_retained_{previous_event['frame_index']}", 0, float(distance))
                    selected[event_key] = event_item
                elif event_key in selected:
                    add_reason(selected[event_key], f"event:change_from_last_retained_{previous_event['frame_index']}", 0, float(distance))
                else:
                    event_item = candidates.setdefault(event_key, {"row": event, "selection_priority": 1, "selection_score": float(distance), "selection_reasons": set()})
                    add_reason(event_item, f"event:change_from_last_retained_{previous_event['frame_index']}", 1, float(distance))

        # Keep short hand-over-paper events that a single group winner would
        # miss. Pick at most one candidate per half-second window, measured
        # against the most recent representative retained before that window.
        # This is still a review proposal, never an annotation or deletion.
        group_start = int(group[0].get("timestamp_ns") or 0)
        event_windows: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            if row.get("semantic_candidate") != "hand_over_paper":
                continue
            timestamp = int(row.get("timestamp_ns") or group_start)
            event_windows[max(0, (timestamp - group_start) // EVENT_WINDOW_NS)].append(row)

        temporal_representatives = {
            key for key in selected
            if key in {source_key(row) for row in group}
        }
        temporal_representatives.update(
            key for key, item in candidates.items()
            if int(item["selection_priority"]) <= 1 and key in {source_key(row) for row in group}
        )
        for window_index, event_rows in sorted(event_windows.items()):
            window_start = group_start + window_index * EVENT_WINDOW_NS
            prior = [row for row in group if int(row.get("timestamp_ns") or group_start) < window_start and source_key(row) in temporal_representatives]
            baseline = prior[-1] if prior else group[0]
            scored = [(hamming(phashes.get(source_key(row)), phashes.get(source_key(baseline))), row) for row in event_rows]
            scored = [(distance, row) for distance, row in scored if distance is not None]
            if not scored:
                continue
            distance, event = max(scored, key=lambda item: (item[0], -int(item[1]["frame_index"]), float(item[1].get("sharpness_laplacian") or 0)))
            if distance < args.event_phash_distance:
                continue
            key = source_key(event)
            item = selected.get(key) or candidates.setdefault(
                key,
                {"row": event, "selection_priority": 1, "selection_score": float(distance), "selection_reasons": set()},
            )
            add_reason(item, f"event:hand_over_paper_500ms_from_{baseline['frame_index']}", 1, float(distance))
            temporal_representatives.add(key)
            # Avoid retaining a neighboring semantic winner for the same
            # event category/window when the novelty winner is stronger.
            for duplicate in event_rows:
                duplicate_key = source_key(duplicate)
                if duplicate_key == key:
                    continue
                duplicate_item = selected.get(duplicate_key) or candidates.get(duplicate_key)
                if duplicate_item is None:
                    continue
                reasons = duplicate_item["selection_reasons"]
                if any(str(reason).startswith(("anchor:", "protected:")) for reason in reasons):
                    continue
                selected.pop(duplicate_key, None)
                candidates.pop(duplicate_key, None)
                suppressed_reasons[duplicate_key] = f"duplicate_hand_over_window_{window_index}:preferred_{event['frame_index']}"

        # Lower-priority class coverage fills remaining room after anchors and
        # change/event candidates. Categories such as printed/blank paper are
        # still represented in overflow rather than silently discarded.
        represented = {item["row"].get("semantic_candidate") for item in selected.values()}
        for category in SECONDARY_CATEGORIES:
            if category in represented:
                continue
            candidate = category_candidate(group, category)
            if candidate is None:
                continue
            key = source_key(candidate)
            existing = selected.get(key) or candidates.setdefault(
                key,
                {"row": candidate, "selection_priority": 3, "selection_score": float(candidate.get(f"clip_{category}") or 0), "selection_reasons": set()},
            )
            add_reason(existing, f"semantic:{category}", 3, float(candidate.get(f"clip_{category}") or 0))

        # The category candidate can coincide with the transition candidate.
        for row in group:
            key = source_key(row)
            if key in selected and key in candidates:
                for reason in candidates[key]["selection_reasons"]:
                    add_reason(selected[key], reason, int(candidates[key]["selection_priority"]), float(candidates[key]["selection_score"]))
                del candidates[key]

    # Promote explicitly reviewed hard cases after the general selector has
    # made its choices. A protected example can replace a routine midpoint.
    for value in args.protect_frame:
        try:
            session_id, frame_index, reason = value.rsplit(":", 2)
        except ValueError:
            parser.error(f"invalid --protect-frame: {value!r}; expected SESSION:INDEX:REASON")
        key = (session_id, frame_index)
        if key not in phashes:
            parser.error(f"protected source frame not found in input: {session_id}/{frame_index}")
        group = next(group for group in groups.values() if any(source_key(row) == key for row in group))
        protected = next(row for row in group if source_key(row) == key)
        middle = group[len(group) // 2]
        middle_key = source_key(middle)
        if middle_key in selected and middle_key not in {source_key(group[0]), source_key(group[-1])} and selected[middle_key]["selection_reasons"] == {"anchor:middle"}:
            del selected[middle_key]
            candidates.setdefault(middle_key, {"row": middle, "selection_priority": 3, "selection_score": 0.0, "selection_reasons": set()})
            add_reason(candidates[middle_key], "routine_anchor:replaced_by_protected_example", 3)
        protected_item = candidates.pop(key, None) or selected.get(key) or {
            "row": protected, "selection_priority": 0, "selection_score": 1.0, "selection_reasons": set()
        }
        add_reason(protected_item, f"protected:{reason}", 0, 1.0)
        selected[key] = protected_item

    # Mandatory anchors and protected examples are always first. Fill the main
    # review queue with event and semantic candidates; overflow remains an
    # explicit second-tier list and is never silently dropped.
    mandatory_keys = set(selected)
    ordered_candidates = sorted(candidates.items(), key=lambda pair: (int(pair[1]["selection_priority"]), -float(pair[1]["selection_score"]), int(pair[0][1])))
    tier1_keys = set(mandatory_keys)
    overflow_keys: set[tuple[str, str]] = set()
    for key, _ in ordered_candidates:
        if len(tier1_keys) < args.max_review_queue:
            tier1_keys.add(key)
        else:
            overflow_keys.add(key)
    chosen = {**selected, **candidates}

    # For every source frame, store its distance to the last retained frame
    # in the same take/segment plus its nearest selected representative.
    manifest_rows: list[dict[str, object]] = []
    for group_key, group in groups.items():
        kept = sorted((row for row in group if source_key(row) in tier1_keys or source_key(row) in overflow_keys), key=lambda row: int(row["frame_index"]))
        last_kept: dict[str, str] | None = None
        for row in group:
            key = source_key(row)
            current_hash = phashes.get(key)
            distance_to_last = hamming(current_hash, phashes.get(source_key(last_kept))) if last_kept else None
            distances = [(hamming(current_hash, phashes.get(source_key(candidate))), candidate) for candidate in kept]
            distances = [(distance, candidate) for distance, candidate in distances if distance is not None]
            nearest_distance, nearest = min(distances, key=lambda item: (item[0], abs(int(item[1]["frame_index"]) - int(row["frame_index"])))) if distances else (None, group[0])
            item = chosen.get(key)
            reasons = sorted(item["selection_reasons"]) if item else ([suppressed_reasons[key]] if key in suppressed_reasons else [])
            if key in tier1_keys:
                selection_status = "review_selected"
                review_tier = 1
                last_kept = row
                disposition = "selected_for_compact_review"
            elif key in overflow_keys:
                selection_status = "overflow_candidate"
                review_tier = 2
                last_kept = row
                disposition = "preserved_for_second_review_tier"
            elif key in suppressed_reasons:
                selection_status = "not_selected_same_event_window"
                review_tier = ""
                disposition = "neighboring_event_candidate_suppressed"
            elif nearest_distance is not None and nearest_distance <= 3:
                selection_status = "not_selected_near_representative"
                review_tier = ""
                disposition = "near_selected_representative_candidate"
            else:
                selection_status = "not_selected_for_review"
                review_tier = ""
                disposition = "not_in_compact_review_queue"

            record = dict(row)
            record.update({
                "source_sha256": checksums.get(key, ""),
                "review_group_type": row.get("review_group_type", group_key[0]),
                "review_group_id": row.get("review_group_id", ""),
                "selection_status": selection_status,
                "review_tier": review_tier,
                "selection_reasons": ";".join(reasons),
                "disposition": disposition,
                "nearest_selected_frame_index": nearest["frame_index"],
                "nearest_selected_phash_distance": nearest_distance if nearest_distance is not None else "",
                "phash_distance_to_last_selected": distance_to_last if distance_to_last is not None else "",
                "phash_hex": f"{current_hash:016x}" if current_hash is not None else "",
                "semantic_ambiguity": "ambiguous" if float(row.get("clip_margin_top2") or 1) < 0.02 else "not_flagged",
                "review_state": "pending" if selection_status in {"review_selected", "overflow_candidate"} else "not_queued",
            })
            manifest_rows.append(record)

    def write_csv(path: Path, items: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for item in items for key in item))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)

    by_key = {source_key(record): record for record in manifest_rows}
    selected_rows = [by_key[key] for key in tier1_keys]
    overflow_rows = [by_key[key] for key in overflow_keys]
    selected_rows.sort(key=lambda row: (row["session_id"], row.get("batch_id", ""), row.get("take_id", ""), int(row["frame_index"])))
    overflow_rows.sort(key=lambda row: (row["session_id"], row.get("batch_id", ""), row.get("take_id", ""), int(row["frame_index"])))
    manifest_rows.sort(key=lambda row: (row["session_id"], int(row["frame_index"])))
    write_csv(args.manifest_output, manifest_rows)
    write_csv(args.review_output, selected_rows)
    write_csv(args.overflow_output, overflow_rows)

    summary: list[dict[str, object]] = []
    for group_key, group in groups.items():
        selected_count = sum(source_key(row) in tier1_keys for row in group)
        overflow_count = sum(source_key(row) in overflow_keys for row in group)
        reasons = sorted({reason for row in group if source_key(row) in chosen for reason in chosen[source_key(row)]["selection_reasons"]})
        summary.append({
            "session_id": group_key[1],
            "group_type": group_key[0],
            "batch_id": group_key[2] if group_key[0] == "take" else "",
            "take_id": group_key[3] if group_key[0] == "take" else "",
            "review_group_id": group[0].get("review_group_id", ""),
            "take_status": group[0].get("take_status", ""),
            "source_frames": len(group),
            "review_selected": selected_count,
            "overflow_candidates": overflow_count,
            "selection_reasons": ";".join(reasons),
        })
    write_csv(args.summary_output, summary)

    print(f"Source rows: {len(rows)}; unique IDs: {len({source_key(row) for row in rows})}")
    print(f"Groups: {len(groups)}; review queue: {len(tier1_keys)}; overflow: {len(overflow_keys)}; missing source files: {len(missing)}")
    print(f"Manifest: {args.manifest_output}\nReview: {args.review_output}\nOverflow: {args.overflow_output}\nSummary: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
