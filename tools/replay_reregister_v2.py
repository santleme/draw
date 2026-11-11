"""F-07 v2: fixed-anchor contract replay; ECC state is always explicit.

Cached maps are anchor-basis fixtures, not physical camera textures. The
observation is expressed in the current pass basis before the unmodified
production registration. Consecutive quads are diagnostic only.

Same path as v1: the real app loop (`draw.loop.SheetInkLoop`) and the real
`_PageCleanWorker` (synchronous `pump()`, scene clock); only the page_clean
observation and the per-pass diagnostic record are v2-specific.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import platform
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import ink_layers as ink  # noqa: E402
from tools import replay_reregister as v1  # noqa: E402

PROTOCOL = Path(__file__).resolve().parents[1] / (
    "docs/F07_REPLAY_V2_A_R1_2026-11-09.json"
)
PROTOCOL_SHA256 = "aa6b9d79d805d4f0c0527d33c618253645882ea0c120cc92a0d8675e357fc81c"
OUTCOMES = ("passed", "failed", "decision_deferred", "vacuous", "invalid")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require_ecc(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError("declare ecc_refine_enabled explicitly as integer 0 or 1")
    return value


@contextmanager
def configured_ecc(ecc_refine_enabled):
    enabled = require_ecc(ecc_refine_enabled)
    values = {"DRAW_ECC_REFINE": str(enabled), "DRAW_IDENTITY_CHAMFER": "0",
              "DRAW_PAGE_TURN_EVENT": "0"}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_protocol(path=PROTOCOL):
    raw = Path(path).read_bytes()
    if sha256(raw) != PROTOCOL_SHA256:
        raise ValueError("preregistration changed: create a new named protocol first")
    protocol = json.loads(raw)
    for name, expected in protocol["production_source_sha256"].items():
        if sha256(inspect.getsource(getattr(ink, name)).encode()) != expected:
            raise ValueError(f"production helper changed: {name}; preregister again")
    scenario = protocol["scenarios"]
    constants = {
        "fps": v1.FPS, "passes_per_warp": v1.PASSES_PER_WARP,
        "settle_frames": v1.SETTLE_FRAMES, "sway_frames": v1.SWAY_FRAMES,
        "sway_amplitude_camera_px": v1.SWAY_AMPLITUDE_PX,
        "detector_noise_camera_px": v1.DETECTOR_NOISE_PX, "detect_ms": v1.DETECT_MS,
        "hand_configs": list(v1.HAND_CONFIGS),
        "move_same_canvas": list(v1.MOVES), "sway_negative": list(v1.SWAY_PROFILES),
        "moves": json.loads(json.dumps(v1.MOVES)),
    }
    if any(scenario[key] != value for key, value in constants.items()):
        raise ValueError("scenario drift: preregister changed constants first")
    if v1.APP_LOGIC != "main" or v1.BOUNDARY_TOL_PX != 3:
        raise ValueError("production/replay criteria drift")
    return protocol


def load_bench(root, protocol, *, package_path):
    """Read-only cache hits; never imports the private cache's inference code."""
    package_bytes = Path(package_path).read_bytes()
    if sha256(package_bytes) != protocol["baseline_input"]["package_sha256"]:
        raise ValueError("package SHA-256 differs from preregistration")
    with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
        return _load_registered_bench(root, protocol, archive)


def _load_registered_bench(root, protocol, archive):
    root = Path(root)
    expected = protocol["baseline_input"]
    prefix = Path(expected["package"]).stem

    def read_registered(relative):
        data = (root / relative).read_bytes()
        if sha256(data) != sha256(archive.read(f"{prefix}/{relative}")):
            raise ValueError(f"package member mismatch: {relative}")
        return data

    index_raw = read_registered("cache/cache_index.json")
    model_hash = sha256(read_registered("model/page_clean_v2_fixed.onnx"))
    if (sha256(index_raw) != expected["cache_index_sha256"]
            or model_hash != expected["model_sha256"]):
        raise ValueError("dataset/model differs from preregistration")
    index = json.loads(index_raw)
    entries, maps = [], []
    for i in range(1, 31):
        name = f"warp:f{i:02d}"
        entry = index[name]
        image_bytes = read_registered(f"warps/f{i:02d}_warp.png")
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None or list(img.shape) != entry["shape"]:
            raise ValueError(f"missing/invalid warp: {name}")
        key = sha256(bytes.fromhex(model_hash) + repr(img.shape).encode() + img.tobytes())
        if key != entry["key"] or entry["model_sha256"] != model_hash:
            raise ValueError(f"cache key mismatch: {name}")
        cache_bytes = read_registered(f"cache/{key}.npy")
        prob = np.load(io.BytesIO(cache_bytes), allow_pickle=False)
        if (prob.dtype != np.float32 or prob.shape != (1216, 864)
                or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1):
            raise ValueError(f"invalid native probability map: {name}")
        entries.append({"name": name, "cache_key": key,
                        "warp_sha256": sha256(image_bytes),
                        "map_sha256": sha256(cache_bytes)})
        maps.append(prob)
    provenance = {
        "source": "real_anchor_basis_cache", "physical_pages": 1,
        "page_id": "bench-2026-11-07-f01-f30", "cache_hits": 30, "cache_misses": 0,
        "native_grid": [864, 1216], "model_sha256": model_hash,
        "package_sha256": expected["package_sha256"],
        "cache_index_sha256": sha256(index_raw), "entries": entries,
    }
    provenance["input_fingerprint"] = sha256(
        json.dumps(provenance, sort_keys=True).encode()
    )
    return [[prob] for prob in maps], maps[-3:], provenance


def corner_displacements(matrix, shape):
    h, w = shape
    corners = np.float64([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]])
    moved = cv2.perspectiveTransform(corners, matrix)
    distances = np.linalg.norm((moved - corners) * [850.0 / w, 1200.0 / h], axis=2)
    return {"median": float(np.median(distances)), "max": float(distances.max())}


def diagnostic_consecutive(previous, current, shape):
    if previous is None:
        return None
    matrix = ink._registration_matrix(previous, current, shape)
    return corner_displacements(matrix, shape)


class ReplayPageClean(v1.ScenePageClean):
    """Observation in the current pass basis, read from the real worker.

    The anchor is the production worker's own state at pass time:
    the identity reference on the first pass of a generation, the
    generation anchor otherwise.
    """

    def __init__(self, *, ecc_refine_enabled):
        super().__init__()
        self.ecc_refine_enabled = require_ecc(ecc_refine_enabled)
        self.previous_quad = None
        self._geometry = None

    def observe(self, page, true_quad, corners):
        if os.environ.get("DRAW_ECC_REFINE") != str(self.ecc_refine_enabled):
            raise ValueError("ECC environment does not match explicit arm")
        worker = self.worker
        first_pass = worker._generation_passes == 0
        anchor = (worker._identity_reference_corners if first_pass
                  else worker._anchor_corners)
        if anchor is None:
            anchor = corners
        registration = ink._registration_matrix(anchor, corners, page.shape)
        observation = np.linalg.inv(registration)
        self._geometry = {
            "roundtrip_corner_max_canonical_px": corner_displacements(
                registration @ observation, page.shape
            )["max"],
            "fixed_anchor_quad_displacement_canonical_px": corner_displacements(
                registration, page.shape
            ),
            "consecutive_quad_displacement_canonical_px": diagnostic_consecutive(
                self.previous_quad, corners, page.shape
            ),
        }
        self.previous_quad = corners.copy()
        return cv2.warpPerspective(page, observation, (page.shape[1], page.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderValue=0.0)


def _ecc_fields(record):
    return {key: value for key, value in record.items() if key.startswith("ecc_")}


class ReplayApp(v1.ReplayApp):
    """The real loop (v1.ReplayApp) with one diagnostic record per pass."""

    def __init__(self, hand_state, turn_event_enabled=False, *, ecc_refine_enabled):
        if turn_event_enabled:
            raise ValueError("v2 R1 does not enable the experimental turn flag")
        self.ecc_refine_enabled = require_ecc(ecc_refine_enabled)
        self.pass_records = []
        super().__init__(hand_state, False, page_clean_factory=partial(
            ReplayPageClean, ecc_refine_enabled=self.ecc_refine_enabled))

    def pump_ink(self):
        identity_before = len(self.worker._identity_events)
        anchor = self.worker._anchor_corners
        anchor_before = None if anchor is None else anchor.copy()
        ran = super().pump_ink()
        if ran:
            identity = list(self.worker._identity_events)[identity_before:]
            anchor_after = self.worker._anchor_corners
            self.pass_records.append({
                "pass_seq": self.worker._passes_total,
                "ecc_refine_enabled": self.ecc_refine_enabled,
                "DRAW_ECC_REFINE": str(self.ecc_refine_enabled),
                **_ecc_fields(self.worker._telemetry_records[-1]),
                **self.worker._page_clean._geometry,
                "identity": identity_measurements(identity, self.ecc_refine_enabled),
                "same_page_anchor_unchanged": (
                    not any(e["decision"] == "same_page" for e in identity)
                    or (anchor_before is not None and anchor_after is not None
                        and np.array_equal(anchor_before, anchor_after))
                ),
            })
        return ran


def identity_measurements(events, ecc_refine_enabled):
    enabled = require_ecc(ecc_refine_enabled)
    return [{**e, "ecc_refine_enabled": enabled, "DRAW_ECC_REFINE": str(enabled),
             "decision": "decision_deferred" if e.get("ecc_rejected") else e["decision"],
             "ecc_rejected": bool(e.get("ecc_rejected")),
             "diagnostic_production_decision": e["decision"]} for e in events]


def finalize_episode(record):
    """Only here may raw diagnostics become a decisive observation."""
    diag = record["diagnostics"]
    if diag["baseline_rejections"] or any(p["ecc_rejected"] for p in diag["passes"]):
        outcome = "decision_deferred"
    elif record["vacuous"]:
        outcome = "vacuous"
    elif any(not np.isfinite(p["roundtrip_corner_max_canonical_px"])
             or p["roundtrip_corner_max_canonical_px"] > 1e-6
             or not p["same_page_anchor_unchanged"] for p in diag["passes"]):
        outcome = "invalid"
    else:
        outcome = "passed" if (
            record["canvas_identical_after_move"] and record["generations_created"] == 0
            and not any(e["decision"] == "new_page" for e in diag["identity"])
        ) else "failed"
    record["outcome"] = outcome
    eligible = outcome in ("passed", "failed")
    record["passes_gate"] = outcome == "passed" if eligible else None
    record["decisive"] = {
        "denominator": int(eligible),
        "identity_votes": [e["decision"] for e in diag["identity"]] if eligible else [],
        "new_confirmed_px_outside_tol": diag["canvas"]["new_px"] if eligible else None,
        "lost_confirmed_px_outside_tol": diag["canvas"]["lost_px"] if eligible else None,
    }
    # Raw canvas/state changes remain exclusively under diagnostics.
    for key in ("new_confirmed_px_outside_tol", "lost_confirmed_px_outside_tol",
                "canvas_identical_after_move", "generations_created", "false_page_turns",
                "identity_decisions"):
        record.pop(key, None)
    return record


class Episode(v1.Episode):
    def report(self, true_before, true_after, identity_true_class=None):
        app = self.app
        after = app.canvas()
        shape = (self.before if self.before is not None else after).shape
        before = np.zeros(shape, bool) if self.before is None else self.before > 0
        after_mask = np.zeros(shape, bool) if after is None else after > 0
        kernel = np.ones((2 * v1.BOUNDARY_TOL_PX + 1,) * 2, np.uint8)
        new_px, lost_px = v1._mismatch(before, after_mask, np.eye(3), kernel)
        generations = self.app.clean_generation - self.generation_before
        identities = identity_measurements(app.identity_events()[self.identity_before:],
                                           app.ecc_refine_enabled)
        passes = app.pass_records[self.passes_before:]
        return {
            "ecc_refine_enabled": app.ecc_refine_enabled,
            "DRAW_ECC_REFINE": str(app.ecc_refine_enabled),
            "vacuous": not passes, "passes_total": len(passes),
            "frames_total": self.frames,
            "passes_during_perturbation": self.passes_during,
            "canvas_identical_after_move": new_px == lost_px == 0,
            "new_confirmed_px_outside_tol": new_px,
            "lost_confirmed_px_outside_tol": lost_px,
            "generations_created": generations,
            "false_page_turns": sum(e["decision"] == "new_page" for e in identities),
            "identity_decisions": [e["decision"] for e in identities],
            "diagnostics": {
                "baseline_rejections": sum(p["ecc_rejected"] for p in
                                           app.pass_records[:self.passes_before]),
                "baseline_passes": self.passes_before,
                "passes": passes, "identity": identities,
                "production_generations_created": generations,
                "canvas": {"new_px": new_px, "lost_px": lost_px,
                           "confirmed_before": int(np.count_nonzero(self.before)),
                           "confirmed_after": int(np.count_nonzero(after))},
            },
        }


def summarize(records, ecc_refine_enabled):
    enabled = require_ecc(ecc_refine_enabled)
    if any(r["ecc_refine_enabled"] != enabled for r in records):
        raise ValueError("cannot pool different ECC states")
    outcomes = {name: sum(r["outcome"] == name for r in records) for name in OUTCOMES}
    decisive = [r["decisive"] for r in records if r["decisive"]["denominator"]]
    return {
        "ecc_refine_enabled": enabled, "DRAW_ECC_REFINE": str(enabled),
        "episodes": len(records), **outcomes,
        "decisive_denominator": len(decisive),
        "identity_votes": {decision: sum(r["identity_votes"].count(decision) for r in decisive)
                           for decision in ("same_page", "new_page")},
        "new_confirmed_px_outside_tol": sum(r["new_confirmed_px_outside_tol"] for r in decisive),
        "lost_confirmed_px_outside_tol": sum(r["lost_confirmed_px_outside_tol"] for r in decisive),
        "diagnostic_rejected_episode_passes": sum(p["ecc_rejected"] for r in records
                                                 for p in r["diagnostics"]["passes"]),
    }


def runtime():
    return {"python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "opencv_threads": cv2.getNumThreads()}


def run_arm(root, *, ecc_refine_enabled, package_path, protocol_path=PROTOCOL):
    enabled = require_ecc(ecc_refine_enabled)
    protocol = validate_protocol(protocol_path)
    drawing, settled, provenance = load_bench(root, protocol, package_path=package_path)
    with configured_ecc(enabled):
        return _run_page(drawing, settled, provenance, enabled, protocol)


def _run_page(drawing, settled, provenance, enabled, protocol):
    episodes, baselines = [], {}
    for hand, config in v1.HAND_CONFIGS.items():
        base = v1.settle_drawing(drawing, settled, config, False,
                                app_factory=partial(ReplayApp, ecc_refine_enabled=enabled))
        baselines[hand] = base[0].pass_records
        cases = [("control_static", "static", v1.run_control, ())]
        cases += [("move_same_canvas", name, v1.run_move, (spec,))
                  for name, spec in v1.MOVES.items()]
        cases += [("sway_negative", profile, v1.run_sway, (profile,))
                  for profile in v1.SWAY_PROFILES]
        for scenario, name, runner, args in cases:
            record = runner(base, settled, config, *args, episode_factory=Episode)
            record.update({"scenario": scenario, "case": name, "hand": hand,
                           "page_id": provenance["page_id"], "protocol_id": protocol["id"],
                           "input_fingerprint": provenance["input_fingerprint"]})
            episodes.append(finalize_episode(record))
            print(f"ECC={enabled} {hand}/{scenario}/{name}: {record['outcome']}",
                  file=sys.stderr, flush=True)
    groups = {name: summarize([r for r in episodes if r["scenario"] == name], enabled)
              for name in ("move_same_canvas", "control_static", "sway_negative")}
    passes = [p for records in baselines.values() for p in records]
    passes += [p for r in episodes for p in r["diagnostics"]["passes"]]
    return {
        "ruler_version": 2, "decisive_mode": "A_fixed_anchor",
        "ecc_refine_enabled": enabled, "DRAW_ECC_REFINE": str(enabled),
        "DRAW_IDENTITY_CHAMFER": "0", "DRAW_PAGE_TURN_EVENT": "0",
        "protocol_id": protocol["id"], "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256(Path(__file__).read_bytes() + Path(v1.__file__).read_bytes()),
        "production_source_sha256": protocol["production_source_sha256"],
        "runtime": runtime(), "input": provenance, "groups": groups,
        "baselines": baselines, "episodes": episodes,
        "diagnostics": {"passes_total": len(passes),
                        "ecc_rejected_passes": sum(p["ecc_rejected"] for p in passes),
                        "baseline_rejected_passes": sum(p["ecc_rejected"] for records in
                                                        baselines.values() for p in records)},
        "limitations": ["one physical page; other three pool pages unavailable",
                        "anchor-basis map contract, not raw-camera replay",
                        "no live occlusion: no hand mask simulated (visible=None)",
                        "no ECC flag recommendation"],
    }


def compare_arms(off, on):
    if off["ecc_refine_enabled"] != 0 or on["ecc_refine_enabled"] != 1:
        raise ValueError("comparison requires distinct explicit OFF and ON arms")
    for key in ("protocol_sha256", "runner_sha256", "production_source_sha256", "runtime"):
        if off[key] != on[key]:
            raise ValueError(f"unpaired arms: {key}")
    if off["input"]["input_fingerprint"] != on["input"]["input_fingerprint"]:
        raise ValueError("unpaired arms: different pages/maps")
    def cases(arm):
        return [(r["page_id"], r["scenario"], r["case"], r["hand"]) for r in arm["episodes"]]
    if cases(off) != cases(on):
        raise ValueError("unpaired episodes")
    return {"protocol_id": off["protocol_id"], "input": off["input"],
            "arms": {"DRAW_ECC_REFINE=0": off["groups"], "DRAW_ECC_REFINE=1": on["groups"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True,
                        help="original ZIP with the preregistered SHA-256")
    parser.add_argument("--ecc-refine", type=int, choices=(0, 1), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_arm(args.bench_root, ecc_refine_enabled=args.ecc_refine,
                     package_path=args.package)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
