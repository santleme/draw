"""Régua de re-registro (tools/replay_reregister.py).

A régua só vale se medir o código do app e se o comparador "a menos
de re-warp" não inventar nem esconder diferença. Estes testes cobram
que o replay dirige o laço real (`draw.loop.SheetInkLoop`) e a
`_PageCleanWorker` real, que o caminho síncrono (`pump()`) dá o mesmo
resultado da thread, e o comparador — sem o draw-lab e em escala
reduzida; o veredito da régua (que hoje
reprova o comportamento do app) NÃO é cobrado aqui — ele é a
baseline documentada em docs/FALHAS.md, e a correção futura muda
os números sem quebrar este arquivo.
"""

import copy
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from draw import app as app_module  # noqa: E402
from draw import ink_layers as ink_layers_module  # noqa: E402
from draw import loop as loop_module  # noqa: E402


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "replay_reregister", REPO / "tools" / "replay_reregister.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = _load_tool()
SHAPE = (120, 85)


class _ProbIsWarp:
    """page_clean falso: o "warp" submetido já é o mapa."""

    threshold = 0.7

    def ink_prob(self, warped):
        return warped


def _drive_threaded(sequence, recover_at=None, extra=()):
    """A trabalhadora em thread, como no app: uma passada por submit."""
    real = app_module._PageCleanWorker(_ProbIsWarp()).start()
    try:
        generation = real.new_generation()
        for i, (prob, corners) in enumerate(list(sequence) + list(extra)):
            if i == recover_at:
                generation = real.new_generation(preserve_identity=True)
            before = real.stats()["passes_total"]
            real.submit(prob, generation, corners)
            deadline = time.monotonic() + 5.0
            while real.stats()["passes_total"] <= before:
                assert time.monotonic() < deadline, "passada nao rodou"
                time.sleep(0.001)
        _version, canvas = real.poll(generation, 0)
        events = real.telemetry_snapshot()["page_identity_events"]
    finally:
        real.stop()
    return canvas, events


def _drive_sync(sequence, recover_at=None, extra=()):
    """A mesma trabalhadora no caminho do replay: pump() em relógio de cena."""
    clock = rr.SceneClock()
    worker = rr.make_ink_worker(_ProbIsWarp(), clock)
    generation = worker.new_generation()
    for i, (prob, corners) in enumerate(list(sequence) + list(extra)):
        if i == recover_at:
            generation = worker.new_generation(preserve_identity=True)
        clock.now += 10.0
        worker.submit(prob, generation, corners)
        assert worker.pump(), "passada nao rodou"
        assert worker.error is None
    _version, canvas = worker.poll(generation, 0)
    return canvas, list(worker._identity_events)


@pytest.mark.lento
def test_sync_pump_matches_threaded_worker(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.005)
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=3)
    base = rr.base_quad(SHAPE)
    rng = np.random.default_rng(0)
    sequence = []
    for stage in drawing + [settled]:
        for k in range(4):
            corners = (base + rng.normal(0, 2.0, size=(4, 2))).astype(np.float32)
            sequence.append((stage[k % len(stage)], corners))
    real_canvas, _ = _drive_threaded(sequence)
    sync_canvas, _ = _drive_sync(sequence)
    assert np.array_equal(sync_canvas, real_canvas)


def test_replay_worker_ecc_uses_only_confirmed_canvas(monkeypatch):
    monkeypatch.setenv("DRAW_ECC_REFINE", "1")
    references = []

    def refine(reference, observed, hand_mask=None):
        references.append(None if reference is None else reference.copy())
        return (
            observed,
            np.eye(3),
            {
                "ecc_ms": 0.0,
                "ecc_rejected": False,
                "ecc_converged": False,
                "ecc_mode": "bypass",
            },
        )

    monkeypatch.setattr(ink_layers_module, "_refine_ink_ecc", refine)
    clock = rr.SceneClock()
    worker = rr.make_ink_worker(_ProbIsWarp(), clock)
    generation = worker.new_generation()
    corners = rr.base_quad(SHAPE)
    prob = np.full(SHAPE, 0.8, dtype=np.float32)
    expected_refs = []
    for index in range(4):
        confirmed = worker._accumulator._confirmed
        expected_refs.append(
            None if confirmed is None else confirmed.astype(np.float32) / 255.0
        )
        clock.now = (index + 1) * 10.0
        worker.submit(prob, generation, corners)
        assert worker.pump()
    assert len(references) == len(expected_refs)
    assert references[0] is None and expected_refs[0] is None
    for got, expected in zip(references[1:], expected_refs[1:]):
        np.testing.assert_array_equal(got, expected)
        assert got.dtype == np.float32
    assert not references[1].any()
    assert np.all(references[-1] == 1.0)


@pytest.mark.lento
def test_sync_pump_matches_threaded_identity_recovery(monkeypatch):
    monkeypatch.setattr(ink_layers_module, "_PAGE_CLEAN_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(ink_layers_module, "_IDLE_PROBE_SECONDS", 0.005)
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=4)
    base = rr.base_quad(SHAPE)
    sequence = [
        (stage[k % len(stage)], base.copy())
        for stage in drawing + [settled]
        for k in range(4)
    ]
    moved = base + np.float32([6.0, -4.0])
    same = (rr.render_pass(settled[0], base, moved), moved)
    other = (np.flipud(settled[0]).copy(), base.copy())
    for returned, expected in ((same, "same_page"), (other, "new_page")):
        args = (sequence, len(sequence), [returned] * 3)
        real_canvas, real_events = _drive_threaded(*args)
        sync_canvas, sync_events = _drive_sync(*args)
        assert real_events[-1]["decision"] == sync_events[-1]["decision"] == expected
        assert [e["decision"] for e in real_events] == [
            e["decision"] for e in sync_events
        ]
        assert np.array_equal(sync_canvas, real_canvas)


def test_replay_drives_the_app_loop_and_the_real_worker():
    app = rr.ReplayApp(True)
    assert type(app.loop) is loop_module.SheetInkLoop
    assert isinstance(app.worker, app_module._PageCleanWorker)
    assert app.loop.ink_worker is app.worker
    assert app.loop.detection_worker is app.detector
    assert app.loop.tracker.__class__ is app_module._SheetTracker


def test_replay_doubles_override_no_production_behavior():
    """O replay não volta a espelhar: subclasses só acrescentam cópia/contagem."""
    assert set(vars(rr.ReplayInkWorker)) - {
        "__module__",
        "__doc__",
        "__qualname__",
    } <= {
        "__deepcopy__",
        "__firstlineno__",
        "__static_attributes__",
    }
    assert set(vars(rr.CountingAccumulator)) - {
        "__module__",
        "__doc__",
        "__qualname__",
        "__firstlineno__",
        "__static_attributes__",
    } == {"__init__", "add"}


def test_replay_app_deepcopy_is_independent():
    drawing, settled = rr.synthetic_pages(SHAPE, stages=2, seed=6)
    app, quad = rr.settle_drawing(drawing, settled, (True, "always"), False)
    clone = copy.deepcopy(app)
    assert clone.worker is not app.worker
    assert clone.loop.ink_worker is clone.worker
    assert clone.worker._clock is clone.clock
    assert clone.loop._warp_sheet.__self__ is clone
    passes = app.passes_total()
    rng = np.random.default_rng(1)
    for i in range(30):
        clone.step(rr.Frame(settled[i % 3], quad, rr._noisy(quad, rng)))
    assert app.passes_total() == passes
    assert clone.passes_total() > passes


def test_render_pass_is_identity_when_warp_matches_page():
    _drawing, settled = rr.synthetic_pages(SHAPE, stages=2, seed=1)
    quad = rr.base_quad(SHAPE)
    assert np.array_equal(rr.render_pass(settled[0], quad, quad), settled[0])


def test_comparator_accepts_pure_rewarp_and_flags_extra_ink():
    canvas = np.zeros((200, 150), np.uint8)
    cv2.polylines(canvas, [np.int32([[20, 30], [120, 60], [60, 170]])], False, 255, 3)
    cv2.circle(canvas, (100, 140), 25, 255, 2)
    matrix = np.array([[0.98, 0.05, 6.0], [-0.04, 1.01, -4.0], [0.0, 0.0, 1.0]])
    rewarped = cv2.warpPerspective(canvas, matrix, (150, 200), flags=cv2.INTER_NEAREST)
    same = rr.compare_canvases(
        canvas, rewarped, [("identity", np.eye(3)), ("known", matrix)]
    )
    assert same["canvas_identical_after_move"]
    assert same["new_confirmed_px_outside_tol"] == 0

    doubled = rewarped.copy()
    doubled[100:110, 10:60] = 255
    extra = rr.compare_canvases(
        canvas, doubled, [("identity", np.eye(3)), ("known", matrix)]
    )
    assert not extra["canvas_identical_after_move"]
    assert extra["new_confirmed_px_outside_tol"] > 0

    gone = rr.compare_canvases(canvas, None, [("identity", np.eye(3))])
    assert not gone["canvas_identical_after_move"]
    assert gone["lost_confirmed_px_outside_tol"] == np.count_nonzero(canvas)


def test_sway_offsets_peak_is_the_measured_amplitude():
    rng = np.random.default_rng(5)
    for profile in rr.SWAY_PROFILES:
        offsets = rr.sway_offsets(profile, 300, rng)
        peak = float(np.max(np.linalg.norm(offsets, axis=2)))
        assert abs(peak - rr.SWAY_AMPLITUDE_PX) < 1e-3


@pytest.mark.parametrize("chamfer_flag", ["0", "1"])
def test_reduced_run_reports_contract_fields(monkeypatch, chamfer_flag):
    monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", chamfer_flag)
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    if chamfer_flag == "0":
        monkeypatch.setattr(
            ink_layers_module,
            "_identity_chamfer_fields",
            lambda *a: pytest.fail("Chamfer off nao pode computar no replay"),
        )
    monkeypatch.setattr(rr, "SETTLE_FRAMES", 60)
    monkeypatch.setattr(rr, "SWAY_FRAMES", 90)
    monkeypatch.setattr(rr, "HAND_CONFIGS", {"presente_mexendo": (True, "always")})
    monkeypatch.setattr(rr, "MOVES", {"translate_large": rr.MOVES["translate_large"]})
    monkeypatch.setattr(rr, "DROP_FRAMES", {"long_150f": 45})
    monkeypatch.setattr(rr, "DROP_MODES", ("page_out",))
    monkeypatch.setattr(rr, "SWAY_PROFILES", ("bimodal",))
    drawing, settled = rr.synthetic_pages(SHAPE, stages=3, seed=2)
    results = rr.run_all(drawing, settled, "synthetic")
    assert results["unit"] == "passada"
    control = results["control_static"]["presente_mexendo"]
    assert control["passes_gate"] and not control["vacuous"]
    fields = (
        "canvas_identical_after_move",
        "confirmed_px_delta",
        "generations_created",
        "false_page_turns",
    )
    for section in (
        "move_same_canvas",
        "drop_and_return",
        "drop_and_turn",
        "sway_negative",
    ):
        for by_hand in results[section].values():
            for run in by_hand.values():
                for field in fields:
                    assert field in run, (section, field)
    # A folha que some além do relógio de perda abre geração nova
    # pelo caminho de drop: é o que a régua tem de enxergar.
    lost = results["drop_and_return"]["page_out_long_150f"]["presente_mexendo"]
    assert lost["drops"] == 1 and lost["generations_created"] >= 1
    turned = results["drop_and_turn"]["page_out_long_150f"]["presente_mexendo"]
    assert turned["identity_decisions"] == [
        "deferred_switch_consensus",
        "deferred_switch_consensus",
        "new_page",
    ]
    assert turned["passes_gate"]
    assert set(results["summary"]) >= {
        "control_static",
        "move_same_canvas",
        "drop_and_return",
        "sway_negative",
        "sway_negative_turn_event_enabled",
    }
    if chamfer_flag == "0":
        assert "identity_chamfer" not in results
        assert "identity_events" not in lost
        assert "identity_true_class" not in lost
    else:
        summary = results["identity_chamfer"]
        assert "no occlusion deferral" in summary["replay_visibility"]
        for section, label in (
            ("move_same_canvas", "same_page"),
            ("drop_and_return", "same_page"),
            ("drop_and_turn", "new_page"),
        ):
            for by_hand in results[section].values():
                for run in by_hand.values():
                    assert run["identity_true_class"] == label
                    assert run["identity_chamfer_by_ecc"]["1"]["events"] == 0
                    assert run["identity_chamfer_by_ecc"]["0"]["events"] == len(
                        run["identity_events"]
                    )
        assert summary["by_true_class"]["same_page"]["by_ecc"]["0"]["events"] > 0
        assert summary["by_true_class"]["new_page"]["by_ecc"]["0"]["events"] > 0


@pytest.mark.parametrize("ecc_enabled", ["0", "1"])
@pytest.mark.parametrize("chamfer_flag", ["0", "1"])
@pytest.mark.parametrize("decision", ["same_page", "new_page"])
def test_replay_ecc_rejection_defers_production(
    monkeypatch,
    ecc_enabled,
    chamfer_flag,
    decision,
):
    reference = np.zeros(SHAPE, dtype=np.uint8)
    reference[10:40, 10:40] = 255
    candidate = reference.astype(np.float32) / 255.0 * 0.95
    if decision == "new_page":
        candidate = np.flipud(candidate).copy()
    clock = rr.SceneClock()
    worker = rr.make_ink_worker(_ProbIsWarp(), clock)
    corners = rr.base_quad(SHAPE)
    # Estado real da pagina anterior: passadas da trabalhadora, nao
    # campos privados escritos a mao.
    monkeypatch.setenv("DRAW_ECC_REFINE", "0")
    generation = worker.new_generation()
    for _ in range(4):
        clock.now += 10.0
        worker.submit(reference.astype(np.float32) / 255.0 * 0.95, generation, corners)
        assert worker.pump()
    prior = worker._accumulator._confirmed.copy()
    assert np.array_equal(prior > 0, reference > 0)

    monkeypatch.setenv("DRAW_ECC_REFINE", ecc_enabled)
    monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", chamfer_flag)
    calls = []

    def reject(*args):
        calls.append(args[3])
        raise cv2.error("sem convergencia")

    monkeypatch.setattr(cv2, "findTransformECC", reject)
    generation = worker.new_generation(preserve_identity=True)
    clock.now += 10.0
    worker.submit(candidate, generation, corners)
    assert worker.pump() and worker.error is None
    pending = worker._identity_reference is not None
    assert pending is (ecc_enabled == "1" or decision == "new_page")
    event = worker._identity_events[-1]
    expected_decision = (
        "deferred_ecc_rejected"
        if ecc_enabled == "1"
        else "deferred_switch_consensus"
        if decision == "new_page"
        else "same_page"
    )
    assert event["decision"] == expected_decision
    assert event["ecc_rejected"] == (ecc_enabled == "1")
    assert bool(calls) == (ecc_enabled == "1")
    assert event["ecc_refine_enabled"] == int(ecc_enabled)
    if chamfer_flag == "1":
        assert event["ecc_refine_enabled"] == int(ecc_enabled)
        assert event["shadow_decision"] == (
            "defer" if ecc_enabled == "1" else expected_decision
        )
        expected = ink_layers_module._identity_chamfer_fields(
            candidate > 0.7,
            reference > 0,
            expected_decision,
            event,
        )
        for key, value in expected.items():
            assert event[key] == value
    else:
        assert "shadow_decision" not in event
        assert not any(key.startswith("chamfer_") for key in event)


def test_episode_chamfer_uses_ground_truth_not_decision_and_separates_ecc(monkeypatch):
    monkeypatch.setenv("DRAW_IDENTITY_CHAMFER", "1")
    drawing, settled = rr.synthetic_pages(SHAPE, stages=2, seed=5)
    app, quad = rr.settle_drawing(drawing, settled, (True, "always"), False)
    ep = rr.Episode(app)
    for enabled in (0, 1):
        monkeypatch.setenv("DRAW_ECC_REFINE", str(enabled))
        prior = settled[0] > 0.7
        seen = np.roll(prior, enabled + 1, axis=0)
        app.worker._identity_events.append(
            {
                "decision": "new_page",
                "recall": 0.0,
                "containment": 0.0,
                **ink_layers_module._identity_chamfer_fields(
                    seen, prior, "new_page", {}
                ),
            }
        )
    report = ep.report(quad, quad, identity_true_class="same_page")
    assert report["identity_true_class"] == "same_page"
    assert report["identity_decisions"] == ["new_page", "new_page"]
    for enabled in (0, 1):
        group = report["identity_chamfer_by_ecc"][str(enabled)]
        assert group["events"] == 1
        key = "chamfer_seen_to_reference_median_px"
        metric = group["metrics"][key]
        assert metric["count"] == 1 and metric["missing"] == 0
        assert metric["values"] == [report["identity_events"][enabled][key]]


def test_scene_detector_latency_and_latest_wins():
    dt = rr.DT
    clock = rr.SceneClock()

    def frame(tag):
        return rr.Frame(None, None, np.full((4, 2), tag, np.float32))

    fast = rr.SceneDetectionWorker(0.02, clock)
    fast.submit(frame(1), 0.0)
    assert fast.poll(0) is None
    clock.now = dt
    got = fast.poll(0)
    assert got[1][0, 0] == 1 and got[2:4] == ("paper", 0.0)

    # 100 ms em 30 FPS: a detecção do quadro 0 sai no quadro 3; os
    # quadros 1 e 2 disputam a fila de tamanho 1 e só o 2 roda.
    clock.now = 0.0
    slow = rr.SceneDetectionWorker(0.1, clock)
    results = []
    seq = 0
    for k in range(8):
        clock.now = k * dt
        got = slow.poll(seq)
        if got is not None:
            seq = got[0]
            results.append((k, int(got[1][0, 0]), got[3]))
        slow.submit(frame(k), clock.now)
    assert results[0] == (3, 0, 0.0)
    assert results[1][1] == 2 and results[1][2] == 2 * dt
    assert slow.replaced >= 1
    assert slow.completed < slow.submitted
