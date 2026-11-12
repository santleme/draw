"""Cenário escrever→apagar→redesenhar→mover da fonte sintética."""

import json

import cv2
import numpy as np
import pytest

from tools.sim_source import SyntheticCapture, export_video


def drain(capture):
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return frames
        frames += 1


def scenario(**kwargs):
    options = dict(
        fps=10,
        stroke_seconds=0.2,
        warmup_seconds=0.2,
        settle_seconds=0.3,
        hold_seconds=0.4,
        erase_stroke=-1,
        redraw_text="ui",
        erase_seconds=0.3,
    )
    options.update(kwargs)
    return SyntheticCapture("Draw ok", **options)


def kinds(capture):
    return [phase["kind"] for phase in capture.phases]


def test_default_timeline_unchanged():
    capture = SyntheticCapture("ab", fps=4, stroke_seconds=0.5, settle_seconds=0.5)
    assert kinds(capture) == ["warmup", "write", "settle"]
    assert capture.frame_count == capture.phases[-1]["end"]
    assert capture.erase_rect is None and not capture.redraw_specs
    assert np.array_equal(capture._quad_now, capture.quad)


def test_scenario_timeline_order_and_bounds():
    capture = scenario(move=(20, 10, 1.5))
    assert kinds(capture) == [
        "warmup",
        "write",
        "hold",
        "erase",
        "hold",
        "write",
        "hold",
        "move",
        "settle",
    ]
    start = 0
    for phase in capture.phases:
        assert phase["start"] == start and phase["end"] > start
        start = phase["end"]
    assert capture.frame_count == start


def test_erase_validations():
    with pytest.raises(ValueError, match="alternativas"):
        SyntheticCapture("a", erase_stroke=0, erase_rect=(10, 10, 20, 20))
    with pytest.raises(ValueError, match="exige"):
        SyntheticCapture("a", redraw_text="b")
    with pytest.raises(ValueError, match="fora"):
        SyntheticCapture("ab", erase_stroke=99)
    with pytest.raises(ValueError, match="fora"):
        SyntheticCapture("ab", erase_rect=(-5, 0, 10, 10))
    with pytest.raises(ValueError, match="move"):
        SyntheticCapture("ab", move=(1, 2, 3, 4))


def drain_until(capture, frame_index):
    """Lê até o frame inclusive; útil para parar numa fronteira de fase."""
    while capture.get(cv2.CAP_PROP_POS_FRAMES) <= frame_index:
        ok, _ = capture.read()
        if not ok:
            break


def _raster(bundle, indices):
    mask = np.zeros(bundle["page_ink_full"].shape, np.uint8)
    points, offsets = bundle["points"], bundle["offsets"]
    for index in indices:
        geo = points[offsets[index] : offsets[index + 1]]
        cv2.polylines(
            mask, [geo.astype(np.int32)], False, 255, int(bundle["thickness"][index])
        )
    return mask


def test_erasure_removes_ink_only_inside_rect():
    capture = scenario()
    assert capture.erase_rect is not None
    x, y, w, h = capture.erase_rect
    inside_before = int(capture.page_ink_full[y : y + h, x : x + w].sum())
    assert inside_before > 0
    erase_end = next(p["end"] for p in capture.phases if p["kind"] == "erase")
    drain_until(capture, erase_end - 1)
    assert not capture._ink[y : y + h, x : x + w].any()
    # fora do retângulo, a tinta dos traços sobrevive
    outside = capture._ink.copy()
    outside[y : y + h, x : x + w] = 0
    assert outside.any()


def test_redraw_lands_on_erased_area():
    capture = scenario()
    assert capture.redraw_specs
    points = np.concatenate([spec[3] for spec in capture.redraw_specs])
    x, y, w, h = capture.erase_rect
    cx, cy = points.mean(axis=0)
    assert x - w <= cx <= x + 2 * w and y - h <= cy <= y + 2 * h
    drain(capture)
    # GT por fase: plano inclui todos os traços; estado final sem o retângulo
    assert set(np.unique(capture.stroke_phase)) == {0, 1}
    assert capture.stroke_erased.any()


def test_move_shifts_final_quad():
    moved = scenario(move=(30, 15, 2.0))
    drain(moved)
    assert not np.allclose(moved._quad_now, moved.quad)
    still = scenario(move=None)
    drain(still)
    assert np.array_equal(still._quad_now, still.quad)


def test_ground_truth_bundle_phase_fields(tmp_path):
    capture = scenario(move=(20, 10, 1.5))
    drain(capture)
    gt = tmp_path / "scenario.gt.npz"
    capture.save_ground_truth(gt)
    with np.load(gt, allow_pickle=False) as bundle:
        for key in (
            "page_ink_final",
            "stroke_phase",
            "stroke_erased",
            "erase_rect",
            "events",
            "quad_final",
            "page_to_scene_final",
        ):
            assert key in bundle.files
        assert bundle["page_ink_final"].shape == bundle["page_ink_full"].shape
        assert int(bundle["stroke_phase"].max()) == 1
        events = json.loads(str(bundle["events"]))
        # estado final: traço apagado some, redesenho fica no lugar
        final = bundle["page_ink_final"] > 0
        full = bundle["page_ink_full"] > 0
        assert not np.array_equal(final, full)
        erased_mask = _raster(bundle, np.nonzero(bundle["stroke_erased"])[0]) > 0
        redraw_mask = _raster(bundle, np.nonzero(bundle["stroke_phase"] == 1)[0]) > 0
        redraw_mask = cv2.dilate(redraw_mask.astype(np.uint8), np.ones((3, 3))) > 0
        x, y, w, h = (int(v) for v in bundle["erase_rect"])
        leftover = np.zeros_like(final)
        leftover[y : y + h, x : x + w] = True
        assert not (final & erased_mask & leftover & ~redraw_mask).any()
        assert (final & redraw_mask).any()
    assert [e["kind"] for e in events] == kinds(capture)


def test_scenario_is_deterministic():
    a, b = scenario(), scenario()
    assert a.erase_rect == b.erase_rect
    assert len(a.redraw_specs) == len(b.redraw_specs)
    for sa, sb in zip(a.redraw_specs, b.redraw_specs):
        assert np.array_equal(sa[3], sb[3])


def test_export_manifest_marks_scenario(tmp_path):
    capture = scenario()
    manifest = export_video(capture, tmp_path / "s.mp4")
    assert manifest["pose"] == "scenario"
    assert manifest["strokes_redraw"] == len(capture.redraw_specs)
    assert manifest["strokes_erased"] == int(capture.stroke_erased.sum())
    assert [p["kind"] for p in manifest["phases"]] == kinds(capture)
    fixed = export_video(
        SyntheticCapture("ab", fps=4, settle_seconds=0.5), tmp_path / "f.mp4"
    )
    assert fixed["pose"] == "fixed"


def test_page_expected_matches_simulator_final(tmp_path):
    """O gabarito do avaliador segue a ordem do simulador: a borracha
    correu antes do redesenho, então traço de fase 1 dentro do
    retângulo apagado é tinta esperada — não falso positivo."""
    from tools.eval_scenario import _page_expected

    capture = scenario(move=(20, 10, 1.5))
    drain(capture)
    gt = tmp_path / "scenario.gt.npz"
    capture.save_ground_truth(gt)
    with np.load(gt, allow_pickle=False) as bundle:
        # fase de redraw aplicada e borracha já corrida: o esperado
        # do checkpoint final tem que ser exatamente o estado da folha
        expected = _page_expected(bundle, int(bundle["stroke_phase"].size), erased=True)
        np.testing.assert_array_equal(expected > 0, bundle["page_ink_final"] > 0)
        # e o redesenho DENTRO do retângulo apagado sobrevive no
        # gabarito (era o que o apagar-em-bloco zerava)
        x, y, w, h = (int(v) for v in bundle["erase_rect"])
        redraw = _raster(bundle, np.nonzero(bundle["stroke_phase"] == 1)[0]) > 0
        inside = np.zeros_like(redraw)
        inside[y : y + h, x : x + w] = True
        assert (expected > 0)[inside & redraw].all()


def test_final_checkpoint_terminal_canvas_loss():
    """Canvas perdido no evento final não empresta o IoU de fases
    anteriores: o veredito expõe o status em vez de métricas."""
    from tools.eval_scenario import _final_checkpoint

    events = [{"kind": "hold", "end": 10}, {"kind": "settle", "end": 20}]
    checkpoints = [
        {"frame": 9, "phase": "hold", "iou": 0.9, "confirmed_px": 100},
        {"frame": 19, "phase": "settle", "status": "sem âncora/canvas"},
    ]
    final = _final_checkpoint(checkpoints, events)
    assert "iou" not in final
    assert "canvas ausente" in final["status"]


def test_final_checkpoint_missing_terminal_event():
    """Vídeo truncado antes do último evento: veredito sem métrica,
    com status explicando — nunca o sucesso de uma fase anterior."""
    from tools.eval_scenario import _final_checkpoint

    events = [{"kind": "hold", "end": 10}, {"kind": "settle", "end": 20}]
    checkpoints = [{"frame": 9, "phase": "hold", "iou": 0.9}]
    final = _final_checkpoint(checkpoints, events)
    assert "iou" not in final
    assert "evento final" in final["status"]


def test_lift_return_timeline_and_absent_scene():
    """lift & return: a folha some de cena por inteiro (só mesa nos
    frames de absent) e volta na pose declarada por return_pose."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=None,
        lift=(300, 200),
        return_pose=(-40, 30, 18),
        absent_seconds=0.8,
        lift_seconds=0.4,
        return_seconds=0.4,
    )
    assert kinds(capture) == [
        "warmup",
        "write",
        "lift_out",
        "absent",
        "return",
        "settle",
    ]
    masks = {}
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        if capture.last_scene is not None:
            masks[capture._position - 1] = capture.last_scene["mask"]
    import tools.gen_synthetic as gs

    for phase in capture.phases:
        end = phase["end"] - 1
        if phase["kind"] == "absent":
            # durante a ausência, nada de papel nem tinta na cena
            assert not (masks[end] == gs.PAPER).any()
            assert not (masks[end] == gs.INK).any()
        elif phase["kind"] == "return":
            assert (masks[end] == gs.PAPER).sum() > 0
    # a volta obedece a return_pose: pose final difere da inicial
    assert not np.allclose(capture._quad_now, capture.quad)


def test_lift_return_ground_truth(tmp_path):
    """GT declara as fases novas e a pose final rodada."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=None,
        lift=(200, 150),
        return_pose=(0, 0, 25),
        absent_seconds=0.5,
    )
    drain(capture)
    gt = tmp_path / "lift.gt.npz"
    capture.save_ground_truth(gt)
    with np.load(gt, allow_pickle=False) as bundle:
        events = json.loads(str(bundle["events"]))
        assert [e["kind"] for e in events] == kinds(capture)
        assert "absent" in [e["kind"] for e in events]
        # a tinta sobrevive à saída: estado final = plano completo
        np.testing.assert_array_equal(
            bundle["page_ink_final"] > 0, bundle["page_ink_full"] > 0
        )
        assert not np.allclose(bundle["quad_final"], bundle["quad"])
        assert not np.allclose(bundle["page_to_scene_final"], bundle["page_to_scene"])


def test_lift_validations():
    with pytest.raises(ValueError):
        SyntheticCapture("ab", return_pose=(10, 0, 15))
    with pytest.raises(ValueError):
        SyntheticCapture("ab", lift=(0, 0))
    with pytest.raises(ValueError):
        SyntheticCapture("ab", lift=(10, 0, 0))


def test_lift_starts_from_post_move_pose():
    """Combinação move+lift: a saída parte da pose DEPOIS do move —
    o quad_at_lift é composto em __init__, não pode saltar de volta
    para a pose inicial quando o lift_out começa."""
    capture = scenario(
        erase_stroke=None,
        redraw_text=None,
        move=(60, 30, 5),
        lift=(300, 200),
        return_pose=(0, 0, 0),
        absent_seconds=0.4,
        lift_seconds=0.4,
        return_seconds=0.4,
    )
    lift_out = next(p for p in capture.phases if p["kind"] == "lift_out")
    move_end = next(p for p in capture.phases if p["kind"] == "move")
    # o quad_from do lift é a pose pós-move: distinto da pose inicial
    # (que é o quad_from do move) e igual ao quad que o move produz
    # no seu último frame — sem salto na transição
    assert not np.allclose(lift_out["quad_from"], move_end["quad_from"])
    moved_quads = {}
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        moved_quads[capture._position - 1] = capture._quad_now.copy()
    end_move = moved_quads[move_end["end"] - 1]
    np.testing.assert_allclose(lift_out["quad_from"], end_move, atol=1.0)
