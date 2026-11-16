"""Contratos do falsificador: amostragem, mutação e placar."""

import numpy as np
import pytest

from tools.falsify import mutate_config, sample_config, score_report
from tools.sim_source import SyntheticCapture


def test_sampled_configs_build_valid_captures():
    rng = np.random.default_rng(11)
    for index in range(40):
        cfg = sample_config(rng, index)
        capture = SyntheticCapture(cfg["text"], seed=cfg["seed"], **cfg["params"])
        assert capture.frame_count > 0
        ok, frame = capture.read()
        assert ok and frame.shape == (512, 512, 3)
        capture.release()


def test_mutation_stays_valid_and_near_parent():
    rng = np.random.default_rng(12)
    parent = sample_config(rng, 0)
    for _ in range(40):
        child = mutate_config(parent, rng)
        capture = SyntheticCapture(child["text"], seed=child["seed"], **child["params"])
        assert capture.frame_count > 0
        capture.release()


def test_mutation_never_leaves_parameter_contract():
    """Mutação que escapasse do contrato do SyntheticCapture viraria
    score=-1 por ERRO — não por falha do pipeline. Cada campo mutável
    tem faixa própria."""
    rng = np.random.default_rng(42)
    for index in range(15):
        parent = sample_config(rng, index)
        for _ in range(25):
            child = mutate_config(parent, rng)
            params = child["params"]
            for key, value in params.items():
                if key in ("ink_sheen", "page_curl"):
                    assert 0.0 <= value <= 1.0, (key, value)
                if key == "jpeg_quality":
                    assert 10 <= value <= 100
                if key == "hand_scale":
                    assert 0 < value <= 4
                if key == "fps":
                    assert 0 < value <= 120
                if key == "erase_rect":
                    x, y, w, h = value
                    assert x >= 0 and y >= 0 and w >= 1 and h >= 1
                    assert x + w <= 360 and y + h <= 510
                if key in ("move", "return_pose", "lift"):
                    assert all(np.isfinite(v) for v in value)
                if key == "lift":
                    assert float(np.linalg.norm(value)) >= 1.0
            capture = SyntheticCapture(
                child["text"], seed=child["seed"], **params
            )
            assert capture.frame_count > 0
            capture.release()


def test_run_trial_removes_artifacts_after_error(tmp_path, monkeypatch):
    """Falha na auditoria não pode deixar MP4/GT/manifesto órfãos no
    workdir — o cleanup roda no finally, não só no caminho feliz."""
    import tools.falsify as fz

    def fake_export(capture, output):
        output.write_bytes(b"x")
        output.with_suffix(".gt.npz").write_bytes(b"x")
        output.with_suffix(".json").write_bytes(b"x")
        output.with_name(output.stem + "_ink_full.npy").write_bytes(b"x")
        return {"frames": 1}

    monkeypatch.setattr(fz, "export_video", fake_export)
    monkeypatch.setattr(
        fz,
        "audit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("auditoria falhou")
        ),
    )
    cfg = {"text": "ab", "seed": 1, "params": {"fps": 8, "settle_seconds": 0.3}}
    with pytest.raises(RuntimeError):
        fz.run_trial(cfg, tmp_path, "bad", models={}, keep_artifacts=False)
    for suffix in (".mp4", ".gt.npz", ".json"):
        assert not (tmp_path / f"bad{suffix}").exists()
    assert not (tmp_path / "bad_ink_full.npy").exists()


def test_score_report_weights_documented_failures():
    perfect = {
        "verdict": {"final_iou": 0.9},
        "checkpoints": [{"iou": 0.9}],
    }
    score, comps = score_report(perfect)
    assert score == pytest.approx(0.1 + 0.5 * 0.1)
    assert comps["no_canvas"] == 0.0

    lost = {"verdict": {"final_iou": None, "status": "canvas ausente"}}
    score, comps = score_report(lost)
    assert comps["no_canvas"] == 1.0 and score >= 2.0

    ghost = {
        "verdict": {
            "final_iou": 0.8,
            "residual_confirmed_in_erased_px": 8000,
            "redraw_confirmed_recall": 0.2,
        },
        "checkpoints": [{"iou": 0.8}],
    }
    score, comps = score_report(ghost)
    assert comps["erase_residual"] == 1.0
    assert comps["redraw_loss"] == pytest.approx(0.8)
    worst = {
        "verdict": {"final_iou": 0.8},
        "checkpoints": [{"iou": 0.8}, {"iou": 0.1}],
    }
    assert score_report(worst)[1]["worst_phase_loss"] == pytest.approx(0.9)
