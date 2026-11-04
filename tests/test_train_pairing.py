"""Pareamento imagem/máscara do treino, nos dois layouts aceitos.

Layout images/masks (config/notebook) e layout plano do gerador
(scene_*.png + scene_*_mask.png + scene_*_ink_full.png).
Regressão da divergência encontrada no QA do neural-seg-v1:
o treino rejeitava a saída do gerador.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from train import (
    combined_loss,
    evaluate,
    onnx_delivery_status,
    pair_images_masks,
    validate_split_manifest,
)


def _write_img(path, value=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8, 3), value, dtype=np.uint8))


def _write_mask(path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8), value, dtype=np.uint8))


def test_images_masks_layout(tmp_path):
    split = tmp_path / "train"
    _write_img(split / "images" / "a.png")
    _write_mask(split / "masks" / "a.png")
    pairs = pair_images_masks(split)
    assert len(pairs) == 1
    assert pairs[0][2] is None


def test_flat_generator_layout(tmp_path):
    split = tmp_path / "train"
    _write_img(split / "scene_train_0000.png")
    _write_mask(split / "scene_train_0000_mask.png")
    _write_mask(split / "scene_train_0000_ink_full.png", 255)
    pairs = pair_images_masks(split)
    assert len(pairs) == 1
    assert pairs[0][1].name == "scene_train_0000_mask.png"
    assert pairs[0][2] is not None and pairs[0][2].name == "scene_train_0000_ink_full.png"


def test_onnx_export_wrapper_resizes_logits_to_input_contract():
    import types
    import pytest

    torch = pytest.importorskip("torch")
    if not hasattr(torch, "nn"):
        pytest.skip("PyTorch completo não está instalado")
    from train import make_onnx_export_model

    class LowResolutionModel(torch.nn.Module):
        def forward(self, pixel_values):
            batch = pixel_values.shape[0]
            logits = torch.zeros((batch, 4, 8, 8), device=pixel_values.device)
            return types.SimpleNamespace(logits=logits)

    export_model = make_onnx_export_model(LowResolutionModel(), size=32)
    output = export_model(torch.zeros((2, 3, 32, 32)))
    assert tuple(output.shape) == (2, 4, 32, 32)


def test_training_evaluate_aggregates_ink_tolerance_over_every_image():
    import types
    import pytest

    torch = pytest.importorskip("torch")
    if not hasattr(torch, "zeros"):
        pytest.skip("PyTorch completo não está instalado")
    targets = torch.zeros((2, 8, 8), dtype=torch.long)
    targets[1, 1, 1] = 3
    logits = torch.full((2, 4, 8, 8), -10.0)
    logits[:, 0] = 10.0
    logits[1, 3, 5, 5] = 20.0

    class FixedModel:
        def eval(self):
            return self

        def __call__(self, pixel_values):
            return types.SimpleNamespace(logits=logits[:pixel_values.shape[0]])

    images = torch.zeros((2, 3, 8, 8))
    metrics = evaluate(FixedModel(), [(images, targets)], "cpu")
    assert metrics["ink_tolerance"]["radius_2px"]["pred_ink"] == 1
    assert metrics["ink_tolerance"]["radius_2px"]["gt_ink"] == 1
    assert metrics["ink_tolerance"]["radius_2px"]["precision"] == 0.0
    assert metrics["ink_tolerance"]["radius_2px"]["recall"] == 0.0


def test_flat_layout_missing_mask_fails_instead_of_silently_dropping(tmp_path):
    split = tmp_path / "train"
    _write_img(split / "scene_train_0000.png")
    import pytest
    with pytest.raises(FileNotFoundError, match="máscara ausente"):
        pair_images_masks(split)


def test_split_manifest_rejects_source_leakage():
    manifest = {
        "splits": {
            "hands": {"train": ["hand-a"], "val": ["hand-a"], "test": ["hand-b"]}
        }
    }
    import pytest
    with pytest.raises(ValueError, match="vazamento de fontes"):
        validate_split_manifest(manifest)


def test_split_manifest_accepts_disjoint_base_hands():
    manifest = {
        "splits": {
            "hands": {"train": ["v0"], "val": ["v1"], "test": ["v2"]}
        },
        "hand_base_splits": {"train": ["h0"], "val": ["h1"], "test": ["h2"]},
    }
    report = validate_split_manifest(manifest)
    assert set(report["checked_source_families"]) == {"hands", "hand_base_splits"}


def test_onnx_delivery_requires_executed_successful_validation():
    assert onnx_delivery_status({"status": "ok"}) == "validated"
    for status in ("degraded", "skipped", "failed", None):
        assert onnx_delivery_status({"status": status}) == "blocked_onnx_validation"


def test_sample_weights_reduce_selected_sample_contribution():
    import pytest
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "nn"):
        pytest.skip("PyTorch completo não está instalado")
    logits = torch.zeros((2, 4, 4, 4), requires_grad=True)
    target = torch.zeros((2, 4, 4), dtype=torch.long)
    target[0, 0, 0] = 3
    target[1, 0, 0] = 3
    class_weights = torch.tensor([1.0, 2.0, 3.0, 10.0])
    separate = []
    for index in range(2):
        separate.append(combined_loss(
            logits[index:index + 1], target[index:index + 1], class_weights
        ))
    expected = (separate[0] + 0.25 * separate[1]) / 1.25
    actual = combined_loss(
        logits, target, class_weights, sample_weights=torch.tensor([1.0, 0.25])
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_evaluate_can_score_predictions_on_reference_resolution():
    import types
    import pytest
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "zeros"):
        pytest.skip("PyTorch completo não está instalado")

    class LowResolutionBackgroundModel:
        def eval(self):
            return self

        def __call__(self, pixel_values):
            logits = torch.full((pixel_values.shape[0], 4, 4, 4), -10.0)
            logits[:, 0] = 10.0
            return types.SimpleNamespace(logits=logits)

    input_images = torch.zeros((1, 3, 8, 8))
    input_masks = torch.zeros((1, 8, 8), dtype=torch.long)
    reference_images = torch.zeros((1, 3, 16, 16))
    reference_masks = torch.zeros((1, 16, 16), dtype=torch.long)
    metrics = evaluate(
        LowResolutionBackgroundModel(),
        [(input_images, input_masks)],
        "cpu",
        target_loader=[(reference_images, reference_masks)],
    )
    assert metrics["supervised_px"] == 16 * 16
    assert metrics["mean_iou"] == 1.0
