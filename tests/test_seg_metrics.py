"""Testes de métricas de segmentação e do validador ONNX, sem hardware.

Sem PyTorch e sem ONNX Runtime: a comparação ONNX é testada com
módulos falsos injetados em sys.modules (modelo dummy que devolve
um array NumPy, sessão ONNX falsa que devolve o mesmo array ou um
array degradado). Isso testa a lógica de comparação e de degradação,
não as bibliotecas — que o CI do Draw não tem instaladas.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import seg_metrics, validate_onnx  # noqa: E402


def _mask_with_all_classes():
    """Máscara 10x10 com as 4 classes presentes, para métricas exatas."""
    gt = np.zeros((10, 10), dtype=np.int64)
    gt[0:5, 0:5] = 1  # paper
    gt[0:5, 5:10] = 2  # hand
    gt[5:10, 0:5] = 3  # ink
    return gt


def test_perfect_prediction_scores_one():
    gt = _mask_with_all_classes()
    metrics = seg_metrics.evaluate(gt, gt)
    for cls in seg_metrics.CLASSES:
        assert metrics["iou"][cls] == 1.0
        assert metrics["dice"][cls] == 1.0
        assert metrics["precision"][cls] == 1.0
        assert metrics["recall"][cls] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["accuracy"] == 1.0
    tol = seg_metrics.ink_tolerance_metrics(gt, gt)
    assert tol["precision"] == 1.0 and tol["recall"] == 1.0 and tol["f1"] == 1.0


def test_empty_prediction_scores_zero_for_present_classes():
    gt = _mask_with_all_classes()
    pred = np.zeros_like(gt)  # tudo background
    metrics = seg_metrics.evaluate(pred, gt)
    assert metrics["iou"]["paper"] == 0.0
    assert metrics["iou"]["hand"] == 0.0
    assert metrics["iou"]["ink"] == 0.0
    assert metrics["dice"]["ink"] == 0.0
    assert metrics["recall"]["ink"] == 0.0
    # Background não é zero: parte do ground truth já é background.
    assert 0.0 < metrics["iou"]["background"] < 1.0
    tol = seg_metrics.ink_tolerance_metrics(pred, gt)
    assert tol["precision"] == 0.0 and tol["recall"] == 0.0


def test_ink_shifted_two_pixels_passes_tolerance_but_not_exact_iou():
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[10, 3:17] = 3  # linha fina de ink
    pred = np.zeros((20, 20), dtype=np.int64)
    pred[12, 3:17] = 3  # mesma linha, deslocada 2 px para baixo
    exact = seg_metrics.evaluate(pred, gt)
    assert exact["iou"]["ink"] == 0.0  # sem sobreposição exata
    tol = seg_metrics.ink_tolerance_metrics(pred, gt, tolerance_px=3)
    assert tol["precision"] == 1.0
    assert tol["recall"] == 1.0
    # Com tolerância 1, o deslocamento de 2 px NÃO deve passar.
    tol1 = seg_metrics.ink_tolerance_metrics(pred, gt, tolerance_px=1)
    assert tol1["precision"] == 0.0 and tol1["recall"] == 0.0


def test_ink_tolerance_empty_cases():
    empty = np.zeros((5, 5), dtype=np.int64)
    tol = seg_metrics.ink_tolerance_metrics(empty, empty)
    assert tol["precision"] == 1.0 and tol["recall"] == 1.0
    gt = empty.copy()
    gt[2, 2] = 3
    tol = seg_metrics.ink_tolerance_metrics(empty, gt)
    assert tol["precision"] == 0.0 and tol["recall"] == 0.0


def test_invalid_masks_raise():
    gt = _mask_with_all_classes()
    with pytest.raises(ValueError):
        seg_metrics.evaluate(np.zeros((5, 5), dtype=np.int64), gt)
    bad = gt.copy()
    bad[0, 0] = 4  # classe inexistente
    with pytest.raises(ValueError):
        seg_metrics.evaluate(bad, gt)


def test_ignore_index_255_is_excluded_from_metrics_and_ink_tolerance():
    gt = _mask_with_all_classes()
    gt[5:7, 5:7] = seg_metrics.IGNORE_INDEX
    perfect = gt.copy()
    wrong_in_ignored = gt.copy()
    perfect[5:7, 5:7] = seg_metrics.BACKGROUND
    wrong_in_ignored[5:7, 5:7] = seg_metrics.INK

    baseline = seg_metrics.evaluate(perfect, gt)
    changed = seg_metrics.evaluate(wrong_in_ignored, gt)
    assert changed == baseline
    assert baseline["ignored_px"] == 4
    assert baseline["supervised_px"] == gt.size - 4
    assert baseline["support_px"]["ink"] == 25

    tol = seg_metrics.ink_tolerance_metrics(wrong_in_ignored, gt, tolerance_px=1)
    assert tol["pred_ink"] == 25
    assert tol["matched_pred"] == 25
    assert tol["matched_gt"] == 25


def test_all_ignored_ground_truth_is_rejected():
    ignored = np.full((4, 4), seg_metrics.IGNORE_INDEX, dtype=np.uint8)
    with pytest.raises(ValueError, match="sem pixels supervisionados"):
        seg_metrics.evaluate(np.zeros_like(ignored), ignored)


def test_streaming_confusion_metrics_match_full_evaluation():
    gt = _mask_with_all_classes()
    gt[:2, :2] = seg_metrics.IGNORE_INDEX
    pred = gt.copy()
    pred[:2, :2] = seg_metrics.BACKGROUND
    pred[7, 7] = 0
    assert seg_metrics.evaluate_confusion_matrix(seg_metrics.confusion_matrix(pred, gt), 4) == (
        seg_metrics.evaluate(pred, gt)
    )


def test_scenario_battery_has_exactly_fifteen():
    assert len(seg_metrics.SCENARIOS) == 15
    ids = [s["id"] for s in seg_metrics.SCENARIOS]
    assert len(set(ids)) == len(ids)  # ids únicos
    names = [s["name"] for s in seg_metrics.SCENARIOS]
    for required in (
        "folha limpa",
        "folha com desenho",
        "folha muito desenhada",
        "mão cobrindo canto",
        "mão cobrindo centro",
        "mão cobrindo traço",
        "mão parcialmente fora da imagem",
        "fundo claro",
        "fundo escuro",
        "sombra forte",
        "folha parcialmente fora da imagem",
        "mão sem folha",
        "objeto branco sem folha",
        "traço fino de lápis",
        "traço grosso de caneta",
    ):
        assert required in names


def test_save_predictions_writes_png_and_manifest(tmp_path):
    pred = _mask_with_all_classes()
    predictions = {"01_clean_sheet": pred}
    manifest_path = seg_metrics.save_predictions(tmp_path, predictions, model_name="dummy-v1")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "dummy-v1"
    assert "01_clean_sheet" in manifest["scenarios"]
    png_path = tmp_path / "01_clean_sheet_pred.png"
    assert png_path.exists() and png_path.stat().st_size > 0


def test_evaluate_scenarios_aggregates(tmp_path):
    gt = _mask_with_all_classes()
    result = seg_metrics.evaluate_scenarios(
        {"01_clean_sheet": gt}, {"01_clean_sheet": gt}
    )
    assert result["per_scenario"]["01_clean_sheet"]["metrics"]["mean_iou"] == 1.0
    assert result["aggregate"]["n_evaluated"] == 1
    assert result["aggregate"]["mean_iou"] == 1.0


def _logits_from_mask(mask):
    """Logits dummy: classe correta com valor alto, demais com zero."""
    h, w = mask.shape
    out = np.zeros((1, 4, h, w), dtype=np.float64)
    for cls in range(4):
        out[0, cls][mask == cls] = 10.0
    return out


def test_compare_model_outputs_identical():
    mask = _mask_with_all_classes()
    logits = _logits_from_mask(mask)
    report = validate_onnx.compare_model_outputs(logits, logits.copy())
    assert report["max_abs_diff"] == 0.0
    assert report["argmax_agreement"] == 1.0
    assert report["min_argmax_iou"] == 1.0


def test_compare_model_outputs_shape_mismatch_raises():
    mask = _mask_with_all_classes()
    logits = _logits_from_mask(mask)
    with pytest.raises(ValueError):
        validate_onnx.compare_model_outputs(logits, logits[:, :, :5, :])


def _install_fake_torch_and_ort(onnx_logits):
    """Injeta torch e onnxruntime falsos em sys.modules.

    O modelo dummy devolve logits NumPy; a sessão ONNX falsa devolve
    ``onnx_logits``. Retorna (modelo_dummy, caminho_onnx_falso).
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _NoGrad()
    fake_ort = types.ModuleType("onnxruntime")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    class _Session:
        def __init__(self, path, providers=None):
            self.path = path

        def get_inputs(self):
            return [types.SimpleNamespace(name="input")]

        def run(self, _outputs, feeds):
            return [onnx_logits]

    fake_ort.InferenceSession = _Session

    class _DummyModel:
        training = False

        def eval(self):
            return self

        def __call__(self, x):
            return x  # entrada já é o logits dummy

    sys.modules["torch"] = fake_torch
    sys.modules["onnxruntime"] = fake_ort
    return _DummyModel(), "dummy.onnx"


def test_validate_onnx_dummy_identical_is_ok(monkeypatch):
    mask = _mask_with_all_classes()
    logits = _logits_from_mask(mask)
    model, path = _install_fake_torch_and_ort(logits.copy())
    report = validate_onnx.validate_onnx(model, path, logits)
    assert report["status"] == "ok"
    assert report["degraded"] is False
    assert report["comparison"]["max_abs_diff"] == 0.0


def test_validate_onnx_dummy_degraded_is_flagged(monkeypatch):
    mask = _mask_with_all_classes()
    logits = _logits_from_mask(mask)
    # ONNX degradado: troca paper por hand em metade da folha.
    degraded_mask = mask.copy()
    degraded_mask[0:5, 0:5] = 2
    degraded_logits = _logits_from_mask(degraded_mask)
    model, path = _install_fake_torch_and_ort(degraded_logits)
    report = validate_onnx.validate_onnx(model, path, logits)
    assert report["status"] == "degraded"
    assert report["degraded"] is True
    assert report["comparison"]["min_argmax_iou"] < 0.99


def test_validate_onnx_expected_output_shape_is_enforced(monkeypatch):
    mask = _mask_with_all_classes()
    logits = _logits_from_mask(mask)
    model, path = _install_fake_torch_and_ort(logits.copy())
    report = validate_onnx.validate_onnx(
        model, path, logits, expected_shape=(1, 4, 12, 10)
    )
    assert report["status"] == "degraded"
    assert report["shape_matches"] is False
    assert report["actual_shape"] == [1, 4, 10, 10]


def test_validate_onnx_reports_onnx_shape_mismatch_as_degraded(monkeypatch):
    mask = _mask_with_all_classes()
    logits = _logits_from_mask(mask)
    lower_resolution = logits[:, :, :5, :]
    model, path = _install_fake_torch_and_ort(lower_resolution)
    report = validate_onnx.validate_onnx(model, path, logits)
    assert report["status"] == "degraded"
    assert report["shape_matches"] is False
    assert report["actual_shape"] == [1, 4, 10, 10]
    assert report["onnx_shape"] == [1, 4, 5, 10]
    assert report["comparison"] is None


def test_validate_onnx_skipped_without_libraries(monkeypatch):
    # Sem torch/ort instalados de verdade, o relatório é skipped,
    # nunca "ok" falso. Remove os fakes caso outro teste os tenha posto.
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    # Se o ambiente tiver torch de verdade, este teste não se aplica;
    # nesse caso, pular em vez de falhar por causa do ambiente.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch instalado neste ambiente; skipped não se aplica")
    report = validate_onnx.validate_onnx(object(), "dummy.onnx", np.zeros((1, 4, 2, 2)))
    assert report["status"] == "skipped"
    assert report["degraded"] is None
