"""Testes das vistas geometricas e da perda do train_round (issue #19).

A translacao geometrica nao pode ser circular (np.roll): pixels que
saem por uma borda nao reaparecem na oposta, e a faixa descoberta nao
e fundo rotulado — e pixel ignorado na perda supervisionada.

O train_round importa torch, que nao e dependencia da suite (a CI
roda sem torch): sem torch real, estes testes pulam, como os demais
testes dependentes de torch do repo.
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

try:
    import torch as _REAL_TORCH
except ImportError:  # ambiente sem torch (CI): os testes pulam
    _REAL_TORCH = None

_PATH = (Path(__file__).resolve().parent.parent
         / "tools" / "selflearn" / "train_round.py")
_tr = None


def _train_round():
    global _tr
    if _REAL_TORCH is None or not hasattr(_REAL_TORCH, "nn"):
        pytest.skip("torch real nao disponivel neste ambiente")
    if _tr is None:
        # test_seg_metrics deixa um torch falso em sys.modules e nao o
        # restaura; o torch real capturado na coleta e o que vale aqui.
        sys.modules["torch"] = _REAL_TORCH
        spec = importlib.util.spec_from_file_location("train_round",
                                                      _PATH)
        _tr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_tr)
    return _tr


def _rng():
    return np.random.default_rng(0)


def test_geo_view_translation_does_not_wrap_to_opposite_edge():
    tr = _train_round()
    img = np.full((8, 8), 0.9, dtype=np.float32)
    img[4, 7] = 0.0  # traco encostado na borda direita
    out, flip, dx, dy = tr.geo_view(img, _rng(), flip=False, dx=2, dy=0)
    assert (dx, dy) == (2, 0)
    # o traco saiu pela direita; NAO reaparece na esquerda
    assert (out < 0.1).sum() == 0
    # a faixa descoberta (2 colunas da esquerda) recebe o fundo estimado
    assert np.allclose(out[:, :2], 0.9)


def test_geo_view_mask_fill_is_zero_not_wrapped_ink():
    tr = _train_round()
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[3, 0] = 1.0  # tinta na borda esquerda
    out, *_ = tr.geo_view(mask, _rng(), flip=False, dx=-2, dy=0,
                          fill=0.0)
    # a tinta saiu pela esquerda e nao vira ground truth na direita
    assert out.sum() == 0.0


def test_geo_valid_mask_marks_uncovered_strip():
    tr = _train_round()
    valid = tr.geo_valid_mask((8, 8), dy=0, dx=2)
    assert valid[:, :2].sum() == 0
    assert valid[:, 2:].sum() == 8 * 6
    valid_y = tr.geo_valid_mask((8, 8), dy=-3, dx=0)
    assert valid_y[5:, :].sum() == 0
    assert valid_y[:5, :].sum() == 5 * 8


def test_consistency_targets_apply_classic_witness_veto():
    # Issue #18: o pseudo-rotulo do treino tem de passar pelo veto
    # da testemunha classica (teacher.pseudo_label), nao so pela
    # confianca da propria rede.
    tr = _train_round()
    page = np.full((64, 64, 3), 0.94, dtype=np.float32)
    page[28:32, 10:50] = 0.12  # traco escuro de verdade
    prob = np.full((64, 64), 0.01, dtype=np.float32)
    prob[5, 5] = 0.99    # rede ve tinta onde NAO ha escurecimento
    prob[30, 30] = 0.99  # rede ve tinta sobre o traco
    target, weight = tr.consistency_targets(prob, page)
    # tinta confirmada pela testemunha: alvo 1, peso 1
    assert target[30, 30] == 1.0 and weight[30, 30] == 1.0
    # tinta vetada (sem escurecimento): ignore — nunca alvo de fundo
    assert target[5, 5] == 0.0 and weight[5, 5] == 0.0
    # escurecimento que a rede nao viu: ignore, nao fundo
    assert target[30, 20] == 0.0 and weight[30, 20] == 0.0
    # fundo confirmado (rede confiante no fundo, sem escurecimento)
    assert target[10, 10] == 0.0 and weight[10, 10] == 1.0


def test_supervised_loss_ignores_invalid_pixels():
    tr = _train_round()
    torch = _REAL_TORCH
    sys.modules["torch"] = torch
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    valid = torch.ones(1, 1, 4, 4)
    valid[:, :, :, :2] = 0.0  # metade esquerda sem conteudo valido
    changed = logits.clone()
    changed[:, :, :, :2] = 10.0  # erro grosseiro so na faixa invalida
    base = tr.supervised_loss(logits, target, valid)
    after = tr.supervised_loss(changed, target, valid)
    assert torch.allclose(base, after, atol=1e-6)
    # sem a mascara, o mesmo erro muda a perda (a mascara e o que protege)
    unmasked = tr.supervised_loss(changed, target)
    assert not torch.allclose(tr.supervised_loss(logits, target),
                              unmasked, atol=1e-6)


def test_load_pages_preserves_bgr_and_manifest_lineage(tmp_path):
    tr = _train_round()
    (tmp_path / "input").mkdir()
    (tmp_path / "mask").mkdir()
    red_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    red_bgr[:] = (0, 0, 255)
    cv2.imwrite(str(tmp_path / "input" / "page0.png"), red_bgr)
    cv2.imwrite(str(tmp_path / "mask" / "page0.png"),
                np.zeros((8, 8), dtype=np.uint8))
    row = {
        "file": "page0", "session_id": "train-session",
        "origin": "human_correction", "split": "train",
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")

    pages = tr.load_pages(tmp_path, with_mask=True)

    assert len(pages) == 1
    assert np.allclose(pages[0].image[0, 0], [0.0, 0.0, 1.0])
    assert pages[0].session_id == "train-session"
    assert pages[0].origin == "human_correction"


def test_ema_ramp_does_not_anchor_teacher_to_random_initialization():
    tr = _train_round()
    assert tr.ema_decay_at_step(0) == pytest.approx(0.1)
    assert tr.ema_decay_at_step(1) == pytest.approx(2 / 11)
    assert tr.ema_decay_at_step(100000) == tr.EMA_DECAY


def test_binary_counts_tracks_thin_stroke_recall_and_blank_false_positive():
    tr = _train_round()
    target = np.zeros((16, 16), dtype=bool)
    target[8, 2:14] = True  # 1 px de espessura
    pred = np.zeros_like(target)
    pred[8, 2:10] = True
    counts = tr.binary_counts(pred, target)
    assert counts["thin_gt"] == 12
    assert counts["thin_tp"] == 8
    metrics = tr._metrics_from_counts(counts)
    assert metrics["thin_ink_recall"] == pytest.approx(8 / 12)

    blank = tr.binary_counts(pred, np.zeros_like(target))
    assert blank["gt_ink"] == 0
    assert blank["fp"] == 8


def _lineage_page(tr, session, holdout=False):
    image = np.full((8, 8, 3), 0.9, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    if holdout:
        return tr.PageExample(
            image, mask, session, "independent_label", "h", split="holdout",
            label_source="final_minus_reference", label_reviewed=True,
        )
    return tr.PageExample(image, mask, session, "human_correction", "t")


def test_run_round_rejects_cold_start_without_explicit_permission(tmp_path):
    tr = _train_round()
    labeled = [_lineage_page(tr, "train-session")]
    holdout = [_lineage_page(tr, "holdout-session", holdout=True)]
    with pytest.raises(ValueError, match="rodada fria"):
        tr.run_round(labeled, [], tmp_path, steps=3, holdout=holdout)


def test_lineage_guard_rejects_holdout_session_in_training():
    tr = _train_round()
    labeled = [_lineage_page(tr, "same-session")]
    holdout = [_lineage_page(tr, "same-session", holdout=True)]
    with pytest.raises(ValueError, match="dois lados"):
        tr._guard_lineage(labeled, [], holdout)


def test_lineage_guard_keeps_previous_rounds_sealed():
    tr = _train_round()
    labeled = [_lineage_page(tr, "old-holdout-now-in-train")]
    current_holdout = [_lineage_page(tr, "new-holdout", holdout=True)]
    with pytest.raises(ValueError, match="dois lados"):
        tr._guard_lineage(
            labeled, [], current_holdout, previous_holdout_ids=["old-holdout-now-in-train"]
        )


def test_checkpoint_returns_sealed_sessions_for_next_round(tmp_path):
    tr = _train_round()
    torch = _REAL_TORCH
    model = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "round.pt"
    torch.save(
        {"student": model.state_dict(), "holdout_sessions_not_for_training": ["sealed"]},
        checkpoint,
    )
    loaded = tr._load_student_checkpoint(model, checkpoint, "cpu")
    assert loaded == ["sealed"]


def test_export_failure_preserves_round_metrics_and_blocks_delivery(
    tmp_path, monkeypatch
):
    tr = _train_round()

    def fail_export(*args, **kwargs):
        raise RuntimeError("export failed for regression test")

    monkeypatch.setattr(tr.torch.onnx, "export", fail_export)
    metrics = tr.run_round(
        [_lineage_page(tr, "train")],
        [],
        tmp_path,
        steps=3,
        batch=2,  # BatchNorm no bottleneck 1x1 exige mais de um valor no treino
        patch=16,
        holdout=[_lineage_page(tr, "holdout", holdout=True)],
        allow_random_init=True,
    )

    assert metrics["delivery_status"] == "blocked_onnx_validation"
    assert metrics["onnx_validation"]["status"] == "failed"
    assert (tmp_path / "round_checkpoint.pt").is_file()
    assert json.loads((tmp_path / "metrics.json").read_text())["steps"] == 3
