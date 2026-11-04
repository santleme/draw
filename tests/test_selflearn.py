"""Testes do pacote draw/selflearn: monitor, coleta, selecao, teacher."""

import hashlib
import json

import numpy as np
import pytest

from draw.selflearn.collect import ExampleRecorder
from draw.selflearn.monitor import (
    classic_witness, divergence, entropy_map, margin, suspicion,
    temporal_violations)
from draw.selflearn.select import rank_examples
from draw.selflearn.teacher import pseudo_label, temporal_aggregate


def test_entropy_peaks_at_half_and_low_at_extremes():
    prob = np.array([[0.5, 0.0, 1.0]], dtype=np.float32)
    ent = entropy_map(prob)
    assert ent[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert ent[0, 1] < 0.05
    assert ent[0, 2] < 0.05


def test_margin_and_suspicion_scalars():
    prob = np.full((10, 10), 0.5, dtype=np.float32)
    assert margin(prob).max() == pytest.approx(0.0, abs=1e-6)
    summary = suspicion(prob)
    assert summary["entropy_mean"] > 0.99
    assert summary["margin_low_frac"] == 1.0
    assert summary["ink_frac"] == 1.0


def test_temporal_violations_flags_only_disappeared_ink():
    prev = np.zeros((4, 4), dtype=np.uint8)
    new = np.zeros((4, 4), dtype=np.uint8)
    prev[0, 0] = 255
    prev[1, 1] = 255
    new[1, 1] = 255
    new[2, 2] = 255
    out = temporal_violations(prev, new)
    assert out[0, 0] == 255
    assert out[1, 1] == 0
    assert out[2, 2] == 0
    assert (out > 0).sum() == 1


def test_divergence_separates_fp_and_fn_candidates():
    prob = np.array([[0.9, 0.1]], dtype=np.float32)
    witness = np.array([[0, 255]], dtype=np.uint8)
    fp_c, fn_c = divergence(prob, witness, thr=0.5)
    assert fp_c[0, 0] == 255 and fp_c[0, 1] == 0
    assert fn_c[0, 1] == 255 and fn_c[0, 0] == 0


def test_classic_witness_marks_dark_stroke():
    page = np.full((60, 60, 3), 240, dtype=np.uint8)
    page[28:32, 10:50] = 30
    witness = classic_witness(page)
    assert witness[30, 30] == 255
    assert witness[5, 5] == 0


def test_recorder_writes_files_and_manifest_hash_matches(tmp_path):
    recorder = ExampleRecorder(tmp_path / "collected")
    warp = np.full((20, 30, 3), 200, dtype=np.uint8)
    prob = np.linspace(0, 1, 20 * 30, dtype=np.float32).reshape(20, 30)
    corners = [[0, 0], [30, 0], [30, 20], [0, 20]]
    row = recorder.record("sess1", warp, prob, corners, "abc123", "live")

    image_path = tmp_path / "collected" / "input" / f"{row['file']}.png"
    prob_path = tmp_path / "collected" / "prob" / f"{row['file']}.png"
    assert image_path.exists() and prob_path.exists()
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert digest == row["sha256"]

    lines = (tmp_path / "collected" / "manifest.jsonl") \
        .read_text().strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["session_id"] == "sess1"
    assert saved["model_sha"] == "abc123"
    assert saved["origin"] == "live"
    assert saved["split"] == "train"
    assert saved["corners"] == corners


def test_recorder_concurrent_same_millisecond_never_overwrites(
        tmp_path, monkeypatch):
    # Dois gravadores, mesmo diretorio, mesma sessao, mesmo
    # milissegundo (issue #21): nomes unicos, nenhum arquivo
    # sobrescrito, e cada linha do manifesto bate com o seu arquivo.
    import types

    from draw.selflearn import collect as collect_module

    frozen = types.SimpleNamespace(time=lambda: 1700000000.0)
    monkeypatch.setattr(collect_module, "time", frozen)

    out = tmp_path / "collected"
    rec_a = ExampleRecorder(out)
    rec_b = ExampleRecorder(out)
    corners = [[0, 0], [30, 0], [30, 20], [0, 20]]
    prob = np.full((20, 30), 0.5, dtype=np.float32)
    warp_a = np.full((20, 30, 3), 200, dtype=np.uint8)
    warp_b = np.full((20, 30, 3), 100, dtype=np.uint8)
    row_a = rec_a.record("sess1", warp_a, prob, corners, "m1", "live")
    row_b = rec_b.record("sess1", warp_b, prob, corners, "m1", "live")

    assert row_a["file"] != row_b["file"]
    for row in (row_a, row_b):
        image_path = out / "input" / f"{row['file']}.png"
        prob_path = out / "prob" / f"{row['file']}.png"
        assert image_path.exists() and prob_path.exists()
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        assert digest == row["sha256"]
    lines = (out / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def _item(session, tta, novelty=0.0):
    return {
        "session_id": session,
        "tta_disagreement": tta,
        "temporal_dispersion": 0.0,
        "region_impurity": 0.0,
        "novelty": novelty,
        "strata": set(),
        "name": f"{session}-{tta}",
    }


def test_ranking_orders_high_disagreement_first_and_respects_cap():
    items = [_item("a", 0.9), _item("a", 0.8), _item("a", 0.7),
             _item("b", 0.6)]
    ranked = rank_examples(items, per_session_cap=2, random_quota=0.0)
    sessions = [r["session_id"] for r in ranked]
    assert sessions.count("a") <= 2
    assert ranked[0]["tta_disagreement"] == 0.9
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert "tta_disagreement" in ranked[0]["reasons"]


def test_ranking_random_quota_filled_from_remainder():
    items = [_item("a", 0.9), _item("a", 0.8), _item("a", 0.7),
             _item("a", 0.6), _item("a", 0.5)]
    ranked = rank_examples(items, per_session_cap=4, random_quota=0.25,
                           seed=3)
    quota = [r for r in ranked if "random_quota" in r["reasons"]]
    assert len(ranked) == 4
    assert len(quota) == 1
    assert quota[0]["tta_disagreement"] == 0.5


def test_ranking_random_quota_filled_without_overflow():
    # Coleta pequena: nenhuma sessao estoura o teto, nao ha overflow,
    # e mesmo assim a cota aleatoria tem de ser cumprida (issue #20).
    items = [_item("a", 0.9), _item("a", 0.8), _item("b", 0.7),
             _item("b", 0.6)]
    ranked = rank_examples(items, per_session_cap=2, random_quota=0.5,
                           seed=1)
    quota = [r for r in ranked if "random_quota" in r["reasons"]]
    assert len(ranked) == 4
    assert len(quota) == 2
    sessions = [r["session_id"] for r in ranked]
    assert sessions.count("a") <= 2
    assert sessions.count("b") <= 2
    assert ranked.random_quota_requested == 2
    assert ranked.random_quota_unmet == 0
    # os itens da cota vao ao fim da fila
    assert all("random_quota" in r["reasons"] for r in ranked[-2:])


def test_ranking_random_quota_unmet_is_recorded():
    # Cota maior que a populacao candidata: o nao atendido fica
    # registrado explicitamente em vez de sumir em silencio.
    items = [_item("a", 0.9), _item("b", 0.8)]
    ranked = rank_examples(items, per_session_cap=3, random_quota=1.5,
                           seed=0)
    quota = [r for r in ranked if "random_quota" in r["reasons"]]
    assert ranked.random_quota_requested == 3
    assert len(quota) == 2
    assert ranked.random_quota_unmet == 1


def test_temporal_aggregate_median_closer_than_single_noisy_frame():
    rng = np.random.default_rng(7)
    clean = np.zeros((16, 16), dtype=np.float32)
    clean[4:12, 7:9] = 0.95
    clean += 0.02
    frames = []
    for _ in range(7):
        noisy = clean + rng.normal(0, 0.25, clean.shape).astype(np.float32)
        frames.append(np.clip(noisy, 0, 1))
    stack = np.stack(frames)
    median, dispersion = temporal_aggregate(stack)
    err_median = np.abs(median - clean).mean()
    err_single = np.abs(stack[0] - clean).mean()
    assert err_median < err_single
    assert dispersion.shape == clean.shape


def _page_with_stroke():
    page = np.full((60, 60, 3), 240, dtype=np.uint8)
    page[28:32, 10:50] = 30
    return page


def test_pseudo_label_veto_and_fn_go_to_ignore():
    page = _page_with_stroke()
    prob = np.full((60, 60), 0.02, dtype=np.float32)
    prob[5, 5] = 0.99
    prob[30, 30] = 0.99
    mask, ignore, confidence = pseudo_label(prob, page)
    assert ignore[5, 5] == 255 and mask[5, 5] == 0
    assert mask[30, 30] == 255 and ignore[30, 30] == 0
    assert confidence[30, 30] > 0.9


def test_pseudo_label_witness_dark_low_prob_is_ignore_not_background():
    page = _page_with_stroke()
    prob = np.full((60, 60), 0.02, dtype=np.float32)
    mask, ignore, confidence = pseudo_label(prob, page)
    assert ignore[30, 30] == 255
    assert mask[30, 30] == 0
    assert confidence[30, 30] == 0.0
    assert ignore[5, 5] == 0
    assert confidence[5, 5] > 0.9
