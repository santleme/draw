"""Conformal por classe do juizo dos aceitos (draw/selflearn/conformal.py)."""

import math

import numpy as np
import pytest

from draw.selflearn import conformal


def _pairs(rng, n_items, size=32, ink_frac=0.1, noise=0.15):
    pairs = []
    for _ in range(n_items):
        label = (rng.random((size, size)) < ink_frac).astype(np.uint8)
        prob = np.where(label == 1, 0.8, 0.2) + rng.normal(0, noise, label.shape)
        pairs.append((np.clip(prob, 0, 1), label))
    return pairs


def test_quantile_is_finite_sample_rank():
    scores = np.arange(1, 11) / 10.0
    # ceil(11 * 0.9) = 10 -> o maior
    assert conformal.conformal_quantile(scores, 0.1) == 1.0
    # ceil(11 * 0.5) = 6 -> o sexto
    assert conformal.conformal_quantile(scores, 0.5) == 0.6
    assert math.isinf(conformal.conformal_quantile(scores[:5], 0.1))
    with pytest.raises(ValueError):
        conformal.conformal_quantile(scores, 1.5)


def test_per_class_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)
    calib = conformal.calibrate(_pairs(rng, 20), alpha=0.1)
    assert calib["classes"]["ink"]["supported"]
    cov = conformal.coverage(_pairs(rng, 20), calib)
    for c in conformal.CLASSES:
        assert cov[c]["coverage"] >= 0.88  # 1 - alpha, com folga de amostra


def test_rare_class_gets_its_own_quantile():
    rng = np.random.default_rng(1)
    pairs = []
    for _ in range(20):
        label = (rng.random((32, 32)) < 0.05).astype(np.uint8)
        # tinta rara e mais incerta que o fundo dominante
        prob = np.where(label == 1, rng.uniform(0.4, 1.0, label.shape), 0.05)
        pairs.append((prob, label))
    calib = conformal.calibrate(pairs, alpha=0.1)
    assert calib["classes"]["ink"]["q"] > calib["classes"]["background"]["q"]
    cov = conformal.coverage(pairs, calib)
    assert cov["ink"]["coverage"] >= 0.9


def test_unsupported_class_never_accepted():
    pairs = [(np.full((16, 16), 0.1), np.zeros((16, 16), np.uint8))]
    calib = conformal.calibrate(pairs, alpha=0.1)
    assert not calib["classes"]["ink"]["supported"]
    mask, ignore = conformal.conformal_accept(np.full((4, 4), 0.99), calib)
    assert not mask.any() and (ignore == 255).all()


def test_accept_requires_singleton_set():
    calib = {
        "version": conformal.CALIBRATION_VERSION,
        "classes": {
            "ink": {"q": 0.3, "supported": True, "n": 1000},
            "background": {"q": 0.3, "supported": True, "n": 1000},
        },
    }
    prob = np.array([[0.95, 0.5, 0.05]])
    mask, ignore = conformal.conformal_accept(prob, calib)
    assert mask.tolist() == [[255, 0, 0]]
    assert ignore.tolist() == [[0, 255, 0]]  # 0.5: conjunto vazio -> ignore


def test_ignore_label_and_validation():
    prob = np.full((4, 4), 0.9)
    label = np.full((4, 4), 255, np.uint8)
    calib = conformal.calibrate([(prob, label)], min_pixels=1)
    assert calib["classes"]["ink"]["n"] == 0
    with pytest.raises(ValueError):
        conformal.calibrate([(prob, np.full((4, 4), 2, np.uint8))])
    with pytest.raises(ValueError):
        conformal.calibrate([(prob, np.zeros((3, 3), np.uint8))])
    with pytest.raises(ValueError):
        conformal.calibrate([(prob + 1, np.zeros((4, 4), np.uint8))])
    with pytest.raises(ValueError):
        conformal.prediction_sets(prob, {"version": 0, "classes": {}})
