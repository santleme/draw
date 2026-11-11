"""Monitor de deriva dos aceitos (draw/selflearn/drift.py)."""

import numpy as np
import pytest

from draw.selflearn import drift


def _profile(rng, n_items, ink_mean=0.9, ink_frac=0.05, size=32):
    profile = drift.AcceptedProfile()
    for _ in range(n_items):
        prob = np.clip(rng.normal(0.03, 0.02, (size, size)), 0, 1)
        target = rng.random((size, size)) < ink_frac
        prob[target] = np.clip(rng.normal(ink_mean, 0.03, int(target.sum())), 0, 1)
        weight = np.ones((size, size))
        profile.add_item(prob, target, weight)
    return profile.to_dict()


def test_ks_matches_known_values():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    d, p = drift.ks_2samp(a, a)
    assert d == 0.0 and p == 1.0
    d, p = drift.ks_2samp(a, rng.normal(size=400) + 0.5)
    assert d > 0.2 and p < 1e-6


def test_psi_identical_is_zero_and_shift_crosses_thresholds():
    rng = np.random.default_rng(1)
    a = rng.normal(size=5000)
    assert drift.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert drift.psi_status(drift.psi(a, a + 0.4)) == drift.ATTENTION
    assert drift.psi_status(drift.psi(a, a + 1.0)) == drift.DRIFT
    assert drift.psi_status(0.05) == drift.OK
    assert drift.psi_status(0.1) == drift.ATTENTION
    assert drift.psi_status(0.25) == drift.DRIFT


def test_psi_bins_shrink_with_small_samples():
    assert drift.psi_bins(60, 60) == 2
    assert drift.psi_bins(5000, 5000) == 10
    # sem deriva, 60 itens nao podem acender alarme so pelo vies do PSI
    rng = np.random.default_rng(9)
    values = [
        drift.psi(rng.random(60), rng.random(60), bins=drift.psi_bins(60, 60))
        for _ in range(200)
    ]
    assert np.mean(values) < drift.PSI_ATTENTION / 2


def test_psi_rejects_empty():
    with pytest.raises(ValueError):
        drift.psi([], [1.0])


def test_same_distribution_is_ok():
    report = drift.compare_profiles(
        _profile(np.random.default_rng(2), 60), _profile(np.random.default_rng(3), 60)
    )
    assert report["status"] == drift.OK
    assert report["changes_training"] is False


def test_known_drift_in_ink_is_flagged_per_class():
    ref = _profile(np.random.default_rng(4), 60, ink_mean=0.92, ink_frac=0.04)
    cur = _profile(np.random.default_rng(5), 60, ink_mean=0.72, ink_frac=0.15)
    report = drift.compare_profiles(ref, cur)
    assert report["status"] == drift.DRIFT
    assert report["classes"]["ink"]["status"] == drift.DRIFT
    assert report["classes"]["ink"]["features"]["accepted_fraction"]["ks_p"] < 0.01


def test_small_sample_is_insufficient_not_ok():
    rng = np.random.default_rng(6)
    report = drift.compare_profiles(_profile(rng, 60), _profile(rng, 5))
    feats = report["classes"]["ink"]["features"]
    assert feats["accepted_fraction"]["status"] == drift.INSUFFICIENT
    assert report["status"] != drift.OK


def test_absent_class_is_insufficient():
    profile = drift.AcceptedProfile()
    for _ in range(40):
        profile.add_item(np.full((8, 8), 0.02), np.zeros((8, 8)), np.ones((8, 8)))
    data = profile.to_dict()
    assert data["classes"]["ink"]["accepted_pixels_total"] == 0
    report = drift.compare_profiles(data, data)
    ink = report["classes"]["ink"]["features"]
    assert ink["confidence_pixels"]["status"] == drift.INSUFFICIENT
    assert report["classes"]["background"]["status"] == drift.OK


def test_ignored_pixels_are_not_profiled():
    profile = drift.AcceptedProfile()
    weight = np.zeros((8, 8))
    profile.add_item(np.full((8, 8), 0.99), np.ones((8, 8)), weight)
    data = profile.to_dict()
    assert data["classes"]["ink"]["accepted_pixels_total"] == 0
    assert data["classes"]["ink"]["item_accepted_fraction"] == [0.0]


def test_profile_round_trips_through_json_and_rejects_unknown(tmp_path):
    import json

    data = _profile(np.random.default_rng(7), 31)
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(data))
    assert drift.compare_profiles(path, data)["status"] == drift.OK
    with pytest.raises(ValueError):
        drift.load_profile({"version": 99})


def test_format_report_mentions_thresholds_and_no_reference():
    data = _profile(np.random.default_rng(8), 40)
    text = drift.format_report(drift.compare_profiles(data, data))
    assert "PSI atencao >= 0.1" in text and "ink:" in text
    assert drift.NO_REFERENCE in drift.format_report(drift.no_reference_report())
