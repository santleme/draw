"""Testes do portao real (U3, loop v2).

Pinam a correcao 4 do Lucas (2026-11-02): DIBCO e diagnostico e
nunca promove; o replay real selado decide; abaixo do n minimo
pre-registrado o veredito e INCONCLUSIVO, nunca PASS.
"""

import json
import subprocess
import sys
from pathlib import Path

from draw.selflearn.gate import (
    INCONCLUSIVE,
    PROMOTE,
    REJECT,
    format_verdict,
    judge_candidate,
)

ROOT = Path(__file__).resolve().parent.parent


def _real(primary, worst, sessions=8, pages=40):
    return {
        "primary": primary,
        "worst_group": worst,
        "holdout_sessions": sessions,
        "holdout_pages": pages,
    }


def test_dibco_winner_that_loses_on_real_is_rejected():
    result = judge_candidate(
        _real(0.30, 0.20),
        _real(0.45, 0.35),
        dibco={"iou": 0.223, "baseline_iou": 0.188},
    )
    assert result["verdict"] == REJECT
    assert result["dibco_diagnostic"]["promotes"] is False
    assert result["dibco_diagnostic"]["role"] == "diagnostic_only"


def test_insufficient_n_is_inconclusive_never_pass():
    result = judge_candidate(
        _real(0.90, 0.80, sessions=2, pages=48), _real(0.40, 0.30), dibco={"iou": 0.99}
    )
    assert result["verdict"] == INCONCLUSIVE
    assert result["n"] == {"holdout_sessions": 2, "holdout_pages": 48}
    text = format_verdict(result)
    assert "2 sessoes" in text
    assert "INCONCLUSIVE" in text


def test_insufficient_pages_is_inconclusive():
    result = judge_candidate(_real(0.90, 0.80, sessions=9, pages=10), _real(0.40, 0.30))
    assert result["verdict"] == INCONCLUSIVE


def test_good_candidate_with_enough_n_promotes():
    result = judge_candidate(
        _real(0.50, 0.36), _real(0.45, 0.35), dibco={"iou": 0.15, "baseline_iou": 0.22}
    )
    assert result["verdict"] == PROMOTE


def test_worst_group_regression_rejects_despite_primary_gain():
    result = judge_candidate(_real(0.60, 0.20), _real(0.45, 0.35))
    assert result["verdict"] == REJECT
    assert any("pior grupo" in r for r in result["reasons"])


def test_promote_margin_is_enforced():
    result = judge_candidate(_real(0.46, 0.36), _real(0.45, 0.35), promote_margin=0.02)
    assert result["verdict"] == REJECT


def test_cli_exit_codes_and_dibco_label(tmp_path):
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    dibco = tmp_path / "dibco.json"
    candidate.write_text(json.dumps(_real(0.30, 0.20)))
    baseline.write_text(json.dumps(_real(0.45, 0.35)))
    dibco.write_text(json.dumps({"iou": 0.223}))
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/selflearn/judge_gate.py"),
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--dibco",
            str(dibco),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 1
    assert "REJECT" in run.stdout
    assert "diagnostico" in run.stdout


def test_cli_worst_group_tolerance_flag_reaches_the_gate(tmp_path):
    # judge_candidate aceita worst_group_tolerance, mas a CLI nao
    # expunha a flag: a tolerancia ficava sempre 0 na pratica.
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    candidate.write_text(json.dumps(_real(0.50, 0.33)))
    baseline.write_text(json.dumps(_real(0.45, 0.35)))
    cmd = [
        sys.executable,
        str(ROOT / "tools/selflearn/judge_gate.py"),
        "--candidate",
        str(candidate),
        "--baseline",
        str(baseline),
    ]
    strict = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert strict.returncode == 1  # pior grupo regrediu 0,02 sem tolerancia
    tolerant = subprocess.run(
        cmd + ["--worst-group-tolerance", "0.05"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert tolerant.returncode == 0
    assert "PROMOTE" in tolerant.stdout


# --- Regra de parada (ordem de adocao 2026-11-06) ---

from draw.selflearn.gate import (  # noqa: E402
    evaluate_rounds,
    format_stop_decision,
    session_bootstrap_delta,
)

import pytest  # noqa: E402


def _scores(base, deltas):
    return {f"s{i}": base + d for i, d in enumerate(deltas)}


def _round(label, candidate, baseline=None):
    entry = {"round": label, "candidate_by_session": candidate}
    if baseline is not None:
        entry["baseline_by_session"] = baseline
    return entry


_BASE = _scores(0.80, [0.0, 0.01, -0.01, 0.0, 0.02, -0.02, 0.0, 0.01])
_GAIN = _scores(0.86, [0.0, 0.01, -0.01, 0.0, 0.02, -0.02, 0.0, 0.01])
_FLAT = _scores(0.80, [0.01, -0.02, 0.0, 0.02, -0.01, 0.0, 0.01, -0.01])


def test_bootstrap_delta_of_identical_scores_is_zero():
    delta, low, high = session_bootstrap_delta(_BASE, _BASE)
    assert delta == 0.0 and low == 0.0 and high == 0.0


def test_bootstrap_rejects_unpaired_sessions():
    with pytest.raises(ValueError):
        session_bootstrap_delta({"a": 1.0}, {"b": 1.0})


def test_bootstrap_is_deterministic():
    first = session_bootstrap_delta(_GAIN, _BASE)
    second = session_bootstrap_delta(_GAIN, _BASE)
    assert first == second
    assert first[0] > 0 and first[1] > 0


def test_two_no_gain_rounds_stop_the_loop():
    result = evaluate_rounds(
        [
            _round(1, _FLAT, baseline=_BASE),
            _round(2, _FLAT),
        ]
    )
    assert result["should_stop"] is True
    assert result["consecutive_no_gain"] == 2
    assert [r["outcome"] for r in result["per_round"]] == ["NO_GAIN", "NO_GAIN"]


def test_gain_resets_the_counter_and_moves_the_champion():
    result = evaluate_rounds(
        [
            _round(1, _FLAT, baseline=_BASE),
            _round(2, _GAIN),
            _round(3, _GAIN),
            _round(4, _GAIN),
        ]
    )
    assert result["should_stop"] is True
    assert result["consecutive_no_gain"] == 2
    assert result["champion_round"] == 2
    assert [r["outcome"] for r in result["per_round"]] == [
        "NO_GAIN",
        "GAIN",
        "NO_GAIN",
        "NO_GAIN",
    ]


def test_inconclusive_round_neither_gains_nor_counts():
    small = _scores(0.99, [0.0, 0.0, 0.0])
    result = evaluate_rounds(
        [
            _round(1, _FLAT, baseline=_BASE),
            _round(2, small),
            _round(3, _FLAT),
        ]
    )
    assert result["should_stop"] is True
    assert [r["outcome"] for r in result["per_round"]] == [
        "NO_GAIN",
        INCONCLUSIVE,
        "NO_GAIN",
    ]


def test_holdout_change_mid_sequence_raises():
    other = _scores(0.90, [0.0] * 8)
    other = {f"x{i}": v for i, v in enumerate(other.values())}
    with pytest.raises(ValueError):
        evaluate_rounds(
            [
                _round(1, _GAIN, baseline=_BASE),
                _round(2, other),
            ]
        )


def test_first_round_without_baseline_raises():
    with pytest.raises(ValueError):
        evaluate_rounds([_round(1, _GAIN)])


def test_stop_decision_text_stamps_preregistration():
    result = evaluate_rounds(
        [
            _round(1, _FLAT, baseline=_BASE),
            _round(2, _FLAT),
        ]
    )
    text = format_stop_decision(result)
    assert "parada: SIM" in text
    assert "bootstrap por sessao" in text


def test_accepted_drift_is_signal_that_never_changes_verdict():
    from draw.selflearn import gate as g

    cand = {
        "primary": 0.8,
        "worst_group": 0.7,
        "holdout_sessions": 5,
        "holdout_pages": 25,
    }
    base = {"primary": 0.7, "worst_group": 0.7}
    plain = g.judge_candidate(cand, base)
    for status in ("OK", "ATTENTION", "DRIFT", "INSUFFICIENT", "NO_REFERENCE"):
        result = g.judge_candidate(cand, base, accepted_drift=status)
        assert result["verdict"] == plain["verdict"]
        assert result["judgment_signals"]["changes_verdict"] is False
        assert result["judgment_signals"]["human_review_required"] == (
            status in ("ATTENTION", "DRIFT")
        )
    assert "deriva dos aceitos" in g.format_verdict(
        g.judge_candidate(cand, base, accepted_drift="DRIFT")
    )
    assert plain["judgment_signals"] is None
