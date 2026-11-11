"""Revisao do holdout selado (draw/selflearn/review.py + CLI)."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from draw.selflearn import process, review
from tests.test_selflearn_process import _make_session

_CLI = (
    Path(__file__).resolve().parent.parent / "tools" / "selflearn" / "review_holdout.py"
)


def _pool(tmp_path):
    sessions = tmp_path / "sessions"
    for i in range(9):
        _make_session(sessions, f"s{i:02d}-fake", [True, True, i % 2 == 0])
    out = tmp_path / "pool"
    process.process_pool(sessions, out, min_holdout_sessions=5)
    return sessions, out


def _mark_all(pool, only=None):
    for row in review.holdout_pages(pool):
        if only is None or row["session_id"] in only:
            review.mark_reviewed(
                pool, row["session_id"], row["generation"], accept_draft=True
            )


def test_status_starts_all_pending(tmp_path):
    _, pool = _pool(tmp_path)
    report = review.status(pool)
    assert report["sessions_holdout"] >= 5
    assert report["pages_reviewed"] == 0 and report["sessions_complete"] == 0
    assert not report["gate"]["n_reached"] and report["gate"]["promotes"] is False
    assert "faltam: gen_0001" in review.format_status(report)


def test_mark_requires_explicit_proof_and_existing_page(tmp_path):
    _, pool = _pool(tmp_path)
    row = review.holdout_pages(pool)[0]
    with pytest.raises(ValueError):
        review.mark_reviewed(pool, row["session_id"], row["generation"])
    with pytest.raises(ValueError):
        review.mark_reviewed(
            pool,
            row["session_id"],
            row["generation"],
            accept_draft=True,
            label_path="x.png",
        )
    with pytest.raises(KeyError):
        review.mark_reviewed(pool, "nao-existe", 1, accept_draft=True)


def test_corrected_label_survives_reprocessing_and_counts(tmp_path):
    sessions, pool = _pool(tmp_path)
    row = review.holdout_pages(pool)[0]
    sid, gen = row["session_id"], row["generation"]
    gen_dir = pool / "holdout" / sid / f"gen_{gen:04d}"
    draft = cv2.imread(str(gen_dir / "label_draft.png"), cv2.IMREAD_GRAYSCALE)
    corrected = draft.copy()
    corrected[0:4, 0:4] = 255
    path = tmp_path / "fix.png"
    cv2.imwrite(str(path), corrected)
    entry = review.mark_reviewed(pool, sid, gen, label_path=path, reviewer="santleme")
    assert entry["mode"] == "corrected" and entry["changed_pixels"] == 16
    summary = process.process_pool(sessions, pool, min_holdout_sessions=5)
    assert summary["pages_holdout_reviewed"] == 1
    kept = cv2.imread(str(gen_dir / "label_reviewed.png"), cv2.IMREAD_GRAYSCALE)
    assert (kept == corrected).all()
    log = (pool / "reviews_log.jsonl").read_text().splitlines()
    assert json.loads(log[0])["reviewer"] == "santleme"
    bad = tmp_path / "bad.png"
    cv2.imwrite(str(bad), np.zeros((3, 3), np.uint8))
    with pytest.raises(ValueError):
        review.mark_reviewed(pool, sid, gen, label_path=bad)


def test_session_completes_only_when_all_pages_reviewed(tmp_path):
    _, pool = _pool(tmp_path)
    rows = review.holdout_pages(pool)
    sid = rows[0]["session_id"]
    first = [r for r in rows if r["session_id"] == sid][0]
    review.mark_reviewed(pool, sid, first["generation"], accept_draft=True)
    assert not review.status(pool)["sessions"][sid]["complete"]
    _mark_all(pool, only={sid})
    report = review.status(pool)
    assert report["sessions"][sid]["complete"]
    assert report["sessions_complete"] == 1


def test_legacy_review_without_label_file_is_loud(tmp_path):
    _, pool = _pool(tmp_path)
    row = review.holdout_pages(pool)[0]
    key = f"{row['session_id']}/gen_{row['generation']:04d}"
    (pool / "reviews.json").write_text(json.dumps({key: True}))
    report = review.status(pool)
    assert report["pages_reviewed"] == 1
    assert any(key in p for p in report["problems"])


def test_export_only_complete_sessions_and_loads_in_train_round_format(tmp_path):
    _, pool = _pool(tmp_path)
    sessions = sorted({r["session_id"] for r in review.holdout_pages(pool)})
    _mark_all(pool, only=set(sessions[:-1]))
    out = tmp_path / "holdout_export"
    result = review.export_holdout(pool, out)
    assert result["sessions_exported"] == sessions[:-1]
    assert sessions[-1] in result["skipped"]
    manifest = [
        json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(manifest) == result["pages_exported"]
    from draw.selflearn.split import guard_holdout_example

    for row in manifest:
        guard_holdout_example(dict(row, file=Path(row["file"]).stem))
        assert (out / "input" / row["file"]).is_file()
        assert (out / "mask" / row["file"]).is_file()
    with pytest.raises(FileExistsError):
        review.export_holdout(pool, out)


def test_cli_status_and_mark(tmp_path, capsys):
    _, pool = _pool(tmp_path)
    spec = importlib.util.spec_from_file_location("review_holdout", _CLI)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    row = review.holdout_pages(pool)[0]
    assert cli.main(["--pool", str(pool), "status"]) == 0
    assert "holdout selado" in capsys.readouterr().out
    argv = ["--pool", str(pool), "mark", row["session_id"], str(row["generation"])]
    with pytest.raises(SystemExit):
        cli.main(argv)  # sem prova explicita
    assert cli.main(argv + ["--accept-draft"]) == 0
    assert review.status(pool)["pages_reviewed"] == 1
