import csv
import json

import cv2
import numpy as np
import pytest

from draw.selflearn import external, review, split


class Teacher:
    threshold = 0.7

    def __init__(self):
        self.calls = 0

    def ink_prob(self, image):
        self.calls += 1
        return (image[:, :, 0] < 100).astype(np.float32)


def catalog(tmp_path, groups=("alice", "alice", "bob"), prior=False):
    root = tmp_path / "raw"
    root.mkdir()
    rows = []
    for i, group in enumerate(groups):
        image = np.full((40, 50, 3), 255, np.uint8)
        image[5:15, 5 + i : 15 + i] = 0
        cv2.imwrite(str(root / f"{i}.png"), image)
        rows.append(
            {
                "session_id": group,
                "image_path_in_archive": f"{i}.png",
                "session_used_for_prior_training": str(prior).lower(),
            }
        )
    with (root / "frames.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_train_queues_pseudo_never_approves_and_roundtrip(tmp_path):
    root = catalog(tmp_path)
    pool = tmp_path / "pool"
    model = Teacher()
    report = external.translate(
        root, pool, "drawings", split="train", model=model, images_are_pages=True
    )
    assert report["train"] == 3 and report["reviewed"] == 0 and model.calls == 3
    rows = review.review_pages(pool, "train")
    assert len({r["session_id"] for r in rows}) == 2
    assert not review.status(pool, split="train")["gate"]["n_reached"]
    for row in rows:
        review.mark_reviewed(
            pool, row["session_id"], row["generation"], accept_draft=True, split="train"
        )
    exported = tmp_path / "export"
    assert review.export_reviewed(pool, exported, "train")["pages_exported"] == 3
    metadata = [
        json.loads(s) for s in (exported / "manifest.jsonl").read_text().splitlines()
    ]
    assert all(
        r["origin"] == "pseudo_incumbent" and r["label_reviewed"] for r in metadata
    )
    for row in metadata:
        assert row["source"]["dataset"] == "drawings"
        assert len(row["input_sha256"]) == 64
        split.guard_train_example(row, [])
        with pytest.raises(split.LeakageError):
            split.guard_holdout_example(row)
    assert (pool / ".gitignore").is_file() and (exported / ".gitignore").is_file()


def test_holdout_teacher_is_never_called_and_requires_independent_label(tmp_path):
    root = catalog(tmp_path)
    pool = tmp_path / "pool"
    model = Teacher()
    external.translate(
        root, pool, "drawings", split="holdout", model=model, images_are_pages=True
    )
    assert model.calls == 0
    rows = review.holdout_pages(pool)
    for row in rows:
        with pytest.raises(split.LeakageError):
            review.mark_reviewed(
                pool, row["session_id"], row["generation"], accept_draft=True
            )
        label = np.zeros((40, 50), np.uint8)
        label[5:15, 5:15] = 255
        path = tmp_path / "human.png"
        cv2.imwrite(str(path), label)
        review.mark_reviewed(
            pool, row["session_id"], row["generation"], label_path=path
        )
    result = review.export_holdout(pool, tmp_path / "out")
    assert result["pages_exported"] == 3
    metadata = [
        json.loads(s)
        for s in (tmp_path / "out/manifest.jsonl").read_text().splitlines()
    ]
    for row in metadata:
        assert row["label_source"] == "human_from_scratch"
        split.guard_holdout_example(row)


def test_import_is_idempotent_preserves_review_and_seal(tmp_path):
    root = catalog(tmp_path)
    pool = tmp_path / "pool"
    model = Teacher()
    kw = dict(split="train", model=model, images_are_pages=True)
    first = external.translate(root, pool, "dataset", **kw)
    row = review.review_pages(pool, "train")[0]
    review.mark_reviewed(
        pool, row["session_id"], row["generation"], accept_draft=True, split="train"
    )
    before = (pool / "reviews.json").read_bytes()
    assert external.translate(root, pool, "dataset", **kw) == first
    assert model.calls == 3 and (pool / "reviews.json").read_bytes() == before
    with pytest.raises(split.LeakageError):
        external.translate(root, pool, "dataset", **dict(kw, split="holdout"))
    image = np.zeros((40, 50, 3), np.uint8)
    cv2.imwrite(str(root / "0.png"), image)
    with pytest.raises(split.LeakageError):
        external.translate(root, pool, "dataset", **kw)


def test_prior_training_sessions_cannot_be_holdout(tmp_path):
    root = catalog(tmp_path, prior=True)
    with pytest.raises(split.LeakageError):
        external.translate(
            root, tmp_path / "bad", "dataset", split="holdout", images_are_pages=True
        )
    external.translate(
        root, tmp_path / "ok", "dataset", model=Teacher(), images_are_pages=True
    )
    assert len(review.review_pages(tmp_path / "ok", "train")) == 3
    assert not review.holdout_pages(tmp_path / "ok")


def test_unmanifested_images_require_batch_not_one_session_per_frame(tmp_path):
    root = catalog(tmp_path)
    (root / "frames.csv").unlink()
    with pytest.raises(ValueError):
        external.load_catalog(root)
    rows = external.load_catalog(root, batch="one-author-batch")
    assert {r["group"] for r in rows} == {"one-author-batch"}


@pytest.mark.parametrize("relative", ["../escape.png", "/etc/passwd"])
def test_catalog_rejects_escaping_paths(tmp_path, relative):
    root = catalog(tmp_path)
    (root / "frames.csv").write_text(f"image_path,session_id\n{relative},alice\n")
    with pytest.raises(ValueError):
        external.load_catalog(root)


def test_exact_duplicates_across_sessions_are_rejected(tmp_path):
    root = catalog(tmp_path, groups=("a", "b"))
    (root / "1.png").write_bytes((root / "0.png").read_bytes())
    with pytest.raises(split.LeakageError):
        external.translate(
            root, tmp_path / "pool", "dataset", model=Teacher(), images_are_pages=True
        )


def test_raw_frames_need_rectification_and_missing_pages_are_reported(tmp_path):
    root = catalog(tmp_path)
    with pytest.raises(ValueError, match="retificação"):
        external.translate(root, tmp_path / "bad", "dataset", model=Teacher())
    report = external.translate(
        root,
        tmp_path / "ok",
        "dataset",
        split="train",
        model=Teacher(),
        prepare_page=lambda image: None,
    )
    assert report["pages_queued"] == 0 and len(report["problems"]) == 3


def test_partial_session_never_exports(tmp_path):
    root = catalog(tmp_path, groups=("a", "a"))
    pool = tmp_path / "pool"
    external.translate(
        root, pool, "dataset", split="train", model=Teacher(), images_are_pages=True
    )
    row = review.review_pages(pool, "train")[0]
    review.mark_reviewed(
        pool, row["session_id"], row["generation"], accept_draft=True, split="train"
    )
    assert (
        review.export_reviewed(pool, tmp_path / "out", "train")["pages_exported"] == 0
    )


def test_external_input_mutation_invalidates_review(tmp_path):
    root = catalog(tmp_path)
    pool = tmp_path / "pool"
    external.translate(
        root, pool, "dataset", split="train", model=Teacher(), images_are_pages=True
    )
    row = review.review_pages(pool, "train")[0]
    cv2.imwrite(row["files"]["final_warp"], np.zeros((40, 50, 3), np.uint8))
    with pytest.raises(ValueError, match="mudou"):
        review.mark_reviewed(
            pool, row["session_id"], row["generation"], accept_draft=True, split="train"
        )


def test_output_cannot_live_inside_raw_root(tmp_path):
    root = catalog(tmp_path)
    with pytest.raises(ValueError, match="fora da raiz"):
        external.translate(
            root, root / "output", "dataset", model=Teacher(), images_are_pages=True
        )
    assert not (root / "output").exists()


def test_explicit_missing_manifest_never_falls_back_to_batch(tmp_path):
    root = catalog(tmp_path)
    with pytest.raises(FileNotFoundError):
        external.load_catalog(root, manifest=root / "missing.csv", batch="all")


def test_model_and_preparation_are_part_of_the_immutable_seal(tmp_path):
    root = catalog(tmp_path)
    model = Teacher()
    model.model_path = tmp_path / "model.onnx"
    model.model_path.write_bytes(b"incumbent")
    pool = tmp_path / "pool"
    kw = dict(
        split="train",
        model=model,
        prepare_page=lambda image: image,
        preparation_id={"detector": "v1"},
    )
    external.translate(root, pool, "dataset", **kw)
    with pytest.raises(split.LeakageError):
        external.translate(root, pool, "dataset", **dict(kw, preparation_id="v2"))
    model.model_path.write_bytes(b"challenger")
    with pytest.raises(split.LeakageError):
        external.translate(root, pool, "dataset", **kw)


def test_auto_split_is_sealed_per_group_and_audit_respects_split(tmp_path):
    root = catalog(tmp_path, groups=tuple(f"author-{i // 2}" for i in range(20)))
    pool = tmp_path / "pool"
    external.translate(root, pool, "dataset", model=Teacher(), images_are_pages=True)
    rows = review.review_pages(pool, "train") + review.holdout_pages(pool)
    for group in {r["source"]["group"] for r in rows}:
        assert len({r["split"] for r in rows if r["source"]["group"] == group}) == 1
    audit = review.audit_pool(pool, split="train")
    assert len(audit["pages"]) == len(review.review_pages(pool, "train"))
    assert not audit["status"]["gate"]["n_reached"]
