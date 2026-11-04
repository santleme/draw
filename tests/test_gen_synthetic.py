"""Testes do gerador sintético: determinismo, rótulos e split por fonte.

Sem câmera, sem FreiHAND real (usa mãos placeholder sintéticas).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import gen_synthetic as gs  # noqa: E402


def _hand():
    return gs.synthetic_hand_sources(1)[0]


def test_determinism_same_seed():
    a = gs.compose_scene(7, "train", 3, hand=_hand())
    b = gs.compose_scene(7, "train", 3, hand=_hand())
    assert np.array_equal(a["rgb"], b["rgb"])
    assert np.array_equal(a["mask"], b["mask"])
    assert np.array_equal(a["ink_full"], b["ink_full"])
    c = gs.compose_scene(8, "train", 3, hand=_hand())
    assert not np.array_equal(a["rgb"], c["rgb"])


def test_mask_only_classes_0_to_3():
    for scene_type in ("normal", "hand_no_sheet", "white_object", "empty_table"):
        scene = gs.compose_scene(11, "train", 1, scene_type, hand=_hand())
        assert set(np.unique(scene["mask"]).tolist()) <= {0, 1, 2, 3}
        assert scene["rgb"].shape == (512, 512, 3)
        assert scene["mask"].shape == (512, 512)
        assert scene["ink_full"].shape == (512, 512)


def test_hand_over_stroke_becomes_hand_not_ink():
    # Mão forçada sobre o traço: onde ink_full e mão se sobrepõem,
    # o rótulo visível tem que ser hand (2), nunca ink (3).
    scene = gs.compose_scene(
        21, "train", 2, "normal", hand=_hand(), hand_placement="over_stroke"
    )
    mask, ink_full = scene["mask"], scene["ink_full"]
    assert (ink_full > 0).sum() > 0, "cena sem traço: não testa oclusão"
    assert (mask == 3).sum() > 0, "nenhum traço visível"
    # ink_full contém o desenho completo: visível + ocluído.
    assert (ink_full > 0).sum() >= (mask == 3).sum()
    # Nos pixels de traço cobertos pela mão, o rótulo é hand.
    hand_pixels = mask == 2
    assert hand_pixels.sum() > 0
    # Pelo menos parte do traço some da máscara visível por causa da mão:
    # ink_full > visível, ou a mão cobre traço em algum pixel.
    covered = (ink_full > 0) & hand_pixels
    assert covered.sum() > 0, "mão não cobriu nenhum traço nesta seed"
    assert not np.any(mask[covered] == 3)


def test_ink_full_contains_visible_plus_occluded():
    scene = gs.compose_scene(
        21, "train", 2, "normal", hand=_hand(), hand_placement="covering_much"
    )
    visible = (scene["mask"] == 3).sum()
    full = (scene["ink_full"] > 0).sum()
    assert full > 0
    assert full >= visible


def test_negatives_have_no_paper():
    for scene_type in ("empty_table", "hand_no_sheet", "white_object"):
        scene = gs.compose_scene(5, "train", 0, scene_type, hand=_hand())
        assert (scene["mask"] == 1).sum() == 0, scene_type
        assert (scene["mask"] == 3).sum() == 0, scene_type


def test_split_has_no_source_leakage():
    splits = gs.split_sources([f"src_{i}" for i in range(20)], seed=3)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 20


def test_generate_dataset_split_no_leakage_and_files(tmp_path):
    manifest = gs.generate_dataset(tmp_path, seed=99, counts={"train": 6, "val": 2, "test": 2})
    assert len(manifest["scenes"]) == 10
    for split in ("train", "val", "test"):
        for scene in manifest["scenes"]:
            if scene["split"] != split:
                continue
            for fname in scene["files"]:
                assert (tmp_path / split / fname).is_file()
            mask = cv2.imread(str(tmp_path / split / scene["files"][1]), cv2.IMREAD_GRAYSCALE)
            assert set(np.unique(mask).tolist()) <= {0, 1, 2, 3}
    # Nenhuma fonte (mão, papel, fundo) em dois splits.
    for key in ("hand_source", "paper_source", "background_source"):
        per_split = {}
        for scene in manifest["scenes"]:
            per_split.setdefault(scene["split"], set()).add(scene[key])
        assert per_split["train"].isdisjoint(per_split["val"])
        assert per_split["train"].isdisjoint(per_split["test"])
        assert per_split["val"].isdisjoint(per_split["test"])


def test_placeholder_hand_is_marked():
    scene = gs.compose_scene(2, "train", 0, hand=_hand())
    assert scene["info"]["hand_placeholder"] is True
    assert scene["info"]["hand_source"].startswith("synthetic_hand_")


def test_two_hands_scene():
    hands = gs.synthetic_hand_sources(2)
    scene = gs.compose_scene(4, "train", 0, hand=hands[0], second_hand=hands[1])
    assert (scene["mask"] == 2).sum() > 0


def test_load_hands_empty_dir_returns_empty(tmp_path):
    assert gs.load_hands(tmp_path) == []
    assert gs.load_hands(None) == []


def test_load_hands_uses_documented_jpeg_threshold(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "mask").mkdir()
    cv2.imwrite(str(tmp_path / "rgb" / "00000000.jpg"),
                np.zeros((2, 6, 3), dtype=np.uint8))
    values = np.array([[0, 1, 7, 126, 127, 128]], dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "mask" / "00000000.png"),
                np.repeat(values, 2, axis=0))

    hand = gs.load_hands(tmp_path)[0]

    assert hand.mask[0].tolist() == [0, 0, 0, 0, 0, 255]
    assert hand.base_id == "00000000"


def test_freihand_variants_stay_in_the_same_split():
    hands = []
    for base in range(3):
        for variant in range(4):
            source_id = f"{base + variant * gs.FREIHAND_BASE_HAND_COUNT:08d}"
            hands.append(gs.HandSource(
                source_id,
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.uint8),
                base_id=gs.freihand_base_id(source_id),
            ))

    variant_splits, base_splits = gs.split_hand_sources(hands, seed=42)
    for base in range(3):
        destinations = {
            split for split, ids in variant_splits.items()
            if f"{base:08d}" in ids
        }
        assert len(destinations) == 1
        base_id = f"{base:08d}"
        for variant in range(1, 4):
            source_id = f"{base + variant * gs.FREIHAND_BASE_HAND_COUNT:08d}"
            assert source_id in variant_splits[next(iter(destinations))]
        assert base_id in base_splits[next(iter(destinations))]
