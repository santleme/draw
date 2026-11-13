"""Testes do gerador sintético: determinismo, rótulos e split por fonte.

Sem câmera, sem FreiHAND real (usa mãos placeholder sintéticas).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

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


def test_mask_only_classes_0_to_3_and_ignore():
    # Contrato atual (regra do Lucas, 2026-11-05): classes 0..3 e
    # 255=ignore para pixel ambíguo. Antes era só 0..3 — todo pixel
    # ganhava um rótulo duro, inclusive os de borda sub-pixel.
    for scene_type in ("normal", "hand_no_sheet", "white_object", "empty_table"):
        scene = gs.compose_scene(11, "train", 1, scene_type, hand=_hand())
        assert set(np.unique(scene["mask"]).tolist()) <= {0, 1, 2, 3, 255}
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
    manifest = gs.generate_dataset(
        tmp_path, seed=99, counts={"train": 6, "val": 2, "test": 2}
    )
    assert len(manifest["scenes"]) == 10
    for split in ("train", "val", "test"):
        for scene in manifest["scenes"]:
            if scene["split"] != split:
                continue
            for fname in scene["files"]:
                assert (tmp_path / split / fname).is_file()
            mask = cv2.imread(
                str(tmp_path / split / scene["files"][1]), cv2.IMREAD_GRAYSCALE
            )
            assert set(np.unique(mask).tolist()) <= {0, 1, 2, 3, 255}
    assert manifest["ignore_index"] == 255
    # Nenhuma fonte (mão, papel, fundo) em dois splits.
    for key in ("hand_source", "paper_source", "background_source"):
        per_split = {}
        for scene in manifest["scenes"]:
            per_split.setdefault(scene["split"], set()).add(scene[key])
        assert per_split["train"].isdisjoint(per_split["val"])
        assert per_split["train"].isdisjoint(per_split["test"])
        assert per_split["val"].isdisjoint(per_split["test"])


def test_ambiguity_ignore_unit_rules():
    # Regra do Lucas: pixel ambíguo = 255, nunca rótulo inventado.
    # Unidade direta sobre _ambiguity_ignore com coberturas sintéticas.
    mask = np.zeros((1, 6), dtype=np.uint8)
    mask[0] = [gs.INK, gs.INK, gs.PAPER, gs.PAPER, gs.HAND, gs.BACKGROUND]
    ink_cov = np.array([[0.5, 0.95, 0.5, 0.2, 0.0, 0.0]], dtype=np.float32)
    paper_cov = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.5]], dtype=np.float32)
    hand_cov = np.array([[0.0, 0.0, 0.0, 0.0, 0.5, 0.0]], dtype=np.float32)
    out, causes = gs._ambiguity_ignore(mask, paper_cov, ink_cov, hand_cov)
    # tinta em empate (0,5) -> ignore; tinta dominante (0,95) -> fica.
    assert out[0, 0] == 255 and causes["ink_edge"] == 1
    assert out[0, 1] == gs.INK
    # papel com tinta em empate por cima -> ignore; tinta fraca (0,2) -> fica.
    assert out[0, 2] == 255 and causes["ink_on_paper"] == 1
    assert out[0, 3] == gs.PAPER
    # mão em empate -> ignore.
    assert out[0, 4] == 255 and causes["hand_edge"] == 1
    # fundo com papel em empate por baixo -> ignore.
    assert out[0, 5] == 255 and causes["paper_edge_out"] == 1


def test_ambiguous_pixels_become_ignore_in_scenes():
    scene = gs.compose_scene(
        31, "train", 0, "normal", hand=_hand(), hand_placement="over_stroke"
    )
    mask = scene["mask"]
    assert (mask == 255).sum() > 0, "nenhum pixel ambíguo marcado"
    assert scene["info"]["ignore_fraction"] > 0
    assert sum(scene["info"]["ignore_causes"].values()) >= (mask == 255).sum()
    # Ignore é exceção, não regra: menos de 5% da cena.
    assert scene["info"]["ignore_fraction"] < 0.05


def test_no_layers_no_ignore():
    # Cena vazia: sem papel, sem mão, sem tinta — nada é ambíguo.
    scene = gs.compose_scene(3, "train", 0, "empty_table")
    assert (scene["mask"] == 255).sum() == 0


def test_visible_majority_stays_supervised():
    # O traço visível continua tinta e a mão visível continua mão:
    # o ignore recorta a borda, não o miolo das classes.
    scene = gs.compose_scene(
        21, "train", 2, "normal", hand=_hand(), hand_placement="over_stroke"
    )
    mask, ink_full = scene["mask"], scene["ink_full"]
    occluded = (ink_full > 0) & (mask == gs.HAND)
    visible_ink = (ink_full > 0) & ~occluded
    kept = (mask == gs.INK) & visible_ink
    assert visible_ink.sum() > 0
    assert kept.sum() / visible_ink.sum() >= 0.75
    assert (mask == gs.HAND).sum() > 0


def test_generated_masks_fit_train_contract(tmp_path):
    # O contrato de treino (training/train.py) já aceita 255: valida e
    # não conta ignore como classe.
    from training import train as T

    gs.generate_dataset(tmp_path, seed=7, counts={"train": 4, "val": 1, "test": 1})
    pairs = T.pair_images_masks(tmp_path / "train")
    assert pairs
    for _, mask_path, _ in pairs:
        mask = T.load_mask(mask_path)  # levanta se houver valor inválido
        assert set(np.unique(mask).tolist()) <= {0, 1, 2, 3, 255}
    counts = T.pixel_counts(pairs)
    assert counts.sum() > 0


def test_combined_loss_ignores_255():
    torch = pytest.importorskip("torch")
    from training import train as T

    rng = np.random.default_rng(0)
    logits = torch.tensor(
        rng.normal(size=(1, 4, 8, 8)), dtype=torch.float32, requires_grad=True
    )
    target = torch.full((1, 8, 8), 255, dtype=torch.long)
    target[0, :4, :4] = 1
    target[0, 4:, 4:] = 3
    weights = torch.tensor([1.0, 1.0, 2.0, 4.0])
    loss = T.combined_loss(logits, target, weights)
    assert torch.isfinite(loss)
    loss.backward()
    # Pixel 100% ignorado: loss ligada ao grafo, gradiente zero.
    only_ignore = torch.full((1, 8, 8), 255, dtype=torch.long)
    loss2 = T.combined_loss(logits, only_ignore, weights)
    assert float(loss2) == 0.0


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


def test_load_hands_missing_mask_fails_loudly(tmp_path):
    cv2.imwrite(str(tmp_path / "00000000.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="sem par de máscara"):
        gs.load_hands(tmp_path)


def test_load_hands_unreadable_file_fails_loudly(tmp_path):
    (tmp_path / "00000000.png").write_bytes(b"not an image")
    (tmp_path / "00000000_mask.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="ilegível"):
        gs.load_hands(tmp_path)


def test_load_hands_missing_dir_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="não existe"):
        gs.load_hands(tmp_path / "hands")


def test_load_hands_uses_documented_jpeg_threshold(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "mask").mkdir()
    cv2.imwrite(
        str(tmp_path / "rgb" / "00000000.jpg"), np.zeros((2, 6, 3), dtype=np.uint8)
    )
    values = np.array([[0, 1, 7, 126, 127, 128]], dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "mask" / "00000000.png"), np.repeat(values, 2, axis=0))

    hand = gs.load_hands(tmp_path)[0]

    assert hand.mask[0].tolist() == [0, 0, 0, 0, 0, 255]
    assert hand.base_id == "00000000"


def test_freihand_variants_stay_in_the_same_split():
    hands = []
    for base in range(3):
        for variant in range(4):
            source_id = f"{base + variant * gs.FREIHAND_BASE_HAND_COUNT:08d}"
            hands.append(
                gs.HandSource(
                    source_id,
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    np.zeros((2, 2), dtype=np.uint8),
                    base_id=gs.freihand_base_id(source_id),
                )
            )

    variant_splits, base_splits = gs.split_hand_sources(hands, seed=42)
    for base in range(3):
        destinations = {
            split for split, ids in variant_splits.items() if f"{base:08d}" in ids
        }
        assert len(destinations) == 1
        base_id = f"{base:08d}"
        for variant in range(1, 4):
            source_id = f"{base + variant * gs.FREIHAND_BASE_HAND_COUNT:08d}"
            assert source_id in variant_splits[next(iter(destinations))]
        assert base_id in base_splits[next(iter(destinations))]


def test_clipped_sheet_placement_exits_frame():
    """clipped_sheet: o quad sai do frame por exatamente uma borda,
    5-40% da caixa fora — o regime ausente entre "inteira" e a lasca
    de sheet_out (gap medido do detector, 2026-11-12)."""
    from tools.gen_synthetic import _sample_clipped_quad

    rng = np.random.default_rng(0)
    edges_seen = set()
    for _ in range(60):
        quad, edge, frac = _sample_clipped_quad(rng, 500, 700)
        assert edge in (0, 1, 2, 3)
        assert 0.05 <= frac <= 0.40
        edges_seen.add(edge)
        lo, hi = quad.min(axis=0), quad.max(axis=0)
        # sai pela borda sorteada (bordas adjacentes podem clipar
        # junto — canto de folha fora do quadro é caso real)
        if edge == 0:
            assert lo[0] < 0
        elif edge == 1:
            assert hi[0] > gs.SIZE
        elif edge == 2:
            assert lo[1] < 0
        else:
            assert hi[1] > gs.SIZE
        # e continua majoritariamente visível (não é lasca)
        inside = np.minimum(hi, gs.SIZE) - np.maximum(lo, 0)
        visible = np.clip(inside / np.maximum(hi - lo, 1), 0, 1).prod()
        assert visible > 0.35
    assert len(edges_seen) == 4


def test_clipped_and_rotated_scene_types():
    """Os dois scene_types novos compõem cena normalmente: classes
    válidas, folha presente e sheet_clip registrado no info."""
    clipped = gs.compose_scene(5, "train", 0, "clipped_sheet", hand=_hand())
    assert set(np.unique(clipped["mask"]).tolist()) <= {0, 1, 2, 3, 255}
    assert clipped["info"]["sheet_clip"] is not None
    rotated = gs.compose_scene(6, "train", 0, "rotated_sheet", hand=_hand())
    assert set(np.unique(rotated["mask"]).tolist()) <= {0, 1, 2, 3, 255}
    # determinismo dos tipos novos
    b = gs.compose_scene(6, "train", 0, "rotated_sheet", hand=_hand())
    assert np.array_equal(rotated["rgb"], b["rgb"])


def test_generate_dataset_clip_rotation_fractions(tmp_path):
    """Frações ligadas produzem os scene_types novos no manifesto e
    registram a dieta; desligadas, o comportamento é o histórico."""
    manifest = gs.generate_dataset(
        tmp_path / "ds",
        seed=1,
        counts={"train": 40, "val": 4, "test": 4},
        clip_fraction=0.5,
        rotation_fraction=0.5,
    )
    kinds = [s["scene_type"] for s in manifest["scenes"]]
    assert "clipped_sheet" in kinds
    assert "rotated_sheet" in kinds
    assert manifest["diet"]["clip_fraction"] == 0.5
    assert manifest["diet"]["rotation_fraction"] == 0.5
    assert manifest["diet"]["rotation_max_deg"] == 40.0
    clips = [
        s["sheet_clip"]
        for s in manifest["scenes"]
        if s["scene_type"] == "clipped_sheet"
    ]
    assert all(c["edges"] and 0 < c["clip_fraction"] for c in clips)


def test_generate_dataset_fractions_validation(tmp_path):
    with pytest.raises(ValueError):
        gs.generate_dataset(tmp_path / "a", clip_fraction=1.5)
    with pytest.raises(ValueError):
        gs.generate_dataset(tmp_path / "b", rotation_fraction=-0.1)
    with pytest.raises(ValueError):
        gs.generate_dataset(tmp_path / "c", rotation_max=120)


def test_rotated_sheet_beyond_legacy_ceiling():
    """rotated_sheet: amostra ALÉM do ±18° histórico — toda a faixa
    é (18, rotation_max] (bancada lift & return)."""
    from tools.gen_synthetic import _sample_wide_rotation_quad

    rng = np.random.default_rng(1)
    for _ in range(80):
        quad = _sample_wide_rotation_quad(rng, 500, 700, 40.0)
        top = quad[1] - quad[0]
        deg = abs(np.degrees(np.arctan2(top[1], top[0])))
        assert deg > 18, f"{deg:.1f}° dentro do envelope legado"
        assert deg <= 40.0 + 1e-6


def test_rotated_sheet_rotation_max_below_ceiling():
    """rotation_max <= 18 não tem faixa "além" — cai no uniforme
    cheio (documentado; validado em (0, 90])."""
    from tools.gen_synthetic import _sample_wide_rotation_quad

    rng = np.random.default_rng(2)
    for _ in range(40):
        quad = _sample_wide_rotation_quad(rng, 500, 700, 15.0)
        top = quad[1] - quad[0]
        deg = abs(np.degrees(np.arctan2(top[1], top[0])))
        assert deg <= 15.0 + 1e-6


def test_clipped_quad_composes_scale_and_wide_rotation():
    """Clipe + rotação ampla são ortogonais: o quad clipado sai pela
    borda sorteada E vem na faixa além de ±18°, e o amostrador de
    escala decide o tamanho (loguniform produz folhas pequenas)."""
    from tools.gen_synthetic import _sample_clipped_quad

    # rotação ampla na base do clipe
    rng = np.random.default_rng(3)
    for _ in range(60):
        quad, edge, frac = _sample_clipped_quad(rng, 500, 700, "legacy", 18.0, 40.0)
        # o ângulo original pode ser medido na aresta "topo" do quad —
        # após o deslocamento o ângulo se preserva (translação pura)
        top = quad[1] - quad[0]
        deg = abs(np.degrees(np.arctan2(top[1], top[0])))
        assert deg > 18 or deg > 162, f"{deg:.1f}° fora da faixa ampla"

    # escala loguniforme compõe com o clipe
    rng = np.random.default_rng(4)
    small_seen = False
    for _ in range(80):
        quad, edge, frac = _sample_clipped_quad(rng, 500, 700, "loguniform")
        lo, hi = quad.min(axis=0), quad.max(axis=0)
        if (hi[0] - lo[0]) * (hi[1] - lo[1]) < 0.30 * gs.SIZE * gs.SIZE:
            small_seen = True
            break
    assert small_seen, "loguniform nunca produziu folha pequena clipada"


def test_clip_and_rotation_independent_rolls(tmp_path):
    """clip_fraction=1 e rotation_fraction=1: o sorteio de rotação
    não encolhe atrás do clipe — o clipe já sai na faixa ampla."""
    from tools.gen_synthetic import _sample_clipped_quad

    # composição direta: os dois hits viram clipped_wide
    rng = np.random.default_rng(5)
    wide_seen = False
    for _ in range(40):
        quad, edge, frac = _sample_clipped_quad(rng, 500, 700, "legacy", 18.0, 40.0)
        top = quad[1] - quad[0]
        deg = abs(np.degrees(np.arctan2(top[1], top[0])))
        if deg > 25:
            wide_seen = True
    assert wide_seen
