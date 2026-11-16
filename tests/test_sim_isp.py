"""Contratos do gêmeo óptico/físico: ISP, penumbra, especular e curvatura."""

import math

import cv2
import numpy as np
import pytest

from tools import gen_synthetic as gs
from tools import sim_isp
from tools.sim_source import SyntheticCapture


def _blank_scene(v=128):
    return np.full((gs.SIZE, gs.SIZE, 3), v, np.uint8)


def test_ae_reacts_to_scene_luminance_with_lag():
    rng = np.random.default_rng(0)
    isp = sim_isp.IspTwin(rng, 10, ae=True, awb=False)
    dark = _blank_scene(60)
    gains = []
    for _ in range(12):
        isp.process(dark)
        gains.append(isp.gain)
    # Cena escura: a exposição sobe gradualmente, não em um frame.
    assert gains[0] < gains[-1] and gains[-1] > 1.0
    assert gains[1] - gains[0] < gains[-1] - gains[0]
    bright = _blank_scene(230)
    for _ in range(12):
        isp.process(bright)
    assert isp.gain < 1.0


def test_awb_pushes_channels_toward_gray_world():
    rng = np.random.default_rng(1)
    isp = sim_isp.IspTwin(rng, 10, ae=False, awb=True)
    warm = _blank_scene(120)
    warm[..., 2] = 200  # vermelho dominante
    for _ in range(40):
        out = isp.process(warm)
    assert isp.wb[2] < 1.0 < isp.wb[0]
    means = out.reshape(-1, 3).mean(axis=0)
    assert abs(means[2] - means[0]) < 60


def test_jpeg_round_trip_and_invalid_params():
    rng = np.random.default_rng(2)
    isp = sim_isp.IspTwin(rng, 10, ae=False, awb=False, jpeg_quality=40)
    out = isp.process(_blank_scene())
    assert out.shape == (gs.SIZE, gs.SIZE, 3) and out.dtype == np.uint8
    with pytest.raises(ValueError):
        sim_isp.IspTwin(rng, 10, jpeg_quality=5)


def test_motion_kernel_scales_with_speed():
    assert sim_isp.motion_kernel(0.5, 0.5) is None
    slow = sim_isp.motion_kernel(3, 0)
    fast = sim_isp.motion_kernel(12, 0)
    assert slow.shape[0] < fast.shape[0] <= 31
    assert fast.sum() == pytest.approx(1.0)
    img = np.zeros((40, 40), np.float32)
    img[20, 10] = 1.0
    blurred = sim_isp.blur_along(img, 8, 0)
    assert blurred.sum() == pytest.approx(1.0, abs=0.05)
    assert (blurred > 0).sum() > (img > 0).sum()


def test_cast_shadow_darkens_offset_region():
    rgb = _blank_scene(200).astype(np.uint8)
    cov = np.zeros((100, 100), np.float32)
    cv2.circle(cov, (50, 50), 40, 1.0, -1)
    before = rgb.copy()
    sim_isp.cast_shadow(rgb, cov, 256, 256, (20.0, 0.0), 8.0, 0.4)
    # Sprite cobre 216–296; sombra deslocada +20 cobre ~236–316 com
    # cauda de penumbra até ~340. Umbra escurece forte, penumbra leve,
    # longe da sombra nada muda.
    umbra = int(rgb[256, 300, 0])
    penumbra = int(rgb[256, 340, 0])
    far = int(rgb[256, 100, 0])
    assert umbra < penumbra <= int(before[256, 256, 0])
    assert far == int(before[256, 100, 0])


def test_ink_sheen_lightens_ink_only_in_band():
    rgb_w = _blank_scene(240)
    ink = np.zeros((gs.SIZE, gs.SIZE), np.uint8)
    cv2.line(ink, (50, 256), (462, 256), 255, 4)
    rgb_w[ink > 0] = (20, 20, 20)
    sim_isp.ink_sheen(rgb_w, ink, (256, 256), (1.0, 0.0), 60.0, 0.9)
    center = rgb_w[256, 256].astype(int)
    edge = rgb_w[256, 60].astype(int)
    assert center.mean() > 150  # dentro da faixa: cinza claro
    assert edge.mean() < 60  # fora da faixa: tinta intacta


def test_scene_to_page_map_without_curl_matches_homography():
    rng = np.random.default_rng(3)
    quad = gs._sample_normal_quad(rng, 360, 510)
    src = np.float32([[0, 0], [360, 0], [360, 510], [0, 510]])
    h_mat = cv2.getPerspectiveTransform(src, quad)
    mx, my = sim_isp.scene_to_page_map(np.linalg.inv(h_mat), 360, 510, None)
    paper = np.random.default_rng(4).integers(0, 255, (510, 360, 3), dtype=np.uint8)
    ink = np.zeros((510, 360), np.uint8)
    ink[100:200, 100:200] = 255
    mapped = sim_isp.warp_paper_mapped(paper, ink, mx, my)
    direct = gs._warp_paper(paper, ink, quad)
    for a, b in zip(mapped, direct):
        assert np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) < 0.5


def test_curl_foreshortens_and_marks_beyond_edge_invalid():
    rng = np.random.default_rng(5)
    quad = gs._sample_normal_quad(rng, 360, 510)
    src = np.float32([[0, 0], [360, 0], [360, 510], [0, 510]])
    h_mat = cv2.getPerspectiveTransform(src, quad)
    curl = {
        "axis": (1.0, 0.0),
        "bend_start": 200.0,
        "edge": 360.0,
        "theta_max": math.radians(60),
        "intensity": 0.85,
    }
    mx, my = sim_isp.scene_to_page_map(np.linalg.inv(h_mat), 360, 510, curl)
    # A dobra encolhe a folha na direção do eixo e deixa de cobrir
    # o que a pose plana cobriria — pixels além da borda projetada
    # ficam fora da página (mapa -1).
    invalid = (mx < 0) | (my < 0)
    assert 0 < float(invalid.mean()) < 0.95
    shade = sim_isp.curl_shade(360, 510, curl)
    assert shade.shape == (510, 360)
    assert shade[255, 50] == pytest.approx(1.0)
    assert shade[255, 350] < 0.8


def test_capture_optics_flags_validate_and_stay_deterministic():
    with pytest.raises(ValueError):
        SyntheticCapture("a", jpeg_quality=3)
    with pytest.raises(ValueError):
        SyntheticCapture("a", ink_sheen=1.5)
    with pytest.raises(ValueError):
        SyntheticCapture("a", page_curl=-0.1)
    with pytest.raises(ValueError):
        SyntheticCapture("a", background="mesa_inexistente")
    kwargs = dict(
        fps=4,
        stroke_seconds=0.3,
        warmup_seconds=0.5,
        settle_seconds=0.5,
        auto_exposure=True,
        auto_white_balance=True,
        motion_blur=True,
        jpeg_quality=70,
        hand_shadow=True,
        ink_sheen=0.5,
        page_curl=0.4,
        hand=True,
    )
    a, b = SyntheticCapture("ab", **kwargs), SyntheticCapture("ab", **kwargs)
    frames = 0
    while True:
        oka, fa = a.read()
        okb, fb = b.read()
        assert oka == okb
        if not oka:
            break
        assert np.array_equal(fa, fb)
        frames += 1
    assert frames == a.frame_count


def test_curl_bundle_exports_dense_maps(tmp_path):
    cap = SyntheticCapture(
        "a", fps=4, warmup_seconds=0.25, settle_seconds=0.25, page_curl=0.5
    )
    while cap.read()[0]:
        pass
    path = tmp_path / "curl.gt.npz"
    cap.save_ground_truth(path)
    cap.release()
    with np.load(path, allow_pickle=False) as bundle:
        assert "scene_to_page_x" in bundle and "scene_to_page_final_x" in bundle
        assert bundle["scene_to_page_x"].shape == (gs.SIZE, gs.SIZE)
        # ink_full da cena bate com o mapa denso — contrato da avaliação.
        remapped = cv2.remap(
            bundle["page_ink_full"],
            bundle["scene_to_page_x"],
            bundle["scene_to_page_y"],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        assert np.array_equal(remapped, bundle["ink_full"])
