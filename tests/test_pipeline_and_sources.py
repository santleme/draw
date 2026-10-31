"""Testes de DrawPipeline, _SheetTracker, _detect_sheet_combined e fontes .drawcap/TCP."""

import socket
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draw import app, config, hand, paper, pipeline, protocol  # noqa: E402
from tests import selftest  # noqa: E402


class _StubHandMasker:
    """Simula HandMasker.detect devolvendo máscara vazia + 21 marcos no frame."""

    def __init__(self, tip_cam_xy, off_sheet_tip_xy=None):
        self._tip_cam_xy = tip_cam_xy
        self._off_sheet_xy = off_sheet_tip_xy

    def detect(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        hands = []
        if self._off_sheet_xy is not None:
            pts_off = np.zeros((21, 2), dtype=np.float32)
            pts_off[hand.INDEX_FINGER_TIP] = self._off_sheet_xy
            hands.append(pts_off)
        if self._tip_cam_xy is not None:
            pts = np.zeros((21, 2), dtype=np.float32)
            pts[hand.INDEX_FINGER_TIP] = self._tip_cam_xy
            hands.append(pts)
        return mask, hands


class _StubNeuralDetector:
    def __init__(self, corners, verdict, logit):
        self._corners = corners
        self._verdict = verdict
        self.last_presence_logit = logit

    def detect_sheet_with_confidence(self, _frame):
        return self._corners, self._verdict


def test_sheet_tracker_state_machine():
    tr = pipeline._SheetTracker()
    q1 = np.array(
        [[200.0, 100.0], [900.0, 100.0], [900.0, 600.0], [200.0, 600.0]],
        dtype=np.float32,
    )
    assert tr.should_detect(0)
    assert tr.update(q1, "paper", True, 10.0) == "seen"
    assert not tr.should_detect(1)
    assert tr.should_detect(5)

    # Detecção próxima mistura suavemente sem salto.
    q_near = q1 + 10.0
    assert tr.update(q_near, "paper", True, 10.16) == "seen"
    assert float(np.abs(tr.corners - (0.7 * q1 + 0.3 * q_near)).max()) < 1e-4

    # Salto > 40 px exige persistência temporal de >= 0,3 s.
    q_jump = q1 + 120.0
    assert tr.update(q_jump, "paper", True, 11.00) is None
    assert float(np.abs(tr.corners - q_jump).max()) > 100.0
    assert tr.update(q_jump, "paper", True, 11.15) is None
    assert tr.update(q_jump, "paper", True, 11.35) == "seen"
    assert float(np.abs(tr.corners - q_jump).max()) < 1.0

    # Presença sem posição renova o relógio e segura os cantos.
    assert tr.update(None, "paper", True, 12.00) is None
    assert tr.corners is not None

    # Ausência por > 10 quadros E > 0,5 s derruba a folha.
    dropped = None
    for k in range(12):
        ev = tr.update(None, None, True, 12.10 + k * 0.06)
        if ev == "drop":
            dropped = k
            break
    assert dropped is not None and tr.corners is None


def test_detect_sheet_combined_veto_and_presence_channel():
    scene_pos, _, _ = selftest.build_markerless_scene()
    scene_pos = cv2.resize(scene_pos, (1280, 720))

    # 1. Veto forte (logit <= -3.0) bloqueia o detector clássico mesmo numa cena positiva.
    veto_det = _StubNeuralDetector(None, None, logit=-8.5)
    c, v = pipeline._detect_sheet_combined(scene_pos, veto_det)
    assert c is None and v is None

    # 2. Logit negativo fraco (-0.4 > -3.0) permite que a reserva clássica encontre a folha.
    weak_det = _StubNeuralDetector(None, None, logit=-0.4)
    c2, v2 = pipeline._detect_sheet_combined(scene_pos, weak_det)
    assert c2 is not None and v2 == "paper"

    # 3. Cena vazia onde clássico retorna None, mas neural viu presença positiva (+4.2):
    blank_dark = np.full((720, 1280, 3), 35, dtype=np.uint8)
    pres_det = _StubNeuralDetector(None, None, logit=4.2)
    c3, v3 = pipeline._detect_sheet_combined(blank_dark, pres_det)
    assert c3 is None and v3 == "paper"


def test_pipeline_uniform_cadence_cursor_fusion_and_stamp_export():
    scene_base, _, true_corners = selftest.build_markerless_scene()
    scale_x = 1280.0 / float(scene_base.shape[1])
    scale_y = 720.0 / float(scene_base.shape[0])
    frame_base = cv2.resize(scene_base, (1280, 720))
    scaled_corners = (
        true_corners * np.array([scale_x, scale_y], dtype=np.float32)
    ).astype(np.float32)

    # Desenha um traço novo sobre a própria folha retificada de frame_base
    # e reprojeta na cena 720p:
    sheet_drawn = paper.warp_sheet(frame_base, scaled_corners)
    cv2.line(sheet_drawn, (260, 760), (540, 760), (20, 20, 20), 6)

    src_rect = np.array(
        [
            [0.0, 0.0],
            [config.PAPER_W - 1.0, 0.0],
            [config.PAPER_W - 1.0, config.PAPER_H - 1.0],
            [0.0, config.PAPER_H - 1.0],
        ],
        dtype=np.float32,
    )
    h_p2c = cv2.getPerspectiveTransform(src_rect, scaled_corners)
    warped_back = cv2.warpPerspective(sheet_drawn, h_p2c, (1280, 720))
    erode_k = np.ones((7, 7), np.uint8)
    mask_poly = np.zeros((720, 1280), dtype=np.uint8)
    cv2.fillConvexPoly(mask_poly, scaled_corners.astype(np.int32), 255)
    mask_poly = cv2.erode(mask_poly, erode_k)
    frame_drawn = frame_base.copy()
    frame_drawn[mask_poly > 0] = warped_back[mask_poly > 0]

    # Projeta um marco 8 próximo ao traço (380, 760) do papel para a câmera
    # e também uma segunda mão fora da folha (canto superior esquerdo 5, 5).
    tip_cam = cv2.perspectiveTransform(
        np.array([[[380.0, 760.0]]], dtype=np.float32), h_p2c
    )[0, 0]
    hand_stub = _StubHandMasker(tip_cam_xy=tip_cam, off_sheet_tip_xy=(5.0, 5.0))

    detect_calls = 0

    def counting_detect(frm):
        nonlocal detect_calls
        detect_calls += 1
        return paper.detect_sheet_with_confidence(frm)

    pipe = pipeline.DrawPipeline(
        hand_masker=hand_stub,
        session_baseline=True,
        detect_fn=counting_detect,
    )

    # Quadro 0 detecta a folha; armamos a página base (C). Semantica
    # de producao restaurada na reconciliacao (condicao 5 da revisao
    # do PR #9): as 12 amostras da base sao espacadas pela cadencia
    # de deteccao (1 a cada 5 quadros), como no bloco de tinta da
    # main — 12 quadros consecutivos NAO fecham a base; ela fecha na
    # 12a amostra espacada, ~60 quadros (~2 s a 30 fps).
    r0 = pipe.process_frame(frame_base, t_capture_ms=1000, now_s=1.0)
    assert r0.sheet_present and detect_calls == 1
    assert pipe.trigger_baseline()
    for i in range(12):
        ri = pipe.process_frame(
            frame_base,
            t_capture_ms=1033 + i * 16,
            now_s=1.033 + i * 0.016,
        )
        assert ri.sheet_present
    assert not pipe.session_ink.ready, (
        "12 quadros consecutivos nao podem fechar a base: "
        "as amostras sao espacadas de 5 em 5 quadros"
    )
    for i in range(12, 60):
        ri = pipe.process_frame(
            frame_base,
            t_capture_ms=1033 + i * 16,
            now_s=1.033 + i * 0.016,
        )
        assert ri.sheet_present
    assert pipe.session_ink.ready, "a base fecha na 12a amostra espacada"
    # Em 61 quadros estáveis (0..60), a detecção só rodou nos múltiplos de 5: 13 vezes!
    assert detect_calls == 13

    # Agora alimentamos 8 quadros com o traço novo: mesmo entre detecções
    # (quadros não múltiplos de 5), o acumulador consolida o traço e a fusão
    # acusa pen_down=True com posição dentro da região do traço!
    last_res = None
    for j in range(8):
        t_ms = 2000 + j * 33
        last_res = pipe.process_frame(
            frame_drawn, t_capture_ms=t_ms, now_s=t_ms / 1000.0
        )

    assert last_res is not None and last_res.sheet_present
    stroke_band = pipe.ink_accumulator.state[745:775, 250:550]
    assert int(stroke_band.sum()) >= 500, (
        f"traco nao consolidou em 8 quadros 1/1: {int(stroke_band.sum())} px"
    )
    assert last_res.cursor.has_position and not last_res.cursor.position_stale
    assert last_res.cursor.pen_down
    assert 240.0 <= last_res.cursor.x <= 560.0
    assert 700.0 <= last_res.cursor.y <= 820.0

    # Salva artefatos (.png + _stamps.npz) e confere carimbos reais:
    with tempfile.TemporaryDirectory() as tmp:
        png_path, npz_path = pipe.save_artifacts(tmp, ts=4242)
        assert png_path is not None and png_path.is_file()
        assert npz_path is not None and npz_path.is_file()
        data = np.load(npz_path)
        assert int((data["mask"] > 0).sum()) == int(pipe.ink_accumulator.state.sum())
        valid_first = data["first_seen_ms"][data["mask"] > 0]
        valid_last = data["last_update_ms"][data["mask"] > 0]
        assert 2000 <= int(valid_first.min()) <= 2000 + 7 * 33
        assert int(valid_last.max()) == 2000 + 7 * 33


def test_async_detector_worker_and_helper_sources():
    scene_pos, _, _ = selftest.build_markerless_scene()
    frame_720 = cv2.resize(scene_pos, (1280, 720))

    # 1. Modo assíncrono fecha a folha em segundo plano e encerra limpo.
    pipe_async = pipeline.DrawPipeline(async_detect=True)
    try:
        acquired = False
        for k in range(12):
            res = pipe_async.process_frame(
                frame_720, t_capture_ms=5000 + k * 33, now_s=5.0 + k * 0.033
            )
            if res.sheet_present:
                acquired = True
                break
            import time as _t

            _t.sleep(0.02)
        assert acquired, "detector assincrono nao adquiriu a folha"
    finally:
        pipe_async.close()

    # 2. Fonte .drawcap e fonte tcp://host:porta com carimbo de captura real.
    ok_jpg, enc = cv2.imencode(".jpg", frame_720, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok_jpg
    jpeg_bytes = enc.tobytes()

    with tempfile.TemporaryDirectory() as tmp:
        drawcap_path = Path(tmp) / "sessao.drawcap"
        with open(drawcap_path, "wb") as f:
            for idx in range(3):
                f.write(
                    protocol.pack_record(
                        10 + idx, 7_500_000 + idx * 33_333, jpeg_bytes
                    )
                )

        assert app.run_check(str(drawcap_path)) == 0
        cap = protocol.HelperCapture(str(drawcap_path))
        ok, decoded = cap.read()
        cap.release()
        assert ok and decoded.shape == (720, 1280, 3)
        assert cap.last_frame_id == 10
        assert cap.last_capture_ts_ms == 7500

    # Loopback TCP ao vivo em porta efêmera:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def _serve_once():
        conn, _ = srv.accept()
        with conn:
            conn.sendall(protocol.pack_record(99, 9_123_000, jpeg_bytes))
        srv.close()

    t = threading.Thread(target=_serve_once, daemon=True)
    t.start()
    rc = app.run_check(f"tcp://127.0.0.1:{port}")
    t.join(timeout=5.0)
    assert rc == 0


def main():
    test_sheet_tracker_state_machine()
    test_detect_sheet_combined_veto_and_presence_channel()
    test_pipeline_uniform_cadence_cursor_fusion_and_stamp_export()
    test_async_detector_worker_and_helper_sources()
    print("TESTES DE PIPELINE, CADENCIA, FUSAO E FONTES OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
