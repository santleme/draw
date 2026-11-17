"""Benchmarks relativos dos estagios do loop (nada aqui vira numero absoluto)."""
import sys, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

import numpy as np
import cv2

from draw import ink, paper, neural, config, app
import selftest


def bench(fn, n=30, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


scene, _gt, _dst = selftest.build_markerless_scene()   # 1600x1200
frame = cv2.resize(scene, (1280, 720), interpolation=cv2.INTER_AREA)
warp = paper.warp_sheet(frame, np.array(
    [[300, 80], [1000, 120], [930, 700], [250, 650]], np.float32))

print(f"frame {frame.shape}, warp {warp.shape}")
print("-" * 62)

# --- detector classico -------------------------------------------------
t = bench(lambda: paper.detect_sheet_with_confidence(frame))
print(f"classico detect_sheet_with_confidence : {t:7.2f} ms")

# --- pre/pos do neural, sem session.run --------------------------------
class _Sess:
    def get_inputs(self): return [type("I", (), {"name": "image", "shape": [1, 3, 384, 384]})()]
    def get_outputs(self): return [type("O", (), {"name": n})() for n in ("mask_logits", "presence_logit")]
    def run(self, names, feed):
        x = feed["image"]
        # mascara "papel" sintetica no centro
        m = np.full((1, 2, 384, 384), -4.0, np.float32)
        m[0, 1, 40:340, 60:320] = 4.0
        return [m, np.array([6.0], np.float32)]

det = neural.NeuralSheetDetector("unused.onnx")
det._session = _Sess()
det._resolved_input_size = 384
t_pred = bench(lambda: det.predict(frame))
print(f"neural.predict (sem session.run)     : {t_pred:7.2f} ms")
presence, mask = det.predict(frame)
t_fit = bench(lambda: det._fit(mask, frame), n=20)
print(f"neural._fit (contorno+ajuste)        : {t_fit:7.2f} ms")

# --- componentes do predict -------------------------------------------
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
t = bench(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
print(f"   BGR->RGB 1280x720                : {t:7.2f} ms")
t = bench(lambda: cv2.resize(frame, (384, 384), interpolation=cv2.INTER_AREA))
print(f"   resize 1280x720->384             : {t:7.2f} ms")
small = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_AREA)
t = bench(lambda: cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
print(f"   BGR->RGB no 384 (ordem invertida): {t:7.2f} ms")
t = bench(lambda: float(np.exp(np.float32(1.0))))
lg = np.zeros((1, 2, 384, 384), np.float32)
def _softmax():
    cb, cp = lg[0, 0], lg[0, 1]
    peak = np.maximum(cb, cp)
    return np.exp(cp - peak) / (np.exp(cb - peak) + np.exp(cp - peak))
t = bench(_softmax)
print(f"   softmax 2 canais 384             : {t:7.2f} ms")
t = bench(lambda: cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST))
print(f"   resize mascara 384->1280x720     : {t:7.2f} ms")

# contorno na resolucao do modelo x no frame
mask_small = cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)
t = bench(lambda: neural._largest_contour(mask))
print(f"   _largest_contour no frame 720p   : {t:7.2f} ms")
t = bench(lambda: neural._largest_contour(mask_small))
print(f"   _largest_contour no 384          : {t:7.2f} ms")

print("-" * 62)
# --- tinta --------------------------------------------------------------
t = bench(lambda: ink.extract_ink(warp), n=20)
print(f"extract_ink 850x1200                 : {t:7.2f} ms")
gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
t = bench(lambda: ink._separable_veto(gray), n=20)
print(f"   _separable_veto (2 medianas)      : {t:7.2f} ms")

acc = ink.InkAccumulator()
obs = (ink.extract_ink(warp) > 0).astype(np.uint8) * 255
t = bench(lambda: acc.update(obs, None, t_capture_ms=1000), n=20)
print(f"InkAccumulator.update                : {t:7.2f} ms")

state = acc.state
t = bench(lambda: (state * 255).astype(np.uint8), n=20)
print(f"   state*255 -> uint8 (int64 tmp)    : {t:7.2f} ms")
u8 = np.empty((config.PAPER_H, config.PAPER_W), np.uint8)
def _fast():
    np.multiply(state, 255, out=u8, dtype=np.uint8)
    return u8
t = bench(_fast, n=20)
print(f"   np.multiply(state,255,dtype=uint8): {t:7.2f} ms")

buf = np.empty((config.PAPER_H, config.PAPER_W, 3), np.uint8)
t = bench(lambda: ink.render_canvas(u8, out=buf), n=20)
print(f"render_canvas(out=buffer)            : {t:7.2f} ms")

print("-" * 62)
# --- composicao da vista ------------------------------------------------
canvas = np.full((config.PAPER_H, config.PAPER_W, 3), 255, np.uint8)
t = bench(lambda: app._compose_view(frame, canvas), n=20)
print(f"_compose_view (resize+hstack+texto)  : {t:7.2f} ms")
t = bench(lambda: frame.copy())
print(f"   frame.copy() 1280x720             : {t:7.2f} ms")
t = bench(lambda: cv2.resize(frame, (1600 - 6 - 453, 640), interpolation=cv2.INTER_AREA))
print(f"   resize camera direto p/ 1141x640  : {t:7.2f} ms")
