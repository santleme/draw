import sys
from pathlib import Path

# Rodam da raiz do repositório: este arquivo mora em
# docs/analise/probes/, então a raiz é parents[3].
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))
"""Sondas dos furos suspeitados no nucleo (neural, papel, tinta, tracker)."""

import numpy as np
import cv2

from draw import neural, paper, ink

print("=" * 70)
print("P1. _fit_plausible aceita NaN?")
nan_quad = np.array([[np.nan, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
print("   quad NaN -> _fit_plausible =", neural._fit_plausible(nan_quad, (720, 1280)))
inf_quad = np.array([[np.inf, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
print("   quad inf -> _fit_plausible =", neural._fit_plausible(inf_quad, (720, 1280)))
# faixa fina e longa (arestas opostas iguais, mas area quase nula)
thin = np.array([[0, 0], [1000, 0], [1000, 2], [0, 2]], dtype=np.float32)
print("   faixa 1000x2 px -> _fit_plausible =", neural._fit_plausible(thin, (720, 1280)))

print("=" * 70)
print("P2. fit_quad publico ignora a guarda? (mesma mascara nos dois caminhos)")
mask = np.zeros((720, 1280), dtype=np.uint8)
# mascara que gera um quad plausivel
cv2.fillPoly(mask, [np.array([[300, 200], [1000, 180], [980, 650], [320, 660]])], 255)
q_pub = neural.fit_quad(mask)
print("   fit_quad ->", None if q_pub is None else q_pub.round(1).tolist())
print("   _fit_plausible(fit_quad) ->", None if q_pub is None else neural._fit_plausible(q_pub, mask.shape))

print("=" * 70)
print("P3. contrato ONNX: nome da entrada hardcoded + teste de contrato nao pega")

class _Out:
    def __init__(self, n): self.name = n
class _In:
    def __init__(self, name, shape): self.name = name; self.shape = shape
class _Sess:
    """Sessao falsa que SO aceita a entrada com o nome real do modelo."""
    def __init__(self, input_name, in_shape):
        self._in = _In(input_name, in_shape)
    def get_inputs(self): return [self._in]
    def get_outputs(self): return [_Out("mask_logits"), _Out("presence_logit")]
    def run(self, names, feed):
        if self._in.name not in feed:
            raise RuntimeError(f"feed invalido: esperava {self._in.name!r}, recebi {list(feed)}")
        return [np.zeros((1, 2, 8, 8), np.float32), np.array([3.0], np.float32)]

det = neural.NeuralSheetDetector("unused.onnx")
det._session = _Sess("input", [1, 3, 8, 8])
try:
    det.predict(np.zeros((720, 1280, 3), np.uint8))
    print("   entrada 'input' -> OK")
except Exception as e:
    print("   entrada 'input' -> QUEBRA:", type(e).__name__, e)

# entrada retangular
det2 = neural.NeuralSheetDetector("unused.onnx")
det2._session = _Sess("image", [1, 3, 384, 640])
print("   shape [1,3,384,640] -> _session_input_size =", neural._session_input_size(det2._session), "(ignora 640)")

print("=" * 70)
print("P4. _SheetTracker segura quad congelado para sempre com 'paper' sem quad?")
from draw.app import _SheetTracker
tr = _SheetTracker()
tr.update(np.array([[100, 100], [500, 100], [500, 400], [100, 400]], np.float32), "paper", True, 0.0)
print("   inicial:", tr.corners is not None)
now = 1.0
for i in range(200):
    now += 0.5
    ev = tr.update(None, "paper", True, now)
print(f"   depois de 100 s de 'paper' sem quad: corners != None -> {tr.corners is not None}, veredito {tr.verdict}, evento {ev}")

print("=" * 70)
print("P5. _pending_hits e escrito e lido?")
import inspect, re
src = inspect.getsource(sys.modules["draw.app"])
writes = len(re.findall(r"_pending_hits\s*=", src))
reads = len(re.findall(r"_pending_hits(?!\s*=)", src))
print(f"   escritas: {writes}, leituras: {reads}")

print("=" * 70)
print("P6. InkAccumulator guarda referencia do visible_mask do chamador?")
acc = ink.InkAccumulator()
visible = np.ones((1200, 850), dtype=bool)
observed = np.zeros((1200, 850), np.uint8)
observed[100:200, 100:200] = 255
for i in range(6):
    acc.update(observed, visible, t_capture_ms=1000 * i)
print("   canvas apos 6 frames com visible=True:", int((acc.state).sum()), "px")
visible[:] = False   # chamador recicla o buffer no frame seguinte
for i in range(6, 12):
    acc.update(observed, visible, t_capture_ms=1000 * i)
print("   votos visiveis da janela apos reciclar visible=False:", int(acc._vis_votes[-1].sum()))
print("   -> a deque guarda a MESMA referencia; reciclar o buffer reescreve o passado")

print("=" * 70)
print("P7. warp_sheet com quadrilatero degenerado (3 pontos colineares)")
frame = np.full((720, 1280, 3), 200, np.uint8)
bad = np.array([[100, 100], [300, 100], [500, 100], [700, 100]], np.float32)
try:
    w = paper.warp_sheet(frame, bad)
    print("   warp saiu, shape", w.shape, "media", float(w.mean()), "unique", np.unique(w)[:5])
except Exception as e:
    print("   warp levantou:", type(e).__name__, e)

print("=" * 70)
print("P8. dtype/temporarios do InkAccumulator.update")
acc2 = ink.InkAccumulator()
o = np.zeros((1200, 850), np.uint8); o[10:20, 10:20] = 255
c = acc2.update(o, None, 0)
print("   out dtype", c.dtype, "| state*255 dtype:", (acc2.state * 255).dtype,
      "| bytes por frame:", (acc2.state * 255).nbytes // 1024, "KiB")

print("=" * 70)
print("P9. page_clean: shape da saida do modelo nao e validada")
from draw import page_clean
pc = page_clean.PageCleanInk("unused.onnx")
class _S:
    def get_inputs(self): return [_In("input", [1,3,256,256])]
    def run(self, names, feed):
        return [np.zeros((1, 1, 128, 128), np.float32)]  # metade da resolucao
pc._session = _S(); pc._input_name = "input"
try:
    m = pc.ink_mask(np.zeros((1200, 850, 3), np.uint8))
    print("   saida 128x128 no patch 256 -> mask", m.shape, "silencioso")
except Exception as e:
    print("   ->", type(e).__name__, str(e)[:120])

print("=" * 70)
print("P10. custo relativo: mascara cheia x limiar via logits")
import time
logits = np.random.randn(1200, 850).astype(np.float32) * 3
t0 = time.perf_counter()
for _ in range(20):
    prob = 1.0 / (1.0 + np.exp(-logits))
    m1 = np.where(prob > 0.7, 255, 0).astype(np.uint8)
t1 = time.perf_counter()
thr = float(np.log(0.7 / 0.3))
for _ in range(20):
    m2 = np.where(logits > thr, 255, 0).astype(np.uint8)
t2 = time.perf_counter()
print(f"   sigmoide+where: {(t1-t0)/20*1000:.2f} ms | so logit: {(t2-t1)/20*1000:.2f} ms | iguais: {np.array_equal(m1, m2)}")
