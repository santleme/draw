import sys
from pathlib import Path

# Rodam da raiz do repositório: este arquivo mora em
# docs/analise/probes/, então a raiz é parents[3].
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))
"""Sondas do caminho padrao de tinta (page-clean): borracha e canvas velho."""
import sys, time, threading
import numpy as np
import cv2
from draw import app as app_module, config, paper
import selftest

H, W = config.PAPER_H, config.PAPER_W

print("=" * 70)
print("P11. A uniao simples do _PageCleanWorker apaga alguma coisa?")

class _FakePageClean:
    model_path = "fake.onnx"; threshold = 0.7
    def __init__(self, masks):
        self.masks = list(masks); self.i = -1
    def load(self): return True
    def ink_mask(self, warped):
        self.i += 1
        return np.asarray(self.masks[min(self.i, len(self.masks) - 1)])

ink_then_erase = []
m = np.zeros((H, W), np.uint8); m[200:300, 200:400] = 255
ink_then_erase.append(m)          # frame 1: tinta
ink_then_erase.append(np.zeros((H, W), np.uint8))  # frame 2: borracha
fake = _FakePageClean(ink_then_erase)
worker = app_module._PageCleanWorker(fake).start()
gen = worker.new_generation()
warp = np.zeros((H, W, 3), np.uint8)
worker.submit(warp, gen); time.sleep(0.4)
got1 = worker.poll(gen, 0)
print("   apos 1a passada: pixels de tinta publicados =", int((got1[1] > 0).sum()) if got1 else None)
worker.submit(warp, gen); time.sleep(0.4)
got2 = worker.poll(gen, got1[0] if got1 else 0)
print("   apos 2a passada (mascara vazia): pixels de tinta =", int((got2[1] > 0).sum()) if got2 else None)
print("   -> a borracha NAO remove tinta: a mascara de 20.000 px fica para sempre")
worker.stop()

print("=" * 70)
print("P12. Canvas da folha anterior vaza para a folha nova (run())")

scene, _gt, _dst = selftest.build_markerless_scene()
sheet_frame = cv2.resize(scene, (1280, 720), interpolation=cv2.INTER_AREA)
blank_frame = np.full((720, 1280, 3), 40, np.uint8)

class _SeqCapture:
    """Folha por 60 quadros, mesa vazia por 90, folha de novo por 60."""
    def __init__(self):
        self.i = 0; self.n = 210; self.released = False
    def read(self):
        i = self.i
        if i >= self.n:
            return False, None
        self.i += 1
        time.sleep(0.005)
        if 60 <= i < 150:
            return True, blank_frame.copy()
        return True, sheet_frame.copy()
    def release(self):
        self.released = True

class _BlobOnce:
    """Tinta so na primeira folha; depois da troca, mascara vazia."""
    model_path = "fake.onnx"; threshold = 0.7
    def __init__(self): self.calls = 0
    def load(self): return True
    def ink_mask(self, warped):
        self.calls += 1
        m = np.zeros((H, W), np.uint8)
        if self.calls <= 4:
            m[300:500, 300:500] = 255
        return m

shown = []
class _Spy:
    def imshow(self, _t, frame): shown.append((app_module._fit(frame, 640).copy(), time.monotonic()))
    def waitKey(self, _d): return -1
    def destroyAllWindows(self): pass

real_open = app_module._open_capture
real_pc = app_module.page_clean_module.PageCleanInk
real_imshow, real_waitkey, real_destroy = cv2.imshow, cv2.waitKey, cv2.destroyAllWindows
spy = _Spy()
app_module._open_capture = lambda _s: _SeqCapture()
app_module.page_clean_module.PageCleanInk = lambda _p: _BlobOnce()
cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = spy.imshow, spy.waitKey, spy.destroyAllWindows
try:
    app_module.run(source=0, save_dir="/tmp/draw-probe-captures")
except Exception as e:
    print("   run() levantou:", type(e).__name__, e)
finally:
    app_module._open_capture = real_open
    app_module.page_clean_module.PageCleanInk = real_pc
    cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = real_imshow, real_waitkey, real_destroy

print(f"   quadros exibidos: {len(shown)}")
# painel direito = canvas; esquerdo = camera
def right_ink(view):
    h, w = view.shape[:2]
    panel = view[:, int(w * 0.72):]
    return int(np.count_nonzero(np.all(panel < 60, axis=2)))
def left_bright(view):
    h, w = view.shape[:2]
    cam = view[:, :int(w * 0.69)]
    return float(cam.mean())

series = [(i, left_bright(v), right_ink(v)) for i, (v, _t) in enumerate(shown)]
sheet_gap = [i for i, b, _ in series if b < 60]
back = [i for i, b, _ in series if b >= 60 and sheet_gap and i > sheet_gap[-1]]
print(f"   quadros com mesa vazia: {len(sheet_gap)} (ultimo {sheet_gap[-1] if sheet_gap else None})")
print(f"   primeiro quadro com folha de volta: {back[0] if back else None}")
if back:
    first_back = back[0]
    after = [(i, ink) for i, b, ink in series[first_back:] if ink > 0]
    print(f"   quadros DEPOIS da folha voltar com tinta no painel: {len(after)} (primeiros: {after[:5]})")
    print("   -> tinta da folha ANTERIOR reaparece na folha nova" if after else
          "   -> nenhum vazamento observado nesta amostra")
