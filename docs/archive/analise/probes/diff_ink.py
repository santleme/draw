"""Diferencial do InkAccumulator: arvore antiga (23d3a36) x nova."""
import importlib.util, sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, "/home/user/draw")
from draw import ink as new

spec = importlib.util.spec_from_file_location(
    "old_ink", Path("/home/user/.draw-analysis-wt/draw/ink.py"))
old = importlib.util.module_from_spec(spec)
sys.modules["old_ink"] = old
# o modulo antigo usa imports relativos (from . import config)
old.__package__ = "old_pkg"
spec.loader.exec_module(old) if False else None

# import relativo exige pacote: carrega a arvore antiga inteira
sys.path.insert(0, "/home/user/.draw-analysis-wt")
for mod in [m for m in list(sys.modules) if m.startswith("draw")]:
    del sys.modules[mod]
from draw import ink as old  # noqa: E402
sys.path.pop(0)
for mod in [m for m in list(sys.modules) if m.startswith("draw")]:
    del sys.modules[mod]
sys.path.insert(0, "/home/user/draw")
from draw import ink as new  # noqa: E402

H, W = 1200, 850
rng = np.random.default_rng(11)
a_old, a_new = old.InkAccumulator(), new.InkAccumulator()
observed = np.zeros((H, W), np.uint8)
visible = np.ones((H, W), bool)
for i in range(60):
    if i < 30:
        observed[100:112, 100:300] = 255
    if i >= 30:
        observed[100:112, 180:240] = 0
    ys, xs = rng.integers(0, H, size=40), rng.integers(0, W, size=40)
    observed[ys, xs] = 255
    visible[:] = True
    if 20 <= i < 30:
        visible[90:130, 130:170] = False
    c_old = a_old.update(observed.copy(), visible.copy(), t_capture_ms=1000 + 33 * i)
    c_new = a_new.update(observed.copy(), visible.copy(), t_capture_ms=1000 + 33 * i)
    assert c_old.dtype == c_new.dtype == np.uint8
    assert np.array_equal(c_old, c_new), f"canvas divergiu no frame {i}"
    assert np.array_equal(a_old.state, a_new.state)
    assert np.array_equal(a_old.first_seen_ms(), a_new.first_seen_ms())
    assert np.array_equal(a_old.last_update_ms(), a_new.last_update_ms())
print("diferencial: 60 frames, canvas + estado + carimbos identicos")
print("tinta final:", int(a_new.state.sum()), "px")
