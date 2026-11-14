"""Resumo tabular das avaliações QuickDraw em captures/quickdraw-sessions/."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "captures" / "quickdraw-sessions"
for ev in sorted(root.glob("*.eval.json")):
    r = json.loads(ev.read_text())
    p = r["pipeline"]
    s = ev.name[: -len(".eval.json")]
    print(
        f"{s:12s} iou={r['iou']:.3f} prec={r['precision']:.3f} "
        f"rec={r['recall']:.3f} fp={r['fp']:6d} fn={r['fn']:5d} "
        f"strokes={r['stroke_counts']} det={p['detections']} "
        f"passes={p['ink_passes']} status={p['status']}"
    )
    for st in r["strokes"]:
        if st["status"] != "captured":
            print("   MISS", st)
