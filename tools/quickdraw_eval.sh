#!/usr/bin/env bash
# Avalia as sessões QuickDraw em captures/quickdraw-sessions/ pelo
# pipeline real (tools/eval_capture.py) com o conjunto de flags
# validado na bancada t16. Gera *.eval.json + *.canvas.npy por sessão.
# Uso: tools/quickdraw_eval.sh [sessão ...]   (sem args = todas)
set -u
cd "$(dirname "$0")/.."

export DRAW_INK_FOLLOW_MOVE=1 DRAW_SHEET_COVERAGE=1 DRAW_CANVAS_PERSIST=1 \
  DRAW_INK_SKIP_STALE=1 DRAW_INK_REMOVAL=1 DRAW_SEG_HAND_OCCLUSION=1 \
  DRAW_ANCHOR_ARBITRATION=1 DRAW_ANCHOR_FOLLOW=1 DRAW_INK_MOTION_GATE=1 \
  DRAW_ANCHOR_VOTE_RESET=1 DRAW_ANCHOR_FAST_CLEAN=1

# PYTHON sobrescreve (worktrees compartilham o venv do checkout principal).
PY=${PYTHON:-.venv/bin/python}

root=captures/quickdraw-sessions
if [ "$#" -gt 0 ]; then
  sessions="$*"
else
  sessions=$(cd "$root" && ls *.gt.npz | sed 's/\.gt\.npz$//')
fi

for s in $sessions; do
  $PY tools/eval_capture.py \
    --video "$root/$s.mp4" \
    --gt "$root/$s.gt.npz" \
    --out "$root/$s.eval.json" \
    --save-canvas "$root/$s.canvas.npy" \
    > /dev/null 2>&1 \
    && echo "$s ok" || echo "$s FAIL"
done
