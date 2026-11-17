# Harness audit — 2026-11-02

The harness is the set of rulers, gates, and protocols that decide
whether a model is good. This audit inventories what exists, where
each project rule lives (code, document, or conversation), and what
is missing. Numbers cited were measured in the laboratory by the agent.

## Inventory

| Piece | Where | What it decides | State |
|---|---|---|---|
| Loop promotion gate | `draw/selflearn/gate.py` + CLI `tools/selflearn/judge_gate.py` | PROMOTE/REJECT/INCONCLUSIVE on the sealed real replay; INCONCLUSIVE under the minimum n (5 sessions/25 pages); DIBCO diagnostic only | Code + tests. Solid. |
| Static replay ruler | laboratory (`draw-lab/replay-ruler-2026-11-01/`) | DIBCO IoU/recall per stroke width, FP on empty warps f01–f03, ink fraction on drawn warps | Works, but lives outside the repo, with the model (v2), date, and VM paths written into the code. |
| Sequence/event rulers | same laboratory | Growth on an empty page, session f01–f30, new-stroke delay | Same. |
| TRI rulers (printed/invariance) | inside the TRI training body (laboratory) | Printed leak ≤0.5%; position invariance | Embedded in the body; output lacks the n (pairs, counts) needed for interval and minimum n. |
| Minimal runner | `tools/runner/` | Synthetic metric per epoch detects a dead run | By design (rescoped 2026-10-31): it never promotes or vetoes a live candidate. Consistent with the rules. |
| Launch protocol | laboratory (`draw-lab/launch-protocol/`) | Launch gates (import audit, probe, smoke, log/checkpoint, allocation discipline) | Document + `probe_env.py`, outside the repo. Rule lives only in text. |
| Collection + split + telemetry | `draw/selflearn/` (collection, split by session), `draw/telemetry.py` | Raw material for the sealed holdout and session observability | Code. The sealed holdout at the required n does not exist yet (see P1). |
| Suite hygiene | `tests/` | Test order must not change the result | It had a real leak (below), fixed in this round. |

## What this round implemented

1. **`sys.modules` leak in `tests/test_seg_metrics.py`**
   (commit `2731018`). The fake torch/onnxruntime modules were
   assigned directly into `sys.modules` and never restored; anyone
   importing torch afterwards, in the same process, got the fake
   (`test_selflearn_train_round.py` worked around it by capturing
   the real torch at collection — a fragile patch, dependent on
   collection order). Fix: injection via `monkeypatch.setitem`, fakes
   marked, and a regression test that runs after the
   `validate_onnx` tests and fails if any fake remains. Proof: without
   the fix the new test fails (1 failed/17 passed); with it,
   157 green suites in both environments.
2. **Single judging point, `tools/judge.py`** (commit
   `17b798c`). One command reads the measurements (direct JSON, or
   via the adapters for the replay ruler's `static_results.json` and
   the TRI body's `metrics.json`) and emits the verdict gate by
   gate: value, pre-registered cutoff, n, minimum n, and 95% CI
   (Wilson for proportions with counts; deterministic bootstrap
   for per-page means). Rules in code: headline DIBCO IoU is
   `diagnostic_only` and never decides alone; promotion
   requires the sealed real replay from `gate.py` with n; insufficient
   n on any provided gate refuses the promotion verdict
   (INCONCLUSIVE); PROMOTE comes out labeled as a proposal — the swap
   still requires the live test and the owner's word. The whole
   report carries "measured in the laboratory by the agent".
   Exit codes equal to those of `judge_gate.py` (0/1/2).
   12 new tests.
3. **Real gate CLI exposes `--worst-group-tolerance`** (in this
   commit). `gate.judge_candidate` accepted the tolerance, but the
   CLI had no flag — in practice it was always 0 for anyone
   judging through the official tool. A CLI test covers both
   sides (rejects without, promotes with declared tolerance).

## Prioritized list of what is missing

### P1 — blocks real judging

- **Sealed holdout still lacks the n.** The gate requires
  ≥5 sessions and ≥25 independently labeled pages; the
  current replay has ~48 old files without that seal. Without
  collecting sealed sessions, every real verdict is INCONCLUSIVE —
  correctly. This is not code: it is collection at the desk.
- **The TRI body does not emit the n of the new rulers.** `ruler_print`
  emits printed pages and pixels (enough for the judge), but
  `ruler_invariance` does not emit the number of pairs nor the
  printed counts at the center. Without n, `tools/judge.py` marks both
  invariance gates INCONCLUSIVE by design. Fix in the TRI
  body (laboratory) **before** the round's launch:
  emit `pairs`, `print_center_px`, `print_center_leak_px`, and
  `leak_px`.
- **The replay ruler lives outside the repo and with constants from
  the era.** `replay_ruler.py` pins model v2, its sha, the
  2026-11-01 date, and the VM venv path. Parameterize model/sha/
  date by argument (or a thin runner in the repo that assembles the
  cache per candidate) so that judging a new candidate does not depend on
  editing the laboratory.

### P2 — ruler consistency

- **Drawn warp has no incumbent recall on the static
  ruler.** There it is only ink fraction (observation); the
  recall ≥0.60 per warp gate today is only fed by the
  training body's metrics (R1). Unify: the rulers should emit
  directly the measurement schema of `tools/judge.py`
  (an `--emit-measurements` on each one).
- **Pooled value × macro CI.** On the DIBCO gates, the value is the
  pooled IoU/recall and the 95% CI is the per-page mean bootstrap;
  the judge notes the distinction in the report, but ideally
  the ruler always emits both sides (pooled and per-page).
- **Launch protocol outside the repo.** `CHECKLIST.md` and
  `probe_env.py` live in the laboratory. Copy to the repo (e.g.
  `docs/LAUNCH_PROTOCOL.md` + `tools/probe_env.py`) so the
  protocol travels with the code it protects.

### P3 — cheap tripwires

- **Verdict fixture in CI.** A JSON of measurements from a
  known candidate + the expected verdict, run in CI:
  changing a pre-registered constant of `tools/judge.py` or of
  `gate.py` breaks CI visibly, instead of passing
  unnoticed.
- **Patch-comment in `test_selflearn_train_round.py`.** It
  still describes the `test_seg_metrics.py` leak as a current
  fact and restores torch by hand. With the leak fixed at the
  source, the restoration is redundant (harmless); simplify
  when someone touches the file.
- **Verdict ↔ `tools/runner/candidates.jsonl`.** The judge emits
  full JSON (`--out`); appending that record to the candidate's
  line in `candidates.jsonl` would close the trail
  candidate → verdict with no manual work.
