# Latency audit of the page-clean path (U0, loop v2)

Lucas's question (2026-11-02): the model pass (~96.6 ms on
his Ryzen, lab measurement) does not explain the estimated
2.3–3.5 s update delay. His hypothesis: the temporal consensus
of the accumulator — in his memory, a window of 15
observations and entry at 0.45, at ~6 Hz a stroke would need ~7
hits, more than 1 s just to appear.

## What the code on main does today (verified, not from memory)

1. **The accumulator in the hypothesis does not run on the default path.**
   `draw/ink.py::InkAccumulator` (window=15, enter=0.45,
   exit=0.20, min_visible=4) is only used behind `--legacy-ink`.
2. **The voting accumulator that existed on the default path was
   removed.** `_ConfirmedInkAccumulator` (window 3, settled-page
   lock, 1 s probe, reopening with 200 px) left in
   `8b9ff45` ("revert: remove settle lock, probe gate and vote
   accumulator from the page-clean live path"). Whoever claims that
   probe/reopening/vote explain the current delay is measuring a
   path that main does not have.
3. **The default path is a simple union:** `_PageCleanWorker` runs
   one full pass per submitted warp, accumulates the union of the
   generation's masks and publishes in the same pass in which the ink
   appears. Contract pinned in
   `tests/test_page_clean_live_contract.py`: new ink on the published
   canvas within 3 passes (in practice, 1).
4. **The cadence is 250 ms between pass starts**
   (`_PAGE_CLEAN_INTERVAL_SECONDS = 0.25`).
5. **The loop only submits a warp when sheet detection runs**
   (`tracker.should_detect`: every `_SHEET_DETECT_INTERVAL = 5`
   frames, or with no sheet/lost sheet) — the submit is not per
   frame.

## Measured in this VM with this unit's instrumentation

Rule: `tools/selflearn/measure_latency.py` over the real warps
f01–f08 (`draw-lab/real-negatives-2026-10-31/proc/warps`),
`telemetry_snapshot()` stamps from `_PageCleanWorker`. Numbers
from this shared VM, not from Lucas's desk.

| Pass source | n | queue (median) | pass (median) | submit→publication |
|---|---|---|---|---|
| Instant fake ink | 6 | 248.5 ms | 0.1 ms | 249.4 ms |
| Real model v2 (`page_clean_v2_fixed.onnx`, sha 0e7fecb1…) | 4 | 0.0 ms | 4,700.8 ms | 4,741.7 ms |
| Real installed w8 model (`w8_page_clean_student_w8.onnx`, sha a470bd28…) | 6 | 1.7 ms | 2,877.3 ms | 2,882.4 ms |

Reading:

- With an instant pass, the submit→publication floor is the
  **whole cadence**: submitted just after the previous
  publication, the warp waits out the rest of the 250 ms between starts.
- With the real model in this VM (under load: v2 ~4.7 s/pass and
  w8 ~2.9 s/pass — far above its historical ~0.6–1.1 s at rest,
  and above the Ryzen's 96.6–168 ms), the pass swallows the
  cadence: queue ~0–2 ms in both.
- Projecting to the Ryzen with his numbers (pass
  96.6–168 ms < cadence 250 ms): the worker's floor is
  ~250–350 ms per update. **Neither the cadence nor the pass,
  added together, close 2.3–3.5 s.**

## Where the rest may live (ordered hypotheses, not verdicts)

1. **Upstream of the submit:** latest-frame capture, sheet
   detection every 5 frames, tracker jump persistence
   (0.3 s) and the corner median — the warp is only submitted on
   a detection frame.
2. **Downstream of the publication:** the loop collects with `poll()` once
   per frame and renders; a slow frame = publication waiting
   in the queue.
3. **Perception at the page turn**, which was the concrete
   complaint of 2026-11-01: a new generation only opens with a
   drop/reappearance of the sheet (the coverage event remains
   off by default since the hotfix), and a drop requires 0.5 s + 10 frames without a sheet.

## What this unit delivers and what is missing

- `_PageCleanWorker.telemetry_snapshot()`: counters
  (submits, submits replaced by latest-wins, passes,
  publications) and, per pass, submit/start/end/
  publication stamps with the derived durations. No change in
  behaviour; the session collection (U1) records this snapshot in
  the manifest.
- **Missing:** the glass-to-ink stopwatch at Lucas's desk
  (capture→submit and publication→screen are not visible from inside
  the worker) and a real session recorded with telemetry
  on. No parameter changes on the basis of this audit: changing
  cadence or threshold without that number would be guessing, and the cost in
  transient false positives (empty warps/furniture) has not even
  been measured yet.
