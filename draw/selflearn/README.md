# draw/selflearn: the loop that learns on its own (Loop A, slice 1)

Draw in production is only inference with frozen weights. This package
is the beginning of the closed offline loop that changes that: the app collects what
it sees, the monitor points out where the network probably erred, an ordered
queue decides what is worth correcting, an offline teacher generates
pseudo-labels with veto, and a training round in the UniMatch
V1 style (EMA teacher, weak-to-strong) learns from labeled + pseudo.
Learning happens between sessions, never inside the app in real time.

## The loop

```
sessao ao vivo
  -> [0 MONITOR]  monitor.py: entropia/margem ranqueiam suspeita;
     tinta que some entre passadas e sinal de erro; a testemunha
     classica cruza com a rede e separa candidato a FP e a FN
  -> [1 COLETA]   collect.py: ExampleRecorder grava warp + mapa de
     probabilidade + linha de manifesto com sha256, sessao, modelo
     e origem (a linhagem de cada exemplo)
  -> [2 SELECAO]  select.py: score auditavel (desacordo TTA +
     dispersao temporal + impureza de regiao + novidade + estrato
     de risco), teto por sessao e cota aleatoria garantida
  -> [3 TEACHER]  teacher.py: voto temporal (mediana das passadas
     alinhadas) e pseudo-rotulo com veto da testemunha: tinta sem
     escurecimento vai para ignore, nunca para fundo; escurecimento
     que a rede nao viu tambem vai para ignore
  -> [5 TREINO]   tools/selflearn/train_round.py: uma rodada com
     replay integral dos rotulados em todo batch, teacher EMA
     (rampa inicial, teto 0.999) rotulando a vista fraca, aluno na
     vista forte, perda supervisionada BCE+Dice + consistencia
     mascarada pela confianca. Exige linhagem e holdout independente,
     preserva warm start, avalia pagina inteira so ao final e exporta
     ONNX + metrics.json + candidate_real.json. Ver docs/TREINO.md
```

## Modules

- `monitor.py` pure suspicion and divergence functions (step 0).
- `collect.py` example recorder in dataset format (U1).
- `select.py` active learning queue with reasons per item (U2).
- `teacher.py` temporal aggregation and pseudo-label with veto (U3).
- `tools/selflearn/train_round.py` the training round (minimum U5).
  `--smoke` exercises the recipe on CPU with its own synthetic dataset:
  supervised loss falls, EMA moves, lineage/holdout guards run, and the
  ONNX is checked against PyTorch. It is not real-world quality evidence.

Tests: `tests/test_selflearn.py`.

## Loop v2: the six corrections of 2026-11-02

The second slice of the loop closes Lucas's corrections on the
first: the engine existed, what was missing was the real raw material and the
right gates.

- `session.py` (U1) automatic session collection, wired into the app:
  each generation records reference (first warp; in legacy, the warp
  of the C key), final warp, final canvas and `manifest.json` with
  sha256 of the files and of the models, keys with relative time and
  pass telemetry. Local only
  (`captures/sessions/<session_id>/`, outside git); another
  person's desk requires consent.
- `split.py` (U2) train x holdout split **by session**,
  deterministic by hash, stratified so that the holdout includes
  easy sessions and sessions with blank pages. Guards that raise
  `LeakageError`: a holdout session does not enter training, uncertainty
  selection, pseudo-labeling or threshold tuning; the
  holdout label is independent (final photo - reference, reviewed by
  hand, `draft_independent_label` + `guard_holdout_example`) and
  a pseudo-label never enters it. Correction of the prediction (with its
  anchoring bias) is training, never holdout.
- `gate.py` + `tools/selflearn/judge_gate.py` (U3) the promotion
  gate: who decides is the sealed real replay; DIBCO IoU comes out
  labeled as diagnostic and promotes no one (round 4
  won on it and lost on the real one). Below the pre-registered
  minimum n (default: 5 sessions and 25 holdout pages), the
  verdict is INCONCLUSIVE with the n values printed - the replay of ~48
  old files still has no statistical power.
- Latency telemetry (U0): `_PageCleanWorker` exposes
  `telemetry_snapshot()` (submits, latest-wins replacements,
  submit/start/end/publication timestamps per pass) without changing
  behavior, and `tools/selflearn/measure_latency.py` measures the
  stages over recorded warps. The audit is in
  `docs/AUDITORIA_LATENCIA.md`: the temporal consensus of the hypothesis
  does not run on the standard main path (the vote accumulator was
  removed in 8b9ff45; the legacy 15-window only runs with
  `--legacy-ink`), and the measured floor of the worker does not close the
  2.3-3.5 s - the glass-to-ink stopwatch on the desk is missing.

Loop v2 tests: `tests/test_selflearn_session.py`,
`tests/test_selflearn_split.py`, `tests/test_selflearn_gate.py`,
`tests/test_page_clean_telemetry.py`.

## What is NOT implemented yet

- U4: minimal correction interface (edit the prediction with brush and
  scribble, budget in minutes and gestures).
- Historical stress set and automatic pre-registration of the round in the
  gate (the current gate judges primary/worst-group/n; promotion
  of a model remains a human decision).
- PrevMatch-lite (vote of checkpoints from previous rounds).
- Loop B (larger offline teacher with per-channel distillation to the
  student) and Loop C (masked pre-training of the backbone).
- Integration with the Kaggle runner (tools/runner) and the automatic
  upload of the versioned dataset.
- Per-tile change detection in the app (real-time front of the
  addendum): the monitor here is offline, not wired to the app.
