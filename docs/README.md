# docs/ — documents and records

Two kinds of files live here. **Living documents** are updated as the
project changes. **Dated records** were written for a specific
experiment or review and are kept unchanged as evidence — a
pre-registration only proves something if its bytes never move.

## Living documents

- `ESTADO_ATUAL.md` — state of the capture system: what works, the
  current failure map and the prioritized evidence gaps.
- `FALHAS.md` — taxonomy of the failures that matter, used to judge
  any model or behavior change. Guarded by `tests/test_falhas_doc.py`.
- `SETUP_WINDOWS.md` — camera setup on Windows with WSL.
- `TREINO.md` — training, evaluation and export guide.
- [SIMULATOR.md](SIMULATOR.md) — text-driven synthetic camera, MP4/ground
  truth export and evaluation through the real capture loop.
- `MEMORIA_DO_PROJETO.md` — running memory of decisions, situations
  and mistakes.

## Dated records

- `PLANO_DE_AVANCO_DADOS_E_MODELO_2026-11-06.md` — the 2026-11-06
  advance plan for data, model and product rules.
- `REVISAO_DADOS_KAGGLE_2026-11-06.md` — initial review of the real
  data held on Kaggle, 2026-11-06.
- `F07_B3_PRE_PORTAO_RESULTADO_2026-11-07.md`,
  `F07_B3_PRE_REGISTRO_2026-11-07.md`,
  `F07_B4_PRE_REGISTRO_2026-11-07.md` (+ `.json`),
  `F07_B4_R1_PRE_REGISTRO_2026-11-07.md` (+ `.json`) —
  pre-registrations of the F-07 temporal experiments, written before
  the runs they describe.
- `F07_REPLAY_V2_A_R1_2026-11-09.md` (+ `.json`) and
  `F07_REPLAY_V2_A_R1_RESULTADOS_2026-11-09.md` — the replay v2
  mode-A run record and its results.

`icon.svg` is the project icon used by the root README.
