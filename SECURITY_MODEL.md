# Draw security threat model

## Scope and authority

- **User-confirmed:** security scans cover the whole `santleme/draw` repository. No other repositories are included; the maintainer confirmed there is no separate infrastructure repository.
- **User-confirmed protection priorities:** developer-machine access and cloud/CI credentials; model, dataset, and release integrity; private webcam frames and handwriting. No ranking among these three priorities was supplied.
- **User-confirmed attacker scope:** all plausible attackers remain in scope. There are no additional exclusions or accepted risks.
- Finding labels: **Component** only, using the stable names below.
- This file is the repository's security model; future scans and reviews start from it. The system description below is inferred from code and documentation; those inferences alone do not justify changing a finding's severity.

## System overview and trust boundaries

Draw is a desktop Python/OpenCV application that captures a webcam or operator-selected MJPEG URL, detects paper with ONNX models, rectifies it, and extracts ink. The normal runtime performs inference, not training. Captures and learning examples are written to local disk. Separate offline tools prepare data, train/export models, and run jobs on Kaggle or other accelerator environments. A React website is a separate static deployment. The experimental Windows helper streams timestamped JPEGs to a WSL receiver and is not integrated into the main app.

Key flows:
1. Webcam or configured network stream → native decoder/inference process → displayed ink and local captures/session records.
2. GitHub Release or configured model mirror → manifest verification → local model files → ONNX runtime.
3. Captures, external datasets and checkpoints → offline selection/training/evaluation → candidate models and release artifacts.
4. Developer recipes/scripts → credential-bearing Kaggle connector → private cloud kernels and fetched outputs.
5. Website visitor → static React application and browser-local sketch pad; no application API or account system was found in the website source.
6. Repository contributions and package registries → CI/build hosts → static deployment or artifacts.

These cross the camera/network-to-native-process boundary, external-file-to-loader boundary, local-data-to-cloud boundary, browser/static-host boundary, and untrusted-contribution-to-CI boundary.

## Components

| Component | Code and responsibilities |
| --- | --- |
| **Capture runtime** | `main.py`, `draw/` except `draw/selflearn/` and the model-verification utility: camera/MJPEG input, native decoding and inference, display, saved images, session reports and telemetry. |
| **Camera bridge** | `bridge/`, `tools/helper_receiver.py`: experimental Windows TCP helper, JPEG framing, recorded stream files and WSL receiver. |
| **Model distribution** | `models/MANIFEST.json`, `tools/fetch_models.py`, `draw/manifest.py`: model provenance, release downloads, checksum/size verification and runtime policy. |
| **Learning and cloud tooling** | `training/`, `notebooks/`, `draw/selflearn/`, `tools/runner/`, `tools/selflearn/`, remaining dataset/evaluation/operator tools: private sessions, external data/checkpoints, training, cloud jobs and model promotion. |
| **Website** | `site/`: static React/TypeScript pages, browser-local drawing, static archive images and outbound source links. |
| **Delivery and setup** | `.github/`, `scripts/`, `requirements*.txt`, `pyproject.toml`, `.replit`: CI, dependency installation, webcam setup and static deployment configuration. |

Assign a finding to the component owning its root cause; prefer the explicit model-distribution and bridge mappings over the broader runtime/tooling mappings.

## Entry points and reachable actors

- The desktop CLI accepts a webcam index or MJPEG URL, custom model paths, and an output directory; environment variables can choose models and download mirrors. Operators choose these values, but files or remote stream content can originate elsewhere.
- Image, JPEG, ONNX, dataset, checkpoint, manifest, recipe and recorded-stream parsers consume external or previously saved material. File provenance and what an attacker can influence must be established per finding.
- `bridge/PROTOCOL.md` assumes same-PC Windows ↔ WSL communication without TLS. The helper sketch now binds `127.0.0.1:8765` by default (or a specific interface IP such as the `vEthernet (WSL)` adapter); `0.0.0.0`/`::` requires `--allow-all-interfaces` and prints a security warning. Every connection must pass a shared-token handshake before any frame is sent. Actual reachability still depends on host/WSL networking and firewall rules, the stream is not encrypted, and the helper remains an untested sketch. The same-PC design is **not** evidence that LAN access is impossible, and the maintainer's no-exclusions answer is **not** a waiver for exposure or malicious local processes.
- Public visitors can reach deployed static website content. No backend, multi-tenant data store or login flow was found; do not invent one.
- Contributors and dependency/release providers can influence material consumed by developers and CI, subject to the repository/hosting permissions that actually apply.

## Assets and sensitivity

All three groups are explicitly important to the maintainer:
- **Private captures:** raw desk frames, handwriting, rectified pages, saved canvases, session examples and metadata. These may contain personal or confidential material; legal classifications and retention policy are not confirmed.
- **Execution and credentials:** developer machines, filesystem access, camera access, Kaggle connector credentials, CI tokens, release/deployment authority and cloud compute.
- **Integrity:** model binaries/manifests, checkpoints, dataset lineage, held-out evaluation data, promotion decisions, build outputs and release assets.

## Attacker profiles

Keep plausible malicious model/dataset/checkpoint/stream suppliers, compromised dependencies, untrusted contributors, website visitors, LAN peers, and other local users/processes in scope. Consider compromised mirrors/cloud outputs where the finding establishes that influence. Do not assume every actor has every permission, nor assume an already fully compromised host merely to manufacture an exploit.

## Existing controls and their limits

- **Bridge (camera stream).** `bridge/windows_helper.py` binds loopback by default and refuses wildcard binds without `--allow-all-interfaces` (with a warning). The protocol has a token handshake (`DRAH` HELLO + 16..256-byte token, compared with `hmac.compare_digest`, `DROK` reply, 5 s timeout; documented in `bridge/PROTOCOL.md`); the receiver (`tools/helper_receiver.py`) refuses to connect without `DRAW_HELPER_TOKEN`/`--token-file` and never logs the token. The receiver rejects `payload_len` outside `1..16 MiB` before reading or allocating, and `decode_jpeg` returns `None` (never raises) for payloads without `FF D8`…`FF D9`, a missing/truncated SOF, dimensions above 8192 px/side or 40 MP, or decoder rejection. Tests: `tests/test_bridge_security.py`, `python tools/helper_receiver.py --selfcheck`. Limits: no TLS; the helper capture loop is still a stub that has never run on Windows.
- **Model integrity.** `draw/manifest.py` is strict by default: a sha256/size mismatch, or a model file with no manifest entry (including a missing manifest), raises before the session opens. Only `DRAW_MODEL_STRICT=0`/`false`/`no`/`off` or the app flag `--allow-unverified` downgrades both to a warning; any other value is strict. Every runtime loader verifies: `draw/neural.py`, `draw/page_clean.py` and the MediaPipe `.task` in `draw/hand.py`. All three default models, including `hand_landmarker.task`, are registered in `models/MANIFEST.json`. `python main.py --check` fails on `.onnx`/`.task` files in `models/` with no entry. The manifest is read with a 1 MiB cap and each entry is validated (64-hex sha256, non-negative integer `bytes`). A refused optional model (page-clean) makes the app fall back to its non-model path rather than load it. Tests: `tests/test_manifest.py`. The test suite opts out (`tests/conftest.py`) because its ONNX fixtures are synthetic.
- **External-input parsers.** `draw/safe_io.py` bounds images (PNG/JPEG only, 64 MiB, 16384 px/side, 64 MP, checked from the header before native decoding; malformed input returns `None`), JSON (16 MiB) and JSONL (256 MiB, 1 M lines, 1 MiB/line, object rows), turning syntax errors into `InputLimitError` with path and line. It is used by the session-pool scanner (`draw/selflearn/process.py`, which also skips corrupt session manifests and refuses generation paths that escape the session folder), the self-learning dataset loader (`tools/selflearn/train_round.py`), the model manifest and the bridge receiver. Every `torch.load` in `draw/`, `tools/` and `training/` passes `weights_only=True` without a full-unpickler fallback (enforced by an AST test), except the byte-for-byte frozen bodies `training/train_ink3.py` and `training/train_body_tri.py`. `tools/vectorize.py` loads `.npz` stamps with `allow_pickle=False`. Runner recipes validate `kernel_slug` and `datasets` against the Kaggle slug alphabet before they reach API paths. Tests: `tests/test_input_hardening.py`. ONNX graphs are only opened after manifest verification in the app; offline evaluation tools still open operator-chosen ONNX files with onnxruntime directly.
- **Private captures and self-learning.** `.gitignore` covers `captures/`, `sessions/`, `logs/`, `dataset/`, model binaries, `*.drawcap` recordings and `selflearn_round_out/`. Because `--save-dir` can point anywhere, the app's save directory, each session folder (`<root>/sessions/<session_id>/`), `ExampleRecorder` output and the pool output each get a `.gitignore` containing `*` (`safe_io.ensure_private_dir`). Lineage is enforced on write as well as read: `ExampleRecorder.record` refuses rows without `session_id`/`origin` or with an unknown split; `select.rank_examples` requires an explicit `holdout_session_ids` and runs `guard_selection_pool`; `process_pool` keeps sealed holdouts sealed; `train_round.py --split-ledger` refuses holdout pages from unsealed sessions and keeps every sealed session out of training. Consent and human promotion remain procedural (`draw/selflearn/README.md`); they are not enforced by code.
- **Downloads and credentials.** `draw/net_policy.py` applies the runner's rules to every other network path: HTTPS only (HTTP only on loopback when a caller opts in for tests), exact host allowlist, no embedded URL credentials, and every redirect hop re-checked or refused. `tools/fetch_models.py` allows the GitHub Release hosts plus the configured mirror host, stops reading at the registered size and refuses names with path components. `tools/runner/collect.py` runs `curl --proto =https --max-redirs 0` without `-L`, treats any non-200/206 status as failure, sends no credentials and restricts hosts to `www.kaggle.com`/`storage.googleapis.com` (override: `DRAW_COLLECT_ALLOWED_HOSTS`). `training/finetune_real.py` replaced `urlretrieve` with an allowlisted HTTPS opener. `tools/runner/runner_kaggle.py` is unchanged: credential injection limited to `www.kaggle.com`, redirects refused, private kernels with internet disabled. No other script in `tools/` or `training/` makes network calls or handles credentials. The `storage.googleapis.com` default was inferred, not observed against a live kernel output.
- **CI and dependencies.** `.github/workflows/ci.yml` keeps workflow-level `permissions: contents: read`; only the change-filter job adds `pull-requests: read`; no job has write scope. Third-party actions are pinned to commit SHAs. Every install in CI is pinned: `pytest==9.2.0`, `ruff==0.17.4`, `torch==2.15.0` (CPU index), `transformers==5.22.2`, and `requirements-dev.txt` pins `onnx`, `numpy` (2.5.4 on Python ≥ 3.12, 2.4.4 on 3.11), `opencv-contrib-python` and `onnxruntime`. `training/requirements.txt` now pins `transformers`. The runtime `requirements.txt` keeps ranges on purpose for app users; `pyproject.toml` declares no dependencies. Branch protection, release credentials and hosting controls are not established.
- `.replit` declares static website deployment. External firewall rules, hosted response headers, account ACLs and cloud dataset-sharing settings were not verified.

## Accepted risk, unknowns and reranking policy

**No additional exclusions or accepted risks were supplied.** Deliberate local operation is not automatically a vulnerability, but documentation of an experimental or local feature does not remove it from scope.

Open deployment questions: whether the Windows helper is used and reachable beyond the machine; actual website hosting/security headers; private-dataset sharing/retention; cloud/CI identity privileges; and any controls maintained outside this repository. Proceed without assuming answers or weakening scope.

Start from each finding's scanner severity and confirmed exploit prerequisites. Change priority only for user-confirmed facts above or explicit repository-authored security commitments, not for this document's code inferences. A confirmed path to one of the maintainer's named sensitive assets may justify a rare increase, but not when the scanner says the exploit is difficult or unverified. Because the maintainer excluded no attacker or asset and accepted no additional risk, most findings should retain the scanner's severity.

## Evidence

Grounded in `README.md` (usage, private training data, layout), `bridge/PROTOCOL.md` (same-PC security perimeter), `bridge/windows_helper.py`, `tools/helper_receiver.py`, `draw/app.py`, `tools/fetch_models.py`, `draw/manifest.py`, `draw/selflearn/README.md`, `tools/runner/runner_kaggle.py`, `site/src/`, `.github/workflows/ci.yml` and `.replit`. Source inspected at commit `58d9f4da9f0449046e8578ff79246cd341ded48a`; the "Existing controls" section was updated with the hardening that closed the bridge, model-verification, parser, privacy, download and CI findings. User confirmations were given by the maintainer (@santleme) on 2026-11-07 during security-scan setup. Update this file when components, trust boundaries, priorities or accepted risks change.
