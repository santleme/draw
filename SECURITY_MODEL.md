# Draw security threat model

## Scope and authority

- **User-confirmed:** security scans cover the whole `santleme/draw` repository. No other repositories are included; Lucas confirmed there is no separate infrastructure repository.
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
- `bridge/PROTOCOL.md` explicitly assumes same-PC Windows ↔ WSL communication, with no TLS or authentication and a warning not to expose the port. The helper sketch actually binds `0.0.0.0:8765` by default (`bridge/windows_helper.py`); actual reachability depends on host/WSL networking and firewall rules. The same-PC design is **not** evidence that LAN access is impossible, and Lucas's no-exclusions answer is **not** a waiver for exposure or malicious local processes.
- Public visitors can reach deployed static website content. No backend, multi-tenant data store or login flow was found; do not invent one.
- Contributors and dependency/release providers can influence material consumed by developers and CI, subject to the repository/hosting permissions that actually apply.

## Assets and sensitivity

All three groups are explicitly important to Lucas:
- **Private captures:** raw desk frames, handwriting, rectified pages, saved canvases, session examples and metadata. These may contain personal or confidential material; legal classifications and retention policy are not confirmed.
- **Execution and credentials:** developer machines, filesystem access, camera access, Kaggle connector credentials, CI tokens, release/deployment authority and cloud compute.
- **Integrity:** model binaries/manifests, checkpoints, dataset lineage, held-out evaluation data, promotion decisions, build outputs and release assets.

## Attacker profiles

Keep plausible malicious model/dataset/checkpoint/stream suppliers, compromised dependencies, untrusted contributors, website visitors, LAN peers, and other local users/processes in scope. Consider compromised mirrors/cloud outputs where the finding establishes that influence. Do not assume every actor has every permission, nor assume an already fully compromised host merely to manufacture an exploit.

## Existing controls and their limits

- `tools/fetch_models.py` uses an HTTPS GitHub Release by default, verifies manifest sha256/size before installation, removes failed partial downloads and installs with `os.replace`. The mirror can be overridden; the checked-in manifest is the trust anchor.
- `draw/manifest.py` checks registered models but defaults to a warning on mismatch; `DRAW_MODEL_STRICT=1` makes mismatches fatal. Unregistered files are not covered. This is an observed policy, not a user-accepted risk.
- `draw/selflearn/README.md` specifies local session collection outside git, consent for another person's desk, session-separated holdouts, lineage checks and human model promotion. These are documented design commitments; they do not prove every implementation path enforces them.
- `tools/runner/runner_kaggle.py` restricts credential injection to `www.kaggle.com`, refuses HTTP redirects, and requests private kernels with internet disabled. These controls do not establish every other training/upload path's behavior.
- `.github/workflows/ci.yml` declares read-only repository content permissions and adds pull-request read permission for change filtering. It runs lint/tests/builds; actual branch protection, release credentials and hosting controls are not established.
- `.replit` declares static website deployment. External firewall rules, hosted response headers, account ACLs and cloud dataset-sharing settings were not verified.

## Accepted risk, unknowns and reranking policy

**No additional exclusions or accepted risks were supplied.** Deliberate local operation is not automatically a vulnerability, but documentation of an experimental or local feature does not remove it from scope.

Open deployment questions: whether the Windows helper is used and reachable beyond the machine; actual website hosting/security headers; private-dataset sharing/retention; cloud/CI identity privileges; and any controls maintained outside this repository. Proceed without assuming answers or weakening scope.

Start from each finding's scanner severity and confirmed exploit prerequisites. Change priority only for user-confirmed facts above or explicit repository-authored security commitments, not for this document's code inferences. A confirmed path to one of Lucas's named sensitive assets may justify a rare increase, but not when the scanner says the exploit is difficult or unverified. Because Lucas excluded no attacker or asset and accepted no additional risk, most findings should retain the scanner's severity.

## Evidence

Grounded in `README.md` (usage, private training data, layout), `bridge/PROTOCOL.md` (same-PC security perimeter), `bridge/windows_helper.py`, `tools/helper_receiver.py`, `draw/app.py`, `tools/fetch_models.py`, `draw/manifest.py`, `draw/selflearn/README.md`, `tools/runner/runner_kaggle.py`, `site/src/`, `.github/workflows/ci.yml` and `.replit`. Source inspected at commit `58d9f4da9f0449046e8578ff79246cd341ded48a`. User confirmations were given by Lucas on 2026-11-07 during security-scan setup. Update this file when components, trust boundaries, priorities or accepted risks change.
