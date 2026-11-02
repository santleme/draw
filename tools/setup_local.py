"""One-command local setup for a fresh (or newly synced) clone.

Pulls, merges and sync automations only move tracked files. The
official model binaries (models/*.onnx, models/*.task) are gitignored
by design, so no sync ever brings them: without them the app falls
back to the classic sheet detector and the legacy ink stack, which is
exactly how a healthy, up-to-date clone ends up "not seeing the
paper" on a new desk.

This script closes that gap in one command:

    python tools/setup_local.py

1. Downloads and verifies the official models
   (tools/fetch_models.py, checked against models/MANIFEST.json).
2. Runs the built-in diagnostics (python main.py --check).

It exits non-zero if either step fails, and the output says which
one: models (step 1) or camera/environment (step 2).
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(step, argv):
    print(f"\n=== {step}: {' '.join(argv)}", flush=True)
    return subprocess.run(argv, cwd=REPO_ROOT).returncode


def main():
    python = sys.executable
    if _run("models", [python, "tools/fetch_models.py"]) != 0:
        print(
            "\nSETUP FAILED at the models step: the official model "
            "binaries are missing or do not match models/MANIFEST.json. "
            "Fix this before debugging detection: without "
            "draw_seg_v3_dyn.onnx the app falls back to the classic "
            "sheet detector, and without page_clean_unet.onnx it falls "
            "back to the legacy ink stack."
        )
        return 1
    if _run("diagnostics", [python, "main.py", "--check"]) != 0:
        print(
            "\nSETUP FAILED at the diagnostics step: the models are "
            "in place, so look at the FAIL lines above (camera, "
            "permissions, video device) - detection code is not the "
            "first suspect."
        )
        return 1
    print("\nSETUP OK: models verified and diagnostics passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
