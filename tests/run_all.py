"""Runner unificado da bateria de testes, verificacao formal e selfchecks.

Roda toda a verificacao sem webcam do repositorio num comando so:
  1. tests/selftest.py (marcadores, sem marcadores, aninhada, textura)
  2. verification/run_verification.py (198 sondas geometricas)
  3. verification/run_verification.py --sabotage (contraprova: TEM de falhar)
  4. todos os tests/test_*.py + tests/cadence_fusion.py
  5. todos os selfchecks das ferramentas em tools/

Uso: python tests/run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _steps():
    py = sys.executable
    items = [
        ("selftest", [py, "tests/selftest.py"], True),
        ("verification", [py, "verification/run_verification.py"], True),
        (
            "verification --sabotage (deve falhar)",
            [py, "verification/run_verification.py", "--sabotage"],
            False,
        ),
    ]
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        rel = path.relative_to(ROOT)
        items.append((str(rel), [py, str(rel)], True))
    items.append(("tests/cadence_fusion.py", [py, "tests/cadence_fusion.py"], True))
    for tool in (
        "tools/helper_receiver.py",
        "tools/run_log.py",
        "tools/step0_latency.py",
        "tools/vectorize.py",
    ):
        items.append((f"{tool} --selfcheck", [py, tool, "--selfcheck"], True))
    return items


def main():
    failures = []
    t_start = time.perf_counter()
    for label, cmd, expect_zero in _steps():
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        dt = time.perf_counter() - t0
        ok = (proc.returncode == 0) if expect_zero else (proc.returncode != 0)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label:<44} ({dt:.2f}s)")
        if not ok:
            failures.append((label, proc.returncode, proc.stdout))

    total_dt = time.perf_counter() - t_start
    if failures:
        print(f"\nFALHARAM {len(failures)} etapa(s) em {total_dt:.2f}s:")
        for label, rc, out in failures:
            print(f"\n--- {label} (exit={rc}) ---\n{out}")
        return 1
    print(f"\nTODA A BATERIA PASSOU ({len(_steps())} etapas em {total_dt:.2f}s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
