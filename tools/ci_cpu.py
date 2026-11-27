"""Portable CPU-only CI runner. Tests are independent scripts, not one shared process."""
from pathlib import Path
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("tests/test_*.py", "tests/numerical/test_*.py", "tools/test_*.py",
            "tools/rep/test_*.py")
REQUIRED = {"tests/test_dense.py", "tests/test_moe.py", "tools/test_experiment.py",
            "tools/test_bf16_harness.py", "tools/rep/test_validate_result.py"}

def run():
    paths = sorted({p.relative_to(ROOT).as_posix() for g in PATTERNS for p in ROOT.glob(g)})
    if not REQUIRED <= set(paths):
        raise RuntimeError("required regression suite missing: " + str(REQUIRED-set(paths)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join(
        str(ROOT / p) for p in ("src", "tools", "tools/rep")), OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", PYTHONUTF8="1")
    failed = []
    for path in paths:
        print(f"\n=== {path} ===", flush=True)
        try:
            r = subprocess.run([sys.executable, path], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            print("TIMEOUT", flush=True)
            failed.append(path)
            continue
        print(r.stdout, end="")
        print(r.stderr, end="")
        if r.returncode:
            failed.append(path)
        elif re.search(r"\bskipped=\d+", r.stderr+r.stdout):
            # BF16 is hardware-dependent on hosted CPU. Its explicit skip is
            # visible, not a claim of BF16 validation; all other skips fail.
            if path == "tools/test_bf16_harness.py":
                print("BF16 CPU capability skip: not validated on this runner")
            else:
                failed.append(path + " (unexpected skip)")
    if failed:
        print("Failed:", ", ".join(failed), file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT/"tools"/"rep"))
    from validate_result import validate, read_json
    evidence = sorted((ROOT/"experiments"/"results").glob("T*/attempt-*/result.json"))
    if not evidence:
        raise RuntimeError("No REP attempt evidence found")
    for p in evidence:
        r = read_json(p)
        # Checks are self-declared here; task-specific acceptance is reviewed
        # separately. This gate validates byte integrity and structure only.
        outcome = validate(p.parent, r["task_id"], r["source_commit"], list(r["checks"]))
        print(f"Evidence {p.parent.relative_to(ROOT)}: {outcome} (structural)")
    print(f"CPU scripts: {len(paths)}; REP reports: {len(evidence)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
