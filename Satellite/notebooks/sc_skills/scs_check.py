"""scs_check — does the copied reference code work on the data (configure and design, step 2)?

Collaborator's code: the copy, run on the data at the reference values, must reproduce what the collaborator's notebook
printed. The package's ReferenceCode.parity() yields (case, quantity, ours, printed), ours written the way the
collaborator's code prints it. Two numbers are the same when they differ by at most SAME_WITHIN (Jun 2026-10-31 "when you
check whether same, just 4 decimal is enough"); two printed lines are the same when they are the same text except numbers
that are the same, taken on the printed decimals, with the same whitespace between tokens (same_text, after
llm_band_sc/lb_her14_run.py within_0001). Every check that our numbers reproduce a reference uses same_number and
same_text.

A package written in R (the R trial, Jun 2026-10-30 "what i want to test is in real data points whether R rynth
works"): we run the R script ourselves, with no model, inside the same bubblewrap box the model will use, on a few
cases of the real data. A case passes when the script exits 0 and prints a JSON line holding "weights" and
"prediction". If every case passes, the skill uses the original R code; otherwise the method needs a Python port, which
must give the same weights and prediction as R on those cases with R run outside the box.

    python scs_check.py <method>
"""
import decimal
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import scs_method

BWRAP = shutil.which("bwrap")
BOX_MOUNTS = ["--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
              "--ro-bind", "/bin", "/bin", "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev",
              "--tmpfs", "/tmp"]


SAME_WITHIN = decimal.Decimal("0.0001")          # Jun 2026-10-31 "just 4 decimal is enough" (plan departure 3)


def _number(tok):
    try:
        d = decimal.Decimal(tok)
    except decimal.InvalidOperation:
        return None
    return d if d.is_finite() else None


def same_number(a, b):
    """Two numbers differing by at most SAME_WITHIN; NaN equals NaN. Integers, booleans and infinities compare
    exactly."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, int) and isinstance(b, int):
        return a == b
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(decimal.Decimal(repr(a)) - decimal.Decimal(repr(b))) <= SAME_WITHIN


def same_text(a, b):
    """Line a equals line b except numbers that differ by at most SAME_WITHIN on the printed decimals."""
    if a == b:
        return True
    pa, pb = re.split(r"(\s+)", a), re.split(r"(\s+)", b)
    if len(pa) != len(pb):
        return False
    for x, y in zip(pa, pb):
        if x == y:
            continue
        nx, ny = _number(x), _number(y)
        if nx is None or ny is None or abs(nx - ny) > SAME_WITHIN:
            return False
    return True


def same(a, b):
    """same_text for two strings, same_number for two numbers, == otherwise."""
    if isinstance(a, str) and isinstance(b, str):
        return same_text(a, b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return same_number(a, b)
    return a == b


def parity(pkg):
    rows = list(pkg.reference_code.parity())
    bad = [r for r in rows if not same(r[2], r[3])]
    near = [r for r in rows if r[2] != r[3] and r not in bad]
    cases = len({r[0] for r in rows})
    print(f"{pkg.name}: {len(rows):,} printed values or lines over {cases} cases compared with the collaborator's "
          f"printout; {len(rows) - len(bad) - len(near):,} identical, {len(near)} the same within {SAME_WITHIN}, "
          f"{len(bad)} differ")
    for r in (bad or near)[:15]:
        print("   ", r)
    return not bad and len(rows) > 0


def run_in_box(cmd, case_dir, r_env, timeout=120):
    """Run `cmd` inside bubblewrap: the case folder read-write as the working directory, the R installation read-only,
    no network, an empty /tmp, one thread."""
    case_dir, r_env = os.path.realpath(case_dir), os.path.realpath(r_env)
    argv = [BWRAP, *BOX_MOUNTS, "--ro-bind", r_env, r_env, "--bind", case_dir, case_dir, "--chdir", case_dir,
            "--unshare-all", "--die-with-parent", "--clearenv",
            "--setenv", "PATH", f"{r_env}/bin:/usr/bin:/bin", "--setenv", "HOME", case_dir,
            "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1", *cmd]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def r_trial(script, cases, r_env):
    """-> [(case, passed, reason, parsed output)]. Each case folder is copied to a fresh temporary folder with the
    script, and the script runs there with Rscript."""
    out = []
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "case"
            shutil.copytree(case, work)
            shutil.copy(script, work / Path(script).name)
            try:
                p = run_in_box(["Rscript", Path(script).name], work, r_env)
            except subprocess.TimeoutExpired:
                out.append((str(case), False, "timed out", None))
                continue
            rec = None
            for line in p.stdout.splitlines()[::-1]:
                try:
                    rec = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            if p.returncode != 0:
                out.append((str(case), False, f"exit {p.returncode}: {p.stderr.strip()[-300:]}", rec))
            elif not isinstance(rec, dict) or "weights" not in rec or "prediction" not in rec:
                out.append((str(case), False, "no JSON line with weights and prediction", rec))
            else:
                out.append((str(case), True, "", rec))
    return out


def check(pkg):
    rc = pkg.reference_code
    if rc.origin == "collaborator":
        return parity(pkg)
    if rc.language == "R":
        t = rc.trial
        res = r_trial(t["script"], t["cases"], t["r_env"])
        for case, ok, reason, _ in res:
            print(f"  {case}: {'works' if ok else 'FAILS, ' + reason}")
        ok = all(r[1] for r in res) and len(res) > 0
        print(f"{pkg.name}: R trial on {len(res)} cases: " + ("the original R code is used" if ok else
                                                              "a Python port is needed"))
        return ok
    raise SystemExit(f"{pkg.name}: no check defined for {rc.origin} code in {rc.language}")


if __name__ == "__main__":
    ok = check(scs_method.load(sys.argv[1]))
    raise SystemExit(0 if ok else 1)
