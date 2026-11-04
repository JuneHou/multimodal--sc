"""SCM checks run before any model call, besides the pipeline's own tests (plan, "SCM checks").

    source     every function in the CRAN source's R files (code/Synth/R) is identical, parsed and deparsed, to the
               installed Synth 1.1-10 function of the same name, and the two hold the same functions
    native     with both design choices at Synth's values, scm_native.R gives Synth's weights, V and synthetic path
               within 0.0001 on both cases at both reference ways
    refusals   every spec of the pipeline's refusal battery (scs_tests.spec_battery), written to the settings file and
               run through scripts/fit.R in the box, is refused by fit.R exactly when the package's Python checks
               refuse it, and both name the same fields

    show_data  scripts/show_data.py runs in the box and prints the table, for every variable, both cases, both skills

    fit_capture  every fit run in the box reaches the run record: printed, filtered with grep -v, redirected to a
               file, cut by head, or started from Python

    python scm_checks.py [source] [native] [refusals] [show_data] [fit_capture]      all five when none is named
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SC = HERE.parent / "sc_skills"
sys.path.insert(0, str(SC))

import scs_method   # noqa: E402

METHOD = "../../scm_simulation"


def check_source():
    r = r"""
inst <- asNamespace("Synth")
env <- new.env()
for (f in list.files("code/Synth/R", pattern = "[.]R$", full.names = TRUE)) sys.source(f, envir = env, keep.source = FALSE)
src <- Filter(function(n) is.function(get(n, envir = env)), ls(env, all.names = TRUE))
ins <- Filter(function(n) is.function(get(n, envir = inst)), ls(inst, all.names = TRUE))
same <- vapply(src, function(n) n %in% ins && identical(deparse(get(n, envir = env)), deparse(get(n, envir = inst))), TRUE)
cat(jsonlite::toJSON(list(source = src, installed = ins, identical = src[same], differ = src[!same]), auto_unbox = FALSE))
"""
    p = subprocess.run([str(scs_method.load_module(METHOD).RSCRIPT), "-e", r], capture_output=True, text=True, cwd=HERE)
    out = json.loads(p.stdout.strip().splitlines()[-1])
    extra = sorted(set(out["installed"]) - set(out["source"]))
    print(f"source: {len(out['source'])} functions in the CRAN source, {len(out['installed'])} in the installed "
          f"package; {len(out['identical'])} identical, {len(out['differ'])} differ {out['differ']}, "
          f"installed only {extra}")
    return not out["differ"] and not extra and len(out["source"]) > 0


def check_native():
    import scs_check
    import scs_spec
    mod = scs_method.load_module(METHOD)
    pkg = mod.PACKAGE
    n, n_identical, bad = 0, 0, []
    for ctx in mod.CONTEXTS():
        for way, spec in mod.REFERENCE_WAYS.items():
            v = scs_spec.validate(pkg.spec, dict(spec), ctx)
            a, b = pkg.engine.reference(ctx, v), pkg.engine.native(ctx, v, True)
            n += 1
            pairs = [(x, y) for k in ("weights", "v") for x, y in zip(a[k], b[k])] + \
                    [(a["predictions"][k], b["predictions"][k]) for k in a["predictions"]]
            same_ids = a["donor_ids"] == b["donor_ids"] and a["predictions"].keys() == b["predictions"].keys()
            if not same_ids or not all(scs_check.same_number(x, y) for x, y in pairs):
                bad.append((ctx["case"], way))
            n_identical += same_ids and all(x == y for x, y in pairs)
    print(f"native: {n} fits (two cases by two reference ways); {n_identical} identical to Synth, "
          f"{n - n_identical - len(bad)} the same within 0.0001, {len(bad)} differ {bad}")
    return not bad and n == 4


FIELD_PATTERNS = [r"field '([^']+)'", r"fields ((?:'[^']+',?)+)", r"missing required field\(s\): ([^;]+)",
                  r"unknown field\(s\): ([^;]+)", r"but got ([^;]+)"]


def named_fields(message):
    out = set()
    for part in message.split("; "):
        for pat in FIELD_PATTERNS:
            for m in re.finditer(pat, part):
                out |= {x.strip().strip("'") for x in re.split(r",\s*", m.group(1)) if x.strip()}
    return out


def check_refusals():
    import scs_build
    import scs_loop
    import scs_spec
    import scs_tests
    mod = scs_method.load_module(METHOD)
    pkg, use = mod.PACKAGE, mod.USES["simulation"]
    case, ctx = use.cases()[0], mod.CONTEXTS()[0]
    root = scs_tests.built(pkg)
    S = pkg.spec

    def python_says(fmt, given):
        try:
            (scs_spec.settings_only if fmt == "configure" else scs_spec.validate)(S, given, ctx)
            return False, set()
        except scs_spec.SpecError as e:
            return True, named_fields(str(e))

    def fit_r_says(fmt, given):
        c = scs_loop.Case(root / scs_build.skill_name(pkg, fmt), case, use)
        try:
            f = scs_build.SPEC_FILE[fmt]
            c.do_shell(f"cat > {f} <<'EOF'\n{json.dumps(given)}\nEOF")
            out, meta = c.do_shell(scs_tests.fit_command(pkg, f))
        finally:
            c.close()
        m = re.search(r"^refused: (.*)$", out, re.M)
        return (True, named_fields(m.group(1))) if m else (False, set()), meta["exit"], out

    jobs = [(fmt, g) for fmt in ("configure", "design") for g in scs_tests.spec_battery(S, fmt == "configure")]
    with ThreadPoolExecutor(8) as ex:
        r_results = list(ex.map(lambda j: fit_r_says(*j), jobs))
    bad, n_refused = [], 0
    for (fmt, given), ((r_ref, r_fields), code, out) in zip(jobs, r_results):
        p_ref, p_fields = python_says(fmt, given)
        n_refused += p_ref
        if p_ref != r_ref or p_fields != r_fields:
            bad.append((fmt, json.dumps(given)[:120], ("python", p_ref, sorted(p_fields)),
                        ("fit.R", r_ref, sorted(r_fields), code, out[-160:])))
    print(f"refusals: {len(jobs)} specs through fit.R in the box ({n_refused} refused by the Python checks); "
          f"{len(bad)} where fit.R and the Python checks disagree")
    for b in bad[:8]:
        print("   ", b)
    return not bad


def check_show_data():
    """scripts/show_data.py runs in the box for every variable, on both cases, in both skills."""
    import scs_build
    import scs_loop
    mod = scs_method.load_module(METHOD)
    pkg, use = mod.PACKAGE, mod.USES["simulation"]
    root = (HERE / "skills").resolve()
    bad, n = [], 0
    for case in use.cases():
        for fmt in ("configure", "design"):
            c = scs_loop.Case(root / scs_build.skill_name(pkg, fmt), case, use)
            try:
                for arg in ("", " Z1", " Z2", " Y_observed"):
                    out, meta = c.do_shell("python scripts/show_data.py" + arg)
                    n += 1
                    if meta["exit"] != 0 or "by period (rows) and unit (columns)" not in out or "donor_29" not in out:
                        bad.append((case["prompt_id"], fmt, arg, out[-200:]))
            finally:
                c.close()
    print(f"show_data: {n} runs in the box (two cases, two skills, four calls); {len(bad)} failed")
    for b in bad[:4]:
        print("   ", b)
    return not bad and n == 16


def check_fit_capture():
    """Every fit run in the box reaches the run record, whatever the command does with fit.R's output: printed,
    filtered with grep -v, redirected to a file, cut by head, or started from Python. Both skills, one case."""
    import scs_build
    import scs_loop
    mod = scs_method.load_module(METHOD)
    pkg, use = mod.PACKAGE, mod.USES["simulation"]
    root = (HERE / "skills").resolve()
    case = use.cases()[0]
    bad, n = [], 0
    for fmt in ("configure", "design"):
        f = scs_build.SPEC_FILE[fmt]
        keys = list(pkg.spec.settings) + ([] if fmt == "configure" else list(pkg.spec.choices))
        spec = json.dumps({k: mod.README_WAY[k] for k in keys})
        ways = {"printed": f"Rscript scripts/fit.R {f}",
                "grep -v": f"Rscript scripts/fit.R {f} 2>&1 | grep -v 'period'",
                "redirected": f"Rscript scripts/fit.R {f} > fit_output.txt 2>&1; wc -l fit_output.txt",
                "cut by head": f"Rscript scripts/fit.R {f} 2>&1 | head -5",
                "from Python": f"python -c \"import subprocess; subprocess.run(['Rscript', 'scripts/fit.R', '{f}'], "
                               f"capture_output=True)\""}
        c = scs_loop.Case(root / scs_build.skill_name(pkg, fmt), case, use)
        try:
            c.do_shell(f"cat > {f} <<'EOF'\n{spec}\nEOF")
            for way, cmd in ways.items():
                before = len(c.fit_records)
                out, meta = c.do_shell(cmd)
                n += 1
                got = c.fit_records[before:]
                if meta["exit"] != 0 or len(got) != 1 or got[0].get("engine_used") != "synth":
                    bad.append((fmt, way, len(got), out[-200:]))
        finally:
            c.close()
    print(f"fit_capture: {n} fits in the box (two skills, five ways of running fit.R); {len(bad)} not recorded once")
    for b in bad[:4]:
        print("   ", b)
    return not bad and n == 10


CHECKS = {"source": check_source, "native": check_native, "refusals": check_refusals, "show_data": check_show_data,
          "fit_capture": check_fit_capture}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CHECKS)
    results = {n: CHECKS[n]() for n in names}
    print("SCM checks: " + ", ".join(f"{n} {'pass' if ok else 'FAIL'}" for n, ok in results.items()))
    raise SystemExit(0 if all(results.values()) else 1)
