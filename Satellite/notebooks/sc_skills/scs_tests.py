"""scs_tests — tests before any model call, for one method package. No test calls a model.

Generic tests (every method):
    copy        the copy of the reference code equals a fresh copy (scs_copy.check)
    check       the copied code works on the data (scs_check: parity for collaborator's code, R trial for R)
    refusals    a spec with a wrong field is refused with a message naming the field
    ... (the build, sandbox and loop tests are added with those modules)

Reproduction of the existing SHC skills (packages that carry OLD, the implementation they had before):
    same_refusals   scs_spec gives the same result as the old validator, message for message, on a battery of specs
    same_dispatch   the copied code is chosen for exactly the choice combinations the old dispatch chose it for
    same_fits       scs_fit gives the same fit as the old fit on every case at the reference values, and on the
                    native engine for one non-reference spec per case

    python scs_tests.py --method <method> [test ...]      every test, or the named ones
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):     # one thread, as the box runs, before
    os.environ[_v] = "1"                                                        # numpy is loaded
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import scs_method  # noqa: E402

BAD_VALUES = [None, "x", True, False, 0, -1, 0.5, 1.5, 3, 100, 1e9, [], [0], [1, 2], [-1], {}]
INT_PAIR_VALUES = [1, 5, 10, 50, 200, None]
OPTIONAL_VALUES = [None, [], ["x"], "x", ["VV"], ["VV", "VV"], ["B2", "NDVI"]]


# ----------------------------------------------------------------------------- batteries
def choice_combinations(S):
    opts = {k: (tuple(f.allowed) if f.kind == "enum" else (True, False)) for k, f in S.choices.items()}
    return [dict(zip(opts, c)) for c in itertools.product(*opts.values())]


def spec_battery(S, settings_only=False):
    ref = {k: S.reference[k] for k in (S.settings if settings_only else list(S.settings) + list(S.choices))}
    out = [[], "x", None, dict(ref)]
    for k in ref:
        out.append({kk: vv for kk, vv in ref.items() if kk != k})
        out += [{**ref, k: v} for v in BAD_VALUES]
    out += [{**ref, "extra_field": 1}, {**ref, "P10": 1}]
    ints = [k for k, f in S.settings.items() if f.kind in ("int", "int_or_null")]
    for a, b in itertools.combinations(ints, 2):
        out += [{**ref, a: x, b: y} for x in INT_PAIR_VALUES for y in INT_PAIR_VALUES]
    if settings_only:
        out += [{**ref, k: S.reference[k]} for k in S.choices] + [{**ref, k: v} for k in S.optional for v in OPTIONAL_VALUES]
        return out
    for combo in choice_combinations(S):
        out.append({**ref, **combo})
        out += [{**ref, **combo, k: v} for k in S.optional for v in OPTIONAL_VALUES]
    return out


def outcome(fn, given, ctx, errors):
    try:
        return "ok", json.dumps(fn(given, ctx), sort_keys=True)
    except errors as e:
        return "refused", str(e)


# ----------------------------------------------------------------------------- generic tests
def test_copy(pkg, mod):
    import scs_copy
    return scs_copy.check(pkg)


def test_check(pkg, mod):
    import scs_check
    return scs_check.check(pkg)


def test_refusals(pkg, mod):
    """Every field, given a wrong value or left out, is refused with a message naming it."""
    import scs_spec
    S, ctx = pkg.spec, _contexts(mod)[0]
    bad = []
    for k in list(S.settings) + list(S.choices):
        for given in ({kk: vv for kk, vv in S.reference.items() if kk != k}, {**S.reference, k: "not a value"}):
            try:
                scs_spec.validate(S, given, ctx)
                bad.append((k, "accepted"))
            except scs_spec.SpecError as e:
                if k not in str(e):
                    bad.append((k, str(e)))
    print(f"refusals: {2 * (len(S.settings) + len(S.choices))} wrong specs, {len(bad)} not refused by name")
    return not bad


# ----------------------------------------------------------------------------- reproduction of the SHC skills
def _contexts(mod):
    old = getattr(mod, "OLD", None)
    return old["contexts"]() if old else mod.CONTEXTS()


def test_same_refusals(pkg, mod):
    import scs_spec
    old, S = mod.OLD, pkg.spec
    errors = (scs_spec.SpecError, old["error"])
    ctxs = _contexts(mod)
    by_sensor = {}
    for c in ctxs:
        by_sensor.setdefault(c["sensor"], c)
    n, diff = 0, []
    for ctx in by_sensor.values():
        for so in (False, True):
            new_fn = (lambda g, c: scs_spec.settings_only(S, g, c)) if so else (lambda g, c: scs_spec.validate(S, g, c))
            old_fn = old["settings_only"] if so else old["validate"]
            for given in spec_battery(S, so):
                a, b = outcome(new_fn, given, ctx, errors), outcome(old_fn, given, ctx, errors)
                n += 1
                if a != b:
                    diff.append((ctx["sensor"], "configure" if so else "design", given, a, b))
    print(f"same_refusals: {n:,} specs (design and configure, one case per sensor); {len(diff)} differ")
    for d in diff[:5]:
        print("   ", d)
    return not diff


def test_same_dispatch(pkg, mod):
    import scs_spec
    S = pkg.spec
    combos = choice_combinations(S)
    diff = [c for c in combos if scs_spec.reference_choices(S, {**S.reference, **c}) != mod.OLD["at_reference"]({**S.reference, **c})]
    print(f"same_dispatch: {len(combos)} choice combinations; {len(diff)} routed differently")
    return not diff


def same_value(a, b):
    """Numbers the same within scs_check.SAME_WITHIN (Jun 2026-10-31 "just 4 decimal is enough", plan departure 3);
    integers, booleans, text and ids exactly."""
    import numpy as np
    import pandas as pd
    import scs_check
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return (isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame) and list(a.columns) == list(b.columns)
                and a.index.equals(b.index) and all(same_value(a[c].to_numpy(), b[c].to_numpy()) for c in a.columns))
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            return all(scs_check.same_number(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
        if a.dtype.kind == "O":
            return all(same_value(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
        return bool(np.all(a == b))
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)) and \
            scs_check.same_number(float(a), float(b))
    return a == b


def identical_value(a, b):
    """Exact equality, NaN equal to NaN (reported beside same_value)."""
    import numpy as np
    import pandas as pd
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(identical_value(a[k], b[k]) for k in a)
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame) and a.equals(b)
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and bool(np.all((a == b) | (pd.isna(a) & pd.isna(b)))) if a.dtype.kind in "fc" else \
            a.shape == b.shape and bool(np.all(a == b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_same_fits(pkg, mod):
    import scs_fit
    import scs_spec
    S = pkg.spec
    native = {**S.reference, **mod.OLD["native_choices"]}
    n, n_identical, diff = 0, 0, []
    for ctx in _contexts(mod):
        for given in (dict(S.reference), native):
            spec = scs_spec.validate(S, given, ctx)
            a = scs_fit.run(pkg, ctx, spec)
            b = mod.OLD["fit"](ctx, spec)
            n += 1
            n_identical += identical_value(a, b)
            if not same_value(a, b):
                bad = [k for k in a if k not in b or not same_value(a[k], b[k])]
                diff.append((ctx["sensor"], ctx.get("site"), a["engine_used"], bad))
    print(f"same_fits: {n} fits ({n // 2} cases at the reference values and with {mod.OLD['native_choices']}); "
          f"{n_identical} identical, {n - n_identical - len(diff)} the same within 0.0001, {len(diff)} differ")
    for d in diff[:5]:
        print("   ", d)
    return not diff


# ----------------------------------------------------------------------------- build, box and loop
_BUILT = {}


def built(pkg):
    """A fresh build of both skills in a temporary folder, once per process."""
    if pkg.name not in _BUILT:
        import scs_build
        root = Path(tempfile.mkdtemp(prefix="scs_build_"))
        scs_build.build(pkg, root)
        _BUILT[pkg.name] = root
    return _BUILT[pkg.name]


def _use(mod):
    return next(iter(mod.USES.values()))


def test_build(pkg, mod):
    """The skill files equal what the builder writes (and, for a method that had skills, the existing skills)."""
    import scs_build
    root = built(pkg)
    d = scs_build.compare(pkg, root)
    print(f"build: {len(scs_build.files(pkg))} files; a fresh build differs in {len(d)}")
    ok = not d
    if hasattr(mod, "OLD"):
        d2 = scs_build.compare(pkg, mod.OLD["skills"])
        print(f"build: against the existing skills in {mod.OLD['skills']}: {len(d2)} differences")
        ok &= not d2
    return ok


def _spec_text(pkg, fmt):
    S = pkg.spec
    keys = list(S.settings) if fmt == "configure" else list(S.settings) + list(S.choices)
    return json.dumps({k: S.reference[k] for k in keys})


def _box_fit(pkg, case, use, fmt):
    import scs_build
    import scs_loop
    skill = built(pkg) / scs_build.skill_name(pkg, fmt)
    c = scs_loop.Case(skill, case, use)
    try:
        f = scs_build.SPEC_FILE[fmt]
        c.do_shell(f"cat > {f} <<'EOF'\n{_spec_text(pkg, fmt)}\nEOF")
        out, meta = c.do_shell(fit_command(pkg, f))
        return out, meta, list(c.fit_records)
    finally:
        c.close()


def test_box_fit(pkg, mod):
    """The fit script in the box at the reference code's values gives the copied code's output on every case."""
    import scs_fit
    import scs_spec
    use = _use(mod)
    n, n_identical, bad = 0, 0, []
    for case in use.cases():
        spec = scs_spec.validate(pkg.spec, dict(pkg.spec.reference), case["ctx"])
        ref = scs_fit.run(pkg, case["ctx"], spec, engine="reference")
        for fmt in ("configure", "design"):
            out, meta, recs = _box_fit(pkg, case, use, fmt)
            n += 1
            if meta["exit"] != 0 or len(recs) != 1 or FIT_MARK_SHOWN(out):
                bad.append((case["prompt_id"], fmt, "no fit record", out[-300:]))
                continue
            r = recs[0]
            n_identical += (r["donor_ids"] == list(ref["donor_ids"]) and r["weights"] == [float(x) for x in ref["weights"]]
                            and r["predictions"] == ref["predictions"] and r["fit_mse"] == ref["fit_mse"])
            same = (r["donor_ids"] == list(ref["donor_ids"]) and same_value(r["weights"], [float(x) for x in ref["weights"]])
                    and same_value(r["predictions"], ref["predictions"]) and same_value(r["fit_mse"], ref["fit_mse"]))
            if not same:
                bad.append((case["prompt_id"], fmt, "differs"))
    print(f"box_fit: {n} fit script runs in the box (both skills, every case) against the copied code run here: "
          f"{n_identical} identical, {n - n_identical - len(bad)} the same within 0.0001, {len(bad)} differ")
    for b in bad[:5]:
        print("   ", b)
    return not bad and n > 0


def FIT_MARK_SHOWN(text):
    import scs_loop
    return scs_loop.FIT_MARK in text


def test_box_refusal(pkg, mod):
    """A settings file with a wrong field, run through the fit script in the box, is refused naming the field."""
    import scs_build
    import scs_loop
    use = _use(mod)
    case = use.cases()[0]
    S = pkg.spec
    s1 = next(iter(S.settings))
    c1 = next(iter(S.choices))
    trials = [("configure", {k: S.reference[k] for k in S.settings if k != s1}, s1),
              ("configure", {**{k: S.reference[k] for k in S.settings}, c1: S.reference[c1]}, c1),
              ("design", {**S.reference, c1: "not an option"}, c1)]
    bad = []
    for fmt, given, field in trials:
        c = scs_loop.Case(built(pkg) / scs_build.skill_name(pkg, fmt), case, use)
        try:
            f = scs_build.SPEC_FILE[fmt]
            c.do_shell(f"cat > {f} <<'EOF'\n{json.dumps(given)}\nEOF")
            out, meta = c.do_shell(fit_command(pkg, f))
            if meta["exit"] == 0 or "refused" not in out or field not in out or c.fit_records:
                bad.append((fmt, field, out[-200:]))
        finally:
            c.close()
    print(f"box_refusal: {len(trials)} wrong files; {len(bad)} not refused by name")
    return not bad


def test_isolation(pkg, mod):
    """Inside the box: no home folder, no project, no network, the skill files read-only."""
    import scs_build
    import scs_loop
    use = _use(mod)
    c = scs_loop.Case(built(pkg) / scs_build.skill_name(pkg, "design"), use.cases()[0], use)
    project = str(Path(__file__).resolve().parents[2])
    probes = {"home folder hidden": (f"ls {Path.home()}", lambda o, e: e != 0),
              "project hidden": (f"ls {project}", lambda o, e: e != 0),
              "no network": ("python -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\"",
                             lambda o, e: e != 0),
              "SKILL.md read-only": ("echo x >> SKILL.md", lambda o, e: e != 0),
              "scripts read-only": ("touch scripts/new_file", lambda o, e: e != 0),
              "case folder writable": ("touch note.txt && echo written", lambda o, e: e == 0 and "written" in o)}
    bad = []
    try:
        for name, (cmd, ok) in probes.items():
            out, meta = c.do_shell(cmd)
            if not ok(out, meta["exit"]):
                bad.append((name, out[-200:]))
    finally:
        c.close()
    print(f"isolation: {len(probes)} probes; {len(bad)} failed")
    for b in bad:
        print("   ", b)
    return not bad


class FakeModel:
    """A scripted stand-in for a model: returns the given replies in turn and keeps every message list it was sent."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages):
        self.seen.append([dict(m) for m in messages])
        text = self.replies.pop(0) if self.replies else "```bash\necho waiting\n```"
        return text, {"finish_reason": "stop"}


FIT_RUNNER = {".py": "python", ".R": "Rscript"}      # the fit script runs with the interpreter of its language


def fit_command(pkg, f):
    name = pkg.texts.fit_script_name
    return f"{FIT_RUNNER[Path(name).suffix]} scripts/{name} {f}"


def _fence(cmd):
    return f"```bash\n{cmd}\n```"


def test_dry_run(pkg, mod):
    """A scripted fake model reads the skill, runs the fit script and writes an answer: the case ends with it."""
    import scs_build
    import scs_fit
    import scs_loop
    import scs_spec
    use = _use(mod)
    case = use.cases()[0]
    bad = []
    for fmt in ("configure", "design"):
        spec = scs_spec.validate(pkg.spec, dict(pkg.spec.reference), case["ctx"])
        fit = scs_fit.run(pkg, case["ctx"], spec, engine="reference")
        fit = json.loads(json.dumps({"predictions": fit["predictions"]}))
        f = scs_build.SPEC_FILE[fmt]
        model = FakeModel([_fence("cat SKILL.md"), _fence(f"python scripts/show_data.py"),
                           _fence(f"cat > {f} <<'EOF'\n{_spec_text(pkg, fmt)}\nEOF"),
                           _fence(fit_command(pkg, f)),
                           _fence(f"cat > answer.json <<'EOF'\n{json.dumps(use.dry_answer(case, fit))}\nEOF")])
        rec = scs_loop.run_case(built(pkg) / scs_build.skill_name(pkg, fmt), case, use, model)
        if rec["ended"] != "answer" or rec["n_steps"] != 5 or len(rec["fit_runs"]) != 1 or not rec["data_shown"]:
            bad.append((fmt, rec["ended"], rec["n_steps"], len(rec["fit_runs"])))
    print(f"dry_run: 2 scripted runs; {len(bad)} did not end with the answer after one fit")
    return not bad


def test_loop_rules(pkg, mod):
    """The step cap, an empty reply, an empty command output and resuming behave as in the SHC runs."""
    import scs_build
    import scs_loop
    import scs_run
    use = _use(mod)
    case = use.cases()[0]
    skill = built(pkg) / scs_build.skill_name(pkg, "configure")
    bad = []
    rec = scs_loop.run_case(skill, case, use, FakeModel([]))
    if rec["ended"] != "step_cap" or rec["n_steps"] != scs_loop.STEP_CAP:
        bad.append(("step cap", rec["ended"], rec["n_steps"]))
    m = FakeModel(["", _fence("true")])
    scs_loop.run_case(skill, case, use, m, step_cap=3)
    if m.seen[1][-2]["content"] != "(empty reply)":
        bad.append(("empty reply", m.seen[1][-2]))
    if m.seen[2][-1]["content"] != "(no output)":
        bad.append(("empty output", m.seen[2][-1]))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        p.write_text("".join(json.dumps({"prompt_id": k, "ended": e}) + "\n"
                             for k, e in (("a", "answer"), ("b", "step_cap"), ("c", "call_error"))))
        if scs_run.resume_set(p) != {"a", "b"}:
            bad.append(("resume", scs_run.resume_set(p)))
    print(f"loop_rules: step cap, empty reply, empty output, resume; {len(bad)} failed")
    for b in bad:
        print("   ", b)
    return not bad


# ----------------------------------------------------------------------------- profile
def test_profile_request(pkg, mod):
    """The request text; for a method that had a profile run, it equals that run's request byte for byte."""
    import scs_profile
    text = scs_profile.request(pkg, _use(mod))
    ok = bool(text.strip()) and "{" + "task}" not in text
    old = getattr(mod, "OLD", {}).get("profile")
    if old:
        same = text == Path(old["request"]).read_text()
        print(f"profile_request: {len(text)} characters; " + ("byte-identical to " if same else "DIFFERS from ")
              + str(old["request"]))
        ok &= same
    return ok


def test_profile_collect(pkg, mod):
    """Collecting the stored generator runs again gives the tree hashes their solver.json recorded."""
    import scs_profile
    old = getattr(mod, "OLD", {}).get("profile") or {}
    if not old.get("runs"):
        print("profile_collect: no stored generator runs for this method")
        return True
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"           # read the stored working copies without refreshing their index
    root, bad = Path(old["root"]), []
    for g in old["runs"]:
        with tempfile.TemporaryDirectory() as tmp:
            m = scs_profile.collect(root / "work" / g, Path(tmp) / "out", Path(tmp) / "skills",
                                    root / "generated" / g / "run.log", root / "codex_home")
            want = json.loads((root / "skills" / g / "solver.json").read_text())
            for k in ("code_provided_sha", "code_written_sha", "env_prefix"):
                if m[k] != want[k]:
                    bad.append((g, k, m[k], want[k]))
    print(f"profile_collect: {len(old['runs'])} stored runs collected again; {len(bad)} values differ from solver.json")
    for b in bad:
        print("   ", b)
    return not bad


def test_profile_probe(pkg, mod):
    """A working copy prepared by script, probed in the generator's box: the project, the data and the home folder
    are absent, the paper is readable, no key is in the environment, the framework starts, Codex writes only inside."""
    import subprocess
    import scs_profile
    repo = HERE_PROFILE / "work" / "gpt-5.4"           # a local clone source at commit 4a8cc0c
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        r = scs_profile.prepare(pkg, _use(mod), "gpt-5.4", tmp, repo=repo)
        absent = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "data"),
                  str(Path.home())]
        p = subprocess.run(["bash", str(Path(__file__).resolve().parent / "scs_box.sh"), "probe", "gpt-5.4",
                            str(r["work"]), str(r["codex_home"]), *absent], capture_output=True, text=True, timeout=300)
        out = p.stdout
        want = [*(f"absent: {a}" for a in absent), "paper: readable", "env names with KEY: 0", "framework: starts",
                "codex sandbox write inside: ok", "codex sandbox write outside: denied"]
        bad = [w for w in want if w not in out]
        if "network: reachable" not in out:
            bad.append("network")
    print(f"profile_probe: {len(want) + 1} probe lines expected; {len(bad)} missing")
    for b in bad:
        print("   ", b)
    if bad:
        print(out[-1500:], p.stderr[-800:])
    return not bad


def same_table(a, b):
    """Two tables read as text: the same columns and rows, each cell the same text except numbers the same within
    0.0001 (scs_check.same_text)."""
    import scs_check
    return list(a.columns) == list(b.columns) and a.shape == b.shape and \
        all(scs_check.same_text(x, y) for c in a.columns for x, y in zip(a[c], b[c]))


def test_same_steps(pkg, mod):
    """From the stored records, the donors table and the reasoning rows equal the existing analysis tables."""
    import io
    import pandas as pd
    import scs_steps
    hooks = mod.STEPS
    t = scs_steps.tables(pkg, mod, _use(mod), hooks)
    ok = True
    read = lambda text: pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    ours = read(t["donors"].to_csv(index=False))
    old = read(Path(hooks["old_tables"]["donors"]).read_text())
    # the existing tables were built for monthly and daily together and then split, so each carries the other data
    # set's setting columns, empty in every row; those columns must be empty, every other column must be equal
    # rows are compared by their identity and columns by name: the existing daily table follows the match file's row
    # order, the generic one the record files'
    key = ["data", "skill", "model", "sensor", "site_id", "prompt_id"]
    by_key = lambda df: df.sort_values(key, kind="stable").reset_index(drop=True)
    extra = [c for c in old.columns if c not in ours.columns]
    extra_empty = all((old[c] == "").all() for c in extra)
    same_d = (extra_empty and not [c for c in ours.columns if c not in old.columns]
              and same_table(by_key(ours[[c for c in old.columns if c in ours.columns]]), by_key(old.drop(columns=extra))))
    print(f"same_steps: donors table, {len(ours)} rows and {len(ours.columns)} columns: "
          + ("equal to " if same_d else "DIFFERS from ") + Path(hooks["old_tables"]["donors"]).name
          + f" (whose {len(extra)} columns of the other data set are " + ("empty in every row)" if extra_empty else
                                                                            "NOT empty)"))
    ok &= same_d
    old_r = read(Path(hooks["old_tables"]["reasoning"]).read_text())
    old_r = old_r[old_r["data"] == hooks["data"]].reset_index(drop=True)
    new_r = read(t["reasoning"].to_csv(index=False))
    same_r = same_table(by_key(new_r), by_key(old_r))
    print(f"same_steps: reasoning rows, {len(new_r)} ours and {len(old_r)} in the existing table: "
          + ("equal" if same_r else "DIFFER"))
    ok &= same_r
    if "old_match" in hooks:
        m = pd.read_csv(hooks["old_match"])
        old_m = {x.prompt_id: (None if pd.isna(x.matched_run) else int(x.matched_run)) for x in m.itertuples() if x.ok}
        new_m = {x.prompt_id: x.matched_run for x in t["matched"].itertuples() if x.ok}
        new_m = {k: (None if pd.isna(v) else int(v)) for k, v in new_m.items()}
        same_m = new_m == old_m
        print(f"same_steps: fit behind each answer, {len(new_m)} answers: " + ("equal to " if same_m else "DIFFERS from ")
              + Path(hooks["old_match"]).name)
        ok &= same_m
    return ok


HERE_PROFILE = Path(__file__).resolve().parents[1] / "skillfoundry_profile"

GENERIC = ["copy", "check", "refusals", "build", "box_fit", "box_refusal", "isolation", "dry_run", "loop_rules",
           "profile_request", "profile_collect", "profile_probe"]
REPRODUCTION = ["same_refusals", "same_dispatch", "same_fits", "same_steps"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("tests", nargs="*")
    a = ap.parse_args(argv)
    mod = scs_method.load_module(a.method)
    pkg = mod.PACKAGE
    names = a.tests or GENERIC + (REPRODUCTION if hasattr(mod, "OLD") else [])
    results = {}
    for name in names:
        print(f"-- {name}")
        results[name] = bool(globals()[f"test_{name}"](pkg, mod))
    print("\n" + "; ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in results.items()))
    return all(results.values())


if __name__ == "__main__":
    raise SystemExit(0 if main(sys.argv[1:]) else 1)
