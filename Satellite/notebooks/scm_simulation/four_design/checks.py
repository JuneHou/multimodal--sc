"""Checks of scm-four-design before any model call, checks 1 to 5 of the plan's Verification
(~/.claude/plans/use-nature-skills-analyze-wiggly-coral.md, top section). No model is called.

    reproduction  through fit.R in the box, the README way written in the four choices gives the scm package's README
                  way fit (Synth), and the package's reference gives Synth's fit of the same Synth arguments, within
                  0.0001 in weights, V and predictions, on both panels at sigma 1.0, 0.5 and 0.2
    combinations  one summary {t: 1} for each t in 1..40 gives the every_outcome fit, under judgement with 42 ones and
                  under pre_period_fit; a summary with weight 1/40 on every period gives the fit of the same summary
                  with weight 1 on every period
    options       one spec for each option of each choice runs through fit.R in the box on both panels at sigma 1.0,
                  exits 0 and records a fit whose spec holds the translated arguments and whose design is the spec
    refusals      every spec of the pipeline's refusal battery and each of the five rules broken once is refused or
                  accepted alike by the Python checks and by four_fields.R through fit.R in the box, naming the same
                  fields, in both skills
    refits        arm_checks.R on each translated spec of `options` reproduces the fit script's predictions; leaving
                  period 40 out of a judgement spec gives one V entry fewer; leaving a period out of a summary gives
                  the refit of that summary recomputed without the period

    python four_design/checks.py [reproduction] [combinations] [options] [refusals] [refits]   (from scm_simulation/)
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM.parent / "sc_skills"))
sys.path.insert(0, str(SIM))

import scs_method   # noqa: E402

METHOD = "../../scm_simulation/four_design"
mod = scs_method.load_module(METHOD)
base = mod.base
pkg = mod.PACKAGE
S = pkg.spec

import scs_build    # noqa: E402
import scs_check    # noqa: E402
import scs_loop     # noqa: E402
import scs_spec     # noqa: E402
import scs_tests    # noqa: E402
from scm_checks import named_fields   # noqa: E402
import her_tables as h                # noqa: E402

ONES = {"option": "judgement", "values": [1] * 42}
README_FOUR = {"predictors": {"option": "every_outcome", "covariate_periods": [40]}, "v": ONES,
               "weights": {"option": "convex"}, "penalty": {"option": "none"}}
REF = dict(mod.REFERENCE)


def same(a, b):
    return scs_check.same_number(float(a), float(b))


def same_fit(a, b, keys=("weights", "v", "predictions")):
    """Weights, V and predictions within 0.0001, and the same donors and periods."""
    if a["donor_ids"] != b["donor_ids"] or a["predictions"].keys() != b["predictions"].keys():
        return False
    for k in keys:
        x, y = a[k], b[k]
        pairs = [(x[t], y[t]) for t in x] if isinstance(x, dict) else list(zip(x, y))
        if isinstance(x, list) and len(x) != len(y):
            return False
        if not all(same(p, q) for p, q in pairs):
            return False
    return True


def box(spec, case, use, fmt="design"):
    """spec written to the skill's file and fit.R run in the box -> (output, exit status, fit records)."""
    c = scs_loop.Case(scs_tests.built(pkg) / scs_build.skill_name(pkg, fmt), case, use)
    try:
        f = scs_build.SPEC_FILE[fmt]
        c.do_shell(f"cat > {f} <<'EOF'\n{json.dumps(spec)}\nEOF")
        out, meta = c.do_shell(scs_tests.fit_command(pkg, f))
        return out, meta["exit"], list(c.fit_records)
    finally:
        c.close()


def uses():
    return [(s, base.USES[base.use_name(s)]) for s in base.SIGMAS]


def check_reproduction():
    readme = base.README_WAY
    jobs = [(s, use, case, name, spec) for s, use in uses() for case in use.cases()
            for name, spec in (("README way", README_FOUR), ("reference", REF))]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: box(j[4], j[2], j[1]), jobs))
    bad = []
    for (s, use, case, name, spec), (out, code, recs) in zip(jobs, res):
        if code != 0 or len(recs) != 1:
            bad.append((s, case["ctx"]["case"], name, "no fit", out[-200:]))
            continue
        ctx = case["ctx"]
        if name == "README way":
            ref = base.PACKAGE.engine.reference(ctx, scs_spec.validate(base.SPEC, dict(readme), ctx))
        else:
            ref = base.PACKAGE.engine.reference(ctx, recs[0]["spec"])
        if not same_fit(recs[0], ref):
            bad.append((s, ctx["case"], name, "differs"))
    print(f"reproduction: {len(jobs)} fits in the box (three noise levels, two panels, the README way and the "
          f"reference) against Synth run here; {len(bad)} differ {bad[:4]}")
    return not bad


def engine(spec, ctx):
    return mod.ENGINE.native(ctx, scs_spec.validate(S, spec, ctx), False)       # the fit script's rule


def check_combinations():
    ctx = base.CONTEXTS()
    pre = range(1, 41)
    unit = [{str(t): 1} for t in pre]
    bad, n = [], 0
    for c in ctx:
        for v in (ONES, {"option": "pre_period_fit"}):
            a = engine({**README_FOUR, "v": v}, c)
            b = engine({**README_FOUR, "v": v, "predictors": {"option": "summaries", "covariate_periods": [40],
                                                              "summaries": unit}}, c)
            n += 1
            if not same_fit(a, b):
                bad.append((c["case"], v["option"], "each period as a summary"))
        for v in ({"option": "judgement", "values": [1, 1, 1]}, {"option": "pre_period_fit"}):
            fits = [engine({**README_FOUR, "v": v, "predictors": {"option": "summaries", "covariate_periods": [40],
                                                                  "summaries": [{str(t): w for t in pre}]}}, c)
                    for w in (1 / 40, 1)]
            n += 1
            if not same_fit(*fits):
                bad.append((c["case"], v["option"], "average and sum"))
    print(f"combinations: {n} pairs of fits; {len(bad)} differ {bad}")
    return not bad


OPTION_SPECS = {
    "predictors every_outcome": REF,
    "predictors summaries": {**REF, "predictors": {"option": "summaries", "covariate_periods": [36, 37, 38, 39, 40],
                                                   "summaries": [{"40": 1}, {"35": 0.5, "36": 0.5}]}},
    "v judgement": {**REF, "v": ONES},
    "v pre_period_fit": REF,
    "v validation": {**REF, "predictors": {"option": "every_outcome", "covariate_periods": [30]},
                     "v": {"option": "validation", "training_end": 30}},
    "weights convex": REF,
    "weights extrapolation": {**REF, "weights": {"option": "extrapolation"}},
    "penalty none": REF,
    "penalty distance": {**REF, "penalty": {"option": "distance", "lambda": 0.1}},
}
_OPTION_FITS = {}


def option_fits():
    if not _OPTION_FITS:
        use = base.USES["simulation"]
        jobs = [(name, case) for name in OPTION_SPECS for case in use.cases()]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda j: box(OPTION_SPECS[j[0]], j[1], use), jobs))
        _OPTION_FITS.update({(name, case["ctx"]["case"]): r for (name, case), r in zip(jobs, res)})
    return _OPTION_FITS


def check_options():
    bad = []
    keys = {"predictors", "predictors_op", "time_predictors_prior", "special_predictors", "outcome_combinations",
            "controls_identifier", "custom_v", "time_optimize_ssr", "optimxmethod", "genoud", "quadopt", "margin_ipop",
            "sigf_ipop", "bound_ipop", "weights", "interpolation_penalty"}
    for (name, case), (out, code, recs) in option_fits().items():
        if code != 0 or len(recs) != 1:
            bad.append((name, case, "no fit", out[-200:]))
            continue
        r = recs[0]
        if not keys <= set(r["spec"]) or r["design"] != json.loads(json.dumps(OPTION_SPECS[name])):
            bad.append((name, case, "record"))
        if name == "weights extrapolation" and r["engine_used"] != "native":
            bad.append((name, case, "engine"))
        if name == "penalty distance" and (r["engine_used"] != "native" or r["spec"].get("penalty_lambda") != 0.1):
            bad.append((name, case, "engine"))
    print(f"options: {len(option_fits())} fits in the box (nine options, two panels); {len(bad)} failed {bad[:4]}")
    return not bad


RULE_BREAKS = [
    {**REF, "predictors": {"option": "every_outcome", "covariate_periods": [41]}},
    {**REF, "predictors": {"option": "summaries", "covariate_periods": [40], "summaries": [{"0": 1}]}},
    {**REF, "v": {"option": "judgement"}},
    {**REF, "v": {"option": "pre_period_fit", "training_end": 30}},
    {**REF, "v": {"option": "validation", "training_end": 30}},
    {**REF, "v": {"option": "judgement", "values": [1, 1, 1]}},
    {**REF, "weights": {"option": "extrapolation"}, "penalty": {"option": "distance", "lambda": 0.1}},
]


def check_refusals():
    use, ctx = base.USES["simulation"], base.CONTEXTS()[0]
    case = use.cases()[0]

    def python_says(fmt, given):
        try:
            (scs_spec.settings_only if fmt == "configure" else scs_spec.validate)(S, given, ctx)
            return False, set()
        except scs_spec.SpecError as e:
            return True, named_fields(str(e))

    def fit_r_says(fmt, given):
        out, code, recs = box(given, case, use, fmt)
        m = re.search(r"^refused: (.*)$", out, re.M)
        return ((True, named_fields(m.group(1))) if m else (False, set())), code, out

    jobs = ([(fmt, g) for fmt in ("configure", "design") for g in scs_tests.spec_battery(S, fmt == "configure")]
            + [("design", g) for g in RULE_BREAKS])
    with ThreadPoolExecutor(8) as ex:
        r_results = list(ex.map(lambda j: fit_r_says(*j), jobs))
    bad, n_refused = [], 0
    for (fmt, given), ((r_ref, r_fields), code, out) in zip(jobs, r_results):
        p_ref, p_fields = python_says(fmt, given)
        n_refused += p_ref
        if p_ref != r_ref or p_fields != r_fields or (r_ref and code == 0) or (not r_ref and code != 0):
            bad.append((fmt, json.dumps(given)[:140], ("python", p_ref, sorted(p_fields)),
                        ("fit.R", r_ref, sorted(r_fields), code, out[-160:])))
    broken = [python_says("design", g)[0] for g in RULE_BREAKS]
    print(f"refusals: {len(jobs)} specs through fit.R in the box ({n_refused} refused by the Python checks, every rule "
          f"break refused: {all(broken)}); {len(bad)} where fit.R and the Python checks disagree")
    for b in bad[:8]:
        print("   ", b)
    return not bad and all(broken)


def check_refits():
    bad, n = [], 0
    for (name, case), (out, code, recs) in option_fits().items():
        if code != 0 or len(recs) != 1:
            bad.append((name, case, "no fit"))
            continue
        r = recs[0]
        a = h.run_checks(case, r["spec"], "1.0")
        n += 1
        if max(abs(a["predictions"][t] - r["predictions"][t]) for t in r["predictions"]) > h.SAME:
            bad.append((name, case, "arm_checks.R differs from the fit"))
    judgement = option_fits()[("v judgement", "dgp1")][2][0]["spec"]
    w40 = h.without_period(judgement, 40)
    n += 1
    if len(w40["custom_v"]) != len(judgement["custom_v"]) - 1:
        bad.append(("v judgement", "dgp1", "V entries after leaving period 40 out"))
    else:
        h.run_checks("dgp1", w40, "1.0")
    s3 = {**OPTION_SPECS["predictors summaries"],
          "predictors": {"option": "summaries", "covariate_periods": [36, 37, 38, 39],
                         "summaries": [{"38": 1 / 3, "39": 1 / 3, "40": 1 / 3}, {"35": 0.5, "36": 0.5}]}}
    s2 = {**s3, "predictors": {**s3["predictors"], "summaries": [{"38": 0.5, "39": 0.5}, {"35": 0.5, "36": 0.5}]}}
    rec3 = engine(s3, base.CONTEXTS()[0])
    left = h.run_checks("dgp1", h.without_period(rec3["spec"], 40), "1.0")
    s2_spec = engine(s2, base.CONTEXTS()[0])["spec"]              # leaving 40 out also drops it from the V search
    s2_spec["time_optimize_ssr"] = [t for t in s2_spec["time_optimize_ssr"] if int(t) != 40]
    recomputed = h.run_checks("dgp1", s2_spec, "1.0")
    n += 1
    if max(abs(left["predictions"][t] - recomputed["predictions"][t]) for t in recomputed["predictions"]) > h.SAME:
        bad.append(("summary", "dgp1", "period 40 left out differs from the summary recomputed without it"))
    print(f"refits: {n} checks of arm_checks.R and the rewrites; {len(bad)} failed {bad}")
    return not bad


CHECKS = {"reproduction": check_reproduction, "combinations": check_combinations, "options": check_options,
          "refusals": check_refusals, "refits": check_refits}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CHECKS)
    results = {n: bool(CHECKS[n]()) for n in names}
    print("\n" + "; ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in results.items()))
    raise SystemExit(0 if all(results.values()) else 1)
