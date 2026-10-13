"""lb_skill_traces — read the skill-arm transcripts cell by cell and count what they show (plan approved 2026-09-24).

Every count is printed with its denominator; every phrase-list hit is stored with +-90 characters of context in an
evidence column; a sample of the non-hits of each list is printed so the lists can be corrected before a figure is
quoted. Discarded phrase candidates are kept below as comments with the reason.

Questions (plan table): H1 harness friction; H2 token cap; A1 says the answer is the last month; A2 had a non-P09
alternative in its own runs and submitted P09; A3 stated criterion; A4 distinct settings across parcels; B1 reasons
for raw / exhaustive / zeta 0 / inverse-variance; B2 choices changed after a fit; C1 program class by AST; C2 blow-up
mechanism; C3 thinking states a simplification; I1 iteration vs matching error; D1 DeepSeek failures.

Writes lb_skill_traces.csv (one row per cell) and trace_workbooks/skills_<model>.xlsx (one sheet per arm, rows =
cells, columns = turns: thinking / command / output; last columns answer, rationale, fitter runs).
    python lb_skill_traces.py
"""
import ast
import collections
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import lb_lib as lb
import lb_skill_loop as L

PAD = 90
SHORT = {"DeepSeek-V4.1-Flash-thinking-low": "DeepSeek", "GLM-5.3": "GLM", "gpt-oss-120b": "gpt-oss", "openai:gpt-5.5": "GPT-5.5"}
ARMS = ("shc_configure", "shc_design", "shc_implement")
TOL = {"sentinel1": 0.006, "sentinel2": 0.0006}          # fit.py prints 2 / 3 decimals; the models copy the print

# ----------------------------------------------------------------------------- phrase lists (case-insensitive)
SAYS_LAST_MONTH = [r"persist", r"\bP09\b", r"last (observed |actual )?(month|value|observation|period)",
                   r"most recent (month|block|observation|period|value)", r"identical to", r"same as (the )?(july|last|latest|p09|final)",
                   r"\bnaive\b", r"carr(y|ied|ies) (it |the [\w-]+ )?forward", r"equal(s)? (to )?the (last|final|latest|observed)"]
# discarded: r"recent" alone (matches any mention of the recent months); r"block 6" (depends on m);
#            r"july" alone (S1 P09 is July but the rationale may discuss July for other reasons)
MIN_ERROR = [r"low(est|er)? (matching |in-sample |fit )?(error|mse)", r"smallest (matching )?(error|mse)", r"best[- ](fit|match)",
             r"minimi[sz]", r"error (is |was )?(low|small|minimal)"]
STABILITY = [r"\bstable\b", r"\bflat\b", r"\bsteady\b", r"little change", r"\bplateau\b", r"stabili"]
RECENCY = [r"most recent", r"recent (months|behavio|trend|pattern|state|regime|dynamics)", r"\blatest\b", r"recency"]
SIMPLICITY = [r"\bsimpl", r"straightforward", r"parsimon", r"interpretab", r"\bdirect(ly)?\b"]
# discarded for SIMPLICITY: r"basic" (rare and ambiguous); r"clean" (describes data as often as method)
SHORT_RECORD = [r"only nine", r"nine months", r"9 months", r"short (record|series|history|sample)", r"too few", r"not enough (data|months|observations)",
                r"few(er)? (observations|data points|months)", r"limited (data|history|record)"]
SEASON_NOT = [r"season\w* (is |are |cannot|can't|not|hard|difficult|poorly|unreliab|weak|noisy|would)", r"(no|little|weak|unreliable) season",
              r"deseasonali\w+ (is |would |could )?(not|unreliab|risky|unstable)", r"annual (sine|cycle|harmonic)"]
SCALE = [r"observed scale", r"convex combination", r"on the (observed|raw|original) scale", r"keeps? .{0,40}scale", r"scale of the (data|observations|bands)"]
EXHAUSTIVE_WHY = [r"exhaustive", r"all (possible )?subsets", r"every (possible )?(subset|combination)", r"global(ly)? (optim|best)", r"greedy .{0,40}(miss|sub-?optimal|local)"]
ZETA_ZERO_WHY = [r"zeta", r"no threshold", r"tolerance", r"unused", r"irrelevant"]
INVVAR_WHY = [r"inverse[- ]variance", r"equal footing", r"dominat", r"different (scales|magnitudes|units)", r"comparable scale", r"rescal"]
SIMPLIFY = [r"simplif", r"\bnaive\b", r"approximat", r"simpler", r"just use", r"\bskip", r"shortcut", r"good enough", r"placeholder", r"fallback"]


def hits(text, patterns, pad=PAD):
    """First match of any pattern with +-pad characters of context, or ''."""
    if not text:
        return ""
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            a, b = max(0, m.start() - pad), min(len(text), m.end() + pad)
            return ("…" if a else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")
    return ""


def think(s):
    return (s.get("usage") or {}).get("reasoning") or ""


# ----------------------------------------------------------------------------- records
def load():
    base = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")}
    recs = {}
    for exp in ARMS:
        for f in sorted(glob.glob(str(lb.LB / f"lb_replies_{exp}_*.jsonl")) + [f for f in glob.glob(str(lb.LB / "runs" / "*" / f"lb_replies_{exp}_*.jsonl")) if "/superseded_" not in f]):
            for l in open(f):
                r = json.loads(l)
                recs[(r["model"], r["prompt_id"])] = r
    out = []
    for (m, pid), r in recs.items():
        p = base[pid[:-len(r["experiment"][4:]) - 1] + "_shc"]
        r["_base"] = p
        r["_p09"] = {b: p["train"][b][-1] for b in p["train"]}
        r["_metric"] = (float(np.mean([lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b]) for b in p["truth"]]))
                        if r.get("ok") else np.nan)
        out.append(r)
    return out


def is_p09(r, vals):
    return vals is not None and all(abs(vals[b] - r["_p09"][b]) <= TOL[r["sensor"]] for b in r["_p09"])


def metric_of(r, vals):
    p = r["_base"]
    return float(np.mean([lb.her_nmse(p["truth"][b], vals[b], p["train"][b]) for b in p["truth"]]))


# ----------------------------------------------------------------------------- C1: program class by AST
def program_class(src):
    """(class, solver, constraint) from the program's syntax tree, not from comments or names in strings."""
    if not src:
        return "no program", "", ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "syntax error", "", ""
    calls, kws, attrs, subs = collections.Counter(), set(), collections.Counter(), 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            calls[name] += 1
            kws |= {k.arg for k in n.keywords if k.arg}
        if isinstance(n, ast.Attribute):
            attrs[n.attr] += 1
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
            subs += 1
    solver = next((s for s in ("nnls", "minimize", "lstsq", "solve", "polyfit", "linregress", "lsq_linear") if calls[s]), "")
    if solver in ("nnls", "lsq_linear"):
        constraint = "nonneg"
    elif solver == "minimize":
        constraint = "simplex" if "constraints" in kws and "bounds" in kws else ("nonneg" if "bounds" in kws else "unconstrained")
    elif solver in ("lstsq", "solve"):
        constraint = "unconstrained"
    else:
        constraint = ""
    if solver in ("polyfit", "linregress"):
        return "trend extrapolation", solver, ""
    if solver in ("nnls", "minimize", "lstsq", "solve", "lsq_linear"):
        return ("fits weights over blocks" if subs else "fits weights"), solver, constraint
    src_l = src.lower()
    last = bool(re.search(r"\[-1\]|\bp09\b|iloc\[-1\]|\.max\(\)|tail\(1\)|\[-1:\]", src)) or "last" in src_l
    mean = calls["mean"] or attrs["mean"] or calls["average"]
    if mean and last:
        return "mean of recent months", "", ""
    if last:
        return "returns last month", "", ""
    if mean:
        return "mean of months", "", ""
    return "other", "", ""


def weights_from_output(out):
    m = re.search(r"weights?\s*[:=]\s*\[?([-\d.eE+, \s]+)", out or "")
    if not m:
        return None
    try:
        w = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]
        return w or None
    except ValueError:
        return None


# ----------------------------------------------------------------------------- per-cell rows
def analyse(recs):
    rows = []
    for r in recs:
        exp, m, steps = r["experiment"], r["model"], r["steps"]
        fits = r.get("fit_runs") or []
        rat = r.get("rationale") or ""
        last_think = " ".join(think(s) for s in steps[-2:])
        row = {"experiment": exp, "arm": exp[4:], "model": SHORT.get(m, m), "sensor": r["sensor"], "site_id": r["site_id"],
               "prompt_id": r["prompt_id"], "ok": bool(r.get("ok")), "ended": r.get("ended"), "n_steps": r.get("n_steps"),
               "metric": r["_metric"], "answer_is_p09": is_p09(r, r.get("values")) if r.get("ok") else None,
               "fit_runs": len(fits), "blocks_kept_final": len(fits[-1]["donor_ids"]) if fits else None}
        # H1 / H2
        row["turns_no_fence"] = sum(1 for s in steps if (s.get("command") or {}).get("kind") == "error")
        row["turns_multi_block"] = sum(1 for s in steps if (s.get("command") or {}).get("n_blocks", 0) > 1)
        langs = set()
        for s in steps:                                        # opening fences only: every odd-numbered ``` in a reply
            fences = re.findall(r"```(\w*)", s.get("reply") or "")
            langs |= {lang or "(untagged)" for lang in fences[0::2]}
        row["fence_langs"] = " ".join(sorted(langs))
        row["heredoc_unrecognised"] = sum(1 for s in steps if (s.get("command") or {}).get("kind") == "error" and "<<" in (s.get("reply") or ""))
        row["turns_length_cap"] = sum(1 for s in steps if (s.get("usage") or {}).get("finish_reason") == "length")
        row["reasoning_chars_median"] = float(np.median([len(think(s)) for s in steps])) if steps else np.nan
        row["reasoning_chars_max"] = max((len(think(s)) for s in steps), default=0)
        # A1
        row["says_last_month"] = hits(rat, SAYS_LAST_MONTH) or hits(last_think, SAYS_LAST_MONTH)
        # A2
        alt = [f for f in fits if len(f["donor_ids"]) >= 2 or not is_p09(r, f["predictions"])]
        row["alternative_in_own_runs"] = bool(alt) if len(fits) > 1 else None
        row["alternative_had_lower_mse"] = (min(f["fit_mse"] for f in alt) < fits[-1]["fit_mse"] - 1e-12) if alt and fits else None
        # A3
        row["criterion_min_error"] = hits(rat, MIN_ERROR)
        row["criterion_stability"] = hits(rat, STABILITY)
        row["criterion_recency"] = hits(rat, RECENCY)
        row["criterion_simplicity"] = hits(rat, SIMPLICITY)
        # A4 / B
        sp = fits[-1]["spec"] if fits else {}
        row["final_spec"] = json.dumps({k: sp[k] for k in sorted(sp) if k != "band_subset"}) if sp else ""
        if exp == "shc_design" and sp:
            row["why_raw"] = (hits(rat, SHORT_RECORD) or hits(rat, SEASON_NOT) or hits(rat, SCALE) or hits(rat, SIMPLICITY) or hits(rat, MIN_ERROR)) if sp["fit_space"] == "raw" else ""
            row["why_raw_theme"] = ("short record" if hits(rat, SHORT_RECORD) else "seasonality not estimable" if hits(rat, SEASON_NOT) else
                                    "observed scale" if hits(rat, SCALE) else "simplicity" if hits(rat, SIMPLICITY) else
                                    "fit error" if hits(rat, MIN_ERROR) else "not stated") if sp["fit_space"] == "raw" else ""
            row["why_exhaustive"] = hits(rat, EXHAUSTIVE_WHY) if sp["selection"] == "exhaustive" else ""
            row["why_zeta_zero"] = hits(rat, ZETA_ZERO_WHY) if sp["zeta"] == 0 else ""
            row["why_invvar"] = hits(rat, INVVAR_WHY) if sp["band_weighting"] == "inverse_variance" else ""
            if len(fits) > 1:
                f0, f1 = fits[0]["spec"], fits[-1]["spec"]
                row["choices_changed"] = " ".join(k for k in ("fit_space", "weight_constraint", "intercept", "band_weighting", "readd_seasonal", "selection") if f0.get(k) != f1.get(k))
        # C
        if exp == "shc_implement":
            prog = r.get("program")
            if not prog:                                       # program run inline: python - <<'PY' ... PY
                for s in reversed(steps):
                    mm = re.search(r"python3?\s+-\s*<<\s*['\"]?(\w+)['\"]?\n(.*?)\n\1\s*$", (s.get("command") or {}).get("text") or "", re.S)
                    if mm:
                        prog = mm.group(2); row["program_inline"] = True; break
            cls, solver, constraint = program_class(prog)
            if r.get("ok") and len(set(round(v, 6) for v in r["values"].values())) == 1 and len(r["values"]) > 1:
                cls = "bug: one value for every band"           # e.g. a mean taken across bands
            row["program_class"], row["program_solver"], row["program_constraint"] = cls, solver, constraint
            wstep = next((s for s in steps if "predict.py" in ((s.get("command") or {}).get("text") or "") and "cat >" in (s.get("command") or {}).get("text", "")), None)
            row["thinking_simplifies"] = hits(think(wstep), SIMPLIFY) if wstep else ""
            runs = [s for s in steps if re.search(r"python3? ((\./)?predict\.py|-\s*<<)", (s.get("command") or {}).get("text") or "")]
            w = weights_from_output(runs[-1].get("output")) if runs else None
            row["program_max_abs_weight"] = max(abs(x) for x in w) if w else np.nan
            row["program_weight_sum"] = sum(w) if w else np.nan
        # I1
        if len(fits) > 1 and r.get("ok"):
            row["final_is_min_mse"] = fits[-1]["fit_mse"] <= min(f["fit_mse"] for f in fits) + 1e-12
            row["metric_first_run"] = metric_of(r, fits[0]["predictions"])
            row["metric_final"] = r["_metric"]
        # D1
        if r.get("ended") == "call_error":
            row["call_error"] = (steps[-1].get("error") or "")[:200]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report
def frac(n, d):
    return f"{n}/{d}" if d else "—"


def report(T):
    pd.set_option("display.width", 220)
    models = [SHORT[m] for m in SHORT if SHORT[m] in set(T.model)]
    print("\n=== H1. harness friction, all turns, per model (turns with no fence / with extra blocks ignored / unrecognised heredocs; fence languages seen) ===")
    for m in models:
        g = T[T.model == m]
        langs = collections.Counter(l for s in g.fence_langs for l in s.split())
        print(f"  {m:8s} turns {int(g.n_steps.sum()):4d}: no fence {int(g.turns_no_fence.sum()):3d}; extra blocks {int(g.turns_multi_block.sum()):3d}; "
              f"unrecognised heredoc {int(g.heredoc_unrecognised.sum()):2d}; fence tags {dict(langs)}")
    print("\n=== H2. token cap and reasoning length ===")
    for m in models:
        g = T[T.model == m]
        print(f"  {m:8s} turns cut by the cap {int(g.turns_length_cap.sum()):3d} in {int((g.turns_length_cap > 0).sum()):2d} cells; "
              f"reasoning chars per turn median {np.nanmedian(g.reasoning_chars_median):.0f}, max {int(g.reasoning_chars_max.max())}")
    print("\n=== A1. configure, single-block answers: rationale/thinking says the answer is the last month ===")
    for m in models:
        g = T[(T.arm == "configure") & (T.model == m) & T.ok & (T.blocks_kept_final == 1)]
        print(f"  {m:8s} {frac(int((g.says_last_month != '').sum()), len(g))}   e.g. {g[g.says_last_month != ''].says_last_month.head(1).tolist()}")
    print("  misses (single-block cells whose rationale did not match):")
    for _, x in T[(T.arm == "configure") & T.ok & (T.blocks_kept_final == 1) & (T.says_last_month == "")].head(6).iterrows():
        print(f"    {x.model:8s} {x.prompt_id}: {(x.final_spec)[:0]}{str(next((r for r in RECS if r['prompt_id']==x.prompt_id and SHORT.get(r['model'])==x.model), {}).get('rationale'))[:160]!r}")
    print("\n=== A2. configure, cells with >1 fitter run: a non-P09 alternative existed in the model's own runs and P09 was submitted; and it had a lower matching error ===")
    for m in models:
        g = T[(T.arm == "configure") & (T.model == m) & T.ok & (T.fit_runs > 1)]
        a = g[g.alternative_in_own_runs == True]
        print(f"  {m:8s} cells with >1 run {len(g):2d}; alternative existed {len(a):2d}; submitted P09 anyway {int(a.answer_is_p09.sum()):2d}; "
              f"of those the alternative had lower matching error {int((a.answer_is_p09 & (a.alternative_had_lower_mse == True)).sum()):2d}")
    print("\n=== A3. stated criterion in the rationale (multi-label), A and B ===")
    for arm in ("configure", "design"):
        for m in models:
            g = T[(T.arm == arm) & (T.model == m) & T.ok]
            if len(g):
                print(f"  {arm:9s} {m:8s} n={len(g):2d}: min error {int((g.criterion_min_error != '').sum()):2d}, stability {int((g.criterion_stability != '').sum()):2d}, "
                      f"recency {int((g.criterion_recency != '').sum()):2d}, simplicity {int((g.criterion_simplicity != '').sum()):2d}, none of these {int(((g.criterion_min_error == '') & (g.criterion_stability == '') & (g.criterion_recency == '') & (g.criterion_simplicity == '')).sum()):2d}")
    print("\n=== A4. configure: distinct final settings across parcels ===")
    for m in models:
        g = T[(T.arm == "configure") & (T.model == m) & (T.final_spec != "")]
        c = collections.Counter(g.final_spec)
        top, n = c.most_common(1)[0] if c else ("", 0)
        print(f"  {m:8s} {len(c)} distinct settings over {len(g)} cells; most common used in {n}: {top}")
    print("\n=== B1. design: reasons for raw fit space (theme), exhaustive, zeta 0, inverse-variance ===")
    for m in models:
        g = T[(T.arm == "design") & (T.model == m) & T.ok & (T.final_spec != "")]
        if not len(g):
            continue
        raw = g[g.why_raw_theme != ""]
        n_exh = int(g.final_spec.str.contains("exhaustive").sum())
        n_zeta0 = int(g.final_spec.str.contains(r'"zeta": 0\.0[,}]', regex=True).sum())
        n_inv = int(g.final_spec.str.contains("inverse_variance").sum())
        print(f"  {m:8s} raw {len(raw)}/{len(g)}: themes {dict(collections.Counter(raw.why_raw_theme))}; "
              f"exhaustive with a stated reason {frac(int((g.why_exhaustive != '').sum()), n_exh)}; "
              f"zeta 0 with a stated reason {frac(int((g.why_zeta_zero != '').sum()), n_zeta0)}; "
              f"inverse-variance with a reason {frac(int((g.why_invvar != '').sum()), n_inv)}")
    print("  raw cells whose rationale states none of the themes:")
    for _, x in T[(T.arm == "design") & (T.why_raw_theme == "not stated")].head(5).iterrows():
        print(f"    {x.model:8s} {x.prompt_id}")
    print("\n=== B2. design: choices changed between the first and the final fitter run ===")
    for m in models:
        g = T[(T.arm == "design") & (T.model == m) & (T.fit_runs > 1)]
        c = collections.Counter(k for s in g.get("choices_changed", pd.Series(dtype=str)).fillna("") for k in s.split())
        print(f"  {m:8s} cells with >1 run {len(g):2d}; cells where a choice changed {int((g.get('choices_changed', pd.Series(dtype=str)).fillna('') != '').sum()):2d}; fields {dict(c)}")
    print("\n=== C1. implement: program class (by syntax tree) ===")
    for m in models:
        g = T[(T.arm == "implement") & (T.model == m) & T.ok]
        if len(g):
            print(f"  {m:8s} n={len(g)}: {dict(collections.Counter(g.program_class))}; solvers {dict(collections.Counter(g.program_solver[g.program_solver != '']))}; "
                  f"constraints {dict(collections.Counter(g.program_constraint[g.program_constraint != '']))}")
    print("\n=== C2. implement: blow-ups (metric > 5): class, constraint, weights printed by the program ===")
    for _, x in T[(T.arm == "implement") & (T.metric > 5)].sort_values("metric", ascending=False).iterrows():
        print(f"  {x.model:8s} {x.prompt_id:30s} metric {x.metric:8.1f}  {x.program_class:26s} {x.program_solver:8s} {x.program_constraint:13s} "
              f"max|w| {x.program_max_abs_weight:7.2f}  sum w {x.program_weight_sum:6.2f}")
    print("\n=== C3. implement: thinking at the moment predict.py is written states a simplification ===")
    for m in models:
        g = T[(T.arm == "implement") & (T.model == m) & T.ok]
        if len(g):
            print(f"  {m:8s} {frac(int((g.thinking_simplifies != '').sum()), len(g))}  e.g. {g[g.thinking_simplifies != ''].thinking_simplifies.head(1).tolist()}")
    print("\n=== I1. cells with >1 fitter run: final run is the minimum matching error; P10 error first run -> final ===")
    for arm in ("configure", "design"):
        for m in models:
            g = T[(T.arm == arm) & (T.model == m) & T.ok & (T.fit_runs > 1)]
            if len(g):
                print(f"  {arm:9s} {m:8s} n={len(g):2d} (median runs {int(g.fit_runs.median())}): final = min error {frac(int(g.final_is_min_mse.sum()), len(g))}; "
                      f"P10 error mean {g.metric_first_run.mean():.3f} -> {g.metric_final.mean():.3f}; better {int((g.metric_final < g.metric_first_run - 1e-12).sum())}, worse {int((g.metric_final > g.metric_first_run + 1e-12).sum())}")
    print("\n=== D1. DeepSeek: how cells ended; fitter runs per cell; gateway error reasons ===")
    g = T[T.model == "DeepSeek"]
    for arm in ("configure", "design", "implement"):
        h = g[g.arm == arm]
        hr = h[h.fit_runs > 0]
        print(f"  {arm:9s} ended {dict(collections.Counter(h.ended))}; cells with fitter runs {len(hr)}, runs per such cell median {hr.fit_runs.median() if len(hr) else 0:.0f}, max {int(h.fit_runs.max()) if len(h) else 0}")
    for e, n in collections.Counter(g.get("call_error", pd.Series(dtype=str)).dropna()).most_common(5):
        print(f"    {n:2d}x {e[:160]}")


# ----------------------------------------------------------------------------- workbooks
def workbooks(recs):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment
    out = lb.LB / "trace_workbooks"; out.mkdir(exist_ok=True)
    by_model = collections.defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)
    for m, rs in by_model.items():
        wb = Workbook(); wb.remove(wb.active)
        for exp in ARMS:
            cells = sorted((r for r in rs if r["experiment"] == exp), key=lambda r: r["prompt_id"])
            if not cells:
                continue
            ws = wb.create_sheet(exp[4:])
            note = "" if any(think(s) for r in cells for s in r["steps"]) else " (no reasoning text is returned by this gateway; thinking columns are empty)"
            ws.append([f"{SHORT.get(m, m)} — {exp}{note}"])
            nmax = max(r["n_steps"] for r in cells)
            ws.append(["cell", "ok", "metric"] + [f"turn {k}" for k in range(1, nmax + 1)] + ["answer", "rationale", "fitter runs"])
            for r in cells:
                row = [r["prompt_id"], r["ok"], round(r["_metric"], 4) if r["ok"] else ""]
                for s in r["steps"]:
                    c = s.get("command") or {}
                    row.append(f"[thinking] {think(s)[:600]}\n[command] {(c.get('text') or c.get('message') or '')[:400]}\n[output] {(s.get('output') or '')[:400]}")
                row += [""] * (nmax - r["n_steps"])
                fits = "\n".join(f"{i}: {json.dumps({k: v for k, v in f['spec'].items() if k != 'band_subset'})} -> blocks {f['donor_ids']} w {[round(x, 3) for x in f['weights']]} mse {f['fit_mse']:.4g}"
                                 for i, f in enumerate(r.get("fit_runs") or [], 1))
                row += [json.dumps(r.get("values")), r.get("rationale") or "", fits or (r.get("program") or "")[:3000]]
                ws.append(row)
            for col in ws.columns:
                for c in col:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
            for i in range(4, nmax + 7):
                ws.column_dimensions[ws.cell(row=2, column=i).column_letter].width = 60
            ws.column_dimensions["A"].width = 30
        wb.save(out / f"skills_{lb.safe_name(m)}.xlsx")
        print(f"  wrote trace_workbooks/skills_{lb.safe_name(m)}.xlsx ({len(rs)} cells)")


RECS = []


def main():
    global RECS
    RECS = load()
    T = analyse(RECS)
    T.to_csv(lb.LB / "lb_skill_traces.csv", index=False)
    print(f"{len(T)} cells -> lb_skill_traces.csv")
    report(T)
    print("\n=== workbooks ===")
    workbooks(RECS)


if __name__ == "__main__":
    main()
