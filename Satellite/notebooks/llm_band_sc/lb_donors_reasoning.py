"""Donors, search path and answer-time reasoning of the simple SHC skills (shc-configure, shc-design), monthly
(notebook 10) and daily (notebook 14). Plan ~/.claude/plans/quirky-growing-wombat.md (2026-10-23).

    python lb_donors_reasoning.py

Every answer is described by the fit behind it: the fit whose predictions, rounded to the answer's decimals, equal the
answer (the last such fit). Daily reads it from lb_match_skill_daily.csv; monthly computes it here with the same rule
(departure 1) and writes lb_match_skill_monthly.csv.

A fit is placed at the step whose output shows its "blocks kept:" line, rebuilt from the stored fit exactly as the
fitter prints it (skills*/shc-*/scripts/fit.py). Fits are stored only when the fitter's record line reaches stdout, so a
fit whose output the model cut (| head, | grep) is shown in the step but not stored; those lines are counted.

Writes
    lb_match_skill_monthly.csv        one row per monthly record, the daily file's columns
    lb_donors_simple_monthly.csv      the fit behind each monthly answer, and notebook 10's block per case (departure 3)
    lb_donors_simple_daily.csv        the fit behind each daily answer, and notebook 14's Table 1 row per case (departure 2)
    lb_search_path_simple.csv         every stored fit of every answered case
    lb_answer_reasoning_simple.csv    reasoning and replies from the step that wrote the answer's spec to the answer step
"""
import lb_one_thread  # noqa: F401  first import: one thread
import json
import re
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import lb_skill_score as monthly_score
from lb_skill_score_daily import matches as matches_daily

LB = Path(__file__).resolve().parent
MODELS = [("gpt-oss-120b", "gpt-oss-120b"), ("GLM-5.3", "GLM-5.3"),
          ("DeepSeek-V4.1-Flash-thinking-low", "DeepSeek-V4.1-Flash-thinking-low"), ("openai:gpt-5.5", "openai_gpt-5.5")]
SKILLS = ("configure", "design")
CHOICES = ["fit_space", "weight_constraint", "intercept", "band_weighting", "selection", "readd_seasonal"]
SETTINGS = {"daily": ["time_bandwidth_candidates", "m_min", "m_step", "m_max", "max_donors", "nu", "zeta"],
            "monthly": ["m", "n", "kernel_bandwidth", "zeta", "beta_trend_degree"]}
# notebook values: lb_her_nb14.py lines 32-43 (TIME_BW_CANDIDATES, M_MIN, M_STEP, M_MAX_USER, MAX_DONORS, NU_VARIANCE_SHARE,
# ZETA_VARIANCE_SHARE); lb_her_nb10.py M, KERNEL_BANDWIDTH, ZETA, BETA_TREND_DEGREE and n = 1 of its held-out blocks (In [6])
NOTEBOOK = {"daily": {"time_bandwidth_candidates": [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0], "m_min": 10, "m_step": 5,
                      "m_max": None, "max_donors": 60, "nu": 0.1, "zeta": 0.001},
            "monthly": {"m": 4, "n": 1, "kernel_bandwidth": 2.0, "zeta": 1e-4, "beta_trend_degree": 1}}
NB_CHOICES = {"fit_space": "latent_std", "weight_constraint": "simplex", "intercept": "none", "band_weighting": "equal",
              "selection": "greedy_forward", "readd_seasonal": False}
KEPT = re.compile(r"^blocks kept: .*$", re.M)
FIT_HEAD = {"daily": re.compile(r"^visits given: \d+; visits to predict", re.M),
            "monthly": re.compile(r"^blocks available: \d+ \(m = ", re.M)}


def check_notebook_values():
    t14, t10 = (LB / "lb_her_nb14.py").read_text(), (LB / "lb_her_nb10.py").read_text()
    for pat in (r"TIME_BW_CANDIDATES = \[15\.0, 30\.0, 45\.0, 60\.0, 90\.0, 120\.0, 180\.0\]", r"M_MIN = 10\b", r"M_STEP = 5\b",
                r"M_MAX_USER = None", r"MAX_DONORS = 60\b", r"NU_VARIANCE_SHARE = 0\.1\b", r"ZETA_VARIANCE_SHARE = 0\.001\b"):
        assert re.search(pat, t14), pat
    for pat in (r"\nM = 4\n", r"KERNEL_BANDWIDTH = 2\.0", r"ZETA = 1e-4", r"BETA_TREND_DEGREE = 1\b"):
        assert re.search(pat, t10), pat


def last_records(path):
    d = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            d[r["prompt_id"]] = r
    return d


def record_file(data, skill, fname):
    return LB / (f"lb_replies_daily_shc_{skill}_{fname}_low.jsonl" if data == "daily" else f"lb_replies_shc_{skill}_{fname}_low.jsonl")


def cmd_text(step):
    c = step.get("command") or {}
    return (c.get("text") or c.get("message") or "") if isinstance(c, dict) else str(c)


def reasoning(step):
    rs = (step.get("usage") or {}).get("reasoning")
    return rs if isinstance(rs, str) else ""


# ----------------------------------------------------------------------------- the fit behind the answer
def ndec(v):
    return max(0, -Decimal(repr(v)).as_tuple().exponent)


def matches_monthly(fit, values):
    """The daily rule on the monthly shape: same bands, and every prediction rounded to the answer's decimals equals it."""
    preds = fit.get("predictions") or {}
    if set(preds) != set(values):
        return False
    return all(round(float(v), ndec(values[b])) == values[b] for b, v in preds.items())


def monthly_match(recs):
    rows = []
    for (skill, key), d in recs.items():
        for r in d.values():
            fits = r["fit_runs"]
            base = {"row": f"shc_{skill}|{key}", "site_id": r["site_id"], "sensor": r["sensor"], "prompt_id": r["prompt_id"],
                    "ok": bool(r["ok"]), "ended": r.get("ended")}
            if not r["ok"]:
                rows.append({**base, "n_fit_runs": len(fits), "n_matching": None, "matched_run": None, "engine": None})
                continue
            hit = [i for i, f in enumerate(fits) if matches_monthly(f, r["values"])]
            rows.append({**base, "n_fit_runs": len(fits), "n_matching": len(hit), "matched_run": hit[-1] + 1 if hit else None,
                         "engine": fits[hit[-1]]["engine_used"] if hit else None})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- blocks as the fitter prints them
class DailyCase:
    """Visit numbers and dates of one daily case, and notebook 14's block rule (lb_her_nb14.py:419-470)."""

    def __init__(self, cases, sensor, site):
        c = cases[(cases.sensor == sensor) & (cases.site_id == site)]
        pre = c[c.part == "pre"][["period_id", "date"]].drop_duplicates().sort_values("date")
        post = c[c.part == "post"][["period_id", "date"]].drop_duplicates().sort_values("date")
        self.pid, self.date = list(pre.period_id), dict(zip(c.period_id, c.date))
        self.T0, self.n = len(pre), len(post)

    def starts(self, m, max_donors):
        N = self.T0 - m - self.n + 1
        assert N >= 1, (self.T0, m, self.n)
        s = list(range(N))
        if max_donors is not None and N > max_donors:
            s = sorted(set(int(round(v)) for v in np.linspace(0, N - 1, max_donors)))
        return s

    def block(self, i, m, max_donors):
        """(kept-line text, pre start date, pre end date, forward start date, forward end date) of donor i."""
        s = self.starts(m, max_donors)[i - 1]
        p = self.pid
        text = f"block {i} ({p[s]}-{p[s + m - 1]} -> {p[s + m]}-{p[s + m + self.n - 1]})"
        return text, self.date[p[s]], self.date[p[s + m - 1]], self.date[p[s + m]], self.date[p[s + m + self.n - 1]]


def monthly_block(i, m, n):
    """Donor i of notebook 10's make_historical_blocks_from_periods(P01-P09, m, n), printed as the monthly fitter prints it."""
    assert 1 <= i <= 9 - m - n + 1
    return f"block {i} (P{i:02d}-P{i + m - 1:02d} -> P{i + m:02d})"


def kept_line(data, fr, case):
    if data == "daily":
        return "blocks kept: " + "; ".join(case.block(i, fr["m_star"], fr["spec"]["max_donors"])[0] for i in fr["donor_ids"])
    return "blocks kept: " + "; ".join(monthly_block(i, fr["spec"]["m"], fr["spec"]["n"]) for i in fr["donor_ids"])


def weights_line(fr):
    return "weights: " + ", ".join(f"{w:.4f}" for w in fr["weights"]) + (
        f"; intercept {fr['intercept']:.4f}" if fr["spec"].get("intercept") == "fitted" else "")


def place_fits(r, data, case):
    """Step of each stored fit: the earliest step at or after the previous fit's step with an unused copy of its kept line."""
    outs = [s.get("output") or "" for s in r["steps"]]
    avail = [Counter(KEPT.findall(o)) for o in outs]
    steps, prev, weights_shown = [], 0, []
    for fr in r["fit_runs"]:
        line = kept_line(data, fr, case)
        at = next((i for i in range(prev, len(outs)) if avail[i][line] > 0), None)
        if at is not None:
            avail[at][line] -= 1
            prev = at
            weights_shown.append(weights_line(fr) in outs[at])
        else:
            weights_shown.append(None)
        steps.append(None if at is None else r["steps"][at]["step"])
    unstored = sum(sum(c.values()) for c in avail)              # kept lines left over: fits shown but not stored
    return steps, weights_shown, unstored


# ----------------------------------------------------------------------------- settings and choices
def same(a, b):
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if a is None or b is None or isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a is b or a == b and type(a) is type(b)
    return float(a) == float(b)


def show(v):
    return json.dumps(v) if not isinstance(v, str) else v


# ----------------------------------------------------------------------------- notebook rows
def notebook10_blocks():
    """Departure 3: notebook 10 selects one block per case (In [15], mean_selected_donors 1.000000 at m = 4). Where its
    printed per-case error (In [14]) equals the error of the following month of a block, that block is the selection."""
    base = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(LB / "lb_prompts_shc.jsonl")}
    her = pd.read_csv(LB / "lb_her_nb10_results.csv")
    rows = []
    for x in her.itertuples():
        b = base[f"{'s1' if x.sensor == 'sentinel1' else 's2'}_{x.site_id}_shc"]
        hits = []
        for i in range(1, 6):                                  # m = 4, n = 1: donors 1-5, following months P05-P09
            pred = {d: v[i + 3] for d, v in b["train"].items()}
            err = monthly_score.joint_validation_nmse(monthly_score.her_heldout_rows(b, pred))
            if round(err, 6) == round(float(x.her_joint_validation_nmse), 6):
                hits.append(i)
        rows.append({"sensor": x.sensor, "site_id": x.site_id, "printed_error": x.her_joint_validation_nmse,
                     "block": hits[0] if len(hits) == 1 else None, "n_blocks_matching": len(hits)})
    return pd.DataFrame(rows)


def notebook14_rows(cases):
    """Departure 2: Table 1 as printed, with dates of the printed donors from notebook 14's block rule at its m*."""
    t = pd.read_csv(LB / "lb_her14_printed_table1.csv")
    rows = []
    for x in t.itertuples():
        case = DailyCase(cases, x.sensor, x.site_id)
        assert (case.T0, case.n) == (x.T0_pre, x.n_post), (x.site_id, case.T0, case.n)
        assert len(case.starts(x.m_star, 60)) == x.N_donors_available, (x.site_id, x.m_star)
        ids = [int(tok.split(":")[0][1:]) for tok in x.donor_weights.split("|") if not tok.startswith("+")]
        blocks = [case.block(i, x.m_star, 60) for i in ids]
        rows.append({"data": "daily", "skill": "notebook 14", "model": "", "sensor": x.sensor, "site_id": x.site_id,
                     "m": x.m_star, "blocks_available": x.N_donors_available, "J": x.J_star,
                     "donors": x.donor_weights, "blocks": "; ".join(b[0] for b in blocks),
                     "block_dates": "; ".join(f"{b[1]}..{b[2]} -> {b[3]}..{b[4]}" for b in blocks),
                     "max_weight": x.max_weight, "selection_mode": x.selection_rule})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- trace cases
def representative_cases(path, reason, donors):
    """Two cases per data set, one per sensor (Jun 2026-10-23: chosen after the reasoning analysis). A case is typical for
    a skill and model when its number of fits and its answer-time text length (replies for GPT-5.5) lie within that
    group's interquartile range, its answer comes from the fit the group mostly answers from (last or earlier), and,
    monthly, it keeps the most recent block as the group mostly does. Per sensor the case typical for the most groups is
    chosen, then the one answered by the most groups, then the one nearest the group medians."""
    nf = path.groupby(["data", "skill", "model", "prompt_id"]).fit.max().rename("nfits")
    t = reason.set_index(["data", "skill", "model", "prompt_id"]).join(nf).join(
        donors.set_index(["data", "skill", "model", "prompt_id"])[["keeps_most_recent"]]).reset_index()
    t["L"] = np.where(t.model == "openai:gpt-5.5", t.replies.fillna("").str.len(), t.reasoning_chars)
    t["last"] = t.matched_run == t.nfits
    parts = []
    for _, g in t.groupby(["data", "skill", "model"]):
        g = g.copy()
        ok = g["last"] == g["last"].mode()[0]
        dist = 0
        for c in ("nfits", "L"):
            q1, q3 = g[c].quantile(.25), g[c].quantile(.75)
            ok &= g[c].between(q1, q3)
            dist = dist + (g[c] - g[c].median()).abs() / max(q3 - q1, 1)
        if g.data.iat[0] == "monthly":
            ok &= g.keeps_most_recent == g.keeps_most_recent.mode()[0]
        g["typical"], g["dist"] = ok, dist
        parts.append(g)
    s = pd.concat(parts).groupby(["data", "sensor", "site_id"]).agg(
        typical=("typical", "sum"), answered=("prompt_id", "size"), dist=("dist", "sum")).reset_index()
    s = s.sort_values(["data", "sensor", "typical", "answered", "dist"], ascending=[True, True, False, False, True])
    chosen = s.groupby(["data", "sensor"]).head(1)
    print("trace cases\n" + chosen.to_string(index=False))
    return [(x.data, x.sensor, x.site_id) for x in chosen.itertuples()]


VISIT = re.compile(r"^\s*P\d+ = \d{4}-\d{2}(-\d{2})?: ")
CSV_ROW = re.compile(r"^[^,\s][^,]*(,[^,]*){3,}$")
JSON_VISIT = re.compile(r'[ \t]*"P\d+":\s*\{[^{}]*\},?[ \t]*\n?')
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def count_word(n):
    return WORDS.get(n, str(n))


def skill_files(data, skill):
    root = LB / ("skills_daily" if data == "daily" else "skills") / f"shc-{skill}"
    files = {str(f.relative_to(root)): f.read_text() for f in sorted(root.rglob("*")) if f.is_file() and f.suffix in (".md", ".py")}
    lines = {}
    for name, t in files.items():
        for l in t.split("\n"):
            if len(l.strip()) >= 8:
                lines.setdefault(l.strip(), name)
    return files, lines


def collapse_runs(lines, is_item, label):
    """Runs of five or more listing lines keep their first two and last two lines and the count omitted."""
    out, i = [], 0
    while i < len(lines):
        j = i
        while j < len(lines) and is_item(lines[j]):
            j += 1
        if j - i >= 5:
            out += lines[i:i + 2] + [f"[{j - i - 4} {label} omitted]"] + lines[j - 2:j]
        else:
            out += lines[i:max(j, i + 1)]
        i = max(j, i + 1)
    return out


def structured_output(out, files, flines):
    """Departure 4 (Jun yes 2026-10-23): the output reduced by its structure; nothing else is cut."""
    for name, t in files.items():
        t = t.strip("\n")
        if t.count("\n") >= 4 and t in out:
            out = out.replace(t, f"[{name}, {t.count(chr(10)) + 1} lines, the skill's own file]")
    lines, res, i = out.split("\n"), [], 0
    while i < len(lines):                                    # the skill's files printed in part
        j, names = i, []
        while j < len(lines) and (lines[j].strip() in flines or (not lines[j].strip() and j > i)):
            if lines[j].strip():
                names.append(flines[lines[j].strip()])
            j += 1
        while j > i and not lines[j - 1].strip():
            j -= 1
        if j - i >= 5:
            res.append(f"[{j - i} lines of the skill's own files: {', '.join(sorted(set(names)))}]")
            i = j
        else:
            res.append(lines[i])
            i += 1
    res = collapse_runs(res, lambda l: bool(VISIT.match(l)), "visit lines")
    res = collapse_runs(res, lambda l: bool(CSV_ROW.match(l)), "data rows")
    text = "\n".join(res)

    def json_run(m):
        items = JSON_VISIT.findall(m.group(0))
        if len(items) < 5:
            return m.group(0)
        return "".join(items[:2]) + f"  [{len(items) - 4} visits omitted]\n" + "".join(items[-2:])
    return re.sub(r"(?:[ \t]*\"P\d+\":\s*\{[^{}]*\},?[ \t]*\n?){5,}", json_run, text)


def describe_fits(fits, data):
    """The settings that vary across the fits of one step, in the form of Jun's example (2026-10-23)."""
    desc = []
    for f in SETTINGS[data] + CHOICES + ["band_subset"]:
        vals = []
        for fr in fits:
            v = fr["spec"].get(f)
            if not any(same(v, u) for u in vals):
                vals.append(v)
        if len(vals) < 2:
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer() for v in vals):
            iv = sorted(int(v) for v in vals)
            if len(iv) >= 3 and iv == list(range(iv[0], iv[-1] + 1)):
                desc.append(f"{f} {iv[0]} to {iv[-1]}")
                continue
        try:
            vals = sorted(vals, key=lambda v: (float(v) if isinstance(v, (int, float)) else str(v)))
        except TypeError:
            pass
        desc.append(f"{f} {count_word(len(vals))} values ({', '.join(show(v) for v in vals)})")
    return ", ".join(desc) if desc else "no setting varies"


def write_traces(recs, cases, chosen):
    fence = "~~~~"
    md = ["# Step traces of the chosen cases", ""]
    for data, sensor, site in chosen:
        case = DailyCase(cases, sensor, site) if data == "daily" else None
        for skill in SKILLS:
            files, flines = skill_files(data, skill)
            for key, _ in MODELS:
                r = next(x for x in recs[data][(skill, key)].values() if x["sensor"] == sensor and x["site_id"] == site)
                steps, _, _ = place_fits(r, data, case)
                fits_at = defaultdict(list)
                for fr, st in zip(r["fit_runs"], steps):
                    if st is not None:
                        fits_at[st].append(fr)
                ans = [i for i, f in enumerate(r["fit_runs"]) if r["ok"] and
                       (matches_daily(f, r["values"]) if data == "daily" else matches_monthly(f, r["values"]))]
                head = (f"## {data} · {sensor} {site} · shc-{skill} · {key}")
                behind = (f"answer from fit {ans[-1] + 1} of {len(r['fit_runs'])}, placed at step {steps[ans[-1]]}" if ans
                          else ("answer matches no stored fit" if r["ok"] else "no answer"))
                md += [head, "", f"ended {r.get('ended')} · {len(r['steps'])} steps · {behind}", ""]
                for s in r["steps"]:
                    out = s.get("output") or ""
                    shown = max(len(fits_at[s["step"]]), len(FIT_HEAD[data].findall(out)), len(KEPT.findall(out)))
                    if shown > 3:
                        stored = len(fits_at[s["step"]])
                        out_s = (f"[{shown} fits" + (f", {stored} of them stored" if stored != shown else "") +
                                 f"; varying {describe_fits(fits_at[s['step']], data) if stored else 'settings not stored'}"
                                 "; one row each in the search path table]")
                    else:
                        out_s = structured_output(out, files, flines)
                    cmd = cmd_text(s)
                    rs = "not logged" if key == "openai:gpt-5.5" else (reasoning(s) or "(empty)")
                    md += [f"### Step {s['step']} · exit {s.get('exit')}", "", "Reasoning", fence, rs, fence,
                           "Reply", fence, s.get("reply") or "", fence]
                    md += (["Command: the code block in the reply", ""] if cmd and cmd in (s.get("reply") or "")
                           else ["Command", fence, cmd, fence])
                    md += ["Output", fence, out_s, fence, ""]
                md += ["Rationale", fence, r.get("rationale") or "", fence, ""]
    (LB / "lb_traces_representative.md").write_text("\n".join(md) + "\n")


# ----------------------------------------------------------------------------- main
def main():
    check_notebook_values()
    cases = pd.read_csv(LB / "lb_cases_daily.csv")
    recs = {data: {(skill, key): last_records(record_file(data, skill, fname)) for skill in SKILLS for key, fname in MODELS}
            for data in ("monthly", "daily")}

    # what is logged
    logged = []
    for data in ("monthly", "daily"):
        for (skill, key), d in recs[data].items():
            steps = [s for r in d.values() for s in r["steps"]]
            lens = [len(reasoning(s)) for s in steps if reasoning(s)]
            logged.append({"data": data, "skill": skill, "model": key, "records": len(d), "steps": len(steps),
                           "steps_with_reasoning": len(lens), "median_chars": statistics.median(lens) if lens else None})
    logged = pd.DataFrame(logged)

    # the fit behind each answer
    mm = monthly_match(recs["monthly"])
    mm.to_csv(LB / "lb_match_skill_monthly.csv", index=False)
    md = pd.read_csv(LB / "lb_match_skill_daily.csv")
    for (skill, key), d in recs["daily"].items():                 # the daily file reproduces with the daily rule
        for x in md[(md.row == f"daily_shc_{skill}|{key}") & (md.ok == True)].itertuples():
            r = d[x.prompt_id]
            hit = [i for i, f in enumerate(r["fit_runs"]) if matches_daily(f, r["values"])]
            assert (hit[-1] + 1 if hit else None) == (None if pd.isna(x.matched_run) else int(x.matched_run)), x.prompt_id
    match = {"monthly": mm, "daily": md}

    donors, path, reason, unstored, placed = [], [], [], Counter(), Counter()
    for data in ("monthly", "daily"):
        m = match[data]
        for (skill, key), d in recs[data].items():
            rowkey = f"{'daily_' if data == 'daily' else ''}shc_{skill}|{key}"
            answered = m[(m.row == rowkey) & (m.ok == True) & m.matched_run.notna()]
            for x in answered.itertuples():
                r = d[x.prompt_id]
                case = DailyCase(cases, r["sensor"], r["site_id"]) if data == "daily" else None
                steps, wshown, left = place_fits(r, data, case)
                unstored[(data, skill, key)] += left
                placed[(data, skill, key, "placed")] += sum(s is not None for s in steps)
                placed[(data, skill, key, "fits")] += len(steps)
                placed[(data, skill, key, "weights_also_shown")] += sum(w is True for w in wshown)
                k = int(x.matched_run) - 1
                fr = r["fit_runs"][k]
                spec = fr["spec"]
                saw = any("HER_SPEC" in (s.get("output") or "") for s in r["steps"])
                nb = NOTEBOOK[data]
                if data == "daily":
                    mstar, maxd = fr["m_star"], spec["max_donors"]
                    blocks = [case.block(i, mstar, maxd) for i in fr["donor_ids"]]
                    n_blocks = len(case.starts(mstar, maxd))
                    bdates = "; ".join(f"{b[1]}..{b[2]} -> {b[3]}..{b[4]}" for b in blocks)
                    btext = "; ".join(b[0] for b in blocks)
                else:
                    mstar, n_blocks = spec["m"], 9 - spec["m"] - spec["n"] + 1
                    btext, bdates = "; ".join(monthly_block(i, spec["m"], spec["n"]) for i in fr["donor_ids"]), ""
                donors.append({"data": data, "skill": skill, "model": key, "sensor": r["sensor"], "site_id": r["site_id"],
                               "prompt_id": r["prompt_id"], "matched_run": k + 1, "engine": fr["engine_used"],
                               "m": mstar, "n": spec.get("n", case.n if case else None), "blocks_available": n_blocks,
                               "J": len(fr["donor_ids"]), "donor_ids": " ".join(map(str, fr["donor_ids"])),
                               "weights": " ".join(f"{w:.4f}" for w in fr["weights"]), "max_weight": max(fr["weights"]),
                               "blocks": btext, "block_dates": bdates,
                               "keeps_most_recent": (n_blocks in fr["donor_ids"]) if data == "monthly" else None,
                               "selection_mode": fr.get("selection_mode"), "fit_mse": fr["fit_mse"],
                               "saw_notebook_settings": saw,
                               "settings_equal_notebook": all(same(spec.get(f), nb[f]) for f in SETTINGS[data]),
                               **{f"spec_{f}": show(spec.get(f)) for f in SETTINGS[data] + CHOICES + ["band_subset"]}})
                prev = None
                for j, (f, st) in enumerate(zip(r["fit_runs"], steps)):
                    sp = f["spec"]
                    changed = "" if prev is None else "; ".join(
                        f"{a}: {show(prev.get(a))} -> {show(sp.get(a))}" for a in SETTINGS[data] + CHOICES + ["band_subset"]
                        if not same(prev.get(a), sp.get(a)))
                    path.append({"data": data, "skill": skill, "model": key, "sensor": r["sensor"], "site_id": r["site_id"],
                                 "prompt_id": r["prompt_id"], "fit": j + 1, "step": st, "file": f["file"],
                                 "changed_from_previous": changed, "m": f.get("m_star", sp.get("m")),
                                 "J": len(f["donor_ids"]), "fit_mse": f["fit_mse"], "engine": f["engine_used"],
                                 "behind_answer": j == k, **{f"spec_{a}": show(sp.get(a)) for a in SETTINGS[data] + CHOICES}})
                    prev = sp
                # answer-time reasoning: from the spec-writing step of the fit behind the answer to the answer step
                names = {fr["file"], Path(fr["file"]).name}
                cmds = [cmd_text(s) for s in r["steps"]]
                ans = max((i for i, c in enumerate(cmds) if "answer.json" in c), default=None)
                fit_at = next((i for i, s in enumerate(r["steps"]) if s["step"] == steps[k]), None)
                upto = fit_at if fit_at is not None else ans
                start = max((i for i in range(0, (upto if upto is not None else -1) + 1) if any(nm in cmds[i] for nm in names)),
                            default=upto)
                end = ans if ans is not None else upto
                win = r["steps"][start:end + 1] if start is not None and end is not None and end >= start else []
                gpt = key == "openai:gpt-5.5"
                reason.append({"data": data, "skill": skill, "model": key, "sensor": r["sensor"], "site_id": r["site_id"],
                               "prompt_id": r["prompt_id"], "matched_run": k + 1, "fit_step": steps[k],
                               "window": f"{win[0]['step']}-{win[-1]['step']}" if win else "",
                               "reasoning": "not logged" if gpt else "\n".join(f"[step {s['step']}] {reasoning(s)}" for s in win if reasoning(s)),
                               "replies": "\n".join(f"[step {s['step']}] {s.get('reply') or ''}" for s in win),
                               "rationale": r.get("rationale") or "",
                               "reasoning_chars": 0 if gpt else sum(len(reasoning(s)) for s in win)})
    donors = pd.DataFrame(donors)
    nb14 = notebook14_rows(cases)
    nb10 = notebook10_blocks()
    nb10_rows = pd.DataFrame([{"data": "monthly", "skill": "notebook 10", "model": "", "sensor": x.sensor, "site_id": x.site_id,
                               "m": 4, "n": 1, "blocks_available": 5, "J": 1,
                               "donor_ids": "" if pd.isna(x.block) else str(int(x.block)),
                               "blocks": "not identified" if pd.isna(x.block) else monthly_block(int(x.block), 4, 1),
                               "keeps_most_recent": None if pd.isna(x.block) else int(x.block) == 5,
                               "printed_error": x.printed_error} for x in nb10.itertuples()])
    pd.concat([nb10_rows, donors[donors.data == "monthly"]], ignore_index=True).to_csv(LB / "lb_donors_simple_monthly.csv", index=False)
    pd.concat([nb14, donors[donors.data == "daily"]], ignore_index=True).to_csv(LB / "lb_donors_simple_daily.csv", index=False)
    pd.DataFrame(path).to_csv(LB / "lb_search_path_simple.csv", index=False)
    pd.DataFrame(reason).to_csv(LB / "lb_answer_reasoning_simple.csv", index=False)

    chosen = representative_cases(pd.DataFrame(path), pd.DataFrame(reason), donors)
    write_traces(recs, cases, chosen)

    # checks and summaries
    persist = sum(int(x.block == 5) for x in nb10.itertuples() if not pd.isna(x.block))
    print("notebook 10 blocks:", nb10.block.value_counts(dropna=False).to_dict(), "| matching P09 carried forward:", persist)
    print(nb10[nb10.n_blocks_matching != 1].to_string(index=False) if (nb10.n_blocks_matching != 1).any() else "every case has one matching block")
    print("\nreasoning logged\n" + logged.to_string(index=False))
    print("\nmonthly match: answered with a matching fit per row")
    print(mm[mm.ok].groupby("row").agg(ok=("ok", "sum"), matched=("matched_run", lambda s: int(s.notna().sum()))).to_string())
    print("\nfit placement per row: fits, placed at a step, weights line also in that step, kept lines shown but not stored")
    for (data, skill, key) in sorted({k[:3] for k in placed}):
        print(f"  {data:7s} {skill:9s} {key[:12]:12s} fits {placed[(data, skill, key, 'fits')]:4d} placed {placed[(data, skill, key, 'placed')]:4d}"
              f" weights {placed[(data, skill, key, 'weights_also_shown')]:4d} unstored {unstored[(data, skill, key)]:4d}")
    return donors, logged


if __name__ == "__main__":
    main()
