"""lb_skills_build_daily — generate the two daily SHC skills under skills_daily/ (plan ~/.claude/plans/simple-skills-daily.md,
step 7). The texts are the monthly texts of lb_skills_build.py with each monthly word, number and column replaced
(Decisions row "skill texts"), the seasonal sentences removed, and the settings and choices of rows C1 to C3.

    skills_daily/shc-configure/   the model decides the seven settings; the six choices are fixed in scripts/fit.py
    skills_daily/shc-design/      the model decides settings and choices; scripts/fit.py takes the full spec

Every text below is what the model reads, byte for byte. The scripts directory carries the package modules the
fitter needs (lb_her_nb14, lb_spec_daily, lb_fit_daily) as VERBATIM copies plus a small lb_lib shim, so no fitting
logic is duplicated: gate DK6 checks each copy against the package file. skills_daily/MANIFEST.json records the
sha256 of every generated file.

    python lb_skills_build_daily.py           (re)generate skills_daily/ and MANIFEST.json
    python lb_skills_build_daily.py --check   compare skills_daily/ with the manifest and the package (gate DK6)
"""
import hashlib
import json
import shutil
import sys

import lb_lib as lb
import lb_spec_daily as S

SKILLS = lb.LB / "skills_daily"
MANIFEST = SKILLS / "MANIFEST.json"
BUNDLED = ["lb_her_nb14.py", "lb_spec_daily.py", "lb_fit_daily.py"]   # verbatim copies

# ----------------------------------------------------------------------------- texts
METHOD_MD = """# Synthetic historical control for one parcel

The parcel's own past is its comparison. Nothing from any other parcel is used.

## Historical blocks
From the visits given, form overlapping blocks. A block is m consecutive visits (its
matching part) followed by the n visits that came after it (its forward part), where n is
the number of visits to predict. Blocks slide one visit at a time, so there are
(visits given) − m − n + 1 of them; when there are more than max_donors, that many are kept,
spread evenly over the history. The last block's forward part ends at the last visit given.
The matching window is the parcel's most recent m visits.

## What is matched
Each band's visits can be matched as observed, or after preparation: the series is
smoothed along time, measured in days since the first visit, with a local-linear Gaussian
kernel whose bandwidth is chosen for each band by leave-one-out cross-validation among the
candidate bandwidths, and the smoothed series is standardized to mean 0 and standard
deviation 1 over the visits given. Bands are stacked, so one set of block weights serves
every band.

## Matching window length
The matching error R is the mean over bands of each band's mean squared matching error
divided by the variance of the band's values over the visits given. For each m from m_min
to m_max in steps of m_step, weights are fitted on all blocks, and the longest m whose R
is at most nu is used. If no m passes, the longest m tried is used.

## Weights and selection
Weights over the blocks are fitted so the weighted combination of the blocks' matching
parts reproduces the matching window, across all bands jointly, minimizing mean squared
error. A selection rule decides which blocks carry weight: stepwise (keep the single
best-fitting block, then add one block at a time while R falls by more than a
tolerance) or exhaustive (every non-empty subset, lowest R wins).

## Prediction
Each visit to predict is predicted as the same weighted combination of what followed each
kept block at the same position of its forward part — the k-th visit to predict from the
k-th visit of each block's forward part, as observed — or, when the fit was made on the
prepared series, in the prepared space with the standardization undone and the seasonal
term added back, which is zero for this method.

## What the fit can and cannot see
The fit sees the visits given only. The visits to predict, listed with their dates in
`visits_to_predict.csv`, are withheld: no script can read their values and no setting may
refer to them. The matching error reported by the fitter is the error on the matching
window inside the visits given, not the error on the visits to predict.
"""

SKILL_A = """---
name: shc-configure
description: Predict a land parcel's satellite band values at each visit listed in
  visits_to_predict.csv from its own history of visits with the synthetic historical control
  method, by deciding the method's settings — candidate smoothing bandwidths, the matching
  window lengths tried, the number of blocks, the error threshold that picks the window
  length, selection tolerance — and running the bundled fitter. Use this whenever a parcel's
  band values are to be forecast at listed later visits from its own past and the fitting
  code is provided.
---

# Configure and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier visits are weighted to
reproduce its most recent visits, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction; read it if anything below is unclear.

## What is fixed in this skill, and what you decide

The fitter makes these choices for you: each band is smoothed and standardized before
matching; every band enters with equal weight; weights are non-negative and sum to one,
with no constant term; selection is stepwise; the prediction is the weighted combination
of the observed values that followed each kept block.

You decide the settings, all seven, in `settings.json`:

- `time_bandwidth_candidates` — the bandwidths of the smoother, in days, among which each
  band's is chosen by leave-one-out cross-validation
- `m_min`, `m_step`, `m_max` — the matching window lengths tried, in visits
- `max_donors` — the most blocks used
- `nu` — the matching window length is the longest one tried whose matching error R is at
  most this
- `zeta` — a further block is kept only if it lowers the matching error R by more than this

`references/settings.md` gives each field's type, range and meaning, and the rules that
tie them: `m_min` must be at most `m_max`, and `m_max` plus the number of visits to predict
must be at most the number of visits given, so that at least one block exists. There are no
defaults; the fitter refuses an incomplete file and names what is missing.

## Workflow

1. Look at the data: `python scripts/show_data.py`. Every visit given, every band, with
   the date of each, and the visits to predict. Nothing else about this parcel exists.
2. Decide the settings from what you see. Short records reward thought here: a longer
   matching window (`m_min` to `m_max`) matches more shape but leaves fewer blocks; a wider
   bandwidth smooths more; a large `zeta` keeps one block, a small one lets more in.
3. Write `settings.json`, then run `python scripts/fit.py settings.json`. It prints the
   matching window length chosen, the blocks it kept, their weights, the matching error
   over the visits given, and the prediction at every visit to predict.
4. You may change the settings and run again. Every run is recorded. The matching error is
   an in-sample quantity; a smaller one is not by itself a better forecast.
5. Finish by writing `answer.json`, with every band at every visit to predict:
   ```json
   {"prediction": {"<period_id>": {"<band>": <value>, ...}, ...}, "rationale": "<why these settings, in a few sentences>"}
   ```
   The prediction you submit is your answer. If it differs from the fitter's, say why.
"""

SKILL_B = """---
name: shc-design
description: Predict a land parcel's satellite band values at each visit listed in
  visits_to_predict.csv from its own history of visits with the synthetic historical control
  method, deciding both its settings and its methodological choices — what is matched, how
  weights are constrained, how bands are weighted, how blocks are selected, how the
  prediction is formed — and running the bundled fitter. Use this whenever a parcel's band
  values are to be forecast at listed later visits from its own past and the design of the
  estimator is open.
---

# Design and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier visits are weighted to
reproduce its most recent visits, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction and the alternatives at each step.

## Everything is yours to decide

`spec.json` holds two kinds of field. The settings (`time_bandwidth_candidates`, `m_min`,
`m_step`, `m_max`, `max_donors`, `nu`, `zeta`) are numbers. The choices are each one of a
few options:

- `fit_space` — match the prepared series (`latent_std`) or the observed values (`raw`)
- `weight_constraint` — `simplex` (non-negative, sum to one), `nonneg`, or `unconstrained`
- `intercept` — `none` or `fitted`
- `band_weighting` — `equal`, `inverse_variance`, or `subset` (with `band_subset`)
- `selection` — `greedy_forward` or `exhaustive`
- `readd_seasonal` — predict from observed forward values (`false`) or in the prepared
  space with the seasonal term added back (`true`)

`references/settings.md` defines every field and the rules that tie them, for instance
that weights fitted on the standardized prepared series keep the observed scale only when
they sum to one, so `nonneg` and `unconstrained` need `raw` or `readd_seasonal`. The fitter
refuses a specification that breaks a rule and names the fields, so read the rules before
writing the file. There are no defaults.

## Workflow

1. `python scripts/show_data.py` — every visit given, every band, and the visits to predict.
2. Read `references/method.md` and `references/settings.md`; decide the design from the data.
3. Write `spec.json`; run `python scripts/fit.py spec.json`. It prints the matching window
   length chosen, the blocks kept, weights (and intercept if any), the matching error over
   the visits given, and the prediction at every visit to predict.
4. Revise and run again if you judge it right; every run is recorded. The matching error is
   in-sample: a design that fits the visits given better does not necessarily predict the
   visits to predict better.
5. Finish by writing `answer.json` as in the format below, with every band at every visit
   to predict. The prediction you submit is your answer; if it differs from the fitter's,
   say why.
   ```json
   {"prediction": {"<period_id>": {"<band>": <value>, ...}, ...}, "rationale": "<why this design, in a few sentences>"}
   ```
"""

KIND_LABEL = {"int": "<int>", "float": "<float>", "int_or_null": "<int or null>", "bwlist": "<list of numbers>"}


def settings_md(arm):
    rows = ["| field | type and range | meaning |", "|---|---|---|"]
    for f, (kind, allowed, meaning) in S.CLASS1.items():
        rng, sep, why = meaning.partition(": ")
        rows.append(f"| `{f}` | {rng} | {why} |")
    out = [f"# Fields for {'shc-configure' if arm == 'A' else 'shc-design'}", "",
           f"`{'settings.json' if arm == 'A' else 'spec.json'}` holds exactly the fields below; there are no defaults, and a file "
           "with a field missing, a field misspelled, or a value outside its range is refused with the field named.", "",
           "## Settings", ""] + rows
    if arm == "B":
        out += ["", "## Choices", "", "| field | options | meaning |", "|---|---|---|"]
        for f, (kind, allowed, meaning) in S.CLASS2.items():
            opts = ", ".join(f"`{a}`" for a in allowed) if kind == "enum" else "`true`, `false`"
            out.append(f"| `{f}` | {opts} | {meaning} |")
        out.append(f"| `band_subset` | list of band names | {S.OPTIONAL['band_subset'][2]} |")
        out += ["", "## Rules that tie the fields together", ""] + [f"- {r}" for r in S.CROSS_FIELD_RULES]
        out += ["", "## Shape of the file (every value is yours)", "", "```json",
                json.dumps({**{f: KIND_LABEL[S.CLASS1[f][0]] for f in S.CLASS1},
                            **{f: ("<true|false>" if k == "bool" else "<" + "|".join(a) + ">") for f, (k, a, _) in S.CLASS2.items()},
                            "band_subset": "<list of band names, only with subset>"}, indent=1), "```"]
    else:
        out += ["", "## Rules", ""] + [f"- {S.CROSS_FIELD_RULES[i]}" for i in S.CONFIGURE_RULES]
        out += ["", "## Shape of the file (every value is yours)", "", "```json",
                json.dumps({f: KIND_LABEL[S.CLASS1[f][0]] for f in S.CLASS1}, indent=1), "```"]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- scripts the model runs
LB_LIB_SHIM = '''"""lb_lib shim for the skill scripts: the few names the bundled package modules read, and the readers of data.csv
(the visits given) and visits_to_predict.csv in the working directory. No project path exists here."""
from pathlib import Path
import pandas as pd

LB = Path(".")
DATA_CSV = LB / "data.csv"
TO_PREDICT_CSV = LB / "visits_to_predict.csv"
SENSORS = ("sentinel1", "sentinel2")
BANDS = {"sentinel1": ["VV", "VH", "VV_minus_VH"], "sentinel2": ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI"]}
DECIMALS = {"sentinel2": 3, "sentinel1": 2}


def disp(band):
    return "VV-VH" if band == "VV_minus_VH" else band


def undisp(band):
    return "VV_minus_VH" if band == "VV-VH" else band


def sensor_of(bands):
    return "sentinel1" if "VV" in set(bands) else "sentinel2"


def given():
    """data.csv (band, period_id, date, y) -> (long frame with the package band names, sensor). Values are read
    exactly as written (round-trip float parsing)."""
    d = pd.read_csv(DATA_CSV, float_precision="round_trip")
    sensor = sensor_of(d.band.unique())
    d["band"] = [undisp(b) for b in d["band"]]
    d["date"] = pd.to_datetime(d["date"])
    return d, sensor


def to_predict():
    """visits_to_predict.csv (period_id, date) -> [(period_id, date)] in time order."""
    v = pd.read_csv(TO_PREDICT_CSV, dtype=str)
    return list(zip(v["period_id"], v["date"]))
'''

SHOW_DATA = '''"""Print this parcel's visits given, every band, and the visits to predict."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb_lib as lb

d, sensor = lb.given()
bands = lb.BANDS[sensor]
dec = lb.DECIMALS[sensor]
w = d.pivot(index=["period_id", "date"], columns="band", values="y").reset_index().sort_values("date")
print(f"{'Sentinel-1 radar' if sensor == 'sentinel1' else 'Sentinel-2 optical'}: mean band values of one land parcel at "
      f"{len(w)} satellite visits, {w['date'].iloc[0]:%Y-%m-%d} to {w['date'].iloc[-1]:%Y-%m-%d}. "
      f"Bands: {', '.join(lb.disp(b) for b in bands)}.")
for _, r in w.iterrows():
    print(f"{r.period_id} = {r.date:%Y-%m-%d}: " + ", ".join(f"{lb.disp(b)} {r[b]:.{dec}f}" for b in bands))
tp = lb.to_predict()
print(f"The {len(tp)} visits to predict, listed in visits_to_predict.csv, are below. Their values are withheld.")
for p, day in tp:
    print(f"{p} = {day}")
'''

FIT_PY = '''"""Run the synthetic historical control on this parcel's visits given under the given file.
    python scripts/fit.py %(file)s
Prints the matching window length chosen, the blocks kept, the weights, the matching error over the visits given, and
the prediction at every visit to predict; appends a record of the run to .fit_log.jsonl. The fit sees the visits
given only.%(fixed_doc)s"""
import json
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb_lib as lb
import lb_spec_daily
import lb_fit_daily

%(fixed_code)s

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/fit.py %(file)s")
    try:
        given = json.load(open(sys.argv[1]))
    except FileNotFoundError:
        raise SystemExit(f"{sys.argv[1]} not found; write it first")
    except json.JSONDecodeError as e:
        raise SystemExit(f"{sys.argv[1]} is not valid JSON: {e}")
    d, sensor = lb.given()
    tp = lb.to_predict()
    pid_of = dict(zip(d["date"], d["period_id"]))
    pre = d[["band", "date", "y"]]
    n_given = d["date"].nunique()
    try:
%(validate)s
    except lb_spec_daily.SpecError as e:
        raise SystemExit(f"refused: {e}")
    try:
        out = lb_fit_daily.run_case(pre, tp, sensor, spec)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"the fit could not run: {e}")
    dates = sorted(d["date"].unique())
    pid = [pid_of[x] for x in dates]
    m, n = out["m_star"], len(tp)
    kept = [f"block {i} ({pid[s]}-{pid[s + m - 1]} -> {pid[s + m]}-{pid[s + m + n - 1]})"
            for i, s in zip(out["donor_ids"], out["block_starts"])]
    rule = ("the longest m tried with R <= nu" if out["selection_mode"] == "paper_rule_stacked"
            else "no m tried had R <= nu; the longest m tried")
    dec = lb.DECIMALS[sensor]
    print(f"visits given: {n_given}; visits to predict: {n}")
    print("bandwidths chosen, days: " + ", ".join(f"{lb.disp(b)} {h:g}" for b, h in out["bandwidths"].items()))
    print(f"matching window length: m = {m} ({rule}); blocks available: {out['n_blocks']}")
    print("blocks kept: " + "; ".join(kept))
    print("weights: " + ", ".join(f"{w_:.4f}" for w_ in out["weights"]) + (f"; intercept {out['intercept']:.4f}" if spec["intercept"] == "fitted" else ""))
    print(f"matching error over the matching window, visits given only: R = {out['R']:.6f}, mean squared error = {out['fit_mse']:.6f}")
    print("prediction:")
    for p, day in tp:
        print(f"{p} = {day}: " + ", ".join(f"{b} {v[p]:.{dec}f}" for b, v in out["predictions"].items()))
    rec = {"file": sys.argv[1], "spec": spec, "m_star": m, "selection_mode": out["selection_mode"],
           "donor_ids": out["donor_ids"], "weights": [float(x) for x in out["weights"]], "intercept": out["intercept"],
           "R": out["R"], "fit_mse": out["fit_mse"], "bandwidths": out["bandwidths"], "predictions": out["predictions"],
           "engine_used": out["engine_used"], "near_ties": out["near_ties"]}
    with open(".fit_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\\n")
    print("@@FIT@@" + json.dumps(rec))          # captured by the harness and removed from what the model sees


if __name__ == "__main__":
    main()
'''

FIXED_A_CODE = '''# The six choices this skill fixes. A settings file that names any of them is refused, not overridden.
FIXED = {"weight_constraint": "simplex", "intercept": "none", "band_weighting": "equal",
         "fit_space": "latent_std", "readd_seasonal": False, "selection": "greedy_forward"}
'''
VALIDATE_A = '''        c1 = lb_spec_daily.class1_only(given, sensor, n_given, tp)
        spec = lb_spec_daily.validate_spec({**c1, **FIXED}, sensor, n_given, tp)'''
VALIDATE_B = '''        spec = lb_spec_daily.validate_spec(given, sensor, n_given, tp)'''


def build():
    if SKILLS.exists():
        shutil.rmtree(SKILLS)
    files = {}
    for arm, name, skill_md in (("A", "shc-configure", SKILL_A), ("B", "shc-design", SKILL_B)):
        root = SKILLS / name
        (root / "references").mkdir(parents=True)
        (root / "scripts").mkdir()
        files[f"{name}/SKILL.md"] = skill_md
        files[f"{name}/references/method.md"] = METHOD_MD
        files[f"{name}/references/settings.md"] = settings_md(arm)
        files[f"{name}/scripts/show_data.py"] = SHOW_DATA
        files[f"{name}/scripts/lb_lib.py"] = LB_LIB_SHIM
        files[f"{name}/scripts/fit.py"] = FIT_PY % {
            "file": "settings.json" if arm == "A" else "spec.json",
            "fixed_doc": "\nThe six choices of the method are fixed here; only the seven settings are read." if arm == "A" else "",
            "fixed_code": FIXED_A_CODE if arm == "A" else "",
            "validate": VALIDATE_A if arm == "A" else VALIDATE_B}
        for mod in BUNDLED:
            files[f"{name}/scripts/{mod}"] = (lb.LB / mod).read_text()
    manifest = {}
    for rel, text in files.items():
        p = SKILLS / rel
        p.write_text(text)
        manifest[rel] = hashlib.sha256(text.encode()).hexdigest()
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def check():
    """Gate DK6: every file under skills_daily/ matches the manifest, and every bundled package module is
    byte-identical to the package file."""
    manifest = json.loads(MANIFEST.read_text())
    for rel, sha in manifest.items():
        assert hashlib.sha256((SKILLS / rel).read_bytes()).hexdigest() == sha, f"skill file changed: {rel}"
    for rel in manifest:
        mod = rel.rsplit("/", 1)[-1]
        if mod in BUNDLED:
            assert (SKILLS / rel).read_bytes() == (lb.LB / mod).read_bytes(), f"bundled copy differs from package: {rel}"
    print(f"DK6 ok: {len(manifest)} skill files match MANIFEST.json; bundled modules byte-identical to the package")
    return manifest


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        m = build()
        print(f"built {len(m)} files under {SKILLS}")
        for rel in m:
            print("  ", rel)
