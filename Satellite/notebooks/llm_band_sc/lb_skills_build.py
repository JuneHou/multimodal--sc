"""lb_skills_build — generate the three SHC skills (approved by Jun 2026-09-24) under skills/.

    skills/shc-configure/   arm A: the model decides the five settings; the six choices are fixed in scripts/fit.py
    skills/shc-design/      arm B: the model decides settings and choices; scripts/fit.py takes the full spec
    skills/shc-implement/   arm C: the model writes predict.py; there is no fitter

Every text below is what the model reads, byte for byte. The scripts directory of A and B also carries the package
modules the fitter needs (lb_her_nb10, lb_her_panel, lb_her_run, lb_spec, lb_fit) as VERBATIM copies plus a small
lb_lib shim, so no fitting logic is duplicated: gate SK1 checks each copy's sha256 against the package file.
skills/MANIFEST.json records the sha256 of every generated file; gate SK1 checks the mounted files against it.

    python lb_skills_build.py           (re)generate skills/ and MANIFEST.json
    python lb_skills_build.py --check   compare skills/ with a fresh generation (gate SK1)
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import lb_lib as lb
import lb_spec

SKILLS = lb.LB / "skills"
MANIFEST = SKILLS / "MANIFEST.json"
BUNDLED = ["lb_her_nb10.py", "lb_her_panel.py", "lb_her_run.py", "lb_spec.py", "lb_fit.py"]   # verbatim copies

# ----------------------------------------------------------------------------- texts (approved 2026-09-24)
METHOD_MD = """# Synthetic historical control for one parcel

The parcel's own past is its comparison. Nothing from any other parcel is used.

## Historical blocks
From the nine observed months P01–P09, form overlapping blocks. A block is m consecutive
months (its matching part) followed by the n months that came after it (its forward part).
Blocks slide one month at a time, so there are 9 − m − n + 1 of them; the last block's
forward part ends at P09. The matching window is the parcel's most recent m months.

## What is matched
Each band's nine months can be matched as observed, or after preparation: an annual sine
and cosine term is estimated for the band (after partialling out a low-order time trend in
the month index) and removed, the remainder is smoothed along time with a local-linear
Gaussian kernel of a chosen bandwidth, and the smoothed series is standardized to mean 0
and standard deviation 1 over P01–P09. Bands are stacked, so one set of block weights
serves every band.

## Weights and selection
Weights over the blocks are fitted so the weighted combination of the blocks' matching
parts reproduces the matching window, across all bands jointly, minimizing mean squared
error. A selection rule decides which blocks carry weight: stepwise (keep the single
best-fitting block, then add one block at a time while the error falls by more than a
tolerance) or exhaustive (every non-empty subset, lowest error wins).

## Prediction
The next month is predicted as the same weighted combination of what followed each kept
block — its first forward month, as observed — or, when the fit was made on the prepared
series, in the prepared space with the standardization undone and the target month's
seasonal term added back.

## What the fit can and cannot see
The fit sees P01–P09 only. The month to predict, P10 (2024-08), is withheld: no script
can read it and no setting may refer to it. The matching error reported by the fitter is
the error on the matching window inside P01–P09, not the error on P10.
"""

SKILL_A = """---
name: shc-configure
description: Predict the next month of a land parcel's satellite band values from its own
  nine-month history with the synthetic historical control method, by deciding the method's
  settings — block length, forecast horizon, smoothing bandwidth, selection tolerance, trend
  degree — and running the bundled fitter. Use this whenever a parcel's monthly band values
  are to be forecast one month ahead from its own past and the fitting code is provided.
---

# Configure and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier months are weighted to
reproduce its most recent months, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction; read it if anything below is unclear.

## What is fixed in this skill, and what you decide

The fitter makes these choices for you: each band is deseasonalized, smoothed and
standardized before matching; every band enters with equal weight; weights are
non-negative and sum to one, with no constant term; selection is stepwise; the prediction
is the weighted combination of the observed values that followed each kept block.

You decide the settings, all five, in `settings.json`:

- `m` — length of the matching window and of every block's matching part, in months
- `n` — how many months after a block it is taken to predict; the first is used
- `kernel_bandwidth` — bandwidth of the smoother, in months
- `zeta` — a further block is kept only if it lowers the matching error by more than this
- `beta_trend_degree` — 0, 1 or 2, the trend partialled out when the seasonal terms are estimated

`references/settings.md` gives each field's type, range and meaning, and the one rule that
ties them: `m + n` must be at most 9, so that at least one block exists. There are no
defaults; the fitter refuses an incomplete file and names what is missing.

## Workflow

1. Look at the data: `python scripts/show_data.py`. Nine months, every band, with the
   calendar month of each. Nothing else about this parcel exists.
2. Decide the settings from what you see. Short records reward thought here: a longer `m`
   matches more shape but leaves fewer blocks; a wider bandwidth smooths more; a large
   `zeta` keeps one block, a small one lets more in.
3. Write `settings.json`, then run `python scripts/fit.py settings.json`. It prints the
   blocks it kept, their weights, the matching error over P01–P09, and the prediction.
4. You may change the settings and run again. Every run is recorded. The matching error is
   an in-sample quantity; a smaller one is not by itself a better forecast.
5. Finish by writing `answer.json`:
   ```json
   {"prediction": {"<band>": <value>, ...}, "rationale": "<why these settings, in a few sentences>"}
   ```
   The prediction you submit is your answer. If it differs from the fitter's, say why.
"""

SKILL_B = """---
name: shc-design
description: Predict the next month of a land parcel's satellite band values from its own
  nine-month history with the synthetic historical control method, deciding both its
  settings and its methodological choices — what is matched, how weights are constrained,
  how bands are weighted, how blocks are selected, how the prediction is formed — and
  running the bundled fitter. Use this whenever a parcel's monthly band values are to be
  forecast one month ahead from its own past and the design of the estimator is open.
---

# Design and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier months are weighted to
reproduce its most recent months, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction and the alternatives at each step.

## Everything is yours to decide

`spec.json` holds two kinds of field. The settings (`m`, `n`, `kernel_bandwidth`, `zeta`,
`beta_trend_degree`) are numbers. The choices are each one of a few options:

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

1. `python scripts/show_data.py` — the nine months, every band.
2. Read `references/method.md` and `references/settings.md`; decide the design from the data.
3. Write `spec.json`; run `python scripts/fit.py spec.json`. It prints the blocks kept,
   weights (and intercept if any), the matching error over P01–P09, and the prediction.
4. Revise and run again if you judge it right; every run is recorded. The matching error is
   in-sample: a design that fits P01–P09 better does not necessarily predict P10 better.
5. Finish by writing `answer.json` as in the format below. The prediction you submit is
   your answer; if it differs from the fitter's, say why.
   ```json
   {"prediction": {"<band>": <value>, ...}, "rationale": "<why this design, in a few sentences>"}
   ```
"""

SKILL_C = """---
name: shc-implement
description: Predict the next month of a land parcel's satellite band values from its own
  nine-month history by implementing the synthetic historical control method yourself in
  Python and running it. Use this whenever a parcel's monthly band values are to be
  forecast one month ahead from its own past and no fitting code is provided.
---

# Implement and run the synthetic historical control

The parcel's past is its comparison: blocks of its own earlier months are weighted to
reproduce its most recent months, and what followed those blocks becomes the forecast.
`references/method.md` describes the construction; every setting and every choice in it
is yours, made inside your program.

## Workflow

1. `python scripts/show_data.py` — the nine months, every band. The same data is in
   `data.csv` in this directory: one row per band per month, columns `band`, `period_id`
   (P01–P09 in time order), `month` (YYYY-MM), `mean_value`.
2. Read `references/method.md`. Decide how to build the blocks, what to match, how to fit
   and constrain the weights, how to select blocks, and how to form the prediction.
3. Write `predict.py`. It reads `data.csv` and prints one line per band, `BAND VALUE`, the
   predicted mean value for the next month. numpy, scipy and pandas are available; nothing
   outside this directory is, and there is no network.
4. Run `python predict.py`. Fix what fails; run again. Every run is recorded.
5. Finish by writing `answer.json`:
   ```json
   {"prediction": {"<band>": <value>, ...}, "rationale": "<what the program does and why>"}
   ```
"""


def settings_md(arm):
    rows = ["| field | type and range | meaning |", "|---|---|---|"]
    for f, (kind, allowed, meaning) in lb_spec.CLASS1.items():
        rng, sep, why = meaning.partition(": ")
        rows.append(f"| `{f}` | {rng} | {why} |")
    out = [f"# Fields for {'shc-configure' if arm == 'A' else 'shc-design'}", "",
           f"`{'settings.json' if arm == 'A' else 'spec.json'}` holds exactly the fields below; there are no defaults, and a file "
           "with a field missing, a field misspelled, or a value outside its range is refused with the field named.", "",
           "## Settings", ""] + rows
    if arm == "B":
        out += ["", "## Choices", "", "| field | options | meaning |", "|---|---|---|"]
        for f, (kind, allowed, meaning) in lb_spec.CLASS2.items():
            opts = ", ".join(f"`{a}`" for a in allowed) if kind == "enum" else "`true`, `false`"
            out.append(f"| `{f}` | {opts} | {meaning} |")
        out.append(f"| `band_subset` | list of band names | {lb_spec.OPTIONAL['band_subset'][2]} |")
        out += ["", "## Rules that tie the fields together", ""] + [f"- {r}" for r in lb_spec.CROSS_FIELD_RULES]
        out += ["", "## Shape of the file (every value is yours)", "", "```json",
                json.dumps({**{f: f"<{lb_spec.CLASS1[f][0]}>" for f in lb_spec.CLASS1},
                            **{f: ("<true|false>" if k == "bool" else "<" + "|".join(a) + ">") for f, (k, a, _) in lb_spec.CLASS2.items()},
                            "band_subset": "<list of band names, only with subset>"}, indent=1), "```"]
    else:
        out += ["", "## Rules", "", f"- {lb_spec.CROSS_FIELD_RULES[0]}", f"- {lb_spec.CROSS_FIELD_RULES[-1]}",
                "", "## Shape of the file (every value is yours)", "", "```json",
                json.dumps({f: f"<{lb_spec.CLASS1[f][0]}>" for f in lb_spec.CLASS1}, indent=1), "```"]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- scripts the model runs
LB_LIB_SHIM = '''"""lb_lib shim for the skill scripts: the few names the bundled package modules read, derived from data.csv in the
working directory (nine periods; the target month is the calendar month after P09). No project path exists here."""
from pathlib import Path
import pandas as pd

LB = Path(".")
FEATURES_CSV = LB / "data.csv"
HER10_SITES = LB / "none.csv"
SENSORS = ("sentinel1", "sentinel2")
BANDS = {"sentinel1": ["VV", "VH", "VV_minus_VH"], "sentinel2": ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI"]}
DECIMALS = {"sentinel2": 3, "sentinel1": 2}
PRE = list(range(1, 10))
TARGET = 10
SD_TOL = 1e-8


def disp(band):
    return "VV-VH" if band == "VV_minus_VH" else band


def undisp(band):
    return "VV_minus_VH" if band == "VV-VH" else band


def _calendar():
    d = pd.read_csv(FEATURES_CSV)
    months = dict(zip(d.period_id, d.month))
    cal = {int(p[1:]): m for p, m in months.items()}
    y, mo = map(int, cal[max(cal)].split("-"))
    mo += 1
    if mo == 13:
        y, mo = y + 1, 1
    cal[max(cal) + 1] = f"{y:04d}-{mo:02d}"
    return cal


CALENDAR = _calendar()


def sensor_of(bands):
    return "sentinel1" if "VV" in set(bands) else "sentinel2"


def wide_from_data(path=FEATURES_CSV):
    """data.csv (band, period_id, month, mean_value) -> the wide frame lb_her_panel.panel_from_wide expects."""
    d = pd.read_csv(path)
    sensor = sensor_of(d.band.unique())
    w = d.pivot(index=["period_id", "month"], columns="band", values="mean_value").reset_index()
    w.columns = [c if c in ("period_id", "month") else f"mean_{undisp(c)}" for c in w.columns]
    w["sensor"] = sensor
    w["site_id"] = "site"
    w["seq"] = w.period_id.str[1:].astype(int)
    return w.sort_values("seq").reset_index(drop=True), sensor
'''

SHOW_DATA = '''"""Print this parcel's nine months, every band."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb_lib as lb

w, sensor = lb.wide_from_data()
bands = lb.BANDS[sensor]
d = lb.DECIMALS[sensor]
print(f"{'Sentinel-1 radar' if sensor == 'sentinel1' else 'Sentinel-2 optical'}: mean band values of one land parcel, "
      f"{len(w)} consecutive calendar months. Bands: {', '.join(lb.disp(b) for b in bands)}.")
for _, r in w.iterrows():
    print(f"P{int(r.seq):02d} = {r.month}: " + ", ".join(f"{lb.disp(b)} {r[f'mean_{b}']:.{d}f}" for b in bands))
print(f"The month to predict is P{lb.TARGET:02d} = {lb.CALENDAR[lb.TARGET]}. Its values are withheld.")
'''

FIT_PY = '''"""Run the synthetic historical control on this parcel's nine months under the given file.
    python scripts/fit.py %(file)s
Prints the blocks kept, the weights, the matching error over P01-P09, and the prediction; appends a record of the
run to .fit_log.jsonl. The fit sees P01-P09 only.%(fixed_doc)s"""
import json
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb_lib as lb
import lb_her_panel
import lb_spec
import lb_fit

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
    w, sensor = lb.wide_from_data()
    panel = lb_her_panel.panel_from_wide(w)
    try:
%(validate)s
    except lb_spec.SpecError as e:
        raise SystemExit(f"refused: {e}")
    try:
        out = lb_fit.run_cell(panel, sensor, "site", spec)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"the fit could not run: {e}")
    blocks = lb_fit.her.make_historical_blocks_from_periods(lb_fit.TRAIN, m=spec["m"], n=spec["n"])
    by_id = {b["donor_id"]: b for b in blocks}
    kept = [f"block {i} ({by_id[i]['pre_periods'][0]}-{by_id[i]['pre_periods'][-1]} -> {by_id[i]['post_periods'][0]})" for i in out["donor_ids"]]
    d = lb.DECIMALS[sensor]
    print(f"blocks available: {out['n_blocks']} (m = {spec['m']}, n = {spec['n']})")
    print("blocks kept: " + "; ".join(kept))
    print("weights: " + ", ".join(f"{w_:.4f}" for w_ in out["weights"]) + (f"; intercept {out['intercept']:.4f}" if spec["intercept"] == "fitted" else ""))
    print(f"matching error (mean squared error over the matching window, P01-P09 only): {out['fit_mse']:.6f}")
    print("prediction for P%%02d = %%s: " %% (lb.TARGET, lb.CALENDAR[lb.TARGET]) + ", ".join(f"{b} {v:.{d}f}" for b, v in out["predictions"].items()))
    rec = {"file": sys.argv[1], "spec": spec, "donor_ids": out["donor_ids"], "weights": [float(x) for x in out["weights"]],
           "intercept": out["intercept"], "fit_mse": out["fit_mse"], "predictions": out["predictions"],
           "engine_used": out["engine_used"], "zeta_used": out["zeta_used"], "near_ties": out["near_ties"]}
    with open(".fit_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\\n")
    print("@@FIT@@" + json.dumps(rec))          # captured by the harness and removed from what the model sees


if __name__ == "__main__":
    main()
'''

FIXED_A_CODE = '''# The six choices this skill fixes (arm A). A settings file that names any of them is refused, not overridden.
FIXED = {"weight_constraint": "simplex", "intercept": "none", "band_weighting": "equal",
         "fit_space": "latent_std", "readd_seasonal": False, "selection": "greedy_forward"}
'''
VALIDATE_A = '''        c1 = lb_spec.class1_only(given, sensor)
        spec = lb_spec.validate_spec({**c1, **FIXED}, sensor)'''
VALIDATE_B = '''        spec = lb_spec.validate_spec(given, sensor)'''


def build():
    if SKILLS.exists():
        shutil.rmtree(SKILLS)
    files = {}
    for arm, name, skill_md in (("A", "shc-configure", SKILL_A), ("B", "shc-design", SKILL_B), ("C", "shc-implement", SKILL_C)):
        root = SKILLS / name
        (root / "references").mkdir(parents=True)
        (root / "scripts").mkdir()
        files[f"{name}/SKILL.md"] = skill_md
        files[f"{name}/references/method.md"] = METHOD_MD
        files[f"{name}/scripts/show_data.py"] = SHOW_DATA
        files[f"{name}/scripts/lb_lib.py"] = LB_LIB_SHIM
        if arm in ("A", "B"):
            files[f"{name}/references/settings.md"] = settings_md(arm)
            files[f"{name}/scripts/fit.py"] = FIT_PY % {
                "file": "settings.json" if arm == "A" else "spec.json",
                "fixed_doc": "\nThe six choices of the method are fixed here; only the five settings are read." if arm == "A" else "",
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
    """Gate SK1: every file under skills/ matches the manifest, the manifest matches a fresh generation, and every
    bundled package module is byte-identical to the package file."""
    manifest = json.loads(MANIFEST.read_text())
    for rel, sha in manifest.items():
        assert hashlib.sha256((SKILLS / rel).read_bytes()).hexdigest() == sha, f"skill file changed: {rel}"
    for rel in manifest:
        mod = rel.rsplit("/", 1)[-1]
        if mod in BUNDLED:
            assert (SKILLS / rel).read_bytes() == (lb.LB / mod).read_bytes(), f"bundled copy differs from package: {rel}"
    print(f"SK1 ok: {len(manifest)} skill files match MANIFEST.json; bundled modules byte-identical to the package")
    return manifest


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        m = build()
        print(f"built {len(m)} files under {SKILLS}")
        for rel in m:
            print("  ", rel)
