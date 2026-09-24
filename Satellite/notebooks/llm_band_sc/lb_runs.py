"""lb_runs — score each repeat run of the SHC design separately, then average across runs.

Why this exists: the ARC endpoint is not reproducible. The same prompt, model, seed 0,
reasoning_effort low and temperature 0 returned different answers on 2026-09-10 and 2026-09-15
(20 of 27 gpt-oss replies differ; its Sentinel-1 mean moves 0.457 -> 0.613, a gap as large as the
gap between models). One deterministic call per site therefore does not pin down a model, so the
runs are kept side by side under runs/ and reported as a mean over runs with the spread shown.

Averaging: the metric is computed per run and then averaged over runs (a repeat measure — "what
does a typical single run score"). The predictions are NOT averaged first; that would be a
two-run ensemble, a different estimator that generally scores better, and lb_score.py already has
a separate `ensemble` row for that kind of thing.

Raw runs: runs/run1_2026-09-10/ (the run committed in 5de649b, which the 10 September report and
deck are built from; no reasoning text) and runs/run2_2026-09-15/ (the reasoning-capture rerun).

    python lb_runs.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lb_lib as lb

RUNS_DIR = lb.LB / "runs"


def runs():
    """[(run label, directory)] in chronological order."""
    return [(d.name, d) for d in sorted(RUNS_DIR.iterdir()) if d.is_dir()]


def prompts_for(exp, _cache={}):
    """One experiment's prompt rows keyed by prompt_id, loaded once."""
    if exp not in _cache:
        _cache[exp] = {json.loads(l)["prompt_id"]: json.loads(l)
                       for l in open(lb.LB / f"lb_prompts_{exp}.jsonl")}
    return _cache[exp]


def score_run(directory):
    """Per experiment, site and method, the collaborator's outcome metric on both denominators.

    A run folder may hold more than one arm. The truth and train rows are taken from the experiment
    each reply names, NOT from the filename: `lb_replies_shc_steps_*.jsonl` matches the glob
    `lb_replies_shc_*.jsonl`, and its prompt_ids (…_steps) do not exist in lb_prompts_shc.jsonl,
    so keying off the filename raises KeyError as soon as a steps run is stored under runs/."""
    rows = []
    for f in sorted(directory.glob("lb_replies_shc*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if not r.get("ok"):
                continue
            exp = r["experiment"]
            if not exp.startswith("shc"):            # this scorer is the SHC design's metric only
                continue
            p = prompts_for(exp)[r["prompt_id"]]
            sensor = p["sensor"]
            bands = [lb.disp(b) for b in lb.BANDS[sensor]]
            eff = r.get("reasoning_effort", "default")
            method = r["model"] if eff == "low" else f"{r['model']} ({eff} effort)"
            per = [lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b]) for b in bands]
            per10 = [lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b] + [p["truth"][b]]) for b in bands]
            rows.append({"experiment": exp, "method": method, "sensor": sensor,
                         "site_id": p["target_site"],
                         "joint_nmse": float(np.mean(per)), "joint_nmse_p10var": float(np.mean(per10))})
    return pd.DataFrame(rows)


def main():
    parts = []
    for label, directory in runs():
        d = score_run(directory)
        if not len(d):
            print(f"{label}: no scored cells (empty or unparsed run)")
            continue
        d.insert(0, "run", label)
        parts.append(d)
        print(f"{label}: {len(d)} scored cells, arms {sorted(d.experiment.unique())}, "
              f"methods {sorted(d.method.unique())}")
    long = pd.concat(parts, ignore_index=True)
    long.to_csv(lb.LB / "lb_scores_shc_runs.csv", index=False)

    # per site: mean over the runs that answered it
    per_site = (long.groupby(["experiment", "method", "sensor", "site_id"])
                    .agg(joint_nmse=("joint_nmse", "mean"),
                         p10var_not_reported=("joint_nmse_p10var", "mean"),
                         n_runs=("run", "nunique")).reset_index())
    per_site.to_csv(lb.LB / "lb_scores_shc_runmean.csv", index=False)

    # the collaborator's SHC reference is deterministic; take it from the single-run scores
    S = pd.read_csv(lb.LB / "lb_scores_shc.csv")
    ref = S[(S.method == "shc") & (S.ok == True)].set_index(["sensor", "site_id"])[lb.METRIC]

    out = []
    for (exp, method, sensor), g in per_site.groupby(["experiment", "method", "sensor"]):
        row = {"experiment": exp, "method": method, "sensor": sensor, "sites": len(g),
               "sites_with_2_runs": int((g.n_runs > 1).sum()),
               "mean_joint_nmse": g.joint_nmse.mean(),
               "median_joint_nmse": g.joint_nmse.median(),
               "mean_joint_nmse": g.joint_nmse.mean(),
               "beats_shc": f"{int((g.set_index('site_id')[lb.METRIC] < ref.loc[sensor].loc[g.site_id]).sum())} / {len(g)}"}
        for label, _ in runs():                      # each run's own mean, for the spread
            d = long[(long.run == label) & (long.experiment == exp)
                     & (long.method == method) & (long.sensor == sensor)]
            row[f"mean_{label}"] = d[lb.METRIC].mean() if len(d) else np.nan
        out.append(row)
    summary = pd.DataFrame(out).sort_values(["experiment", "sensor", "mean_joint_nmse"])
    summary.to_csv(lb.LB / "lb_summary_shc_runs.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== SHC design, mean over runs (her outcome metric, variance over P01-P10) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return long, per_site, summary


if __name__ == "__main__":
    main()
