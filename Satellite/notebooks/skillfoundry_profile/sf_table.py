"""sf_table — the side by side table of plan row 15 on the 27 monthly cells, scored as the collaborator's notebook 10
scores its held-out P10 validation (`Satellite/data/scripts/scripts/validation/shc/10_shc_monthly_multiple_outcomes.ipynb`,
cells 25, 26, 28, 30 and 34; her functions extracted in llm_band_sc/lb_her_nb10.py).

    python sf_table.py [--generators gpt-5.4 deepseek]

The evaluation is llm_band_sc/lb_skill_score.py's, so the simple and the profile skills are scored by one implementation:
per cell and band her heldout_validation_for_sensor rows (cell 25, her_heldout_rows), per cell the pandas mean of
standardized_sq_error over bands (joint_validation_nmse); a cell with no valid answer.json is a skipped site, as a site
whose validation raises is skipped in her cell 26. No comparison uses a tolerance. Summaries (her_summary), each as her
notebook prints it:
    status            processed and skipped cells per sensor (cell 34)
    by sensor         mean standardized_sq_error over the processed cells' bands (cell 30, mean_standardized_P10_sq_error)
    all cells         number, mean and median of joint_validation_nmse over the processed cells of both sensors (cell 28,
                      n_site_sensor, mean_validation_nmse, median_validation_nmse)
    by sensor x band  mean sq_error and mean standardized_sq_error (cell 30)
The simple skills' records are read by lb_skill_score.load(); the profile records are the last record per cell of
runs/replies_profile_*.jsonl. Reference rows are her SHC as run by lb_shc.py, P09 carried forward and the P07 to P09
mean, from llm_band_sc/lb_scores_shc.csv, through the same summaries.
Writes into results/
    cells_monthly.csv           per cell and band: her validation columns, joint_validation_nmse, status, source file
    table_monthly.md            per model and skill: status, by sensor and all cells
    table_monthly_bands.md      per model, skill, sensor and band: mean sq_error and mean standardized_sq_error
    numbers_table_monthly.csv   one number-registry row per number of the two tables, written at generation
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sf_run as S                   # puts llm_band_sc on sys.path
import lb_lib as lb                  # noqa: E402
import lb_skill_score as K           # noqa: E402

HERE = S.HERE
ROOT = HERE.parent.parent.parent
OUT = HERE / "results"
SENSORS = K.SENSORS
SIMPLE = {"shc_configure": "shc-configure", "shc_design": "shc-design", "shc_implement": "shc-implement"}
NB10 = "Satellite/data/scripts/scripts/validation/shc/10_shc_monthly_multiple_outcomes.ipynb"


def skill_order(generators):
    """Row order within each model: settings only, then code provided, then code written; simple before profile."""
    return (["shc-configure", "shc-design"] + [f"{g} code provided" for g in generators]
            + ["shc-implement"] + [f"{g} code written" for g in generators])


def records_simple():
    _, recs = K.load()
    for (method, pid), (r, f) in recs.items():
        if r["experiment"] in SIMPLE and method in S.DEFAULT_MODELS:
            yield SIMPLE[r["experiment"]], r, pid[:-len(r["experiment"][4:]) - 1] + "_shc", Path(f)


def records_profile(generators):
    last = {}
    for g in generators:
        for fmt in S.FORMATS:
            for f in sorted((HERE / "runs").glob(f"replies_profile_{g}_{fmt}_*_low.jsonl")):
                for line in open(f):
                    r = json.loads(line)
                    last[(r["experiment"], r["model"], r["prompt_id"])] = (r, f)
    for (exp, model, pid), (r, f) in last.items():
        yield f"{r['generator']} {r['format'].replace('_', ' ')}", r, pid.split("_profile_")[0] + "_shc", f


def cell_rows(base, generators):
    rows = []
    for skill, r, key, f in list(records_simple()) + list(records_profile(generators)):
        b = base[key]
        status = "processed" if r.get("ok") else "skipped"
        for v in K.her_heldout_rows(b, r["values"] if r.get("ok") else None):
            rows.append({"model": r["model"], "skill": skill, "sensor": b["sensor"], "site_id": b["target_site"],
                         "status": status, **v, "file": str(f.relative_to(ROOT))})
    rows += [{**v, "file": "Satellite/notebooks/llm_band_sc/lb_scores_shc.csv"} for v in K.reference_rows(base)]
    C = pd.DataFrame(rows)
    joint = (C[C.status == "processed"].groupby(["model", "skill", "sensor", "site_id"])["standardized_sq_error"]
             .mean().rename("joint_validation_nmse"))
    return C.merge(joint, on=["model", "skill", "sensor", "site_id"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="+", default=list(S.GENERATORS), choices=S.GENERATORS)
    a = ap.parse_args()
    base = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")}
    C = cell_rows(base, a.generators)
    OUT.mkdir(exist_ok=True)
    C.to_csv(OUT / "cells_monthly.csv", index=False)
    today = dt.date.today().isoformat()
    f6 = lambda x: f"{x:.6f}" if np.isfinite(x) else "–"
    arms = [(m, s) for m in S.DEFAULT_MODELS for s in skill_order(a.generators)] + [("reference", l) for _, l in K.REFERENCES]
    head = ("| model | skill | " + " | ".join(f"{t} processed | {t} skipped | {t} mean_standardized_P10_sq_error"
                                             for _, t in SENSORS)
            + " | n_site_sensor | mean_validation_nmse | median_validation_nmse |")
    lines = [head, "|" + "---|" * (2 + 3 * len(SENSORS) + 3)]
    blines = ["| model | skill | sensor | band | sq_error | standardized_sq_error |", "|---|---|---|---|---|---|"]
    reg = []
    for model, skill in arms:
        g = C[(C.model == model) & (C.skill == skill)]
        if g.empty:
            continue
        src = "; ".join(sorted(set(g.file)))
        what = f"{model}, {skill}, monthly cells"
        summ, bt = K.her_summary(g)
        cols = []
        for s, t in SENSORS:
            n_p, n_s = summ[f"{s}_processed"], summ[f"{s}_skipped"]
            cols += [str(n_p), str(n_s), f6(summ[f"{s}_mean_standardized_P10_sq_error"])]
            key = f"tbl_m_{model}_{skill.replace(' ', '_')}_{s}"
            reg += [(f"{key}_status", f"{n_p}, {n_s}", f"processed and skipped {t} cells, {what}", src,
                     f"sf_table.py, as {NB10} cell 34", today),
                    (f"{key}_mean_std", cols[-1], f"mean standardized P10 squared error over the bands of the processed "
                     f"{t} cells, {what}", src, f"sf_table.py, as {NB10} cell 30", today)]
            for b in lb.BANDS[s]:
                sq, st = bt.loc[(bt.sensor == s) & (bt.band == b), ["sq_error", "standardized_sq_error"]].iloc[0]
                blines.append(f"| {model} | {skill} | {s} | {b} | {f6(sq)} | {f6(st)} |")
                reg.append((f"{key}_{b}", f"{f6(sq)}, {f6(st)}", f"mean sq_error and standardized_sq_error of band {b} "
                            f"over the processed {t} cells, {what}", src, f"sf_table.py, as {NB10} cell 30", today))
        n = summ["n_site_sensor"]
        cols += [str(n), f6(summ["mean_validation_nmse"]) if n else "–", f6(summ["median_validation_nmse"]) if n else "–"]
        reg.append((f"tbl_m_{model}_{skill.replace(' ', '_')}_all", ", ".join(cols[-3:]),
                    f"processed cells of both sensors, mean and median joint_validation_nmse over them, {what}", src,
                    f"sf_table.py, as {NB10} cell 28", today))
        lines.append(f"| {model} | {skill} | " + " | ".join(cols) + " |")
    (OUT / "table_monthly.md").write_text("\n".join(lines) + "\n")
    (OUT / "table_monthly_bands.md").write_text("\n".join(blines) + "\n")
    pd.DataFrame(reg, columns=["name", "value", "what it counts", "source file", "how it is computed", "date read"]
                 ).to_csv(OUT / "numbers_table_monthly.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
