"""lb_skill_score — score the skill arms as the collaborator's notebook 10 scores its held-out P10, and describe how each
model operated the skill.

The evaluation is hers, from data/scripts/scripts/validation/shc/10_shc_monthly_multiple_outcomes.ipynb (functions
extracted in lb_her_nb10.py):
    per cell and band   her heldout_validation_for_sensor rows (cell 25): error = actual P10 - prediction, sq_error,
                        training_variance = nanvar(P01 to P09, ddof 1), NaN when not finite or below SD_TOL,
                        standardized_sq_error = sq_error / training_variance
    per cell            joint_validation_nmse = the pandas mean of standardized_sq_error over bands (cell 25); a cell
                        with no valid answer.json is a skipped site, as a site whose validation raises is skipped (cell 26)
    summaries           processed and skipped sites per sensor (cell 34); mean standardized_sq_error per sensor (cell 30,
                        mean_standardized_P10_sq_error); number, mean and median of joint_validation_nmse over the
                        processed sites of both sensors (cell 28, n_site_sensor, mean_validation_nmse,
                        median_validation_nmse); mean sq_error and standardized_sq_error per sensor and band (cell 30)
Reference rows apply the same summaries to her SHC as lb_shc.py runs it, P09 carried forward and the P07 to P09 mean
(lb_scores_shc.csv). Reads lb_replies_shc_configure|shc_design|shc_implement_*.jsonl (main folder and runs/*), last
record per (method, prompt_id).
Writes
    lb_scores_skill.csv        per cell: experiment, method, sensor, site, status, joint_validation_nmse, how the cell ended
                               and ran (steps, fitter runs, last fit, references read, data shown, python runs, seconds)
    lb_scores_skill_bands.csv  per cell and band: her validation columns
    lb_summary_skill.csv       her summaries per experiment and method, and for the reference rows
    lb_skill_trace.csv         per cell: the sequence of fitter runs (settings/spec of each, blocks, weights, fit MSE),
                               program sha (arm C)
    lb_skill_reasoning.csv     R1: reasoning-like keys per model and gateway across every stored raw response
Prints her summaries per experiment and method beside the reference rows, and the R2 cap table.
    python lb_skill_score.py
"""
import glob
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import lb_lib as lb

EXPERIMENTS = ("shc_configure", "shc_design", "shc_implement")
SENSORS = (("sentinel1", "Sentinel-1"), ("sentinel2", "Sentinel-2"))
REFERENCES = (("shc", "her SHC"), ("persistence", "P09 carried forward"), ("last3_mean", "P07 to P09 mean"))


def load():
    base = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")}
    recs = {}
    files = []
    for exp in EXPERIMENTS:
        files += glob.glob(str(lb.LB / f"lb_replies_{exp}_*.jsonl")) + [f for f in glob.glob(str(lb.LB / "runs" / "*" / f"lb_replies_{exp}_*.jsonl")) if "/superseded_" not in f]
    for f in sorted(files):
        for l in open(f):
            r = json.loads(l)
            eff = r.get("reasoning_effort", "low")
            method = r["model"] if eff == "low" else f"{r['model']} ({eff} effort)"
            recs[(method, r["prompt_id"])] = (r, f)
    return base, recs


def her_heldout_rows(base, prediction):
    """Her cell 25 rows for one cell, one per band in her band order and with her band names; prediction None = no answer."""
    rows = []
    for b in lb.BANDS[base["sensor"]]:
        d = lb.disp(b)
        actual = float(base["truth"][d])
        synthetic = float(prediction[d]) if prediction is not None else np.nan
        if np.isfinite(actual) and np.isfinite(synthetic):
            error = actual - synthetic
            sq_error = float(error ** 2)
        else:
            synthetic = error = sq_error = np.nan
        train_var = float(np.nanvar(np.asarray(base["train"][d], dtype=float), ddof=1))
        if not np.isfinite(train_var) or train_var < lb.SD_TOL:
            train_var = np.nan
        std = sq_error / train_var if (np.isfinite(sq_error) and np.isfinite(train_var)) else np.nan
        rows.append({"band": b, "actual_P10": actual, "synthetic_P10": synthetic, "error": error, "sq_error": sq_error,
                     "training_variance": train_var, "standardized_sq_error": std})
    return rows


def joint_validation_nmse(rows):
    """Her cell 25: the pandas mean of standardized_sq_error over the cell's bands."""
    return float(pd.Series([r["standardized_sq_error"] for r in rows], dtype=float).mean())


def reference_rows(base):
    """Her cell 25 rows of the reference methods, from their per-band scores in lb_scores_shc.csv."""
    ref = pd.read_csv(lb.LB / "lb_scores_shc.csv")
    rows = []
    for m, label in REFERENCES:
        for _, s in ref[ref.method == m].iterrows():
            b = base[f"{'s1' if s.sensor == 'sentinel1' else 's2'}_{s.site_id}_shc"]
            for v in her_heldout_rows(b, None):
                std = float(s[f"nmse_{lb.disp(v['band'])}"])
                v.update(standardized_sq_error=std, sq_error=std * v["training_variance"])
                rows.append({"model": "reference", "skill": label, "sensor": s.sensor, "site_id": s.site_id,
                             "status": "processed" if s.ok else "skipped", **v})
    return rows


def her_summary(g):
    """Her held-out summaries for one arm; g holds its rows per cell and band with status and joint_validation_nmse."""
    out, bands = {}, []
    for s, _ in SENSORS:
        h = g[g.sensor == s]
        sites = h.drop_duplicates("site_id")
        out[f"{s}_processed"] = int((sites.status == "processed").sum())
        out[f"{s}_skipped"] = int((sites.status == "skipped").sum())
        out[f"{s}_mean_standardized_P10_sq_error"] = h.loc[h.status == "processed", "standardized_sq_error"].mean()
        for b in lb.BANDS[s]:
            hb = h[(h.band == b) & (h.status == "processed")]
            bands.append({"sensor": s, "band": b, "sq_error": hb.sq_error.mean(), "standardized_sq_error": hb.standardized_sq_error.mean()})
    j = g[g.status == "processed"].drop_duplicates(["sensor", "site_id"]).joint_validation_nmse
    out.update(n_site_sensor=len(j), mean_validation_nmse=j.mean(), median_validation_nmse=j.median())
    return out, pd.DataFrame(bands)


def walk_keys(obj, hits, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(w in k.lower() for w in ("reason", "think", "thought")):
                hits.append((f"{path}.{k}".lstrip("."), bool(v) if not isinstance(v, (int, float)) else v != 0))
            walk_keys(v, hits, f"{path}.{k}")
    elif isinstance(obj, list):
        for v in obj:
            walk_keys(v, hits, path + "[]")


def main():
    base, recs = load()
    rows, bands, trace, inv, cap, rlen = [], [], [], defaultdict(Counter), Counter(), defaultdict(list)
    for (method, pid), (r, f) in recs.items():
        p = base.get(pid[:-len(r["experiment"][4:]) - 1] + "_shc")
        if p is None:
            continue
        status = "processed" if r.get("ok") else "skipped"
        hrows = her_heldout_rows(p, r["values"] if r.get("ok") else None)
        joint = joint_validation_nmse(hrows) if r.get("ok") else np.nan
        for v in hrows:
            bands.append({"experiment": r["experiment"], "method": method, "sensor": p["sensor"], "site_id": p["target_site"],
                          "status": status, **v, "joint_validation_nmse": joint})
        fits = r.get("fit_runs") or []
        last = fits[-1] if fits else None
        rows.append({"experiment": r["experiment"], "method": method, "sensor": p["sensor"], "site_id": p["target_site"],
                     "status": status, "joint_validation_nmse": joint, "ended": r.get("ended"), "steps": r.get("n_steps"),
                     "fit_runs": len(fits),
                     "last_fit_blocks": " ".join(map(str, last["donor_ids"])) if last else "",
                     "last_fit_weights": " ".join(f"{w:.4f}" for w in last["weights"]) if last else "",
                     "last_fit_mse": last["fit_mse"] if last else np.nan, "refs_read": " ".join(r.get("refs_read") or []),
                     "data_shown": r.get("data_shown"), "python_runs": r.get("python_runs"), "seconds": r.get("seconds"),
                     "file": f.split("/")[-1]})
        for i, fr in enumerate(fits, 1):
            trace.append({"experiment": r["experiment"], "method": method, "sensor": p["sensor"], "site_id": p["target_site"],
                          "run": i, **{k: fr["spec"].get(k) for k in fr["spec"]}, "donor_ids": " ".join(map(str, fr["donor_ids"])),
                          "weights": " ".join(f"{w:.4f}" for w in fr["weights"]), "fit_mse": fr["fit_mse"],
                          "pred": json.dumps({k: round(v, 4) for k, v in fr["predictions"].items()})})
        if r.get("program"):
            trace.append({"experiment": r["experiment"], "method": method, "sensor": p["sensor"], "site_id": p["target_site"],
                          "run": "program", "program_sha256": hashlib.sha256(r["program"].encode()).hexdigest()})
        for s in r.get("steps") or []:
            u = s.get("usage") or {}
            hits = []
            walk_keys(u.get("raw_response"), hits)
            for k, populated in hits:
                inv[(r["model"], u.get("gateway", r.get("gateway")))][(k, populated)] += 1
            if u:
                cap[(r["experiment"], method, u.get("finish_reason"))] += 1
                if isinstance(u.get("reasoning"), str):
                    rlen[(r["experiment"], method)].append(len(u["reasoning"]))
    S = pd.DataFrame(rows); B = pd.DataFrame(bands); T = pd.DataFrame(trace)
    S.to_csv(lb.LB / "lb_scores_skill.csv", index=False); B.to_csv(lb.LB / "lb_scores_skill_bands.csv", index=False)
    T.to_csv(lb.LB / "lb_skill_trace.csv", index=False)
    inv_rows = [{"model": m, "gateway": g, "key": k, "populated": pop, "n": n} for (m, g), c in inv.items() for (k, pop), n in c.items()]
    pd.DataFrame(inv_rows).to_csv(lb.LB / "lb_skill_reasoning.csv", index=False)
    if S.empty:
        print("no skill-arm replies found"); return
    print(f"\n{len(S)} cells, scored as notebook 10 scores its held-out P10\n")
    R = pd.DataFrame(reference_rows(base))
    summ = []
    for (e, m), g in B.groupby(["experiment", "method"]):
        summ.append({"experiment": e, "method": m, **her_summary(g)[0]})
    for m, label in REFERENCES:
        g = R[R.skill == label].copy()
        g["joint_validation_nmse"] = g.groupby(["sensor", "site_id"])["standardized_sq_error"].transform("mean")
        summ.append({"experiment": "reference", "method": label, **her_summary(g)[0]})
    pd.DataFrame(summ).to_csv(lb.LB / "lb_summary_skill.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nhow the skill was operated (per experiment and method): read SKILL.md / read references / showed data / ended")
    for (e, m), g in S.groupby(["experiment", "method"]):
        print(f"  {e:14s} {m:36s} SKILL.md {int(g.refs_read.str.contains('SKILL.md').sum()):2d}/{len(g)}  "
              f"references {int(g.refs_read.str.contains('references').sum()):2d}/{len(g)}  data {int(g.data_shown.sum()):2d}/{len(g)}  "
              f"ended: {dict(Counter(g.ended))}")
    print("\nR2 cap accounting — finish_reason over all turns:")
    for (e, m, fr), n in sorted(cap.items()):
        rl = rlen.get((e, m), [])
        print(f"  {e:14s} {m:36s} {str(fr):8s} {n:4d}   median reasoning chars {int(np.median(rl)) if rl else '-'}")
    print("\nR1 reasoning inventory — keys in the raw responses (populated / total):")
    if inv_rows:
        I = pd.DataFrame(inv_rows)
        for (m, g), grp in I.groupby(["model", "gateway"]):
            print(f"  {m:36s} {g:7s} " + "; ".join(f"{k}: {int(grp[(grp.key == k) & grp.populated].n.sum())}/{int(grp[grp.key == k].n.sum())}" for k in sorted(grp.key.unique())))


if __name__ == "__main__":
    main()
