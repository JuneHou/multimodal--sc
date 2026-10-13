"""lb_skill_score — score the skill arms and describe how each model operated the skill.

Reads lb_replies_shc_configure|shc_design|shc_implement_*.jsonl (main folder and runs/*), last record per
(method, prompt_id). Metric: lb_lib.METRIC on the SUBMITTED answer — per band, squared error at P10 over the
variance of that band across P01-P09, mean over bands (lb.her_nmse with the base prompt record's truth/train).
Writes
    lb_scores_skill.csv   per cell: experiment, method, sensor, site, ok, ended, steps, joint_nmse, answer vs last fit
    lb_skill_trace.csv    per cell: the sequence of fitter runs (settings/spec of each, blocks, weights, fit MSE),
                          whether SKILL.md / references were read, whether the data was shown, program sha (arm C)
    lb_skill_reasoning.csv  R1: reasoning-like keys per model and gateway across every stored raw response
Prints the summary beside the reference rows, and the R2 cap table.
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
    rows, trace, inv, cap, rlen = [], [], defaultdict(Counter), Counter(), defaultdict(list)
    for (method, pid), (r, f) in recs.items():
        p = base.get(pid[:-len(r["experiment"][4:]) - 1] + "_shc")
        if p is None:
            continue
        bands = list(p["truth"])
        joint = float(np.mean([lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b]) for b in bands])) if r.get("ok") else np.nan
        fits = r.get("fit_runs") or []
        last = fits[-1] if fits else None
        tol = 0.6 * 10.0 ** (-lb.DECIMALS[p["sensor"]])        # fit.py prints at the sensor's decimals; the model copies that
        same = (last is not None and r.get("values") is not None
                and all(abs(float(r["values"][b]) - float(last["predictions"][b])) <= tol for b in bands))
        rows.append({"experiment": r["experiment"], "method": method, "sensor": p["sensor"], "site_id": p["target_site"],
                     "ok": bool(r.get("ok")), "ended": r.get("ended"), "steps": r.get("n_steps"), lb.METRIC: joint,
                     "fit_runs": len(fits), "answer_equals_last_fit": same if last is not None else None,
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
    S = pd.DataFrame(rows); T = pd.DataFrame(trace)
    S.to_csv(lb.LB / "lb_scores_skill.csv", index=False); T.to_csv(lb.LB / "lb_skill_trace.csv", index=False)
    inv_rows = [{"model": m, "gateway": g, "key": k, "populated": pop, "n": n} for (m, g), c in inv.items() for (k, pop), n in c.items()]
    pd.DataFrame(inv_rows).to_csv(lb.LB / "lb_skill_reasoning.csv", index=False)
    if S.empty:
        print("no skill-arm replies found"); return
    print(f"\n{len(S)} cells. Metric: {lb.METRIC_TEXT}\n")
    summ = S[S.ok].groupby(["experiment", "method", "sensor"])[lb.METRIC].agg(n="size", mean="mean", median="median").reset_index()
    ok = S.groupby(["experiment", "method", "sensor"]).agg(cells=("ok", "size"), ok=("ok", "sum"), steps=("steps", "mean"),
                                                            fit_runs=("fit_runs", "mean")).reset_index()
    print(summ.merge(ok, on=["experiment", "method", "sensor"]).to_string(index=False))
    ref = pd.read_csv(lb.LB / "lb_summary_shc.csv")
    ref = ref[ref.method.isin(["shc", "persistence", "last3_mean"])][["method", "sensor", "n", "mean_joint_nmse", "median_joint_nmse"]]
    print("\nreference rows (lb_summary_shc.csv, same metric):\n" + ref.to_string(index=False))
    print("\nhow the skill was operated (per experiment and method): read SKILL.md / read references / showed data / answer = last fit")
    for (e, m), g in S.groupby(["experiment", "method"]):
        print(f"  {e:14s} {m:36s} SKILL.md {int(g.refs_read.str.contains('SKILL.md').sum()):2d}/{len(g)}  "
              f"references {int(g.refs_read.str.contains('references').sum()):2d}/{len(g)}  data {int(g.data_shown.sum()):2d}/{len(g)}  "
              f"answer=last fit {int(g.answer_equals_last_fit.fillna(False).sum()):2d}/{int(g.fit_runs.gt(0).sum())}  ended: {dict(Counter(g.ended))}")
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
