"""lb_cases — why the step-by-step prompt changes the answer, and eight worked cases.

Companion to lb_arms.py, which says WHAT the two prompting arms score. This says WHY.

The claim under test: the step-by-step prompt collapses the model onto "repeat the last observed
month", which is what the collaborator's band-space SHC does at 26 of 27 cells (its stepwise donor
selection always keeps the most recent block). If so, "steps help Sentinel-2 and hurt Sentinel-1"
is not a fact about steps -- it is a fact about whether carrying P09 forward beats the model's own
instinct on that sensor.

The claim is checked TWO independent ways, which must agree or it is withdrawn:
  A. carry-forward distance -- |prediction - P09| in units of the site's own pre-period SD,
     computed from the predictions themselves;
  B. against the `persistence` method already scored in lb_scores_shc.csv -- if the step-by-step arm
     has collapsed onto P09, its per-site error must match persistence's per-site error.
A is about the predictions, B is about the errors, and neither is derived from the other.

    python lb_cases.py              # mechanism test + all eight cases
    python lb_cases.py mechanism    # just the test
"""
import json
import sys

import numpy as np
import pandas as pd

import lb_arms as la
import lb_lib as lb

# One model per family for the walkthroughs: the frontier model (OpenAI snapshot, complete in both
# arms), the richest raw chain, and the open-weight model that never truncates.
CASE_MODELS = ["openai:gpt-5.5", "Kimi-K3-thinking-low", "gpt-oss-120b"]

# sensor, arm, verdict, site -- chosen from the per-site errors, not picked for effect.
# Site 0022 appears twice on purpose: the same series failing under the direct prompt and rescued
# under the step-by-step prompt is the mechanism visible on one site.
CASES = [
    ("sentinel1", "shc",       "RIGHT", "treatment_0020"),
    ("sentinel1", "shc",       "WRONG", "treatment_0022"),
    ("sentinel1", "shc_steps", "RIGHT", "treatment_0022"),
    ("sentinel1", "shc_steps", "WRONG", "treatment_0027"),
    ("sentinel2", "shc",       "RIGHT", "treatment_0014"),
    ("sentinel2", "shc",       "WRONG", "treatment_0004"),
    ("sentinel2", "shc_steps", "RIGHT", "treatment_0012"),
    ("sentinel2", "shc_steps", "WRONG", "treatment_0003"),
]
ARM_NAME = {"shc": "direct prompt", "shc_steps": "step-by-step prompt"}


def load():
    """(model, arm, prompt_id) -> record, keeping the last run that answered it.

    la.sources() yields runs/* in sorted order then the main folder, so the later, reasoning-capturing
    run wins for the ARC models and the single OpenAI run wins for openai:gpt-5.5."""
    keep, meta = {}, {}
    for run, f in la.sources():
        for line in open(f):
            r = json.loads(line)
            exp = r.get("experiment")
            if exp not in la.ARMS:
                continue
            p = la.prompts_for(exp)[r["prompt_id"]]
            meta[r["prompt_id"]] = p
            k = (r["model"], exp, r["prompt_id"])
            if r.get("ok") or k not in keep:
                keep[k] = (r, run)
    return keep, meta


def trace(rec):
    """(text, kind) for a record's reasoning: ARC returns the raw chain, OpenAI only a summary."""
    u = rec.get("usage") or ((rec.get("attempts") or [{}])[-1].get("usage") or {})
    t = u.get("reasoning")
    return (t or ""), ("summary" if u.get("reasoning_kind") == "summary" else "raw chain")


def cf_distance(values, p, bands):
    """|prediction - P09| per band, in units of that band's own pre-period SD, averaged over bands.
    0 means the prediction is exactly the last observed month carried forward."""
    out = []
    for b in bands:
        train = p["train"][b]
        sd = float(np.std(train, ddof=1))
        if sd > lb.SD_TOL:
            out.append(abs(values[b] - train[-1]) / sd)
    return float(np.mean(out)) if out else np.nan


def mechanism(keep, meta):
    print("=" * 96)
    print("CHECK A — DISTANCE FROM CARRYING P09 FORWARD (0 = the last observed month, repeated)")
    print("          in units of each band's own pre-period SD, averaged over bands then over sites")
    print("=" * 96)
    rows = []
    for (model, arm, pid), (r, run) in keep.items():
        if not r.get("ok"):
            continue
        p = meta[pid]
        bands = [lb.disp(b) for b in lb.BANDS[p["sensor"]]]
        rows.append({"model": model, "arm": arm, "sensor": p["sensor"],
                     "site_id": p["target_site"], "d": cf_distance(r["values"], p, bands)})
    D = pd.DataFrame(rows)
    piv = D.pivot_table(index=["sensor", "model"], columns="arm", values="d", aggfunc="mean")
    piv.columns = [f"{ARM_NAME[c]} (mean)" for c in piv.columns]
    med = D.pivot_table(index=["sensor", "model"], columns="arm", values="d", aggfunc="median")
    for c in med.columns:
        piv[f"{ARM_NAME[c]} (median)"] = med[c]
    print(piv.round(3).to_string())

    print("\n" + "=" * 96)
    print("CHECK B — THE STEP-BY-STEP ARM AGAINST THE `persistence` REFERENCE, SITE BY SITE")
    print("          persistence = P09 carried forward, already scored independently")
    print("=" * 96)
    S = pd.read_csv(lb.LB / "lb_scores_shc.csv")
    ok = S[S.ok == True]
    per = ok[ok.method == "persistence"].set_index(["sensor", "site_id"])[lb.METRIC]
    shc = ok[ok.method == "shc"].set_index(["sensor", "site_id"])[lb.METRIC]
    out = []
    for (model, arm, pid), (r, run) in keep.items():
        if not r.get("ok"):
            continue
        p = meta[pid]
        bands = [lb.disp(b) for b in lb.BANDS[p["sensor"]]]
        v = float(np.mean([lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b] + [p["truth"][b]])
                           for b in bands]))
        key = (p["sensor"], p["target_site"])
        out.append({"model": model, "arm": arm, "sensor": p["sensor"], "site_id": p["target_site"],
                    "v": v, "persistence": per.get(key, np.nan), "her_shc": shc.get(key, np.nan)})
    O = pd.DataFrame(out)
    t = []
    for (sensor, model, arm), g in O.groupby(["sensor", "model", "arm"]):
        t.append({"sensor": sensor, "model": model, "arm": ARM_NAME[arm], "sites": len(g),
                  "arm mean": g.v.mean(), "persistence mean": g.persistence.mean(),
                  "mean |arm − persistence|": (g.v - g.persistence).abs().mean(),
                  "sites within 0.01": int(((g.v - g.persistence).abs() < 0.01).sum())})
    T = pd.DataFrame(t).sort_values(["sensor", "model", "arm"])
    print(T.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nReading: if the step-by-step rows sit at ~0 mean absolute difference from persistence")
    print("while the direct rows do not, the two checks agree and the mechanism holds.")
    return D, O


def show_case(keep, meta, sensor, arm, verdict, site, S):
    pid = next(k[2] for k in keep if k[2].startswith(f"{lb.SHORT[sensor]}_{site}")
               and keep[k][0].get("experiment") == arm)
    p = meta[pid]
    bands = [lb.disp(b) for b in lb.BANDS[sensor]]
    d = lb.DECIMALS[sensor]
    print("\n" + "=" * 96)
    print(f"{verdict}  —  {sensor}, site {site}, {ARM_NAME[arm]}")
    print("=" * 96)
    print("  input, the site's own months P01–P09" + " " * 14 + "  ".join(f"{b:>8s}" for b in bands))
    for i in range(len(p["train"][bands[0]])):
        print(f"    P{i+1:02d}" + " " * 44 + "  ".join(f"{p['train'][b][i]:8.{d}f}" for b in bands))
    print(f"    P10 WITHHELD TRUTH" + " " * 30 + "  ".join(f"{p['truth'][b]:8.{d}f}" for b in bands))
    print()
    ok_rows = []
    for model in CASE_MODELS:
        r, run = keep.get((model, arm, pid), (None, None))
        if r is None:
            print(f"    {model:24s} not run"); continue
        if not r.get("ok"):
            u = (r.get("attempts") or [{}])[-1].get("usage") or {}
            print(f"    {model:24s} NO ANSWER ({u.get('finish_reason')})")
            ok_rows.append((model, r, run, None)); continue
        v = float(np.mean([lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b] + [p["truth"][b]])
                           for b in bands]))
        print(f"    {model:24s}" + " " * 20 + "  ".join(f"{r['values'][b]:8.{d}f}" for b in bands)
              + f"   error {v:.3f}   distance from P09 {cf_distance(r['values'], p, bands):.3f}")
        ok_rows.append((model, r, run, v))
    ok = S[S.ok == True]
    for label, method in (("band-space SHC", "shc"), ("persistence (P09)", "persistence"),
                          ("mean of last 3 months", "last3_mean")):
        row = ok[(ok.method == method) & (ok.sensor == sensor) & (ok.site_id == site)]
        if len(row):
            print(f"    {label:24s}" + " " * 20 + " " * (10 * len(bands))
                  + f"   error {float(row[lb.METRIC].iloc[0]):.3f}")
    print()
    for model, r, run, v in ok_rows:
        t, kind = trace(r)
        if not t:
            print(f"    — {model} ({kind}): none returned"); continue
        body = " ".join(t.split())
        print(f"    — {model} [{kind}, {len(t)} chars, run {run}]:")
        print(f"        {body[:520]}{'…' if len(body) > 520 else ''}")


def main():
    keep, meta = load()
    D, O = mechanism(keep, meta)
    if len(sys.argv) > 1 and sys.argv[1] == "mechanism":
        return
    S = pd.read_csv(lb.LB / "lb_scores_shc.csv")
    for sensor, arm, verdict, site in CASES:
        show_case(keep, meta, sensor, arm, verdict, site, S)


if __name__ == "__main__":
    main()
