"""lb_arms — the complete direct-prompt vs step-by-step comparison, every model, both sensors.

Arms: `shc` (the direct prompt: band guide, the site's P01-P09 table, one-line answer format) and
`shc_steps` (the same plus the collaborator's block/weight-fitting procedure written out in prose).
Metric throughout: her outcome metric, squared error at P10 over the band's variance, mean over bands.
Every reported figure uses `lb.METRIC` = `joint_nmse`, the P01-P09 denominator her stored per-site
values use. `joint_nmse_p10var` is still computed and never reported. Lower is better.

Where each arm's replies live, and how many runs each has:
  shc        open-weight : runs/run1_2026-09-10/, runs/run2_2026-09-15/      (2 runs)
             gpt-5.5     : runs/gpt55_run1_2026-09-17/, gpt55_run2_...       (2 runs)
  shc_steps  open-weight : the main folder, run of 2026-09-15/16              (1 run)
             gpt-5.5     : runs/gpt55_steps_run1_2026-09-19/                  (1 run)

Two accuracy tables are printed, and the difference between them matters:
  ALL SITES   every site the arm answered. Not a like-for-like comparison when the arms answered
              different numbers of sites (Kimi answers 26 direct, 19 with steps; GLM 3 and 0).
  MATCHED     only sites answered by that model in BOTH arms. This is the comparison to read;
              the all-sites S2 figure for a model that dropped cells is survivorship, not skill.

    python lb_arms.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lb_lib as lb

RUNS = lb.LB / "runs"
ARMS = ("shc", "shc_steps", "shc_configure", "shc_design", "shc_implement")


def prompts_for(exp, _c={}):
    if exp not in _c:
        _c[exp] = {json.loads(l)["prompt_id"]: json.loads(l)
                   for l in open(lb.LB / f"lb_prompts_{exp}.jsonl")}
    return _c[exp]


def sources():
    """(run label, file) for every reply file that belongs to one of the two arms.
    The main folder's `shc` files are a copy of run2 and are skipped, so no run is counted twice."""
    out = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        out += [(d.name, f) for f in sorted(d.glob("lb_replies_shc*.jsonl"))]
    out += [("main_2026-09-15", f) for f in sorted(lb.LB.glob("lb_replies_shc_steps_*.jsonl"))]
    return out


def load():
    rows = []
    for run, f in sources():
        for line in open(f):
            r = json.loads(line)
            exp = r.get("experiment")
            if exp not in ARMS:
                continue
            model, ok = r["model"], bool(r.get("ok"))
            u = r.get("usage") or {}
            # a failed call has no top-level usage: its finish reason / error is on the last attempt
            last = (r.get("attempts") or [{}])[-1]
            why = u.get("finish_reason") or (last.get("usage") or {}).get("finish_reason") \
                or (str(last.get("error", "")).split(":")[0] or None)
            rec = {"arm": exp, "model": model, "run": run, "prompt_id": r["prompt_id"], "ok": ok,
                   "finish": why,
                   "completion_tokens": u.get("completion_tokens", np.nan)}
            p = prompts_for(exp)[r["prompt_id"]]
            rec["sensor"], rec["site_id"] = p["sensor"], p["target_site"]
            if ok:
                bands = [lb.disp(b) for b in lb.BANDS[p["sensor"]]]
                rec["joint_nmse"] = float(np.mean(
                    [lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b]) for b in bands]))
                rec["joint_nmse_p10var"] = float(np.mean(
                    [lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b] + [p["truth"][b]])
                     for b in bands]))
            rows.append(rec)
    return pd.DataFrame(rows)


def per_site(df):
    """One value per (arm, model, sensor, site): the mean over the runs that answered it."""
    g = df[df.ok].groupby(["arm", "model", "sensor", "site_id"])
    return g.agg(v=(lb.METRIC, "mean"), v_p10var=("joint_nmse_p10var", "mean"),
                 n_runs=("run", "nunique")).reset_index()


def fmt(x, n=3):
    return "—" if pd.isna(x) else f"{x:.{n}f}"


def main():
    df = load()
    ps = per_site(df)
    S = pd.read_csv(lb.LB / "lb_scores_shc.csv")
    ok = S[S.ok == True]
    ref = ok[ok.method == "shc"].set_index(["sensor", "site_id"])[lb.METRIC]
    own = ok[ok.method == "own_mean"].set_index(["sensor", "site_id"])[lb.METRIC]
    N = {"sentinel1": 12, "sentinel2": 15}
    # every model actually present, in a preferred order; unknown names (e.g. a new `gateway:model`)
    # are appended rather than dropped, so a new gateway cannot silently vanish from the tables
    order = ["gpt-oss-120b", "Kimi-K3-thinking-low", "GLM-5.3", "DeepSeek-V4-Flash-thinking-low",
             "gpt-5.5", "hokie:gpt-5.5", "openai:gpt-5.5"]
    present = set(df.model)
    models = [m for m in order if m in present] + sorted(present - set(order))
    pd.set_option("display.width", 250)

    # ---------- 1. completion ----------
    print("\n" + "=" * 96)
    print("TABLE 1 — CELLS ANSWERED (of 12 Sentinel-1 / 15 Sentinel-2), per run")
    print("=" * 96)
    t = []
    for m in models:
        for arm in ARMS:
            d = df[(df.model == m) & (df.arm == arm)]
            if not len(d):
                continue
            row = {"model": m, "arm": arm, "runs": d.run.nunique()}
            for s in ("sentinel1", "sentinel2"):
                a = d[(d.sensor == s)]
                row[s] = "—" if not len(a) else f"{int(a.ok.sum() / max(a.run.nunique(),1))} / {N[s]}"
            fin = d[~d.ok].finish.fillna("no reply").value_counts().to_dict()
            row["failures (all runs)"] = ", ".join(f"{k}×{v}" for k, v in fin.items()) if fin else "none"
            t.append(row)
    print(pd.DataFrame(t).to_string(index=False))

    # ---------- 2. accuracy, all answered sites ----------
    print("\n" + "=" * 96)
    print("TABLE 2 — ACCURACY ON ALL ANSWERED SITES (her metric, P01–P10 variance; lower better)")
    print("         *not* like-for-like where the two arms answered different site counts")
    print("=" * 96)
    t = []
    for s in ("sentinel1", "sentinel2"):
        for m in models:
            for arm in ARMS:
                g = ps[(ps.model == m) & (ps.arm == arm) & (ps.sensor == s)]
                if not len(g):
                    continue
                beats = int((g.set_index("site_id").v < ref.loc[s].loc[g.site_id]).sum())
                t.append({"sensor": s, "model": m, "arm": arm, "sites": len(g),
                          "mean": fmt(g.v.mean()), "median": fmt(g.v.median()),
                          "beats her SHC": f"{beats} / {len(g)}"})
        for label, series in (("her SHC", ref), ("own mean P01–P09", own)):
            v = series.loc[s]
            t.append({"sensor": s, "model": label, "arm": "—", "sites": len(v),
                      "mean": fmt(v.mean()), "median": fmt(v.median()), "beats her SHC": ""})
    print(pd.DataFrame(t).to_string(index=False))

    # ---------- 3. matched sites ----------
    print("\n" + "=" * 96)
    print("TABLE 3 — STEP-BY-STEP vs DIRECT, MATCHED SITES ONLY (both arms answered the site)")
    print("         this is the like-for-like comparison")
    print("=" * 96)
    t = []
    for s in ("sentinel1", "sentinel2"):
        for m in models:
            a = ps[(ps.model == m) & (ps.arm == "shc") & (ps.sensor == s)].set_index("site_id")
            b = ps[(ps.model == m) & (ps.arm == "shc_steps") & (ps.sensor == s)].set_index("site_id")
            common = sorted(set(a.index) & set(b.index))
            if not common:
                continue
            da, db = a.loc[common].v, b.loc[common].v
            better = int((db < da).sum())
            t.append({"sensor": s, "model": m, "matched sites": f"{len(common)} / {N[s]}",
                      "direct": fmt(da.mean()), "steps": fmt(db.mean()),
                      "change": fmt(db.mean() - da.mean()),
                      "%": f"{100*(db.mean()-da.mean())/da.mean():+.1f}%",
                      "steps better at": f"{better} / {len(common)}",
                      "her SHC": fmt(ref.loc[s].loc[common].mean())})
    print(pd.DataFrame(t).to_string(index=False))

    # ---------- 4. token spend ----------
    print("\n" + "=" * 96)
    print("TABLE 4 — COMPLETION TOKENS PER ANSWERED CALL")
    print("         median and mean both shown: steps make gpt-oss BIMODAL (median falls, tail")
    print("         explodes), so either statistic alone misrepresents it")
    print("=" * 96)
    t = []
    for m in models:
        row = {"model": m}
        for arm, tag in ((a, a.replace("shc_steps", "steps").replace("shc", "direct")) for a in ARMS):
            d = df[(df.model == m) & (df.arm == arm) & df.ok].completion_tokens.dropna()
            row[f"{tag} median"] = fmt(d.median(), 0) if len(d) else "—"
            row[f"{tag} mean"] = fmt(d.mean(), 0) if len(d) else "—"
            row[f"{tag} max"] = fmt(d.max(), 0) if len(d) else "—"
        t.append(row)
    print(pd.DataFrame(t).to_string(index=False))

    df.to_csv(lb.LB / "lb_arms_cells.csv", index=False)
    ps.to_csv(lb.LB / "lb_arms_per_site.csv", index=False)
    print(f"\nwritten: lb_arms_cells.csv ({len(df)} records), lb_arms_per_site.csv ({len(ps)} site values)")


if __name__ == "__main__":
    main()
