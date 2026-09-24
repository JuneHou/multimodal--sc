"""lb_why — why the predictions are wrong: a failure analysis, not a model ranking.

Companion to lb_arms.py (what each arm scores) and lb_cases.py (the mechanism and worked cases).
This asks, of every wrong answer: what kind of wrong is it?

Five diagnostics, each isolating one candidate cause:

  1 NO ANSWER      the call never produced a parsable line at all -- by reason.
  2 DIRECTION      did the withheld month continue the recent direction, or reverse it? A model that
                   extrapolates is wrong exactly when the series turns, so error is split by whether
                   P10 continued or reversed the P08->P09 move.
  3 BIAS vs SPREAD signed error per band: a systematic offset is a different fault from noise, and
                   only the first is fixable by a prompt.
  4 BAND SHARE     which bands carry the squared error. The metric is a mean over bands, so one
                   band can own a site's score.
  5 SELF-CONSISTENCY  whether a prediction contradicts itself. VV-VH must equal VV minus VH. NDVI and
                   NDWI have algebraic definitions in B4/B8 and B3/B8 -- but the observed data does
                   NOT satisfy them, because each index is formed per day and the monthly composite
                   takes each band's median independently. So the test is comparative: does the model
                   impose an identity the data does not have? Enforcing false algebra is its own
                   error source, and it is invisible in the joint metric.

    python lb_why.py
"""
import json

import numpy as np
import pandas as pd

import lb_arms as la
import lb_lib as lb

MODELS = ["openai:gpt-5.5", "gpt-5.5", "Kimi-K3-thinking-low", "gpt-oss-120b", "GLM-5.3"]
ARM = {"shc": "direct", "shc_steps": "steps"}


def records():
    """Every reply of the two arms, with its prompt, keeping the last run that answered."""
    keep = {}
    for run, f in la.sources():
        for line in open(f):
            r = json.loads(line)
            if r.get("experiment") not in la.ARMS:
                continue
            k = (r["model"], r["experiment"], r["prompt_id"])
            if r.get("ok") or k not in keep:
                keep[k] = (r, run)
    for (model, arm, pid), (r, run) in keep.items():
        yield model, arm, pid, r, la.prompts_for(arm)[pid]


def d1_no_answer():
    print("=" * 100)
    print("1  NO ANSWER — the call never produced a parsable line")
    print("=" * 100)
    rows = []
    for model, arm, pid, r, p in records():
        if r.get("ok"):
            continue
        last = (r.get("attempts") or [{}])[-1]
        why = (last.get("usage") or {}).get("finish_reason") or str(last.get("error", "?")).split(":")[0]
        rows.append({"model": model, "arm": ARM[arm], "sensor": p["sensor"], "reason": why})
    if not rows:
        print("  none"); return
    D = pd.DataFrame(rows)
    print(pd.crosstab([D.model, D.arm], [D.sensor, D.reason]).to_string())
    print("\n  'length'     = the output budget was spent on reasoning before any answer was written")
    print("  'repetition' = a degenerate loop in the reasoning")
    print("  'HTTP 400'   = the model name is not served by the gateway")


def d2_direction():
    print("\n" + "=" * 100)
    print("2  DIRECTION — error when the withheld month CONTINUED the last move, vs REVERSED it")
    print("=" * 100)
    rows = []
    for model, arm, pid, r, p in records():
        if not r.get("ok") or model not in MODELS:
            continue
        bands = [lb.disp(b) for b in lb.BANDS[p["sensor"]]]
        cont, rev, err = [], [], []
        for b in bands:
            tr = p["train"][b]
            last_move = tr[-1] - tr[-2]
            truth_move = p["truth"][b] - tr[-1]
            e = lb.her_nmse(p["truth"][b], r["values"][b], tr + [p["truth"][b]])
            if not np.isfinite(e):
                continue
            (cont if last_move * truth_move > 0 else rev).append(e)
            err.append(e)
        rows.append({"model": model, "arm": ARM[arm], "sensor": p["sensor"],
                     "continued": np.mean(cont) if cont else np.nan,
                     "reversed": np.mean(rev) if rev else np.nan,
                     "n_cont": len(cont), "n_rev": len(rev)})
    D = pd.DataFrame(rows)
    t = D.groupby(["sensor", "model", "arm"]).agg(
        err_continued=("continued", "mean"), err_reversed=("reversed", "mean"),
        bands_continued=("n_cont", "sum"), bands_reversed=("n_rev", "sum")).reset_index()
    t["ratio rev/cont"] = t.err_reversed / t.err_continued
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  A model that projects the recent direction forward is wrong precisely when the series")
    print("  turns. ratio >> 1 means the error is concentrated in the reversals.")


def d3_bias():
    print("\n" + "=" * 100)
    print("3  BIAS vs SPREAD — signed error, in units of the band's own pre-period SD")
    print("=" * 100)
    rows = []
    for model, arm, pid, r, p in records():
        if not r.get("ok") or model not in MODELS:
            continue
        for b in [lb.disp(x) for x in lb.BANDS[p["sensor"]]]:
            sd = float(np.std(p["train"][b], ddof=1))
            if sd > lb.SD_TOL:
                rows.append({"model": model, "arm": ARM[arm], "sensor": p["sensor"], "band": b,
                             "z": (r["values"][b] - p["truth"][b]) / sd})
    D = pd.DataFrame(rows)
    t = D.groupby(["sensor", "model", "arm"]).z.agg(
        bias="mean", spread="std", n="size").reset_index()
    t["|bias| / spread"] = t.bias.abs() / t.spread
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n  bias near 0 with large spread = unbiased but noisy; the error is not a fixable offset.")
    return D


def d4_bands(D):
    print("\n" + "=" * 100)
    print("4  BAND SHARE — share of the total squared error carried by each band (Sentinel-2)")
    print("=" * 100)
    rows = []
    for model, arm, pid, r, p in records():
        if not r.get("ok") or p["sensor"] != "sentinel2" or model not in MODELS:
            continue
        for b in [lb.disp(x) for x in lb.BANDS["sentinel2"]]:
            e = lb.her_nmse(p["truth"][b], r["values"][b], p["train"][b] + [p["truth"][b]])
            if np.isfinite(e):
                rows.append({"model": model, "arm": ARM[arm], "band": b, "e": e})
    B = pd.DataFrame(rows)
    piv = B.pivot_table(index=["model", "arm"], columns="band", values="e", aggfunc="mean")
    piv = piv[[lb.disp(x) for x in lb.BANDS["sentinel2"]]]
    share = piv.div(piv.sum(axis=1), axis=0) * 100
    print("mean error per band:")
    print(piv.round(2).to_string())
    print("\npercent of each row's total error:")
    print(share.round(1).to_string())


def d5_consistency():
    print("\n" + "=" * 100)
    print("5  SELF-CONSISTENCY — does a prediction obey algebra the OBSERVED DATA does not obey?")
    print("=" * 100)
    s1, s2 = [], []
    for model, arm, pid, r, p in records():
        if not r.get("ok") or model not in MODELS:
            continue
        v, tr, th = r["values"], p["train"], p["truth"]
        if p["sensor"] == "sentinel1":
            s1.append({"model": model, "arm": ARM[arm],
                       "pred": abs(v["VV"] - v["VH"] - v["VV-VH"]),
                       "truth": abs(th["VV"] - th["VH"] - th["VV-VH"]),
                       "input": float(np.mean([abs(tr["VV"][i] - tr["VH"][i] - tr["VV-VH"][i])
                                               for i in range(len(tr["VV"]))]))})
        else:
            def ndvi(d): return (d["B8"] - d["B4"]) / (d["B8"] + d["B4"])
            def ndwi(d): return (d["B3"] - d["B8"]) / (d["B3"] + d["B8"])
            ti = [{k: tr[k][i] for k in tr} for i in range(len(tr["B2"]))]
            s2.append({"model": model, "arm": ARM[arm],
                       "pred NDVI": abs(ndvi(v) - v["NDVI"]), "pred NDWI": abs(ndwi(v) - v["NDWI"]),
                       "truth NDVI": abs(ndvi(th) - th["NDVI"]), "truth NDWI": abs(ndwi(th) - th["NDWI"]),
                       "input NDVI": float(np.mean([abs(ndvi(x) - x["NDVI"]) for x in ti])),
                       "input NDWI": float(np.mean([abs(ndwi(x) - x["NDWI"]) for x in ti]))})
    print("Sentinel-1, |VV - VH - (VV-VH)| — an exact identity in the data by construction:")
    print(pd.DataFrame(s1).groupby(["model", "arm"]).mean().round(4).to_string())
    print("\nSentinel-2, |index recomputed from bands - index as given|.")
    print("The indices are formed per day and each band is composited independently, so the OBSERVED")
    print("rows do not satisfy the formula. 'input'/'truth' are how far the real data departs from it;")
    print("'pred' is how far the model's answer does. pred << truth means the model imposed the")
    print("textbook formula on data that does not follow it.")
    print(pd.DataFrame(s2).groupby(["model", "arm"]).mean().round(4).to_string())


def main():
    d1_no_answer()
    d2_direction()
    D = d3_bias()
    d4_bands(D)
    d5_consistency()


if __name__ == "__main__":
    main()
