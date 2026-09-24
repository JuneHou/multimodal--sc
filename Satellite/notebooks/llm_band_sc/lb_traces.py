"""lb_traces — what the stored reasoning says, and whether the answer matches it.

The reasoning text has so far been quoted anecdotally. This reads all of it.

Two questions, and the second is the one that matters:

  SAID  which method does the trace name? Detected by phrase, reported with counts, so a claim about
        "the models say they average" is backed by a number rather than one quotation.

  DID   which simple rule is the answer actually closest to? Each prediction is compared with six
        rules computed from the same nine rows -- repeat P09, mean of the last two, mean of the last
        three, the site's own nine-month mean, a slope extrapolation from the last three, and a
        least-squares line through all nine -- in units of each band's pre-period SD. The nearest
        rule is the one the answer behaves like, whatever the trace claims.

SAID against DID separates three distinct faults that all score as "wrong":
  - a bad method chosen and followed;
  - a sound method named and then not executed (the answer matches a different rule);
  - no method at all, the answer landing on the trivial fallback.

Availability is reported first, because the research gateway returns the model's own chain while the
GPT-5 family returns only a summary written by a separate summarizer, and the two are not comparable
as evidence of what the model did.

    python lb_traces.py
"""
import json
import re

import numpy as np
import pandas as pd

import lb_arms as la
import lb_lib as lb

ARM = {"shc": "direct", "shc_steps": "steps"}

# phrase -> method named. Matched against the lower-cased trace.
SAYS = {
    "average":     [r"moving average", r"\baverage of", r"\bavg\b", r"\bmean of", r"3-month", r"three-month"],
    "trend/slope": [r"\btrend\b", r"\bslope\b", r"linear", r"extrapolat"],
    "seasonal":    [r"season", r"senescen", r"phenolog", r"summer", r"august"],
    "smoothing":   [r"smooth", r"kernel", r"blend"],
    "weights":     [r"\bweight", r"sum to 1", r"convex", r"simplex"],
    "persistence": [r"carry forward", r"carried forward", r"same as (?:p09|july|last)", r"repeat",
                    r"unchanged", r"\bp09 value"],
}


def rules(p, band):
    """Six predictors of P10 built from the same nine rows the model was given."""
    tr = np.asarray(p["train"][band], float)
    t = np.arange(1, len(tr) + 1)
    fin = np.isfinite(tr)
    out = {"repeat P09": tr[-1],
           "mean last 2": np.nanmean(tr[-2:]),
           "mean last 3": np.nanmean(tr[-3:]),
           "own 9-month mean": np.nanmean(tr),
           "slope from last 3": tr[-1] + (tr[-1] - tr[-3]) / 2}
    if fin.sum() >= 2:
        b, a = np.polyfit(t[fin], tr[fin], 1)
        out["line through all 9"] = a + b * (len(tr) + 1)
    return out


def main():
    keep = {}
    for run, f in la.sources():
        for line in open(f):
            r = json.loads(line)
            if r.get("experiment") not in la.ARMS:
                continue
            k = (r["model"], r["experiment"], r["prompt_id"])
            if r.get("ok") or k not in keep:
                keep[k] = (r, run)

    rows = []
    for (model, arm, pid), (r, run) in keep.items():
        p = la.prompts_for(arm)[pid]
        u = (r.get("usage") or ((r.get("attempts") or [{}])[-1].get("usage") or {}))
        txt = u.get("reasoning") or ""
        kind = "summary" if u.get("reasoning_kind") == "summary" else "chain"
        rec = {"model": model, "arm": ARM[arm], "sensor": p["sensor"], "ok": bool(r.get("ok")),
               "kind": kind, "chars": len(txt)}
        low = txt.lower()
        for name, pats in SAYS.items():
            rec[name] = any(re.search(q, low) for q in pats)
        if r.get("ok"):
            bands = [lb.disp(b) for b in lb.BANDS[p["sensor"]]]
            dist = {}
            for b in bands:
                sd = float(np.std(p["train"][b], ddof=1))
                if sd <= lb.SD_TOL:
                    continue
                for name, val in rules(p, b).items():
                    dist.setdefault(name, []).append(abs(r["values"][b] - val) / sd)
            means = {k2: float(np.mean(v)) for k2, v in dist.items()}
            rec["nearest rule"] = min(means, key=means.get)
            rec["distance"] = means[rec["nearest rule"]]
        rows.append(rec)
    D = pd.DataFrame(rows)

    print("=" * 100)
    print("AVAILABILITY — is there any reasoning text, and what kind")
    print("=" * 100)
    a = D.groupby(["model", "arm"]).agg(
        calls=("chars", "size"), with_text=("chars", lambda s: int((s > 0).sum())),
        median_chars=("chars", "median"), kind=("kind", lambda s: s.mode()[0])).reset_index()
    print(a.to_string(index=False))
    print("\n  'chain'   = the model's own reasoning, returned by the research gateway")
    print("  'summary' = written by a separate summarizer; the GPT-5 family never returns the chain")

    print("\n" + "=" * 100)
    print("SAID — method named in the trace, % of calls that have a trace")
    print("=" * 100)
    T = D[D.chars > 0]
    said = T.groupby(["model", "arm"])[list(SAYS)].mean().mul(100).round(0).astype(int)
    print(said.to_string())

    print("\n" + "=" * 100)
    print("DID — the simple rule each answer is actually closest to")
    print("=" * 100)
    ok = D[D.ok]
    for sensor in ("sentinel1", "sentinel2"):
        s = ok[ok.sensor == sensor]
        if s.empty:
            continue
        print(f"\n{sensor}:")
        print(pd.crosstab([s.model, s.arm], s["nearest rule"]).to_string())

    print("\n" + "=" * 100)
    print("SAID vs DID — where the trace names averaging, what did the answer do?")
    print("=" * 100)
    both = ok[(ok.chars > 0)]
    for arm in ("direct", "steps"):
        s = both[both.arm == arm]
        if s.empty:
            continue
        print(f"\n{arm} prompt:")
        print(pd.crosstab(s["average"], s["nearest rule"], rownames=["trace says average"]).to_string())

    print("\n" + "=" * 100)
    print("HOW CLOSE — distance from the nearest rule (0 = the answer IS that rule)")
    print("=" * 100)
    print(ok.groupby(["sensor", "model", "arm"]).distance.agg(["median", "mean"]).round(3).to_string())
    D.to_csv(lb.LB / "lb_traces.csv", index=False)
    print(f"\nwritten: lb_traces.csv ({len(D)} calls)")


if __name__ == "__main__":
    main()
