"""lb_why_p09 — does the stored reasoning say P09 was chosen because it performs best?

The SHC design only. The advisor's reading of the P09 result is that the model copies the last
month because that is the cheapest way to the lowest error. That is a claim about a choice, so
this reads the models' own account of the choice and reports what is there.

Two kinds of object live in usage["reasoning"] and they are never merged: the research gateway
returns the model's own chain, the OpenAI gateway returns a summary written by a separate
summarizer (usage["reasoning_kind"] == "summary"). Every count below is reported per kind.

Three questions per trace:

  Q1  performance justification for sitting on P09?  A claim that the last month is the best
      estimate, that the series is flat, or that moving away would add error.

  Q2  on the step-by-step arm, why was the last window kept?
        evaluated  the five windows are compared and one is kept on the result
        asserted   no comparison appears at all
        exploited  the trace reasons that the last window reproduces P09 and that this is what
                   scores best, and keeps it for that reason
      The third is the advisor's hypothesis. It is separable because step (3) scores each window
      against the smoothed P06-P09 values while the last window is P05-P08 -- a one-month offset
      that fits best almost by construction. A model that keeps it because the fit is guaranteed,
      rather than because it measured it, is taking the cheapest route to the lowest error.

  Q3  does the stated reason match the answer?  Distance of the prediction from the P09 value in
      units of each band's own P01-P09 SD, attached to every trace.

Phrase lists are spelled out below so they can be read and disputed. Run with --misses to print
traces that matched nothing, so the lists can be corrected before any number is quoted.

    python lb_why_p09.py
    python lb_why_p09.py --misses 8
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

import lb_arms as la
import lb_lib as lb

ARM = {"shc": "direct", "shc_steps": "steps"}

# --------------------------------------------------------------------------- phrase lists
# Q1: a performance reason for staying at or near the last observed month.
#
# The first pass of this list was far too loose and every count it produced was a false positive,
# so the categories below are narrowed and the discarded ones are kept, commented, as the record
# of what was tried:
#   "series is flat"  matched the bare words flat/stable/steady. Every hit was ordinary series
#       description -- "VV fairly stable around -10, Jan dip -11.27, Jul high -9.80. Diff
#       increasing trend after Jan" -- one band steady while others move, followed by a forecast
#       that does move. It is not a reason to sit on P09. Now requires the claim to reach a
#       prediction verb and an unchanged-valued outcome in the same clause.
#   "moving away would add error"  matched lowest error / minimise error / avoid overfitting.
#       Every hit on the step-by-step arm was the model restating step (3) of the prompt, which
#       instructs exactly that. Restating the instruction is not a preference, so the pattern now
#       excludes clauses that mention the windows or the prompt's own wording.
SAYS_P09_IS_BEST = {
    "last month is the best estimate": [
        r"most recent\b[^.]{0,60}\b(best|reliable|informative|representative)",
        r"last (?:month|value|observation)\b[^.]{0,60}\b(best|reliable|safest|anchor)",
        r"\bbest (?:estimate|predictor|guess)\b[^.]{0,40}\b(p09|last|recent)",
        r"anchor (?:on|to) p09", r"start from p09", r"use p09 as",
    ],
    "flat, therefore predict the same value": [
        r"\b(?:flat|stable|steady|unchanged|little changed?|no real change|nearly constant)\b"
        r"[^.]{0,70}\b(?:so|therefore|hence|thus)\b[^.]{0,70}"
        r"\b(?:same|unchanged|repeat|equal to p09|as p09|no change)\b",
        r"\b(?:predict|forecast|output|p10)\b[^.]{0,60}"
        r"\b(?:same as (?:p09|july|last)|unchanged|equal to p09|repeat p09)\b",
    ],
    "notices the procedure reduces to repeating P09, and calls it trivial": [
        r"\b(?:too (?:easy|trivial|simple)|trivial(?:ly)?|merely|just) (?:repeat|repeats|repeating)",
        r"(?:would |will )?just repeat(?:s|ing)? (?:the )?last",
        r"\bpersistence forecast\b", r"\bthat implies persistence\b",
        r"\bmaybe too (?:easy|trivial|simple)\b", r"\bseems too (?:easy|trivial|simple)\b",
    ],
}

# Q2: how the last window came to be kept, on the step-by-step arm.
WINDOW_EVALUATED = [
    r"\bw1\b.*\bw2\b", r"window 1\b.*window 2", r"try(?:ing)? each window", r"each window alone",
    r"\bcompare[d]? (?:the )?(?:five|5|all) windows", r"tested? (?:each|all) window",
    r"\b(?:sse|mse|rmse|error)\b[^.]{0,40}\bw[1-5]\b", r"\bw[1-5]\b[^.]{0,40}\b(?:sse|mse|rmse|fit)",
]
WINDOW_EXPLOITED = [
    # "less obviously" / "not obviously" negate the claim, so the adverb must not be preceded by one
    r"w5\b[^.]{0,80}\b(?<!less )(?<!not )(?:by construction|trivially|automatically|obviously|of course)",
    r"(?:by construction|trivially|automatically)\b[^.]{0,80}\bw5\b",
    r"w5\b[^.]{0,80}\b(?:overlap|shifted|lag|offset|closest to the target window)",
    r"(?:overlap|lag|offset)[^.]{0,80}\bw5\b",
    r"w5\b[^.]{0,60}\b(?:will|must|has to|is bound to) (?:fit|win|be best)",
    r"(?:just|simply) (?:use|take|output|return|copy|repeat) p09",
    r"p09\b[^.]{0,60}\b(?:lowest|smallest|best) (?:error|mse|rmse|fit)",
    r"(?:lowest|smallest|best) (?:error|mse|rmse)\b[^.]{0,60}\bp09\b",
    r"easiest\b[^.]{0,60}\b(?:answer|fit|way)",
]
# a trace that keeps only the last window at all (to separate "kept W5" from "kept something else")
KEPT_W5 = [r"\bw5[- ]only\b", r"only w5\b", r"keep(?:ing)? (?:just |only )?w5\b",
           r"w5\b[^.]{0,40}\balone\b", r"\bbase w5\b", r"final(?:ly)?[^.]{0,30}\bw5\b"]

# ----------------------------------------------------------------------------------------------
# The advisor's assumption has two readings and they are not the same claim, so they are counted
# apart. "Easiest way to get the best performance" can mean:
#
#   FIT      the model picks the window that scores best on the prompt's own selection rule.
#            That rule compares each window with the smoothed P06-P09 values, and the last window
#            P05-P08 is those same months shifted one, so it wins before any arithmetic. This is
#            WINDOW_EXPLOITED above. It is about the selection score, not about P10.
#
#   OUTCOME  the model picks P09 because it judges P09 will be close to the withheld P10. This is
#            the reading that makes the assumption a claim about forecasting, and it is what the
#            patterns below look for. A trace that argues P10 will DIFFER from P09 and then
#            outputs P09 anyway is evidence against the assumption, not for it.
REASONS_ABOUT_P10 = [
    r"p10\b[^.]{0,80}\b(?:likely|probably|expect|should|will|would|similar|higher|lower|increase|decreas)",
    r"\b(?:august|aug)\b[^.]{0,80}\b(?:likely|probably|expect|higher|lower|increase|decreas|similar|peak)",
    r"next month\b[^.]{0,70}\b(?:likely|probably|expect|higher|lower|increase|decreas|similar)",
    r"seasonal[^.]{0,80}\b(?:p10|august|aug|next month)",
]
SAYS_P10_NEAR_P09 = [
    r"p10\b[^.]{0,60}\b(?:=|equals?|same as|similar to|close to|near)\b[^.]{0,20}p09",
    r"p09\b[^.]{0,60}\b(?:is|as)\b[^.]{0,40}\bgood (?:estimate|predictor|proxy)\b[^.]{0,30}p10",
    r"expect p10\b[^.]{0,40}\b(?:unchanged|flat|same)",
]
# A match on the list above is only a forecast judgement if it is not simply restating what the
# procedure emits. "If only W5 kept, P10=P09 raw" is the procedure's output, not a claim that P09
# predicts P10 well -- on the first pass all seven matches were of that kind, and every one was
# followed by the model doubting the result. Matches whose surrounding text carries any of these
# words are therefore reclassified as procedural rather than counted as forecast judgements.
PROCEDURAL_CONTEXT = [r"\bw5\b", r"\bwindow", r"\bkept\b", r"\bkeep\b", r"\bif only\b",
                      r"\bfollower", r"\bstep ?4\b", r"\bselection\b"]
DOUBT = [r"\btoo (?:easy|trivial|simple)\b", r"\btrivial\b", r"\bnot satisfying\b",
         r"\bwhy (?:elaborate|mention|bother)\b", r"\bmaybe not desired\b", r"\bseems? wrong\b",
         r"\blikely expects?\b", r"\bnot (?:the )?intended\b", r"\bpointless\b",
         r"\bignoring trend\b", r"\bis it really that simple\b", r"\bbut why\b"]
SAYS_P10_DIFFERS = [
    r"p10\b[^.]{0,70}\b(?:differ|not (?:be )?equal|won'?t be|should not be|shouldn'?t be|unlikely to (?:be )?equal)",
    r"p10\b[^.]{0,70}\b(?:higher|lower|greater|less|above|below)\b[^.]{0,25}\bthan p09\b",
    r"\b(?:august|aug)\b[^.]{0,70}\b(?:higher|lower|increase|decreas|drop|rise)\b[^.]{0,40}\b(?:than (?:july|p09)|from (?:july|p09))",
    r"(?:senescen|harvest|peak|decline)[^.]{0,70}\b(?:august|aug|p10)\b",
]


PAD = 90     # characters of surrounding text kept with every match, so a hit can be judged


def hits(text, patterns, pad=None):
    """First match per pattern, returned with its surrounding sentence rather than the bare token:
    a match on the word "stable" cannot be judged without what sits either side of it."""
    out = []
    for p in patterns:
        m = re.search(p, text, flags=re.I | re.S)
        if m:
            w = PAD if pad is None else pad
            a, b = max(0, m.start() - w), min(len(text), m.end() + w)
            out.append((p, ("..." if a else "") + text[a:b].replace("\n", " ") + ("..." if b < len(text) else "")))
    return out


# --------------------------------------------------------------------------- reading traces
def traces():
    """Every stored trace on the two SHC arms, with its kind and the answer's distance from P09."""
    rows = []
    for run, f in la.sources():
        for line in open(f):
            r = json.loads(line)
            exp = r.get("experiment")
            if exp not in la.ARMS:
                continue
            u = r.get("usage") or {}
            txt = u.get("reasoning")
            if not txt:
                continue
            p = la.prompts_for(exp)[r["prompt_id"]]
            s = p["sensor"]
            bands = [lb.disp(b) for b in lb.BANDS[s]]
            d = np.nan
            if r.get("ok"):
                per = []
                for b in bands:
                    y = np.asarray(p["train"][b], float)
                    sd = float(np.nanstd(y, ddof=1))
                    if np.isfinite(sd) and sd >= lb.SD_TOL:
                        per.append(abs(r["values"][b] - y[-1]) / sd)
                d = float(np.mean(per)) if per else np.nan
            rows.append({"arm": ARM[exp], "model": r["model"], "run": run,
                         "gateway": u.get("gateway", "arc"),
                         "kind": u.get("reasoning_kind", "own chain"),
                         "sensor": s, "site_id": p["target_site"], "prompt_id": r["prompt_id"],
                         "ok": bool(r.get("ok")), "chars": len(txt), "dist_from_p09": d,
                         "trace": txt})
    return pd.DataFrame(rows)


def classify(df):
    recs = []
    for _, r in df.iterrows():
        t = r["trace"]
        row = {k: r[k] for k in ("arm", "model", "gateway", "kind", "sensor", "site_id",
                                 "prompt_id", "ok", "chars", "dist_from_p09")}
        for label, pats in SAYS_P09_IS_BEST.items():
            h = hits(t, pats)
            row[f"q1: {label}"] = bool(h)
            row[f"q1 evidence: {label}"] = h[0][1] if h else ""
        ev, ex, kept = hits(t, WINDOW_EVALUATED), hits(t, WINDOW_EXPLOITED), hits(t, KEPT_W5)
        row["q2 evaluated"], row["q2 exploited"], row["q2 kept W5"] = bool(ev), bool(ex), bool(kept)
        row["q2 evidence evaluated"] = ev[0][1] if ev else ""
        row["q2 evidence exploited"] = ex[0][1] if ex else ""
        row["q2 evidence kept W5"] = kept[0][1] if kept else ""
        for key, pats in (("reasons about P10", REASONS_ABOUT_P10),
                          ("says P10 differs from P09", SAYS_P10_DIFFERS)):
            h = hits(t, pats)
            row[key] = bool(h)
            row[f"evidence: {key}"] = h[0][1] if h else ""
        # split the P10-equals-P09 matches into a forecast judgement and a restatement of the
        # procedure's own output, by what surrounds the match
        near = hits(t, SAYS_P10_NEAR_P09)
        ctx = near[0][1] if near else ""
        # the procedural test reads a wider window than the quoted evidence: at 90 characters one
        # match missed the words "it will fit best" sitting just outside it and was miscounted as
        # a forecast judgement.
        wide = hits(t, SAYS_P10_NEAR_P09, pad=250)
        wctx = wide[0][1] if wide else ""
        procedural = bool(near) and any(re.search(p, wctx, re.I) for p in PROCEDURAL_CONTEXT)
        row["forecast judgement: P10 near P09"] = bool(near) and not procedural
        row["procedure yields P09 (restated, not a forecast)"] = procedural
        row["doubts the procedural result"] = procedural and bool(hits(ctx, DOUBT))
        row["evidence: P10 near P09"] = ctx
        # evaluated and exploited co-occur -- a trace can enumerate the windows AND notice that the
        # last one is the target shifted. Collapsing to one verdict would hide that, so keep both.
        row["q2 verdict"] = ("evaluated and exploited" if (ev and ex) else "exploited only" if ex
                             else "evaluated only" if ev else "asserted, no comparison")
        row["q1 any"] = any(row[f"q1: {k}"] for k in SAYS_P09_IS_BEST)
        recs.append(row)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- reporting
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", type=int, default=0,
                    help="print this many traces that matched nothing, for correcting the lists")
    a = ap.parse_args()
    pd.set_option("display.width", 250)

    T = traces()
    C = classify(T)
    C.to_csv(lb.LB / "lb_why_p09.csv", index=False)

    print("=== trace coverage: what exists to read (denominator for every count below) ===")
    cov = T.groupby(["arm", "model", "gateway", "kind"]).agg(
        traces=("chars", "size"), median_chars=("chars", "median")).reset_index()
    print(cov.to_string(index=False))

    print("\n=== Q1  does the trace give a performance reason for sitting on P09? ===")
    cols = [f"q1: {k}" for k in SAYS_P09_IS_BEST]
    g = C.groupby(["arm", "model", "kind"])
    q1 = g[cols + ["q1 any"]].sum().astype(int)
    q1.insert(0, "n_traces", g.size())
    print(q1.to_string())

    print("\n=== Q2  step-by-step arm: how did the last window come to be kept? ===")
    s = C[C.arm == "steps"]
    if len(s):
        order = ["evaluated and exploited", "exploited only", "evaluated only", "asserted, no comparison"]
        q2 = s.pivot_table(index=["model", "kind"], columns="q2 verdict", values="ok", aggfunc="size")
        q2 = q2.reindex(columns=order, fill_value=0).fillna(0).astype(int)
        q2.insert(0, "n_traces", s.groupby(["model", "kind"]).size())
        q2["says it kept W5"] = s.groupby(["model", "kind"])["q2 kept W5"].sum().astype(int)
        print(q2.to_string())

    print("\n=== THE ADVISOR'S ASSUMPTION, step-by-step arm only ===")
    print("    Is P09 chosen because the model judges it will be closest to P10 (the outcome),")
    print("    or because the last window wins the prompt's selection rule by construction (the fit)?")
    if len(s):
        SAT = 0.05      # a prediction within this many SD of P09 is treated as having sat on P09
        a = s.copy()
        a["sat on P09"] = a.dist_from_p09 < SAT
        t1 = a.groupby(["model", "kind"]).agg(
            n_traces=("ok", "size"),
            fit_wins_by_construction=("q2 exploited", "sum"),
            reasons_about_P10=("reasons about P10", "sum"),
            FORECAST_P10_near_P09=("forecast judgement: P10 near P09", "sum"),
            restates_procedure_gives_P09=("procedure yields P09 (restated, not a forecast)", "sum"),
            and_doubts_it=("doubts the procedural result", "sum"),
            says_P10_differs=("says P10 differs from P09", "sum"),
            sat_on_P09=("sat on P09", "sum")).astype(int)
        print("\n" + t1.to_string())
        print("\n  The decisive cell: traces that argue P10 will DIFFER from P09, and output P09 regardless.")
        both = a[a["says P10 differs from P09"] & a["sat on P09"]]
        tot = a[a["says P10 differs from P09"]]
        for (m, k), g in tot.groupby(["model", "kind"]):
            n = int((g.dist_from_p09 < SAT).sum())
            print(f"    {m:24s} {k:10s}  {n} of {len(g)} traces that expect a change still returned P09")
        print(f"\n  Overall: {len(both)} of {len(tot)} across all models.")

    print("\n=== Q3  did the answer actually sit on P09?  mean |prediction - P09| in P01-P09 SD ===")
    q3 = C[C.ok].groupby(["arm", "model", "kind"]).dist_from_p09.agg(["size", "mean", "std"])
    print(q3.round(3).to_string())

    if a.misses:
        print(f"\n=== traces that matched nothing at all (first {a.misses}) ===")
        none = C[~C["q1 any"] & ~C["q2 evaluated"] & ~C["q2 exploited"] & ~C["q2 kept W5"]]
        print(f"{len(none)} of {len(C)} traces matched no pattern")
        for _, r in none.head(a.misses).iterrows():
            t = T[T.prompt_id == r.prompt_id]
            t = t[t.model == r.model].trace.iloc[0]
            print(f"\n--- {r.model} / {r.arm} / {r.site_id} / {r['kind']} / {r.chars} chars")
            print("   " + t[:600].replace("\n", "\n   "))

    print(f"\nwrote lb_why_p09.csv ({len(C)} traces)")


if __name__ == "__main__":
    main()
