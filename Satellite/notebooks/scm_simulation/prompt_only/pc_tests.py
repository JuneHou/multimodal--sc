"""Gates G1 to G7 of the plan's Verification. No test calls a model.

    python prompt_only/pc_tests.py
"""
import json
import math

import numpy as np
import pandas as pd

import pc_lib as pc

FORBIDDEN = ("z1", "z2", "synthetic", "difference", "regression", "weight")


def table(text, header):
    """Data rows of the table whose header line equals header."""
    lines = text.splitlines()
    i = lines.index(header)
    rows = []
    for l in lines[i + 1:]:
        if not l.strip():
            break
        rows.append(l.split(","))
    return rows


def g1_g2(prompts):
    bad = []
    for r in prompts:
        w = pc.wide(r["sigma"], r["case"])
        t = table(r["prompt"], "period,target")
        want = [[str(p), pc.fmt(w.loc[p, "treated"])] for p in pc.PRE]
        if t != want:
            bad.append((r["prompt_id"], "target table"))
        if any(int(row[0]) >= 41 for row in t):
            bad.append((r["prompt_id"], "G2 target row after 40"))
        if r["method"] == "with_donors":
            ds = pc.donors(w)
            if len(ds) != 29:
                bad.append((r["prompt_id"], "donor count"))
            d = table(r["prompt"], "period," + ",".join(ds))
            want_d = [[str(p)] + [pc.fmt(w.loc[p, x]) for x in ds] for p in pc.PRE + pc.POST]
            if d != want_d:
                bad.append((r["prompt_id"], "donor table"))
        low = r["prompt"].lower()
        bad += [(r["prompt_id"], f"forbidden word {f}") for f in FORBIDDEN if f in low]
        if r["prompt"] != pc.prompt(r["method"], r["sigma"], r["case"]):
            bad.append((r["prompt_id"], "prompts.jsonl differs from the builder"))
    print(f"G1/G2 prompt content and held-back periods, {len(prompts)} prompts:", "pass" if not bad else f"FAIL {bad}")
    return not bad


def g3():
    good = ", ".join(f"{t} {t / 10:.3f}" for t in pc.POST)
    want = {str(t): t / 10 for t in pc.POST}
    cases = [
        ("valid line", f"counterfactual: {good}", want),
        ("text before the line", f"I used the mean.\nSo:\ncounterfactual: {good}", want),
        ("markdown bold", f"**counterfactual:** {good}", want),
        ("upper case, backticks", f"`Counterfactual: {good}`", want),
        ("negative and signed values", "counterfactual: " + ", ".join(f"{t} -{t}.5" for t in pc.POST),
         {str(t): -(t + 0.5) for t in pc.POST}),
        ("last line valid of two", f"counterfactual: 41 1.0\ncounterfactual: {good}", want),
        ("last line invalid of two", f"counterfactual: {good}\ncounterfactual: 41 1.0", None),
        ("a period missing", "counterfactual: " + ", ".join(f"{t} 1.0" for t in pc.POST[:-1]), None),
        ("a period twice", "counterfactual: " + ", ".join(f"{t} 1.0" for t in pc.POST) + ", 41 2.0", None),
        ("a period outside 41 to 60", "counterfactual: " + ", ".join(f"{t} 1.0" for t in pc.POST[1:]) + ", 61 1.0", None),
        ("two numbers after a period", "counterfactual: " + ", ".join(f"{t} 1.0 2.0" for t in pc.POST), None),
        ("no answer line", "The impact is about -3.", None),
        ("empty reply", "", None),
    ]
    bad = []
    for name, text, expect in cases:
        got = pc.parse(text)
        if got != expect:
            bad.append(name)
    print(f"G3 parser, {len(cases)} planted replies:", "pass" if not bad else f"FAIL {bad}")
    return not bad


def g4():
    bad = []
    for sigma in pc.SIGMAS:
        g = pd.read_csv(pc.mod.TRUTH_BY_SIGMA[sigma])
        for dgp in pc.DGPS:
            y0 = g[g.dgp == dgp].set_index("period")["Y0_true"]
            s = pc.scored(sigma, dgp, {str(t): float(y0[t]) for t in pc.POST})
            if not (abs(s["cf_rmse"]) < 1e-12 and abs(s["att_error"]) < 1e-12):
                bad.append((sigma, dgp, s))
    print("G4 scoring, true path scores 0 within 1e-12 at every panel:", "pass" if not bad else f"FAIL {bad}")
    return not bad


# README Section 5 cells as printed, data/simulation/README.md lines 431 to 458 (read 2026-11-06), sigma 1.0, 0.5, 0.2
PRINTED = {
    "dgp1": {"ATT bias": ["−0.017", "−0.008", "−0.003"], "ATT RMSE": ["0.317", "0.208", "0.160"],
             "counterfactual RMSE": ["1.368", "0.871", "0.637"],
             "within ±0.25 of the truth": ["60%", "79%", "88%"], "within ±0.50 of the truth": ["88%", "97%", "99%"]},
    "dgp2": {"ATT bias": ["+0.022", "+0.012", "+0.004"], "ATT RMSE": ["0.507", "0.263", "0.131"],
             "counterfactual RMSE": ["1.257", "0.640", "0.290"],
             "within ±0.25 of the truth": ["38%", "67%", "95%"], "within ±0.50 of the truth": ["68%", "94%", "100%"]},
}


def g5():
    bad = []
    for dgp, rows in PRINTED.items():
        for k, sigma in enumerate(pc.SIGMAS):
            col = pc.readme_column(sigma, dgp)
            bad += [(dgp, sigma, r) for r, cells in rows.items() if col[r] != cells[k]]
    print("G5 README column equals the 30 printed cells:", "pass" if not bad else f"FAIL {bad}")
    return not bad


def g6():
    y = pd.Series({t: 2.0 * t + 1.0 for t in range(1, 61)})
    donors = pd.DataFrame({"donor_a": {t: float(t) for t in range(1, 61)},
                           "donor_b": {t: t + 10.0 for t in range(1, 61)}})
    r = pc.rules_from(y, donors)
    want = {"pre_mean": {str(t): 42.0 for t in pc.POST},          # mean of 2t+1 over 1..40 = 2 * 20.5 + 1
            "last_value": {str(t): 81.0 for t in pc.POST},        # 2*40+1
            "linear_trend": {str(t): 2.0 * t + 1.0 for t in pc.POST},
            "did_donor_mean": {str(t): t + 21.5 for t in pc.POST}}   # donor mean t+5, plus 42 minus 25.5
    bad = [k for k in want if any(not math.isclose(r[k][p], want[k][p], abs_tol=1e-9) for p in want[k])]
    print("G6 rules on a planted series:", "pass" if not bad else f"FAIL {bad}")
    return not bad


def g7():
    """The README Section 3 path of pc_readme_sc.R against her replication 1 row, within 0.0001 (Jun "when you check
    whether same, just 4 decimal is enough", 2026-10-31); counts exact."""
    paths = pd.read_csv(pc.README_SC_PATHS, dtype={"sigma": str})
    wts = pd.read_csv(pc.RESULTS / "readme_sc_weights.csv", dtype={"sigma": str})
    bad, worst = [], 0.0
    for sigma in pc.SIGMAS:
        truth = pd.read_csv(pc.mod.TRUTH_BY_SIGMA[sigma])
        for dgp in pc.DGPS:
            her = pd.read_csv(pc.SIM.parent.parent / "data" / "simulation" / "mc_results_r" / f"sigma_{sigma}"
                              / f"mc_replications_{dgp}.csv").query("replication == 1").iloc[0]
            y = pc.wide(sigma, dgp)["treated"]
            ys = paths[(paths.sigma == sigma) & (paths.dgp == dgp)].set_index("period")["y_syn"]
            y0 = truth[truth.dgp == dgp].set_index("period")["Y0_true"]
            w = wts[(wts.sigma == sigma) & (wts.dgp == dgp)]["weight"].to_numpy()
            ours = {"att_hat": np.mean([y[t] - ys[t] for t in pc.POST]),
                    "counterfactual_rmse": np.sqrt(np.mean([(ys[t] - y0[t]) ** 2 for t in pc.POST])),
                    "pre_rmspe": np.sqrt(np.mean([(y[t] - ys[t]) ** 2 for t in pc.PRE])),
                    "max_weight": w.max(), "sum_sq_weights": (w ** 2).sum()}
            for k, v in ours.items():
                worst = max(worst, abs(v - her[k]))
                if abs(v - her[k]) > 1e-4:
                    bad.append((sigma, dgp, k))
            if int((w > 1e-3).sum()) != int(her["n_donors_used"]):
                bad.append((sigma, dgp, "n_donors_used"))
    print(f"G7 README Section 3 path equals her replication 1 on six panels (largest difference {worst:.1e}):",
          "pass" if not bad else f"FAIL {bad}")
    return not bad


def main():
    prompts = [json.loads(l) for l in pc.PROMPTS.read_text().splitlines()]
    ok = [g1_g2(prompts), g3(), g4(), g5(), g6(), g7()]
    print("all gates:", "pass" if all(ok) else "FAIL")
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
