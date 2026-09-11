"""lb_baselines — plain forecasting rules for the SHC design and the diagnosis of what the models do.

Rules (P10 from the target's own P01-P09, per band; nothing else is used):
    persistence   P09 carried forward (= the band-space SHC at 26 of 27 cells)
    last3_mean    mean of P07-P09
    own_mean      mean of P01-P09 (the existing no-estimation reference)
    linear9       OLS line on P01-P09, extrapolated to P10
    linear4       OLS line on P06-P09, extrapolated to P10
    holt_damped   additive Holt with damping; alpha, beta, phi by in-sample one-step SSE on P01-P09 (grid)
The rules are scored by lb_score.py as extra method rows of lb_scores_shc.csv (same metric, same cells).

Diagnosis (this file's main; reads lb_scores_shc.csv and the prompts): every quantity is the move from P09
in units of the band's P01-P09 SD (ddof 1).  Writes
    lb_diag_models.csv    per model x cell x band: model / truth / rule moves, nearest rule, direction agreement
    lb_diag_summary.csv   per model x sensor: nearest-rule counts, slope and R^2 of model move on linear9 move,
                          direction-agreement shares (and the same shares for the rules)
    lb_diag_s2_sites.csv  the P01-P10 series and every method's P10 value at the high-error Sentinel-2 sites
    python lb_baselines.py
"""
import json

import numpy as np
import pandas as pd

import lb_lib as lb

RULES = ["persistence", "last3_mean", "own_mean", "linear9", "linear4", "holt_damped"]
DIAG_MODELS = ["Kimi-K3-thinking-low", "gpt-oss-120b", "gpt-oss-120b (default effort)", "ensemble"]
S2_SITES = ["treatment_0004", "treatment_0005", "treatment_0010"]
HOLT_GRID = dict(alpha=np.round(np.arange(0.1, 1.01, 0.1), 2), beta=np.round(np.arange(0.0, 1.01, 0.1), 2),
                 phi=np.array([0.8, 0.85, 0.9, 0.95, 1.0]))


def linear_extrap(y, t, t_next):
    y, t = np.asarray(y, float), np.asarray(t, float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    b, a = np.polyfit(t[ok], y[ok], 1)
    return float(a + b * t_next)


def holt_forecast(y, alpha, beta, phi):
    """Additive damped Holt, level/trend initialised from the first two points; returns (one-step forecast after
    the last point, in-sample one-step SSE over points 3..n)."""
    y = np.asarray(y, float)
    l, b = y[0], y[1] - y[0]
    sse = 0.0
    for t in range(1, len(y)):
        f = l + phi * b
        if t >= 2:
            sse += (y[t] - f) ** 2
        l_new = alpha * y[t] + (1 - alpha) * (l + phi * b)
        b = beta * (l_new - l) + (1 - beta) * phi * b
        l = l_new
    return float(l + phi * b), float(sse)


def holt_damped(y):
    y = np.asarray(y, float)
    if not np.isfinite(y).all() or len(y) < 3:
        return np.nan
    best = (np.inf, np.nan)
    for a in HOLT_GRID["alpha"]:
        for be in HOLT_GRID["beta"]:
            for ph in HOLT_GRID["phi"]:
                f, sse = holt_forecast(y, a, be, ph)
                if sse < best[0] - 1e-12:
                    best = (sse, f)
    return best[1]


def rule_predictions(train):
    """train: {display band: [P01..P09 values]} -> {rule: {band: P10 value}}."""
    t = np.array(lb.PRE, float)
    out = {r: {} for r in RULES}
    for b, y in train.items():
        y = np.asarray(y, float)
        out["persistence"][b] = float(y[-1])
        out["last3_mean"][b] = float(np.nanmean(y[-3:]))
        out["own_mean"][b] = float(np.nanmean(y))
        out["linear9"][b] = linear_extrap(y, t, lb.TARGET)
        out["linear4"][b] = linear_extrap(y[-4:], t[-4:], lb.TARGET)
        out["holt_damped"][b] = holt_damped(y)
    return out


# ----------------------------------------------------------------------------- diagnosis
def diagnose():
    P = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")}
    S = pd.read_csv(lb.LB / "lb_scores_shc.csv")
    # P10 values per method: reconstruct from the score file is impossible (it stores errors), so rebuild rules here
    # and take the model values from the reply files.
    import lb_score
    R = lb_score.load_replies("shc")
    rows = []
    for pid, p in P.items():
        s, site = p["sensor"], p["target_site"]
        bands = list(p["train"].keys())
        rules = rule_predictions(p["train"])
        vals = {m: (R[(m, pid)]["values"] if (m, pid) in R and R[(m, pid)].get("ok") else None) for m in DIAG_MODELS[:3]}
        k, g = vals["Kimi-K3-thinking-low"], vals["gpt-oss-120b"]
        vals["ensemble"] = {b: 0.5 * (k[b] + g[b]) for b in bands} if k and g else None
        for b in bands:
            y = np.asarray(p["train"][b], float)
            sd = float(np.nanstd(y, ddof=1))
            if not np.isfinite(sd) or sd < lb.SD_TOL:
                continue
            p09 = float(y[-1]); truth_mv = (p["truth"][b] - p09) / sd
            rule_mv = {r: (rules[r][b] - p09) / sd for r in RULES}
            for m in DIAG_MODELS:
                if vals[m] is None:
                    continue
                mv = (vals[m][b] - p09) / sd
                dist = {r: abs(mv - rule_mv[r]) for r in RULES}
                nearest = min(dist, key=dist.get)
                rows.append({"model": m, "sensor": s, "site_id": site, "band": b, "sd_train": sd, "p09": p09,
                             "truth": p["truth"][b], "model_value": vals[m][b], "model_move": mv, "truth_move": truth_mv,
                             **{f"{r}_move": rule_mv[r] for r in RULES},
                             "nearest_rule": nearest, "nearest_dist": dist[nearest],
                             "direction_ok": int(np.sign(mv) == np.sign(truth_mv)),
                             **{f"direction_ok_{r}": int(np.sign(rule_mv[r]) == np.sign(truth_mv)) for r in RULES}})
    D = pd.DataFrame(rows); D.to_csv(lb.LB / "lb_diag_models.csv", index=False)
    summ = []
    for (m, s), d in D.groupby(["model", "sensor"]):
        x, yv = d["linear9_move"].values, d["model_move"].values
        ok = np.isfinite(x) & np.isfinite(yv)
        slope, icpt = np.polyfit(x[ok], yv[ok], 1) if ok.sum() > 2 else (np.nan, np.nan)
        r2 = float(np.corrcoef(x[ok], yv[ok])[0, 1] ** 2) if ok.sum() > 2 else np.nan
        cnt = d["nearest_rule"].value_counts()
        row = {"model": m, "sensor": s, "n_cells_bands": len(d), "slope_on_linear9": float(slope), "r2_on_linear9": r2,
               "mean_abs_model_move": float(d["model_move"].abs().mean()), "mean_abs_truth_move": float(d["truth_move"].abs().mean()),
               "direction_ok_model": float(d["direction_ok"].mean())}
        row.update({f"direction_ok_{r}": float(d[f"direction_ok_{r}"].mean()) for r in RULES if r != "persistence"})
        row.update({f"nearest_{r}": int(cnt.get(r, 0)) for r in RULES})
        summ.append(row)
    SM = pd.DataFrame(summ); SM.to_csv(lb.LB / "lb_diag_summary.csv", index=False)
    # the high-error Sentinel-2 sites: series and every method's P10 value
    det = []
    for pid, p in P.items():
        if p["sensor"] != "sentinel2" or p["target_site"] not in S2_SITES:
            continue
        rules = rule_predictions(p["train"])
        for b in p["train"]:
            row = {"site_id": p["target_site"], "band": b, **{f"P{q:02d}": v for q, v in zip(lb.PRE, p["train"][b])},
                   "P10_truth": p["truth"][b]}
            for r in ("persistence", "linear9", "holt_damped", "own_mean"):
                row[f"pred_{r}"] = rules[r][b]
            for m in DIAG_MODELS[:3]:
                rr = R.get((m, pid)); row[f"pred_{m}"] = rr["values"][b] if rr and rr.get("ok") else np.nan
            shc = S[(S.method == "shc") & (S.site_id == p["target_site"]) & (S.sensor == "sentinel2")]
            det.append(row)
    DT = pd.DataFrame(det); DT.to_csv(lb.LB / "lb_diag_s2_sites.csv", index=False)
    return D, SM, DT


def main():
    pd.set_option("display.width", 250)
    D, SM, DT = diagnose()
    cols = ["model", "sensor", "n_cells_bands", "slope_on_linear9", "r2_on_linear9", "mean_abs_model_move", "mean_abs_truth_move",
            "direction_ok_model", "direction_ok_linear9", "direction_ok_holt_damped", "direction_ok_own_mean"]
    print("=== which rule does each model track? moves from P09 in units of the P01-P09 SD; slope/R^2 of model move on the 9-month linear-trend move ===")
    print(SM[cols].sort_values(["sensor", "model"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== nearest rule per model (count of cells x bands) ===")
    print(SM[["model", "sensor"] + [f"nearest_{r}" for r in RULES]].sort_values(["sensor", "model"]).to_string(index=False))
    print(f"\nwrote lb_diag_models.csv ({len(D)} rows), lb_diag_summary.csv, lb_diag_s2_sites.csv ({len(DT)} rows)")


if __name__ == "__main__":
    main()
