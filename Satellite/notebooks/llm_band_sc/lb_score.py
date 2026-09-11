"""lb_score — score every reply with the collaborator's metrics, next to the references.

SHC (her notebook 26): per band, squared error / variance (ddof=1) of the target's own P01-P09 values;
joint = mean over bands. Rows: each model; own_mean (the target's P01-P09 mean, nothing estimated); her
stored SHC values (joint per site; her sensor and sensor x band tables).
SCM (her notebook 14 method on the monthly data): her scaler per sensor x feature (mean / SD ddof=1 over
the sample = targets + donors at P01-P09); per-cell validation RMSE over features at P10 (standardized);
per-feature |error|; training RMSE, ratio and flag, largest weight and effective donors for the SCM
reference (her SLSQP solver on the standardized P01-P09 matrix). Models have no training fit (n/a).
Writes lb_scores_shc.csv, lb_summary_shc.csv, lb_summary_shc_band.csv, lb_scores_scm.csv,
lb_summary_scm.csv, lb_summary_scm_feature.csv.     python lb_score.py
"""
import json

import numpy as np
import pandas as pd

import lb_lib as lb
import lb_shc
import lb_baselines

ARMS = ["abl_noguide", "abl_anon", "abl_nocal", "abl_rule"]


def load_replies(exp):
    out = {}
    for mf in sorted(lb.LB.glob(f"lb_replies_{exp}_*.jsonl")):
        for line in open(mf):
            r = json.loads(line)
            eff = r.get("reasoning_effort", "default")
            method = r["model"] if eff == "low" else f"{r['model']} ({eff} effort)"
            r["method"] = method
            out[(method, r["prompt_id"])] = r
    return out


def score_shc():
    P = [json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")]
    R = load_replies("shc")
    RA = {arm: load_replies(arm) for arm in ARMS}                          # ablation arms (plan 2026-09-10, C)
    her = pd.read_csv(lb.HER26_SITES).set_index(["sensor", "site_id"]).her_joint_validation_nmse      # notebook held-out P10
    models = sorted({m for m, _ in R})
    SHC = lb_shc.run_all().set_index(["sensor", "site_id"])                 # our run of her held-out SHC
    rows = []
    for p in P:
        s, site = p["sensor"], p["target_site"]
        bands = [lb.disp(b) for b in lb.BANDS[s]]
        preds = {"own_mean": {b: float(np.nanmean(p["train"][b])) for b in bands},
                 "shc": {b: float(SHC.loc[(s, site), f"pred_{b}"]) for b in bands}}
        for m in models:
            r = R.get((m, p["prompt_id"]))
            preds[m] = r["values"] if r and r.get("ok") else None
        preds.update(lb_baselines.rule_predictions(p["train"]))              # plain rules (plan 2026-09-10, A)
        k, g = preds.get("Kimi-K3-thinking-low"), preds.get("gpt-oss-120b")
        preds["ensemble"] = {b: 0.5 * (k[b] + g[b]) for b in bands} if k and g else None
        for arm, Ra in RA.items():
            pid = p["prompt_id"][:-4] + "_" + arm[4:]
            for (m, q), r in Ra.items():
                if q == pid:
                    preds[f"{m} [{arm[4:]}]"] = r["values"] if r.get("ok") else None
        for method, pv in preds.items():
            row = {"method": method, "sensor": s, "site_id": site, "ok": pv is not None}
            if pv is not None:
                per = {b: lb.her_nmse(p["truth"][b], pv[b], p["train"][b]) for b in bands}                       # notebook step 11: variance over P01-P09
                per10 = {b: lb.her_nmse(p["truth"][b], pv[b], p["train"][b] + [p["truth"][b]]) for b in bands}   # slide 16 step 10: variance over P01-P10
                row.update({f"nmse_{b}": v for b, v in per.items()})
                row.update({f"nmse10_{b}": v for b, v in per10.items()})
                row["joint_nmse"] = float(np.mean(list(per.values())))
                row["joint_nmse_p10var"] = float(np.mean(list(per10.values())))
            if method == "shc":
                row.update(selected_blocks=SHC.loc[(s, site), "selected_blocks"], weights=SHC.loc[(s, site), "weights"],
                           her_stored_joint_nmse=float(her.get((s, site), np.nan)))
            rows.append(row)
    S = pd.DataFrame(rows); S.to_csv(lb.LB / "lb_scores_shc.csv", index=False)
    ok = S[S.ok == True]
    summ = ok.groupby(["method", "sensor"]).agg(n=("joint_nmse", "size"), mean_joint_nmse=("joint_nmse", "mean"),
                                                 median_joint_nmse=("joint_nmse", "median"),
                                                 mean_joint_nmse_p10var=("joint_nmse_p10var", "mean"),
                                                 median_joint_nmse_p10var=("joint_nmse_p10var", "median")).reset_index()
    summ.to_csv(lb.LB / "lb_summary_shc.csv", index=False)
    band_cols = [c for c in S.columns if c.startswith("nmse_")]      # step-11 per band (P01-P09 variance)
    band = ok.groupby(["method", "sensor"])[band_cols].mean().stack().reset_index()
    band.columns = ["method", "sensor", "band", "mean_nmse"]; band["band"] = band["band"].str[5:]
    band = band.dropna()
    band.to_csv(lb.LB / "lb_summary_shc_band.csv", index=False)
    return S, summ, band


def score_scm():
    F = pd.read_csv(lb.FEATURES_CSV)
    P = [json.loads(l) for l in open(lb.LB / "lb_prompts_scm.jsonl")]
    R = load_replies("scm")
    models = sorted({m for m, _ in R})
    rows = []
    for s in lb.SENSORS:
        cols = lb.FEAT[s]
        d = F[F.sensor == s].copy()
        means, stds = lb.her_scaler(d, cols, lb.PRE)                  # pooled over targets + donors, P01-P09
        d[cols] = (d[cols] - means) / stds
        mat = lambda site, periods: d[d.site_id == site].set_index("seq").reindex(periods)[cols].to_numpy(float)
        for p in [q for q in P if q["sensor"] == s]:
            site, donors = p["target_site"], p["donors"]
            Yt, Yd = mat(site, lb.PRE), [mat(j, lb.PRE) for j in donors]
            w = lb.her_solve(Yt.reshape(-1), np.column_stack([x.reshape(-1) for x in Yd]))
            actual = mat(site, [lb.TARGET])[0]
            X10 = np.stack([mat(j, [lb.TARGET])[0] for j in donors])           # (J, F)
            pred_scm = w @ X10
            train_pred = np.tensordot(w, np.stack(Yd), axes=(0, 0))
            train_rmse = lb.her_rmse(Yt - train_pred)
            preds = {"scm": pred_scm}
            for m in models:
                r = R.get((m, p["prompt_id"]))
                if r and r.get("ok"):
                    native = np.array([r["values"][lb.disp(b)] for b in lb.BANDS[s]], float)
                    preds[m] = (native - means[cols].to_numpy()) / stds[cols].to_numpy()
                else:
                    preds[m] = None
            for method, pv in preds.items():
                row = {"method": method, "sensor": s, "site_id": site, "ok": pv is not None}
                if pv is not None:
                    err = actual - pv
                    row["validation_rmse_std"] = lb.her_rmse(err)
                    row.update({f"err_{lb.disp(b)}": float(e) for b, e in zip(lb.BANDS[s], err)})
                if method == "scm":
                    ratio = row["validation_rmse_std"] / train_rmse if train_rmse > 0 else np.nan
                    row.update(training_rmse_std=train_rmse, validation_to_training_rmse_ratio=ratio,
                               validation_flag="good" if ratio <= 1.5 else ("caution" if ratio <= 2.0 else "poor"),
                               largest_donor_weight=float(np.max(w)), effective_number_of_donors=float(1 / np.sum(w ** 2)),
                               weights=" ".join(f"{v:.4f}" for v in w))
                rows.append(row)
    S = pd.DataFrame(rows); S.to_csv(lb.LB / "lb_scores_scm.csv", index=False)
    ok = S[S.ok == True]
    summ = ok.groupby(["method", "sensor"]).agg(n=("validation_rmse_std", "size"), mean_validation_rmse=("validation_rmse_std", "mean"),
                                                 median_validation_rmse=("validation_rmse_std", "median")).reset_index()
    summ.to_csv(lb.LB / "lb_summary_scm.csv", index=False)
    err_cols = [c for c in S.columns if c.startswith("err_")]
    long = ok.melt(id_vars=["method", "sensor", "site_id"], value_vars=err_cols, var_name="feature", value_name="err").dropna()
    long["feature"] = long["feature"].str[4:]; long["validation_rmse"] = long["err"].abs()   # one period: RMSE = |error| = MAE
    feat = long.groupby(["method", "sensor", "feature"]).validation_rmse.agg(
        mean_validation_rmse="mean", median_validation_rmse="median", min_validation_rmse="min",
        max_validation_rmse="max", sd_validation_rmse="std", number_of_target_sites="size").reset_index()
    feat.to_csv(lb.LB / "lb_summary_scm_feature.csv", index=False)
    return S, summ, feat


def main():
    pd.set_option("display.width", 220)
    S1, summ1, band1 = score_shc()
    print("=== SHC design: P10 predicted from P01-P09; her outcome metric (slide 16 step 10 / notebook 10): squared error / outcome variance over P01-P10 (ddof 1), mean over bands ===")
    print("    rows: models; shc = our run of her held-out SHC (reproduces her 27 stored values); own_mean = P01-P09 mean")
    cols = ["method", "sensor", "n", "mean_joint_nmse_p10var", "median_joint_nmse_p10var", "beats_shc"]
    ok1 = S1[S1.ok == True]
    ref = ok1[ok1.method == "shc"].set_index(["sensor", "site_id"]).joint_nmse_p10var
    wins = {}
    for (m, s), d in ok1.groupby(["method", "sensor"]):
        d = d.set_index("site_id"); wins[(m, s)] = f"{int((d.joint_nmse_p10var < ref.loc[s].loc[d.index]).sum())} / {len(d)}"
    summ1["beats_shc"] = [wins[(m, s)] for m, s in zip(summ1.method, summ1.sensor)]
    print("    rules (plan 2026-09-10 A): persistence = P09, last3_mean, linear9 / linear4 = OLS line on P01-P09 / P06-P09, holt_damped (in-sample grid), ensemble = mean of Kimi-low and gpt-oss-low")
    print("    '[noguide]/[anon]/[nocal]/[rule]' = prompt ablations (plan C); beats_shc = sites below our run of her SHC")
    print(summ1.sort_values(["sensor", "mean_joint_nmse_p10var"])[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n(her held-out variant, variance over P01-P09, kept in lb_summary_shc.csv as mean_joint_nmse / median_joint_nmse)")
    S2, summ2, feat2 = score_scm()
    print("\n=== SCM design (her notebook-14 metric): standardized validation RMSE at P10 ===")
    print(summ2.sort_values(["sensor", "method"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n_rep = {e: len(list(lb.LB.glob(f"lb_replies_{e}_*.jsonl"))) for e in ("shc", "scm", *ARMS)}
    print(f"\nreply files: {n_rep}")


if __name__ == "__main__":
    main()
