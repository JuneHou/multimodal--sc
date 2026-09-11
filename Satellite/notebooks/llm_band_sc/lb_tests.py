"""lb_tests — gates before any API call (plan 'Verification').
1. Her notebook-14 solver + scaler + validation RMSE, run on her shipped biweekly table with her window
   P01-P08 / P09-P10, reproduce her shipped 02_scm_weights.csv and 06_scm_site_summary.csv.
2. Our monthly extraction equals her notebook-26 feature values printed in the notebook output.
3. 27 parsed per-site notebook-26 values; their sensor means equal her printed sensor table.
4. parse() round trip; 5. no forbidden word in any prompt; then one prompt per experiment x sensor.
    python lb_tests.py
"""
import json

import numpy as np
import pandas as pd

import lb_lib as lb


def gate_her14():
    F = pd.read_csv(lb.HER14 / "01_extracted_features.csv")
    W = pd.read_csv(lb.HER14 / "02_scm_weights.csv")
    S = pd.read_csv(lb.HER14 / "06_scm_site_summary.csv")
    sample = pd.read_csv(lb.ROOT / "Satellite" / "data" / "daily_datasets" / "selected_site_sample.csv")
    dw = drmse = 0.0; n = 0
    for sensor in lb.SENSORS:
        cols = lb.FEAT[sensor]
        d = F[F.sensor == sensor].copy()
        means, stds = lb.her_scaler(d, cols, range(1, 9), period_col="sequential_period")
        d[cols] = (d[cols] - means) / stds
        for t in sorted(d[d.group == "treatment"].site_id.unique()):
            donors = sample[(sample.group == "counterfactual") & (sample.matched_treatment_site_id == t)].sort_values("control_rank").site_id.astype(str).tolist()
            mat = lambda s, periods: d[d.site_id == s].set_index("sequential_period").reindex(periods)[cols].to_numpy(float)
            y = mat(t, range(1, 9)).reshape(-1)
            X = np.column_stack([mat(j, range(1, 9)).reshape(-1) for j in donors])
            w = lb.her_solve(y, X)
            hers = W[(W.sensor == sensor) & (W.treatment_site == t)].set_index("donor_site").loc[donors, "weight"].to_numpy()
            dw = max(dw, float(np.abs(w - hers).max()))
            pred = np.tensordot(w, np.stack([mat(j, [9, 10]) for j in donors]), axes=(0, 0))
            rmse = lb.her_rmse(mat(t, [9, 10]) - pred)
            hers_r = float(S[(S.sensor == sensor) & (S.treatment_site == t)].validation_rmse_standardized.iloc[0])
            drmse = max(drmse, abs(rmse - hers_r)); n += 1
    assert dw <= 1e-6 and drmse <= 1e-8, (dw, drmse)
    print(f"1. her notebook-14 reproduced on {n} cells: max|dw| = {dw:.1e}, max|d validation RMSE| = {drmse:.1e}")


def gate_extraction():
    ours = pd.read_csv(lb.FEATURES_CSV)
    hers = pd.read_csv(lb.HER26_FEATURE_ROWS)
    hers = hers[hers.period_id.str[1:].astype(int) <= lb.TARGET]
    worst, n = 0.0, 0
    for r in hers.itertuples():
        o = ours[(ours.site_id == r.site_id) & (ours.sensor == r.sensor) & (ours.period_id == r.period_id)]
        if not len(o):
            continue
        v = float(o[f"mean_{r.band}"].iloc[0]); worst = max(worst, abs(v - r.her_mean_value)); n += 1
    assert n >= 10 and worst <= 1e-6, (n, worst)
    print(f"2. monthly extraction equals her printed values on {n} rows: max|diff| = {worst:.1e}")


def gate_her26():
    S = pd.read_csv(lb.HER26_SITES); T = pd.read_csv(lb.HER26_TABLES)
    assert len(S) == 27 and S.groupby("sensor").size().to_dict() == {"sentinel1": 12, "sentinel2": 15}, S.groupby("sensor").size()
    m = S.groupby("sensor").her_joint_validation_nmse.mean()
    hers = T[T.kind == "sensor"].set_index("sensor").standardized_sq_error
    d = float((m - hers.loc[m.index]).abs().max())
    assert d <= 5e-6, (m, hers)
    print(f"3. her 27 per-site values parsed; sensor means {m.round(6).to_dict()} match her table (max|diff| = {d:.1e}); "
          f"{int((T.kind == 'sensor_band').sum())} sensor x band rows")


def gate_shc_reproduction():
    import lb_shc
    df = lb_shc.run_all(); her = pd.read_csv(lb.HER26_SITES)
    m = df.merge(her, on=["sensor", "site_id"]); d = float((m.joint_validation_nmse - m.her_joint_validation_nmse).abs().max())
    assert len(m) == 27 and d <= 1e-5, (len(m), d)
    print(f"3b. our run of her held-out SHC reproduces her 27 stored per-site values: max|diff| = {d:.1e}")


def gate_parse_and_words():
    assert lb.parse("sentinel1", "VV -9.9, VH -15.8, VV-VH 5.9") == {"VV": -9.9, "VH": -15.8, "VV-VH": 5.9}
    assert lb.parse("sentinel2", "B2 0.041, B3 0.066, B4 0.052, B8 0.390, B11 0.230, B12 0.115, NDVI 0.765, NDWI -0.710")["NDWI"] == -0.71
    assert lb.parse("sentinel2", "B2 0.041, B3 0.066") is None
    n = 0
    for exp in ("shc", "scm"):
        for line in open(lb.LB / f"lb_prompts_{exp}.jsonl"):
            p = json.loads(line); n += 1
            low = p["prompt"].lower()
            assert not any(w in low for w in lb.FORBIDDEN), p["prompt_id"]
            assert f"P{lb.TARGET:02d}" in p["prompt"] and all(np.isfinite(v) for v in p["truth"].values()), p["prompt_id"]
            if exp == "scm":
                assert len(p["donors"]) == lb.N_DONORS
    print(f"4-5. parser ok; {n} prompts, no forbidden word, truth finite, 5 donors per SCM prompt")


def gate_prompt_builder():
    """6. the default prompt builder reproduces the 27 stored SHC prompts byte for byte (ablations share the code)."""
    import pandas as pd
    from lb_prompts import rows_of
    df = pd.read_csv(lb.FEATURES_CSV)
    n = 0
    for line in open(lb.LB / "lb_prompts_shc.jsonl"):
        p = json.loads(line); n += 1
        assert lb.prompt_shc(p["sensor"], rows_of(df, p["target_site"], p["sensor"])) == p["prompt"], p["prompt_id"]
    print(f"6. default prompt builder reproduces the {n} stored SHC prompts exactly")


def gate_rules():
    """7. ladder rules on a planted line; Holt with beta=0, phi=1 is simple exponential smoothing; persistence
    equals our run of her SHC wherever it keeps block 5 alone at weight 1; own_mean equals the existing row."""
    import lb_baselines as lbb
    y = [2.0 + 0.5 * t for t in range(1, 10)]                         # exact line: next value 7.0
    r = lbb.rule_predictions({"b": y})
    assert abs(r["linear9"]["b"] - 7.0) < 1e-9 and abs(r["linear4"]["b"] - 7.0) < 1e-9 and r["persistence"]["b"] == 6.5
    assert abs(r["last3_mean"]["b"] - 6.0) < 1e-9 and abs(r["own_mean"]["b"] - 4.5) < 1e-9
    f, _ = lbb.holt_forecast(y, 0.3, 0.0, 1.0)                        # SES with a fixed initial trend of 0.5 -> stays on the line
    assert abs(f - 7.0) < 1e-9
    z = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0]
    f, _ = lbb.holt_forecast(z, 1.0, 0.0, 1.0)                        # alpha 1, beta 0: level = last value, trend fixed at z1-z0
    assert abs(f - (5.0 + 2.0)) < 1e-9
    import lb_shc
    S = lb_shc.run_all().set_index(["sensor", "site_id"])
    worst, n = 0.0, 0
    for line in open(lb.LB / "lb_prompts_shc.jsonl"):
        p = json.loads(line)
        row = S.loc[(p["sensor"], p["target_site"])]
        if str(row["selected_blocks"]).strip() != "5":
            continue
        n += 1
        pr = lbb.rule_predictions(p["train"])
        for b in p["train"]:
            worst = max(worst, abs(pr["persistence"][b] - float(row[f"pred_{b}"])))
    assert worst < 1e-9, worst
    print(f"7. rules exact on a planted line; Holt reduces correctly; persistence = our run of her SHC at the {n} single-block cells (max diff {worst:.1e})")


def gate_ablation_prompts():
    """8-9. anon labels round-trip through the parser; ablation prompt files: 27 rows each, same truth as the
    base file, no forbidden word, the intended text present or absent."""
    for s in lb.SENSORS:
        labs = lb.anon_labels(s)
        txt = ", ".join(f"{l} {0.1 * (k + 1):.3f}" for k, l in enumerate(labs))
        got = lb.parse(s, txt, labs)
        assert list(got.keys()) == [lb.disp(b) for b in lb.BANDS[s]] and abs(got[lb.disp(lb.BANDS[s][-1])] - 0.1 * len(labs)) < 1e-9
    base = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")}
    from lb_prompts_ablation import ARMS
    for arm in ARMS:
        n = 0
        for line in open(lb.LB / f"lb_prompts_{arm}.jsonl"):
            p = json.loads(line); n += 1
            b = base[p["prompt_id"][: -len(arm[4:]) - 1] + "_shc"]
            assert p["truth"] == b["truth"] and p["train"] == b["train"]
            low = p["prompt"].lower()
            assert not any(w in low for w in lb.FORBIDDEN)
            assert ("Band guide" in p["prompt"]) == (arm in ("abl_nocal", "abl_rule"))
            assert (lb.CALENDAR[lb.TARGET] in p["prompt"]) == (arm != "abl_nocal")
            assert ("x1 " in p["prompt"]) == (arm == "abl_anon") and ("Sentinel" in p["prompt"]) != (arm in ("abl_noguide", "abl_anon"))
            assert ("rule you used" in p["prompt"]) == (arm == "abl_rule")
        assert n == 27
    print("8-9. anon labels round-trip; 4 ablation prompt files x 27 cells, same truth, no forbidden word, intended edits present")


def show_prompts():
    for exp in ("shc", "scm"):
        seen = set()
        for line in open(lb.LB / f"lb_prompts_{exp}.jsonl"):
            p = json.loads(line)
            if p["sensor"] in seen:
                continue
            seen.add(p["sensor"])
            print(f"\n===== {p['prompt_id']} =====\n{p['prompt']}")


if __name__ == "__main__":
    gate_her14(); gate_extraction(); gate_her26(); gate_shc_reproduction(); gate_parse_and_words()
    gate_prompt_builder(); gate_rules(); gate_ablation_prompts(); show_prompts()
    print("\nlb_tests: all gates passed")
