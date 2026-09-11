"""Build 41_llm_band_sc.ipynb from the lb_*.csv tables (S1 before S2). python build_nb41.py"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
C.append(nbf.v4.new_markdown_cell("""# 41 — LLMs as the weight solver on the collaborator's band-feature designs (monthly data)

Plan `quirky-growing-wombat` (2026-09-09). Two designs, both validation only, both predicting **P10 from
inputs up to P09** on the monthly composites: **SHC** (her notebook 26: the target site's own P01–P09) and
**SCM** (her notebook 14 method: the target site P01–P09 and its 5 matched donor sites P01–P10). Scored with
her metrics exactly (`lb_lib.her_nmse`, `lb_lib.her_rmse`, `lb_lib.her_scaler`, `lb_lib.her_solve` — gate 1 of
`lb_tests.py` reproduces her shipped notebook-14 weights and RMSE to 1e-7). All of P01–P10 are pre-hurricane
for every site; nothing is treated in validation. Every number is read from the CSVs."""))
C.append(nbf.v4.new_code_cell("""import json, pandas as pd
import lb_lib as lb
pd.set_option("display.width", 220); f4 = lambda v: f"{v:.4f}"
summ_shc = pd.read_csv("lb_summary_shc.csv"); band_shc = pd.read_csv("lb_summary_shc_band.csv")
summ_scm = pd.read_csv("lb_summary_scm.csv"); feat_scm = pd.read_csv("lb_summary_scm_feature.csv")
S_shc = pd.read_csv("lb_scores_shc.csv"); S_scm = pd.read_csv("lb_scores_scm.csv")
print("methods:", sorted(S_shc.method.unique()))"""))
C.append(nbf.v4.new_markdown_cell("## 1. SHC design — her outcome metric (slide 16 step 10 / `data/shc/10`): squared error / outcome variance over P01–P10 (ddof 1) per band, mean over the 8 or 3 bands\n\n`shc` = our run of her held-out SHC pipeline (`lb_shc.py`, reproduces her 27 stored per-site values); `own_mean` = the target's P01–P09 mean, nothing estimated. The held-out variant (variance over P01–P09, her notebook step 11) is in the same CSV as `joint_nmse`."))
C.append(nbf.v4.new_code_cell("""for s in ("sentinel1", "sentinel2"):
    print(f"\\n=== {s} ===")
    print(summ_shc[summ_shc.sensor == s].sort_values("mean_joint_nmse_p10var").to_string(index=False, float_format=f4))
    print(band_shc[band_shc.sensor == s].pivot(index="band", columns="method", values="mean_nmse").to_string(float_format=f4))"""))
C.append(nbf.v4.new_code_cell("""piv = S_shc[S_shc.ok == True].pivot_table(index=["sensor","site_id"], columns="method", values="joint_nmse_p10var")
print(piv.to_string(float_format=f4))"""))
C.append(nbf.v4.new_markdown_cell("## 2. SCM design — her notebook-14 metric: standardized validation RMSE at P10 (her pooled scaler over targets + donors at P01–P09)\n\n`scm` = her SLSQP simplex solver on the same standardized P01–P09 matrix, with her training RMSE, ratio, flag, largest weight and effective donors."))
C.append(nbf.v4.new_code_cell("""for s in ("sentinel1", "sentinel2"):
    print(f"\\n=== {s} ===")
    print(summ_scm[summ_scm.sensor == s].sort_values("mean_validation_rmse").to_string(index=False, float_format=f4))
    print(feat_scm[feat_scm.sensor == s].pivot(index="feature", columns="method", values="mean_validation_rmse").to_string(float_format=f4))
print(S_scm[S_scm.method == "scm"][["sensor","site_id","training_rmse_std","validation_rmse_std","validation_to_training_rmse_ratio","validation_flag","largest_donor_weight","effective_number_of_donors","weights"]].to_string(index=False, float_format=f4))
print(S_scm[S_scm.ok == True].pivot_table(index=["sensor","site_id"], columns="method", values="validation_rmse_std").to_string(float_format=f4))"""))
C.append(nbf.v4.new_markdown_cell("## 3. Replies that did not parse, and one prompt per design and sensor"))
C.append(nbf.v4.new_code_cell("""for exp in ("shc", "scm"):
    for mf in sorted(lb.LB.glob(f"lb_replies_{exp}_*.jsonl")):
        recs = [json.loads(l) for l in open(mf)]
        bad = [r for r in recs if not r["ok"]]
        print(f"{mf.name}: {len(recs)} replies, {len(bad)} failed" + ("" if not bad else " -> " + ", ".join(r["prompt_id"] for r in bad)))
for exp in ("shc", "scm"):
    seen = set()
    for line in open(lb.LB / f"lb_prompts_{exp}.jsonl"):
        p = json.loads(line)
        if p["sensor"] in seen: continue
        seen.add(p["sensor"]); print(f"\\n===== {p['prompt_id']} =====\\n{p['prompt']}")"""))
C.append(nbf.v4.new_markdown_cell("""## 4. What do the models actually do? Plain rules, prediction diagnosis, prompt ablations (plan 2026-09-10)

**A. Ladder of plain forecasts** of P10 from the target's own P01–P09, per band, scored exactly as the models: `persistence` = P09 carried forward (equals `shc` at the 26 single-block cells), `last3_mean` = mean of P07–P09, `linear9` / `linear4` = OLS line on P01–P09 / P06–P09 extrapolated one step, `holt_damped` = additive damped Holt with α, β, φ chosen by in-sample one-step SSE on P01–P09 (grid), `ensemble` = mean of the Kimi-low and gpt-oss-low answers. `beats_shc` = sites where the method is below our run of her SHC.

**B. Diagnosis**: every quantity is the move from P09 in units of the band's P01–P09 SD (ddof 1); `slope_on_linear9` / `r2` = regression of the model's move on the 9-month linear-trend move (1 = full trend, 0 = persistence, negative = moves against the trend); `direction_ok` = share of cells × bands where the model moved in the direction the truth moved, with the same share for the rules; `nearest_*` = which rule the model's value is closest to.

**C. Prompt ablations** (`[noguide]` no band guide, `[anon]` no guide, sensor or band names, `[nocal]` no calendar dates, `[rule]` the base prompt plus a second line naming the rule used) appear in the table of section 1 once their reply files exist; the rules the models state are listed at the end."""))
C.append(nbf.v4.new_code_cell("""ok = S_shc[S_shc.ok == True]
ref = ok[ok.method == "shc"].set_index(["sensor", "site_id"]).joint_nmse_p10var
for s in ("sentinel1", "sentinel2"):
    t = summ_shc[summ_shc.sensor == s].sort_values("mean_joint_nmse_p10var").copy()
    wins = []
    for m in t.method:
        d = ok[(ok.method == m) & (ok.sensor == s)].set_index("site_id")
        wins.append(f"{int((d.joint_nmse_p10var < ref.loc[s].loc[d.index]).sum())} / {len(d)}")
    t["beats_shc"] = wins
    print(s); print(t[["method", "n", "mean_joint_nmse_p10var", "median_joint_nmse_p10var", "beats_shc"]].to_string(index=False, float_format=lambda v: f"{v:.4f}")); print()"""))
C.append(nbf.v4.new_code_cell("""piv = ok.pivot_table(index=["sensor","site_id"], columns="method", values="joint_nmse_p10var")
cols = [c for c in ["Kimi-K3-thinking-low","gpt-oss-120b","gpt-oss-120b (default effort)","ensemble","last3_mean","holt_damped","linear9","linear4","persistence","shc","own_mean"] if c in piv]
print(piv[cols].round(3).to_string())"""))
C.append(nbf.v4.new_code_cell("""diag = pd.read_csv("lb_diag_summary.csv")
cols = ["model","sensor","n_cells_bands","slope_on_linear9","r2_on_linear9","mean_abs_model_move","mean_abs_truth_move","direction_ok_model","direction_ok_linear9","direction_ok_holt_damped","direction_ok_last3_mean","direction_ok_own_mean"]
print(diag[cols].sort_values(["sensor","model"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(); print(diag[["model","sensor"] + [c for c in diag.columns if c.startswith("nearest_")]].sort_values(["sensor","model"]).to_string(index=False))"""))
C.append(nbf.v4.new_markdown_cell("The three Sentinel-2 sites that set the model means (0004, 0005, 0010): the last three input months, the truth at P10, and every method's P10 value per band. At 0004 the P09 composite is a bright outlier in the visible and SWIR bands (residual cloud/haze) and P10 returns to the earlier level; persistence carries the outlier and the models extrapolate it further, the last-3 mean and the own mean revert."))
C.append(nbf.v4.new_code_cell("""d = pd.read_csv("lb_diag_s2_sites.csv")
cols = ["site_id","band","P07","P08","P09","P10_truth","pred_persistence","pred_last3_mean" if "pred_last3_mean" in d else "pred_own_mean","pred_linear9","pred_holt_damped","pred_Kimi-K3-thinking-low","pred_gpt-oss-120b"]
print(d[[c for c in cols if c in d]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))"""))
C.append(nbf.v4.new_markdown_cell("### Rules the models state for themselves (ablation `[rule]`, second line of the reply; empty until that arm has run)"))
C.append(nbf.v4.new_code_cell("""import glob, lb_lib as lb
for f in sorted(glob.glob("lb_replies_abl_rule_*.jsonl")):
    print("==", f)
    for line in open(f):
        r = json.loads(line)
        if r.get("ok"):
            print(f"  {r['prompt_id']:28s} {lb.rule_text(r.get('reply', ''))}")"""))
nb["cells"] = C; nbf.write(nb, "41_llm_band_sc.ipynb"); print("wrote 41_llm_band_sc.ipynb")
