"""lb_spec_tests — gates for the 2026-09-23 build (plan: quirky-growing-wombat). Called at the end of lb_tests.py.
Part 0 gates: G1 extractor, G2 panel, G3 drift, G14 public surface + parity. Parts 1-2 gates are added as they land.
    python lb_spec_tests.py
"""
import json

import numpy as np
import pandas as pd

import lb_lib as lb
import lb_extract_her
import lb_her_nb10 as her
import lb_her_panel
import lb_her_run


def gate_g1_extractor():
    """The committed lb_her_nb10.py is what a fresh extraction produces, and every manifest slice matches the live notebook."""
    lb_extract_her.check()
    sha, blocks, missing = lb_extract_her.extract()
    assert not missing, missing
    assert sha == her.NOTEBOOK_SHA256, "notebook changed since extraction"
    live = {(b["cell"], b["name"]): b["sha256"] for b in blocks}
    for m in her.MANIFEST:
        assert live[(m["cell"], m["name"])] == m["sha256"], f"manifest slice differs: {m['name']}"
    print(f"G1 ok: {len(her.MANIFEST)} blocks match the live notebook; sha256 {sha[:16]}...")


def gate_g2_panel():
    """Her panel schema from our wide CSV: her columns, exact sin/cos of the calendar month, her sort order."""
    p = lb_her_panel.her_feature_panel()
    assert set(lb_her_panel.REQUIRED) <= set(p.columns), set(lb_her_panel.REQUIRED) - set(p.columns)
    assert len(p) == 1560 and p.period_id.nunique() == 10, (len(p), p.period_id.nunique())
    th = 2.0 * np.pi * p.calendar_month_num.astype(float) / 12.0
    assert np.array_equal(p.x_sin_month.to_numpy(), np.sin(th).to_numpy())
    assert np.array_equal(p.x_cos_month.to_numpy(), np.cos(th).to_numpy())
    assert (p.period_num == p.period_id.str[1:].astype(int)).all()
    key = p[["sensor", "site_id", "band", "period_num"]]
    assert key.equals(key.sort_values(["sensor", "site_id", "band", "period_num"]).reset_index(drop=True))
    p9 = lb_her_panel.her_feature_panel(max_seq=9)
    assert "P10" not in set(p9.period_id) and len(p9) == 1560 * 9 // 10
    print("G2 ok: her panel schema, exact sin/cos, her sort order, 1560 rows; max_seq=9 holds no P10")


def gate_g3_drift():
    """Her extracted code vs the frozen condensed implementation on all 27 cells (prints, then asserts)."""
    lb_her_run.drift(tol=1e-9)


def _bits_equal(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    fa, fb = np.isfinite(a), np.isfinite(b)
    return a.shape == b.shape and np.array_equal(fa, fb) and a[fa].tobytes() == b[fb].tobytes()


def gate_g14_public_surface():
    """lb_shc.run_all keeps its columns and is bit-identical to lb_shc_parity.csv; heldout_shc keeps its return shape."""
    import lb_shc
    res = lb_shc.run_all()
    ref = lb_her_run.read_parity()
    assert list(res.columns[:6]) == lb_her_run.PUBLIC_COLUMNS, list(res.columns)
    preds = [c for c in res.columns if c.startswith("pred_")]
    assert len(preds) == 11, preds
    m = ref.merge(res, on=["sensor", "site_id"], suffixes=("_ref", ""))
    assert len(m) == 27
    for c in ["joint_validation_nmse", "latent_mse"] + preds:
        assert _bits_equal(m[f"{c}_ref"], m[c]), f"{c} differs from lb_shc_parity.csv"
    assert (m.selected_blocks_ref == m.selected_blocks).all() and (m.weights_ref == m.weights).all()
    df = pd.read_csv(lb.FEATURES_CSV)
    df = df[df.role == "target"]
    r = res.iloc[0]
    h = lb_shc.heldout_shc(lb_her_run._series(df, r.site_id, r.sensor), r.sensor)
    assert set(h) == {"rows", "joint", "selected_blocks", "weights", "latent_mse", "predictions"}, set(h)
    assert list(h["rows"].columns) == ["band", "actual_P10", "synthetic_P10", "standardized_sq_error"]
    assert h["joint"] == r.joint_validation_nmse and set(h["predictions"]) == {lb.disp(b) for b in lb.BANDS[r.sensor]}
    print("G14 ok: lb_shc.run_all bit-identical to lb_shc_parity.csv on 27 cells; heldout_shc surface unchanged")


def _truth_and_train(full, site, sensor):
    s = lb_her_run._series(full, site, sensor)
    return ({lb.disp(b): s[10][f"mean_{b}"] for b in lb.BANDS[sensor]},
            {lb.disp(b): [s[q][f"mean_{b}"] for q in range(1, 10)] for b in lb.BANDS[sensor]})


def gate_g4_her_path_bit_identity():
    """lb_fit at HER_SPEC, engine=auto, on the 9-period panel: bit-identical to lb_shc_parity.csv."""
    import lb_fit
    import lb_spec
    panel9 = lb_her_panel.her_feature_panel(max_seq=9)
    ref = lb_her_run.read_parity()
    full = pd.read_csv(lb.FEATURES_CSV)
    full = full[full.role == "target"]
    worst_joint = 0.0
    for _, r in ref.iterrows():
        out = lb_fit.run_cell(panel9, r.sensor, r.site_id, lb_spec.HER_SPEC, engine="auto")
        assert out["engine_used"] == "her"
        assert out["donor_ids"] == [int(x) for x in r.selected_blocks.split()], (r.site_id, out["donor_ids"])
        assert _bits_equal(out["weights"], [float(x) for x in r.weights_full.split()]), (r.site_id, "weights")
        assert _bits_equal([out["fit_mse"]], [r.latent_mse]), (r.site_id, "fit_mse")
        for b in lb.BANDS[r.sensor]:
            assert _bits_equal([out["predictions"][lb.disp(b)]], [r[f"pred_{lb.disp(b)}"]]), (r.site_id, b)
        truth, train = _truth_and_train(full, r.site_id, r.sensor)
        joint = float(np.mean([lb.her_nmse(truth[d], out["predictions"][d], train[d]) for d in truth]))
        worst_joint = max(worst_joint, abs(joint - r.joint_validation_nmse))
    assert worst_joint < 1e-12, worst_joint
    print(f"G4 ok: her path at HER_SPEC on the 9-period panel: blocks, weights, fit MSE and every prediction "
          f"bit-identical to lb_shc_parity.csv (27 cells); joint via lb.her_nmse within {worst_joint:.1e}")


def gate_g5_native_path():
    """The general implementation forced on her settings agrees with the reference to 1e-10, blocks identical."""
    import lb_fit
    import lb_spec
    panel9 = lb_her_panel.her_feature_panel(max_seq=9)
    ref = lb_her_run.read_parity()
    worst = 0.0
    for _, r in ref.iterrows():
        out = lb_fit.run_cell(panel9, r.sensor, r.site_id, lb_spec.HER_SPEC, engine="native")
        assert out["engine_used"] == "native"
        assert out["donor_ids"] == [int(x) for x in r.selected_blocks.split()], (r.site_id, out["donor_ids"])
        wref = np.array([float(x) for x in r.weights_full.split()])
        worst = max(worst, float(np.max(np.abs(out["weights"] - wref))), abs(out["fit_mse"] - r.latent_mse),
                    max(abs(out["predictions"][lb.disp(b)] - r[f"pred_{lb.disp(b)}"]) for b in lb.BANDS[r.sensor]))
    assert worst <= 1e-10, worst
    print(f"G5 ok: general path forced on her settings, 27 cells: blocks identical; max |diff| over weights, "
          f"fit MSE and predictions = {worst:.2e}")


def gate_g6_no_defaults():
    """The fit, sandbox and prompt code never read a spec with a fall-back and never import her settings."""
    import re
    files = [f for f in ("lb_fit.py", "lb_sandbox.py", "lb_spec_prompts.py") if (lb.LB / f).exists()]
    for fn in files:
        src = (lb.LB / fn).read_text()
        for pat in (r"spec\.get\(", r"getattr\(\s*spec", r"HER_SPEC", r"spec\[[^\]]+\]\s+or\s"):
            assert not re.search(pat, src), f"{fn}: forbidden pattern {pat}"
    print(f"G6 ok: no spec fall-back and no HER_SPEC in {files}")


def gate_g7_spec_validation():
    import lb_spec as S
    for f in S.REQUIRED:
        try:
            S.validate_spec({k: v for k, v in S.HER_SPEC.items() if k != f}, "sentinel1")
            raise AssertionError(f"spec without {f} was accepted")
        except S.SpecError as e:
            assert f in str(e), (f, str(e))
    try:
        S.validate_spec({**S.HER_SPEC, "bogus": 1}, "sentinel1")
        raise AssertionError("unknown field accepted")
    except S.SpecError as e:
        assert "bogus" in str(e)
    assert S.validate_spec(S.HER_SPEC, "sentinel2") == S.HER_SPEC
    assert S.validate_spec({**S.HER_SPEC, "band_subset": None}, "sentinel1") == S.HER_SPEC     # null = not given
    try:
        S.validate_spec({**S.HER_SPEC, "band_subset": ["VV"]}, "sentinel1")
        raise AssertionError("band_subset list accepted without subset weighting")
    except S.SpecError as e:
        assert "band_subset" in str(e)
    for spec, field in (({**S.HER_SPEC, "band_weighting": "subset", "band_subset": ["P10"]}, "band_subset"),
                        ({**S.HER_SPEC, "m": "P10"}, "m"), ({**S.HER_SPEC, "m": 8, "n": 2}, "n")):
        try:
            S.validate_spec(spec, "sentinel1")
            raise AssertionError(f"{field} case accepted")
        except S.SpecError as e:
            assert field in str(e), (field, str(e))
    c1 = {k: S.HER_SPEC[k] for k in S.CLASS1}
    assert S.class1_only(c1, "sentinel1") == c1
    try:
        S.class1_only({**c1, "selection": "exhaustive"}, "sentinel1")
        raise AssertionError("arm A accepted a Class-2 key")
    except S.SpecError as e:
        assert "protocol violation" in str(e)
    print(f"G7 ok: each of the {len(S.REQUIRED)} required fields omitted raises naming it; unknown field raises; "
          "P10 and infeasible m+n refused naming the field; arm A refuses Class-2 keys")


def smoke_class2(site="treatment_0015", sensor="sentinel1"):
    """Every Class-2 branch executes on one cell (values are printed, not asserted)."""
    import lb_fit
    import lb_spec as S
    panel9 = lb_her_panel.her_feature_panel(max_seq=9)
    cases = {
        "her": {}, "exhaustive": {"selection": "exhaustive"},
        "nonneg raw": {"weight_constraint": "nonneg", "fit_space": "raw"},
        "unconstr readd": {"weight_constraint": "unconstrained", "readd_seasonal": True},
        "raw+intercept": {"fit_space": "raw", "intercept": "fitted"},
        "inverse_variance": {"band_weighting": "inverse_variance"},
        "subset": {"band_weighting": "subset", "band_subset": ["VV", "VH"]},
        "readd": {"readd_seasonal": True}, "readd+intercept": {"readd_seasonal": True, "intercept": "fitted"},
        "m3 n2 h1": {"m": 3, "n": 2, "kernel_bandwidth": 1.0},
    }
    for name, ch in cases.items():
        spec = S.validate_spec({**S.HER_SPEC, **ch}, sensor)
        out = lb_fit.run_cell(panel9, sensor, site, spec)
        print(f"  {name:18s} engine={out['engine_used']:6s} blocks={out['donor_ids']!s:10s} w={np.round(out['weights'], 4)!s:28s} "
              f"a={out['intercept']:+.4f} mse={out['fit_mse']:.4g} zeta_used={out['zeta_used']} ties={out['near_ties']} "
              f"pred={ {k: round(v, 3) for k, v in out['predictions'].items()} }")


def gate_g9_prompts():
    """Each arm's prompt file: 27 rows, truth/train byte-identical to the base records, nine data lines, no P10
    value, no forbidden word, the field list present, and for arm B every cross-field rule."""
    import re
    import lb_guard
    import lb_spec
    import lb_spec_prompts
    base = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(lb.LB / "lb_prompts_shc.jsonl")}
    data_line = re.compile(r"^P0\d = \d{4}-\d{2}: ")
    for arm, (exp, suffix) in lb_spec_prompts.ARMS.items():
        recs = [json.loads(l) for l in open(lb.LB / f"lb_prompts_{exp}.jsonl")]
        assert len(recs) == 27, (exp, len(recs))
        for r in recs:
            b = base[r["prompt_id"][:-len(suffix) - 1] + "_shc"]
            assert r["truth"] == b["truth"] and r["train"] == b["train"] and r["experiment"] == exp and r["arm"] == arm
            p = r["prompt"]
            assert sum(bool(data_line.match(l)) for l in p.splitlines()) == 9, r["prompt_id"]
            assert "P10" in p and lb.CALENDAR[lb.TARGET] in p
            p10_values = ", ".join(f"{d} {v}" for d, v in zip([lb.disp(b) for b in lb.BANDS[r["sensor"]]],
                                                              lb_guard.p10_value_strings(r["sensor"], r["target_site"])))
            assert p10_values not in p, (r["prompt_id"], "P10 values rendered in the prompt")
            low = p.lower()
            assert not any(w in low for w in lb.FORBIDDEN), r["prompt_id"]
            if arm in ("A", "B"):
                assert all(f"  {f}: " in p for f in lb_spec.CLASS1), r["prompt_id"]
            if arm == "B":
                assert all(f"  {f}: " in p for f in lb_spec.CLASS2) and all(rule in p for rule in lb_spec.CROSS_FIELD_RULES)
            if arm == "A":
                assert not any(f"  {f}: " in p for f in lb_spec.CLASS2), r["prompt_id"]
            if arm == "C":
                assert "def predict(panel, sensor, site_id)" in p
    print("G9 ok: three arm prompt files, 27 rows each, base truth/train kept, nine data lines, no P10 value, no forbidden word")


def gate_g10_canary():
    """With every P10 value replaced by a sentinel in the FULL panel, the fit code never shows it: neither her
    path nor the native path with seasonal re-add reads a P10 value."""
    import lb_fit
    import lb_guard
    import lb_spec
    panel, sentinels = lb_guard.canary_panel()
    assert "P10" in set(panel.period_id)
    for spec in (lb_spec.HER_SPEC, {**lb_spec.HER_SPEC, "readd_seasonal": True, "intercept": "fitted"},
                 {**lb_spec.HER_SPEC, "fit_space": "raw", "weight_constraint": "unconstrained", "selection": "exhaustive"}):
        for sensor, site in (("sentinel1", "treatment_0015"), ("sentinel2", "treatment_0014")):
            out = lb_fit.run_cell(panel, sensor, site, lb_spec.validate_spec(spec, sensor))
            blob = repr({k: v for k, v in out.items() if k != "candidates"}) + out["candidates"].to_string()
            lb_guard.assert_no_sentinel(blob, sentinels, f"run_cell {sensor} {site} {spec['fit_space']}")
    tp = lb_guard.training_panel()
    lb_guard.assert_no_sentinel(tp.to_string(), sentinels, "training_panel")
    print("G10 ok: sentinel P10 values never appear in any fit output (her path, re-add path, raw exhaustive path) or the training panel")


def _her_program():
    """Her extracted functions plus a predict() driver: the sandbox positive control (gate G13)."""
    src = (lb.LB / "lb_her_nb10.py").read_text()
    body = src.split("# ---- manifest ----")[0]
    body = body.split('"""', 2)[2]                          # drop the generated header docstring
    return body + '''

def predict(panel, sensor, site_id):
    fp = panel.copy()
    fp["band"] = fp["band"].replace({"VV-VH": "VV_minus_VH"})
    fp["period_num"] = fp["period_id"].str.extract(r"(\\d+)")[0].astype(int)
    fp["calendar_month_num"] = pd.to_datetime(fp["month"] + "-01").dt.month
    theta = 2.0 * np.pi * fp["calendar_month_num"].astype(float) / 12.0
    fp["x_sin_month"] = np.sin(theta)
    fp["x_cos_month"] = np.cos(theta)
    fp = fp.sort_values(["sensor", "site_id", "band", "period_num"]).reset_index(drop=True)
    latent_df, _beta = estimate_sensor_latents(fp, site_id, sensor, VALIDATION_TRAIN_PERIODS)
    problem, err = build_joint_problem(latent_df, sensor, VALIDATION_TREATED_PRE, VALIDATION_BLOCKS)
    if problem is None:
        raise ValueError(err)
    sel, w, mse, _p, _c = stepwise_donor_selection(problem)
    out = {}
    for band in BANDS[sensor]:
        vals = [get_observed_value(fp, site_id, sensor, band,
                                   VALIDATION_BLOCKS[problem["original_donor_cols"][c]]["post_periods"][0]) for c in sel]
        out[band.replace("VV_minus_VH", "VV-VH")] = float(np.asarray(vals, dtype=float) @ w)
    return out
'''


def gate_g12_g13_sandbox():
    """G12: programs that break the contract are refused with the reason named. G13: her own code as a program
    reproduces lb_shc_parity.csv through the sandbox. G11: the directory inventory is checked on every run."""
    import lb_guard
    import lb_sandbox
    panel9 = lb_guard.training_panel()
    cell = lb_guard.cell_panel(panel9, "sentinel1", "treatment_0015")
    good = "def predict(panel, sensor, site_id):\n    return {b: 1.0 for b in panel['band'].unique()}\n"
    cases = {
        "disallowed import": ("import os\n" + good, "import os is not allowed"),
        "file read": ("import pandas as pd\ndef predict(panel, sensor, site_id):\n    x = pd.read_csv('other')\n    return {}\n", "file access `read_csv`"),
        "path string": (good.replace("1.0", "len('/data/x')"), "path-like"),
        "no predict": ("def guess(panel):\n    return {}\n", "no top-level `def predict"),
        "wrong return": ("def predict(panel, sensor, site_id):\n    return 3\n", "predict must return a dict"),
        "timeout": ("def predict(panel, sensor, site_id):\n    while True:\n        pass\n", "timeout"),
    }
    for name, (src, reason) in cases.items():
        rec = lb_sandbox.run_user_code(src, cell, timeout_s=8)
        assert not rec["ok"] and reason in (rec["error"] or ""), (name, rec["error"], rec["stderr"][-300:])
    rec = lb_sandbox.run_user_code(good, cell)
    assert rec["ok"] and set(rec["predictions"]) == {"VV", "VH", "VV-VH"}, rec
    print(f"G12 ok: {len(cases)} contract breaches refused with the reason named; a trivial valid program runs")
    ref = lb_her_run.read_parity()
    prog = _her_program()
    worst, n = 0.0, 0
    for _, r in ref.iterrows():
        rec = lb_sandbox.run_user_code(prog, lb_guard.cell_panel(panel9, r.sensor, r.site_id))
        assert rec["ok"], (r.site_id, rec["error"], rec["stderr"][-500:])
        for b in lb.BANDS[r.sensor]:
            worst = max(worst, abs(rec["predictions"][lb.disp(b)] - r[f"pred_{lb.disp(b)}"]))
        n += 1
    assert worst <= 1e-8, worst
    print(f"G13 ok: her code as a sandboxed program reproduces every prediction of lb_shc_parity.csv on {n} cells (max |diff| {worst:.1e}); G11 inventory checked on each run")


def main():
    gate_g1_extractor()
    gate_g2_panel()
    gate_g3_drift()
    gate_g14_public_surface()
    gate_g4_her_path_bit_identity()
    gate_g5_native_path()
    gate_g6_no_defaults()
    gate_g7_spec_validation()
    gate_g10_canary()
    gate_g12_g13_sandbox()
    print("lb_spec_tests: Part 0, 1 and 2 gates passed")


if __name__ == "__main__":
    main()
