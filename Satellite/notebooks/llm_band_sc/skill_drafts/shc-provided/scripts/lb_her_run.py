"""lb_her_run — drive the collaborator's extracted SHC code (lb_her_nb10) on our monthly band means.

Part 0 of the 2026-09-23 plan. Her functions are imported and called; her source is never edited.
Class-1 settings (m, n, kernel_bandwidth, zeta, beta_trend_degree) are applied by rebinding the
module-level names her functions read, inside a context manager that restores them on exit.
The treated matching window is VALIDATION_TRAIN_PERIODS[-m:], as in her cell 4: its length is m by
construction (her build_band_problem needs the target and the block columns to have equal length),
so it is not a separate setting.

    python lb_her_run.py             run all 27 cells at her settings; compare with her stored values
    python lb_her_run.py --drift     per-cell differences against the frozen condensed implementation
    python lb_her_run.py --freeze    write lb_shc_parity.csv (full precision) and lb_shc_candidates.csv
"""
import contextlib
import sys

import numpy as np
import pandas as pd

import lb_lib as lb
import lb_her_nb10 as her
import lb_her_panel

PARITY = lb.LB / "lb_shc_parity.csv"
CANDIDATES = lb.LB / "lb_shc_candidates.csv"
PUBLIC_COLUMNS = ["sensor", "site_id", "joint_validation_nmse", "selected_blocks", "weights", "latent_mse"]


@contextlib.contextmanager
def her_globals(**overrides):
    """Temporarily rebind names in lb_her_nb10 (her functions read their globals at call time).
    Refuses a name that is not hers. Restores on exit and asserts the restore. Not reentrant."""
    unknown = [k for k in overrides if not hasattr(her, k)]
    if unknown:
        raise KeyError(f"not a name in lb_her_nb10: {unknown}")
    saved = {k: getattr(her, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(her, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(her, k, v)
        assert all(getattr(her, k) is v for k, v in saved.items()), "her_globals failed to restore"


def validation_blocks(train_periods, m, n):
    return her.make_historical_blocks_from_periods(train_periods, m=m, n=n)


def class1_overrides(spec):
    """Class-1 fields -> the names to rebind. The training span is never a field: P01-P09 is the guard."""
    train = her.VALIDATION_TRAIN_PERIODS
    m, n = int(spec["m"]), int(spec["n"])
    return dict(KERNEL_BANDWIDTH=float(spec["kernel_bandwidth"]), ZETA=float(spec["zeta"]),
                BETA_TREND_DEGREE=int(spec["beta_trend_degree"]),
                VALIDATION_TREATED_PRE=train[-m:], VALIDATION_BLOCKS=validation_blocks(train, m, n))


def heldout_cell(panel, sensor, site, spec=None):
    """One cell through her heldout_validation_for_sensor. spec=None means her extracted settings, untouched.
    Adds: donor_ids (her cell-26 mapping local col -> block donor_id), latent_mse (the final_mse her
    cell 25 discards, recomputed with the same solve_convex_weights call her selector makes)."""
    with her_globals(**(class1_overrides(spec) if spec is not None else {})):
        res = her.heldout_validation_for_sensor(panel, site, sensor)
        blocks = her.VALIDATION_BLOCKS
        sel = res["selected_local_cols"]
        orig = res["problem"]["original_donor_cols"]
        donor_ids = [blocks[orig[c]]["donor_id"] for c in sel]
        w2, latent_mse = her.solve_convex_weights(res["problem"]["z_stacked"], res["problem"]["Z_stacked"], sel)
    assert np.array_equal(w2, res["weights"]), "re-solve did not reproduce her final weights"
    res.update(donor_ids=donor_ids, latent_mse=float(latent_mse), original_donor_cols=list(orig),
               n_original_donors=res["problem"]["n_original_donors"])
    return res


def run_all(features_csv=lb.FEATURES_CSV, spec=None, panel=None):
    """-> (per-cell DataFrame in the historical lb_shc.run_all column layout + weights_full, candidates DataFrame)."""
    panel = lb_her_panel.her_feature_panel(features_csv) if panel is None else panel
    rows, cands = [], []
    for sensor in her.BANDS:
        for site in sorted(panel.loc[panel.sensor == sensor, "site_id"].unique()):
            r = heldout_cell(panel, sensor, site, spec)
            row = {"sensor": sensor, "site_id": site, "joint_validation_nmse": r["joint_validation_nmse"],
                   "selected_blocks": " ".join(str(d) for d in r["donor_ids"]),
                   "weights": " ".join(f"{w:.4f}" for w in r["weights"]), "latent_mse": r["latent_mse"],
                   "weights_full": " ".join(f"{w:.17g}" for w in r["weights"])}
            for _, b in r["results"].iterrows():
                row[f"pred_{lb.disp(b['band'])}"] = b["synthetic_P10"]
            rows.append(row)
            cd = r["candidate_diagnostics"].copy()
            cd.insert(0, "site_id", site); cd.insert(0, "sensor", sensor)
            cands.append(cd)
    return pd.DataFrame(rows), pd.concat(cands, ignore_index=True)


def read_parity(path=PARITY):
    """The frozen reference, read exactly: it is written with %.17g, and pandas' default float parser is
    not round-trip exact (off by an ulp on some values), so bit-level gates must read it this way."""
    return pd.read_csv(path, float_precision="round_trip")


def _series(df, site, sensor):
    d = df[(df.site_id == site) & (df.sensor == sensor)]
    return {int(r.seq): {c: getattr(r, c) for c in df.columns if c.startswith("mean_")} for r in d.itertuples(index=False)}


def drift(tol=1e-9):
    """Her extracted code vs the frozen condensed implementation, per cell, at full precision. Prints, then asserts."""
    import lb_shc_condensed as old
    df = pd.read_csv(lb.FEATURES_CSV)
    panel = lb_her_panel.her_feature_panel()
    print(f"{'sensor':9s} {'site':15s} {'d_joint':>10s} {'d_latent':>10s} {'blocks_old':>10s} {'blocks_new':>10s} {'max|dw|':>10s} {'max|dpred|':>10s}")
    worst, bad = 0.0, []
    for sensor in her.BANDS:
        for site in sorted(panel.loc[panel.sensor == sensor, "site_id"].unique()):
            o = old.heldout_shc(_series(df, site, sensor), sensor)
            n = heldout_cell(panel, sensor, site)
            dj = n["joint_validation_nmse"] - o["joint"]
            dl = n["latent_mse"] - o["latent_mse"]
            same_blocks = list(n["donor_ids"]) == list(o["selected_blocks"])
            dw = np.max(np.abs(np.asarray(n["weights"]) - np.asarray(o["weights"]))) if same_blocks else np.nan
            newp = {lb.disp(b["band"]): b["synthetic_P10"] for _, b in n["results"].iterrows()}
            dp = max(abs(newp[k] - o["predictions"][k]) for k in o["predictions"])
            print(f"{sensor:9s} {site:15s} {dj:10.2e} {dl:10.2e} {str(o['selected_blocks']):>10s} {str(n['donor_ids']):>10s} {dw:10.2e} {dp:10.2e}")
            worst = max(worst, abs(dj), abs(dl), 0.0 if np.isnan(dw) else dw, dp)
            if not same_blocks or abs(dj) > tol or abs(dl) > tol or dp > tol or (not np.isnan(dw) and dw > tol):
                bad.append((sensor, site))
    print(f"\nlargest absolute difference over all 27 cells and all quantities: {worst:.3e}")
    if bad:
        raise SystemExit(f"DRIFT: {len(bad)} cell(s) beyond {tol:g} or with different blocks: {bad}")
    print(f"drift ok: all 27 cells within {tol:g}, selected blocks identical")


def compare_to_her_stored(res):
    hers = pd.read_csv(lb.HER10_SITES)
    m = hers.merge(res, on=["sensor", "site_id"])
    d = (m.joint_validation_nmse - m.her_joint_validation_nmse).abs()
    print(f"{len(m)} cells vs her stored values (printed to 6 decimals): max |diff| = {d.max():.3e}")
    for s, g in m.groupby("sensor"):
        print(f"  {s}: mean {g.joint_validation_nmse.mean():.6f}   her stored mean {g.her_joint_validation_nmse.mean():.6f}")
    return d.max()


def main(argv):
    if "--drift" in argv:
        return drift()
    res, cands = run_all()
    print(res[PUBLIC_COLUMNS].to_string(index=False))
    compare_to_her_stored(res)
    if "--freeze" in argv:
        res.to_csv(PARITY, index=False, float_format="%.17g")
        cands.to_csv(CANDIDATES, index=False, float_format="%.17g")
        print(f"froze {PARITY.name} ({len(res)} rows) and {CANDIDATES.name} ({len(cands)} candidate rows)")


if __name__ == "__main__":
    main(sys.argv[1:])
