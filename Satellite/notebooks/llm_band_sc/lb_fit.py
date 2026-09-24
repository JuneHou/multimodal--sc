"""lb_fit — one cell of the held-out SHC design under a spec (lb_spec), on a panel that holds P01-P09 only.

Her code (lb_her_nb10) is called unchanged for every step at which the spec asks for the one behaviour
her code implements; our branches run only where the spec departs from it. Nothing in this module
knows her settings as values: the dispatch condition below names the behaviours her code has, which is
what routing to her functions requires, and is not a default (gate G6 audits this file for fall-backs).

    run_cell(panel, sensor, site, spec, engine="auto")
        engine = "auto"   her functions when the Class-2 fields match her code's behaviour, else our branches
                 "her"    force her functions (error if the spec's Class-2 fields do not match)
                 "native" force our general implementation even on her behaviour (gate G5 only)
    -> dict: engine_used, donor_ids, weights, intercept, fit_mse, predictions {display band: value},
             candidates (DataFrame of every fit tried), zeta_used, n_blocks, near_ties, spec

The fit reads periods P01-P09 only. P10 is scored elsewhere (lb_spec_score) from the prompt record's truth.
The target month's calendar month (for readd_seasonal) is permitted: every prompt already states it.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

import lb_lib as lb
import lb_her_nb10 as her
import lb_her_run

TRAIN = list(her.VALIDATION_TRAIN_PERIODS)          # P01-P09; fixed, never a field
TARGET_MONTH = int(lb.CALENDAR[lb.TARGET][5:7])     # calendar month of the target period
TIE_TOL = 1e-9


def _matches_her_code_behaviour(spec):
    """True when every Class-2 field asks for what her code does, so her functions can run unchanged."""
    return (spec["weight_constraint"] == "simplex" and spec["intercept"] == "none"
            and spec["band_weighting"] == "equal" and spec["fit_space"] == "latent_std"
            and spec["readd_seasonal"] is False and spec["selection"] == "greedy_forward")


# ----------------------------------------------------------------------------- problem
def build_problem(panel, sensor, site, spec):
    """The stacked design under the spec: z (target), Z (block columns), plus what prediction needs."""
    m, n = spec["m"], spec["n"]
    treated_pre = TRAIN[-m:]
    blocks = her.make_historical_blocks_from_periods(TRAIN, m=m, n=n)
    bands = list(her.BANDS[sensor])
    site_df = panel[(panel.site_id == site) & (panel.sensor == sensor)]
    raw = {b: dict(zip(g.period_id, g.mean_value.astype(float))) for b, g in site_df.groupby("band")}
    beta = {}
    if spec["fit_space"] == "latent_std":
        with lb_her_run.her_globals(**lb_her_run.class1_overrides(spec)):
            latent_df, beta_df = her.estimate_sensor_latents(panel, site, sensor, TRAIN)
        fit = {b: dict(zip(g.period_id, g.l_hat_std.astype(float))) for b, g in latent_df.groupby("band")}
        for _, r in beta_df.iterrows():
            beta[r["band"]] = {k: float(r[k]) for k in ("beta_sin", "beta_cos", "latent_mean", "latent_sd")}
    else:
        fit = {b: {p: raw[b][p] for p in TRAIN} for b in bands}
    included = [b for b in bands if lb.disp(b) in spec["band_subset"]] if spec["band_weighting"] == "subset" else bands
    scale = {}
    for b in bands:
        if spec["band_weighting"] == "inverse_variance":
            s = float(np.std([fit[b][p] for p in TRAIN], ddof=1))
            scale[b] = 1.0 / s if s > her.SD_TOL else 1.0
        else:
            scale[b] = 1.0
    z_parts, Z_parts = [], []
    for b in included:
        z_parts.append(np.array([fit[b][p] for p in treated_pre], float) * scale[b])
        Z_parts.append(np.column_stack([[fit[b][p] for p in blk["pre_periods"]] for blk in blocks]) * scale[b])
    return {"z": np.concatenate(z_parts), "Z": np.vstack(Z_parts), "blocks": blocks, "bands": bands,
            "included": included, "scale": scale, "beta": beta, "treated_pre": treated_pre,
            "fwd_raw": {b: np.array([raw[b][blk["post_periods"][0]] for blk in blocks], float) for b in bands},
            "fwd_fit": {b: np.array([fit[b][blk["post_periods"][0]] for blk in blocks], float) for b in bands}}


# ----------------------------------------------------------------------------- weights
def solve_weights(z, Z, cols, spec, native=False):
    """-> (w, intercept, mse). Her solve_convex_weights for simplex without intercept unless native is forced."""
    cols = list(cols)
    X = Z[:, cols]
    J = X.shape[1]
    wc, ic = spec["weight_constraint"], spec["intercept"]
    if wc == "simplex" and ic == "none" and not native:
        w, mse = her.solve_convex_weights(z, Z, cols)
        return np.asarray(w, float), 0.0, float(mse)
    a = 0.0
    if wc == "unconstrained":
        A = np.column_stack([X, np.ones(len(z))]) if ic == "fitted" else X
        coef = np.linalg.lstsq(A, z, rcond=None)[0]
        w = np.asarray(coef[:J], float)
        a = float(coef[J]) if ic == "fitted" else 0.0
    elif wc == "nonneg" and ic == "none":
        w, _ = nnls(X, z)
    else:                                      # simplex forced native, simplex + intercept, nonneg + intercept
        if J == 1 and wc == "simplex" and ic == "none":
            w = np.array([1.0])                # her short-circuit
        else:
            k = J + (1 if ic == "fitted" else 0)
            x0 = np.concatenate([np.full(J, 1.0 / J), [0.0]]) if ic == "fitted" else np.full(J, 1.0 / J)
            bounds = [(0.0, 1.0) if wc == "simplex" else (0.0, None)] * J + ([(None, None)] if ic == "fitted" else [])
            cons = [{"type": "eq", "fun": lambda x: x[:J].sum() - 1.0}] if wc == "simplex" else []
            off = (lambda x: x[J]) if ic == "fitted" else (lambda x: 0.0)
            obj = lambda x: float(np.mean((z - X @ x[:J] - off(x)) ** 2))
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"ftol": her.OPT_TOL, "maxiter": her.OPT_MAXITER})
            if not res.success:
                raise RuntimeError(f"SLSQP failed: {res.message}")
            w = np.asarray(res.x[:J], float)
            a = float(res.x[J]) if ic == "fitted" else 0.0
        if wc == "simplex":                    # her post-processing (cell 19), replicated for the native simplex path
            w[np.abs(w) < 1e-12] = 0.0
            if w.sum() <= 0:
                raise RuntimeError("Estimated weights sum to zero.")
            w = w / w.sum()
    mse = float(np.mean((z - X @ w - a) ** 2))
    return w, a, mse


# ----------------------------------------------------------------------------- selection
def _near_ties(rows, round_col, mse_col):
    """Rounds whose best and second-best candidate MSE are within TIE_TOL: the tiebreak (donor order) decided.
    Her candidate table calls the round `selection_step`; ours calls it `round`."""
    if rows is None or len(rows) == 0 or mse_col not in rows:
        return None
    if round_col not in rows:
        round_col = "selection_step" if "selection_step" in rows else "round"
        if round_col not in rows:
            return None
    n = 0
    for _, g in rows.groupby(round_col):
        v = np.sort(g[mse_col].dropna().to_numpy())
        if len(v) >= 2 and v[1] - v[0] < TIE_TOL:
            n += 1
    return n


def select_blocks(z, Z, spec, native=False):
    """-> (selected cols, w, intercept, mse, candidates DataFrame, zeta_used)."""
    N = Z.shape[1]
    if spec["selection"] == "greedy_forward":
        if spec["weight_constraint"] == "simplex" and spec["intercept"] == "none" and not native:
            problem = {"z_stacked": z, "Z_stacked": Z, "n_donors": N, "n_original_donors": N,
                       "original_donor_cols": list(range(N)), "by_band": {}}
            with lb_her_run.her_globals(ZETA=float(spec["zeta"])):
                sel, w, mse, _path, cands = her.stepwise_donor_selection(problem)
            return list(sel), np.asarray(w, float), 0.0, float(mse), cands, True
        selected, remaining, current, rows = [], list(range(N)), np.inf, []
        while remaining:
            trials = []
            for c in remaining:
                w, a, mse = solve_weights(z, Z, selected + [c], spec, native)
                imp = np.inf if not np.isfinite(current) else current - mse
                trials.append((c, mse, imp))
                rows.append({"round": len(selected) + 1, "candidate": c, "trial_mse": mse, "improvement": imp})
            c, mse, imp = min(trials, key=lambda t: t[1])         # ties: first in donor order, as hers
            if len(selected) == 0 or imp > spec["zeta"]:
                selected.append(c); remaining.remove(c); current = mse
            else:
                break
        w, a, mse = solve_weights(z, Z, selected, spec, native)
        return selected, w, a, mse, pd.DataFrame(rows), True
    rows, best = [], None                                            # exhaustive: every non-empty subset
    for k in range(1, N + 1):
        for sub in itertools.combinations(range(N), k):
            try:
                w, a, mse = solve_weights(z, Z, list(sub), spec, native)
            except RuntimeError as e:
                rows.append({"round": k, "subset": sub, "trial_mse": np.nan, "error": str(e)}); continue
            rows.append({"round": k, "subset": sub, "trial_mse": mse, "error": ""})
            key = (mse, len(sub), sub)                               # lowest MSE, then fewest blocks, then lowest ids
            if best is None or key < best[0]:
                best = (key, list(sub), w, a, mse)
    if best is None:
        raise RuntimeError("every block subset failed to fit")
    _, sel, w, a, mse = best
    return sel, w, a, mse, pd.DataFrame(rows), False


# ----------------------------------------------------------------------------- prediction
def predict(prob, spec, sel, w, a):
    """Her rule unless readd_seasonal: the observed values that followed each kept block, weighted (+ intercept
    in the space it was fitted). With readd_seasonal: predict the standardized latent, undo the standardization,
    add the target month's seasonal term from the band's fitted sine/cosine coefficients."""
    th = 2.0 * np.pi * TARGET_MONTH / 12.0
    preds = {}
    for b in prob["bands"]:
        a_b = a / prob["scale"][b]
        if spec["readd_seasonal"]:
            bt = prob["beta"][b]
            zhat = float(prob["fwd_fit"][b][sel] @ w + a_b)
            preds[lb.disp(b)] = zhat * bt["latent_sd"] + bt["latent_mean"] + bt["beta_sin"] * np.sin(th) + bt["beta_cos"] * np.cos(th)
        else:
            preds[lb.disp(b)] = float(prob["fwd_raw"][b][sel] @ w + a_b)
    return preds


# ----------------------------------------------------------------------------- one cell
def run_cell(panel, sensor, site, spec, engine="auto"):
    assert engine in ("auto", "her", "native"), engine
    her_behaviour = _matches_her_code_behaviour(spec)
    if engine == "her" and not her_behaviour:
        raise ValueError("engine='her' needs every Class-2 field at the behaviour her code implements")
    if her_behaviour and engine != "native":
        with lb_her_run.her_globals(**lb_her_run.class1_overrides(spec)):
            latent_df, _beta = her.estimate_sensor_latents(panel, site, sensor, TRAIN)
            problem, err = her.build_joint_problem(latent_df, sensor, her.VALIDATION_TREATED_PRE, her.VALIDATION_BLOCKS)
            if problem is None:
                raise ValueError(err)
            sel, w, mse, _path, cands = her.stepwise_donor_selection(problem)
            blocks = her.VALIDATION_BLOCKS
        orig = problem["original_donor_cols"]
        preds = {}                                                   # her cell-25 prediction, on P01-P09 values only
        for b in her.BANDS[sensor]:
            dv = np.array([her.get_observed_value(panel, site, sensor, b, blocks[orig[c]]["post_periods"][0]) for c in sel], float)
            preds[lb.disp(b)] = float(dv @ w)
        return {"engine_used": "her", "donor_ids": [blocks[orig[c]]["donor_id"] for c in sel],
                "weights": np.asarray(w, float), "intercept": 0.0, "fit_mse": float(mse), "predictions": preds,
                "candidates": cands, "zeta_used": True, "n_blocks": len(blocks),
                "near_ties": _near_ties(cands, "selection_step", "trial_mse"), "spec": dict(spec)}
    prob = build_problem(panel, sensor, site, spec)
    sel, w, a, mse, cands, zeta_used = select_blocks(prob["z"], prob["Z"], spec, native=(engine == "native"))
    return {"engine_used": "native", "donor_ids": [prob["blocks"][c]["donor_id"] for c in sel],
            "weights": np.asarray(w, float), "intercept": float(a), "fit_mse": float(mse),
            "predictions": predict(prob, spec, sel, w, a), "candidates": cands, "zeta_used": zeta_used,
            "n_blocks": len(prob["blocks"]), "near_ties": _near_ties(cands, "round", "trial_mse"), "spec": dict(spec)}
