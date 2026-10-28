"""lb_fit_daily — one daily case of notebook 14's SHC under a spec (lb_spec_daily), on the visits given only.

Her notebook 14 statements (lb_her_nb14.CELL_SRC) run unchanged for every step at which the spec asks for the one
behaviour her code implements: her cells 1, 2, 4 to 8 and 10 execute in order in one namespace, on a panel holding
this case's visits given and the dates of the visits to predict with no values, Class-1 settings rebound after her
cell 2 as lb_her14_run binds names. Our branches run only where the spec departs from her code. Nothing in this
module knows her settings as values (lb_spec_daily.HER_SPEC is not imported here).

    run_case(pre, to_predict, sensor, spec, engine="auto")
        pre         DataFrame band (her band names), date, y: the visits given
        to_predict  list of (period_id, date) of the visits to predict
        engine      "auto"   her statements when the Class-2 fields match her code's behaviour, else our branches
                    "her"    force her statements (error if the Class-2 fields do not match)
                    "native" force our general implementation even on her behaviour (gate DK5 only)
    -> dict: engine_used, m_star, selection_mode, n_blocks, donor_ids, weights, intercept, R, fit_mse,
             bandwidths {band: days}, predictions {display band: {period_id: value}}, m_rows, path, near_ties, spec

The prediction at a visit to predict is her cell 10 y_star_synthetic: the weighted values of the kept blocks at the
same position of their pseudo-post windows (with readd_seasonal, the weighted latent values, unstandardized, plus her
seasonal term, which notebook 14 sets to zero). Observed values after the visits given never enter.
"""
import contextlib
import io
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

import lb_lib as lb
import lb_her_nb14 as her14

SITE = "site"
SPEC_NAME = "no_lag"
TIE_TOL = 1e-9
SD_ZERO = 1e-12                       # her build_stacked_case treats an SD at or below 1e-12 as zero
HER_CELLS = [1, 2, 3, 4, 5, 6, 7, 8, 10]
STACK_CELLS = [1, 2, 3, 4, 5, 6, 7]


def class1_binds(spec):
    """Class-1 fields -> her names to rebind after her cell 2 (before the cells whose defaults read them)."""
    return dict(TIME_BW_CANDIDATES=[float(h) for h in spec["time_bandwidth_candidates"]], M_MIN=int(spec["m_min"]),
                M_STEP=int(spec["m_step"]), M_MAX_USER=spec["m_max"], MAX_DONORS=int(spec["max_donors"]),
                NU_VARIANCE_SHARE=float(spec["nu"]), ZETA_VARIANCE_SHARE=float(spec["zeta"]))


CLASS2_FIELDS = ("weight_constraint", "intercept", "band_weighting", "fit_space", "readd_seasonal", "selection")


def _matches_her_code_behaviour(spec):
    """True when every Class-2 field asks for what her code does, so her statements can run unchanged."""
    return (spec["weight_constraint"] == "simplex" and spec["intercept"] == "none"
            and spec["band_weighting"] == "equal" and spec["fit_space"] == "latent_std"
            and spec["readd_seasonal"] is False and spec["selection"] == "greedy_forward")


def her_namespace(pre, to_predict, sensor, spec, cells):
    """Her cells in order in one namespace. Her cell 3 reads her panel files; here its feature_panel is this case:
    the visits given, and the visits to predict with y missing."""
    post_dates = [pd.Timestamp(d) for _, d in to_predict]
    given = pre.assign(site_id=SITE, sensor=sensor)[["site_id", "sensor", "band", "date", "y"]]
    ahead = pd.DataFrame([(SITE, sensor, b, d, np.nan) for b in lb.BANDS[sensor] for d in post_dates],
                         columns=["site_id", "sensor", "band", "date", "y"])
    fp = pd.concat([given, ahead], ignore_index=True)
    fp["date"] = pd.to_datetime(fp["date"])
    fp = fp.sort_values(["site_id", "sensor", "band", "date"]).reset_index(drop=True)
    ns = {"__name__": "her_notebook_14"}
    with contextlib.redirect_stdout(io.StringIO()):
        for i in cells:
            if i == 3:
                ns["feature_panel"] = fp
                continue
            exec(compile(her14.CELL_SRC[i], f"<notebook 14 cell {i}>", "exec"), ns)
            if i == 2:
                ns.update(class1_binds(spec))
    return ns


def _failure(ns):
    st = ns.get("status")
    if st is not None and len(st) and (st["status"] == "failed").any():
        return "; ".join(st.loc[st["status"] == "failed", "message"].astype(str).unique())
    return "no case was fitted"


def _bandwidths(ns, sensor):
    return {b: float(ns["series_store"][(SITE, sensor, b, SPEC_NAME)]["time_bandwidth"])
            for b in lb.BANDS[sensor] if (SITE, sensor, b, SPEC_NAME) in ns["series_store"]}


# ----------------------------------------------------------------------------- her statements
def _run_her(pre, to_predict, sensor, spec):
    try:
        ns = her_namespace(pre, to_predict, sensor, spec, HER_CELLS)
    except RuntimeError as e:                     # her cell 10 raises when no case was fitted
        raise ValueError(f"{e}") from None
    key = (SITE, sensor, SPEC_NAME)
    if key not in ns["final_fit_store"]:
        raise ValueError(_failure(ns))
    fit = ns["final_fit_store"][key]
    pid = {pd.Timestamp(d): p for p, d in to_predict}
    eff = ns["daily_effects"]
    preds = {}
    for b in fit["band_order"]:
        rows = eff.loc[eff["band"].eq(b)]
        preds[lb.disp(b)] = {pid[pd.Timestamp(d)]: float(v) for d, v in zip(rows["date"], rows["y_star_synthetic"])}
    blocks, sel = fit["blocks"], fit["selected_cols"]
    return {"engine_used": "her", "m_star": int(fit["m_star"]), "selection_mode": fit["selection_mode"],
            "n_blocks": len(blocks), "donor_ids": [blocks[c]["donor_id"] for c in sel],
            "block_starts": [blocks[c]["start"] for c in sel],
            "weights": np.asarray(fit["weights"], float), "intercept": 0.0, "R": float(fit["final_R"]),
            "fit_mse": float(fit["final_mse"]), "bandwidths": _bandwidths(ns, sensor), "predictions": preds,
            "m_rows": ns["m_selection"], "path": ns["donor_selection_path"], "near_ties": None, "spec": dict(spec)}


# ----------------------------------------------------------------------------- weights
def solve_weights(ns, z, Z, cols, spec, native=False):
    """-> (w, intercept, mse). Her solve_paper_weights for simplex without intercept unless native is forced."""
    cols = list(cols)
    X = Z[:, cols]
    J = X.shape[1]
    wc, ic = spec["weight_constraint"], spec["intercept"]
    if wc == "simplex" and ic == "none" and not native:
        fit = ns["solve_paper_weights"](z, Z, cols)
        return np.asarray(fit["weights"], float), 0.0, float(fit["mse"])
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
            x0 = np.concatenate([np.full(J, 1.0 / J), [0.0]]) if ic == "fitted" else np.full(J, 1.0 / J)
            bounds = [(0.0, 1.0) if wc == "simplex" else (0.0, None)] * J + ([(None, None)] if ic == "fitted" else [])
            cons = [{"type": "eq", "fun": lambda x: x[:J].sum() - 1.0}] if wc == "simplex" else []
            off = (lambda x: x[J]) if ic == "fitted" else (lambda x: 0.0)
            obj = lambda x: float(np.mean((z - X @ x[:J] - off(x)) ** 2))
            res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"ftol": ns["WEIGHT_OPT_TOL"], "maxiter": ns["WEIGHT_OPT_MAXITER"]})
            if not res.success:
                raise RuntimeError(f"SLSQP failed: {res.message}")
            w = np.asarray(res.x[:J], float)
            a = float(res.x[J]) if ic == "fitted" else 0.0
        if wc == "simplex":                    # her post-processing (cell 6), replicated for the native simplex path
            w[np.abs(w) < 1e-12] = 0.0
            if w.sum() <= 0:
                raise RuntimeError("Estimated weights sum to zero.")
            w = w / w.sum()
    mse = float(np.mean((z - X @ w - a) ** 2))
    return w, a, mse


# ----------------------------------------------------------------------------- our branches
class _Problem:
    """The stacked design of one case in the spec's fitting space, built from her stacked case."""

    def __init__(self, ns, sensor, spec):
        case = ns["build_stacked_case"](SITE, sensor, SPEC_NAME)
        self.ns, self.spec, self.case = ns, spec, case
        self.pre_dates, self.post_dates = list(case["pre_dates"]), list(case["post_dates"])
        self.T0, self.n = len(self.pre_dates), len(self.post_dates)
        self.bands = list(case["band_order"])
        self.raw, self.fitv, self.rho, self.scale = {}, {}, {}, {}
        for b in self.bands:
            look = case["band_objs"][b]["pre_fit"].set_index("date")
            y = look.loc[self.pre_dates, "y"].to_numpy(float)
            sc = case["scalers"][b]
            self.raw[b] = y
            if spec["fit_space"] == "latent_std":
                self.fitv[b] = (look.loc[self.pre_dates, "l_hat"].to_numpy(float) - sc["mean"]) / sc["sd"]
                self.rho[b] = sc["rho"]
            else:
                self.fitv[b] = y
                self.rho[b] = 1.0 / sc["var_y"]
            if spec["band_weighting"] == "inverse_variance":
                s = float(np.std(self.fitv[b], ddof=1))
                self.scale[b] = 1.0 / s if s > SD_ZERO else 1.0
            else:
                self.scale[b] = 1.0
        self.included = ([b for b in self.bands if lb.disp(b) in spec["band_subset"]]
                         if spec["band_weighting"] == "subset" else list(self.bands))

    def design(self, m, blocks):
        z, cols, slices, cur = [], [[] for _ in blocks], {}, 0
        for b in self.included:
            z.append(self.fitv[b][self.T0 - m:] * self.scale[b])
            for j, blk in enumerate(blocks):
                cols[j].append(self.fitv[b][list(blk["pre_positions"])] * self.scale[b])
            slices[b] = slice(cur, cur + m)
            cur += m
        return np.concatenate(z), np.column_stack([np.concatenate(c) for c in cols]), slices

    def R(self, z, Z, cols, w, a, slices):
        """Her R: mean over the fitted bands of MSE_b / Var(y_b), MSE_b in the band's own fitting space."""
        r = z - Z[:, list(cols)] @ w - a
        vals = [float(np.mean((r[sl] / self.scale[b]) ** 2)) * self.rho[b] for b, sl in slices.items()]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    def blocks(self, m):
        return self.ns["make_historical_blocks"](list(range(self.T0)), m=m, n=self.n)


def _choose_m(P, spec, native):
    feasible_max = P.T0 - P.n
    if spec["m_max"] is not None:
        feasible_max = min(feasible_max, int(spec["m_max"]))
    candidates = list(range(int(spec["m_min"]), feasible_max + 1, int(spec["m_step"])))
    if not candidates:
        raise ValueError(f"No feasible m: T0={P.T0}, n={P.n}, M_MIN={spec['m_min']}")
    rows = []
    for m in candidates:
        try:
            blocks = P.blocks(m)
            z, Z, slices = P.design(m, blocks)
            w, a, mse = solve_weights(P.ns, z, Z, range(Z.shape[1]), spec, native)
            R = P.R(z, Z, range(Z.shape[1]), w, a, slices)
            rows.append({"m": m, "N_donors": len(blocks), "R_mean_all_in": R, "stacked_mse_all_in": mse,
                         "passes_nu": bool(np.isfinite(R) and R <= spec["nu"]), "error": ""})
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            rows.append({"m": m, "N_donors": np.nan, "R_mean_all_in": np.nan, "stacked_mse_all_in": np.nan,
                         "passes_nu": False, "error": str(e)})
    frame = pd.DataFrame(rows)
    passing = frame.loc[frame["passes_nu"] & np.isfinite(frame["R_mean_all_in"])]
    if not passing.empty:
        return int(passing["m"].max()), "paper_rule_stacked", frame
    usable = frame.loc[np.isfinite(frame["R_mean_all_in"])]
    if usable.empty:
        raise ValueError("No usable m candidate produced a fit.")
    return int(usable["m"].max()), "fallback_longest_m", frame


def _select(P, z, Z, slices, spec, native):
    """-> (selected cols, w, intercept, mse, R, path rows, near_ties)."""
    N = Z.shape[1]
    if spec["selection"] == "greedy_forward":
        single = []
        for j in range(N):
            w, a, mse = solve_weights(P.ns, z, Z, [j], spec, native)
            single.append(mse)
        selected = [int(np.argmin(single))]                       # her first block: lowest single-block MSE
        remaining = [j for j in range(N) if j not in selected]
        w, a, mse = solve_weights(P.ns, z, Z, selected, spec, native)
        current = P.R(z, Z, selected, w, a, slices)
        rows, ties = [], 0
        while remaining:
            trials = []
            for c in remaining:
                w, a, mse = solve_weights(P.ns, z, Z, selected + [c], spec, native)
                trials.append((c, P.R(z, Z, selected + [c], w, a, slices)))
            c, best = min(trials, key=lambda t: t[1])            # ties: first in block order, as hers
            vals = np.sort([t[1] for t in trials if np.isfinite(t[1])])
            ties += int(len(vals) >= 2 and vals[1] - vals[0] < TIE_TOL)
            improvement = current - best
            stop = bool(improvement <= spec["zeta"])
            rows.append({"J": len(selected), "R_J": current, "best_next": c, "R_J_plus_1": best,
                         "improvement": improvement, "stop_now": stop})
            if stop:
                break
            selected.append(c)
            remaining.remove(c)
            current = best
        w, a, mse = solve_weights(P.ns, z, Z, selected, spec, native)
        return selected, w, a, mse, P.R(z, Z, selected, w, a, slices), pd.DataFrame(rows), ties
    rows, best = [], None                                         # exhaustive: every non-empty subset
    for k in range(1, N + 1):
        for sub in itertools.combinations(range(N), k):
            try:
                w, a, mse = solve_weights(P.ns, z, Z, list(sub), spec, native)
            except RuntimeError as e:
                rows.append({"size": k, "subset": sub, "R": np.nan, "error": str(e)})
                continue
            R = P.R(z, Z, sub, w, a, slices)
            rows.append({"size": k, "subset": sub, "R": R, "error": ""})
            if not np.isfinite(R):
                continue
            key = (R, len(sub), sub)                              # lowest R, then fewest blocks, then lowest ids
            if best is None or key < best[0]:
                best = (key, list(sub), w, a, mse, R)
    if best is None:
        raise RuntimeError("every block subset failed to fit")
    _, sel, w, a, mse, R = best
    return sel, w, a, mse, R, pd.DataFrame(rows), None


def _predict(P, spec, blocks, sel, w, a):
    preds = {}
    for b in P.bands:
        a_b = a / P.scale[b]
        out = {}
        for k in range(P.n):
            pos = [blocks[c]["pseudo_post_positions"][k] for c in sel]
            if spec["readd_seasonal"]:
                sc = P.case["scalers"][b]
                zhat = float(np.dot(w, P.fitv[b][pos])) + a_b
                out[k] = zhat * sc["sd"] + sc["mean"] + 0.0          # her seasonal_hat = 0.0 (cells 5 and 10)
            else:
                out[k] = float(np.dot(w, P.raw[b][pos])) + a_b
        preds[b] = out
    return preds


def _run_native(pre, to_predict, sensor, spec, native):
    ns = her_namespace(pre, to_predict, sensor, spec, STACK_CELLS)
    try:
        P = _Problem(ns, sensor, spec)
    except ValueError as e:
        raise ValueError(f"{e}; {_failure(ns)}") from None
    m_star, mode, m_rows = _choose_m(P, spec, native)
    blocks = P.blocks(m_star)
    z, Z, slices = P.design(m_star, blocks)
    sel, w, a, mse, R, path, ties = _select(P, z, Z, slices, spec, native)
    raw = _predict(P, spec, blocks, sel, w, a)
    pid = [p for p, _ in to_predict]
    assert [pd.Timestamp(d) for _, d in to_predict] == [pd.Timestamp(d) for d in P.post_dates]
    preds = {lb.disp(b): {pid[k]: v for k, v in raw[b].items()} for b in P.bands}
    return {"engine_used": "native", "m_star": m_star, "selection_mode": mode, "n_blocks": len(blocks),
            "donor_ids": [blocks[c]["donor_id"] for c in sel], "block_starts": [blocks[c]["start"] for c in sel],
            "weights": np.asarray(w, float), "intercept": float(a), "R": float(R), "fit_mse": float(mse),
            "bandwidths": _bandwidths(ns, sensor), "predictions": preds, "m_rows": m_rows, "path": path,
            "near_ties": ties, "spec": dict(spec)}


# ----------------------------------------------------------------------------- one case
def run_case(pre, to_predict, sensor, spec, engine="auto"):
    """`spec` is a validated spec (lb_spec_daily) holding every Class-1 and Class-2 field; nothing is filled in here."""
    assert engine in ("auto", "her", "native"), engine
    missing = [k for k in CLASS2_FIELDS if k not in spec]
    if missing:
        raise ValueError(f"spec is missing Class-2 field(s) {missing}")
    her_behaviour = _matches_her_code_behaviour(spec)
    if engine == "her" and not her_behaviour:
        raise ValueError("engine='her' needs every Class-2 field at the behaviour her code implements")
    if her_behaviour and engine != "native":
        return _run_her(pre, to_predict, sensor, spec)
    return _run_native(pre, to_predict, sensor, spec, native=(engine == "native"))
