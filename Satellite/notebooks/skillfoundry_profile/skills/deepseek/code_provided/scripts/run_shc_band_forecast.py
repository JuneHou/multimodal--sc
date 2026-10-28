#!/usr/bin/env python3
"""Forecast a parcel's next-time-point satellite band values with the
Synthetic Historical Control (SHC) method of Chen, Yang & Yang.

The SHC method reuses one unit's own distant history as a donor pool. A
"treated block" is the most recent ``m`` observations; each "historical
block" is an older window that mirrors the treated block's pre/post split.
The treated pre-segment is matched with a convex (simplex) combination of
historical pre-segments; those same weights are applied to the historical
"pseudo-post" segments to build the counterfactual, i.e. the forecast.

This script is self-contained and uses only the Python standard library so
it runs offline on any machine with the agent's bash shell. It:

  1. reads a ``data.csv`` describing one parcel's band history,
  2. optionally detrends and standardizes each band,
  3. estimates the latent (smooth) component with local-linear kernel
     regression whose bandwidth is chosen by leave-one-out cross validation,
  4. matches the treated block to historical blocks via a simplex-constrained
     least-squares problem (projected-gradient solver),
  5. writes ``answer.json`` with ``{"prediction": {...}, "rationale": ...}``
     and prints a machine-readable JSON summary.

See SKILL.md for the full runbook, environment setup, and troubleshooting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading and format detection
# ---------------------------------------------------------------------------

BAND_KEYS = ("band", "band_name", "bandname", "band_id", "channel", "variable", "var")
TIME_KEYS = ("time", "date", "timestamp", "t", "period", "datetime", "index")
VALUE_KEYS = ("value", "reflectance", "band_value", "val", "y", "measurement")


def _column_lookup(fieldnames: list[str]) -> dict[str, str | None]:
    lowered = {name.lower().strip(): name for name in fieldnames}
    found: dict[str, str | None] = {"band": None, "time": None, "value": None}
    for key, target in (("band", BAND_KEYS), ("time", TIME_KEYS), ("value", VALUE_KEYS)):
        for candidate in target:
            if candidate in lowered:
                found[key] = lowered[candidate]
                break
    return found


def _to_float(raw: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_long(rows: list[dict[str, str]], cols: dict[str, str | None]) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    time_as_str: dict[str, dict[float, float]] = {}
    seq_counter = 0
    for row in rows:
        band = str(row[cols["band"]]).strip()
        value = _to_float(row[cols["value"]])
        if not band or value is None:
            continue
        raw_time = str(row[cols["time"]]).strip()
        parsed = _to_float(raw_time)
        if parsed is None:
            # Non-numeric timestamps: assign a stable order of first appearance.
            if raw_time not in time_as_str.setdefault(band, {}):
                seq_counter += 1
                time_as_str[band][raw_time] = float(seq_counter)
            t = time_as_str[band][raw_time]
        else:
            t = parsed
        series.setdefault(band, []).append((t, value))
    return series


def load_wide(rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, list[tuple[float, float]]]:
    """Wide format: either rows are time points and columns are bands, or the
    first column is the band and remaining columns are time points."""
    first = fieldnames[0]
    first_is_band = first.lower().strip() in BAND_KEYS
    series: dict[str, list[tuple[float, float]]] = {}

    if not first_is_band:
        # rows = time, columns = bands
        for row in rows:
            t = _to_float(row[first])
            if t is None:
                continue
            for band in fieldnames[1:]:
                value = _to_float(row[band])
                if value is None:
                    continue
                series.setdefault(band, []).append((t, value))
    else:
        # rows = bands, columns = time points
        for row in rows:
            band = str(row[first]).strip()
            if not band:
                continue
            for idx, time_col in enumerate(fieldnames[1:]):
                value = _to_float(row[time_col])
                if value is None:
                    continue
                t = _to_float(time_col)
                t = t if t is not None else float(idx)
                series.setdefault(band, []).append((t, value))
    return series


def load_series(path: Path) -> dict[str, list[tuple[float, float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        fieldnames = [name for name in reader.fieldnames if name is not None]
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} has no data rows")

    cols = _column_lookup(fieldnames)
    if cols["band"] and cols["value"]:
        series = load_long(rows, cols)
        # If the long parser found a single band with duplicated time stamps it
        # is more likely a wide table with an unusual header.
        if len(series) >= 1 and any(len(v) > 1 for v in series.values()):
            return _sort_series(series)
    return _sort_series(load_wide(rows, fieldnames))


def _sort_series(series: dict[str, list[tuple[float, float]]]) -> dict[str, list[tuple[float, float]]]:
    cleaned: dict[str, list[tuple[float, float]]] = {}
    for band, points in series.items():
        if not points:
            continue
        # Average any duplicate time stamps, then sort ascending.
        buckets: dict[float, list[float]] = {}
        for t, value in points:
            buckets.setdefault(t, []).append(value)
        ordered = sorted((t, sum(vals) / len(vals)) for t, vals in buckets.items())
        cleaned[band] = ordered
    return cleaned


# ---------------------------------------------------------------------------
# Numerical helpers (pure Python)
# ---------------------------------------------------------------------------


def gaussian_kernel(u: float) -> float:
    return math.exp(-0.5 * u * u)


def local_linear_predict(
    xs: list[float],
    ys: list[float],
    h_abs: float,
    targets: list[float],
    exclude_index: int | None = None,
) -> list[float]:
    """Local-linear (degree-1) kernel smoother evaluated at ``targets``.

    ``h_abs`` is the bandwidth in the same units as ``xs``. Points far from the
    target simply receive negligible Gaussian weight. If the local design is
    singular this falls back to a weighted mean.
    """
    preds: list[float] = []
    for xt in targets:
        s0 = s1 = s2 = t0 = t1 = 0.0
        for idx, (x, y) in enumerate(zip(xs, ys)):
            if exclude_index is not None and idx == exclude_index:
                continue
            z = (x - xt) / h_abs
            if abs(z) > 8.0:
                continue
            w = gaussian_kernel(z)
            if w <= 0.0:
                continue
            s0 += w
            s1 += w * z
            s2 += w * z * z
            t0 += w * y
            t1 += w * z * y
        if s0 <= 0.0:
            preds.append(float("nan"))
            continue
        det = s0 * s2 - s1 * s1
        if abs(det) < 1e-12:
            preds.append(t0 / s0)
            continue
        a = (t0 * s2 - t1 * s1) / det
        preds.append(a)
    return preds


def select_bandwidth(xs: list[float], ys: list[float], grid: list[float]) -> tuple[float, float]:
    """Pick the bandwidth minimizing leave-one-out CV error. Returns (h_abs, cv)."""
    n = len(xs)
    if n < 3:
        return (max(1.0, float(n)), float("inf"))
    best_h = grid[0]
    best_cv = float("inf")
    for h_frac in grid:
        h_abs = max(h_frac * n, 1e-6)
        errors: list[float] = []
        for i in range(n):
            pred = local_linear_predict(xs, ys, h_abs, [xs[i]], exclude_index=i)[0]
            if math.isnan(pred):
                continue
            errors.append((pred - ys[i]) ** 2)
        if not errors:
            continue
        cv = sum(errors) / len(errors)
        if cv < best_cv:
            best_cv = cv
            best_h = h_abs
    return best_h, best_cv


def _gram_and_c(columns: list[list[float]], target: list[float]) -> tuple[list[list[float]], list[float]]:
    n = len(columns)
    m = len(target)
    gram = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ci = columns[i]
        for j in range(i, n):
            cj = columns[j]
            dot = 0.0
            for k in range(m):
                dot += ci[k] * cj[k]
            gram[i][j] = dot
            gram[j][i] = dot
    c = [sum(columns[i][k] * target[k] for k in range(m)) for i in range(n)]
    return gram, c


def solve_simplex_least_squares(
    columns: list[list[float]],
    target: list[float],
    *,
    ridge: float = 1e-6,
    iterations: int = 200,
) -> list[float]:
    """Solve min_w ||target - sum_i w_i * columns_i||^2 + ridge*||w||^2
    subject to w >= 0 and sum(w) = 1 via the Frank-Wolfe (conditional gradient)
    algorithm with exact line search.

    Starting from the best single column this converges quickly to a sparse
    simplex solution, mirroring the sparsity assumption (Assumption 3b) of the
    SHC paper. ``columns`` is a list of N vectors, each of length m.
    """
    n = len(columns)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    m = len(target)
    if m == 0:
        return [1.0 / n] * n

    gram, c = _gram_and_c(columns, target)

    def grad_of(w: list[float]) -> list[float]:
        out = [0.0] * n
        for i in range(n):
            gi = 0.0
            row = gram[i]
            for j in range(n):
                if w[j]:
                    gi += row[j] * w[j]
            out[i] = 2.0 * (gi + ridge * w[i]) - 2.0 * c[i]
        return out

    # Warm start: vertex minimizing <grad, e_i> = best single column.
    start = min(range(n), key=lambda i: gram[i][i] + ridge - 2.0 * c[i])
    w = [0.0] * n
    w[start] = 1.0

    for _ in range(iterations):
        grad = grad_of(w)
        s_idx = min(range(n), key=lambda i: grad[i])
        if w[s_idx] >= 1.0 - 1e-12:
            break
        # Frank-Wolfe gap and exact line search for the quadratic objective.
        d = [(-w[i]) for i in range(n)]
        d[s_idx] += 1.0
        gap = sum(grad[i] * (-d[i]) for i in range(n))  # = <grad, w - s>
        if gap <= 1e-12:
            break
        # d^T (G + ridge I) d
        gd = [0.0] * n
        for i in range(n):
            acc = 0.0
            row = gram[i]
            for j in range(n):
                if d[j]:
                    acc += row[j] * d[j]
            gd[i] = acc + ridge * d[i]
        denom = sum(d[i] * gd[i] for i in range(n))
        if denom <= 1e-18:
            break
        step = gap / denom
        if step > 1.0:
            step = 1.0
        if step <= 0.0:
            break
        for i in range(n):
            if d[i]:
                w[i] += step * d[i]
        if step < 1e-12:
            break
    total = sum(w)
    if total <= 0.0:
        return [1.0 / n] * n
    return [wi / total for wi in w]


def matching_mse(columns: list[list[float]], target: list[float], weights: list[float]) -> float:
    m = len(target)
    if m == 0:
        return float("inf")
    total = 0.0
    for k in range(m):
        approx = sum(weights[i] * columns[i][k] for i in range(len(columns)))
        total += (approx - target[k]) ** 2
    return total / m


# ---------------------------------------------------------------------------
# SHC forecast for a single standardized series
# ---------------------------------------------------------------------------


class SHCConfig:
    def __init__(self, args: argparse.Namespace, n_obs: int) -> None:
        self.donor_mode = args.donor_mode
        self.ridge = args.ridge
        self.bandwidth_grid = args.bandwidth_grid
        self.max_donors = args.max_donors
        self.max_blocks = getattr(args, "max_blocks", 120)
        self.stepwise_tol = args.stepwise_tol
        self.mse_tol = args.mse_tol
        self.lags = args.lags
        upper = min(args.pre_max, max(3, n_obs - 2))
        lower = min(args.pre_min, upper)
        self.m_min = max(3, lower)
        self.m_max = max(self.m_min, upper)


def _solve_beta(xs: list[list[float]], ys: list[float], mu_x: list[float], mu_y: list[float]) -> list[float]:
    """Least-squares beta from partialling-out residuals (Robinson 1988)."""
    p = len(xs[0]) if xs else 0
    if p == 0:
        return []
    a = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for t in range(len(ys)):
        vx = [xs[t][j] - mu_x[j] for j in range(p)]
        vy = ys[t] - mu_y
        for i in range(p):
            b[i] += vx[i] * vy
            for j in range(p):
                a[i][j] += vx[i] * vx[j]
    # Solve a @ beta = b with Gaussian elimination and tiny ridge.
    for i in range(p):
        a[i][i] += 1e-8
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            continue
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        for r in range(p):
            if r == col:
                continue
            factor = a[r][col] / a[col][col]
            if factor == 0.0:
                continue
            for c in range(col, p):
                a[r][c] -= factor * a[col][c]
            b[r] -= factor * b[col]
    return [b[i] / a[i][i] if abs(a[i][i]) > 1e-12 else 0.0 for i in range(p)]


def forecast_standardized(z: list[float], cfg: SHCConfig) -> dict:
    """Forecast the next value of a standardized series using SHC.

    Returns a diagnostics dictionary with ``forecast`` in standardized units.
    """
    n = len(z)
    diag: dict = {"n_obs": n, "status": "ok", "notes": []}
    if n < 6:
        fallback = z[-1] if z else 0.0
        diag.update({"status": "fallback", "forecast": fallback, "method": "last_value"})
        diag["notes"].append("series shorter than 6 points; used last value")
        return diag

    xs = [float(i) for i in range(n)]

    # Controls: lagged values of the same standardized series.
    lag = cfg.lags
    X: list[list[float]] = []
    y_used: list[float] = []
    idx_used: list[int] = []
    for t in range(n):
        if t - lag < 0:
            continue
        X.append([z[t - 1 - j] for j in range(lag)] if lag > 0 else [])
        y_used.append(z[t])
        idx_used.append(t)

    # First stage: smooth means of y (and controls) with a LOOCV bandwidth.
    bw, cv = select_bandwidth(xs, z, cfg.bandwidth_grid)
    mu_y = local_linear_predict(xs, z, bw, xs)
    beta: list[float] = []
    if lag > 0:
        mu_x_cols = [local_linear_predict(xs, [X[i][j] for i in range(len(X))], bw, xs) for j in range(lag)]
        # Align mu_x to y_used index space.
        mu_x_used = [[mu_x_cols[j][idx_used[i]] for j in range(lag)] for i in range(len(y_used))]
        mu_y_used = [mu_y[idx_used[i]] for i in range(len(y_used))]
        beta = _solve_beta(X, y_used, mu_x_used, mu_y_used)

    # Residual series y* = y - x'beta (beta empty -> y).
    y_star = list(z)
    if lag > 0:
        for i in range(len(y_used)):
            t = idx_used[i]
            y_star[t] = z[t] - sum(beta[j] * X[i][j] for j in range(lag))

    ell = local_linear_predict(xs, y_star, bw, xs)
    ell = [v if not math.isnan(v) else y_star[i] for i, v in enumerate(ell)]

    # Candidate pre-durations m; choose the largest m whose matching MSE is
    # within tolerance (paper Section 2.4).
    to = n  # 1-based index of last observed point
    best = None
    best_smallest = None
    for m in range(cfg.m_min, cfg.m_max + 1):
        if to - m < 2:
            continue
        target = [ell[to - m + i] for i in range(m)]  # treated pre-segment
        columns = []
        post_values = []
        for i in range(1, to - m + 1):
            if len(columns) >= cfg.max_blocks:
                break
            start = to - i - m  # 0-based start of historical pre-segment
            columns.append([ell[start + k] for k in range(m)])
            post_idx = to - i  # 0-based index of pseudo-post point
            post_values.append(y_star[post_idx])
        if not columns:
            continue
        weights = solve_simplex_least_squares(columns, target, ridge=cfg.ridge)
        mse = matching_mse(columns, target, weights)
        record = {
            "m": m,
            "mse": mse,
            "n_blocks": len(columns),
            "weights": weights,
            "post_values": post_values,
            "target": target,
            "columns": columns,
        }
        if best_smallest is None or mse < best_smallest["mse"] - 1e-15:
            best_smallest = record
        if mse <= cfg.mse_tol:
            best = record  # keep the largest acceptable m
    if best is None:
        best = best_smallest
    if best is None:
        diag.update({"status": "fallback", "forecast": z[-1], "method": "last_value"})
        diag["notes"].append("series too short for any historical block; used last value")
        return diag

    m = best["m"]
    columns = best["columns"]
    target = best["target"]
    post_values = best["post_values"]

    donor_indices = list(range(1, to - m + 1))

    # Optional parsimonious re-selection over the donor pool.
    if cfg.donor_mode == "single":
        single = min(
            range(len(columns)),
            key=lambda i: matching_mse([columns[i]], target, [1.0]),
        )
        cols2 = [columns[single]]
        pv2 = [post_values[single]]
        di = [donor_indices[single]]
        weights = [1.0]
    elif cfg.donor_mode == "stepwise":
        order: list[int] = []
        selected_cols: list[list[float]] = []
        selected_pv: list[float] = []
        prev_mse = matching_mse(columns, target, solve_simplex_least_squares(columns, target, ridge=cfg.ridge))
        remaining = list(range(len(columns)))
        while remaining and len(order) < cfg.max_donors:
            best_mse = None
            best_pick = None
            for cand in remaining:
                trial_cols = selected_cols + [columns[cand]]
                w = solve_simplex_least_squares(trial_cols, target, ridge=cfg.ridge)
                mse = matching_mse(trial_cols, target, w)
                if best_mse is None or mse < best_mse:
                    best_mse = mse
                    best_pick = cand
            trial_cols = selected_cols + [columns[best_pick]]
            trial_pv = selected_pv + [post_values[best_pick]]
            w = solve_simplex_least_squares(trial_cols, target, ridge=cfg.ridge)
            mse = matching_mse(trial_cols, target, w)
            selected_cols = trial_cols
            selected_pv = trial_pv
            order.append(best_pick)
            remaining.remove(best_pick)
            if prev_mse - mse <= cfg.stepwise_tol:
                break
            prev_mse = mse
        cols2 = selected_cols
        pv2 = selected_pv
        di = [donor_indices[i] for i in order]
        weights = solve_simplex_least_squares(cols2, target, ridge=cfg.ridge)
    else:  # all-in
        cols2 = columns
        pv2 = post_values
        di = donor_indices
        weights = solve_simplex_least_squares(cols2, target, ridge=cfg.ridge)

    ystar_pred = sum(weights[i] * pv2[i] for i in range(len(weights)))
    z_pred = ystar_pred
    if lag > 0:
        x_next = [z[n - 1 - j] for j in range(lag)]
        z_pred = ystar_pred + sum(beta[j] * x_next[j] for j in range(lag))

    top = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:5]
    diag.update(
        {
            "status": "ok",
            "method": f"shc_{cfg.donor_mode}",
            "forecast": z_pred,
            "m": m,
            "bandwidth": bw,
            "bandwidth_cv": cv,
            "matching_mse": matching_mse(cols2, target, weights),
            "n_donors": len(cols2),
            "lags": lag,
            "top_donor_weights": [
                {
                    "block_lag": to - di[i],
                    "weight": round(weights[i], 6),
                }
                for i in top
                if weights[i] > 0
            ],
            "mse_curve": [],
        }
    )
    return diag


# ---------------------------------------------------------------------------
# Band-level orchestration: detrend + standardize + SHC + invert
# ---------------------------------------------------------------------------


def detrend_series(values: list[float]) -> tuple[list[float], float]:
    """Remove an OLS linear trend. Returns (residuals, next-point trend value)."""
    n = len(values)
    if n < 3:
        return list(values), 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (values[i] - my) for i, x in enumerate(xs))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = my - slope * mx
    resid = [values[i] - (intercept + slope * xs[i]) for i in range(n)]
    return resid, intercept + slope * n


def forecast_band(times: list[float], values: list[float], args: argparse.Namespace) -> dict:
    n = len(values)
    diag: dict = {"n_obs": n, "status": "ok"}
    if n < 4:
        diag.update({"prediction": values[-1] if values else 0.0, "status": "fallback", "method": "last_value"})
        return diag

    resid = list(values)
    trend_next = 0.0
    if args.detrend == "linear":
        resid, trend_next = detrend_series(values)
    elif args.detrend == "auto":
        resid, trend_next = detrend_series(values)
        sd = statistics.pstdev(values) if n > 1 else 0.0
        slope_span = abs(trend_next - statistics.fmean(values))
        if sd <= 0 or slope_span < 0.2 * sd:
            resid = list(values)
            trend_next = 0.0

    mu = statistics.fmean(resid)
    sd = statistics.pstdev(resid) if n > 1 else 0.0
    if sd <= 1e-12:
        diag.update({"prediction": values[-1], "status": "constant", "method": "mean"})
        return diag
    z = [(v - mu) / sd for v in resid]

    cfg = SHCConfig(args, n)
    shc = forecast_standardized(z, cfg)
    pred_resid = shc["forecast"] * sd + mu
    prediction = pred_resid + trend_next

    diag.update(shc)
    diag["prediction"] = prediction
    diag["detrend"] = args.detrend
    diag["trend_next"] = trend_next
    diag["series_mean"] = mu
    diag["series_std"] = sd
    return diag


def build_rationale(diags: dict[str, dict], args: argparse.Namespace) -> str:
    ok = [b for b, d in diags.items() if d.get("status") == "ok"]
    fallback = [b for b, d in diags.items() if d.get("status") != "ok"]
    ms = sorted({d.get("m") for d in diags.values() if d.get("m")})
    donors = [d.get("n_donors") for d in diags.values() if d.get("n_donors")]
    bws = [d.get("bandwidth") for d in diags.values() if d.get("bandwidth")]
    parts = [
        "Forecast produced with the Synthetic Historical Control (SHC) method of "
        "Chen, Yang & Yang: each band's own distant history is used as the donor pool, "
        "because no cross-sectional controls are available for a single parcel.",
    ]
    if ok:
        parts.append(
            f"Applied the estimator to {len(ok)} band(s) with pre-duration m in {ms}, "
            f"LOOCV bandwidth ~{round(statistics.fmean(bws), 2) if bws else 'n/a'} "
            f"(in standardized units) and {round(statistics.fmean(donors), 1) if donors else 'n/a'} "
            f"historical controls on average."
        )
    if args.detrend != "none":
        parts.append(f"Detrending mode '{args.detrend}' was used because SHC matching requires recurring, non-growing latent patterns.")
    if fallback:
        parts.append(
            f"Bands {', '.join(sorted(fallback))} had too short or degenerate histories and used a documented fallback."
        )
    parts.append(
        "Weights are simplex-constrained (non-negative, sum to one) and chosen to minimize the "
        "pre-segment matching MSE; the same weights are applied to the historical pseudo-post "
        "segments to form each counterfactual."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 run_shc_band_forecast.py --input data.csv --out answer.json "
            "--summary-out summary.json\n"
        ),
    )
    parser.add_argument("--input", type=Path, default=Path("data.csv"),
                        help="Input CSV with the parcel's band history (default: data.csv).")
    parser.add_argument("--out", type=Path, default=Path("answer.json"),
                        help="Path for answer.json (default: ./answer.json).")
    parser.add_argument("--summary-out", type=Path, default=None,
                        help="Optional path for a diagnostics JSON summary.")
    parser.add_argument("--bands", type=str, default=None,
                        help="Comma-separated subset of band names to forecast (default: all).")
    parser.add_argument("--pre-min", type=int, default=4, dest="pre_min",
                        help="Minimum pre-duration m for the treated block (default: 4).")
    parser.add_argument("--pre-max", type=int, default=24, dest="pre_max",
                        help="Maximum pre-duration m for the treated block (default: 24).")
    parser.add_argument("--mse-tol", type=float, default=0.05, dest="mse_tol",
                        help="Max pre-segment matching MSE (standardized units) to accept a larger m (default: 0.05).")
    parser.add_argument("--donor-mode", choices=["all-in", "stepwise", "single"], default="all-in",
                        dest="donor_mode", help="Donor-pool strategy (default: all-in).")
    parser.add_argument("--max-blocks", type=int, default=120, dest="max_blocks",
                        help="Cap on historical blocks per band, using the most recent (default: 120).")
    parser.add_argument("--max-donors", type=int, default=8, dest="max_donors",
                        help="Max donors in stepwise mode (default: 8).")
    parser.add_argument("--stepwise-tol", type=float, default=1e-4, dest="stepwise_tol",
                        help="MSE-improvement stopping tolerance for stepwise mode (default: 1e-4).")
    parser.add_argument("--ridge", type=float, default=1e-6,
                        help="Ridge penalty for numerical stability of the simplex QP (default: 1e-6).")
    parser.add_argument("--lags", type=int, default=0,
                        help="Number of lagged self-controls in the semi-parametric regression (default: 0).")
    parser.add_argument("--detrend", choices=["auto", "linear", "none"], default="auto",
                        help="Detrending to apply before SHC matching (default: auto).")
    parser.add_argument("--bandwidth-grid", type=float, nargs="+", dest="bandwidth_grid",
                        default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0],
                        help="Candidate bandwidths as fractions of the series length.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        series = load_series(args.input)
    except (OSError, ValueError) as exc:
        print(f"error: failed to read {args.input}: {exc}", file=sys.stderr)
        return 2
    if not series:
        print(f"error: no usable band series found in {args.input}", file=sys.stderr)
        return 2

    if args.bands:
        wanted = {b.strip() for b in args.bands.split(",") if b.strip()}
        series = {b: v for b, v in series.items() if b in wanted}
        missing = wanted - set(series)
        if missing:
            print(f"warning: requested bands not found: {sorted(missing)}", file=sys.stderr)

    predictions: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}
    for band, points in series.items():
        if len(points) < 2:
            print(f"warning: band {band!r} has fewer than 2 points; skipped", file=sys.stderr)
            continue
        times = [t for t, _ in points]
        values = [v for _, v in points]
        diag = forecast_band(times, values, args)
        diagnostics[band] = diag
        predictions[band] = round(float(diag["prediction"]), 6)

    if not predictions:
        print("error: no band produced a prediction", file=sys.stderr)
        return 3

    answer = {"prediction": predictions, "rationale": build_rationale(diagnostics, args)}
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: could not write {args.out}: {exc}", file=sys.stderr)
        return 4

    summary = {
        "input": str(args.input),
        "output": str(args.out),
        "n_bands": len(predictions),
        "bands": sorted(predictions),
        "predictions": predictions,
        "donor_mode": args.donor_mode,
        "detrend": args.detrend,
        "lags": args.lags,
        "band_diagnostics": {
            band: {k: v for k, v in diag.items() if k not in {"columns", "target", "post_values"}}
            for band, diag in diagnostics.items()
        },
    }
    if args.summary_out is not None:
        try:
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write {args.summary_out}: {exc}", file=sys.stderr)
            return 4
    print(json.dumps({"status": "ok", "n_bands": len(predictions), "predictions": predictions}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
