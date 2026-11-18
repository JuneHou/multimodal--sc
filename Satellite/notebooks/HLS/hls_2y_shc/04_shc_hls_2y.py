#!/usr/bin/env python3
"""Per-site SHC for HLS NDVI (2-year treatment history), following an_08's
executable estimator.

CONSOLIDATED HERE on 2026-11-11: this script, and its output, used to be
triplicated verbatim under each of hls_2y_random_control/, hls_2y_nearest_
control/ and hls_2y_nearest20_control/ -- SHC has no geographic donors, so
all three copies read the same treatment-only panel and produced BYTE-
IDENTICAL output (verified: md5 match on site_summary.csv at every cadence,
and on site_periods_full_<cadence>.csv across all three source panels,
before this move). One copy, one results folder
(test results/validation/hls_flooding_128/hls_2y_shc/), is authoritative now; nothing design-specific
was ever in this script to begin with.

Intentional adaptations:
* Use each site's finite available observations; retain calendar gaps.
* Smooth and form historical blocks on OBSERVATION POSITIONS, not elapsed time.
* Use historical, strictly forward-in-time placebo ATT comparisons only.
* Record real dates, gaps, spans, threshold failures, and placebo resolution.

This script is self-contained: notebook 16 and an_08 are not runtime dependencies.
It follows an_08's CODE thresholds (.10 / .001), not its conflicting Markdown.
The rank comparison is not CWZ inference or automatically a calibrated p-value.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

CALS = Path('/Users/gaoyujuan/REAP Dropbox/Gao yujuan/Virginia Tech/CALS')
ROOT = CALS / 'test results' / 'hls_2y_shc'
EVENT = pd.Timestamp('2024-09-27')
DAMAGE_END = pd.Timestamp('2025-01-01')
POOLS = ('all_in', 'selected', 'single')
WEIGHT_OPT_TOL = 1e-12
WEIGHT_OPT_MAXITER = 5000
EIGEN_TOL_REL = 1e-10
PD_NULL_RIDGE_REL = 1e-8

# Estimator functions below are copied verbatim from an_08's code cell 5.

AN08_SOURCE_SHA256 = '9336bf49aac107073f821bf35e889ffdd2ce75ea023e54584a48711468a100e8'

def gaussian_kernel(u):
    return np.exp(-0.5 * np.asarray(u, dtype=float) ** 2)

def local_linear_predict(t_train, y_train, t_eval, bandwidth, exclude_index=None):
    t_train = np.asarray(t_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
    out = np.full(len(t_eval), np.nan, dtype=float)
    for q, t0 in enumerate(t_eval):
        good = np.isfinite(t_train) & np.isfinite(y_train)
        if exclude_index is not None:
            good[int(exclude_index)] = False
        tt, yy = t_train[good], y_train[good]
        w = gaussian_kernel((tt - t0) / bandwidth)
        X = np.column_stack([np.ones(len(tt)), tt - t0])
        XtW = X.T * w
        out[q] = (np.linalg.pinv(XtW @ X) @ (XtW @ yy))[0]
    return out

def select_time_bandwidth(t, y, grid):
    """LOOCV over `grid`, exactly an_06's objective but with the grid passed in."""
    rows = []
    for h in grid:
        pred = np.array([
            local_linear_predict(t, y, [t[i]], h, exclude_index=i)[0]
            for i in range(len(y))
        ])
        rows.append({"bandwidth": h, "loo_mse": float(np.mean((y - pred) ** 2))})
    table = pd.DataFrame(rows).sort_values(["loo_mse", "bandwidth"]).reset_index(drop=True)
    return float(table.iloc[0]["bandwidth"]), table

def _fista_simplex(target, X, C2=None, ridge=0.0, max_iter=20000, tol=1e-13):
    def project(v):
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1
        idx = np.arange(1, len(v) + 1)
        rho = np.nonzero(u - css / idx > 0)[0][-1]
        return np.maximum(v - css[rho] / (rho + 1.0), 0)
    A = X.T @ X
    if C2 is not None and C2.shape[1] and ridge > 0:
        A = A + ridge * (C2 @ C2.T)
    b = X.T @ target
    step = 1.0 / (2 * max(float(np.linalg.eigvalsh(A)[-1]), 1e-12))
    w = np.full(X.shape[1], 1.0 / X.shape[1])
    y, t = w.copy(), 1.0
    for _ in range(max_iter):
        w_new = project(y - step * (2 * (A @ y - b)))
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = w_new + ((t - 1) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w, t = w_new, t_new
    return w

def solve_weights_manual(target, donor_matrix, selected_cols=None):
    target = np.asarray(target, dtype=float)
    donor_matrix = np.asarray(donor_matrix, dtype=float)
    selected_cols = (list(range(donor_matrix.shape[1]))
                     if selected_cols is None else list(selected_cols))
    X = donor_matrix[:, selected_cols]
    J = X.shape[1]
    if J == 1:
        resid = target - X[:, 0]
        return {"weights": np.array([1.0]), "mse": float(np.mean(resid ** 2)),
                "solver": "closed_form"}
    gram = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(gram)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    null_mask = eigvals <= EIGEN_TOL_REL * scale
    C2 = eigvecs[:, null_mask] if null_mask.any() else np.empty((J, 0))
    ridge = PD_NULL_RIDGE_REL * scale if C2.shape[1] > 0 else 0.0

    def objective(w):
        resid = target - X @ w
        out = float(np.sum(resid ** 2))
        if C2.shape[1]:
            out += float(ridge * np.sum((C2.T @ w) ** 2))
        return out

    start = np.full(J, 1.0 / J)
    scale0 = objective(start) or 1.0
    res = minimize(lambda w: objective(w) / scale0, start, method="SLSQP",
                   bounds=[(0.0, 1.0)] * J,
                   constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                   options={"ftol": WEIGHT_OPT_TOL, "maxiter": WEIGHT_OPT_MAXITER})
    if res.success:
        w, solver = np.asarray(res.x, dtype=float), "slsqp"
    else:
        w, solver = _fista_simplex(target, X, C2=C2, ridge=ridge), "fista_fallback"
    w[np.abs(w) < 1e-12] = 0.0
    if w.sum() <= 0:
        raise RuntimeError("Estimated weights sum to zero.")
    w = w / w.sum()
    resid = target - X @ w
    return {"weights": w, "mse": float(np.mean(resid ** 2)), "solver": solver}

def stepwise_donors_manual(target, donor_matrix, blocks, zeta):
    N = donor_matrix.shape[1]
    single_mse = np.array([np.mean((target - donor_matrix[:, j]) ** 2) for j in range(N)])
    selected = [int(np.argmin(single_mse))]
    remaining = [j for j in range(N) if j not in selected]
    current = solve_weights_manual(target, donor_matrix, selected)
    while remaining:
        trials = [(j, solve_weights_manual(target, donor_matrix, selected + [j]))
                  for j in remaining]
        best_j, best_fit = min(trials, key=lambda x: x[1]["mse"])
        if current["mse"] - best_fit["mse"] <= zeta:
            break
        selected.append(best_j)
        remaining.remove(best_j)
        current = best_fit
    return {"selected_cols": selected,
            "weights": np.asarray(current["weights"], dtype=float),
            "mse": current["mse"]}


def span_days(dates):
    dates = pd.DatetimeIndex(dates)
    return int((dates[-1] - dates[0]).days) if len(dates) else 0


def gap_count(dates, cadence):
    """Number of adjacent observed pairs skipping at least one nominal period."""
    dates = pd.DatetimeIndex(dates)
    if len(dates) < 2:
        return 0
    if cadence == 'monthly':
        return int((np.diff(dates.to_period('M').asi8) > 1).sum())
    nominal = 7 if cadence == 'weekly' else 14
    return int((np.diff(dates.asi8) > pd.Timedelta(days=nominal).value).sum())


def site_frames(panel_dir, cadence, args):
    """Keep available per-site NDVI observations; do not fill missing periods."""
    # NDVI comes from the full grid itself, NOT from conus_random10_<cadence>_ndvi.csv.
    # That panel is the donor-trimmed one, clipped to the ORIGINAL 2023-2025
    # window even when treatment has been extended -- cross-referencing it by
    # date silently turned every extended pre-event date into NaN (absent =
    # "invalid"), so n_pre read back as 12-13 even though the grid itself
    # already had 25. site_periods_full_<cadence>.csv now carries ndvi
    # directly, so no merge against a differently-windowed file is needed.
    full = panel_dir / f'site_periods_full_{cadence}.csv'
    periods = pd.read_csv(full if full.exists()
                          else panel_dir / f'site_periods_{cadence}.csv',
                          dtype={'site': str})
    if not {'site', 'date', 'phase', 'ndvi'} <= set(periods):
        raise ValueError('Period grid requires site/date/phase/ndvi')
    periods['date'] = pd.to_datetime(periods.date, errors='raise')
    if periods[['site', 'date']].isna().any().any():
        raise ValueError('Missing site/date keys')
    if periods.duplicated(['site', 'date']).any():
        raise ValueError('Duplicate site/date keys')
    periods['ndvi'] = pd.to_numeric(periods.ndvi, errors='coerce')
    periods['phase'] = periods.phase.astype(str).str.lower()
    frames, audit = {}, []
    for site, grid in periods.groupby('site', sort=True):
        grid = grid.sort_values('date').copy()
        if grid.empty:
            audit.append(dict(site=site, eligible=False, reason='No site period grid'))
            continue
        usable_phase = grid.phase.isin(['pre', 'post'])
        # A composite can START before the intervention and still be post: the
        # period CONTAINING the event is labelled by which side most of its
        # days fall on, so comparing its start date to EVENT rejects a correct
        # label. Judge a post period by its END, and only a period that truly
        # ends before the event is inconsistent.
        ends = (pd.to_datetime(grid.period_end, errors='raise')
                if 'period_end' in grid else grid.date)
        if ((grid.phase.eq('pre') & grid.date.ge(EVENT))
            | (grid.phase.eq('post') & ends.lt(EVENT))).any():
            raise ValueError(f'{site}: phase inconsistent with intervention date')
        if 'period_end' in grid:
            # A PRE composite must not reach into the event or past it.
            if (grid.phase.eq('pre') & ends.ge(EVENT)).any():
                raise ValueError(f'{site}: a pretreatment composite includes event/post days')
        valid = grid.ndvi.notna() & np.isfinite(grid.ndvi) & grid.ndvi.between(-1, 1)
        pre = grid.loc[valid & grid.phase.eq('pre')]
        available_post = grid.loc[valid & grid.phase.eq('post')]
        if args.n_post == 'damage':
            post = available_post.loc[available_post.date.lt(DAMAGE_END)]
            requested = len(post)
        elif args.n_post == 'full':
            # Every available post period, not just the first 3 months --
            # att_3month below still restricts to DAMAGE_END independently,
            # so this does not cost that number anything. t0 (and so the
            # pre-period / matching-window search) is len(pre), unaffected
            # by how many post periods are requested.
            post = available_post
            requested = len(post)
        else:
            requested = int(args.n_post)
            post = available_post.iloc[:requested]
        n_pre, n_post = len(pre), len(post)
        reason = ''
        if n_pre < args.min_pre:
            reason = f'Too few pre observations: {n_pre} < {args.min_pre}'
        elif n_post < 1 or n_post != requested:
            reason = 'Insufficient post observations for requested horizon'
        elif n_pre < args.m_min + n_post:
            reason = 'No feasible matching length: m_min + n_post > T0; horizon NOT shortened'
        used = pd.concat([pre, post], ignore_index=True)
        audit.append(dict(site=site, eligible=not bool(reason), reason=reason,
            n_pre=n_pre, n_post=n_post, n_post_requested=requested,
            n_post_available=len(available_post),
            n_invalid_or_missing_grid_rows=int((usable_phase & ~valid).sum()),
            pre_gap_pairs=gap_count(pre.date, cadence),
            post_gap_pairs=gap_count(post.date, cadence),
            pre_start=pre.date.min(), pre_end=pre.date.max(),
            post_start=post.date.min(), post_end=post.date.max(),
            pre_span_days=span_days(pre.date), post_span_days=span_days(post.date)))
        if not reason:
            frames[site] = (used.rename(columns={'date': 'calendar_period'})[
                ['calendar_period', 'ndvi']].reset_index(drop=True), n_post)
    return frames, pd.DataFrame(audit)


def fit_one(frame, n_post, args):
    """Fit an_08's estimator using positions in the available observation series."""
    dates = pd.DatetimeIndex(frame.calendar_period)
    if (not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans):
        raise ValueError('Unique chronological dates are required; gaps are allowed')
    y = frame.ndvi.to_numpy(float)
    if not np.isfinite(y).all() or (np.abs(y) > 1).any():
        raise ValueError('NDVI must be finite and within [-1, 1]')
    t0 = len(y) - n_post
    cap = t0 - n_post if args.m_max is None else min(args.m_max, t0 - n_post)
    if t0 < args.min_pre or n_post < 1 or cap < args.m_min:
        raise ValueError('Insufficient pre/post history or no feasible m')
    ypre = y[:t0]
    # Kernel smoothing needs the REAL calendar gap between observations, not
    # their position in the series: a 21-day gap (one missed week) must get
    # less kernel weight than three real consecutive weeks, and a position-
    # only x (np.arange) treats them identically. Matching below (matrices())
    # deliberately keeps using observation positions -- see module docstring.
    x = ((dates[:t0] - dates[0]) / pd.Timedelta(days=7)).to_numpy(dtype=float)
    time_span = x[-1] - x[0] + 1.0
    normalized_grid = np.linspace(1.0 / time_span, args.bw_max, args.bw_grid_size)
    bandwidth, cv = select_time_bandwidth(x, ypre, normalized_grid * time_span)
    latent = local_linear_predict(x, ypre, x, bandwidth)
    sd = float(np.std(latent, ddof=1))
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        raise ValueError('Pretreatment latent series has zero/invalid variance')
    latent_z = (latent - latent.mean()) / sd
    variance = float(np.var(latent_z, ddof=1))
    nu, zeta = args.nu_share * variance, args.zeta_share * variance

    def matrices(m):
        # Adjacent donor starts: stride ONE observation, as in an_08.
        starts = np.arange(t0 - m - n_post + 1)
        pre_idx = starts[:, None] + np.arange(m)
        post_idx = starts[:, None] + m + np.arange(n_post)
        return pre_idx, post_idx, latent_z[-m:], latent_z[pre_idx].T

    scan = []
    for m in range(args.m_min, cap + 1):
        _, _, target, donors = matrices(m)
        result = solve_weights_manual(target, donors)
        scan.append(dict(m=m, latent_mse=result['mse'], nu=nu,
                         passes=result['mse'] <= nu, n_donors=donors.shape[1]))
    mscan = pd.DataFrame(scan)
    passing = mscan.loc[mscan.passes]
    if len(passing):
        m = int(passing.m.max())
        mode = 'nu_rule'
    elif args.strict_threshold:
        raise ValueError('No matching length passes fixed nu; strict mode rejects this fit')
    else:
        m = int(mscan.sort_values(['latent_mse', 'm']).iloc[0].m)
        mode = 'fallback_min_mse'
    pre_idx, post_idx, target, donors = matrices(m)
    n_donors = donors.shape[1]
    all_in = solve_weights_manual(target, donors)
    stepwise = stepwise_donors_manual(target, donors, None, zeta)
    selected_weights = np.zeros(n_donors)
    selected_weights[stepwise['selected_cols']] = stepwise['weights']
    single_index = int(np.argmin(np.mean((target[:, None] - donors) ** 2, axis=0)))
    single_weights = np.zeros(n_donors)
    single_weights[single_index] = 1.
    pool_weights = dict(all_in=all_in['weights'], selected=selected_weights,
                        single=single_weights)
    target_idx = np.r_[np.arange(t0 - m, t0), np.arange(t0, len(y))]
    actual_months = dates[t0:].month.to_numpy()
    paths, summaries, weights = [], [], []
    for pool, mass in pool_weights.items():
        if (mass < -1e-7).any() or not np.isclose(mass.sum(), 1, atol=1e-6):
            raise ValueError('Weight solver violated the simplex constraint')
        raw_cf = np.r_[ypre[pre_idx].T @ mass, ypre[post_idx].T @ mass]
        latent_cf = np.r_[latent[pre_idx].T @ mass, latent[post_idx].T @ mass]
        observed = y[target_idx]
        gaps = observed - raw_cf
        summaries.append(dict(pool=pool, m=m, n=n_post, T0=t0, N=n_donors,
            J=int((mass > 1e-10).sum()), selection_mode=mode,
            passes_nu=mode == 'nu_rule', bandwidth=bandwidth,
            normalized_bandwidth=bandwidth / time_span, nu_share=args.nu_share,
            zeta_share=args.zeta_share, nu=nu, zeta=zeta,
            pre_rmse_raw=float(np.sqrt(np.mean(gaps[:m] ** 2))),
            pre_rmse_latent=float(np.sqrt(np.mean((target - donors @ mass) ** 2))),
            mean_effect=float(gaps[m:].mean()), last_effect=float(gaps[-1]),
            matching_span_days=span_days(dates[t0 - m:t0]),
            post_span_days=span_days(dates[t0:]),
            pre_mse_window=float(np.mean(gaps[m - args.prefit_periods:m] ** 2))))
        paths.append(pd.DataFrame(dict(pool=pool,
            period=['pre'] * m + ['post'] * n_post,
            k=np.r_[np.arange(-m, 0), np.arange(1, n_post + 1)],
            calendar_date=dates[target_idx], observed=observed,
            counterfactual=raw_cf, effect=gaps,
            latent_counterfactual=latent_cf,
            effect_against_latent=observed - latent_cf)))
        for j in range(n_donors):
            donor_pre, donor_post = dates[pre_idx[j]], dates[post_idx[j]]
            weights.append(dict(pool=pool, donor_id=j + 1, weight=float(mass[j]),
                donor_pre_start=donor_pre[0], donor_pre_end=donor_pre[-1],
                donor_pseudopost_start=donor_post[0], donor_pseudopost_end=donor_post[-1],
                donor_pre_span_days=span_days(donor_pre),
                donor_post_span_days=span_days(donor_post),
                season_matched=bool(np.array_equal(donor_post.month.to_numpy(), actual_months))))
    cv = cv.assign(normalized_h=lambda d: d.bandwidth / time_span)
    return dict(summary=pd.DataFrame(summaries), paths=pd.concat(paths, ignore_index=True),
                weights=pd.DataFrame(weights), mscan=mscan, bandwidth_cv=cv,
                latent=pd.DataFrame(dict(calendar_date=dates[:t0], observed=ypre,
                    latent=latent, latent_z=latent_z, smoother_residual=ypre - latent)))


def historical_placebos(frame, n_post, actual_fit, args):
    """Chronological refits; same horizon, fixed q-period pre-MSE screen at 1x."""
    t0 = len(frame) - n_post
    actual = actual_fit['summary'].query("pool == 'selected'").iloc[0]
    actual_effect = float(actual.mean_effect)
    actual_mse = float(actual.pre_mse_window)
    stride = args.placebo_stride or n_post
    minimum = max(args.min_pre, args.m_min + n_post)
    cutoffs = list(reversed(range(t0 - n_post, minimum - 1, -stride)))
    audit = [dict(kind='actual', cutoff_index=t0, status='ok', eligible=False,
        pre_start=frame.calendar_period.iloc[0], pre_end=frame.calendar_period.iloc[t0 - 1],
        post_start=frame.calendar_period.iloc[t0], post_end=frame.calendar_period.iloc[-1],
        n_pre=t0, n_post=n_post, pre_mse=actual_mse, mean_effect=actual_effect,
        selection_mode=actual.selection_mode, passes_nu=bool(actual.passes_nu))]
    actual_months = pd.DatetimeIndex(frame.calendar_period.iloc[t0:]).month.to_numpy()
    kept, successful = [], 0
    for cutoff in cutoffs:
        row = dict(kind='placebo', cutoff_index=cutoff, eligible=False,
            n_pre=cutoff, n_post=n_post,
            pre_start=frame.calendar_period.iloc[0], pre_end=frame.calendar_period.iloc[cutoff - 1],
            post_start=frame.calendar_period.iloc[cutoff],
            post_end=frame.calendar_period.iloc[cutoff + n_post - 1])
        months = pd.DatetimeIndex(frame.calendar_period.iloc[cutoff:cutoff + n_post]).month.to_numpy()
        row['season_matched'] = bool(np.array_equal(months, actual_months))
        if args.placebo_season_match and not row['season_matched']:
            row['status'] = 'season_excluded'
            audit.append(row)
            continue
        try:
            # Pass only history and this pseudo-post window. No later data.
            placebo = fit_one(frame.iloc[:cutoff + n_post].copy(), n_post, args)
            s = placebo['summary'].query("pool == 'selected'").iloc[0]
            mse, effect = float(s.pre_mse_window), float(s.mean_effect)
            eligible = mse <= actual_mse
            row.update(status='ok', pre_mse=mse, mean_effect=effect,
                eligible=bool(eligible), selection_mode=s.selection_mode,
                passes_nu=bool(s.passes_nu), m=int(s.m), J=int(s.J),
                bandwidth=float(s.bandwidth), nu=float(s.nu), zeta=float(s.zeta))
            successful += 1
            if eligible:
                kept.append(effect)
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            row.update(status='fit_failed', error=str(exc))
        audit.append(row)
    effects = np.asarray(kept, float)
    if args.alternative == 'two-sided':
        extreme = np.abs(effects) >= abs(actual_effect)
    elif args.alternative == 'less':
        extreme = effects <= actual_effect
    else:
        extreme = effects >= actual_effect
    k = len(effects)
    return dict(actual_mean_effect=actual_effect, actual_pre_mse=actual_mse,
        n_candidate_blocks=len(cutoffs), n_successful_placebos=successful,
        n_eligible_placebos=k, n_extreme=int(extreme.sum()), alternative=args.alternative,
        placebo_rank_p=float((1 + extreme.sum()) / (k + 1)) if k else np.nan,
        placebo_p_floor=1 / (k + 1) if k else np.nan,
        placebo_n_post=n_post, placebo_stride=stride,
        status='ok' if k else 'no_eligible_placebos'), pd.DataFrame(audit)


def plot_site(fit, site, cadence, output):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    colors = dict(all_in='#4576aa', selected='#d35439', single='#8c68a9')
    for pool in POOLS:
        path = fit['paths'].query('pool == @pool').sort_values('calendar_date')
        axes[0].plot(path.calendar_date, path.counterfactual, color=colors[pool], label=pool)
        axes[1].plot(path.calendar_date, path.effect, color=colors[pool], label=pool)
    observed = fit['paths'].query("pool == 'selected'").sort_values('calendar_date')
    axes[0].plot(observed.calendar_date, observed.observed, color='black', label='Observed')
    for ax in axes:
        ax.axvline(EVENT, color='grey', ls='--')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('NDVI')
    axes[1].set_ylabel('Observed - raw counterfactual')
    axes[1].axhline(0, color='grey', lw=.7)
    fig.suptitle(f'{site} | {cadence} | gaps retained; blocks use observation positions')
    fig.tight_layout()
    safe_site = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(site))
    fig.savefig(output / f'{safe_site}_counterfactual_effect.png', dpi=170)
    plt.close(fig)


def fit_cadence(cadence, args):
    out = args.output / cadence / 'tables'
    figures = args.output / cadence / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    frames, eligibility = site_frames(args.panels, cadence, args)
    eligibility.to_csv(out / 'site_eligibility.csv', index=False)
    if not frames:
        raise ValueError(f'No eligible sites at {cadence}; see site_eligibility.csv')
    collections = {name: [] for name in ('summary', 'paths', 'weights', 'mscan', 'bandwidth_cv', 'latent')}
    reports, comparisons, audits, failures = [], [], [], []
    for index, (site, (frame, n_post)) in enumerate(sorted(frames.items()), 1):
        try:
            fit = fit_one(frame, n_post, args)
            row = fit['summary'].query("pool == 'selected'").iloc[0]
            selected = fit['paths'].query("pool == 'selected' and period == 'post'")
            selected_weights = fit['weights'].query("pool == 'selected'")
            season_weight = float(selected_weights.loc[selected_weights.season_matched, 'weight'].sum())
            report = dict(site=site, cadence=cadence, **row.to_dict(),
                n_pre=int(row.T0), n_post=n_post,
                att_damage=float(row.mean_effect),
                att_3month=float(selected.loc[selected.calendar_date.lt(DAMAGE_END), 'effect'].mean()),
                post_start=selected.calendar_date.min(), post_end=selected.calendar_date.max(),
                pre_gap_pairs=gap_count(frame.calendar_period.iloc[:-n_post], cadence),
                post_gap_pairs=gap_count(frame.calendar_period.iloc[-n_post:], cadence),
                any_season_matched_block=bool(selected_weights.season_matched.any()),
                season_matched_donor_weight=season_weight)
            if args.placebos:
                comparison, audit = historical_placebos(frame, n_post, fit, args)
                comparison.update(site=site, cadence=cadence)
                comparisons.append(comparison)
                audits.append(audit.assign(site=site, cadence=cadence))
                report.update({key: value for key, value in comparison.items()
                               if key not in ('site', 'cadence')})
            reports.append(report)
            for name in collections:
                collections[name].append(fit[name].assign(site=site, cadence=cadence))
            if not args.no_figures and index <= args.plot_sites:
                figures.mkdir(parents=True, exist_ok=True)
                plot_site(fit, site, cadence, figures)
            print(f'{cadence} {index}/{len(frames)} {site}: m={int(row.m)}, '
                  f'mode={row.selection_mode}, ATT={row.mean_effect:+.4f}', flush=True)
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            failures.append(dict(site=site, error=str(exc)))
            print(f'{cadence} {site}: FAILED {exc}', flush=True)
    pd.DataFrame(failures, columns=['site', 'error']).to_csv(out / 'failed_sites.csv', index=False)
    if not reports:
        raise ValueError(f'All sites failed at {cadence}; see failed_sites.csv')
    summary = pd.DataFrame(reports)
    summary.to_csv(out / 'site_summary.csv', index=False)
    names = dict(summary='pool_summary.csv', paths='site_paths.csv', weights='site_weights.csv',
                 mscan='site_mscan.csv', bandwidth_cv='bandwidth_loocv.csv', latent='pretreatment_latent.csv')
    combined = {}
    for key, filename in names.items():
        combined[key] = pd.concat(collections[key], ignore_index=True)
        combined[key].to_csv(out / filename, index=False)
    combined['weights'].query("pool == 'selected'").rename(
        columns={'season_matched': 'matches_target'}).to_csv(
        out / 'season_match_feasibility.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out / 'historical_placebo_comparison.csv', index=False)
    (pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()).to_csv(
        out / 'historical_placebo_audit.csv', index=False)
    post = combined['paths'].query("pool == 'selected' and period == 'post'")
    post.groupby('calendar_date', as_index=False).agg(n_sites=('site', 'nunique'),
        mean_observed=('observed', 'mean'), mean_counterfactual=('counterfactual', 'mean'),
        mean_effect=('effect', 'mean')).to_csv(out / 'att_by_period.csv', index=False)
    pd.DataFrame([dict(cadence=cadence, n_sites=len(summary),
        mean_att=float(summary.mean_effect.mean()), median_att=float(summary.mean_effect.median()),
        n_fallback_fits=int(summary.selection_mode.eq('fallback_min_mse').sum()),
        n_failed_sites=len(failures),
        note='Descriptive site averages; horizons/observed calendars may differ. No aggregate inference.')]
        ).to_csv(out / 'att_summary.csv', index=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(an08_source_sha256=AN08_SOURCE_SHA256,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        # The only real input: SHC reads no conus_<design>_<cadence>_ndvi.csv
        # at all (it has no donors). site_periods_<cadence>.csv is a fallback
        # path only, used when _full doesn't exist.
        input_sha256={name: hashlib.sha256((args.panels / name).read_bytes()).hexdigest()
            for name in (f'site_periods_full_{cadence}.csv',)
            if (args.panels / name).exists()},
        time_scale='available observation positions; real calendar gaps retained',
        inference='historical placebo ranks only; no t intervals, bootstrap or CWZ')
    (out / 'run_config.json').write_text(json.dumps(config, indent=2))
    print(f'{cadence}: {len(summary)} fits; '
          f'{summary.selection_mode.eq("fallback_min_mse").sum()} threshold fallbacks; '
          f'{len(failures)} failures. Calendar spans and season diagnostics are in tables.')
    return summary, bool(failures)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', type=Path, default=ROOT / 'panels')
    parser.add_argument('--output', type=Path, default=ROOT / 'shc')
    parser.add_argument('--cadence', nargs='+', choices=['monthly', 'biweekly', 'weekly'],
                        default=['monthly', 'biweekly', 'weekly'])
    parser.add_argument('--n-post', default='damage', help="'damage': observed post periods before 2025-01-01; 'full': every available post period through whatever the treatment panel covers; or a positive observation count")
    parser.add_argument('--min-pre', type=int, default=12, help='Same minimum for actual and placebo fits, in observations')
    parser.add_argument('--m-min', type=int, default=4)
    parser.add_argument('--m-max', type=int, default=None, help='Default: full feasible T0-n range as in an_08')
    parser.add_argument('--nu-share', type=float, default=.10)
    parser.add_argument('--zeta-share', type=float, default=.001)
    parser.add_argument('--bw-grid-size', type=int, default=50)
    parser.add_argument('--bw-max', type=float, default=.40)
    parser.add_argument('--strict-threshold', action='store_true', help='Reject no-passing-m fits instead of labeled an_08 fallback')
    parser.add_argument('--prefit-periods', type=int, default=4)
    parser.add_argument('--alternative', choices=['two-sided', 'less', 'greater'], default='two-sided')
    parser.add_argument('--placebo-stride', type=int, default=None, help='Default n_post: nonoverlapping pseudo-post positions')
    parser.add_argument('--placebo-season-match', action='store_true', help='Require same ordered month-of-year sequence for placebo post windows')
    parser.add_argument('--no-placebos', dest='placebos', action='store_false')
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--plot-sites', type=int, default=6)
    args = parser.parse_args(argv)
    if args.n_post not in ('damage', 'full'):
        try:
            args.n_post = int(args.n_post)
        except ValueError:
            parser.error('--n-post must be damage, full, or a positive integer')
        if args.n_post < 1:
            parser.error('--n-post must be positive')
    if (not 1 <= args.prefit_periods <= args.m_min <= args.min_pre
        or (args.m_max is not None and args.m_max < args.m_min)
        or args.bw_grid_size < 2 or not 0 < args.bw_max <= 1
        or args.bw_max < 1 / args.min_pre
        or not np.isfinite([args.nu_share, args.zeta_share, args.bw_max]).all()
        or args.nu_share <= 0 or args.zeta_share < 0
        or (args.placebo_stride is not None and args.placebo_stride < 1)
        or args.plot_sites < 0):
        parser.error('Invalid matching, bandwidth, prefit or placebo settings')
    return args


def main(argv=None):
    args = parse_args(argv)
    problems = []
    for cadence in args.cadence:
        try:
            _, failed = fit_cadence(cadence, args)
            if failed:
                problems.append(cadence + ': some sites failed; inspect failed_sites.csv')
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            problems.append(cadence + ': ' + str(exc))
    if problems:
        raise SystemExit('\n'.join(problems))


if __name__ == '__main__':
    main()
