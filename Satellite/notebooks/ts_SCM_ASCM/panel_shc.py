"""panel_shc — Synthetic Historical Control (Chen, Yang & Yang, SSRN 4995085) on one site's
own history, for a vector outcome (the 980-d TerraMind latent), covariate-free.

Written 2026-09-04 from the paper and the 4 Sep meeting only (memory
`project-shc-corrected-design`); nothing here descends from the withdrawn SHC code.

Model (paper eq 2 with p = 0):   y_t = l_t + delta_t d_t + eps_t,   d_t = 1 for t > T0.
Every function takes one site × sensor series `Y` of shape (T, D): rows are periods
1..T in order, columns the D outcome coordinates; a period that is not usable is an
all-NaN row.

Stage 1  smooth_latent / choose_bandwidth   eq 21 (p = 0): local-linear kernel regression of
         each coordinate on time over the PRE-period only; l_hat_t = fitted intercept.
         Bandwidth by the paper's leave-one-out CV, pooled over the D coordinates.
Blocks   blocks(T0, m, n)                   eq 7: treated block = last m pre + n post periods;
         historical block i shifted by z_i = n + i - 1; N = T0 - n - (m - 1).
Stage 2  fit_weights                        eq 23/24: simplex least squares of the treated
         pre segment of l_hat on the historical blocks' pre segments (stacked m*D rows).
         stepwise_donors                    eqs 32-33: best single block, then add the block
         with the largest MSE_pre decrease while the decrease exceeds zeta.
Effect   counterfactual                     meeting formula: l_hat0_{T0+k} = sum_j w_j
         l_hat_{hist j, post k}; Delta_{T0+k} = y_{T0+k} - l_hat0_{T0+k}.  Secondary, paper
         eq 27: the same weights on the RAW forward rows.
m rules  choose_m                           2.4: (a) largest m with MSE_pre <= nu;
         (b) m minimising a validation error supplied by the caller.

Tolerances follow the paper's Brexit application: nu = 0.1 % and zeta = 0.001 % of the
series' pre-period sample variance (pooled over coordinates).
"""
import numpy as np

import panel_align as pa

H_GRID = np.logspace(-1.5, 1.0, 26)     # bandwidth grid (units of T0 periods), LOO-CV over it
NU_FRAC, ZETA_FRAC = 1e-3, 1e-5         # paper fn 16/17: 0.1 % and 0.001 % of the sample variance
MIN_NEIGHBOURS = 2                      # a target period needs >= 2 finite neighbours


# ----------------------------------------------------------------- stage 1: latent component
def smoother_matrix(T0, h, finite_rows, loo=False):
    """S (T0 x T0): row t is the intercept row of the local-linear weighted LS fit at
    target t, regressor (tau - t)/T0, Gaussian kernel K((tau - t)/(h T0)); rows tau that
    are not finite get zero weight; `loo` also zeroes tau = t. Rows with fewer than
    MIN_NEIGHBOURS usable weights are NaN."""
    S = np.full((T0, T0), np.nan)
    tau = np.arange(T0, dtype=float)
    for t in range(T0):
        u = (tau - t) / T0
        k = np.exp(-0.5 * (u / h) ** 2) * finite_rows
        if loo:
            k = k.copy(); k[t] = 0.0
        if (k > 1e-12).sum() < MIN_NEIGHBOURS:
            continue
        Z = np.column_stack([np.ones(T0), u])
        G = Z.T @ (k[:, None] * Z)
        if np.linalg.cond(G) > 1e12:
            continue
        S[t] = np.linalg.solve(G, (Z.T * k))[0]     # intercept row of (Z'KZ)^-1 Z'K
    return S


def smooth_latent(Y_pre, h):
    """l_hat (T0 x D) = S(h) @ Y_pre, coordinate by coordinate along time. Missing periods
    (all-NaN rows) get zero weight; their own l_hat row is still estimated from the
    neighbours."""
    Y = np.asarray(Y_pre, float)
    fin = np.isfinite(Y).all(1).astype(float)
    S = smoother_matrix(Y.shape[0], h, fin)
    Yz = np.where(np.isfinite(Y), Y, 0.0)
    L = S @ Yz
    L[np.isnan(S).any(1)] = np.nan
    return L


def choose_bandwidth(Y_pre, grid=H_GRID):
    """Paper 2.3: leave-one-out CV, CV(h) = sum_t sum_d (y_td - l_hat^(-t)_td)^2 over the
    finite pre periods, pooled over the D coordinates. Returns (h, flag, rows)."""
    Y = np.asarray(Y_pre, float)
    fin = np.isfinite(Y).all(1)
    Yz = np.where(np.isfinite(Y), Y, 0.0)
    rows, score = [], {}
    for h in grid:
        S = smoother_matrix(Y.shape[0], h, fin.astype(float), loo=True)
        ok = fin & ~np.isnan(S).any(1)
        if ok.sum() < fin.sum():                    # not every finite period is evaluable
            score[h] = np.nan
        else:
            e = Y[ok] - (S[ok] @ Yz)
            score[h] = float(np.mean(e ** 2))
    usable = [g for g in grid if np.isfinite(score[g])]
    if not usable:
        raise RuntimeError("no bandwidth on the grid is evaluable at every pre period")
    h = float(min(usable, key=lambda g: score[g]))
    flag = "floor" if h == usable[0] else "ceiling" if h == usable[-1] else "interior"
    for g in grid:
        rows.append({"h": float(g), "cv": score[g], "chosen": bool(g == h), "flag": flag})
    return h, flag, rows


# ----------------------------------------------------------------- pseudo-panel (eq 7)
def blocks(T0, m, n):
    """Treated block and the N = T0 - n - (m - 1) historical blocks as 1-based period lists.
    Returns (treated_pre, treated_post, [(pre_i, post_i)] for i = 1..N oldest-shift-last:
    block i is the treated window shifted back by z_i = n + i - 1)."""
    N = T0 - n - (m - 1)
    assert m >= 1 and n >= 1 and N >= 1, (T0, m, n, N)
    treated_pre = list(range(T0 - m + 1, T0 + 1))
    treated_post = list(range(T0 + 1, T0 + n + 1))
    hist = []
    for i in range(1, N + 1):
        z = n + i - 1
        hist.append(([q - z for q in treated_pre], [q - z for q in treated_post]))
    return treated_pre, treated_post, hist


# ----------------------------------------------------------------- stage 2: weights
def _design(L, treated_pre, hist):
    """Stacked target (m*D,) and design (m*D, N) from l_hat rows (period p -> L[p-1])."""
    y = np.concatenate([L[q - 1] for q in treated_pre])
    X = np.column_stack([np.concatenate([L[q - 1] for q in pre]) for pre, _ in hist])
    return y, X


RIDGE_APPLIED = {"count": 0}                    # how often eq 24's ridge was needed (diagnostic)


def fit_weights(y, X, cols):
    """Simplex least squares on the selected columns (paper eq 23). If the Gram X'X is
    singular, eq 24: add the penalty ||varsigma^(1/2) C2' w||^2 with C2 the eigenvectors of
    the zero eigenvalues and varsigma a small positive number (R nearPD's role), by
    augmenting the least-squares system; counted in RIDGE_APPLIED."""
    Xs = X[:, cols]
    if Xs.shape[1] == 1:
        w = np.array([1.0])
    else:
        G = Xs.T @ Xs
        ev, V = np.linalg.eigh(G)
        null = ev <= 1e-10 * ev.max()
        if null.any():
            RIDGE_APPLIED["count"] += 1
            vs = 1e-6 * ev.max()
            Xa = np.vstack([Xs, np.sqrt(vs) * V[:, null].T])
            ya = np.concatenate([y, np.zeros(int(null.sum()))])
            w = pa.simplex_scm(ya, Xa)
        else:
            w = pa.simplex_scm(y, Xs)
        w = np.where(np.abs(w) < 1e-12, 0.0, w); w = w / w.sum()
    mse = float(np.mean((y - Xs @ w) ** 2))          # eq 31 on the stacked coordinates
    return w, mse


def stepwise_donors(y, X, zeta):
    """Paper 2.4 stepwise matching. Returns (selected column indices, w, mse, path rows,
    candidate rows) — every candidate tried at every step is recorded."""
    N = X.shape[1]
    selected, remaining, cur = [], list(range(N)), np.inf
    path, cands = [], []
    step = 0
    while remaining:
        step += 1
        trials = []
        for i in remaining:
            w, mse = fit_weights(y, X, selected + [i])
            trials.append((i, w, mse))
            cands.append({"step": step, "candidate": i + 1, "trial_mse": mse,
                          "improvement": (cur - mse) if np.isfinite(cur) else np.nan,
                          "trial_w": " ".join(f"{v:.4f}" for v in w)})
        i, w, mse = min(trials, key=lambda z: z[2])
        gain = cur - mse if np.isfinite(cur) else np.inf
        accept = (not selected) or gain > zeta
        path.append({"step": step, "added": i + 1, "mse": mse, "improvement": gain, "accepted": accept})
        if not accept:
            break
        selected.append(i); remaining.remove(i); cur = mse
        w_sel = w
    return selected, w_sel, cur, path, cands


# ----------------------------------------------------------------- effect
def counterfactual(L, Y, w, cols, hist, treated_post):
    """Meeting formula: l_hat0 at each post period = sum_j w_j l_hat[hist j post k]
    (smoothed latent at the donors' forward rows, all inside the pre-period); effect =
    Y_post - l_hat0. Secondary (paper eq 27): the same weights on the raw Y forward rows.
    Returns dict of (n, D) arrays: l0, y_post, delta, y0_raw, delta_raw."""
    n = len(treated_post)
    l0 = np.stack([sum(w[j] * L[hist[c][1][k] - 1] for j, c in enumerate(cols)) for k in range(n)])
    y0 = np.stack([sum(w[j] * Y[hist[c][1][k] - 1] for j, c in enumerate(cols)) for k in range(n)])
    y_post = np.stack([Y[q - 1] for q in treated_post])
    return {"l0": l0, "y_post": y_post, "delta": y_post - l0, "y0_raw": y0, "delta_raw": y_post - y0}


# ----------------------------------------------------------------- one full SHC fit
def shc_fit(Y, T0, m, n, h=None):
    """Everything for one site × sensor × (m, n): stage 1 on Y[:T0], blocks, stepwise
    donors, weights, counterfactual for periods T0+1..T0+n (must exist in Y for the
    effect; NaN otherwise). Returns a dict."""
    Y = np.asarray(Y, float)
    assert Y.shape[0] >= T0, Y.shape
    Y_pre = Y[:T0]
    fin = np.isfinite(Y_pre).all(1)
    var_pre = float(np.nanvar(Y_pre[fin])) if fin.sum() >= 2 else np.nan
    if h is None:
        h, h_flag, h_rows = choose_bandwidth(Y_pre)
    else:
        h_flag, h_rows = "fixed", []
    L = smooth_latent(Y_pre, h)
    treated_pre, treated_post, hist = blocks(T0, m, n)
    used = sorted({q for q in treated_pre} | {q for pre, post in hist for q in pre + post})
    if any(np.isnan(L[q - 1]).any() for q in used):
        return {"ok": False, "note": "l_hat missing at a block period", "h": h, "h_flag": h_flag,
                "h_rows": h_rows, "treated_pre": treated_pre, "treated_post": treated_post, "hist": hist}
    y, X = _design(L, treated_pre, hist)
    nu, zeta = NU_FRAC * var_pre, ZETA_FRAC * var_pre
    sel, w, mse, path, cands = stepwise_donors(y, X, zeta)
    singles = [fit_weights(y, X, [i])[1] for i in range(X.shape[1])]
    i1 = int(np.argmin(singles)); mse_single = singles[i1]
    w_all, mse_all = fit_weights(y, X, list(range(X.shape[1])))
    n_post_have = int(sum(q <= Y.shape[0] and np.isfinite(Y[q - 1]).all() for q in treated_post))
    Yfull = np.vstack([Y, np.full((max(0, T0 + n - Y.shape[0]), Y.shape[1]), np.nan)])
    cf = counterfactual(L, Yfull, w, sel, hist, treated_post)
    return {"ok": True, "note": "", "h": h, "h_flag": h_flag, "h_rows": h_rows, "L": L,
            "treated_pre": treated_pre, "treated_post": treated_post, "hist": hist,
            "selected": [c + 1 for c in sel], "w": w, "mse_pre": mse, "N": X.shape[1],
            "n_eff": float(1.0 / np.sum(w ** 2)), "path": path, "candidates": cands,
            "single_block": i1 + 1, "mse_single": mse_single, "w_all": w_all, "mse_all": mse_all,
            "var_pre": var_pre, "nu": nu, "zeta": zeta, "passes_nu": bool(mse <= nu),
            "n_post_observed": n_post_have, **cf}


def choose_m(results_by_m, validation_by_m=None):
    """(a) largest m whose MSE_pre <= nu; (b) m with the smallest validation error (caller
    supplies {m: error}). Returns dict."""
    ok = [m for m, r in results_by_m.items() if r["ok"] and r["passes_nu"]]
    out = {"m_rule_a": max(ok) if ok else None}
    if validation_by_m:
        v = {m: e for m, e in validation_by_m.items() if np.isfinite(e)}
        out["m_rule_b"] = min(v, key=v.get) if v else None
    return out
