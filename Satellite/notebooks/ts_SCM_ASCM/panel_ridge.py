"""panel_ridge — Experiment 3: the text-SC ridge estimator on a latent panel.

Wang, Liu, Zhang, Roth & Richardson, "Event Causality Identification with Synthetic
Control" (arXiv:2509.18156), Section 3.2, fit synthetic-control weights on text
embeddings by UNCONSTRAINED ridge regression:

    min_w  ||u_study - sum_j w_j u_j||^2  +  lambda * sum_j w_j^2,   lambda = 1.0

No simplex, no sum-to-one, no non-negativity — the estimator sits between our SCM
(`panel_scm`, simplex) and our ASCM (`panel_ascm`, simplex-anchored ridge with sum
w = 1). Their setup is structurally close to ours: 2–5 donors (retrieve n = 100, filter
to 2–5) on 1536-d text-embedding-ada-002 vectors — but T0 = 1, a single pre-treatment
embedding, so they have no held-out pre-period and never measure counterfactual
fidelity (evaluation is a downstream GPT-3.5 binary judgment). This module runs their
estimator through OUR P09/P10 holdout, with the same donors, scaler, schemes and flags
as `panel_scm` / `panel_ascm`, so the comparison is measured rather than argued.

Two details the paper leaves undetermined, both carried as explicit variants:

1. **lambda = 1.0 does not transport.** ada-002 vectors are returned L2-normalized, so
   their loss is O(1) and lambda = 1 is real shrinkage. Our design is 980 dims x 8
   periods of z-scored values (SSE ~ 7,840), where a literal lambda = 1 is numerically
   indistinguishable from unregularized least squares. Both readings are run.
2. **The loss sums from j = 0 while the penalty sums from j = 1**, and the paper never
   says what w_0 is (its prose says "weights w_1...w_J"). Read here as an UNPENALIZED
   intercept — of independent interest, since an unpenalized intercept is equivalent to
   the per-unit pre-period demeaning used by Tian, Lee & Panchenko (arXiv:2304.02272).

Variants (keyed `variant`; `panel_lib` already uses `arm` for main/quality40):

    paper         z-scored design as-is,          lambda = 1.0 fixed,  no intercept
    unitnorm      each site x period z-scored vector L2-normalized and the per-period
                  loss averaged — restores the O(1) loss scale of the single-period
                  ada-002 case, so lambda = 1 means what it means in the paper
    cv            z-scored design as-is,          lambda by leave-one-fit-period-out CV
    cv_intercept  same CV, plus the unpenalized intercept (the j = 0 term)

INTERPRETIVE TRAP, guarded in the notebook reading: `y` is z-scored against the pooled
P01–P08 mean, so w -> 0 predicts the pooled mean and gives RMSPE exactly 1 — the mean
baseline. Unconstrained ridge therefore has an escape hatch SCM/ASCM do not: as lambda
grows it converges TO the baseline rather than to a donor mix. A CV-chosen lambda
landing at ~1.00 means "shrank to the mean", not "matched the baseline". `flag_level`
(pass iff RMSPE < 1) stays meaningful, since shrinkage approaches 1 from above.

Scoring is identical for every variant: weights are always applied to the standard
z-scored design (`Panel.design`), never to a variant's preprocessed one, so
`valid_rmse` is comparable across variants and against SCM/ASCM.
"""
import numpy as np
import pandas as pd

from panel_lib import SENSORS, TRAIN8, TRAIN9, POST, J, LAMBDAS, flags

VARIANTS = ("paper", "unitnorm", "cv", "cv_intercept")
PAPER_LAMBDA = 1.0                    # Wang et al. Section 3.2, set and not tuned


# ---------------------------------------------------------------- the estimator
def ridge_fit(y, X, lam, intercept=False):
    """Closed-form unconstrained ridge in the paper's SSE convention:

        min_{w, b}  ||y - X w - b||^2  +  lam * ||w||^2

    `b` is UNPENALIZED (the paper's j = 0 term) and is 0.0 unless `intercept`.
    Returns (w (J,), b float). Deliberately NO sum-to-one and NO non-negativity
    assert — dropping the simplex is the whole point of this estimator.
    """
    Jn = X.shape[1]
    if intercept:
        xbar, ybar = X.mean(axis=0), float(y.mean())
        Xc, yc = X - xbar, y - ybar
    else:
        xbar, ybar, Xc, yc = np.zeros(Jn), 0.0, X, y
    A = Xc.T @ Xc + lam * np.eye(Jn)
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(ybar - xbar @ w) if intercept else 0.0
    assert np.all(np.isfinite(w)) and np.isfinite(b), (lam, intercept)
    return w, b


def unitnorm_design(y, X, n_periods):
    """The `unitnorm` variant's preprocessing (fitting only, never scoring).

    Each stacked period block is one site x period z-scored 980-vector. L2-normalize
    every such vector (treated and each donor) to unit norm, then divide through by
    sqrt(n_periods) so the objective is the AVERAGE per-period squared error — the
    O(1) loss scale of the paper's single-period ada-002 case, which is what makes
    lambda = 1.0 mean there what it means here.
    """
    D = len(y) // n_periods
    yb = y.reshape(n_periods, D)
    Xb = X.reshape(n_periods, D, X.shape[1])
    yn = np.linalg.norm(yb, axis=1, keepdims=True)
    xn = np.linalg.norm(Xb, axis=1, keepdims=True)
    yn[yn == 0] = 1.0
    xn[xn == 0] = 1.0
    s = np.sqrt(n_periods)
    return (yb / yn / s).reshape(-1), (Xb / xn / s).reshape(-1, X.shape[1])


def _design_for(panel, variant, tid, sensor, seqs, dl):
    """(y, X) used for FITTING under `variant`; scoring always uses panel.design."""
    y, X = panel.design(tid, sensor, seqs, dl)
    if variant == "unitnorm":
        return unitnorm_design(y, X, len(seqs))
    return y, X


def _spec(variant):
    """(fixed lambda or None if CV, intercept flag) for a variant."""
    return ({"paper": (PAPER_LAMBDA, False),
             "unitnorm": (PAPER_LAMBDA, False),
             "cv": (None, False),
             "cv_intercept": (None, True)}[variant])


def rmse_with_b(panel, w, b, tid, sensor, seqs, dl):
    """RMSPE in P01–P08 SD units on the STANDARD z-scored design, intercept included.
    Mirrors `Panel.rmse_of` (which has no intercept term) so every variant, SCM and
    ASCM are scored by the same formula on the same scale."""
    y, X = panel.design(tid, sensor, seqs, dl)
    r = y - X @ w - b
    return float(np.sqrt(np.mean(r * r)))


def null_rmse(panel, tid, sensor, seqs, dl):
    """RMSPE of the w = 0 predictor — i.e. the pooled P01–P08 mean, the point
    unconstrained ridge shrinks toward. Pooled over all 60 sites this is 1 by
    construction of the scaler, but PER SITE it varies around 1, so a holdout RMSPE
    just under 1 can be the null rather than predictive power. Every ridge row
    carries this alongside `valid_rmse`; `gain_over_null` is the honest metric."""
    return rmse_with_b(panel, np.zeros(len(dl)), 0.0, tid, sensor, seqs, dl)


# ---------------------------------------------------------------- fitting
def fit_weights(panel, donors, variants=VARIANTS, arm="main"):
    """Ridge fits for w8 (P01–P08) and w9 (P01–P09) per variant x sensor x site.

    Returns (weights DataFrame, lambda-CV DataFrame,
             W: {(variant, sensor, tid, window): (w, b, dl, seqs, lam)}).
    """
    w_rows, cv_rows, W = [], [], {}
    for variant in variants:
        fixed_lam, intercept = _spec(variant)
        for sensor in SENSORS:
            for tid in panel.treatments:
                dl = donors[arm].get((sensor, tid))
                if dl is None:
                    continue
                for wname, window in (("P01-08", TRAIN8), ("P01-09", TRAIN9)):
                    seqs = panel.usable(tid, sensor, window, dl, arm)
                    if len(seqs) < 2:
                        W[(variant, sensor, tid, wname)] = None
                        continue
                    y, X = _design_for(panel, variant, tid, sensor, seqs, dl)

                    if fixed_lam is not None:
                        lam, at_bound = fixed_lam, False
                    else:
                        cv_sse = np.zeros(len(LAMBDAS))
                        for hold in seqs:
                            rest = [q for q in seqs if q != hold]
                            if len(rest) < 2:
                                continue
                            y_r, X_r = _design_for(panel, variant, tid, sensor,
                                                   rest, dl)
                            y_h, X_h = panel.design(tid, sensor, [hold], dl)
                            for li, lam_i in enumerate(LAMBDAS):
                                wf, bf = ridge_fit(y_r, X_r, lam_i, intercept)
                                r = y_h - X_h @ wf - bf
                                cv_sse[li] += float(r @ r)
                        best = int(np.argmin(cv_sse))
                        lam = float(LAMBDAS[best])
                        at_bound = best in (0, len(LAMBDAS) - 1)
                        for li, lam_i in enumerate(LAMBDAS):
                            cv_rows.append({"variant": variant, "sensor": sensor,
                                            "treatment_site_id": tid,
                                            "fit_window": wname,
                                            "lambda": float(lam_i),
                                            "cv_sse": float(cv_sse[li]),
                                            "chosen": li == best})

                    w, b = ridge_fit(y, X, lam, intercept)
                    W[(variant, sensor, tid, wname)] = (w, b, dl, seqs, lam)
                    sq = float(np.sum(w ** 2))
                    neg = w[w < 0]
                    for c, wi in zip(dl, w):
                        w_rows.append({"variant": variant, "sensor": sensor,
                                       "treatment_site_id": tid,
                                       "fit_window": wname, "donor": c,
                                       "weight": float(wi), "lambda": lam,
                                       "lambda_at_boundary": at_bound,
                                       "intercept": b,
                                       "sum_weights": float(w.sum()),
                                       "weight_l2_norm": float(np.sqrt(sq)),
                                       "largest_donor_weight": float(w.max()),
                                       "n_negative_weights": int((w < 0).sum()),
                                       "sum_negative_weight": float(-neg.sum()),
                                       "effective_donors":
                                           float(1.0 / sq) if sq > 1e-12 else np.nan,
                                       "n_train_periods": len(seqs)})
    wdf = pd.DataFrame(w_rows)
    assert len(wdf) == len(variants) * 2 * len(panel.treatments) * 2 * J, len(wdf)
    return wdf, pd.DataFrame(cv_rows), W


def validate(panel, W, variants=VARIANTS, arm="main"):
    """Both validation schemes for every fitted variant x site x sensor.

    `frozen_joint`: w8 scored on P09+P10 pooled (the collaborator's notebook-14 split).
    `expanding`:    w8 -> P09, w9 -> P10 (the advisor's design).
    Schema matches `panel_scm_validation.csv` plus variant/lambda/sum_weights/intercept.
    """
    val_rows = []
    for variant in variants:
        for sensor in SENSORS:
            for tid in panel.treatments:
                f8 = W.get((variant, sensor, tid, "P01-08"))
                f9 = W.get((variant, sensor, tid, "P01-09"))
                if f8 is None or f9 is None:
                    continue
                w8, b8, dl, seqs8, lam8 = f8
                w9, b9, _, seqs9, lam9 = f9
                tr8 = rmse_with_b(panel, w8, b8, tid, sensor, seqs8, dl)
                tr9 = rmse_with_b(panel, w9, b9, tid, sensor, seqs9, dl)
                entries = [
                    ("frozen_joint", "pooled", w8, b8, lam8, tr8,
                     panel.usable(tid, sensor, [9, 10], dl, arm), len(seqs8)),
                    ("expanding", "P09", w8, b8, lam8, tr8,
                     panel.usable(tid, sensor, [9], dl, arm), len(seqs8)),
                    ("expanding", "P10", w9, b9, lam9, tr9,
                     panel.usable(tid, sensor, [10], dl, arm), len(seqs9))]
                for scheme, ep, w, b, lam, tr, seqs, ntr in entries:
                    row = {"variant": variant, "arm": arm, "sensor": sensor,
                           "treatment_site_id": tid, "scheme": scheme,
                           "eval_period": ep, "lambda": lam,
                           "sum_weights": float(w.sum()), "intercept": b,
                           "train_rmse": tr, "n_train_periods": ntr,
                           "n_valid_periods": len(seqs)}
                    if not seqs:
                        row.update({"valid_rmse": np.nan, "ratio": np.nan,
                                    "flag_ratio": "no_data", "flag_level": "no_data",
                                    "null_rmse": np.nan, "gain_over_null": np.nan})
                    else:
                        v = rmse_with_b(panel, w, b, tid, sensor, seqs, dl)
                        ratio, fr, fl = flags(tr, v)
                        nl = null_rmse(panel, tid, sensor, seqs, dl)
                        row.update({"valid_rmse": v, "ratio": ratio,
                                    "flag_ratio": fr, "flag_level": fl,
                                    "null_rmse": nl, "gain_over_null": nl - v})
                    val_rows.append(row)
    val = pd.DataFrame(val_rows)
    assert len(val) == len(variants) * 2 * len(panel.treatments) * 3, len(val)
    return val


def effects(panel, W, variants=VARIANTS, arm="main"):
    """P11–P20 latent-gap norms with the frozen w8 per variant."""
    eff_rows = []
    for variant in variants:
        for sensor in SENSORS:
            for tid in panel.treatments:
                fit = W.get((variant, sensor, tid, "P01-08"))
                if fit is None:
                    continue
                w8, b8, dl, _, _ = fit
                for q in POST:
                    have_t = panel.ok(tid, sensor, q, arm)
                    have_d = all(panel.ok(c, sensor, q, arm) for c in dl)
                    g = np.nan
                    if have_t and have_d:
                        zt = panel.Z(tid, sensor, q)
                        zs = np.tensordot(w8, np.stack(
                            [panel.Z(c, sensor, q) for c in dl]), axes=1) + b8
                        g = float(np.sqrt(np.mean((zt - zs) ** 2)))
                    eff_rows.append({"variant": variant, "sensor": sensor,
                                     "treatment_site_id": tid, "period": f"P{q}",
                                     "latent_gap_norm": g,
                                     "treated_latent_present": have_t,
                                     "donors_present": have_d})
    eff = pd.DataFrame(eff_rows)
    assert len(eff) == len(variants) * 2 * len(panel.treatments) * len(POST), len(eff)
    return eff


def run_ridge(panel, donors, variants=VARIANTS, arm="main"):
    """Full Experiment-3 pass: fits (+ lambda CV) -> validation -> effects.
    Returns dict(weights=..., lambda_cv=..., validation=..., effects=..., W=...)."""
    wdf, cv, W = fit_weights(panel, donors, variants, arm)
    val = validate(panel, W, variants, arm)
    eff = effects(panel, W, variants, arm)
    return {"weights": wdf, "lambda_cv": cv, "validation": val, "effects": eff, "W": W}


# ---------------------------------------------------------------- comparison
KEYS = ["sensor", "treatment_site_id", "scheme", "eval_period"]


def compare_all(scm_val, ascm_val, ridge_val, arm="main"):
    """Tidy comparison: one row per sensor x site x scheme x eval_period with SCM,
    ASCM and every ridge variant's holdout RMSPE, the w = 0 null, and the winner.

    `beats_null` — not `best_rmse < 1` — is the verdict column: with per-site nulls
    scattered around 1, an RMSPE just below 1 proves nothing on its own.
    """
    out = (scm_val.query("arm == @arm")[KEYS + ["valid_rmse"]]
           .rename(columns={"valid_rmse": "valid_rmse_scm"})
           .merge(ascm_val.query("arm == @arm")[KEYS + ["valid_rmse"]]
                  .rename(columns={"valid_rmse": "valid_rmse_ascm"}), on=KEYS))
    for variant in sorted(ridge_val["variant"].unique()):
        sub = (ridge_val.query("arm == @arm and variant == @variant")
               [KEYS + ["valid_rmse"]]
               .rename(columns={"valid_rmse": f"valid_rmse_ridge_{variant}"}))
        out = out.merge(sub, on=KEYS)
    nulls = (ridge_val.query("arm == @arm")[KEYS + ["null_rmse"]]
             .drop_duplicates(subset=KEYS))
    out = out.merge(nulls, on=KEYS)
    cols = [c for c in out.columns if c.startswith("valid_rmse_")]
    out["best_method"] = out[cols].idxmin(axis=1).str.replace("valid_rmse_", "",
                                                              regex=False)
    out["best_rmse"] = out[cols].min(axis=1)
    out["gain_over_null"] = out["null_rmse"] - out["best_rmse"]
    out["beats_null"] = out["gain_over_null"] > 0
    out["beats_mean_baseline"] = out["best_rmse"] < 1.0
    return out


def plot_three_way(cmp_, path, scheme="frozen_joint", sensors=SENSORS):
    """Holdout RMSPE by site for SCM / ASCM / each ridge variant, per sensor.

    Two references are drawn, and the per-site one is the one that matters: the black
    rule is that site's w = 0 null (the pooled-mean predictor, where unconstrained
    ridge shrinks to), the grey dashed line is RMSPE 1. A point below the dashed line
    but sitting on the black rule has not predicted anything.

    Drawn as dots rather than bars so the y-axis can zoom to the region where
    everything actually sits (~0.7-1.3) without a truncated bar baseline.
    """
    import matplotlib.pyplot as plt
    cols = [c for c in cmp_.columns if c.startswith("valid_rmse_")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, sensor in zip(axes, sensors):
        sub = cmp_.query("sensor == @sensor and scheme == @scheme")
        sites = sorted(sub["treatment_site_id"].unique())
        s = sub.set_index("treatment_site_id").loc[sites]
        x = np.arange(len(sites), dtype=float)
        span = 0.62
        for i, c in enumerate(cols):
            dx = (i - (len(cols) - 1) / 2) * span / max(len(cols) - 1, 1)
            ax.scatter(x + dx, s[c], s=34, zorder=4,
                       label=c.replace("valid_rmse_", ""))
        for xi, nv in zip(x, s["null_rmse"]):
            ax.plot([xi - span / 1.6, xi + span / 1.6], [nv, nv], color="k",
                    lw=1.8, zorder=5,
                    label="w = 0 null (pooled mean)" if xi == 0 else None)
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ax.text(0.995, 1.0, "RMSPE = 1", transform=ax.get_yaxis_transform(),
                fontsize=7, color="0.4", ha="right", va="bottom")
        for xi in x[:-1]:
            ax.axvline(xi + 0.5, color="0.9", lw=0.6, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([t[-2:] for t in sites])
        ax.set_xlim(-0.6, len(sites) - 0.4)
        ax.set_xlabel("treatment site")
        ax.set_ylabel("holdout RMSPE (SD units)")
        ax.set_title(f"{sensor} — {scheme}")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=8, ncol=len(lb),
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig
