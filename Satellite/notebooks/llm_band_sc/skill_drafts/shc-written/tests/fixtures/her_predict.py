
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---- cell 4: PRE_PERIODS (notebook lines 26-29) ----
PRE_PERIODS = [
    "P01", "P02", "P03", "P04", "P05",
    "P06", "P07", "P08", "P09", "P10",
]

# ---- cell 4: HURRICANE_PERIOD (notebook lines 31-31) ----
HURRICANE_PERIOD = "P11"

# ---- cell 4: POST_PERIODS (notebook lines 33-36) ----
POST_PERIODS = [
    "P12", "P13", "P14", "P15", "P16",
    "P17", "P18", "P19", "P20", "P21",
]

# ---- cell 4: ALL_PERIODS (notebook lines 38-38) ----
ALL_PERIODS = PRE_PERIODS + [HURRICANE_PERIOD] + POST_PERIODS

# ---- cell 4: M (notebook lines 46-46) ----
M = 4

# ---- cell 4: N_POST (notebook lines 47-47) ----
N_POST = 2

# ---- cell 4: TREATED_PRE (notebook lines 49-49) ----
TREATED_PRE = PRE_PERIODS[-M:]          # P07-P10

# ---- cell 4: TREATED_POST (notebook lines 50-50) ----
TREATED_POST = POST_PERIODS[:N_POST]    # P12-P13

# ---- cell 4: M_COMPARISON_VALUES (notebook lines 63-65) ----
M_COMPARISON_VALUES = list(
    range(2, len(PRE_PERIODS) - N_POST + 1)
)

# ---- cell 4: M_SENSITIVITY_VALUES (notebook lines 68-71) ----
M_SENSITIVITY_VALUES = [
    m for m in M_COMPARISON_VALUES
    if m != M
]

# ---- cell 4: M_SELECTION_NU_STD (notebook lines 91-91) ----
M_SELECTION_NU_STD = 1.0

# ---- cell 4: VALIDATION_PERIOD (notebook lines 98-98) ----
VALIDATION_PERIOD = "P10"

# ---- cell 4: VALIDATION_TRAIN_PERIODS (notebook lines 99-99) ----
VALIDATION_TRAIN_PERIODS = PRE_PERIODS[:-1]   # P01-P09

# ---- cell 4: VALIDATION_TREATED_PRE (notebook lines 100-100) ----
VALIDATION_TREATED_PRE = VALIDATION_TRAIN_PERIODS[-M:]  # P06-P09

# ---- cell 4: BANDS (notebook lines 107-110) ----
BANDS = {
    "sentinel1": ["VV", "VH", "VV_minus_VH"],
    "sentinel2": ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI"],
}

# ---- cell 4: USE_MONTH_SIN_COS (notebook lines 117-117) ----
USE_MONTH_SIN_COS = True

# ---- cell 4: BETA_TREND_DEGREE (notebook lines 121-121) ----
BETA_TREND_DEGREE = 1

# ---- cell 4: KERNEL_BANDWIDTH (notebook lines 123-123) ----
KERNEL_BANDWIDTH = 2.0

# ---- cell 4: ZETA (notebook lines 125-125) ----
ZETA = 1e-4

# ---- cell 4: SD_TOL (notebook lines 126-126) ----
SD_TOL = 1e-8

# ---- cell 4: OPT_TOL (notebook lines 128-128) ----
OPT_TOL = 1e-10

# ---- cell 4: OPT_MAXITER (notebook lines 129-129) ----
OPT_MAXITER = 5000

# ---- cell 7: CELL7_DERIVED (notebook lines 25-54) ----
CELL7_DERIVED_SRC = r'''feature_panel["period_num"] = (
    feature_panel["period_id"]
    .str.extract(r"(\d+)")[0]
    .astype(int)
)
feature_panel["month_date"] = pd.to_datetime(
    feature_panel["month"] + "-01"
)
feature_panel["calendar_month_num"] = (
    feature_panel["month_date"].dt.month
)
theta = (
    2.0
    * np.pi
    * feature_panel["calendar_month_num"].astype(float)
    / 12.0
)
feature_panel["x_sin_month"] = np.sin(theta)
feature_panel["x_cos_month"] = np.cos(theta)
feature_panel = (
    feature_panel
    .sort_values(["sensor", "site_id", "band", "period_num"])
    .reset_index(drop=True)
)
'''

# ---- cell 11: make_historical_blocks_from_periods (notebook lines 1-19) ----
def make_historical_blocks_from_periods(pre_periods, m, n):
    T0 = len(pre_periods)
    N = T0 - m - n + 1

    if N < 1:
        raise ValueError(
            f"Not enough history: T0={T0}, m={m}, n={n}"
        )

    blocks = []

    for j in range(N):
        blocks.append({
            "donor_id": j + 1,
            "pre_periods": list(pre_periods[j:j + m]),
            "post_periods": list(pre_periods[j + m:j + m + n]),
        })

    return blocks

# ---- cell 11: MAIN_BLOCKS (notebook lines 22-26) ----
MAIN_BLOCKS = make_historical_blocks_from_periods(
    PRE_PERIODS,
    m=M,
    n=N_POST,
)

# ---- cell 11: VALIDATION_BLOCKS (notebook lines 28-32) ----
VALIDATION_BLOCKS = make_historical_blocks_from_periods(
    VALIDATION_TRAIN_PERIODS,
    m=M,
    n=1,
)

# ---- cell 13: gaussian_kernel (notebook lines 1-2) ----
def gaussian_kernel(u):
    return np.exp(-0.5 * np.asarray(u, dtype=float) ** 2)

# ---- cell 13: local_linear_predict (notebook lines 5-55) ----
def local_linear_predict(
    t_train,
    y_train,
    t_eval,
    bandwidth,
):
    t_train = np.asarray(t_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    t_eval = np.atleast_1d(
        np.asarray(t_eval, dtype=float)
    )

    out = np.full(
        len(t_eval),
        np.nan,
        dtype=float,
    )

    good = (
        np.isfinite(t_train)
        & np.isfinite(y_train)
    )

    t_tr = t_train[good]
    y_tr = y_train[good]

    if len(t_tr) < 2:
        return out

    for i, t0 in enumerate(t_eval):

        u = (t_tr - t0) / bandwidth
        weights = gaussian_kernel(u)

        X_loc = np.column_stack([
            np.ones(len(t_tr)),
            t_tr - t0,
        ])

        XtW = X_loc.T * weights
        XtWX = XtW @ X_loc
        XtWy = XtW @ y_tr

        coef_loc = (
            np.linalg.pinv(XtWX)
            @ XtWy
        )

        out[i] = coef_loc[0]

    return out

# ---- cell 13: prepare_band_series (notebook lines 58-136) ----
def prepare_band_series(
    site_sensor_df,
    band,
    allowed_periods,
):
    df = (
        site_sensor_df[
            (site_sensor_df["band"] == band)
            & (
                site_sensor_df["period_id"]
                .isin(allowed_periods)
            )
        ]
        [[
            "period_id",
            "period_num",
            "month",
            "calendar_month_num",
            "mean_value",
            "x_sin_month",
            "x_cos_month",
        ]]
        .drop_duplicates("period_id")
        .copy()
    )

    # Force the exact estimation period order.
    df = (
        df
        .set_index("period_id")
        .reindex(allowed_periods)
        .reset_index()
    )

    # Rebuild period number after reindexing.
    df["period_num"] = (
        df["period_id"]
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    # Recover month information if reindexing created missing metadata.
    month_lookup = (
        site_sensor_df[
            [
                "period_id",
                "month",
                "calendar_month_num",
                "x_sin_month",
                "x_cos_month",
            ]
        ]
        .drop_duplicates("period_id")
        .set_index("period_id")
    )

    for col in [
        "month",
        "calendar_month_num",
        "x_sin_month",
        "x_cos_month",
    ]:
        if col not in df.columns:
            df[col] = np.nan

        missing = df[col].isna()

        if missing.any():
            df.loc[missing, col] = (
                df.loc[missing, "period_id"]
                .map(month_lookup[col])
            )

    df["y"] = pd.to_numeric(
        df["mean_value"],
        errors="coerce",
    )

    return df

# ---- cell 13: _trend_basis (notebook lines 139-162) ----
def _trend_basis(t, degree=1):
    """
    Basis used only to partial out a simple smooth trend while
    estimating the sine/cosine beta coefficients.

    degree=1 -> [1, centered t]
    degree=2 -> [1, centered t, centered t^2]
    """
    t = np.asarray(t, dtype=float)

    tc = (
        t - np.mean(t)
    )

    cols = [
        np.ones(len(t)),
    ]

    for d in range(1, degree + 1):
        cols.append(
            tc ** d
        )

    return np.column_stack(cols)

# ---- cell 13: estimate_seasonal_beta_and_latent (notebook lines 165-324) ----
def estimate_seasonal_beta_and_latent(
    site_sensor_df,
    band,
    allowed_periods,
):
    """
    Estimate:
        y_t = beta_sin * sin(month_t)
            + beta_cos * cos(month_t)
            + l_t
            + error_t

    Practical identification used here:
    - beta_sin and beta_cos are estimated from pre-treatment data
      after partialling out a low-order time trend.
    - only the estimated seasonal component is removed from y.
    - l_t is then estimated by local-linear kernel smoothing.

    No lagged outcome is used.
    """

    df = prepare_band_series(
        site_sensor_df,
        band,
        allowed_periods,
    )

    fit = df[
        np.isfinite(df["y"])
        & np.isfinite(df["x_sin_month"])
        & np.isfinite(df["x_cos_month"])
    ].copy()

    if len(fit) != len(allowed_periods):
        raise ValueError(
            f"{band}: only {len(fit)} usable observations; "
            f"expected {len(allowed_periods)}"
        )

    t = fit["period_num"].to_numpy(float)

    y = fit["y"].to_numpy(float)

    X_season = fit[
        [
            "x_sin_month",
            "x_cos_month",
        ]
    ].to_numpy(float)

    # --------------------------------------------------------------
    # Step A: partial out a simple time trend when estimating beta.
    # --------------------------------------------------------------
    T_basis = _trend_basis(
        t,
        degree=BETA_TREND_DEGREE,
    )

    # Residual-maker M_T = I - P_T.
    P_T = (
        T_basis
        @ np.linalg.pinv(
            T_basis.T @ T_basis
        )
        @ T_basis.T
    )

    M_T = (
        np.eye(len(t))
        - P_T
    )

    y_detrended = (
        M_T @ y
    )

    X_detrended = (
        M_T @ X_season
    )

    # --------------------------------------------------------------
    # Step B: estimate beta_sin and beta_cos.
    # --------------------------------------------------------------
    beta_hat = (
        np.linalg.pinv(
            X_detrended.T
            @ X_detrended
        )
        @ (
            X_detrended.T
            @ y_detrended
        )
    )

    beta_sin = float(
        beta_hat[0]
    )

    beta_cos = float(
        beta_hat[1]
    )

    # --------------------------------------------------------------
    # Step C: remove ONLY the estimated seasonal component.
    # --------------------------------------------------------------
    seasonal_component_hat = (
        X_season @ beta_hat
    )

    residualized_y = (
        y - seasonal_component_hat
    )

    # --------------------------------------------------------------
    # Step D: estimate latent smooth component l_t.
    # --------------------------------------------------------------
    l_hat = local_linear_predict(
        t_train=t,
        y_train=residualized_y,
        t_eval=t,
        bandwidth=KERNEL_BANDWIDTH,
    )

    fit["seasonal_component_hat"] = (
        seasonal_component_hat
    )

    fit["residualized_y"] = (
        residualized_y
    )

    fit["l_hat"] = (
        l_hat
    )

    # Optional descriptive form of the seasonal harmonic:
    # beta_sin*sin(theta) + beta_cos*cos(theta)
    # = amplitude * cos(theta - phase)
    seasonal_amplitude = float(
        np.sqrt(
            beta_sin ** 2
            + beta_cos ** 2
        )
    )

    seasonal_phase_radians = float(
        np.arctan2(
            beta_sin,
            beta_cos,
        )
    )

    beta_dict = {
        "beta_sin": beta_sin,
        "beta_cos": beta_cos,
        "seasonal_amplitude": seasonal_amplitude,
        "seasonal_phase_radians": seasonal_phase_radians,
    }

    return beta_dict, fit

# ---- cell 15: estimate_sensor_latents (notebook lines 1-94) ----
def estimate_sensor_latents(
    feature_panel,
    site_id,
    sensor,
    allowed_periods,
):

    site_df = feature_panel[
        (
            feature_panel["site_id"]
            == site_id
        )
        & (
            feature_panel["sensor"]
            == sensor
        )
    ].copy()

    latent_list = []
    beta_list = []

    for band in BANDS[sensor]:

        beta_dict, band_df = (
            estimate_seasonal_beta_and_latent(
                site_df,
                band,
                allowed_periods,
            )
        )

        if not np.isfinite(
            band_df["l_hat"]
        ).all():
            raise ValueError(
                f"{site_id} {sensor} {band}: "
                "latent estimate contains NaN"
            )

        # ----------------------------------------------------------
        # Standardize latent component separately by band.
        # ----------------------------------------------------------
        l_mean = float(
            band_df["l_hat"].mean()
        )

        l_sd = float(
            band_df["l_hat"].std(ddof=1)
        )

        if (
            not np.isfinite(l_sd)
            or l_sd < SD_TOL
        ):
            l_sd = 1.0

        band_df["l_hat_std"] = (
            band_df["l_hat"]
            - l_mean
        ) / l_sd

        band_df["band"] = band
        band_df["sensor"] = sensor
        band_df["site_id"] = site_id

        latent_list.append(
            band_df
        )

        beta_list.append({
            "site_id": site_id,
            "sensor": sensor,
            "band": band,
            **beta_dict,
            "latent_mean": l_mean,
            "latent_sd": l_sd,
            "n_periods_used": len(
                band_df
            ),
        })

    latent_df = pd.concat(
        latent_list,
        ignore_index=True,
    )

    beta_df = pd.DataFrame(
        beta_list
    )

    return (
        latent_df,
        beta_df,
    )

# ---- cell 17: build_band_problem (notebook lines 1-68) ----
def build_band_problem(
    latent_df,
    band,
    treated_pre,
    historical_blocks,
):

    band_df = (
        latent_df[
            latent_df["band"] == band
        ]
        .copy()
        .sort_values("period_num")
    )

    z_map = dict(
        zip(
            band_df["period_id"],
            band_df["l_hat_std"],
        )
    )

    y_map = dict(
        zip(
            band_df["period_id"],
            band_df["y"],
        )
    )

    z_target = np.array(
        [z_map.get(p, np.nan) for p in treated_pre],
        dtype=float,
    )

    y_target = np.array(
        [y_map.get(p, np.nan) for p in treated_pre],
        dtype=float,
    )

    Z_pre = np.column_stack([
        np.array(
            [z_map.get(p, np.nan) for p in b["pre_periods"]],
            dtype=float,
        )
        for b in historical_blocks
    ])

    Y_pre = np.column_stack([
        np.array(
            [y_map.get(p, np.nan) for p in b["pre_periods"]],
            dtype=float,
        )
        for b in historical_blocks
    ])

    y_all = band_df["y"].to_numpy(float)
    y_scale_var = float(np.nanvar(y_all, ddof=1))

    if not np.isfinite(y_scale_var) or y_scale_var < SD_TOL:
        y_scale_var = np.nan

    return {
        "z_target": z_target,
        "Z_pre": Z_pre,
        "y_target": y_target,
        "Y_pre": Y_pre,
        "y_scale_var": y_scale_var,
    }

# ---- cell 17: build_joint_problem (notebook lines 71-145) ----
def build_joint_problem(
    latent_df,
    sensor,
    treated_pre,
    historical_blocks,
):

    by_band = {}

    for band in BANDS[sensor]:
        by_band[band] = build_band_problem(
            latent_df,
            band,
            treated_pre,
            historical_blocks,
        )

    # Treated target must be complete for every band.
    for band, bp in by_band.items():
        if not np.isfinite(bp["z_target"]).all():
            return None, (
                f"{band}: treated latent target contains missing values"
            )

    n_original_donors = len(historical_blocks)

    # Keep only donor columns complete across all bands.
    valid = np.ones(
        n_original_donors,
        dtype=bool,
    )

    for bp in by_band.values():
        for j in range(n_original_donors):
            if not np.isfinite(bp["Z_pre"][:, j]).all():
                valid[j] = False

    original_donor_cols = np.where(valid)[0].tolist()

    if not original_donor_cols:
        return None, "no complete historical donor blocks"

    target_parts = []
    donor_parts = []

    for band in BANDS[sensor]:

        bp = by_band[band]

        bp["Z_pre"] = bp["Z_pre"][:, original_donor_cols]
        bp["Y_pre"] = bp["Y_pre"][:, original_donor_cols]

        target_parts.append(bp["z_target"])
        donor_parts.append(bp["Z_pre"])

    z_stacked = np.concatenate(target_parts)
    Z_stacked = np.vstack(donor_parts)

    if not np.isfinite(z_stacked).all():
        return None, "stacked treated latent vector contains missing values"

    if not np.isfinite(Z_stacked).all():
        return None, "stacked historical donor matrix contains missing values"

    problem = {
        "z_stacked": z_stacked,
        "Z_stacked": Z_stacked,
        "by_band": by_band,
        "n_donors": Z_stacked.shape[1],
        "n_original_donors": n_original_donors,
        # Maps local matrix column -> original historical block index.
        "original_donor_cols": original_donor_cols,
    }

    return problem, None

# ---- cell 19: solve_convex_weights (notebook lines 1-77) ----
def solve_convex_weights(
    target,
    donor_matrix,
    selected_cols=None,
):

    target = np.asarray(target, dtype=float)
    donor_matrix = np.asarray(donor_matrix, dtype=float)

    if selected_cols is None:
        selected_cols = list(
            range(donor_matrix.shape[1])
        )

    X = donor_matrix[:, selected_cols]
    J = X.shape[1]

    if J == 0:
        raise ValueError("No donor columns supplied.")

    if J == 1:
        w = np.array([1.0])
        mse = float(
            np.mean(
                (target - X[:, 0]) ** 2
            )
        )
        return w, mse

    x0 = np.full(
        J,
        1.0 / J,
    )

    def objective(w):
        return float(
            np.mean(
                (target - X @ w) ** 2
            )
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0,
        },
        options={
            "ftol": OPT_TOL,
            "maxiter": OPT_MAXITER,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"SLSQP failed: {result.message}"
        )

    w = np.asarray(
        result.x,
        dtype=float,
    )

    # Clean tiny numerical values and renormalize.
    w[np.abs(w) < 1e-12] = 0.0

    if w.sum() <= 0:
        raise RuntimeError(
            "Estimated weights sum to zero."
        )

    w = w / w.sum()

    return w, objective(w)

# ---- cell 19: stepwise_donor_selection (notebook lines 80-410) ----
def stepwise_donor_selection(problem):
    """
    Stepwise SHC donor selection with full candidate diagnostics.

    Returns
    -------
    selected : list[int]
        Local donor columns finally selected.

    final_w : np.ndarray
        Final convex weights for selected donors.

    final_mse : float
        Final joint standardized latent MSE.

    accepted_path : pd.DataFrame
        Only donors that were actually accepted.

    candidate_diagnostics : pd.DataFrame
        EVERY donor candidate tried at every step, including rejected
        candidates, its re-optimized trial weights, trial MSE, and
        improvement relative to the currently accepted donor set.
    """

    if problem is None:
        return (
            [],
            np.array([]),
            np.nan,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    n_donors = problem["n_donors"]
    target = np.asarray(
        problem["z_stacked"],
        dtype=float,
    )
    donor_matrix = np.asarray(
        problem["Z_stacked"],
        dtype=float,
    )

    original_donor_cols = list(
        problem["original_donor_cols"]
    )

    selected = []
    remaining = list(range(n_donors))
    current_mse = np.inf

    path_rows = []
    candidate_rows = []

    accepted_step = 0

    # Number of historical blocks in this specific problem.
    # This must be dynamic because changing m changes the donor-pool size.
    original_pool_size = int(
        problem.get(
            "n_original_donors",
            max(original_donor_cols) + 1,
        )
    )

    while remaining:

        selection_step = accepted_step + 1
        candidate_results = []

        # -------------------------------------------------------------
        # Try EVERY remaining donor.
        # We re-estimate convex weights for the full trial donor set.
        # -------------------------------------------------------------
        for candidate_local_col in remaining:

            trial_local_cols = (
                selected + [candidate_local_col]
            )

            trial_w, trial_mse = solve_convex_weights(
                target,
                donor_matrix,
                trial_local_cols,
            )

            improvement = (
                np.inf
                if not np.isfinite(current_mse)
                else float(current_mse - trial_mse)
            )

            candidate_original_col = int(
                original_donor_cols[
                    candidate_local_col
                ]
            )

            # Map trial weights back to original donor IDs D1-D5.
            weight_by_original = {
                original_col: 0.0
                for original_col in range(
                    original_pool_size
                )
            }

            for local_col, weight in zip(
                trial_local_cols,
                trial_w,
            ):
                original_col = int(
                    original_donor_cols[local_col]
                )
                weight_by_original[
                    original_col
                ] = float(weight)

            trial_original_cols = [
                int(original_donor_cols[x])
                for x in trial_local_cols
            ]

            candidate_results.append({
                "candidate_local_col":
                    candidate_local_col,
                "candidate_original_col":
                    candidate_original_col,
                "trial_local_cols":
                    trial_local_cols,
                "trial_original_cols":
                    trial_original_cols,
                "trial_w":
                    trial_w,
                "trial_mse":
                    float(trial_mse),
                "improvement":
                    improvement,
                "weight_by_original":
                    weight_by_original,
            })

        # -------------------------------------------------------------
        # Pick the candidate with the lowest trial MSE.
        # -------------------------------------------------------------
        best = min(
            candidate_results,
            key=lambda d: d["trial_mse"],
        )

        best_improvement = best["improvement"]

        # First donor is always accepted.
        accepted = (
            len(selected) == 0
            or best_improvement > ZETA
        )

        # -------------------------------------------------------------
        # Save EVERY candidate tried at this step.
        # -------------------------------------------------------------
        for result in candidate_results:

            row = {
                "selection_step":
                    selection_step,

                "candidate_local_col":
                    result[
                        "candidate_local_col"
                    ],

                "candidate_original_col":
                    result[
                        "candidate_original_col"
                    ],

                "candidate_donor_id":
                    result[
                        "candidate_original_col"
                    ] + 1,

                "selected_before_local_cols":
                    "|".join(
                        str(x)
                        for x in selected
                    ),

                "selected_before_donor_ids":
                    "|".join(
                        str(
                            int(
                                original_donor_cols[x]
                            ) + 1
                        )
                        for x in selected
                    ),

                "trial_local_cols":
                    "|".join(
                        str(x)
                        for x in result[
                            "trial_local_cols"
                        ]
                    ),

                "trial_donor_ids":
                    "|".join(
                        str(x + 1)
                        for x in result[
                            "trial_original_cols"
                        ]
                    ),

                "current_mse_before_addition":
                    (
                        float(current_mse)
                        if np.isfinite(current_mse)
                        else np.inf
                    ),

                "trial_mse":
                    result["trial_mse"],

                "improvement":
                    result["improvement"],

                "best_candidate_this_step":
                    (
                        result[
                            "candidate_local_col"
                        ]
                        ==
                        best[
                            "candidate_local_col"
                        ]
                    ),

                "accepted":
                    (
                        accepted
                        and
                        result[
                            "candidate_local_col"
                        ]
                        ==
                        best[
                            "candidate_local_col"
                        ]
                    ),
            }

            # Save one weight column for each original donor.
            for original_col in range(
                original_pool_size
            ):
                row[
                    f"weight_donor{original_col + 1}"
                ] = result[
                    "weight_by_original"
                ][original_col]

            candidate_rows.append(row)

        # -------------------------------------------------------------
        # Apply the stopping rule.
        # -------------------------------------------------------------
        if accepted:

            accepted_step += 1

            best_local_col = int(
                best["candidate_local_col"]
            )

            selected.append(best_local_col)
            remaining.remove(best_local_col)

            current_mse = float(
                best["trial_mse"]
            )

            best_original_col = int(
                original_donor_cols[
                    best_local_col
                ]
            )

            path_rows.append({
                "step":
                    accepted_step,

                "local_donor_col":
                    best_local_col,

                "original_donor_col":
                    best_original_col,

                "mse_after_addition":
                    current_mse,

                "improvement":
                    best_improvement,
            })

        else:
            # No remaining candidate improves MSE by more than ZETA.
            # We still saved all rejected candidates above.
            break

    if not selected:
        return (
            [],
            np.array([]),
            np.nan,
            pd.DataFrame(path_rows),
            pd.DataFrame(candidate_rows),
        )

    final_w, final_mse = solve_convex_weights(
        target,
        donor_matrix,
        selected,
    )

    return (
        selected,
        final_w,
        final_mse,
        pd.DataFrame(path_rows),
        pd.DataFrame(candidate_rows),
    )

# ---- cell 21: evaluate_pre_fit (notebook lines 1-79) ----
def evaluate_pre_fit(
    problem,
    selected_local_cols,
    weights,
):

    rows = []

    for band, bp in problem["by_band"].items():

        # -------------------------
        # Step 9: latent fit
        # -------------------------
        z_target = bp["z_target"]
        Z_pre = bp["Z_pre"][:, selected_local_cols]

        synthetic_z = Z_pre @ weights

        latent_mse = float(
            np.mean(
                (z_target - synthetic_z) ** 2
            )
        )

        # -------------------------
        # Step 10: observed outcome fit
        # -------------------------
        y_target = bp["y_target"]
        Y_pre = bp["Y_pre"][:, selected_local_cols]

        if (
            np.isfinite(y_target).all()
            and np.isfinite(Y_pre).all()
        ):
            synthetic_y = Y_pre @ weights

            outcome_mse = float(
                np.mean(
                    (y_target - synthetic_y) ** 2
                )
            )
        else:
            outcome_mse = np.nan

        scale_var = bp["y_scale_var"]

        outcome_nmse = (
            outcome_mse / scale_var
            if (
                np.isfinite(outcome_mse)
                and np.isfinite(scale_var)
                and scale_var > SD_TOL
            )
            else np.nan
        )

        rows.append({
            "band": band,
            "latent_mse_std": latent_mse,
            "outcome_mse_raw": outcome_mse,
            "outcome_variance_pre": scale_var,
            "outcome_nmse": outcome_nmse,
        })

    fit_df = pd.DataFrame(rows)

    joint_latent_mse = float(
        fit_df["latent_mse_std"].mean()
    )

    joint_outcome_nmse = float(
        fit_df["outcome_nmse"].mean()
    )

    return (
        fit_df,
        joint_latent_mse,
        joint_outcome_nmse,
    )

# ---- cell 25: get_observed_value (notebook lines 1-25) ----
def get_observed_value(
    feature_panel,
    site_id,
    sensor,
    band,
    period_id,
):

    row = feature_panel[
        (feature_panel["site_id"] == site_id)
        & (feature_panel["sensor"] == sensor)
        & (feature_panel["band"] == band)
        & (feature_panel["period_id"] == period_id)
    ]

    if row.empty:
        return np.nan

    value = row["mean_value"].iloc[0]

    return (
        float(value)
        if pd.notna(value)
        else np.nan
    )

# ---- cell 25: heldout_validation_for_sensor (notebook lines 28-192) ----
def heldout_validation_for_sensor(
    feature_panel,
    site_id,
    sensor,
):

    latent_df, beta_df = estimate_sensor_latents(
        feature_panel,
        site_id,
        sensor,
        VALIDATION_TRAIN_PERIODS,
    )

    problem, problem_error = build_joint_problem(
        latent_df,
        sensor,
        VALIDATION_TREATED_PRE,
        VALIDATION_BLOCKS,
    )

    if problem is None:
        raise ValueError(problem_error)

    (
        selected_local_cols,
        weights,
        _,
        donor_path,
        candidate_diagnostics,
    ) = stepwise_donor_selection(problem)

    if not selected_local_cols:
        raise ValueError(
            "validation donor selection selected no donors"
        )

    rows = []

    for band in BANDS[sensor]:

        actual = get_observed_value(
            feature_panel,
            site_id,
            sensor,
            band,
            VALIDATION_PERIOD,
        )

        donor_values = []

        for local_col in selected_local_cols:

            original_col = problem[
                "original_donor_cols"
            ][local_col]

            block = VALIDATION_BLOCKS[
                original_col
            ]

            pseudo_validation_period = (
                block["post_periods"][0]
            )

            donor_values.append(
                get_observed_value(
                    feature_panel,
                    site_id,
                    sensor,
                    band,
                    pseudo_validation_period,
                )
            )

        donor_values = np.asarray(
            donor_values,
            dtype=float,
        )

        if (
            np.isfinite(actual)
            and np.isfinite(donor_values).all()
        ):
            synthetic = float(
                donor_values @ weights
            )

            error = actual - synthetic
            sq_error = float(error ** 2)

        else:
            synthetic = np.nan
            error = np.nan
            sq_error = np.nan

        # Standardization scale uses P01–P09 only.
        train_values = [
            get_observed_value(
                feature_panel,
                site_id,
                sensor,
                band,
                p,
            )
            for p in VALIDATION_TRAIN_PERIODS
        ]

        train_values = np.asarray(
            train_values,
            dtype=float,
        )

        train_var = float(
            np.nanvar(
                train_values,
                ddof=1,
            )
        )

        if (
            not np.isfinite(train_var)
            or train_var < SD_TOL
        ):
            train_var = np.nan

        standardized_sq_error = (
            sq_error / train_var
            if (
                np.isfinite(sq_error)
                and np.isfinite(train_var)
            )
            else np.nan
        )

        rows.append({
            "site_id": site_id,
            "sensor": sensor,
            "band": band,
            "actual_P10": actual,
            "synthetic_P10": synthetic,
            "error": error,
            "sq_error": sq_error,
            "training_variance": train_var,
            "standardized_sq_error": standardized_sq_error,
        })

    validation_df = pd.DataFrame(rows)

    joint_validation_nmse = float(
        validation_df[
            "standardized_sq_error"
        ].mean()
    )

    return {
        "results": validation_df,
        "selected_local_cols": selected_local_cols,
        "weights": weights,
        "joint_validation_nmse": joint_validation_nmse,
        "problem": problem,
        "donor_path": donor_path,
        "candidate_diagnostics": candidate_diagnostics,
        "beta": beta_df,
        "latent": latent_df,
    }


import pandas as pd, numpy as np
d = pd.read_csv("data.csv")
d["band"] = d["band"].replace({"VV-VH": "VV_minus_VH"})
d["sensor"] = "sentinel1" if "VV" in set(d["band"]) else "sentinel2"
d["site_id"] = "site"
d["period_num"] = d["period_id"].str.extract(r"(\d+)")[0].astype(int)
d["calendar_month_num"] = pd.to_datetime(d["month"] + "-01").dt.month
th = 2.0 * np.pi * d["calendar_month_num"].astype(float) / 12.0
d["x_sin_month"], d["x_cos_month"] = np.sin(th), np.cos(th)
d = d.sort_values(["sensor", "site_id", "band", "period_num"]).reset_index(drop=True)
sensor = d["sensor"].iloc[0]
latent_df, _ = estimate_sensor_latents(d, "site", sensor, VALIDATION_TRAIN_PERIODS)
problem, err = build_joint_problem(latent_df, sensor, VALIDATION_TREATED_PRE, VALIDATION_BLOCKS)
sel, w, mse, _p, _c = stepwise_donor_selection(problem)
for band in BANDS[sensor]:
    vals = [get_observed_value(d, "site", sensor, band, VALIDATION_BLOCKS[problem["original_donor_cols"][c]]["post_periods"][0]) for c in sel]
    print(band.replace("VV_minus_VH", "VV-VH"), float(np.asarray(vals, dtype=float) @ w))
