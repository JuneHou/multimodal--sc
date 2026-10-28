#!/usr/bin/env python3
"""Forecast the next parcel-level satellite band values with a synthetic historical control workflow.

Example:
  python3 run_shc_satellite_forecast.py --input data.csv --out answer.json --details-out diagnostics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


COLUMN_ALIASES = {
    "band": ("band", "band_name", "channel", "feature", "variable"),
    "time": ("time", "timestamp", "date", "step", "t"),
    "value": ("value", "band_value", "measurement", "reflectance", "signal"),
}


@dataclass
class HistoryMatrix:
    bands: list[str]
    time_labels: list[str]
    matrix: list[list[float]]


@dataclass
class MatchResult:
    pre_length: int
    donor_count: int
    donor_end_positions: list[int]
    donor_weights: list[float]
    pre_rmse: float
    normalized_pre_rmse: float
    predicted_post: float


@dataclass
class ForecastResult:
    band: str
    prediction: float
    method: str
    transform: str
    pre_length: int
    donor_count: int
    donor_end_positions: list[int]
    donor_weights: list[float]
    pre_rmse: float
    normalized_pre_rmse: float
    rationale: str


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def infer_columns(fieldnames: list[str]) -> dict[str, str]:
    normalized = {name.strip().lower(): name for name in fieldnames}
    resolved: dict[str, str] = {}
    for role, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                resolved[role] = normalized[alias]
                break
        if role not in resolved:
            alias_text = ", ".join(aliases)
            raise ValueError(f"Could not find a {role} column. Expected one of: {alias_text}")
    return resolved


def load_long_history(path: Path) -> HistoryMatrix:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header row")
        columns = infer_columns(reader.fieldnames)
        band_order: list[str] = []
        time_order: list[str] = []
        band_seen: set[str] = set()
        time_seen: set[str] = set()
        values: dict[tuple[str, str], float] = {}
        row_count = 0
        for row in reader:
            row_count += 1
            band = (row.get(columns["band"]) or "").strip()
            time_label = (row.get(columns["time"]) or "").strip()
            value_text = (row.get(columns["value"]) or "").strip()
            if not band or not time_label or not value_text:
                raise ValueError(
                    f"Row {row_count + 1} is missing one of band/time/value after column inference"
                )
            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_count + 1} has a non-numeric value {value_text!r}"
                ) from exc
            key = (band, time_label)
            if key in values:
                raise ValueError(f"Duplicate observation for band={band!r}, time={time_label!r}")
            values[key] = value
            if band not in band_seen:
                band_seen.add(band)
                band_order.append(band)
            if time_label not in time_seen:
                time_seen.add(time_label)
                time_order.append(time_label)
    if not values:
        raise ValueError(f"{path} contains no data rows")
    matrix: list[list[float]] = []
    for band in band_order:
        row_values: list[float] = []
        missing_times = [time_label for time_label in time_order if (band, time_label) not in values]
        if missing_times:
            raise ValueError(
                f"Band {band!r} is missing {len(missing_times)} time points; first missing value is {missing_times[0]!r}"
            )
        for time_label in time_order:
            row_values.append(values[(band, time_label)])
        matrix.append(row_values)
    if len(time_order) < 6:
        raise ValueError(
            f"{path} needs at least 6 ordered time points for one-step forecasting; found {len(time_order)}"
        )
    return HistoryMatrix(bands=band_order, time_labels=time_order, matrix=matrix)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda idx: abs(augmented[idx][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("Linear system is singular or ill-conditioned")
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot_value = augmented[pivot_index][pivot_index]
        for column in range(pivot_index, size + 1):
            augmented[pivot_index][column] /= pivot_value
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            for column in range(pivot_index, size + 1):
                augmented[row_index][column] -= factor * augmented[pivot_index][column]
    return [augmented[row_index][size] for row_index in range(size)]


def fit_ridge_ols(design: list[list[float]], target: list[float], ridge: float = 1e-6) -> list[float]:
    n_features = len(design[0])
    xtx = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    xty = [0.0 for _ in range(n_features)]
    for row, response in zip(design, target, strict=True):
        for i in range(n_features):
            xty[i] += row[i] * response
            for j in range(n_features):
                xtx[i][j] += row[i] * row[j]
    for index in range(n_features):
        xtx[index][index] += ridge
    return solve_linear_system(xtx, xty)


def local_linear_predict(
    series: list[float],
    target_index: int,
    bandwidth: float,
    *,
    leave_out_target: bool,
) -> float:
    s0 = s1 = s2 = t0 = t1 = 0.0
    for index, value in enumerate(series):
        if leave_out_target and index == target_index:
            continue
        offset = (index - target_index) / bandwidth
        weight = math.exp(-0.5 * offset * offset)
        x_value = float(index - target_index)
        s0 += weight
        s1 += weight * x_value
        s2 += weight * x_value * x_value
        t0 += weight * value
        t1 += weight * x_value * value
    determinant = s0 * s2 - s1 * s1
    if abs(determinant) < 1e-12:
        return mean(series)
    intercept = (t0 * s2 - t1 * s1) / determinant
    return intercept


def choose_bandwidth(series: list[float]) -> float:
    n_obs = len(series)
    if n_obs < 5:
        return max(1.0, n_obs / 2.0)
    candidate_bandwidths = sorted(
        {
            max(1.0, round(multiplier * n_obs, 2))
            for multiplier in (0.18, 0.25, 0.33, 0.5, 0.67)
        }
    )
    best_bandwidth = candidate_bandwidths[0]
    best_score = math.inf
    for bandwidth in candidate_bandwidths:
        squared_errors = []
        for index, value in enumerate(series):
            fitted = local_linear_predict(series, index, bandwidth, leave_out_target=True)
            squared_errors.append((value - fitted) ** 2)
        score = mean(squared_errors)
        if score < best_score:
            best_score = score
            best_bandwidth = bandwidth
    return best_bandwidth


def smooth_series(series: list[float]) -> tuple[list[float], float]:
    bandwidth = choose_bandwidth(series)
    smoothed = [
        local_linear_predict(series, index, bandwidth, leave_out_target=False)
        for index in range(len(series))
    ]
    return smoothed, bandwidth


def project_to_simplex(weights: list[float]) -> list[float]:
    if not weights:
        return []
    sorted_weights = sorted(weights, reverse=True)
    partial_sum = 0.0
    rho = 0
    theta = 0.0
    for index, value in enumerate(sorted_weights, start=1):
        partial_sum += value
        candidate = (partial_sum - 1.0) / index
        if value - candidate > 0.0:
            rho = index
            theta = candidate
    if rho == 0:
        return [1.0 / len(weights) for _ in weights]
    theta = (sum(sorted_weights[:rho]) - 1.0) / rho
    projected = [max(value - theta, 0.0) for value in weights]
    total = sum(projected)
    if total == 0.0:
        return [1.0 / len(weights) for _ in weights]
    return [value / total for value in projected]


def optimize_simplex_weights(
    target_vector: list[float],
    donor_matrix: list[list[float]],
    *,
    ridge: float = 1e-6,
    max_iterations: int = 400,
    tolerance: float = 1e-9,
) -> list[float]:
    donor_count = len(donor_matrix)
    if donor_count == 1:
        return [1.0]
    gram = [[0.0 for _ in range(donor_count)] for _ in range(donor_count)]
    rhs = [0.0 for _ in range(donor_count)]
    for i in range(donor_count):
        rhs[i] = dot(donor_matrix[i], target_vector)
        for j in range(donor_count):
            gram[i][j] = dot(donor_matrix[i], donor_matrix[j])
    lipschitz = max(
        sum(abs(value) for value in row) + ridge
        for row in gram
    )
    step_size = 1.0 / max(2.0 * lipschitz, 1e-6)
    weights = [1.0 / donor_count for _ in range(donor_count)]
    for _ in range(max_iterations):
        gradient = []
        for i in range(donor_count):
            fitted_component = sum(gram[i][j] * weights[j] for j in range(donor_count))
            gradient.append(2.0 * (fitted_component - rhs[i] + ridge * weights[i]))
        candidate = [weight - step_size * grad for weight, grad in zip(weights, gradient, strict=True)]
        updated = project_to_simplex(candidate)
        max_delta = max(abs(new - old) for new, old in zip(updated, weights, strict=True))
        weights = updated
        if max_delta < tolerance:
            break
    return weights


def synthetic_vector(weights: list[float], donor_matrix: list[list[float]]) -> list[float]:
    length = len(donor_matrix[0])
    result = [0.0 for _ in range(length)]
    for weight, donor in zip(weights, donor_matrix, strict=True):
        for index in range(length):
            result[index] += weight * donor[index]
    return result


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(mean([(a - b) ** 2 for a, b in zip(actual, predicted, strict=True)]))


def build_match_result(
    latent_series: list[float],
    raw_residuals: list[float],
    pre_length: int,
    *,
    max_donors: int,
    max_iterations: int,
) -> MatchResult:
    treated_pre = latent_series[-pre_length:]
    donors: list[dict[str, object]] = []
    for end_index in range(pre_length - 1, len(latent_series) - 1):
        donors.append(
            {
                "end_index": end_index,
                "pre": latent_series[end_index - pre_length + 1 : end_index + 1],
                "post": raw_residuals[end_index + 1],
            }
        )
    if len(donors) < 2:
        raise ValueError("Need at least two historical blocks to run synthetic historical control")
    first_choice = min(
        range(len(donors)),
        key=lambda idx: root_mean_squared_error(
            treated_pre,
            donors[idx]["pre"],  # type: ignore[arg-type]
        ),
    )
    selected = [first_choice]
    donor_matrix = [donors[first_choice]["pre"]]  # type: ignore[list-item]
    weights = optimize_simplex_weights(treated_pre, donor_matrix, max_iterations=max_iterations)
    synthetic_pre = synthetic_vector(weights, donor_matrix)
    current_rmse = root_mean_squared_error(treated_pre, synthetic_pre)
    scale = max(standard_deviation(treated_pre), 1e-6)
    improvement_floor = max(1e-6, 0.02 * scale)
    while len(selected) < min(max_donors, len(donors)):
        best_candidate: int | None = None
        best_weights: list[float] | None = None
        best_rmse = current_rmse
        for candidate in range(len(donors)):
            if candidate in selected:
                continue
            trial_indices = selected + [candidate]
            trial_matrix = [donors[index]["pre"] for index in trial_indices]  # type: ignore[list-item]
            trial_weights = optimize_simplex_weights(
                treated_pre,
                trial_matrix,
                max_iterations=max_iterations,
            )
            trial_pre = synthetic_vector(trial_weights, trial_matrix)
            trial_rmse = root_mean_squared_error(treated_pre, trial_pre)
            if trial_rmse < best_rmse - 1e-12:
                best_candidate = candidate
                best_weights = trial_weights
                best_rmse = trial_rmse
        if best_candidate is None or current_rmse - best_rmse <= improvement_floor:
            break
        selected.append(best_candidate)
        donor_matrix = [donors[index]["pre"] for index in selected]  # type: ignore[list-item]
        weights = best_weights if best_weights is not None else optimize_simplex_weights(
            treated_pre,
            donor_matrix,
            max_iterations=max_iterations,
        )
        synthetic_pre = synthetic_vector(weights, donor_matrix)
        current_rmse = root_mean_squared_error(treated_pre, synthetic_pre)
    donor_posts = [float(donors[index]["post"]) for index in selected]
    predicted_post = sum(weight * post for weight, post in zip(weights, donor_posts, strict=True))
    return MatchResult(
        pre_length=pre_length,
        donor_count=len(selected),
        donor_end_positions=[int(donors[index]["end_index"]) + 1 for index in selected],
        donor_weights=[round(weight, 6) for weight in weights],
        pre_rmse=current_rmse,
        normalized_pre_rmse=current_rmse / max(scale, 1e-6),
        predicted_post=predicted_post,
    )


def choose_pre_length(
    latent_series: list[float],
    raw_residuals: list[float],
    *,
    min_pre_length: int,
    max_pre_length: int,
    fit_threshold: float,
    max_donors: int,
    max_iterations: int,
) -> MatchResult:
    results: list[MatchResult] = []
    upper_bound = min(max_pre_length, len(latent_series) - 2)
    for pre_length in range(min_pre_length, upper_bound + 1):
        results.append(
            build_match_result(
                latent_series,
                raw_residuals,
                pre_length,
                max_donors=max_donors,
                max_iterations=max_iterations,
            )
        )
    if not results:
        raise ValueError("History is too short for the requested pre-length search range")
    eligible = [result for result in results if result.normalized_pre_rmse <= fit_threshold]
    if eligible:
        return max(eligible, key=lambda result: (result.pre_length, -result.normalized_pre_rmse))
    return min(results, key=lambda result: (result.normalized_pre_rmse, -result.pre_length))


def build_level_design(
    matrix: list[list[float]],
    target_band_index: int,
) -> tuple[list[list[float]], list[float], list[float]]:
    time_count = len(matrix[0])
    design = []
    target = []
    for time_index in range(1, time_count):
        row = [1.0]
        for band_values in matrix:
            row.append(band_values[time_index - 1])
        design.append(row)
        target.append(matrix[target_band_index][time_index])
    future_row = [1.0]
    for band_values in matrix:
        future_row.append(band_values[-1])
    return design, target, future_row


def difference_matrix(matrix: list[list[float]]) -> list[list[float]]:
    differenced = []
    for band_values in matrix:
        differenced.append(
            [band_values[index + 1] - band_values[index] for index in range(len(band_values) - 1)]
        )
    return differenced


def evaluate_mode(
    history: HistoryMatrix,
    target_band_index: int,
    *,
    transform: str,
    min_pre_length: int,
    max_pre_length: int,
    fit_threshold: float,
    max_donors: int,
    max_iterations: int,
) -> ForecastResult:
    source_matrix = history.matrix if transform == "level" else difference_matrix(history.matrix)
    if len(source_matrix[0]) < max(min_pre_length + 3, 5):
        raise ValueError(f"{transform} mode does not have enough usable history")
    design, target_series, future_row = build_level_design(source_matrix, target_band_index)
    coefficients = fit_ridge_ols(design, target_series)
    ols_prediction = dot(future_row, coefficients)
    residuals = [
        response - dot(row, coefficients)
        for row, response in zip(design, target_series, strict=True)
    ]
    if len(residuals) < min_pre_length + 2:
        raise ValueError("Residual history is too short after residualization")
    latent, _ = smooth_series(residuals)
    match = choose_pre_length(
        latent,
        residuals,
        min_pre_length=min_pre_length,
        max_pre_length=max_pre_length,
        fit_threshold=fit_threshold,
        max_donors=max_donors,
        max_iterations=max_iterations,
    )
    prediction = ols_prediction + match.predicted_post
    method = "synthetic_historical_control"
    if match.normalized_pre_rmse > fit_threshold * 1.75:
        prediction = ols_prediction
        method = "ridge_ols_fallback"
    if transform == "difference":
        prediction += history.matrix[target_band_index][-1]
    rationale = (
        f"{history.bands[target_band_index]} used {transform} mode with {method}, "
        f"pre_length={match.pre_length}, donor_count={match.donor_count}, "
        f"normalized_pre_rmse={match.normalized_pre_rmse:.3f}"
    )
    return ForecastResult(
        band=history.bands[target_band_index],
        prediction=prediction,
        method=method,
        transform=transform,
        pre_length=match.pre_length,
        donor_count=match.donor_count,
        donor_end_positions=match.donor_end_positions,
        donor_weights=match.donor_weights,
        pre_rmse=match.pre_rmse,
        normalized_pre_rmse=match.normalized_pre_rmse,
        rationale=rationale,
    )


def choose_band_forecast(
    history: HistoryMatrix,
    target_band_index: int,
    *,
    mode: str,
    min_pre_length: int,
    max_pre_length: int,
    fit_threshold: float,
    max_donors: int,
    max_iterations: int,
) -> ForecastResult:
    candidates: list[ForecastResult] = []
    transforms = ["level", "difference"] if mode == "auto" else [mode]
    errors: list[str] = []
    for transform in transforms:
        try:
            candidates.append(
                evaluate_mode(
                    history,
                    target_band_index,
                    transform=transform,
                    min_pre_length=min_pre_length,
                    max_pre_length=max_pre_length,
                    fit_threshold=fit_threshold,
                    max_donors=max_donors,
                    max_iterations=max_iterations,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised through failure handling
            errors.append(f"{transform}: {exc}")
    if not candidates:
        raise ValueError("; ".join(errors))
    candidates.sort(
        key=lambda result: (
            0 if result.method == "synthetic_historical_control" else 1,
            result.normalized_pre_rmse,
            0 if result.transform == "level" else 1,
        )
    )
    chosen = candidates[0]
    chosen.rationale = (
        f"{chosen.band}: {chosen.method} on {chosen.transform} history with pre_length={chosen.pre_length}, "
        f"{chosen.donor_count} donor blocks ending at observations {chosen.donor_end_positions}, "
        f"weights={chosen.donor_weights}, pre_rmse={chosen.pre_rmse:.4f}, "
        f"normalized_pre_rmse={chosen.normalized_pre_rmse:.3f}"
    )
    return chosen


def build_rationale(history: HistoryMatrix, results: list[ForecastResult]) -> str:
    intro = (
        "Forecasted one step ahead for each band with a synthetic historical control workflow: "
        "each target band was residualized on lagged multiband controls, the latent residual path was smoothed "
        "with local linear regression, and the last pre-forecast block was matched to earlier historical blocks from "
        "the same parcel. Difference mode was preferred automatically when it produced a tighter pre-match."
    )
    details = " ".join(result.rationale + "." for result in results)
    closing = (
        f"Input covered {len(history.bands)} bands across {len(history.time_labels)} ordered time points; "
        "review the normalized pre-match errors before trusting unusually sharp band jumps."
    )
    return f"{intro} {details} {closing}"


def build_answer(history: HistoryMatrix, results: list[ForecastResult]) -> dict[str, object]:
    predictions = {
        result.band: round(result.prediction, 6)
        for result in results
    }
    return {
        "prediction": predictions,
        "rationale": build_rationale(history, results),
    }


def build_details(history: HistoryMatrix, results: list[ForecastResult]) -> dict[str, object]:
    return {
        "band_count": len(history.bands),
        "time_count": len(history.time_labels),
        "bands": history.bands,
        "time_labels": history.time_labels,
        "per_band": [
            {
                "band": result.band,
                "prediction": round(result.prediction, 6),
                "method": result.method,
                "transform": result.transform,
                "pre_length": result.pre_length,
                "donor_count": result.donor_count,
                "donor_end_positions": result.donor_end_positions,
                "donor_weights": result.donor_weights,
                "pre_rmse": round(result.pre_rmse, 6),
                "normalized_pre_rmse": round(result.normalized_pre_rmse, 6),
            }
            for result in results
        ],
    }


def write_json(payload: dict[str, object], path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path is None:
        print(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Long-format CSV with one row per band and time point.")
    parser.add_argument("--out", type=Path, default=None, help="Write the required answer JSON to this path.")
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional diagnostics JSON with per-band pre-fit metrics and donor selections.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "level", "difference"),
        default="auto",
        help="Use level matching, differenced matching, or automatically choose the tighter pre-fit path.",
    )
    parser.add_argument(
        "--min-pre-length",
        type=int,
        default=4,
        help="Minimum treated pre-window length m to consider during donor matching.",
    )
    parser.add_argument(
        "--max-pre-length",
        type=int,
        default=10,
        help="Maximum treated pre-window length m to consider during donor matching.",
    )
    parser.add_argument(
        "--fit-threshold",
        type=float,
        default=0.35,
        help="Normalized pre-match RMSE threshold used to prefer longer pre-windows.",
    )
    parser.add_argument(
        "--max-donors",
        type=int,
        default=4,
        help="Maximum number of historical donor blocks retained by the stepwise-matching procedure.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=400,
        help="Maximum projected-gradient iterations for simplex-constrained donor weights.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        history = load_long_history(args.input)
        results = [
            choose_band_forecast(
                history,
                band_index,
                mode=args.mode,
                min_pre_length=args.min_pre_length,
                max_pre_length=args.max_pre_length,
                fit_threshold=args.fit_threshold,
                max_donors=args.max_donors,
                max_iterations=args.max_iterations,
            )
            for band_index in range(len(history.bands))
        ]
        answer = build_answer(history, results)
        write_json(answer, args.out)
        if args.details_out is not None:
            args.details_out.parent.mkdir(parents=True, exist_ok=True)
            args.details_out.write_text(
                json.dumps(build_details(history, results), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return 0
    except Exception as exc:
        print(f"synthetic historical control forecast failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
