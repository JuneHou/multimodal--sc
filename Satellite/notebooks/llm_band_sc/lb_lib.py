"""lb_lib — shared pieces for the LLM-as-weight-solver experiments on the collaborator's band-feature
designs (plan `quirky-growing-wombat`, approved 2026-09-09, third version).

Her code, copied: the notebook-14 scaler (pooled mean / SD with ddof=1 over the sample at the fit
periods, SD 0 or NaN -> 1), the notebook-14 SLSQP simplex solver (mean squared error objective,
bounds [0, 1], sum w = 1, ftol 1e-12, maxiter 5000, rows with any non-finite entry dropped), her
notebook-14 validation RMSE (sqrt of nanmean of squared standardized errors) and her notebook-26
validation metric (squared error / variance with ddof=1 of the site's own P01-P09 values, per band;
joint = plain mean over bands).

Wording: all of P01-P10 are pre-hurricane for every site; nothing is treated in validation. Prompts
say TARGET parcel and DONOR parcels only.
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
TS = HERE.parent / "ts_SCM_ASCM"
sys.path.insert(0, str(TS))
import panel_lib as pl            # noqa: E402
import panel_monthly as pm        # noqa: E402

ROOT = pl.ROOT
HER14 = ROOT / "test" / "scm_validation"                      # her shipped notebook-14 outputs (solver gate only)
NB26 = ROOT / "Satellite" / "data" / "shc" / "10_shc_monthly_multiple_outcomes.ipynb"   # her monthly multiple-outcome SHC (Jun 2026-09-09: reference = data/shc)
LB = HERE
FEATURES_CSV = LB / "lb_features_monthly.csv"
HER26_SITES = LB / "lb_her_nb26_results.csv"
HER26_TABLES = LB / "lb_her_nb26_tables.csv"
HER26_FEATURE_ROWS = LB / "lb_her_nb26_feature_rows.csv"

MODELS = ["gpt-oss-120b", "GLM-5.3", "Kimi-K3-thinking-low", "DeepSeek-V4-Flash-thinking-low"]   # Jun 2026-09-09: -thinking-low variants
REASONING_EFFORT = "low"                                    # sent with every request (ARC `reasoning_effort` field)
BASE = "https://llm-api.arc.vt.edu/api/v1"
SENSORS = ("sentinel1", "sentinel2")
SHORT = {"sentinel1": "s1", "sentinel2": "s2"}
BANDS = pl.BANDS                                   # sentinel1: VV, VH, VV_minus_VH ; sentinel2: B2 ... NDWI
FEAT = {s: [f"mean_{b}" for b in BANDS[s]] for s in SENSORS}
PRE = list(range(1, 10))                           # P01-P09 = the inputs
TARGET = 10                                        # P10 = the withheld month
N_DONORS = 5
SD_TOL = 1e-8
DECIMALS = {"sentinel2": 3, "sentinel1": 2}
_cal = pd.read_csv(pm.MONTHLY / "nb25_monthly_long_period_definitions.csv")
CALENDAR = {int(p[1:]): m for p, m in zip(_cal["period_id"], _cal["month_label"])}


def disp(band):
    return band.replace("_minus_", "-")


BAND_GUIDE = {
    "sentinel2": ("Band guide (Sentinel-2 optical, mean over a land parcel about 1 km across, reflectance on a "
                  "0-1 scale): B2 = blue light, B3 = green light, B4 = red light (together the visible colour of "
                  "the ground); B8 = near-infrared (high for healthy vegetation); B11, B12 = shortwave infrared "
                  "(sensitive to moisture and bare soil); NDVI = vegetation greenness index; NDWI = surface water "
                  "/ moisture index (both in [-1, 1])."),
    "sentinel1": ("Band guide (Sentinel-1 radar, mean over a land parcel about 1 km across, backscatter in "
                  "decibels, more negative = weaker return): VV = co-polarised backscatter (high for rough or "
                  "built surfaces, low for smooth water); VH = cross-polarised backscatter (sensitive to "
                  "vegetation volume and structure); VV-VH = their difference (a vegetation / structure ratio)."),
}
FORMAT_LINE = {
    "sentinel2": "B2 0.000, B3 0.000, B4 0.000, B8 0.000, B11 0.000, B12 0.000, NDVI 0.000, NDWI 0.000",
    "sentinel1": "VV 0.00, VH 0.00, VV-VH 0.00",
}


# ----------------------------------------------------------------------------- her code
def her_scaler(df, feature_cols, periods, period_col="seq"):
    """Notebook 14 `get_training_scaler`: mean and SD (pandas, ddof=1) over the rows of the sample at
    the fit periods; SD 0 -> 1, NaN -> 1."""
    training = df.loc[df[period_col].isin(periods)]
    means = training[feature_cols].mean()
    stds = training[feature_cols].std().replace(0, 1).fillna(1)
    return means, stds


def her_solve(y, X):
    """Notebook 14 solver. y (n,), X (n, J) standardized; rows with any non-finite entry dropped."""
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y_, X_ = y[valid], X[valid]
    J = X.shape[1]
    if valid.sum() == 0:
        return np.full(J, np.nan)
    obj = lambda w: float(np.mean((y_ - X_ @ w) ** 2))
    res = minimize(obj, np.repeat(1 / J, J), method="SLSQP", bounds=[(0.0, 1.0)] * J,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"maxiter": 5000, "ftol": 1e-12})
    return res.x


def her_rmse(errors):
    """Notebook 14: sqrt of nanmean of squared (standardized) errors."""
    return float(np.sqrt(np.nanmean(np.asarray(errors, float) ** 2)))


def her_nmse(actual, synthetic, train_values):
    """Notebook 26 held-out metric for one band: squared error / nanvar(train_values, ddof=1)."""
    sq = float((actual - synthetic) ** 2)
    var = float(np.nanvar(np.asarray(train_values, float), ddof=1))
    return sq / var if (np.isfinite(sq) and np.isfinite(var) and var > SD_TOL) else np.nan


# ----------------------------------------------------------------------------- prompts
def fmt_vals(sensor, row):
    d = DECIMALS[sensor]
    parts = []
    for b in BANDS[sensor]:
        v = row.get(f"mean_{b}", np.nan)
        parts.append(f"{disp(b)} {v:.{d}f}" if np.isfinite(v) else f"{disp(b)} n/a")
    return ", ".join(parts)


def anon_labels(sensor):
    """Ablation 'anon': band k -> 'x{k}' (no band names, no sensor name)."""
    return [f"x{k}" for k in range(1, len(BANDS[sensor]) + 1)]


def fmt_vals_labels(sensor, row, labels):
    d = DECIMALS[sensor]
    parts = []
    for b, lab in zip(BANDS[sensor], labels):
        v = row.get(f"mean_{b}", np.nan)
        parts.append(f"{lab} {v:.{d}f}" if np.isfinite(v) else f"{lab} n/a")
    return ", ".join(parts)


def series_lines(sensor, rows, periods, prefix="", calendar=True, labels=None):
    """One line per period: 'P01 = 2023-11: VV -10.06, ...' (or 'missing' when the composite is absent).
    calendar=False drops the dates ('P01: ...'); labels replaces the band names (ablation 'anon')."""
    out = []
    for q in periods:
        r = rows.get(q)
        has = r is not None and any(np.isfinite(r.get(f"mean_{b}", np.nan)) for b in BANDS[sensor])
        body = (fmt_vals_labels(sensor, r, labels) if labels else fmt_vals(sensor, r)) if has else "missing"
        head = f"{prefix}P{q:02d} = {CALENDAR[q]}: " if calendar else f"{prefix}P{q:02d}: "
        out.append(head + body)
    return out


def format_line(sensor, labels=None):
    if labels is None:
        return FORMAT_LINE[sensor]
    d = DECIMALS[sensor]
    return ", ".join(f"{lab} {0:.{d}f}" for lab in labels)


def prompt_shc(sensor, target_rows, guide=True, anon=False, calendar=True, ask_rule=False):
    """The SHC-design prompt. Defaults = the prompts run on 2026-09-09 (byte-identical, see lb_tests gate 6).
    Ablations (plan 2026-09-10): guide=False drops the band guide; anon=True drops the guide, the sensor and the
    band names (x1..xk); calendar=False drops the dates (P01..P10 only); ask_rule=True asks for a second line
    naming the rule used."""
    labels = anon_labels(sensor) if anon else None
    lines = []
    if guide and not anon:
        lines.append(BAND_GUIDE[sensor])
    what = f"{len(labels)} measured quantities of one unit" if anon else "mean band values of one land parcel"
    months = "consecutive calendar months" if calendar else "consecutive months"
    lines.append(f"Below are {what} for {len(PRE)} {months}.")
    lines += series_lines(sensor, target_rows, PRE, calendar=calendar, labels=labels)
    nxt = f"P{TARGET:02d} = {CALENDAR[TARGET]}" if calendar else f"P{TARGET:02d}"
    subj = "the unit's quantities" if anon else "the parcel's mean band values"
    if ask_rule:
        tail = (f"Reply with exactly two lines and nothing else: first the answer line in this format, then a "
                f"second line naming in at most 15 words the rule you used:\n{format_line(sensor, labels)}")
    else:
        tail = f"Reply with exactly one line in this format and nothing else:\n{format_line(sensor, labels)}"
    lines.append(f"Predict {subj} for the next month, {nxt}. " + tail)
    return "\n".join(lines)


def prompt_scm(sensor, target_rows, donor_rows):
    """donor_rows: list of dicts (rank order) mapping period -> row."""
    lines = [BAND_GUIDE[sensor],
             f"Below are mean band values for {1 + len(donor_rows)} land parcels over consecutive calendar months.",
             "Month calendar: " + ", ".join(f"P{q:02d} = {CALENDAR[q]}" for q in range(1, TARGET + 1)) + ".",
             "TARGET parcel:"]
    lines += [l.split(" = ")[0] + ":" + l.split(":", 1)[1] for l in series_lines(sensor, target_rows, PRE)]
    for k, dr in enumerate(donor_rows, 1):
        lines.append(f"DONOR {k}" + (" (most similar parcel in land cover, elevation and slope):" if k == 1 else ":"))
        lines += [l.split(" = ")[0] + ":" + l.split(":", 1)[1] for l in series_lines(sensor, dr, list(range(1, TARGET + 1)))]
    lines.append('A period with no observation is written as "missing".')
    lines.append(f"Using the donors' values at P{TARGET:02d} and the relation between the target and the donors at "
                 f"P01-P{PRE[-1]:02d}, predict the TARGET parcel's mean band values at P{TARGET:02d}. Reply with "
                 f"exactly one line in this format and nothing else:\n{FORMAT_LINE[sensor]}")
    return "\n".join(lines)


FORBIDDEN = ("hurricane", "landslide", "treat")


# ----------------------------------------------------------------------------- ARC client
def api_key():
    k = os.environ.get("ARC_LLM_API_KEY")
    p = Path.home() / ".config" / "arc_llm_key"
    if not k and p.exists():
        k = p.read_text().strip()
    if not k:
        raise SystemExit("no API key: export ARC_LLM_API_KEY or write it to ~/.config/arc_llm_key")
    return k


def call(model, prompt, seed=0, max_tokens=8000, timeout=600, key=None, reasoning_effort=REASONING_EFFORT):
    """Reasoning models (gpt-oss, GLM, Kimi, DeepSeek thinking) spend tokens before the answer line: the cap
    is the ARC ceiling, 8,000 tokens per non-streaming request (Jun, 2026-09-09: set to ceiling)."""
    key = key or api_key()
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "seed": seed}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    r = requests.post(f"{BASE}/chat/completions", json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    usage = dict(j.get("usage", {})); usage["finish_reason"] = ch.get("finish_reason")
    return ch["message"].get("content") or "", usage


def _parse_line(names, text):
    out = {}
    for n in names:
        m = re.search(rf"(?<![A-Za-z0-9-]){re.escape(n)}\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text or "")
        if m:
            out[n] = float(m.group(1))
    return out if len(out) == len(names) else None


def parse(sensor, text, labels=None):
    """'VV -9.9, VH -15.8, VV-VH 5.9' -> {display name: value}; None unless every band is found.
    labels: the names used in the prompt when they differ from the band names (ablation 'anon'); the result is
    always keyed by the real display names. A multi-line reply is parsed line by line and the first line that
    carries every band wins (ablation 'rule' adds a second line of text), then the whole text."""
    names = [disp(b) for b in BANDS[sensor]]
    used = labels or names
    for chunk in [l for l in (text or "").splitlines() if l.strip()] + [text or ""]:
        got = _parse_line(used, chunk)
        if got:
            return {n: got[u] for n, u in zip(names, used)}
    return None


def rule_text(text):
    """Ablation 'rule': everything after the first non-empty line, stripped."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return " ".join(lines[1:]) if len(lines) > 1 else ""
