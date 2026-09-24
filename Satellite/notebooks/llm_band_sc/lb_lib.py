"""lb_lib — shared pieces for the LLM-as-weight-solver experiments on the collaborator's band-feature
designs (plan `quirky-growing-wombat`, approved 2026-09-09, third version).

Her code, copied: the notebook-14 scaler (pooled mean / SD with ddof=1 over the sample at the fit
periods, SD 0 or NaN -> 1), the notebook-14 SLSQP simplex solver (mean squared error objective,
bounds [0, 1], sum w = 1, ftol 1e-12, maxiter 5000, rows with any non-finite entry dropped), her
notebook-14 validation RMSE (sqrt of nanmean of squared standardized errors) and her notebook 10
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
HER10_SITES = LB / "lb_her_nb10_results.csv"
HER10_TABLES = LB / "lb_her_nb10_tables.csv"
HER10_FEATURE_ROWS = LB / "lb_her_nb10_feature_rows.csv"

MODELS = ["gpt-oss-120b", "GLM-5.3", "DeepSeek-V4.1-Flash-thinking-low"]   # Jun 2026-09-09: -thinking-low variants;
# 2026-09-23 (gate 0, `lb_gate0.py --list`): ARC removed Kimi-K3-thinking-low (108 stored replies of 15-16 Sep stand
# as records of that date) and renamed DeepSeek-V4-Flash, which never answered, to DeepSeek-V4.1-Flash; Jun added it.
REASONING_EFFORT = "low"                                    # sent with every request (ARC `reasoning_effort` field)
BASE = "https://llm-api.arc.vt.edu/api/v1"                  # ARC, the open-weight models above
HOKIE_BASE = "https://apim-n1ai-usc-e1f7a372a.azure-api.net/openai/v1"   # Hokie AI (Azure APIM), the GPT-5 family
OPENAI_BASE = "https://api.openai.com/v1"                                # OpenAI direct
KEY_ENV = {"arc": "ARC_LLM_API_KEY", "hokie": "HOKIE_AI_KEY", "openai": "OPENAI_API_KEY"}

# The Responses API is the ONLY endpoint that returns any reasoning text for the GPT-5 family, and what
# it returns is a SUMMARY written by a separate summarizer, never the raw chain: "we don't expose the raw
# reasoning tokens emitted by the model" (OpenAI reasoning guide, read 2026-09-19). Raw traces exist only
# for the ARC open-weight models, in message.reasoning. Anything stored from here is labelled
# reasoning_kind="summary" so the two are never pooled or compared as if they were the same object.
RESPONSES = {   # gateway -> (url, auth header, api-version values to try in order)
    "hokie":  (f"{HOKIE_BASE}/responses", lambda k: {"api-key": k},
               ["preview", "2025-04-01-preview", "2025-03-01-preview", None]),
    "openai": (f"{OPENAI_BASE}/responses", lambda k: {"Authorization": f"Bearer {k}"}, [None]),
}

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


def parse_model(spec):
    """A model is named `gateway:model`, e.g. `openai:gpt-5.5`, and the gateway is part of its identity.

    `gpt-5.5` is an alias, not a thing: Hokie AI resolves it to the snapshot gpt-5.5-2026-04-24 behind
    Azure content filtering, and api.openai.com resolves it independently. The same string on two
    gateways is two different serving paths whose replies must not be pooled, so the gateway travels
    with the name into the reply filename and into every table's model column.

    A bare name keeps its historical default (GPT-5 family -> hokie, everything else -> arc) so every
    command and reply file written before 2026-09-19 still means exactly what it meant then.
    Returns (gateway, model, spec-as-written)."""
    if ":" in spec:
        gw, model = spec.split(":", 1)
        if gw not in KEY_ENV:
            raise SystemExit(f"unknown gateway {gw!r} in {spec!r}; use one of {sorted(KEY_ENV)}")
        return gw, model, spec
    return ("hokie" if spec.lower().startswith("gpt-5") else "arc"), spec, spec


def provider(model):
    """Gateway for a bare model name. Kept for callers that have not moved to parse_model()."""
    return parse_model(model)[0]


def safe_name(spec):
    """Reply-filename form of a model spec: `openai:gpt-5.5` -> `openai_gpt-5.5`, so two gateways
    never append into one another's file."""
    return spec.replace(":", "_").replace("/", "_")


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


# The reported outcome metric. The collaborator's notebook 10 held-out metric takes the variance
# over P01-P09 -- the periods the fit is allowed to see. Scored that way our run of her SHC
# reproduces her 27 stored per-site values to 4.2e-07 and her sensor means 0.492461 / 0.786712.
# The P01-P10 variant put the withheld month into its own scaler and matched nothing of hers
# (max per-site difference 0.277); it is still computed, as `joint_nmse_p10var`, but never reported.
METRIC = "joint_nmse"
METRIC_TEXT = ("squared error at P10 over the variance of that band across P01-P09, averaged over "
               "bands (the collaborator's notebook 10 held-out metric)")


def her_nmse(actual, synthetic, train_values):
    """Notebook 10 held-out metric for one band: squared error / nanvar(train_values, ddof=1)."""
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


SHC_STEPS = (
    "Follow this procedure: (1) First replace each month's raw value with a smoothed version formed "
    "by blending it with its nearby months, so no single month's value is trusted on its own — use "
    "these smoothed values, not the raw ones, in the steps below. (2) From this smoothed series, form "
    "5 overlapping 4-month windows — P01-P04, P02-P05, P03-P06, P04-P07, and P05-P08 — each followed "
    "immediately by a known next month (P05, P06, P07, P08, and P09, respectively). (3) Search for the "
    "best small combination of these 5 windows: try each window alone against the smoothed P06-P09 "
    "values and keep whichever fits best as your first pick, regardless of how good that fit is on its "
    "own; then, in further rounds, try adding each not-yet-chosen window to what you've already picked, "
    "refitting weights (summing to 1, across all the bands together) each time, and add a window only "
    "if it meaningfully improves the fit over your best combination so far; stop once no further window "
    "helps. (4) Predict P10 as that same weighted combination — using the weights from your final, "
    "best-fitting round — of the actual, not smoothed, values that immediately followed each kept "
    "window."
)                                # Jun, 2026-09-15: validated cell-by-cell against her actual notebook
                                 # data/shc/10_shc_monthly_multiple_outcomes.ipynb (cells 4, 11, 13, 17, 19, 25)


def prompt_shc(sensor, target_rows, guide=True, anon=False, calendar=True, ask_rule=False, steps=False):
    """The SHC-design prompt. Defaults = the prompts run on 2026-09-09 (byte-identical, see lb_tests gate 6).
    Ablations (plan 2026-09-10): guide=False drops the band guide; anon=True drops the guide, the sensor and the
    band names (x1..xk); calendar=False drops the dates (P01..P10 only); ask_rule=True asks for a second line
    naming the rule used. steps=True (plan 2026-09-15) inserts SHC_STEPS, her paper's block/weight-fitting
    procedure in prose, before the final Predict line; data given to the model is unchanged."""
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
    if steps:
        lines.append(SHC_STEPS)
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
def api_key(prov="arc"):
    env = KEY_ENV[prov]
    k = os.environ.get(env)
    if prov == "arc" and not k:
        p = Path.home() / ".config" / "arc_llm_key"
        if p.exists():
            k = p.read_text().strip()
    if not k:
        raise SystemExit(f"no API key for {prov}: export {env} in your shell, then rerun")
    return k


def call(model, prompt, seed=0, max_tokens=8000, timeout=600, key=None, reasoning_effort=REASONING_EFFORT):
    """Reasoning models (gpt-oss, GLM, Kimi, DeepSeek thinking) spend tokens before the answer line: the cap
    is the ARC ceiling, 8,000 tokens per non-streaming request (Jun, 2026-09-09: set to ceiling).
    The ARC gateway returns the model's hidden reasoning in a separate `reasoning` field of the message
    (confirmed 2026-09-15 via lb_diag.py on Kimi-K3-thinking-low); captured into `usage["reasoning"]` so
    it is stored alongside every reply rather than discarded. Hokie AI does NOT: the GPT-5 family returns
    only `annotations` and `refusal` beside the content, so `usage["reasoning"]` is None there and the
    reasoning is visible only as `completion_tokens_details.reasoning_tokens`. Established 2026-09-17 by
    dumping every leaf of a live chat/completions payload: no reasoning text under any field name. The
    Responses API, which could return a reasoning SUMMARY, is a live route on the gateway but answered
    HTTP 400 with an empty body on the two shapes tried, and was not pursued further.

    Two gateways, two request shapes. Hokie AI rejects `temperature` and `max_tokens` outright with
    HTTP 400 for the GPT-5 family, so that arm has NO temperature knob at all and must be repeated
    rather than assumed reproducible; its token cap is `max_completion_tokens` and its auth is an
    `api-key` header (the APIM 401 text asks for a subscription key, but that header is refused)."""
    prov, model, _ = parse_model(model)
    key = key or api_key(prov)
    msgs = [{"role": "user", "content": prompt}]
    if prov == "arc":
        body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens, "seed": seed}
        url, auth = f"{BASE}/chat/completions", {"Authorization": f"Bearer {key}"}
    else:                       # hokie / openai: the GPT-5 family rejects temperature and max_tokens
        body = {"model": model, "messages": msgs, "seed": seed, "max_completion_tokens": max_tokens}
        base = HOKIE_BASE if prov == "hokie" else OPENAI_BASE
        auth = {"api-key": key} if prov == "hokie" else {"Authorization": f"Bearer {key}"}
        url = f"{base}/chat/completions"
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    r = requests.post(url, json=body, timeout=timeout,
                      headers={**auth, "Content-Type": "application/json"})
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    usage = dict(j.get("usage", {})); usage["finish_reason"] = ch.get("finish_reason")
    usage["reasoning"] = ch["message"].get("reasoning")
    usage["served_model"] = j.get("model")   # the snapshot that answered: `gpt-5.5` is an alias and two
    usage["gateway"] = prov                  # gateways can resolve it differently, so record both
    return ch["message"].get("content") or "", usage


def call_summary(model, prompt, seed=0, max_tokens=8000, timeout=600, key=None,
                 reasoning_effort=REASONING_EFFORT, summary="auto"):
    """One Responses-API call, returning (answer text, usage) with the reasoning SUMMARY attached.

    Same return shape as call(), so lb_run.py stores it unchanged and lb_arms.py scores it unchanged.
    usage["reasoning"] holds the summary text and usage["reasoning_kind"] is "summary" -- NOT the raw
    chain that ARC returns for gpt-oss / Kimi / GLM. An empty summary is stored as None, not as "".

    Azure gates this endpoint behind an api-version the chat surface does not need, and answered 400
    with an empty body when probed without one, so the versions in RESPONSES are tried in order and
    the last response's status and body are raised if none is accepted."""
    gateway, model, _ = parse_model(model)
    if gateway not in RESPONSES:
        raise SystemExit(f"{gateway} has no Responses API; ARC already returns the raw chain")
    url, auth, versions = RESPONSES[gateway]
    key = key or api_key(gateway)
    body = {"model": model, "input": prompt, "max_output_tokens": max_tokens,
            "reasoning": {k: v for k, v in (("effort", reasoning_effort), ("summary", summary)) if v}}
    last = None
    for ver in versions:
        r = requests.post(url, params={} if ver is None else {"api-version": ver},
                          json=body, timeout=timeout,
                          headers={**auth(key), "Content-Type": "application/json"})
        last = r
        if r.status_code == 200:
            break
    if last.status_code != 200:
        raise RuntimeError(f"responses API {last.status_code} on {gateway}: "
                           f"{(last.text or '<empty body>')[:400]}")
    j = last.json()
    items = j.get("output") or []
    think = " ".join(s.get("text", "") for it in items if it.get("type") == "reasoning"
                     for s in (it.get("summary") or []) if isinstance(s, dict)).strip()
    text = j.get("output_text") or ""
    if not text:
        text = " ".join(c.get("text", "") for it in items if it.get("type") == "message"
                        for c in (it.get("content") or []) if isinstance(c, dict)).strip()
    usage = dict(j.get("usage") or {})
    usage["finish_reason"] = j.get("status")
    usage["reasoning"] = think or None
    usage["reasoning_kind"] = "summary"          # never the raw chain; see RESPONSES above
    usage["endpoint"] = "responses"
    usage["served_model"] = j.get("model")
    usage["gateway"] = gateway
    # normalise the two token names so the scorers do not need to know which endpoint was used
    usage.setdefault("prompt_tokens", usage.get("input_tokens"))
    usage.setdefault("completion_tokens", usage.get("output_tokens"))
    return text, usage


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


ARC_BUFFERED_CAP = 8000     # gate 0, 2026-09-23: the ARC proxy refuses max_tokens > 8000 unless stream=true


def assemble_sse(lines):
    """OpenAI-style SSE chunks -> (message, finish_reason, usage, served_model, delta_keys, n_chunks).
    Every string field in every delta is concatenated, so reasoning text is reassembled whatever the
    delta calls it (reasoning, reasoning_content, ...). Usage arrives in the final chunk when the request
    sets stream_options.include_usage."""
    import json as _json
    msg, finish, usage, model, keys, n = {}, None, {}, None, set(), 0
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        j = _json.loads(data)
        n += 1
        model = model or j.get("model")
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices") or []:
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            for k, v in (ch.get("delta") or {}).items():
                keys.add(k)
                if isinstance(v, str):
                    msg[k] = msg.get(k, "") + v
                elif isinstance(v, list):
                    msg.setdefault(k, []).extend(v)
                elif v is not None:
                    msg.setdefault(k, v)
    return msg, finish, usage, model, sorted(keys), n


def call_turns(model, messages, seed=0, max_tokens=8000, timeout=600, key=None, reasoning_effort=REASONING_EFFORT,
               stream=None):
    """Multi-turn chat (2026-09-23 spec arms). `messages` is the whole conversation so far; one HTTP request.
    Same (text, usage) contract as call(), plus usage["raw_response"] and usage["reasoning_fields"].
    ARC: streams (SSE) when max_tokens exceeds ARC_BUFFERED_CAP or stream=True — the proxy's buffered mode is
    capped at 8000 tokens. In streaming, raw_response holds the fully reassembled message (every delta field,
    lossless for text), the set of delta keys, the usage chunk and the chunk count, rather than every one-token
    chunk verbatim. Non-streaming: raw_response is the entire response JSON."""
    prov, model, _ = parse_model(model)
    key = key or api_key(prov)
    if prov == "arc":
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "seed": seed}
        url, auth = f"{BASE}/chat/completions", {"Authorization": f"Bearer {key}"}
        use_stream = (max_tokens > ARC_BUFFERED_CAP) if stream is None else bool(stream)
    else:
        body = {"model": model, "messages": messages, "seed": seed, "max_completion_tokens": max_tokens}
        base = HOKIE_BASE if prov == "hokie" else OPENAI_BASE
        auth = {"api-key": key} if prov == "hokie" else {"Authorization": f"Bearer {key}"}
        url = f"{base}/chat/completions"
        use_stream = bool(stream)
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    headers = {**auth, "Content-Type": "application/json"}
    if use_stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        r = requests.post(url, json=body, timeout=timeout, stream=True, headers=headers)
        r.raise_for_status()
        msg, finish, usage_raw, served, keys, n = assemble_sse(r.iter_lines())
        usage = dict(usage_raw or {})
        usage["raw_response"] = {"stream": True, "assembled_message": msg, "delta_keys": keys,
                                 "usage_chunk": usage_raw, "n_chunks": n, "finish_reason": finish, "model": served}
    else:
        r = requests.post(url, json=body, timeout=timeout, headers=headers)
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        msg = ch.get("message") or {}
        finish, served = ch.get("finish_reason"), j.get("model")
        usage = dict(j.get("usage", {}))
        usage["raw_response"] = j
    usage["finish_reason"] = finish
    usage["reasoning"] = msg.get("reasoning") if msg.get("reasoning") is not None else msg.get("reasoning_content")
    usage["served_model"] = served
    usage["gateway"] = prov
    usage["streamed"] = use_stream
    usage["reasoning_fields"] = {k: (len(v) if isinstance(v, str) else bool(v)) for k, v in msg.items()
                                 if any(w in k.lower() for w in ("reason", "think", "thought"))}
    return msg.get("content") or "", usage
