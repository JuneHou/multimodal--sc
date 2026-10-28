"""lb_skill_score_daily — the collaborator's notebook 14 evaluation (her cells 10 to 15) for every row of the daily
simple skills: her SHC, the two rule rows and each skill arm and model.

Plan ~/.claude/plans/simple-skills-daily.md, step 12 (Decisions "score", "stores of a skill arm case", "cell 13
residuals of a skill arm", "cases in a comparison", "figures"). Jun "so can we do same evaluation as collaborator for
daily? i need result comparable" (2026-10-15); Jun "a. Each skill arm uses its own, from the fit its answer came from,
matched at the answer's" (2026-10-15).

Per case, her statements of lb_her_nb14 run at the spec of the fit the answer came from (the matched fit) on the
case's visits before Helene and its observed visits after, which gives her series_store, model_parameters,
stack_case_store, final_fit_store and daily_effects. Her cell 10 rows then take the answer in place of her
y_star_synthetic, with her cell 10 formulas after it. The stores of a row's cases are merged into one namespace, and
her cells 11 (validation), 12 (effect tables), 13 (inference), 14 (Tables 1 to 3) and 15 (figures) run there, each
top-level statement in her order; a statement that raises (a row with no fit has no final_fit_store) is logged and the
next one runs. Her printed output is parsed by lb_her14_run.parse_printed into the tables of her printed tables files.

    matched fit   the last fitter run in the record whose every prediction, rounded with round() to the number of
                  decimals of the answer value at the same visit and band, equals that answer value
    her engine    a matched fit with every Class-2 field at her code's behaviour: her cells 1 to 8 and 10 at its spec
    other engine  her cells 1 to 5 at its spec (series_store, model_parameters); no final_fit_store entry
    no match      the case enters cells 10 and 12 only, as the rule rows do

Rows: "her" (her printed tables in the printed files; our run of her code in the effects, inference and figures),
"rule:persistence", "rule:last3_mean", "<experiment>|<model>" from the last record per prompt_id of
lb_replies_daily_*.jsonl, and "her|<row>", her code over the cases of a row that lacks some of her 52 cases.

Writes lb_match_skill_daily.csv, lb_effects_skill_daily.csv, lb_inference_skill_daily.csv,
lb_printed_skill_daily_<table>.csv, lb_score_log_skill_daily.csv and figures_daily/<row>/.

    python lb_skill_score_daily.py            score every record present
    python lb_skill_score_daily.py --her      her row only (gate DK11 runs this through lb_skill_tests_daily.py)
"""
import lb_one_thread  # noqa: F401  first import: one thread, as the skill sandbox runs
import ast
import contextlib
import glob
import io
import json
import multiprocessing as mp
import re
import sys
import traceback
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import lb_lib as lb
import lb_her_nb14 as her14
import lb_her_eval14 as her_eval
import lb_her14_run as H
import lb_fit_daily as F
import lb_spec_daily as S

SPEC_NAME = F.SPEC_NAME
HER_CELLS = F.HER_CELLS                           # 1, 2, 3 (feature_panel given), 4 to 8, 10
SERIES_CELLS = [1, 2, 3, 4, 5]                    # her series_store and model_parameters only
EVAL_CELLS = [11, 12, 13, 14, 15]
FIG_ROOT = lb.LB / "figures_daily"
WORKERS = 32
STORES = ("series_store", "model_parameters", "stack_case_store", "final_fit_store", "daily_effects")

CASES = pd.read_csv(lb.LB / "lb_cases_daily.csv", float_precision="round_trip")
CASE_KEYS = sorted(set(zip(CASES["site_id"], CASES["sensor"])))      # her cell 10 order: site, then sensor


def safe(row):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", row)


def spec_json(spec):
    return json.dumps(spec, sort_keys=True)


# ----------------------------------------------------------------------------- one case, her statements
def case_stores(job):
    """Her cells on one case at one spec. job = (site_id, sensor, spec json, cells). Returns the stores."""
    site_id, sensor, sj, cells = job
    spec = json.loads(sj)
    c = CASES[(CASES["site_id"] == site_id) & (CASES["sensor"] == sensor)]
    fp = c[["site_id", "sensor", "band", "date", "y"]].copy()
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
                ns.update(F.class1_binds(spec))
    return {k: ns[k] for k in STORES if k in ns}


def run_jobs(jobs):
    jobs = sorted(set(jobs))
    with mp.get_context("fork").Pool(min(WORKERS, max(1, len(jobs)))) as pool:
        out = pool.map(case_stores, jobs, chunksize=1)
    return dict(zip(jobs, out))


# ----------------------------------------------------------------------------- answers and matched fits
def ndec(v):
    return max(0, -Decimal(repr(v)).as_tuple().exponent)


def matches(fit, values):
    """Every fit prediction, rounded to the decimals of the answer value at the same visit and band, equals it; and
    the fit and the answer cover the same visits and bands."""
    preds = fit.get("predictions") or {}
    pairs = {(p, lb.disp(b)) for b, pv in preds.items() for p in pv}
    if pairs != {(p, b) for p, bv in values.items() for b in bv}:
        return False
    return all(round(float(v), ndec(values[p][lb.disp(b)])) == values[p][lb.disp(b)]
               for b, pv in preds.items() for p, v in pv.items())


def last_records():
    """{(experiment, model): {(site_id, sensor): record}} from the last record per prompt_id of every daily file."""
    rows = {}
    for f in sorted(glob.glob(str(lb.LB / "lb_replies_daily_*.jsonl"))):
        last = {}
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                last[r["prompt_id"]] = r
        for r in last.values():
            rows.setdefault((r["experiment"], r["model"]), {})[(r["site_id"], r["sensor"])] = r
    return rows


def post_dates(site_id, sensor):
    c = CASES[(CASES["site_id"] == site_id) & (CASES["sensor"] == sensor) & (CASES["part"] == "post")]
    return dict(zip(c["period_id"], c["date"]))


# ----------------------------------------------------------------------------- her cell 10 rows
def with_prediction(effects, pred):
    """Her cell 10 rows with `pred` {(band, date): value} in place of her y_star_synthetic, her formulas after it:
    delta_hat = y_star_obs - y_star_synth, gamma_hat = delta_hat, y0_sa = y_sa - gamma_hat,
    counterfactual_y0 = y0_sa + seasonal_hat (0.0)."""
    e = effects.copy()
    e["y_star_synthetic"] = [pred[(b, pd.Timestamp(d))] for b, d in zip(e["band"], e["date"])]
    e["effect_gamma_hat"] = e["y_star_observed"] - e["y_star_synthetic"]
    e["counterfactual_y0"] = (e["observed_y_sa"] - e["effect_gamma_hat"]) + 0.0
    return e


def built_rows(site_id, sensor, pred, intervention):
    """Her cell 10 columns for a case with no fit of her code (a rule row, a fit of another engine, an answer that
    matches no fit): her row order (band in her band order, then visit), her formulas, fit columns empty."""
    c = CASES[(CASES["site_id"] == site_id) & (CASES["sensor"] == sensor) & (CASES["part"] == "post")]
    rows = []
    for band in lb.BANDS[sensor]:
        cb = c[c["band"] == band].sort_values("date")
        for k, (d, y) in enumerate(zip(pd.to_datetime(cb["date"]), cb["y"]), start=1):
            obs_y = float(y)
            seasonal_hat = 0.0
            y_sa = obs_y
            y_star_synth = float(pred[(band, d)])
            y_star_obs = y_sa
            gamma_hat = y_star_obs - y_star_synth
            y0_sa = y_sa - gamma_hat
            rows.append({"site_id": site_id, "sensor": sensor, "band": band, "specification": SPEC_NAME, "date": d,
                         "post_horizon_k": k, "days_since_event": int((d - intervention).days), "observed_y": obs_y,
                         "observed_y_sa": y_sa, "counterfactual_y0": y0_sa + seasonal_hat,
                         "y_star_observed": y_star_obs, "y_star_synthetic": y_star_synth,
                         "effect_gamma_hat": gamma_hat, "m_star": np.nan, "J_star": np.nan, "n_post": np.nan,
                         "stacked_latent_mse": np.nan, "band_latent_mse": np.nan, "selection_mode": None})
    return pd.DataFrame(rows)


def answer_pred(record, site_id, sensor):
    dates = post_dates(site_id, sensor)
    disp_to_band = {lb.disp(b): b for b in lb.BANDS[sensor]}
    return {(disp_to_band[b], pd.Timestamp(dates[p])): float(v) for p, bv in record["values"].items()
            for b, v in bv.items()}


# ----------------------------------------------------------------------------- her cells 11 to 15 on one row
def statements(src, cell):
    tree = ast.parse(src)
    return [(n.lineno, compile(ast.Module(body=[n], type_ignores=[]), f"<notebook 14 cell {cell}>", "exec"))
            for n in tree.body]


def evaluate(row, cases, figures=True):
    """cases: {(site_id, sensor): {"effects": DataFrame, "series": stores or None, "fit": stores or None}}.
    Merges the stores in her case order and runs her cells 11 to 15. Returns (tables, namespace, log)."""
    ns = {"__name__": "her_notebook_14"}
    with contextlib.redirect_stdout(io.StringIO()):
        for i in (1, 2):
            exec(compile(her14.CELL_SRC[i], f"<notebook 14 cell {i}>", "exec"), ns)
        exec(compile(her_eval.CELL_SRC[1], "<notebook 14 cell 1>", "exec"), ns)
    keys = [k for k in CASE_KEYS if k in cases]
    ns["daily_effects"] = pd.concat([cases[k]["effects"] for k in keys], ignore_index=True)
    series = [k for k in keys if cases[k]["series"] is not None]
    if series:
        ns["series_store"] = {sk: v for k in series for sk, v in cases[k]["series"]["series_store"].items()}
        ns["model_parameters"] = pd.concat([cases[k]["series"]["model_parameters"] for k in series], ignore_index=True)
    fits = [k for k in keys if cases[k]["fit"] is not None]
    if fits:
        ns["stack_case_store"] = {sk: v for k in fits for sk, v in cases[k]["fit"]["stack_case_store"].items()}
        ns["final_fit_store"] = {sk: v for k in fits for sk, v in cases[k]["fit"]["final_fit_store"].items()}
    fig_dir = FIG_ROOT / safe(row)
    if figures and fig_dir.exists():
        for old in fig_dir.glob("*.png"):
            old.unlink()
        fig_dir.rmdir()
    if figures and fits:                       # her cell 15 draws one figure per final_fit_store case
        fig_dir.mkdir(parents=True)
    ns["FIGURE_DIR"] = fig_dir
    printed, log = {}, []
    for cell in EVAL_CELLS:
        if cell == 15 and not figures:
            continue
        src = her_eval.CELL_SRC[cell] if cell in her_eval.CELL_SRC else her14.CELL_SRC[cell]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for line, code in statements(src, cell):
                try:
                    exec(code, ns)
                except Exception as exc:                                   # noqa: BLE001  logged per row
                    log.append({"row": row, "cell": cell, "line": line,
                                "error": f"{type(exc).__name__}: {exc}".splitlines()[0][:300]})
        printed[cell] = buf.getvalue()
    present = {k for k, v in ns.items() if not (isinstance(v, pd.DataFrame) and v.empty)}
    tables = H.parse_printed({c: printed[c].splitlines() for c in (11, 12, 13, 14)}, present)
    ns["_printed"] = printed
    return tables, ns, log


# ----------------------------------------------------------------------------- rows
def her_cases(stores, keys=None):
    keys = CASE_KEYS if keys is None else keys
    out = {}
    for k in keys:
        s = stores[(k[0], k[1], spec_json(S.HER_SPEC), tuple(HER_CELLS))]
        out[k] = {"effects": s["daily_effects"], "series": s, "fit": s}
    return out


def skill_cases(recs, stores, match_rows, row):
    intervention = pd.Timestamp("2024-09-27")
    out = {}
    for k in CASE_KEYS:
        r = recs.get(k)
        base = {"row": row, "site_id": k[0], "sensor": k[1], "prompt_id": r["prompt_id"] if r else None,
                "ok": bool(r and r.get("ok")), "ended": r.get("ended") if r else None}
        if not (r and r.get("ok")):
            match_rows.append({**base, "n_fit_runs": None, "n_matching": None, "matched_run": None, "engine": None,
                               "rerun_equals_record": None})
            continue
        fits = r.get("fit_runs") or []
        hit = [i for i, fr in enumerate(fits) if matches(fr, r["values"])]
        pred = answer_pred(r, *k)
        m = fits[hit[-1]] if hit else None
        engine = m["engine_used"] if m else None
        rec = {**base, "n_fit_runs": len(fits), "n_matching": len(hit), "matched_run": hit[-1] + 1 if hit else None,
               "engine": engine, "rerun_equals_record": None}
        if engine == "her":
            s = stores[(k[0], k[1], spec_json(m["spec"]), tuple(HER_CELLS))]
            e = s["daily_effects"]
            mp_ = {(b, pd.Timestamp(d)): v for b, d, v in zip(e["band"], e["date"], e["y_star_synthetic"])}
            dates, her_band = post_dates(*k), {lb.disp(x): x for x in lb.BANDS[k[1]]}
            rec_pred = {(her_band[lb.disp(b)], pd.Timestamp(dates[p])): v
                        for b, pv in m["predictions"].items() for p, v in pv.items()}
            rec["rerun_equals_record"] = rec_pred == mp_
            out[k] = {"effects": with_prediction(e, pred), "series": s, "fit": s}
        elif m is not None:
            s = stores[(k[0], k[1], spec_json(m["spec"]), tuple(SERIES_CELLS))]
            bw = {b: float(v["time_bandwidth"]) for (_, _, b, _), v in s["series_store"].items()}
            rec["rerun_equals_record"] = {lb.disp(b): v for b, v in bw.items()} == \
                {lb.disp(b): float(v) for b, v in m["bandwidths"].items()}
            out[k] = {"effects": built_rows(*k, pred, intervention), "series": s, "fit": None}
        else:
            out[k] = {"effects": built_rows(*k, pred, intervention), "series": None, "fit": None}
        match_rows.append(rec)
    return out


def rule_cases(rule):
    intervention = pd.Timestamp("2024-09-27")
    r = pd.read_csv(lb.LB / "lb_rules_daily.csv", float_precision="round_trip")
    r = r[r["rule"] == rule]
    out = {}
    for k in CASE_KEYS:
        rk = r[(r["site_id"] == k[0]) & (r["sensor"] == k[1])]
        pred = {(b, pd.Timestamp(d)): v for b, d, v in zip(rk["band"], rk["date"], rk["prediction"])}
        out[k] = {"effects": built_rows(*k, pred, intervention), "series": None, "fit": None}
    return out


def needed_jobs(all_recs):
    jobs = {(k[0], k[1], spec_json(S.HER_SPEC), tuple(HER_CELLS)) for k in CASE_KEYS}
    for recs in all_recs.values():
        for k, r in recs.items():
            if not r.get("ok"):
                continue
            hit = [fr for fr in (r.get("fit_runs") or []) if matches(fr, r["values"])]
            if hit:
                m = hit[-1]
                cells = HER_CELLS if m["engine_used"] == "her" else SERIES_CELLS
                jobs.add((k[0], k[1], spec_json(m["spec"]), tuple(cells)))
    return jobs


def her_row(stores, figures=True):
    """Her row through the scorer at her settings on her 52 cases (gate DK11)."""
    return evaluate("her", her_cases(stores), figures=figures)


def main(argv):
    all_recs = {} if "--her" in argv else last_records()
    print(f"cases {len(CASE_KEYS)}; rows from records: {sorted(all_recs)}")
    stores = run_jobs(needed_jobs(all_recs))
    print(f"her statements run on {len(stores)} case and spec pairs")
    printed, effects, inference, logs, match_rows = {}, [], [], [], []

    def add(row, tables, ns, log, printed_tables=True):
        logs.extend(log)
        effects.append(ns["daily_effects"].assign(row=row))
        if "conformal_inference" in ns and not ns["conformal_inference"].empty:
            inference.append(ns["conformal_inference"].assign(row=row))
        if printed_tables:
            for name, df in tables.items():
                printed.setdefault(name, []).append(df.assign(row=row))
        print(f"{row}: {len(ns['daily_effects'])} effect rows, tables {len(tables)}, statements raised {len(log)}")

    tables, ns, log = her_row(stores)
    add("her", tables, ns, log, printed_tables=False)
    for name in tables:                                   # her row in the printed files is what her notebook printed
        printed.setdefault(name, []).append(pd.read_csv(lb.LB / f"lb_her14_printed_{name}.csv", dtype=str)
                                            .assign(row="her"))
    if "--her" not in argv:
        for rule in ("persistence", "last3_mean"):
            add(f"rule:{rule}", *evaluate(f"rule:{rule}", rule_cases(rule)))
        for (exp, model), recs in sorted(all_recs.items()):
            row = f"{exp}|{model}"
            cases = skill_cases(recs, stores, match_rows, row)
            if not cases:
                print(f"{row}: no case with a valid answer")
                continue
            add(row, *evaluate(row, cases))
            full = len(cases) == len(CASE_KEYS)
            series = {k for k, v in cases.items() if v["series"] is not None}
            fits = {k for k, v in cases.items() if v["fit"] is not None}
            if not (full and len(series) == len(CASE_KEYS) and len(fits) == len(CASE_KEYS)):
                hc = her_cases(stores, [k for k in CASE_KEYS if k in cases])
                for k in hc:
                    hc[k]["series"] = hc[k]["series"] if k in series else None
                    hc[k]["fit"] = hc[k]["fit"] if k in fits else None
                add(f"her|{row}", *evaluate(f"her|{row}", hc))
    made = {safe(r) for r in {e["row"].iat[0] for e in effects}}
    for d in FIG_ROOT.iterdir():               # figures of a row this run did not produce are from an earlier run
        if d.is_dir() and d.name not in made:
            for old in d.glob("*.png"):
                old.unlink()
            d.rmdir()
            print(f"removed figures of an earlier run: {d.name}")
    for name, parts in printed.items():
        df = pd.concat(parts, ignore_index=True)
        df[["row"] + [c for c in df.columns if c != "row"]].to_csv(lb.LB / f"lb_printed_skill_daily_{name}.csv",
                                                                    index=False)
    pd.concat(effects, ignore_index=True).to_csv(lb.LB / "lb_effects_skill_daily.csv", index=False)
    pd.concat(inference, ignore_index=True).to_csv(lb.LB / "lb_inference_skill_daily.csv", index=False)
    pd.DataFrame(logs, columns=["row", "cell", "line", "error"]).to_csv(lb.LB / "lb_score_log_skill_daily.csv",
                                                                       index=False)
    pd.DataFrame(match_rows).to_csv(lb.LB / "lb_match_skill_daily.csv", index=False)
    print("wrote lb_printed_skill_daily_<table>.csv, lb_effects_skill_daily.csv, lb_inference_skill_daily.csv, "
          "lb_score_log_skill_daily.csv, lb_match_skill_daily.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
