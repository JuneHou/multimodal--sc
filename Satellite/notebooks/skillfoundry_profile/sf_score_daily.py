"""sf_score_daily — the daily profile rows through notebook 14's evaluation (plan ~/.claude/plans/profile-skills-daily.md).

A profile answer comes from no notebook 14 fit, so it has no residual series and no test of procedures B and A, as a
simple skill answer that matched no fitter run has none (Jun "b", 2026-10-22). Each case of a row enters her cell 10
as lb_skill_score_daily.built_rows writes it for an answer with no fit, and her cells 11 to 14 run on the row's merged
stores through lb_skill_score_daily.evaluate, unchanged; the statements that need a fit raise and are logged, as for the
rule rows. A case with no valid answer is a skipped case. No figure is drawn and no rerun of notebook 14 is made.

    python sf_score_daily.py             score every record in runs/replies_daily_*.jsonl and write the results
    python sf_score_daily.py --table     rewrite results/table_daily.md from the files already written

Writes results/printed_daily_<table>.csv, effects_daily.csv, score_log_daily.csv and table_daily.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LB_DIR = HERE.parent / "llm_band_sc"
sys.path.insert(0, str(LB_DIR))
import lb_one_thread  # noqa: E402,F401  first import: one thread, as the simple daily scoring runs
import glob  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

import lb_lib as lb  # noqa: E402
import lb_skill_score_daily as SC  # noqa: E402

RESULTS = HERE / "results"
INTERVENTION = pd.Timestamp("2024-09-27")
TABLE2_COLUMNS = ["sensor", "band", "sites", "mean_effect", "sd_across_sites", "min_site", "max_site", "sites_negative",
                  "mean_counterfactual", "percent_change"]


def last_records(pattern=str(HERE / "runs" / "replies_daily_*.jsonl")):
    """{(experiment, model): {(site_id, sensor): record}} from the last record per prompt_id of every daily file."""
    rows = {}
    for f in sorted(glob.glob(pattern)):
        last = {}
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                last[r["prompt_id"]] = r
        for r in last.values():
            rows.setdefault((r["experiment"], r["model"]), {})[(r["site_id"], r["sensor"])] = r
    return rows


def profile_cases(recs):
    """Every case with a valid answer: her cell 10 rows of the answer, no residual series, no fit."""
    return {k: {"effects": SC.built_rows(*k, SC.answer_pred(recs[k], *k), INTERVENTION), "series": None, "fit": None}
            for k in SC.CASE_KEYS if k in recs and recs[k].get("ok")}


def score_row(row, recs):
    """(tables, namespace, log) of one row, or None when no case has a valid answer."""
    cases = profile_cases(recs)
    return SC.evaluate(row, cases, figures=False) if cases else None


def table_md():
    """Notebook 14's Table 2 by band as it prints it, one block per row: notebook 14's printed block first, then each
    simple daily row, then each profile row; a row's case count beside its name when fewer than 52."""
    her = pd.read_csv(LB_DIR / "lb_her14_printed_table2_by_band.csv", dtype=str)
    simple = pd.read_csv(LB_DIR / "lb_printed_skill_daily_table2_by_band.csv", dtype=str)
    simple = simple[~simple["row"].str.startswith("her")]
    match = pd.read_csv(LB_DIR / "lb_match_skill_daily.csv")
    n_simple = match[match["ok"]].groupby("row").size().to_dict()
    prof_f = RESULTS / "printed_daily_table2_by_band.csv"
    prof = pd.read_csv(prof_f, dtype=str) if prof_f.exists() else pd.DataFrame(columns=["row"] + TABLE2_COLUMNS)
    n_prof = {f"{e}|{m}": sum(bool(r.get("ok")) for r in recs.values()) for (e, m), recs in last_records().items()}
    out = ["# Notebook 14's Table 2 by band, notebook 14 and every daily row", "",
           "Notebook 14's block is the table notebook 14 printed. Each other block is the same table printed by notebook "
           "14's code on that row's answered cases.", ""]

    def block(name, df):
        out.extend([f"## {name}", "", "| " + " | ".join(TABLE2_COLUMNS) + " |", "|" + "---|" * len(TABLE2_COLUMNS)])
        out.extend("| " + " | ".join(str(r[c]) for c in TABLE2_COLUMNS) + " |" for _, r in df.iterrows())
        out.append("")

    block("Notebook 14 SHC, printed", her)
    for row in simple["row"].unique():
        n = n_simple.get(row, 52)
        block(row + ("" if n == 52 or row.startswith("rule:") else f" ({n} cases)"), simple[simple["row"] == row])
    for row in prof["row"].unique():
        n = n_prof.get(row, 0)
        block(row + ("" if n == 52 else f" ({n} cases)"), prof[prof["row"] == row])
    (RESULTS / "table_daily.md").write_text("\n".join(out))
    print(f"wrote {RESULTS / 'table_daily.md'}")


def main(argv):
    if "--table" in argv:
        return table_md()
    all_recs = last_records()
    print(f"cases {len(SC.CASE_KEYS)}; profile rows from records: {sorted(all_recs)}")
    printed, effects, logs = {}, [], []
    for (exp, model), recs in sorted(all_recs.items()):
        row = f"{exp}|{model}"
        got = score_row(row, recs)
        if got is None:
            print(f"{row}: no case with a valid answer")
            continue
        tables, ns, log = got
        logs.extend(log)
        effects.append(ns["daily_effects"].assign(row=row))
        for name, df in tables.items():
            printed.setdefault(name, []).append(df.assign(row=row))
        print(f"{row}: {len(ns['daily_effects'])} effect rows, tables {sorted(tables)}, statements raised {len(log)}")
    RESULTS.mkdir(exist_ok=True)
    for name, parts in printed.items():
        df = pd.concat(parts, ignore_index=True)
        df[["row"] + [c for c in df.columns if c != "row"]].to_csv(RESULTS / f"printed_daily_{name}.csv", index=False)
    if effects:
        pd.concat(effects, ignore_index=True).to_csv(RESULTS / "effects_daily.csv", index=False)
    pd.DataFrame(logs, columns=["row", "cell", "line", "error"]).to_csv(RESULTS / "score_log_daily.csv", index=False)
    table_md()


if __name__ == "__main__":
    main(sys.argv[1:])
