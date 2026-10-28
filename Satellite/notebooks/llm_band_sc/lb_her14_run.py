"""lb_her14_run — run the collaborator's notebook 14 (daily SHC) on our copies of her daily panels.

Plan ~/.claude/plans/simple-skills-daily.md, step 3. Her cells, extracted byte for byte into lb_her_nb14.CELL_SRC,
are executed in notebook order in one namespace, as her notebook runs them. PANEL_FILES, which her cell 2 resolves
from her Dropbox as the first existing of treatment.xlsx and treatment.csv, is resolved the same way in our copy of
her folder, `data/daily_Sep2020_Sep2025_datasets/sentinel{1,2}/`, so it is treatment.xlsx, the file her printout
names. The two files are not bit-identical: 17,563 of 62,860 y values differ in the last bits, at most 5.5e-12
(checked 2026-10-15). Jun "yes" (2026-10-15) to installing openpyxl so the satellite Python reads the xlsx.

    python lb_her14_run.py --parity    run her cells 1 to 13; compare what each prints with what her notebook printed
                                       (cell 8: her first output chunk, which holds the whole printout); write
                                       lb_her14_effects.csv (her daily_effects), lb_her14_effect_by_site_band.csv,
                                       lb_her14_effect_summary.csv, lb_her14_inference.csv and lb_her14_parity.txt
    python lb_her14_run.py --her-tables  read the tables her notebook printed in cells 12, 13 and 14 into
                                       lb_her14_printed_<table>.csv, values kept as the printed text

Gate DK2 is cells 5 and 8 printing what hers printed, and cell 12 printing what hers printed except numbers that
differ from hers by 0.0001, the difference taken exactly on the printed decimals: Jun "i don't care the 0.0001"
(2026-10-15). Every other cell's comparison is printed and does not gate. Her SHC row is her printed tables: Jun "if
we can see her result, use her result" (2026-10-15).
"""
import lb_one_thread  # noqa: F401  first import: one thread, as the skill sandbox runs
import contextlib
import decimal
import io
import json
import re
import sys
import time

import pandas as pd

import lb_lib as lb
import lb_her_nb14 as her14
import lb_extract_her14

DAILY = lb.ROOT / "Satellite" / "data" / "daily_Sep2020_Sep2025_datasets"
PANEL_FILES = {s: next(p for p in (DAILY / s / "treatment.xlsx", DAILY / s / "treatment.csv") if p.exists())
               for s in ("sentinel1", "sentinel2")}          # her cell 2 order of candidates
RUN_ORDER = list(her14.CELL_SRC)
GATE_EXACT, GATE_0001 = (5, 8), (12,)                 # plan, Decisions "parity" and "parity, difference counted ..."
STEP = decimal.Decimal("0.0001")                      # Jun "i don't care the 0.0001" (2026-10-15)


def run_cells(panel_files=PANEL_FILES, bind=None, cells=RUN_ORDER):
    """Execute her cells in one namespace. `bind` rebinds names after her cell 2 sets them (none for parity).
    Returns (namespace, {cell: printed text}, {cell: seconds})."""
    ns, printed, seconds = {"__name__": "her_notebook_14"}, {}, {}
    for i in cells:
        if i == 3:
            ns["PANEL_FILES"] = dict(panel_files)
        buf, t0 = io.StringIO(), time.time()
        with contextlib.redirect_stdout(buf):
            exec(compile(her14.CELL_SRC[i], f"<notebook 14 cell {i}>", "exec"), ns)
        printed[i], seconds[i] = buf.getvalue(), time.time() - t0
        if i == 2 and bind:
            unknown = [k for k in bind if k not in ns]
            if unknown:
                raise KeyError(f"not a name her cell 2 sets: {unknown}")
            ns.update(bind)
    return ns, printed, seconds


def her_printed():
    nb, _ = lb_extract_her14.notebook()
    out = {}
    for i in RUN_ORDER:
        chunks = [o for o in nb["cells"][i].get("outputs", []) if o.get("output_type") == "stream" and o.get("name") == "stdout"]
        out[i] = "".join(chunks[0]["text"]) if chunks else ""
    return out


def _num(tok):
    try:
        return decimal.Decimal(tok)
    except decimal.InvalidOperation:
        return None


def within_0001(a, b):
    """True when line a equals line b except numbers that differ by at most 0.0001, compared exactly on the printed
    decimals; the whitespace between tokens must be the same."""
    pa, pb = re.split(r"(\s+)", a), re.split(r"(\s+)", b)
    if len(pa) != len(pb):
        return False
    for x, y in zip(pa, pb):
        if x == y:
            continue
        nx, ny = _num(x), _num(y)
        if nx is None or ny is None or not nx.is_finite() or not ny.is_finite() or abs(nx - ny) > STEP:
            return False
    return True


def compare(printed, hers):
    """Per-cell comparison report and the DK2 verdict."""
    lines, ok_gate = [], True
    for i in RUN_ORDER:
        al = printed[i].splitlines()
        bl = hers[i].splitlines()
        if i in (1, 2):                    # her cell 2 also printed the Dropbox paths her statements resolved
            bl = [l for l in bl if not l.startswith(("sentinel1 panel:", "sentinel2 panel:", "DAILY_DIR", "OUTPUT_DIR"))]
        same = al == bl
        diffs = [(k, x, y) for k, (x, y) in enumerate(zip(al, bl)) if x != y]
        near = [d for d in diffs if within_0001(d[1], d[2])]
        if i in GATE_EXACT:
            passed, rule = same, "gate, identical"
        elif i in GATE_0001:
            passed, rule = len(al) == len(bl) and len(near) == len(diffs), "gate, numbers within 0.0001"
        else:
            passed, rule = True, "reported, not gated"
        if i in GATE_EXACT + GATE_0001:
            ok_gate &= passed
        verdict = "identical" if same else ("differs" + ("" if i not in GATE_EXACT + GATE_0001 else
                                                          ", passes" if passed else ", FAILS"))
        lines.append(f"cell {i:2d}: {verdict} [{rule}] ({len(al)} lines ours, {len(bl)} hers)")
        if not same:
            lines.append(f"    {len(diffs)} differing lines in the common length, {len(near)} of them only in numbers "
                         f"within 0.0001")
            if len(al) != len(bl):
                lines.append(f"    line counts differ: {len(al)} vs {len(bl)}")
            rest = [d for d in diffs if d not in near]
            if rest and sorted(al) == sorted(bl):
                lines.append(f"    the other {len(rest)} lines are the same lines in another order")
            for k, x, y in (rest or near)[:15]:
                lines.append(f"    line {k + 1}\n      ours: {x}\n      hers: {y}")
    lines.append("DK2 " + ("ok: cells 5 and 8 identical, cell 12 identical except numbers within 0.0001"
                           if ok_gate else "FAIL: see the cells marked FAILS"))
    return "\n".join(lines), ok_gate


def parity():
    ns, printed, seconds = run_cells()
    report, ok = compare(printed, her_printed())
    report += "\nseconds per cell: " + ", ".join(f"{i}: {seconds[i]:.1f}" for i in RUN_ORDER)
    (lb.LB / "lb_her14_parity.txt").write_text(report + "\n")
    (lb.LB / "lb_her14_printed.json").write_text(json.dumps({str(i): printed[i] for i in RUN_ORDER}, indent=0))
    print(report)
    ns["daily_effects"].to_csv(lb.LB / "lb_her14_effects.csv", index=False)
    ns["effect_by_site_band"].to_csv(lb.LB / "lb_her14_effect_by_site_band.csv", index=False)
    ns["effect_summary"].to_csv(lb.LB / "lb_her14_effect_summary.csv", index=False)
    ns["conformal_inference"].to_csv(lb.LB / "lb_her14_inference.csv", index=False)
    return ok


# ---- her printed tables (plan, Decisions "her SHC row" and "her printed tables, files") ----

def _her_output(i):
    """Every stdout chunk her cell i printed, joined."""
    nb, _ = lb_extract_her14.notebook()
    return "".join("".join(o["text"]) for o in nb["cells"][i].get("outputs", [])
                   if o.get("output_type") == "stream" and o.get("name") == "stdout").splitlines()


def _rows(lines, h):
    out = []
    for row in lines[h + 1:]:
        if not row.strip():
            break
        out.append(row)
    return out


def _fixed_width(lines, h):
    """A table printed by DataFrame.to_string(index=False): every column is right-aligned, so each column ends where
    its header name ends."""
    header = lines[h]
    ends = [m.end() for m in re.finditer(r"\S+", header)]
    cols = header.split()
    starts = [0] + ends[:-1]
    data = []
    for row in _rows(lines, h):
        if len(row.rstrip()) > ends[-1] or any(row[e:e + 1] not in ("", " ") for e in ends[:-1]):
            raise ValueError(f"row does not fit the header columns: {row!r}")
        data.append([row[s:e].strip() for s, e in zip(starts, ends)])
    return pd.DataFrame(data, columns=cols, dtype=str)


def _indexed(lines, h, index_name):
    """A groupby table printed by to_string(): the header line, a line holding the index name, then the rows."""
    if lines[h + 1].split() != [index_name]:
        raise ValueError(f"expected the index line {index_name!r} after {lines[h]!r}")
    cols = [index_name] + lines[h].split()
    data = [row.split() for row in _rows(lines, h + 1)]
    if any(len(r) != len(cols) for r in data):
        raise ValueError(f"a row of the table under {lines[h]!r} has the wrong number of fields")
    return pd.DataFrame(data, columns=cols, dtype=str)


def _after(lines, anchor, first_col):
    a = lines.index(anchor)
    return next(k for k in range(a + 1, len(lines)) if lines[k].split()[:1] == [first_col])


def _next_line(lines, anchor):
    return lines[lines.index(anchor) + 1]


def _line_starting(lines, prefix):
    return next(l for l in lines if l.startswith(prefix))


def _validation_checks(c11):
    """Her cell 11 checks 1, 2 and 4, each printed as a heading and a line 'passed P of N cases' or 'P of N cases'."""
    rows = []
    for heading in ("1. Weights sum to one:", "2. Weights are non-negative:", "4. Stacked fit error within nu at m*:"):
        m = re.fullmatch(r"\s*(?:passed )?(\S+) of (\S+) cases", _next_line(c11, heading))
        rows.append([heading.rstrip(":"), m.group(1), m.group(2)])
    return pd.DataFrame(rows, columns=["check", "passed", "cases"], dtype=str)


def _effect_summary(c12):
    """Her cell 12 prints effect_summary once per sensor, under a line holding the sensor's name."""
    parts = []
    for sensor in ("sentinel1", "sentinel2"):
        if sensor not in c12:
            continue
        part = _fixed_width(c12, c12.index(sensor) + 1)
        part.insert(0, "sensor", sensor)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _value_counts(lines, anchor):
    """A Series.value_counts() printed by to_string(): the index name on its own line, then 'value count' rows."""
    h = lines.index(anchor) + 1
    name = lines[h].split()
    if len(name) != 1:
        raise ValueError(f"expected the index name line after {anchor!r}")
    return pd.DataFrame([r.split() for r in _rows(lines, h)], columns=[name[0], "count"], dtype=str)


# The printed tables of each cell, and the namespace object each one prints. A row whose namespace lacks the object
# (a row with no fit has no final_fit_store, so no validation_report or table1_model_selection) has no such table.
PRINTED_TABLES = {
    11: ["validation_checks", "selection_mode", "bands_sharing_weights", "pre_rmse_over_sd", "validation_by_case"],
    12: ["effect_summary", "effect_by_site_band"],
    13: ["summary_lines_13", "rejections_by_band", "rejected_by_either", "acf1_by_sensor", "resid_sd_ratio_by_sensor",
         "p_by_statistic", "pointwise_share"],
    14: ["table1", "donor_pool_by_sensor", "table2", "table2_by_band", "table3", "table3_by_band", "summary_lines_14"],
}
PRINTS = {"validation_checks": "validation_report", "selection_mode": "validation_report",
          "bands_sharing_weights": "validation_report", "validation_by_case": "validation_report",
          "pre_rmse_over_sd": "pre_fit_quality", "effect_summary": "effect_summary",
          "effect_by_site_band": "effect_by_site_band", "table1": "table1_model_selection",
          "donor_pool_by_sensor": "table1_model_selection", "table2": "table2_effects", "table2_by_band": "table2_by_band",
          "table3": "table3_inference", "table3_by_band": "table3_by_band", "summary_lines_14": "table3_by_band",
          **{t: "conformal_inference" for t in PRINTED_TABLES[13]}}
SUMMARY_PREFIXES = {13: ("Tests:", "  A, SHC replication:", "  B, CWZ moving block:", "Totals:", "stated setting is n"),
                    14: ("Totals:",)}


def parse_printed(out, present=None):
    """Her printed tables from the printed lines of her cells 11 to 14 ({cell: [lines]}), values kept as the printed
    text. `present`: the namespace names the run holds; a table whose object is absent is not parsed (None = all)."""
    want = [n for c in PRINTED_TABLES if c in out for n in PRINTED_TABLES[c] if present is None or PRINTS[n] in present]
    c11, c12, c13, c14 = (out.get(i, []) for i in (11, 12, 13, 14))
    parse = {
        "validation_checks": lambda: _validation_checks(c11),
        "selection_mode": lambda: _value_counts(c11, "   Selection mode:"),
        "bands_sharing_weights": lambda: _indexed(c11, c11.index(
            "   by construction: one stacked problem per site and sensor, bands enter as stacked rows") + 1, "sensor"),
        "pre_rmse_over_sd": lambda: _indexed(c11, c11.index("5. Pre-treatment smooth, RMSE relative to outcome SD:") + 1,
                                             "sensor"),
        "validation_by_case": lambda: _fixed_width(c11, _after(c11, "Per-case validation:", "site_id")),
        "effect_summary": lambda: _effect_summary(c12),
        "effect_by_site_band": lambda: _fixed_width(c12, _after(c12, "Per-site, per-band effects:", "sensor")),
        "summary_lines_13": lambda: pd.DataFrame(
            [[l] for l in c13 if l.startswith(SUMMARY_PREFIXES[13])], columns=["line"], dtype=str),
        "rejections_by_band": lambda: _fixed_width(c13, _after(c13, "REJECTION COUNTS AT 5 PERCENT, by band", "sensor")),
        "rejected_by_either": lambda: _fixed_width(c13, _after(c13, "SITE-BAND TESTS REJECTED BY EITHER PROCEDURE",
                                                               "site_id")),
        "acf1_by_sensor": lambda: _indexed(c13, c13.index("the applicable case.") + 1, "sensor"),
        "resid_sd_ratio_by_sensor": lambda: _indexed(c13, c13.index("distributions the wrong width.") + 1, "sensor"),
        "p_by_statistic": lambda: _fixed_width(c13, _after(c13, "STATISTIC CHOICE, CWZ moving-block p-values by q",
                                                           "sensor")),
        "pointwise_share": lambda: _fixed_width(c13, _after(
            c13, "Share of post periods whose 95 percent interval excludes zero:", "sensor")),
        "table1": lambda: _fixed_width(c14, _after(c14, "TABLE 1. Selected m and donor pool, by site and sensor",
                                                   "sensor")),
        "donor_pool_by_sensor": lambda: _indexed(c14, c14.index("Donor-pool summary by sensor:") + 1, "sensor"),
        "table2": lambda: _fixed_width(c14, _after(c14, "Panel A. By site and band", "sensor")),
        "table2_by_band": lambda: _fixed_width(c14, _after(c14, "Panel B. By band, across sites", "sensor")),
        "table3": lambda: _fixed_width(c14, _after(c14, "Panel A. By site and band, ordered by the CWZ p-value",
                                                   "sensor")),
        "table3_by_band": lambda: _fixed_width(c14, _after(c14, "Panel B. By band, rejection counts at 5 percent",
                                                           "sensor")),
        "summary_lines_14": lambda: pd.DataFrame(
            [[l] for l in c14 if l.startswith(SUMMARY_PREFIXES[14])], columns=["line"], dtype=str),
    }
    return {n: parse[n]() for n in want}


def her_tables():
    t = parse_printed({i: _her_output(i) for i in (11, 12, 13, 14)})
    for name, df in t.items():
        df.to_csv(lb.LB / f"lb_her14_printed_{name}.csv", index=False)
        print(f"lb_her14_printed_{name}.csv: {len(df)} rows, {len(df.columns)} columns")
    return t


if __name__ == "__main__":
    if "--parity" in sys.argv:
        sys.exit(0 if parity() else 1)
    if "--her-tables" in sys.argv:
        her_tables()
        sys.exit(0)
    print(__doc__)
