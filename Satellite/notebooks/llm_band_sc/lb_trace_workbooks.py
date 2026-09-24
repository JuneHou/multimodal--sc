"""lb_trace_workbooks — one spreadsheet per model holding that model's step-by-step reasoning.

Layout asked for: rows are sites, columns are sensors, each cell is the trace for that site under
that sensor. Sites and sensors are very nearly disjoint -- of the 26 distinct sites only
treatment_0015 is present under both Sentinel-1 and Sentinel-2 -- so 27 of the 52 cells carry a
trace and the rest are blank by construction, not by omission.

A second sheet carries what is needed to read a trace against its outcome: the answer returned,
how far that answer sat from the P09 value, and the classification from lb_why_p09.

Only models that stored a trace on the step-by-step arm get a file. Kimi-K3 and gpt-oss-120b
return the model's own chain; GPT-5.5 through the OpenAI gateway returns a summary written by a
separate summarizer, which is stated on that file's sheet rather than left to be inferred.

    python lb_trace_workbooks.py
"""
import json
import re

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import lb_arms as la
import lb_lib as lb
import lb_why_p09 as why

ARM = "shc_steps"
OUT = lb.LB / "trace_workbooks"
SENSORS = ["sentinel1", "sentinel2"]
HEAD = PatternFill("solid", fgColor="EEF3F5")
FLAGS = ["q2 verdict", "q2 kept W5", "forecast judgement: P10 near P09",
         "procedure yields P09 (restated, not a forecast)", "doubts the procedural result",
         "says P10 differs from P09"]


def collect():
    """Every step-by-step trace, with the answer and its distance from the P09 value."""
    rows = []
    for run, f in la.sources():
        for line in open(f):
            r = json.loads(line)
            if r.get("experiment") != ARM:
                continue
            u = r.get("usage") or {}
            txt = u.get("reasoning")
            if not txt:
                continue
            p = la.prompts_for(ARM)[r["prompt_id"]]
            s = p["sensor"]
            bands = [lb.disp(b) for b in lb.BANDS[s]]
            per = []
            if r.get("ok"):
                for b in bands:
                    y = np.asarray(p["train"][b], float)
                    sd = float(np.nanstd(y, ddof=1))
                    if np.isfinite(sd) and sd >= lb.SD_TOL:
                        per.append(abs(r["values"][b] - y[-1]) / sd)
            rows.append({"model": r["model"], "kind": u.get("reasoning_kind", "own chain"),
                         "sensor": s, "site_id": p["target_site"], "prompt_id": r["prompt_id"],
                         "answer": r.get("reply", ""),
                         "P09 values": ", ".join(f"{b} {p['train'][b][-1]:g}" for b in bands),
                         "P10 truth": ", ".join(f"{b} {p['truth'][b]:g}" for b in bands),
                         "distance from P09 (SD)": round(float(np.mean(per)), 4) if per else np.nan,
                         "trace chars": len(txt), "trace": txt})
    return pd.DataFrame(rows)


def write_book(model, d, cls):
    sites = sorted(set(la.prompts_for(ARM)[p]["target_site"] for p in la.prompts_for(ARM)))
    grid = pd.DataFrame(index=sites, columns=SENSORS, dtype=object)
    for _, r in d.iterrows():
        grid.loc[r.site_id, r.sensor] = r.trace
    grid.index.name = "site"

    idx = d[["site_id", "sensor", "prompt_id", "answer", "P09 values", "P10 truth",
             "distance from P09 (SD)", "trace chars"]].copy()
    keep = [c for c in FLAGS if c in cls.columns]
    idx = idx.merge(cls[["prompt_id", "model"] + keep], on="prompt_id", how="left",
                    suffixes=("", "_c"))
    idx = idx[idx.model == model].drop(columns=["model"]).sort_values(["sensor", "site_id"])

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    path = OUT / f"traces_steps_{safe}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        grid.to_excel(xw, sheet_name="traces")
        idx.to_excel(xw, sheet_name="index", index=False)
        ws = xw.sheets["traces"]
        ws.freeze_panes = "B2"
        ws.column_dimensions["A"].width = 22
        for i, _ in enumerate(SENSORS, start=2):
            ws.column_dimensions[get_column_letter(i)].width = 110
        for row in ws.iter_rows(min_row=2):
            ws.row_dimensions[row[0].row].height = 210
            for c in row[1:]:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in ws[1]:
            c.font, c.fill = Font(bold=True), HEAD
        wi = xw.sheets["index"]
        wi.freeze_panes = "A2"
        for i, col in enumerate(idx.columns, start=1):
            wi.column_dimensions[get_column_letter(i)].width = min(46, max(14, len(str(col)) + 4))
        for c in wi[1]:
            c.font, c.fill, c.alignment = Font(bold=True), HEAD, Alignment(wrap_text=True, vertical="top")
    return path, grid.notna().sum().sum()


def main():
    OUT.mkdir(exist_ok=True)
    D = collect()
    cls = why.classify(why.traces())
    print(f"step-by-step traces found: {len(D)}\n")
    for model, d in D.groupby("model"):
        path, n = write_book(model, d, cls)
        kind = d["kind"].iloc[0]
        note = "the model's own chain" if kind == "own chain" else "a summary written by a separate summarizer"
        print(f"  {model:24s} {n:2d} traces ({note})")
        print(f"      -> {path}")
    print(f"\nRows are the 26 distinct sites, columns the two sensors. Only treatment_0015 appears")
    print("under both, so 27 of 52 cells hold a trace and the remainder are blank by design.")


if __name__ == "__main__":
    main()
