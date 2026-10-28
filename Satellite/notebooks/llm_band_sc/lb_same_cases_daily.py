"""lb_same_cases_daily — her notebook 14 inference counts (procedures A and B) for every daily skill row beside her SHC
on the same site-band tests.

    python lb_same_cases_daily.py

Her side is built only from the numbers her notebook printed: the per site-band rows of her Table 3 panel A
(`lb_her14_printed_table3.csv`, parsed from her saved cell 14 output by `lb_her14_run.py --her-tables`), whose
`cwz_reject_05` (procedure B, CWZ moving block) and `shc_reject_05` (procedure A, SHC bootstrap) are counted over the
row's tests as her panel B counts them (sum of the flags; a disagreement is a test where the two flags differ, her
`methods_agree`). The script first checks that her printed rows give her printed panel B and her Totals line. The row's
own counts come from its Table 3 panel A as her code printed it in the scorer (`lb_printed_skill_daily_table3.csv`),
checked against its panel B and Totals line. Her code rerun over the row's cases (the `her|<row>` rows of the scorer)
is kept beside, because procedure A draws its bootstrap from one random stream over the cases in order.
Writes
    lb_same_cases_daily.csv            per row and scope (all, sensor, band): tests, cases, the row's counts, her
                                       printed counts on the same tests, her code's counts on the same cases
    lb_numbers_same_cases_daily.csv    one number-registry row per number of the file above, written at generation
"""
import datetime as dt
import re

import pandas as pd

import lb_lib as lb

NB14 = "Satellite/data/scripts/scripts/validation/shc/14_shc_daily_sentinel1_sentinel2.ipynb"
KEY = ["sensor", "site_id", "band"]


def flags(df):
    out = df.copy()
    for c in ("cwz_reject_05", "shc_reject_05"):
        out[c] = out[c].map({"True": True, "False": False})
    return out


def counts(df):
    return {"tests": len(df), "cases": df[["sensor", "site_id"]].drop_duplicates().shape[0],
            "cwz_rejects": int(df.cwz_reject_05.sum()), "shc_rejects": int(df.shc_reject_05.sum()),
            "disagreements": int((df.cwz_reject_05 != df.shc_reject_05).sum())}


def totals_line(line):
    m = re.match(r"Totals: CWZ rejects (\d+) of (\d+) site-band tests; SHC bootstrap rejects (\d+); "
                 r"[\d.]+ expected by chance; the procedures disagree on (\d+)\.", line)
    b, n, a, d = map(int, m.groups())
    return {"tests": n, "cwz_rejects": b, "shc_rejects": a, "disagreements": d}


def check(per_test, by_band, line, label):
    """per_test counts equal the printed panel B and Totals line."""
    for (s, b), g in per_test.groupby(["sensor", "band"]):
        pb = by_band[(by_band.sensor == s) & (by_band.band == b)].iloc[0]
        c = counts(g)
        assert (c["tests"], c["cwz_rejects"], c["shc_rejects"], c["disagreements"]) == \
            (int(pb.sites), int(pb.cwz_rejects), int(pb.shc_rejects), int(pb.disagreements)), (label, s, b)
    c = counts(per_test)
    assert {k: c[k] for k in ("tests", "cwz_rejects", "shc_rejects", "disagreements")} == totals_line(line), label


def scopes(df):
    yield "all", df
    for s, g in df.groupby("sensor"):
        yield s, g
    for (s, b), g in df.groupby(["sensor", "band"]):
        yield f"{s} {b}", g


def main():
    her = flags(pd.read_csv(lb.LB / "lb_her14_printed_table3.csv", dtype=str))
    check(her, pd.read_csv(lb.LB / "lb_her14_printed_table3_by_band.csv"),
          pd.read_csv(lb.LB / "lb_her14_printed_summary_lines_14.csv").line.iat[0], "her printed")
    T = flags(pd.read_csv(lb.LB / "lb_printed_skill_daily_table3.csv", dtype=str))
    BB = pd.read_csv(lb.LB / "lb_printed_skill_daily_table3_by_band.csv")
    TL = pd.read_csv(lb.LB / "lb_printed_skill_daily_summary_lines_14.csv")
    her_idx = her.set_index(KEY)
    rows, reg, today = [], [], dt.date.today().isoformat()
    src_row = "Satellite/notebooks/llm_band_sc/lb_printed_skill_daily_table3.csv"
    src_her = "Satellite/notebooks/llm_band_sc/lb_her14_printed_table3.csv"
    for row in [r for r in T.row.unique() if r.startswith("daily_")]:
        own = T[T.row == row]
        check(own, BB[BB.row == row], TL[TL.row == row].line.iat[0], row)
        hcode = T[T.row == f"her|{row}"]
        assert hcode.empty or set(map(tuple, hcode[KEY].values)) == set(map(tuple, own[KEY].values)), row
        her_same = her_idx.loc[list(map(tuple, own[KEY].values))].reset_index()
        for scope, g in scopes(own):
            keys = set(map(tuple, g[KEY].values))
            h = her_same[[k in keys for k in map(tuple, her_same[KEY].values)]]
            hc = hcode[[k in keys for k in map(tuple, hcode[KEY].values)]] if not hcode.empty else None
            c, ch = counts(g), counts(h)
            r = {"row": row, "scope": scope, **c, **{f"her_{k}": v for k, v in ch.items() if k not in ("tests", "cases")}}
            if hc is not None:
                cc = counts(hc)
                r.update({f"hercode_{k}": v for k, v in cc.items() if k not in ("tests", "cases")})
            rows.append(r)
            key = f"same_d_{row.replace('|', '_')}_{scope.replace(' ', '_')}"
            reg.append((f"{key}_row", f"{c['tests']}, {c['cases']}, {c['cwz_rejects']}, {c['shc_rejects']}, "
                        f"{c['disagreements']}",
                        f"{row}, {scope}: site-band tests, cases, procedure B (CWZ) rejections at 5 percent, procedure A "
                        f"(SHC bootstrap) rejections, tests where A and B disagree", src_row,
                        "lb_same_cases_daily.py, counts of her cell 13 flags as her cell 14 panel B counts them", today))
            reg.append((f"{key}_her", f"{ch['cwz_rejects']}, {ch['shc_rejects']}, {ch['disagreements']}",
                        f"her SHC over the same {c['tests']} tests as {row}, {scope}: procedure B rejections, procedure A "
                        f"rejections, disagreements", f"{NB14}; {src_her}",
                        "lb_same_cases_daily.py, from the per site-band flags her cell 14 printed", today))
            if hc is not None:
                reg.append((f"{key}_hercode", f"{cc['cwz_rejects']}, {cc['shc_rejects']}, {cc['disagreements']}",
                            f"her code rerun over the same cases as {row}, {scope}: procedure B rejections, procedure A "
                            f"rejections, disagreements", src_row, "rows her|<row> of the scorer", today))
    D = pd.DataFrame(rows)
    D.to_csv(lb.LB / "lb_same_cases_daily.csv", index=False)
    pd.DataFrame(reg, columns=["name", "value", "what it counts", "source file", "how it is computed", "date read"]
                 ).to_csv(lb.LB / "lb_numbers_same_cases_daily.csv", index=False)
    print(D[D.scope.isin(["all", "sentinel1", "sentinel2"])].to_string(index=False))


if __name__ == "__main__":
    main()
