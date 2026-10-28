"""lb_cases_daily — the 52 daily cases of notebook 14 and the two rule rows (plan ~/.claude/plans/simple-skills-daily.md, step 5).

Her cells 1 to 7 run on her xlsx panels (lb_her14_run.run_cells); every case notebook 14 fits (her printed Table 1)
is built by her build_stacked_case. A case's visits are her pre_dates, numbered P001 to P<T0>, and her post_dates,
numbered on from P<T0+1>. Pre values are y of her pre_fit at her pre_dates, the values the model is given. Observed
post values are y of her series at her post_dates, her cell 10 observed_y, used only for scoring.

Rule rows, the monthly rules counted in visits, per band:
    persistence   the last pre-Helene visit carried forward to every post-Helene visit
    last3_mean    the mean of the last three pre-Helene visits at every post-Helene visit

    python lb_cases_daily.py     write lb_cases_daily.csv and lb_rules_daily.csv; check T0 and n against her printed
                                 Table 1 and the observed values against lb_her14_effects.csv (our run of her cell 10)
"""
import lb_one_thread  # noqa: F401  first import: one thread, as the skill sandbox runs
import numpy as np
import pandas as pd

import lb_her14_run
import lb_lib as lb

SPEC = "no_lag"
RULES = ("persistence", "last3_mean")


def pid(position):
    return f"P{position:03d}"


def build_cases():
    ns, _, _ = lb_her14_run.run_cells(cells=[1, 2, 3, 4, 5, 6, 7])
    t1 = pd.read_csv(lb.LB / "lb_her14_printed_table1.csv")
    rows = []
    for sensor, site in zip(t1["sensor"], t1["site_id"]):
        case = ns["build_stacked_case"](site, sensor, SPEC)
        pre_dates, post_dates = list(case["pre_dates"]), list(case["post_dates"])
        T0 = len(pre_dates)
        for band in case["band_order"]:
            obj = ns["series_store"][(site, sensor, band, SPEC)]
            pre = obj["pre_fit"].set_index("date").loc[pre_dates, "y"].to_numpy(float)
            post = obj["series"].set_index("date").loc[post_dates, "y"].to_numpy(float)
            for k, (d, y) in enumerate(zip(pre_dates, pre), start=1):
                rows.append((sensor, site, band, pid(k), d, y, "pre"))
            for k, (d, y) in enumerate(zip(post_dates, post), start=T0 + 1):
                rows.append((sensor, site, band, pid(k), d, y, "post"))
    return pd.DataFrame(rows, columns=["sensor", "site_id", "band", "period_id", "date", "y", "part"])


def rule_rows(cases):
    out = []
    for (sensor, site, band), g in cases.groupby(["sensor", "site_id", "band"], sort=False):
        pre = g.loc[g["part"].eq("pre")].sort_values("period_id")["y"].to_numpy(float)
        post = g.loc[g["part"].eq("post")].sort_values("period_id")
        value = {"persistence": pre[-1], "last3_mean": float(np.mean(pre[-3:]))}
        for rule in RULES:
            out.append(post[["sensor", "site_id", "band", "period_id", "date"]].assign(rule=rule, prediction=value[rule]))
    return pd.concat(out, ignore_index=True)


def check(cases):
    t1 = pd.read_csv(lb.LB / "lb_her14_printed_table1.csv")
    first = cases.groupby(["sensor", "site_id"]).apply(
        lambda g: pd.Series({"T0": g.loc[g["part"].eq("pre"), "date"].nunique(),
                             "n": g.loc[g["part"].eq("post"), "date"].nunique()}), include_groups=False).reset_index()
    m = t1.merge(first, on=["sensor", "site_id"])
    assert len(m) == len(t1) == 52, f"{len(m)} cases matched of {len(t1)}"
    assert (m["T0"] == m["T0_pre"]).all() and (m["n"] == m["n_post"]).all(), "T0 or n differs from her Table 1"
    eff = pd.read_csv(lb.LB / "lb_her14_effects.csv", parse_dates=["date"])
    post = cases.loc[cases["part"].eq("post")]
    j = post.merge(eff[["sensor", "site_id", "band", "date", "observed_y"]], on=["sensor", "site_id", "band", "date"],
                   how="outer", indicator=True)
    assert (j["_merge"] == "both").all(), "post visits differ from her daily_effects rows"
    same = (j["y"] == j["observed_y"]) | (j["y"].isna() & j["observed_y"].isna())
    assert same.all(), f"{int((~same).sum())} observed values differ from her observed_y"
    print(f"cases: {len(m)}; T0 and n equal her printed Table 1 at every case; {len(j):,} observed post values equal "
          f"her cell 10 observed_y (our run), {int(j['y'].isna().sum())} of them missing")


def main():
    cases = build_cases()
    check(cases)
    rules = rule_rows(cases)
    cases.to_csv(lb.LB / "lb_cases_daily.csv", index=False)
    rules.to_csv(lb.LB / "lb_rules_daily.csv", index=False)
    print(f"lb_cases_daily.csv: {len(cases):,} rows ({int(cases['part'].eq('pre').sum()):,} pre, "
          f"{int(cases['part'].eq('post').sum()):,} post); lb_rules_daily.csv: {len(rules):,} rows")
    print("pre values missing:", int(cases.loc[cases["part"].eq("pre"), "y"].isna().sum()))


if __name__ == "__main__":
    main()
