"""Render the §6/§7 result and pass-count tables of `Docs/9-3-update.md` from
`panel_metric2_p10.csv`, so the document and the CSV cannot drift apart.

Cell = test RMSE (train RMSE), M2, with the tuned hyperparameter appended where the arm has
one (FSC's lambda). Only the arms the document reports are
emitted: of the five alignment arms only D_min, and the reference rows are italicised as
references rather than estimators.

    python panel_metric2_tables.py            # print both sensors' tables to stdout
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import panel_lib as pl

FILLS = {"sentinel1": [("chipmean", "chip-mean fill"), ("histfill", "historical fill")],
         "sentinel2": [("chipmean", "chip-mean fill"), ("histfill", "historical fill"),
                       ("masked", "masked pooling"), ("maskdrop", "masked drop")]}
REPRS = [("chip_mean", "chip mean (5)"), ("gram", "Gram (15)"), ("quantile", "quantile (100)"),
         ("combined", "combined (115)"), ("block245", "2×2 block (245)"),
         ("latent980", "latent980 (980)")]
# (estimator, perm) -> label, in the order the document lists them
LABELS = [
    (("equal_weight", "-"), "*reference — equal-weight donor average*"),
    (("own_history_mean", "-"), "*reference — own-history mean*"),
    (("scm", "-"), "cross-sectional SCM"),
    (("perdim_scm", "-"), "one fit per channel (single-outcome SC, their eq 5)"),
    (("perdim_scm_demeaned", "-"), "one fit per channel + demeaning (their eq 5 + eq 6)"),
    (("perdim_scm_demeaned_std", "-"), "one fit per channel + demeaning + cell standardization (eq 5 + eq 6 + fn 5)"),
    (("scm_demeaned", "-"), "multi-outcome SC + demeaning (their eq 6)"),
    (("scm_demeaned_std", "-"), "multi-outcome SC + demeaning + cell standardization (their fn 5)"),
    (("afsc", "-"), "FSC (augmented, tuned λ)"),
    (("scm", "D_min"), "aligned SCM (D_min)"),
]


def _cell(r):
    if r is None or not np.isfinite(r["rmse_test"]):
        return "—"
    s = f"{r['rmse_test']:.4f} ({r['rmse_train']:.4f})"
    if r.get("estimator") == "afsc" and np.isfinite(r.get("lambda", np.nan)):
        s += f" λ={r['lambda']:.3g}" + ("↑" if r.get("lambda_flag") == "ceiling" else "")
    return s


def _flags(r):
    if r is None or not np.isfinite(r["rmse_test"]):
        return "—"
    return f"{int(r['C2'])} / {int(r['C3'])} of {int(r['n_sites'])}"


def table(P, sensor, kind="rmse"):
    fills = FILLS[sensor]
    head = "| representation | estimator | " + " | ".join(
        f + (" C2 / C3" if kind == "flags" else "") for _, f in fills) + " |"
    out = [head, "|---|---|" + "---|" * len(fills)]
    for rep, rep_label in REPRS:
        first = True
        for (est, perm), label in LABELS:
            if kind == "flags" and est in ("equal_weight", "own_history_mean"):
                continue                          # the references are what C2/C3 compare TO
            rows = [P[(P.fill == f) & (P.sensor == sensor) & (P["repr"] == rep) &
                      (P.estimator == est) & (P.perm == perm)] for f, _ in fills]
            if all(len(r) == 0 for r in rows):
                continue
            cells = [(_flags if kind == "flags" else _cell)(
                None if len(r) == 0 else r.iloc[0]) for r in rows]
            out.append(f"| {'**' + rep_label + '**' if first else ''} | {label} | "
                       + " | ".join(cells) + " |")
            first = False
    return "\n".join(out)


if __name__ == "__main__":
    P = pd.read_csv(pl.TS / "panel_metric2_p10.csv")
    for sensor in ("sentinel1", "sentinel2"):
        print(f"\n===== {sensor}: test RMSE (train RMSE), M2 at P10 =====\n")
        print(table(P, sensor, "rmse"))
        print(f"\n===== {sensor}: pass counts =====\n")
        print(table(P, sensor, "flags"))
