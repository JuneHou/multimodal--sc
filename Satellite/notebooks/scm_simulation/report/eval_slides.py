"""The results slides in the layout of the README Section 5 evaluation tables: one slide per noise level and DGP, the
README's rows, the collaborator's printed values in the first column and one answer in every other column.

Rows are the README Section 5 evaluation rows except donor mismatch and noise share of MSE (Jun 2026-11-05 "this is
analysis, not metrics"). The collaborator's column is copied from README Section 5 as printed. An answer's value is the
per-replication quantity whose README aggregation gives her printed number (Jun 2026-11-05 "select which i will get
same readme numbers"): its ATT error in the ATT bias row, the absolute value in the ATT RMSE row, yes or no in the two
within rows, and its own counterfactual RMSE, random noise (noise_floor.py), pre-treatment RMSPE, covariate balance,
donors above 0.001, CV-RMSE over pre-treatment RMSPE and largest leave-one-donor-out change from
results/sigma_<s>/. A number in brackets counts the refits that worked where others failed (her_tables.py
failed_refits).

    python report/eval_slides.py           writes report/deck/project/slides/r10_dgp1.html ... r02_dgp2.html
    python report/eval_slides.py --print   prints every table as text
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]                 # notebooks/scm_simulation
README = HERE.parents[1] / "data" / "simulation" / "README.md"
SLIDES = HERE / "report" / "deck" / "project" / "slides"
SIGMAS = ("1.0", "0.5", "0.2")
NB = "‑"                                                   # non-breaking hyphen, as the deck writes names
ROWS = ["ATT bias", "ATT RMSE", "counterfactual RMSE", "random noise", "pre-treatment RMSPE", "covariate balance",
        "within ±0.25 of the truth", "within ±0.50 of the truth", "donors carrying weight",
        "CV-RMSE ÷ pre-treatment RMSPE", "max ATT shift, drop one donor"]
LABEL = {"random noise": "— random noise"}                 # indented under counterfactual RMSE, as in the README
# the deck's scm-design is the four-choice design skill (predictors, V, weights, penalty), whose records are the arm
# scm-four-design; the earlier scm-design with two choices leaves the deck and keeps its records (Jun 2026-11-06 "keep
# only four variable case as scm-design", "slides 8 need reflect this design")
GROUPS = [("scm-configure", "configure"), ("scm-four-design", "design")]
# DeepSeek-V4.1-Flash is not a solver model of the deck (Jun "drop all deepseek in this artifact", "Solver model only",
# 2026-11-06); the profile skill the DeepSeek generator wrote stays, run by the three models below
MODELS = [("gpt-oss-120b", f"gpt{NB}oss"), ("GLM-5.3", "GLM"), ("openai:gpt-5.5", f"GPT{NB}5.5")]
DROPPED_SOLVER = "DeepSeek"
PROFILES = [("profile-gpt-5.4", f"GPT{NB}5.4"), ("profile-deepseek", "DeepSeek")]
KEYS = ["arm", "model", "case"]


def readme_tables():
    """{dgp: {row: [value at sigma 1.0, 0.5, 0.2]}} as README Section 5 prints them."""
    lines = README.read_text().splitlines()
    out = {}
    for g in ("1", "2"):
        i = next(k for k, l in enumerate(lines) if l.startswith(f"#### DGP {g}"))
        t = {}
        for l in lines[i + 1:]:
            if l.startswith("####"):
                break
            if not l.startswith("|") or "---" in l or "σ = 1.0" in l:
                continue
            c = [re.sub(r"\*\*", "", x).strip() for x in l.strip("|").split("|")]
            t[re.sub(r"&nbsp;|— ", "", c[0]).strip()] = c[1:4]
        assert set(ROWS) <= set(t), set(ROWS) - set(t)
        out[f"dgp{g}"] = t
    return out


def answers(sigma):
    r = HERE / "results" / f"sigma_{sigma}"
    t = [pd.read_csv(r / f"table_4_{i}.csv") for i in (1, 2, 3)]
    t = t[0].merge(t[1], on=KEYS + ["answered", "note"], how="left").merge(t[2], on=KEYS + ["answered", "note"],
                                                                         how="left")
    t = t.merge(pd.read_csv(r / "noise_floor.csv"), on=KEYS, how="left")
    if "failed_refits" not in t:
        t["failed_refits"] = None
    return t[~t.arm.isin(["README Section 3", "every factor known"])]


def worked(f, kind):
    """Refits of one kind that worked, when some of that kind failed, else None (as report/sigma_slides.py did)."""
    if not isinstance(f["failed_refits"], str):
        return None
    parts = f["failed_refits"].split("; ")
    if kind == "donor":
        bad = sum(p.startswith("donor") for p in parts)
        return int(f["n_positive"]) - bad if bad else None
    bad = sum(p.startswith("period") for p in parts)
    return 40 - bad if bad else None


def signed(v):
    return ("+" if v > 0 else "−") + f"{abs(v):.3f}"


def value(f, row):
    """An answer's value in one README row, as the slide prints it. A case whose model responded but wrote no answer
    counts as wrong: "no" in the within rows and "no answer" elsewhere, never a dash (Jun "count all no answer as
    wrong. youshould not use - when we have response", 2026-11-06). The within rows stay yes or no, since each model
    ran once (Jun "or this is because we run only once, then keep binary", 2026-11-06)."""
    if f is None:
        return "–"                                          # a model not run
    if not f["answered"]:
        return "no" if row.startswith("within") else "no answer"
    e = f["att_error"]
    if str(f.get("note", "")).startswith("no recorded fit"):    # no recorded fit gives this counterfactual
        return {"ATT bias": signed(e), "ATT RMSE": f"{abs(e):.3f}", "counterfactual RMSE": f"{f['cf_rmse']:.3f}",
                "within ±0.25 of the truth": "yes" if abs(e) <= 0.25 else "no",
                "within ±0.50 of the truth": "yes" if abs(e) <= 0.50 else "no"}.get(row, "no fit")
    if row in ("pre-treatment RMSPE", "covariate balance", "donors carrying weight",
               "CV-RMSE ÷ pre-treatment RMSPE", "max ATT shift, drop one donor") and pd.isna(f["pre_rmspe"]):
        return "fit failed"                                 # a profile answer whose Synth fit failed
    v = {"ATT bias": signed(e), "ATT RMSE": f"{abs(e):.3f}", "counterfactual RMSE": f"{f['cf_rmse']:.3f}",
         "random noise": f"{f['noise_floor']:.3f}", "pre-treatment RMSPE": f"{f['pre_rmspe']:.3f}",
         "covariate balance": f"{f['covariate_balance']:.3f}",
         "within ±0.25 of the truth": "yes" if abs(e) <= 0.25 else "no",
         "within ±0.50 of the truth": "yes" if abs(e) <= 0.50 else "no",
         "donors carrying weight": str(int(f["n_positive"])),
         "CV-RMSE ÷ pre-treatment RMSPE": f"{f['cv_rmse'] / f['pre_rmspe']:.2f}",
         "max ATT shift, drop one donor": f"{f['lodo_max_abs_att_change']:.3f}"}[row]
    if v == "nan":
        v = "failed"                                        # every refit of that kind failed
    kind = {"CV-RMSE ÷ pre-treatment RMSPE": "period", "max ATT shift, drop one donor": "donor"}.get(row)
    n = worked(f, kind) if kind else None
    return v if n is None else f"{v} ({n})"


def columns(sigma, case):
    """[(header, answer row or None)] after the README column: configure and design by model, then the profiles."""
    t = answers(sigma)
    t = t[t.case == case]
    cols = []
    for arm, label in GROUPS:
        for m, mlabel in MODELS:
            r = t[(t.arm == arm) & (t.model == m)]
            if not len(r):
                continue                                    # a model not run at this noise level
            assert len(r) == 1, (arm, m, case)
            cols.append((f"{label}<br>{mlabel}", r.iloc[0]))
    for arm, label in PROFILES:
        r = t[(t.arm == arm) & t.answered.astype(bool) & ~t.model.astype(str).str.startswith(DROPPED_SOLVER)]
        vals = {row: {value(x, row) for _, x in r.iterrows()} for row in ROWS}
        assert all(len(v) == 1 for v in vals.values()), (arm, case, vals)   # one identical answer for every model
        cols.append((f"profile<br>{label}", r.iloc[0]))
    return cols


def grid(sigma, case):
    """[(row label, README value, [answer values])], and the headers."""
    her = readme_tables()[case]
    k = SIGMAS.index(sigma)
    cols = columns(sigma, case)
    return [(LABEL.get(row, row), her[row][k], [value(f, row) for _, f in cols]) for row in ROWS], [h for h, _ in cols]


TEXT = {
    ("1.0", "dgp1"): (
        "On DGP 1 at σ = 1 six of seven cases come within 0.25 of the true effect",
        "GLM‑5.3 under scm‑configure misses the effect by 0.326, and the DeepSeek profile skill comes closest, 0.075 "
        "above it. Both scm‑design answers equal the GPT‑5.4 profile skill’s answer, and with GPT‑5.5 under "
        "scm‑configure they have the lowest counterfactual RMSE, 0.992."),
    ("1.0", "dgp2"): (
        "On DGP 2 at σ = 1 one of seven cases comes within 0.25 of the true effect",
        "GLM‑5.3 under scm‑configure misses the effect by 0.045 and has the lowest counterfactual RMSE, 0.771. "
        "GLM‑5.3 under scm‑design misses by 0.336. Leaving out one donor moves the ATT of the DeepSeek profile skill "
        "by 0.560, the largest shift of any answer."),
    ("0.5", "dgp1"): (
        "On DGP 1 at σ = 0.5 every case comes within 0.25 of the true effect",
        "gpt‑oss‑120b under scm‑design wrote a counterfactual of 0 in every period, which scores 0.023 above the true "
        "effect with the highest counterfactual RMSE, 1.545, and no fit behind it. GLM‑5.3 under scm‑design and the "
        "GPT‑5.4 profile skill give one answer, 0.059 above the true effect, with the lowest counterfactual RMSE, "
        "0.545."),
    ("0.5", "dgp2"): (
        "On DGP 2 at σ = 0.5 four of six cases come within 0.25 of the true effect",
        "Both scm‑design answers equal the GPT‑5.4 profile skill’s answer, 0.247 below the true effect, with the "
        "lowest counterfactual RMSE, 0.446. Every leave‑one‑donor‑out refit of GLM‑5.3 under scm‑configure failed."),
    ("0.2", "dgp1"): (
        "On DGP 1 at σ = 0.2 five of six cases come within 0.25 of the true effect",
        "gpt‑oss‑120b under scm‑configure misses the effect by 0.251. Both scm‑design answers equal the GPT‑5.4 "
        "profile skill’s answer, 0.007 above the true effect, with the lowest counterfactual RMSE, 0.291."),
    ("0.2", "dgp2"): (
        "On DGP 2 at σ = 0.2 five of six cases come within 0.25 of the true effect",
        "The DeepSeek profile skill misses the effect by 0.284. gpt‑oss‑120b under scm‑design comes closest, 0.098 "
        "below the true effect, and both scm‑design answers and the GPT‑5.4 profile skill have the lowest "
        "counterfactual RMSE, 0.189."),
}
ASIDE = ("An answer’s ATT is the mean over periods 41 to 60 of the treated unit’s outcome minus the answer’s "
         "counterfactual, as the collaborator’s code computes it. ATT bias is its error, ATT RMSE the error’s absolute "
         "value, and each within row says whether the error lies within the bound. Random noise is σ√(1 + Σw²) at the "
         "answer’s own weights.")


def caption(sigma, case, brackets):
    c = (f"README Section 5 rows, DGP {case[-1]}, σ = {sigma}. README column, the collaborator’s mean over 1,000 "
         "replications, the leave‑one‑out rows over 100. Every other column, one answer.")
    return c + (" A number in brackets counts the refits that worked." if brackets else "")


# header labels on one line and tables at 26px (Jun 2026-11-06 "update the table's first row as what i did in slides 11,
# for all table increase the front a bit"); column widths follow their content ("doesn't have to be same width")
def slide(sigma, case):
    rows, heads = grid(sigma, case)
    sid = f"r{sigma.replace('.', '')}_{case}"
    title, para = TEXT[(sigma, case)]
    th = lambda a, x: f'<th style="text-align:{a};padding:2px 5px">{x}</th>'  # noqa: E731
    head = '<tr style="background:#DDE3DD">' + th("left", "") + th("right", "README Section 5") + \
           "".join(th("right", h.replace("<br>", " ")) for h in heads) + "</tr>"
    td = 'style="text-align:right;padding:2px 5px"'
    body = "\n".join(f'<tr style="background:{"#F7F6F2" if i % 2 else "#FFFFFF"}"><td style="padding:2px 5px">{lab}</td>'
                     f'<td style="text-align:right;padding:2px 5px;font-weight:600">{her}</td>'
                     + "".join(f"<td {td}>{v}</td>" for v in vals) + "</tr>"
                     for i, (lab, her, vals) in enumerate(rows))
    brackets = any("(" in v for _, _, vals in rows for v in vals)
    html = (f'<section id="{sid}" data-transition="fade" style="background:#EEF0EC;color:#17212B;font-family:\'IBM Plex '
            f'Sans\', Arial, sans-serif;padding:128px 128px 160px;display:flex;flex-direction:column;gap:24px">\n'
            f'<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#0E6B73">'
            f'Results · DGP {case[-1]} · σ = {sigma}</p>\n'
            f'<h2 style="font-family:\'Source Serif 4\', Georgia, serif;font-size:60px;font-weight:600;line-height:1.1;'
            f'color:#17212B">{title}</h2>\n'
            f'<table style="font-size:26px;font-variant-numeric:tabular-nums;color:#17212B">\n{head}\n{body}\n</table>\n'
            f'<p style="font-size:24px;line-height:1.45;color:#17212B">{para}</p>\n'
            f'<p style="position:absolute;left:128px;bottom:64px;width:1664px;font-size:24px;color:#5B6670">'
            f'{caption(sigma, case, brackets)}</p>\n'
            f'<aside>{ASIDE}</aside>\n</section>\n')
    return html, sid


def main():
    if "--print" in sys.argv:
        for sigma in SIGMAS:
            for case in ("dgp1", "dgp2"):
                rows, heads = grid(sigma, case)
                print(f"\n== sigma {sigma} {case}\n" + " | ".join(["", "README"] + [h.replace("<br>", " ") for h in heads]))
                for lab, her, vals in rows:
                    print(" | ".join([lab, her] + vals))
        return
    for sigma in SIGMAS:
        for case in ("dgp1", "dgp2"):
            html, sid = slide(sigma, case)
            (SLIDES / f"{sid}.html").write_text(html)
            print("wrote", sid)


if __name__ == "__main__":
    main()
