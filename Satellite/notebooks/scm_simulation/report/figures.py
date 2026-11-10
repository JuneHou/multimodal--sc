"""Figures of the SCM report deck (nature-figure, Python backend, matplotlib). Sized for a 1920 by 1080 slide: a figure
inch is shown as 100 slide pixels, so 17 pt text is about 24 px on the slide; the PNG is exported at 300 dpi for
sharpness. Rscript only computes the README Section 3 weights (her_section3.R); it draws nothing.

    F1  fig1_dgp_paths       DGP 1 and DGP 2, periods 1 to 60: the treated unit's true untreated path and the README
                             Section 3 synthetic path (weights from her_section3.R)
    F2  fig2_case_dgp1       DGP 1, periods 41 to 60: the true untreated path, the README Section 3 counterfactual and
                             every answer's counterfactual, one panel per skill format
    F3  fig3_luck            DGP 1 ATT error, one line per reference and per skill format: the counterfactual that
                             knows every factor over 11 replications (luck.py), then README Section 3 and every
                             answer on the shipped replication

Slide slots, in figure inches at 100 slide pixels each: F1 10.4 by 4.8, F2 9.2 by 6.0, F3 9.6 by 6.0. Every label is
17 pt.

    python report/figures.py      writes report/deck/figs/<name>.{png,svg,pdf} and the alignment JSON
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]                       # notebooks/scm_simulation
OUT = HERE / "report" / "deck" / "figs"
SKILL = Path.home() / ".claude" / "skills" / "nature-figure" / "scripts"
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
sys.path.insert(0, str(HERE.parent / "sc_skills"))
import scs_method  # noqa: E402
MOD = scs_method.load_module("../../scm_simulation")

DATA = HERE.parents[1] / "data" / "simulation" / "simulation_data" / "sigma_1.0"   # the deck's figures are sigma = 1
RSCRIPT = "/data/wang/junh/envs/rsynth/bin/Rscript"

# DeepSeek-V4.1-Flash is not a solver model of the deck (Jun "drop all deepseek in this artifact", "Solver model only",
# 2026-11-06); the profile skill the DeepSeek generator wrote stays
MODELS = [("gpt-oss-120b", "gpt-oss-120b", "#0F4D92"), ("GLM-5.3", "GLM-5.3", "#42949E"),
          ("openai:gpt-5.5", "GPT-5.5", "#B64342")]
DROPPED_SOLVER = "DeepSeek"
# the deck's scm-design is the four-choice design skill, arm scm-four-design (Jun 2026-11-06 "keep only four variable
# case as scm-design")
FORMATS = [("scm-configure", "scm-configure"), ("scm-four-design", "scm-design"),
           ("profile-gpt-5.4", "profile, GPT-5.4 generator"), ("profile-deepseek", "profile, DeepSeek generator")]
TRUE_C, README_C, PROFILE_C = "#272727", "#767676", "#0E6B73"
# the y axis of figures 1 and 2 is the treated unit's Y without treatment, Y_1t^N as the method slide writes it
# (Jun 2026-11-06 "out come is vague", "legend too long", "README is enough")
YN = "Y₁ₜᴺ"                                      # Y₁ₜᴺ as plain characters, side by side

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 17, "axes.titlesize": 17, "axes.labelsize": 17, "xtick.labelsize": 17, "ytick.labelsize": 17,
    "legend.fontsize": 17, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 1.0,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / f"{name}.alignment.json"),
                                       overlay_svg=str(OUT / f"{name}.alignment.svg"), tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def panel(case):
    d = pd.read_csv(DATA / f"{case}_panel.csv")
    return d.pivot(index="period", columns="unit_id", values="Y_observed")


def truth(case):
    t = pd.read_csv(DATA / "ground_truth.csv")
    return t[t.dgp == case].set_index("period")["Y0_true"]


def readme_path(case):
    """Synthetic path of the README Section 3 weights (her_section3.R) over periods 1 to 60."""
    p = subprocess.run([RSCRIPT, str(HERE / "her_section3.R"), str(DATA / f"{case}_panel.csv"),
                        str(DATA / "ground_truth.csv"), case], capture_output=True, text=True, check=True)
    w = json.loads(p.stdout.strip().splitlines()[-1])["weights"]
    y = panel(case)
    return sum(wt * y[d] for d, wt in w.items())


def answers(case):
    """{(format, model): counterfactual series over 41..60} from the last record of each case."""
    out = {}
    for fmt, folder in MOD.record_folders("1.0").items():
        if fmt not in dict(FORMATS):                 # the deck's arms only; the two-choice scm-design is not in it
            continue
        for p in sorted(folder.glob("*.jsonl")):
            last = {}
            for line in p.read_text().splitlines():
                r = json.loads(line)
                last[r["prompt_id"]] = r
            for r in last.values():
                if r["case"] != case or not r["ok"] or r["model"].startswith(DROPPED_SOLVER):
                    continue
                cf = r["answer"]["prediction"]["counterfactual"]
                out[(fmt, r["model"])] = pd.Series({int(k): v for k, v in cf.items()}).sort_index()
    return out


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
    for ax, case, title in zip(axes, ("dgp1", "dgp2"), ("DGP 1, shared factors", "DGP 2, no shared shocks")):
        tr, sp = truth(case), readme_path(case)
        ax.axvspan(40.5, 60.5, color="#EEF0EC", zorder=0)
        ax.plot(tr.index, tr.values, color=TRUE_C, label="true")
        ax.plot(sp.index, sp.values, color=README_C, linestyle="--", label="README")
        ax.set_title(title, loc="left")
        ax.set_xlabel("period")
        ax.set_xlim(0.5, 60.5)
        ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
    axes[0].set_ylabel(YN)
    axes[1].text(41.5, 0.97, "treated", transform=axes[1].get_xaxis_transform(), va="top", color="#4D4D4D")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, "fig1_dgp_paths")


def fig2():
    tr, rd, ans = truth("dgp1"), readme_path("dgp1"), answers("dgp1")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex=True, sharey=True)
    for ax, (fmt, label) in zip(axes.flat, FORMATS):
        x = list(range(41, 61))
        ax.plot(x, tr.loc[x].values, color=TRUE_C, label="true")
        ax.plot(x, rd.loc[x].values, color=README_C, linestyle="--", label="README")
        if fmt.startswith("profile"):
            s = ans[(fmt, "openai:gpt-5.5")]
            ax.plot(s.index, s.values, color=PROFILE_C, label="all three models")
        else:
            for key, name, col in MODELS:                 # DeepSeek-V4.1-Flash answered no configure or design case
                if (fmt, key) in ans:
                    s = ans[(fmt, key)]
                    ax.plot(s.index, s.values, color=col, linewidth=1.6, label=name)
        ax.set_title(label, loc="left")
        ax.set_xticks([41, 50, 60])
    for ax in axes[1]:
        ax.set_xlabel("period")
    for ax in axes[:, 0]:
        ax.set_ylabel(YN)
    handles = [mpl.lines.Line2D([], [], color=TRUE_C, label="true"),
               mpl.lines.Line2D([], [], color=README_C, linestyle="--", label="README"),
               *[mpl.lines.Line2D([], [], color=c, label=n) for k, n, c in MODELS
                 if any((f, k) in ans for f, _ in FORMATS[:2])],
               mpl.lines.Line2D([], [], color=PROFILE_C, label="profile, all three models")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "fig2_case_dgp1")


def fig3():
    luck = pd.read_csv(MOD.results_folder("1.0") / "luck_replications.csv")
    t41 = pd.read_csv(MOD.results_folder("1.0") / "table_4_1.csv")
    t41 = t41[(t41.case == "dgp1") & ~t41.model.astype(str).str.startswith(DROPPED_SOLVER)]
    rows = ["every factor known", "README Section 3", *[f for f, _ in FORMATS]]
    labels = ["every factor known", "README Section 3", "scm-configure", "scm-design",
              "profile, GPT-5.4", "profile, DeepSeek"]
    y = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    v = luck.att_error_known_factors
    ax.scatter(v[1:], [y["every factor known"]] * (len(v) - 1), s=110, color="#A8A8A8", zorder=3)
    ax.scatter(v[:1], [y["every factor known"]], s=150, color=TRUE_C, zorder=4)
    for ref in ("README Section 3",):
        r = t41[t41.arm == ref]
        ax.scatter(r.att_error, [y[ref]] * len(r), s=130, color=README_C, zorder=3)
    for fmt, _ in FORMATS:
        if fmt.startswith("profile"):
            r = t41[(t41.arm == fmt) & t41.answered]
            assert r.att_error.nunique() == 1, f"{fmt}: the four answers differ"
            ax.scatter(r.att_error.iloc[:1], [y[fmt]], s=130, color=PROFILE_C, zorder=3)
            continue
        for key, name, col in MODELS:
            r = t41[(t41.arm == fmt) & (t41.model == key) & t41.answered]
            if len(r):
                ax.scatter(r.att_error, [y[fmt]], s=130, color=col, zorder=3)
    ax.set_yticks([y[r] for r in rows])
    ax.set_yticklabels(labels)
    ax.axvline(0, color="#CFCECE", linewidth=1.0, zorder=1)
    ax.axhline(y["scm-configure"] + 0.5, color="#E2E2E2", linewidth=1.0, zorder=1)
    ax.set_xlabel("ATT error on DGP 1 (estimate minus the true effect)")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    answered = set(t41[t41.arm.isin(["scm-configure", "scm-four-design"]) & t41.answered].model)
    dot = lambda c, l, s=10: mpl.lines.Line2D([], [], marker="o", linestyle="", color=c, markersize=s, label=l)
    handles = [dot(TRUE_C, "shipped replication"), dot("#A8A8A8", "ten new replications"),
               *[dot(c, n) for k, n, c in MODELS if k in answered],
               dot(PROFILE_C, "profile, all three models")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "fig3_luck")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
