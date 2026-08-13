#!/usr/bin/env python
"""Meeting-ready figures + tables for the latent-recovery study.

Numbers come from latent/results/*.json (stdlib json) and, for the two
per-record distributions, from precomputed CSVs under latent/figures/plotdata/.

Two-env split (neither env is modified):
  prep    -> /data/wang/junh/envs/medrag/bin/python  (pandas + pyarrow, no matplotlib)
             reads the parquets, writes latent/figures/plotdata/*.csv
  render  -> /data/wang/junh/envs/causal/bin/python  (pandas + matplotlib, no pyarrow)
             reads the JSONs + the CSVs, writes latent/figures/*.png|svg and tables.md

  python make_report_figures.py prep       # medrag
  python make_report_figures.py render     # causal
  python make_report_figures.py verify     # either (checks values against FINAL_REPORT)

PHI: figures and tables contain aggregates and numbers only -- no note text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LATENT = HERE.parent
RESULTS = LATENT / "results"
FIGDIR = LATENT / "figures"
PLOTDATA = FIGDIR / "plotdata"

# ---------------------------------------------------------------- palette
INK = "#22252a"
GRID = "#d8dbe0"
QWEN = "#1f4e79"      # primary encoder
BIOL = "#9fb8cd"      # secondary encoder (light)
BAD = "#b45f06"       # the invalid digit-string run / failure
NULLC = "#9aa0a6"     # permutation null
OK = "#2f6f4e"
WARN = "#8a6d1f"
NO = "#a03232"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })


def _tidy(ax, ygrid=True):
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


# ---------------------------------------------------------------- stage 1: prep
def prep():
    """medrag env: parquet -> csv. Everything else is read straight from JSON."""
    import pandas as pd
    PLOTDATA.mkdir(parents=True, exist_ok=True)

    # H2 per-target weight cosines, observed and permutation null, v1 and v2
    frames = []
    for ver, f in (("v1", "n2c2_cross_space.parquet"), ("v2", "n2c2_cross_space_v2.parquet")):
        d = pd.read_parquet(RESULTS / f)[
            ["arm", "i", "cos", "null_cos", "l1", "null_l1", "top1_agree", "null_top1_agree"]
        ].copy()
        d["version"] = ver
        frames.append(d)
    h2 = pd.concat(frames, ignore_index=True)
    h2.to_csv(PLOTDATA / "h2_per_target.csv", index=False)

    # L3 per-mixture copy fractions
    l3 = pd.read_parquet(RESULTS / "l3_generation.parquet")[
        ["mixture_id", "family", "K", "top_weight", "copy_fraction_pooled",
         "copy_fraction_best_donor", "copy_fraction_top_weight_donor",
         "gen_ppl", "cycle_rank", "cycle_cosine"]
    ]
    l3.to_csv(PLOTDATA / "l3_copy_fraction.csv", index=False)

    # L4 per-decode severity + copy fractions
    l4 = pd.read_parquet(RESULTS / "l4_interpolation.parquet")[
        ["pair_id", "alpha", "judge_score", "severity_lr_generated",
         "severity_lr_mixture_latent", "gen_ppl", "copy_fraction_met",
         "copy_fraction_notmet", "copy_fraction_pooled", "copies_the_heavier_source"]
    ]
    l4.to_csv(PLOTDATA / "l4_by_alpha.csv", index=False)

    # H1 grid, flattened from the summary JSON (no parquet needed, emitted for convenience)
    for src, out in (("n2c2_weight_recovery_summary.json", "h1_grid_n2c2.csv"),
                     ("weight_recovery_summary.json", "h1_grid_digitstring.csv")):
        cells = json.loads((RESULTS / src).read_text())["cells"]
        pd.DataFrame(cells).to_csv(PLOTDATA / out, index=False)

    print("wrote:")
    for p in sorted(PLOTDATA.glob("*.csv")):
        print("  ", p, p.stat().st_size, "bytes")


# ---------------------------------------------------------------- loading
def load():
    """Load every artifact a figure or table needs. Works in any env with pandas."""
    import pandas as pd

    def j(name):
        return json.loads((RESULTS / name).read_text())

    D = {
        "h1": j("n2c2_weight_recovery_summary.json"),
        "h1_old": j("weight_recovery_summary.json"),
        "h2v1": j("n2c2_cross_space_summary.json"),
        "h2v2": j("n2c2_cross_space_summary_v2.json"),
        "h2v1_old": j("cross_space_summary.json"),
        "l3": j("l3_summary.json"),
        "l4": j("l4_summary.json"),
        "copy": j("copy_diagnostic.json"),
        "uniq": j("uniqueness_check.json"),
        "l1": j("n2c2_l1_report.json"),
        "l1_old": j("l1_report.json"),
    }
    D["h1_cells"] = pd.DataFrame(D["h1"]["cells"])
    D["h1_old_cells"] = pd.DataFrame(D["h1_old"]["cells"])
    if (PLOTDATA / "h2_per_target.csv").exists():
        D["h2_pt"] = pd.read_csv(PLOTDATA / "h2_per_target.csv")
        D["l3_copy"] = pd.read_csv(PLOTDATA / "l3_copy_fraction.csv")
        D["l4_rows"] = pd.read_csv(PLOTDATA / "l4_by_alpha.csv")
    return D


def cell(df, arm, K, wstar, pool, sigma="sigma0"):
    q = df[(df.arm == arm) & (df.K == K) & (df.wstar == wstar)
           & (df.pool == pool) & (df.sigma_name == sigma)]
    if len(q) != 1:
        raise KeyError(f"{arm} K={K} {wstar} {pool} {sigma}: {len(q)} rows")
    return q.iloc[0]


# ---------------------------------------------------------------- fig 1
def fig1(D):
    import matplotlib.pyplot as plt
    _style()
    df = D["h1_cells"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.9),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- panel A: mean L1 vs K, three pools, diffuse weights, sigma = 0
    FLOOR = 3e-6              # values below this are numerical zero; clipped and flagged
    pools = [("oracle", "o", "-"), ("m50", "s", "-"), ("mall", "D", "")]
    names = {"oracle": "oracle pool\n(true donors handed over)",
             "m50": "m = 50 retrieval pool",
             "mall": "m = all (1,263)"}
    clipped = False
    for arm, col, lw, z in (("biolord", BIOL, 1.2, 2), ("qwen3", QWEN, 1.9, 3)):
        for pool, mk, ls in pools:
            sub = df[(df.arm == arm) & (df.wstar == "dirichlet") & (df.pool == pool)
                     & (df.sigma_name == "sigma0")].sort_values("K")
            if not len(sub):
                continue
            y = sub.mean_l1.clip(lower=FLOOR)
            clipped |= bool((sub.mean_l1 < FLOOR).any())
            axA.plot(sub.K, y, marker=mk, ls=(ls or "none"), color=col,
                     lw=lw, zorder=z, mfc=col if arm == "qwen3" else "white",
                     mec=col, clip_on=False)
            if arm == "qwen3":
                last = sub.iloc[-1]
                axA.annotate(names[pool], (last.K, max(last.mean_l1, FLOOR)),
                             xytext=(6, 0), textcoords="offset points",
                             va="center", ha="left", fontsize=8, color=QWEN)
    axA.set_yscale("log")
    axA.set_xticks([2, 5, 10])
    axA.set_xlim(1.6, 19)
    axA.set_ylim(FLOOR / 1.6, 4)
    axA.set_xlabel("K  (number of planted donors)")
    axA.set_ylabel("mean $L_1(\\hat{w}, w^*)$   (log scale)")
    axA.set_title("A   Recovery error by donor pool\n"
                  "diffuse random weights, $\\sigma$ = 0, 500 replicates/cell", loc="left")
    axA.axhline(0.30, color=NULLC, lw=0.9, ls=":", zorder=1)
    axA.annotate("pre-registered support bar   L$_1$ $\\leq$ 0.30", (1.7, 0.30),
                 xytext=(0, -11), textcoords="offset points", fontsize=7.4, color="#6b7075")
    note = "bold, filled = Qwen3-768 (selected)\nthin, open = BioLORD"
    if clipped:
        note += "\n\noracle at K = 2 is $\\approx$ 1e-15 (numerical zero),\nclipped to the axis floor"
    axA.text(0.015, 0.52, note, transform=axA.transAxes, fontsize=7.4,
             color="#4a5057", va="top")
    _tidy(axA)

    # ---- panel B: the dominant-weight story
    q = cell(df, "qwen3", 5, "spiky", "m50")
    old = cell(D["h1_old_cells"], "qwen3", 5, "spiky", "m50")
    vals = [q.mean_w_max, old.mean_w_max]
    labels = ["n2c2 clinical prose\n(valid run)", "digit-string run\n(invalid — contrast)"]
    cols = [QWEN, BAD]
    xs = [0, 1]
    axB.bar(xs, vals, width=0.52, color=cols, edgecolor=cols, zorder=3)
    axB.bar(xs, vals, width=0.52, facecolor="none", hatch=["", "////"],
            edgecolor=["none", "white"], lw=0, zorder=4)
    axB.axhline(0.9, color=INK, lw=1.2, ls="--", zorder=5)
    axB.annotate("planted $w^*_{max}$ = 0.90", (1.42, 0.9), xytext=(0, 4),
                 textcoords="offset points", ha="right", fontsize=8, color=INK)
    for x, v in zip(xs, vals):
        axB.annotate(f"{v:.3f}", (x, v), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold",
                     color=QWEN if x == 0 else BAD)
    ci = D["h1"]["verdicts"]["qwen3"]["K5_m50_sigma0_spiky_wmax_err_ci"]
    axB.annotate(f"error {0.9 - q.mean_w_max:.3f}\nCI {ci[0]:.4f}–{ci[1]:.4f}\n"
                 f"top-1 donor\nfound 100%",
                 (0, q.mean_w_max), xytext=(0, -42), textcoords="offset points",
                 ha="center", va="top", fontsize=7, color="white", zorder=6)
    axB.annotate(f"error {0.9 - old.mean_w_max:.3f}\noracle gate FAILED\n"
                 f"cosine concentration 0.99",
                 (1, old.mean_w_max), xytext=(0, 22), textcoords="offset points",
                 ha="center", va="bottom", fontsize=7, color=BAD, zorder=6)
    axB.set_xticks(xs)
    axB.set_xticklabels(labels, fontsize=8)
    axB.set_xlim(-0.55, 1.55)
    axB.set_ylim(0, 1.0)
    axB.set_ylabel("recovered $\\hat{w}_{max}$")
    axB.set_title("B   The planted 0.9 weight comes back\nK = 5, m = 50, $\\sigma$ = 0, Qwen3-768",
                  loc="left")
    _tidy(axB)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- fig 2
def fig2(D):
    import matplotlib.pyplot as plt
    import numpy as np
    _style()
    df, dfo = D["h1_cells"], D["h1_old_cells"]
    new = [cell(df, a, 5, "dirichlet", "m50").mean_true_donors_in_pool for a in ("qwen3", "biolord")]
    old = [cell(dfo, a, 5, "dirichlet", "m50").mean_true_donors_in_pool for a in ("qwen3", "biolord")]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, new, w, color=QWEN, edgecolor=QWEN, zorder=3, label="n2c2 clinical prose")
    b2 = ax.bar(x + w / 2, old, w, color="white", edgecolor=BAD, hatch="////", lw=1.2,
                zorder=3, label="digit-string run (invalid)")
    for bars, vals, col in ((b1, new, QWEN), (b2, old, BAD)):
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f} / 5", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold", color=col)
    ax.axhline(5, color=INK, lw=1.2, ls="--", zorder=4)
    ax.annotate("all 5 true donors present", (1.62, 5), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Qwen3-768 (selected)", "BioLORD"])
    ax.set_ylim(0, 6.3)
    ax.set_xlim(-0.6, 1.65)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel("mean number of true donors inside the m = 50 pool")
    ax.set_title("The bottleneck is retrieval, not the weight solver\n"
                 "K = 5 planted donors, diffuse random weights, $\\sigma$ = 0, 500 replicates",
                 loc="left")
    ax.legend(loc="upper left", fontsize=8)
    _tidy(ax)

    q_or = cell(df, "qwen3", 5, "dirichlet", "oracle")
    q_all = cell(df, "qwen3", 5, "dirichlet", "mall")
    q_50 = cell(df, "qwen3", 5, "dirichlet", "m50")
    ax.text(0.40, 0.735, "error lives here", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=9.5, fontweight="bold", color=BAD)
    box = ("same solver, same targets, only the pool changes (Qwen3):\n"
           f"  oracle pool     L$_1$ = {q_or.mean_l1:.1e}    gate passed\n"
           f"  m = all pool    L$_1$ = {q_all.mean_l1:.1e}    0.0000 to 4 dp, LP-unique\n"
           f"  m = 50 pool     L$_1$ = {q_50.mean_l1:.3f}      true donors missing")
    ax.text(0.40, 0.715, box, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.8, family="DejaVu Sans Mono", color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f4f6f8", ec=GRID, lw=0.8))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- fig 3
def fig3(D):
    import matplotlib.pyplot as plt
    import numpy as np
    _style()
    pt = D["h2_pt"]
    pt = pt[pt.arm == "qwen3"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    series, positions, colors = [], [], []
    tick, ticklab = [], []
    for k, (ver, lab) in enumerate((("v1", "v1  (raw concept bags)"),
                                    ("v2", "v2  (NegEx negation filter)"))):
        sub = pt[pt.version == ver]
        base = k * 2.6
        series += [sub["cos"].values, sub["null_cos"].values]
        positions += [base + 0.0, base + 1.0]
        colors += [QWEN, NULLC]
        tick += [base + 0.0, base + 1.0]
        ticklab += [f"observed\n{lab}", "permutation\nnull"]

    parts = ax.violinplot(series, positions=positions, widths=0.8,
                          showextrema=False, showmedians=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.30 if col == NULLC else 0.45)
    for vals, pos, col in zip(series, positions, colors):
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax.vlines(pos, q25, q75, color=col, lw=4, zorder=4)
        ax.plot([pos], [np.mean(vals)], marker="o", ms=6, color="white",
                mec=col, mew=1.8, zorder=5)
        ax.text(pos, 1.035, f"mean {np.mean(vals):.3f}", ha="center", va="bottom",
                fontsize=8.2, color=col,
                fontweight="bold" if col == QWEN else "normal")

    # null IQR bands, drawn under the observed violins
    for k, ver in enumerate(("v1", "v2")):
        s = D["h2v1" if ver == "v1" else "h2v2"]["arms"]["qwen3"]
        lo, hi = s["null_cosine_iqr"]
        ax.add_patch(plt.Rectangle((k * 2.6 - 0.62, lo), 2.05, hi - lo,
                                   fc=NULLC, alpha=0.13, ec="none", zorder=1))
        ax.annotate("null IQR", (k * 2.6 - 0.60, hi), xytext=(0, 2),
                    textcoords="offset points", fontsize=7.4, color="#6b7075")

    ax.set_xticks(tick)
    ax.set_xticklabels(ticklab, fontsize=8)
    ax.set_ylim(-0.16, 1.14)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("per-patient cosine between the two weight vectors")
    ax.set_title("H2 refuted: concept-code space and text-latent space disagree\n"
                 "287 real n2c2 patients, m = 50 donors, weights fitted twice; Qwen3-768",
                 loc="left")
    ax.text(0.5, 0.028,
            "observed mean inside the null IQR in both versions   |   top-donor agreement "
            "1.3$\\times$ null vs the 3$\\times$ bar   $\\Rightarrow$  REFUTED",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=NO,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf3f2", ec="#e8cdc9", lw=0.8))
    _tidy(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- fig 4
def fig4(D):
    import matplotlib.pyplot as plt
    import numpy as np
    _style()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.2, 3.9),
                                        gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # ---- A: copy-fraction histogram
    cf = D["l3_copy"]["copy_fraction_pooled"].values
    st = D["copy"]["l3"]["pooled_over_donors"]
    axA.hist(cf, bins=np.linspace(0, 1, 26), color=QWEN, alpha=0.85, zorder=3)
    axA.axvline(st["median"], color=BAD, lw=1.6, zorder=4)
    axA.annotate(f"median {st['median']:.3f}", (st["median"], axA.get_ylim()[1] * 0.94),
                 xytext=(-5, 0), textcoords="offset points", ha="right",
                 fontsize=8.5, fontweight="bold", color=BAD)
    axA.set_xlabel("share of the generated record's 8-grams\nfound in a donor record")
    axA.set_ylabel("mixtures (of 500)")
    axA.set_title("A   H3 is confounded: the decoder copies", loc="left")
    axA.text(0.03, 0.80,
             f"{D['copy']['l3']['share_pooled_copy_over_80pct']:.0%} of decodes are $\\geq$80% copied\n"
             f"{D['copy']['l3']['share_pooled_copy_over_50pct']:.0%} are $\\geq$50% copied\n"
             "$\\Rightarrow$ perplexity 0.98$\\times$ real and cycle rank 2/1,265\npass trivially",
             transform=axA.transAxes, fontsize=7.8, va="top", color=INK)
    _tidy(axA)

    # ---- B: mean severity vs alpha, two scorers
    ba = D["l4"]["by_alpha"]
    al = np.array([float(a) for a in sorted(ba, key=float)])
    lr = np.array([ba[f"{a:.2f}"]["mean_severity_lr_generated"] for a in al])
    jd = np.array([ba[f"{a:.2f}"]["mean_judge"] for a in al])
    axB.plot(al, lr, marker="o", color=QWEN, zorder=3)
    axB.set_ylim(0, 0.45)
    axB.set_ylabel("logistic scorer:  mean P(ADVANCED-CAD)", color=QWEN)
    axB.tick_params(axis="y", colors=QWEN)
    axB.spines["left"].set_color(QWEN)
    axB2 = axB.twinx()
    axB2.spines["top"].set_visible(False)
    axB2.plot(al, jd, marker="^", ls="--", color=BAD, zorder=3)
    axB2.set_ylim(0, 10)
    axB2.set_ylabel("LLM judge:  mean severity (1–10)", color=BAD)
    axB2.tick_params(axis="y", colors=BAD)
    axB2.spines["right"].set_color(BAD)
    axB2.spines["right"].set_visible(True)
    axB.annotate("logistic on latents", (al[-1], lr[-1]), xytext=(-4, 8),
                 textcoords="offset points", ha="right", fontsize=8, color=QWEN)
    axB2.annotate("LLM judge", (al[-1], jd[-1]), xytext=(-4, -12),
                  textcoords="offset points", ha="right", fontsize=8, color=BAD)
    axB.set_xlabel(r"$\alpha$  (weight on the severe / ADVANCED-CAD-met record)")
    axB.set_xticks(al)
    axB.set_title("B   H4 refuted: severity barely moves", loc="left")
    axB.text(0.04, 0.93,
             "mean per-pair Spearman  0.14 (logistic) / 0.10 (judge)\n"
             "pre-registered support bar: $\\geq$ 0.70",
             transform=axB.transAxes, fontsize=7.8, va="top", color=INK)
    _tidy(axB)

    # ---- C: the switch
    byal = D["copy"]["l4"]["by_alpha"]
    met = np.array([byal[f"{a:.2f}"]["met_median"] for a in al])
    notmet = np.array([byal[f"{a:.2f}"]["notmet_median"] for a in al])
    axC.plot(al, met, marker="o", color=QWEN, zorder=3)
    axC.plot(al, notmet, marker="s", ls="--", color=NULLC, zorder=3)
    axC.annotate("copied from the\nSEVERE record", (al[-1], met[-1]), xytext=(-6, 4),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=8, color=QWEN)
    axC.annotate("copied from the\nNON-SEVERE record", (al[1], notmet[1]),
                 xytext=(6, -6), textcoords="offset points", ha="left", va="top",
                 fontsize=8, color="#6b7075")
    axC.axvline(0.75, color=BAD, lw=1.0, ls=":", zorder=2)
    axC.annotate("the decoder switches\nsource record at $\\alpha \\approx$ 0.75",
                 (0.75, 0.98), xytext=(-7, 0), textcoords="offset points",
                 ha="right", va="top", fontsize=8, color=BAD)
    axC.set_ylim(-0.03, 1.03)
    axC.set_xticks(al)
    axC.set_xlabel(r"$\alpha$  (weight on the severe record)")
    axC.set_ylabel("median 8-gram copy fraction from each source")
    axC.set_title("C   Why: it retrieves, it does not blend", loc="left")
    _tidy(axC)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- fig 5
def fig5(D):
    import matplotlib.pyplot as plt
    _style()
    h1 = D["h1"]["verdicts"]["qwen3"]
    df = D["h1_cells"]
    q50 = cell(df, "qwen3", 5, "dirichlet", "m50")
    v2 = D["h2v2"]["arms"]["qwen3"]
    l3v = D["l3"]["verdict"]
    l4v = D["l4"]["verdict"]

    rows = [
        ("H1", "weight identifiability\nplanted mixtures, 500 reps/cell",
         "NEITHER", WARN,
         f"$\\hat{{w}}_{{max}}$ = {h1['K5_m50_sigma0_spiky_mean_w_max']:.3f} vs planted 0.900   |   top-1 donor 100%\n"
         f"diffuse-weights L$_1$ = {h1['K5_m50_sigma0_dirichlet_mean_l1']:.3f} (bar $\\leq$ 0.30)",
         "dominant 0.9 recovered, diffuse weights missed; with the true\n"
         "donors present, recovery is exact and LP-certified unique"),
        ("H2", "cross-representation\nconsistency, 287 real patients",
         "REFUTED", NO,
         f"weight cosine {v2['mean_weight_cosine']:.3f} — inside the null IQR "
         f"[{v2['null_cosine_iqr'][0]:.2f}, {v2['null_cosine_iqr'][1]:.2f}]\n"
         f"top-donor agreement {v2['top1_ratio_vs_null']:.1f}$\\times$ null (bar 3$\\times$); NegEx-robust",
         "concept-code space and text space tell different mixing\n"
         "stories: representation is a causal-modelling choice"),
        ("H3", "generative sensibility\n500 decoded mixtures",
         "PASSED BUT\nCONFOUNDED", WARN,
         f"perplexity ratio {D['l3']['a_perplexity']['ratio_of_medians_generated_over_real']:.3f}   |   "
         f"cycle rank 2 / 1,265\n"
         f"copy fraction median {D['copy']['l3']['pooled_over_donors']['median']:.3f}",
         "the stand-in decoder reproduces a donor near-verbatim —\n"
         "that passes both metrics without synthesising"),
        ("H4", "semantic interpolation\n50 pairs $\\times$ 5 $\\alpha$ levels",
         "REFUTED", NO,
         f"mean Spearman($\\alpha$, severity) = {l4v['mean_spearman']['lr_generated']:.2f} (logistic) / "
         f"{l4v['mean_spearman']['llm_judge']:.2f} (judge)\nbar $\\geq$ 0.70; scorers agree in sign",
         "the decoder switches source record at $\\alpha \\approx$ 0.75 instead\n"
         "of interpolating — the decoder again, not the latent"),
    ]

    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = [0.010, 0.058, 0.228, 0.383, 0.712]
    heads = ["", "hypothesis", "verdict", "headline numbers", "mechanism (what it localises to)"]
    top, rowh = 0.845, 0.185
    for x, h in zip(xs, heads):
        ax.text(x, top + 0.055, h, fontsize=8.6, fontweight="bold", color="#5a6067")
    ax.plot([0.005, 0.995], [top + 0.035, top + 0.035], color=INK, lw=1.0)

    for i, (tag, what, verdict, vcol, nums, mech) in enumerate(rows):
        y = top - i * rowh
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((0.005, y - rowh + 0.045), 0.99, rowh,
                                       fc="#f6f7f9", ec="none", zorder=0))
        ax.text(xs[0], y, tag, fontsize=13, fontweight="bold", color=INK, va="top")
        ax.text(xs[1], y, what, fontsize=8.4, va="top", color=INK)
        ax.text(xs[2], y, verdict, fontsize=9.2, fontweight="bold", color=vcol, va="top")
        ax.text(xs[3], y, nums, fontsize=8.2, va="top", color=INK)
        ax.text(xs[4], y, mech, fontsize=8.2, va="top", color=INK)
        ax.plot([0.005, 0.995], [y - rowh + 0.045, y - rowh + 0.045],
                color=GRID, lw=0.7, zorder=1)

    ax.text(0.005, 0.055,
            "Bottom line: the causal geometry survives the text latent — a planted 0.9 mixing weight comes back as "
            f"{h1['K5_m50_sigma0_spiky_mean_w_max']:.3f}, and with the true donors\npresent recovery is exact "
            "(L$_1$ = 0.0000 to 4 dp) with an LP-certified unique solution. Every failure localises to an off-the-shelf "
            "component: retrieval\n"
            f"({q50.mean_true_donors_in_pool:.2f} of 5 true donors found), representation choice (H2), and the stand-in "
            "decoder (H3/H4) — exactly the three pieces the proposed system would learn.",
            fontsize=8.8, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.55", fc="#f0f4f8", ec="#c9d6e2", lw=0.9))
    ax.set_title("Pre-registered battery — scoreboard\n"
                 "n2c2 2018 Track 1: 288 patients, 1,264 clinical narratives; encoder Qwen3-Embedding-8B (768-d MRL); "
                 "thresholds frozen before any run",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- tables
def _md(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def tables_md(D) -> str:
    df, dfo = D["h1_cells"], D["h1_old_cells"]
    h1 = D["h1"]["verdicts"]["qwen3"]
    u = D["uniq"]["aggregate"]
    parts = []
    parts.append("# Key tables — latent recovery battery (n2c2, 2026-08-11)\n")
    parts.append("Generated by `latent/scripts/make_report_figures.py`; every value read from "
                 "`latent/results/`. Numbers only — no note text (PHI).\n")

    # T1 decision cell
    q = cell(df, "qwen3", 5, "spiky", "m50")
    parts.append("## T1 — H1 decision cell (K = 5, m = 50, sigma = 0, Qwen3-768)\n")
    parts.append(_md([
        ["dominant 0.9: recovered w_max",
         f"**{h1['K5_m50_sigma0_spiky_mean_w_max']:.3f}** (err {h1['K5_m50_sigma0_spiky_wmax_err']:.3f}, "
         f"CI {h1['K5_m50_sigma0_spiky_wmax_err_ci'][0]:.4f}–{h1['K5_m50_sigma0_spiky_wmax_err_ci'][1]:.4f})",
         "support err <= 0.10 — PASS (6x inside)"],
        ["dominant-weight: top-1 donor found", f"**{h1['K5_m50_sigma0_spiky_top1']:.0%}**", "support >= 90% — PASS"],
        ["diffuse: mean L1(w_hat, w*)",
         f"{h1['K5_m50_sigma0_dirichlet_mean_l1']:.3f} "
         f"(CI {h1['K5_m50_sigma0_dirichlet_l1_ci'][0]:.3f}–{h1['K5_m50_sigma0_dirichlet_l1_ci'][1]:.3f})",
         "support <= 0.30 — MISS (refute bar far away)"],
        ["diffuse: top-1 donor found", f"{h1['K5_m50_sigma0_dirichlet_top1']:.1%}", "refute < 60% — not refuted"],
        ["formal verdict", f"**{h1['verdict']}**", "dominant-weight supported, diffuse missed"],
    ], ["metric", "value", "pre-registered bar"]))

    # T2 full sigma0 grid
    # "dirichlet"/"spiky" are the result files' internal keys; display plain names
    WNAME = {"dirichlet": "diffuse", "spiky": "dominant-0.9"}
    parts.append("\n## T2 — full H1 grid, sigma = 0 (both encoders, both weight families)\n")
    rows = []
    for arm in ("qwen3", "biolord"):
        for K in (2, 5, 10):
            for w in ("dirichlet", "spiky"):
                for pool in ("oracle", "m50", "mall"):
                    s = df[(df.arm == arm) & (df.K == K) & (df.wstar == w)
                           & (df.pool == pool) & (df.sigma_name == "sigma0")]
                    if not len(s):
                        continue
                    s = s.iloc[0]
                    rows.append([arm, K, WNAME[w], pool, int(s.N), f"{s.mean_l1:.3e}",
                                 f"{s.top1_recovery:.1%}", f"{s.mean_w_max:.4f}",
                                 f"{s.mean_true_donors_in_pool:.3f}",
                                 f"{s.frac_all_true_donors_in_pool:.1%}"])
    parts.append(_md(rows, ["arm", "K", "weights", "pool", "N", "mean L1", "top-1",
                            "mean w_max", "true donors in pool", "all K in pool"]))

    # T3 localisation
    parts.append("\n## T3 — where the H1 error lives (Qwen3, K = 5, diffuse weights, sigma = 0)\n")
    rows = []
    for pool, lab in (("oracle", "oracle (true donors handed over)"),
                      ("mall", "m = all (1,263 candidates)"),
                      ("m50", "m = 50 (cosine retrieval)")):
        s = cell(df, "qwen3", 5, "dirichlet", pool)
        rows.append([lab, f"{s.mean_l1:.3e}", f"{s.mean_true_donors_in_pool:.3f} / 5",
                     f"{s.frac_all_true_donors_in_pool:.1%}", f"{s.top1_recovery:.1%}"])
    parts.append(_md(rows, ["pool", "mean L1", "true donors present", "all 5 present", "top-1"]))
    fa_d = cell(df, "qwen3", 5, "dirichlet", "m50").frac_all_true_donors_in_pool
    fa_s = cell(df, "qwen3", 5, "spiky", "m50").frac_all_true_donors_in_pool
    parts.append(f"\nShare of replicates with all 5 true donors in the m = 50 pool: "
                 f"{fa_d:.1%} (diffuse) / {fa_s:.1%} (dominant-weight). The FINAL_REPORT's "
                 f"figures are the diffuse family: 5/500 = 1.0% have all 5, and the 2.66/5 "
                 f"comes from that same diffuse cell; the dominant-weight family is 0% exact.\n")
    og = D["h1"]["verdicts"]["oracle_gate"]
    parts.append(f"\nOracle gate: max mean L1 over all oracle sigma=0 cells = "
                 f"{og['qwen3']['max_mean_l1_over_oracle_sigma0_cells']:.3e} (qwen3) / "
                 f"{og['biolord']['max_mean_l1_over_oracle_sigma0_cells']:.3e} (biolord), "
                 f"threshold {og['qwen3']['threshold']} — PASSED both arms.\n")

    # T4 digit-string contrast
    parts.append("\n## T4 — falsifiability: the invalid digit-string run (same test, bad representation)\n")
    rows = []
    for arm in ("qwen3", "biolord"):
        n_s = cell(df, arm, 5, "spiky", "m50")
        o_s = cell(dfo, arm, 5, "spiky", "m50")
        n_d = cell(df, arm, 5, "dirichlet", "m50")
        o_d = cell(dfo, arm, 5, "dirichlet", "m50")
        rows.append([arm, f"{n_s.mean_w_max:.3f}", f"{o_s.mean_w_max:.3f}",
                     f"{n_d.mean_true_donors_in_pool:.2f} / 5", f"{o_d.mean_true_donors_in_pool:.2f} / 5",
                     f"{n_d.mean_l1:.3f}", f"{o_d.mean_l1:.3f}"])
    parts.append(_md(rows, ["arm", "w_max (n2c2 prose)", "w_max (digit strings)",
                            "donors found (prose)", "donors found (digits)",
                            "diffuse L1 (prose)", "diffuse L1 (digits)"]))
    parts.append(f"\nOracle gate on the digit-string run: FAILED "
                 f"({D['h1_old']['verdicts']['oracle_gate']['qwen3']['max_mean_l1_over_oracle_sigma0_cells']:.4f} "
                 f"> 0.01). Neighbour-cosine concentration (mean top-3 cosine): "
                 f"{D['l1_old']['neighbour_profile_overlap']['qwen3_768']['mean_top3_cosine']:.4f} (digit strings) vs "
                 f"{D['l1']['neighbour_profile_overlap']['qwen3_768']['mean_top3_cosine']:.4f} (n2c2 prose).\n")

    # T5 H2
    parts.append("\n## T5 — H2 cross-representation consistency (287 real patients, m = 50)\n")
    rows = []
    for ver, key in (("v1 (raw bags)", "h2v1"), ("v2 (NegEx filtered)", "h2v2")):
        for arm in ("qwen3", "biolord"):
            s = D[key]["arms"][arm]
            rows.append([ver, arm, f"{s['mean_weight_cosine']:.3f}",
                         f"[{s['weight_cosine_ci'][0]:.3f}, {s['weight_cosine_ci'][1]:.3f}]",
                         f"{s['null_mean_cosine']:.3f}",
                         f"[{s['null_cosine_iqr'][0]:.3f}, {s['null_cosine_iqr'][1]:.3f}]",
                         f"{s['top1_agreement']:.1%}", f"{s['top1_ratio_vs_null']:.2f}x",
                         s["verdict"]])
    parts.append(_md(rows, ["version", "arm", "mean weight cosine", "95% CI", "null mean",
                            "null IQR", "top-1 agreement", "vs null", "verdict"]))
    parts.append(f"\nNegation cleanup removed 21.4% of concept mass; verdict unchanged. "
                 f"Negative control (random bags): mean cosine 0.313 / 0.295 — the machinery "
                 f"returns 'refuted' on noise.\n")

    # T6 uniqueness
    parts.append("\n## T6 — uniqueness certificates (20 pseudo-targets, K = 5, Qwen3-768)\n")
    m = u["mall"]
    lp = m["lp_ranges"]
    parts.append(_md([
        ["exact fit feasible", f"{m['n_exact_fit_feasible']} / {m['n_targets']}",
         "a perfect convex fit exists in the m = all pool"],
        ["unconstrained affine solution dim", f"{m['affine_solution_dim'][0]}",
         "494-dimensional before nonnegativity"],
        ["QP strictly convex (m = all)", f"{m['n_qp_strictly_convex']} / {m['n_targets']}",
         f"Gram min eig {m['gram_min_eig_range'][0]:.1e} — not strictly convex"],
        ["LP coords with nonzero range", f"{lp['n_coords_with_nonzero_range']} / {lp['total_coords_probed']}",
         f"max range {lp['max_range_over_everything']:.1e} — every coordinate pinned"],
        ["donors that can carry weight", f"{lp['n_donors_that_can_carry_weight'][0]}",
         "= exactly the K = 5 true donors"],
        ["max off-support mass", f"{m['max_offsupport_mass']:.1e}", "nonnegativity collapses the polytope to a point"],
        ["warm restarts agreeing", f"{D['uniq']['n_restarts_per_target']} + 1 = 11",
         f"max pairwise L1 {m['restarts']['max_pairwise_l1_over_targets']:.1e}; "
         f"max L1 to w* {m['restarts']['max_l1_to_wstar_over_targets']:.1e}"],
        ["m = 50 pool, same targets", f"exact fit feasible {u['m50']['n_exact_fit_feasible']} / 20",
         f"strictly convex {u['m50']['n_qp_strictly_convex']} / 20 — unique but uniquely wrong "
         f"(mean L1 to w* {u['m50']['restarts']['mean_l1_to_wstar_over_targets']:.3f})"],
    ], ["certificate", "value", "reading"]))

    # T7 L3/L4
    parts.append("\n## T7 — H3 / H4 decoder stages\n")
    l3, l4, cp = D["l3"], D["l4"], D["copy"]
    parts.append(_md([
        ["H3 perplexity (median generated / median real)",
         f"{l3['a_perplexity']['generated']['median']:.2f} / "
         f"{l3['a_perplexity']['real_records_full']['median']:.2f} = "
         f"{l3['a_perplexity']['ratio_of_medians_generated_over_real']:.3f}", "<= 1.5x — PASS"],
        ["H3 cycle rank of z* (median of 1,265)", f"{l3['b_cycle_consistency']['median_rank']:.0f}",
         f"top 5% rate {l3['b_cycle_consistency']['top_5pct_rate']:.1%} — PASS"],
        ["H3 similar-mixtures / similar-notes (Spearman)",
         f"{l3['c_similar_mixtures_similar_notes']['sampled_pairs']['spearman_rho']:.3f}",
         f"p = {l3['c_similar_mixtures_similar_notes']['sampled_pairs']['p_value']:.1e} — PASS"],
        ["copy diagnostic (8-gram overlap, pooled over donors)",
         f"median {cp['l3']['pooled_over_donors']['median']:.3f} "
         f"(IQR {cp['l3']['pooled_over_donors']['q25']:.3f}–{cp['l3']['pooled_over_donors']['q75']:.3f})",
         f"{cp['l3']['share_pooled_copy_over_80pct']:.1%} of decodes >= 80% copied — H3 confounded"],
        ["H4 mean Spearman(alpha, severity), logistic",
         f"{l4['verdict']['mean_spearman']['lr_generated']:.3f}", "support >= 0.70 — REFUTED"],
        ["H4 mean Spearman(alpha, severity), LLM judge",
         f"{l4['verdict']['mean_spearman']['llm_judge']:.3f}", "refute < 0.30 on both — REFUTED"],
        ["H4 copy fraction from the severe record, alpha 0.50 -> 0.75",
         f"{cp['l4']['by_alpha']['0.50']['met_median']:.3f} -> {cp['l4']['by_alpha']['0.75']['met_median']:.3f}",
         "the decoder switches records instead of blending"],
    ], ["metric", "value", "bar / reading"]))

    # T8 scoreboard
    parts.append("\n## T8 — scoreboard\n")
    parts.append(_md([
        ["H1 weight identifiability", "neither (dominant-weight supported, diffuse missed)",
         f"w_max {h1['K5_m50_sigma0_spiky_mean_w_max']:.3f} vs 0.900; diffuse L1 "
         f"{h1['K5_m50_sigma0_dirichlet_mean_l1']:.3f}",
         "exact + LP-unique with true donors present; residual error is retrieval"],
        ["H2 cross-representation", "refuted",
         f"cosine {D['h2v2']['arms']['qwen3']['mean_weight_cosine']:.3f} inside null IQR; top-1 "
         f"{D['h2v2']['arms']['qwen3']['top1_ratio_vs_null']:.1f}x null",
         "representation choice is load-bearing for causal interpretation"],
        ["H3 generative sensibility", "passed but confounded — do not quote",
         f"ppl ratio {l3['a_perplexity']['ratio_of_medians_generated_over_real']:.3f}; copy median "
         f"{cp['l3']['pooled_over_donors']['median']:.3f}", "the stand-in decoder copies a donor"],
        ["H4 semantic interpolation", "refuted",
         f"Spearman {l4['verdict']['mean_spearman']['lr_generated']:.2f} / "
         f"{l4['verdict']['mean_spearman']['llm_judge']:.2f}",
         "decoder switches source records at alpha ~ 0.75"],
    ], ["hypothesis", "verdict", "headline numbers", "mechanism"]))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- verify
def verify(D, echo=True):
    """Check every number the FINAL_REPORT quotes against the artifacts."""
    df, dfo = D["h1_cells"], D["h1_old_cells"]
    h1 = D["h1"]["verdicts"]["qwen3"]
    v1, v2 = D["h2v1"]["arms"]["qwen3"], D["h2v2"]["arms"]["qwen3"]
    checks = []

    def chk(name, quoted, actual, fmt="{:.4g}", tol=None):
        ok = (abs(quoted - actual) <= (tol if tol is not None else abs(quoted) * 5e-3 + 1e-12))
        checks.append((ok, name, fmt.format(quoted), fmt.format(actual)))

    chk("H1 dominant-weight w_max = 0.884", 0.884, h1["K5_m50_sigma0_spiky_mean_w_max"], tol=5e-4)
    chk("H1 dominant-weight w_max err = 0.016", 0.016, h1["K5_m50_sigma0_spiky_wmax_err"], tol=5e-4)
    chk("H1 err CI lo = .0155", 0.0155, h1["K5_m50_sigma0_spiky_wmax_err_ci"][0], tol=5e-5)
    chk("H1 err CI hi = .0166", 0.0166, h1["K5_m50_sigma0_spiky_wmax_err_ci"][1], tol=5e-5)
    chk("H1 dominant-weight top-1 = 100%", 1.0, h1["K5_m50_sigma0_spiky_top1"], tol=0)
    chk("H1 diffuse L1 = 0.452", 0.452, h1["K5_m50_sigma0_dirichlet_mean_l1"], tol=5e-4)
    checks.append((h1["verdict"].startswith("neither"), "H1 verdict = 'neither'",
                   "neither", h1["verdict"]))
    chk("oracle gate L1 ~ 1.5e-05", 1.5e-5,
        D["h1"]["verdicts"]["oracle_gate"]["qwen3"]["max_mean_l1_over_oracle_sigma0_cells"], tol=1e-6)
    mall = cell(df, "qwen3", 5, "dirichlet", "mall").mean_l1
    checks.append((round(mall, 4) == 0.0, "m=all L1 = 0.0000 (4 dp)", "0.0000", f"{mall:.3e}"))
    chk("m=50 true donors = 2.66/5", 2.66,
        cell(df, "qwen3", 5, "dirichlet", "m50").mean_true_donors_in_pool, tol=5e-3)
    fa = cell(df, "qwen3", 5, "dirichlet", "m50").frac_all_true_donors_in_pool
    checks.append((abs(fa - 0.01) < 1e-9, "m=50 '1.0% of replicates have all 5' (diffuse)", "1.0%", f"{fa:.1%}"))
    fas = cell(df, "qwen3", 5, "spiky", "m50").frac_all_true_donors_in_pool
    checks.append((fas == 0.0, "m=50 all-5 rate, dominant-weight", "0.0%", f"{fas:.1%}"))
    chk("uniqueness affine dim = 494", 494, D["uniq"]["aggregate"]["mall"]["affine_solution_dim"][0], tol=0)
    chk("uniqueness off-support <= 1.7e-12", 1.7e-12,
        D["uniq"]["aggregate"]["mall"]["max_offsupport_mass"], tol=2e-14)
    checks.append((D["uniq"]["n_restarts_per_target"] + 1 == 11, "11 warm starts", "11",
                   str(D["uniq"]["n_restarts_per_target"] + 1)))
    chk("digit-string donors = 0.88/5", 0.88,
        cell(dfo, "qwen3", 5, "dirichlet", "m50").mean_true_donors_in_pool, tol=5e-3)
    chk("digit-string w_max = 0.589", 0.589,
        cell(dfo, "qwen3", 5, "spiky", "m50").mean_w_max, tol=5e-4)
    checks.append((not D["h1_old"]["verdicts"]["oracle_gate"]["qwen3"]["passed"],
                   "digit-string oracle gate failed", "failed",
                   "failed" if not D["h1_old"]["verdicts"]["oracle_gate"]["qwen3"]["passed"] else "passed"))
    chk("concentration 0.99 (digit strings)", 0.99,
        D["l1_old"]["neighbour_profile_overlap"]["qwen3_768"]["mean_top3_cosine"], tol=5e-3)
    chk("concentration 0.71 (n2c2 prose)", 0.71,
        D["l1"]["neighbour_profile_overlap"]["qwen3_768"]["mean_top3_cosine"], tol=5e-3)
    chk("H2 v1 cosine = 0.425", 0.425, v1["mean_weight_cosine"], tol=5e-4)
    chk("H2 v2 cosine = 0.422", 0.422, v2["mean_weight_cosine"], tol=5e-4)
    inside = v2["null_cosine_iqr"][0] <= v2["mean_weight_cosine"] <= v2["null_cosine_iqr"][1]
    checks.append((inside, "H2 v2 mean inside null IQR", "inside",
                   f"{v2['null_cosine_iqr'][0]:.3f}-{v2['null_cosine_iqr'][1]:.3f}"))
    chk("H2 top-1 ratio = 1.3x null", 1.3, v2["top1_ratio_vs_null"], tol=0.04)
    chk("H3 ppl ratio = 0.983", 0.983,
        D["l3"]["a_perplexity"]["ratio_of_medians_generated_over_real"], tol=5e-4)
    chk("H3 cycle rank = 2 / 1,265", 2.0, D["l3"]["b_cycle_consistency"]["median_rank"], tol=0)
    chk("copy median = 0.835 (84%)", 0.835, D["copy"]["l3"]["pooled_over_donors"]["median"], tol=5e-4)
    checks.append((D["l4"]["verdict"]["verdict"] == "refuted", "H4 refuted", "refuted",
                   D["l4"]["verdict"]["verdict"]))
    chk("corpus rows = 1,264", 1264, D["h1"]["n_corpus_rows"], tol=0)
    chk("patients = 288", 288, D["h1"]["n_patients"], tol=0)
    chk("H2 targets = 287", 287, D["h2v2"]["n_targets"], tol=0)

    if echo:
        bad = [c for c in checks if not c[0]]
        for ok, name, q, a in checks:
            print(f"[{'ok ' if ok else 'MISMATCH'}] {name:<45} report={q:<12} artifact={a}")
        print(f"\n{len(checks) - len(bad)}/{len(checks)} checks pass; {len(bad)} mismatch(es)")
    return checks


# ---------------------------------------------------------------- driver
FIGS = {"fig1_h1_recovery": fig1, "fig2_retrieval": fig2, "fig3_h2_null": fig3,
        "fig4_decoder": fig4, "fig5_scoreboard": fig5}


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGDIR.mkdir(parents=True, exist_ok=True)
    D = load()
    for name, fn in FIGS.items():
        f = fn(D)
        for ext in ("png", "svg"):
            f.savefig(FIGDIR / f"{name}.{ext}")
        plt.close(f)
        print("wrote", FIGDIR / f"{name}.png", "+ .svg")
    (FIGDIR / "tables.md").write_text(tables_md(D))
    print("wrote", FIGDIR / "tables.md")
    print()
    verify(D)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "render"
    if what == "prep":
        prep()
    elif what == "render":
        render()
    elif what == "verify":
        verify(load())
    elif what == "tables":
        (FIGDIR / "tables.md").write_text(tables_md(load()))
        print("wrote", FIGDIR / "tables.md")
    else:
        raise SystemExit(f"unknown stage {what!r}; use prep | render | verify | tables")
