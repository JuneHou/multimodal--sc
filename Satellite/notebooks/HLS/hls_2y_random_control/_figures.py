#!/usr/bin/env python3
"""Figures and effect maps for the hls_2y_random_control runs.

Called by each method script at the end of its run, so one run of a method
script is that method's whole deliverable. Also usable directly:

    python _figures.py --method ascm --cadence monthly

LAYOUTS follow scripts/validation/ascm_r_package/03b_summarize_bysite.py, which
already produced the reference figures. What changes here is the UNIT: those
figures put one panel per DESIGN and averaged the 89 sites inside it. These put
one panel per TREATMENT SITE, grouped into pages by dominant land cover
(forest 77, agriculture 6, developed 6), because that is the comparison asked
for.

The three months after Helene -- the event date to 2025-01-01 -- are shaded in
every time-series panel, and the caterpillar is computed on that window.

MAPS are built for ASCM and TASC only. SHC has no geographic controls at all:
its donors are earlier blocks of the treated site's own series, so there is
nothing to put on a map that its figures do not already say.

SHC's own figures (shc_figure1/run_shc/SHC_SPEC) moved to
scripts/validation/hls_2y_shc/_figures.py on 2026-11-11, alongside a single
consolidated test results/validation/hls_flooding_128/hls_2y_shc/ folder -- all three design pipelines
produced byte-identical SHC output (SHC has no donors), so one copy replaces
three. See that file's docstring for why its figure4 omits the season-
matched benchmark this file's figure4 still draws for ASCM/TASC.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CALS = Path("/Users/gaoyujuan/REAP Dropbox/Gao yujuan/Virginia Tech/CALS")
HLS = CALS / "datasets" / "flooding" / "HLS"
ROOT = CALS / "test results" / "validation" / "hls_flooding_128" / "hls_2y_random_control"
PANELS = ROOT / "panels"
DESIGN = "conus_random10"

EVENT = pd.Timestamp("2024-09-27")
DAMAGE_END = pd.Timestamp("2025-01-01")

STRATA = ["forest", "agriculture", "developed"]
PER_PAGE, NCOLS = 24, 4

OBSERVED, PRIMARY, SECONDARY = "#1c1c1a", "#d95f02", "#377eb8"
# The 3-month average gets its own hue and dash so it cannot be mistaken for
# the per-period effect line.
AVG = "#6a3d9a"
DAMAGE_FILL, GRID, MUTED = "#9ecae1", "#dcdad4", "#6b6b66"
SIG, NOTSIG, BAND = "#d95f02", "#7f8c8d", "#b9c6d2"
# Diverging ends for the effect maps: NDVI loss reads brown-red, gain green.
LOSS, MID, GAIN = (178, 24, 43), (247, 247, 247), (26, 152, 80)

# Each method names its counterfactual and effect columns differently. One
# place to say so, so the figure code below is method-agnostic.
SPEC = {
    "ascm": dict(effects="conus_random10_site_effects.csv",
                 summary="conus_random10_site_summary.csv",
                 counterfactual="ascm_counterfactual", effect="ascm_effect",
                 second="scm_counterfactual", second_effect="scm_effect",
                 att="ascm_att_damage", att_full="ascm_att",
                 lower="conformal_damage_lower", upper="conformal_damage_upper",
                 pvalue="conformal_damage_min_p", pre_rmse="ascm_pre_rmse",
                 label="ASCM", second_label="SCM", weights=True,
                 interval="95% CWZ conformal interval",
                 ptest="conformal p",
                 period_lower="conformal_lower",
                 period_upper="conformal_upper",
                 period_p="conformal_p"),
    "tasc": dict(effects="site_effects.csv", summary="site_summary.csv",
                 counterfactual="counterfactual", effect="effect",
                 second=None, second_effect=None,
                 att="att_damage", att_full="att",
                 lower="att_lower", upper="att_upper",
                 pvalue="placebo_p", pre_rmse="pre_rmse_donor_only",
                 label="TASC", second_label=None, weights=False,
                 interval="95% from pre-period residual spread",
                 ptest="donor placebo p",
                 period_lower=None, period_upper=None, period_p=None),
}


# ----------------------------------------------------------------- inputs
# NLCD's own palette, so a marker's land-cover label matches the colour the
# basemap paints underneath it.
NLCD_COLOURS = {"water": "#466b9f", "developed": "#d99282",
                "barren": "#b3ac9f", "forest": "#68ab5f",
                "shrub_grass": "#ccb879", "agriculture": "#ab6c28",
                "wetlands": "#b8d9eb"}
NLCD_LABELS = {"water": "Open water", "developed": "Developed",
               "barren": "Barren", "forest": "Forest",
               "shrub_grass": "Shrub / grassland",
               "agriculture": "Planted / cultivated",
               "wetlands": "Wetlands"}


def landcover_of():
    """Dominant NLCD class and its share, for treated sites AND donors.

    Treated come from the site-characteristics table (percent); donors are
    derived from the same NLCD extraction that drove the matching
    (candidate_features, fractions), so both describe the same classification
    over the same footprint. Donors whose extraction produced no land cover
    at all are reported as unknown rather than guessed at.
    """
    out = {}
    treated = pd.read_csv(HLS / "treatment" / "site_characteristics"
                          / "treatment_site_landcover.csv")
    pct_col = "dominant %"
    for r in treated.itertuples(index=False):
        row = dict(zip(treated.columns, r))
        key = str(row["dominant"]).replace(" ", "_")
        out[row["site_id"]] = (key, round(float(row[pct_col]), 1))

    donors = pd.read_csv(HLS / "_work" / "site_selection"
                         / "candidate_features.csv")
    shares = [c for c in donors.columns if c.startswith("lc_")]
    # Rows with no land cover at all would make idxmax raise; they are a QC
    # failure in the extraction, not a class, so they stay unknown.
    usable = donors[donors[shares].notna().any(axis=1)]
    dominant = usable[shares].idxmax(axis=1).str.replace("lc_", "",
                                                         regex=False)
    pct = usable[shares].max(axis=1) * 100
    for site, cls, value in zip(usable.site_id, dominant, pct):
        out.setdefault(site, (cls, round(float(value), 1)))
    for site in donors.site_id:
        out.setdefault(site, ("unknown", float("nan")))
    return out


def strata_of():
    """site -> dominant NLCD class, from the treatment site characteristics."""
    lc = pd.read_csv(HLS / "treatment" / "site_characteristics"
                     / "treatment_site_landcover.csv")
    return dict(zip(lc["site_id"], lc["dominant"]))


def coordinates():
    treated = pd.read_csv(HLS / "treatment" / "treatment_sites.csv")
    donors = pd.read_csv(HLS / DESIGN / f"{DESIGN}_control_sites.csv")
    return ({r.site_id: [round(r.latitude, 5), round(r.longitude, 5)]
             for r in treated.itertuples()},
            {r.site_id: [round(r.latitude, 5), round(r.longitude, 5)]
             for r in donors.itertuples()})


def environment_of():
    """site_id -> elevation_m / slope_deg / temp_c, for treated sites AND
    every candidate (donors included) -- the same pre-treatment baseline
    features site selection itself used, read once so the map can show a
    treated-minus-donor difference per clicked donor, not just an aggregate.
    """
    cols = ["site_id", "elevation_m", "slope_deg", "temp_c"]
    treated = pd.read_csv(HLS / "_work" / "site_selection"
                         / "treatment_features.csv", usecols=cols)
    candidates = pd.read_csv(HLS / "_work" / "site_selection"
                            / "candidate_features.csv", usecols=cols)
    out = {}
    for frame in (candidates, treated):   # treated overrides if a site_id collides
        for r in frame.itertuples(index=False):
            out[r.site_id] = dict(elevation_m=r.elevation_m, slope_deg=r.slope_deg,
                                  temp_c=r.temp_c)
    return out


def load(method, cadence):
    spec = SPEC[method]
    tables = ROOT / method / cadence / "tables"
    effects = pd.read_csv(tables / spec["effects"], parse_dates=["date"])
    summary = pd.read_csv(tables / spec["summary"])
    return spec, effects, summary


def season_benchmark(cadence, damage_dates):
    """The raw series with no synthetic control: a difference in CALENDAR time.

    Each post-event period against the period closest to one year (364 days)
    earlier, less the same comparison at the last pre-event period, then the
    untouched donor pool's version of that subtracted. 03b_summarize_bysite.py
    does this with month labels; matching on 364 days instead generalises it to
    the biweekly and weekly grids, where calendar months do not exist.

    A magnitude check, not an estimator -- no inference is attached.
    """
    panel = pd.read_csv(PANELS / f"{DESIGN}_{cadence}_ndvi.csv",
                        parse_dates=["date"])
    treated = panel[panel.candidate == 1].groupby("date")["ndvi"].mean()

    # The panel is ragged, so a plain groupby would average a DIFFERENT set
    # of donors in each period and the benchmark would partly measure which
    # donors were observed. Fix the donor set -- but only over the handful of
    # dates the benchmark actually touches (each damage period, its match a
    # year earlier, and the baseline pair). Demanding completeness over all
    # 77 weekly periods left under 20 donors and no benchmark at all.
    wide = panel[panel.candidate == 0].pivot(
        index="site", columns="date", values="ndvi")
    all_dates = np.array(sorted(wide.columns), dtype="datetime64[ns]")

    def _nearest(target, pool):
        target = np.datetime64(pd.Timestamp(target), "ns")
        i = int(np.argmin(np.abs(pool - target)))
        return pool[i] if abs(pool[i] - target) <= np.timedelta64(20, "D") \
            else None

    pre_dates = [d for d in all_dates if d < np.datetime64(EVENT, "ns")]
    if not pre_dates or not len(damage_dates):
        return None
    needed = {pre_dates[-1]}
    base_then = _nearest(pre_dates[-1] - np.timedelta64(364, "D"), all_dates)
    if base_then is None:
        return None
    needed.add(base_then)
    for d in damage_dates:
        then = _nearest(np.datetime64(pd.Timestamp(d), "ns")
                        - np.timedelta64(364, "D"), all_dates)
        if then is not None:
            needed.update({np.datetime64(pd.Timestamp(d), "ns"), then})
    cols = [pd.Timestamp(d) for d in sorted(needed)]
    complete = wide[wide[cols].notna().all(axis=1)]
    if len(complete) < 20:
        return None
    donors = complete[cols].mean(axis=0)
    dates = np.array(sorted(set(treated.index) & set(donors.index)),
                     dtype="datetime64[ns]")

    def nearest(target):
        target = np.datetime64(pd.Timestamp(target), "ns")
        i = int(np.argmin(np.abs(dates - target)))
        return dates[i] if abs(dates[i] - target) \
            <= np.timedelta64(20, "D") else None

    def event_study(series):
        pre = [d for d in dates if d < np.datetime64(EVENT, "ns")]
        base_now = pre[-1]
        base_then = nearest(base_now - np.timedelta64(364, "D"))
        if base_then is None:
            return None
        base = series[pd.Timestamp(base_now)] - series[pd.Timestamp(base_then)]
        out = []
        for d in damage_dates:
            then = nearest(np.datetime64(pd.Timestamp(d), "ns")
                           - np.timedelta64(364, "D"))
            if then is not None:
                out.append((series[pd.Timestamp(d)]
                            - series[pd.Timestamp(then)]) - base)
        return out or None

    t_vals, d_vals = event_study(treated), event_study(donors)
    if not t_vals or not d_vals:
        return None
    return float(np.mean(t_vals) - np.mean(d_vals))


# ------------------------------------------------------------ small multiples
def _pages(sites):
    for start in range(0, len(sites), PER_PAGE):
        yield start // PER_PAGE, sites[start:start + PER_PAGE]


def _grid(n):
    rows = int(np.ceil(n / NCOLS))
    fig, axes = plt.subplots(rows, NCOLS, figsize=(4.0 * NCOLS, 2.5 * rows),
                             squeeze=False)
    return fig, axes, rows


def _decorate(ax, dates):
    ax.axvline(EVENT, color=MUTED, lw=1, ls=":")
    ax.axvspan(EVENT, DAMAGE_END, color=DAMAGE_FILL, alpha=0.18, lw=0)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def figure1(method, cadence, spec, effects, strata, out):
    """Observed against that site's counterfactual, one panel per site."""
    for stratum in STRATA:
        sites = sorted(s for s in effects.site.unique()
                       if strata.get(s) == stratum)
        if not sites:
            continue
        for page, chunk in _pages(sites):
            fig, axes, rows = _grid(len(chunk))
            for ax, site in zip(axes.ravel(), chunk):
                d = effects[effects.site == site].sort_values("date")
                ax.plot(d.date, d.observed, color=OBSERVED, lw=1.5,
                        label="Observed")
                ax.plot(d.date, d[spec["counterfactual"]], color=PRIMARY,
                        lw=1.4, ls="--", label=spec["label"])
                if spec["second"]:
                    ax.plot(d.date, d[spec["second"]], color=SECONDARY,
                            lw=1.1, ls="-.", label=spec["second_label"])
                _decorate(ax, d.date)
                ax.set_title(site, fontsize=9)
            for ax in axes.ravel()[len(chunk):]:
                ax.axis("off")
            axes[0][0].set_ylabel("NDVI", fontsize=8)
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=len(labels) + 1, fontsize=9, frameon=False)
            fig.suptitle(
                f"Figure 1 - {spec['label']} {cadence}: observed against each "
                f"site's counterfactual\n{stratum} sites "
                f"({len(sites)} total), shading = the three months after Helene",
                fontsize=11)
            fig.tight_layout(rect=(0, 0.035, 1, 0.955))
            suffix = f"_p{page + 1}" if len(sites) > PER_PAGE else ""
            fig.savefig(out / f"figure1_counterfactual_{stratum}{suffix}.png",
                        dpi=150)
            plt.close(fig)


def figure2(method, cadence, spec, effects, summary, strata, out):
    """Observed minus counterfactual per site, with that site's own interval."""
    info = summary.set_index("site")
    for stratum in STRATA:
        sites = sorted(s for s in effects.site.unique()
                       if strata.get(s) == stratum)
        if not sites:
            continue
        for page, chunk in _pages(sites):
            fig, axes, rows = _grid(len(chunk))
            for ax, site in zip(axes.ravel(), chunk):
                d = effects[effects.site == site].sort_values("date")
                ax.plot(d.date, d[spec["effect"]], color=PRIMARY, lw=1.5,
                        marker="o", ms=2.2, label=f"{spec['label']} effect")
                if spec["second_effect"]:
                    ax.plot(d.date, d[spec["second_effect"]], color=SECONDARY,
                            lw=1.0, ls="--", label=f"{spec['second_label']} effect")
                row = info.loc[site]
                # Per-period inference, drawn where the estimator produces it:
                # CWZ conformal gives a bound at every post period, which the
                # jackknife+ this replaced never did.
                if spec["period_lower"] and spec["period_lower"] in d:
                    band = d[d[spec["period_lower"]].notna()]
                    if len(band):
                        ax.fill_between(band.date, band[spec["period_lower"]],
                                        band[spec["period_upper"]],
                                        color=PRIMARY, alpha=0.15, lw=0,
                                        label=spec["interval"])
                else:
                    lo, hi = row.get(spec["lower"]), row.get(spec["upper"])
                    if pd.notna(lo) and pd.notna(hi):
                        ax.fill_between([EVENT, DAMAGE_END], lo, hi,
                                        color="none", edgecolor=PRIMARY,
                                        lw=0.9, ls="--",
                                        label=spec["interval"])
                att = row.get(spec["att"])
                if pd.notna(att):
                    # Deliberately not the effect colour: a different hue AND
                    # a different dash, so the 3-month average never reads as
                    # part of the per-period series.
                    ax.hlines(att, EVENT, DAMAGE_END, color=AVG, lw=2.4,
                              ls=(0, (6, 2)), zorder=6,
                              label="3-month average effect")
                ax.axhline(0, color=MUTED, lw=0.9)
                _decorate(ax, d.date)
                ax.set_title(f"{site}  ATT {att:+.3f}" if pd.notna(att)
                             else site, fontsize=9)
            for ax in axes.ravel()[len(chunk):]:
                ax.axis("off")
            axes[0][0].set_ylabel("Observed minus counterfactual", fontsize=8)
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=len(labels) + 1, fontsize=9, frameon=False)
            fig.suptitle(
                f"Figure 2 - {spec['label']} {cadence}: per-site effect and "
                f"inference\n{stratum} sites ({len(sites)} total), "
                f"shading = the three months after Helene", fontsize=11)
            fig.tight_layout(rect=(0, 0.035, 1, 0.955))
            suffix = f"_p{page + 1}" if len(sites) > PER_PAGE else ""
            fig.savefig(out / f"figure2_effect_inference_{stratum}{suffix}.png",
                        dpi=150)
            plt.close(fig)


def _caterpillar(ax, sub, spec, strata, bench, title, label_counts=True):
    """One facet. With label_counts=False the legend text stays generic,
    because a facetted figure draws ONE legend and per-panel counts in it
    would describe whichever panel happened to be last."""
    sub = sub.sort_values(spec["att"]).reset_index(drop=True)
    y = np.arange(len(sub))
    lo, hi = sub.get(spec["lower"]), sub.get(spec["upper"])
    if lo is not None and hi is not None:
        ax.hlines(y, lo, hi, color=BAND, lw=2.2, zorder=1,
                  label=spec["interval"])
    p = sub.get(spec["pvalue"]) if spec["pvalue"] else None
    sig = p.lt(0.05) if p is not None else pd.Series(False, index=sub.index)
    ax.scatter(sub.loc[sig, spec["att"]], y[sig.to_numpy()], s=26, color=SIG,
               edgecolor="black", lw=.4, zorder=3,
               label=(f"{spec.get('ptest', 'p')} < .05 (n={int(sig.sum())})"
                      if label_counts else f"{spec.get('ptest', 'p')} < .05"))
    ax.scatter(sub.loc[~sig, spec["att"]], y[~sig.to_numpy()], s=26,
               color=NOTSIG, edgecolor="black", lw=.4, zorder=3,
               label=(f"not significant (n={int((~sig).sum())})"
                      if label_counts else "not significant"))
    ax.axvline(0, color=MUTED, lw=1)
    mean = sub[spec["att"]].mean()
    ax.axvline(mean, color=SECONDARY, lw=1.6,
               label=(f"mean ATT {mean:+.4f}" if label_counts
                      else "mean ATT for this panel"))
    if bench is not None:
        ax.axvline(bench, color="black", lw=1.4, ls="--",
                   label=f"season-matched benchmark {bench:+.4f}")
    step = max(1, len(sub) // 22)
    ax.set_yticks(y[::step])
    ax.set_yticklabels(sub["site"][::step], fontsize=6)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    return mean, int(sig.sum())


def figure4(method, cadence, spec, summary, strata, out, effects=None):
    """One figure per PERIOD inside the three months after Helene.

    The old figure 4 collapsed the whole window into one number per site. This
    walks the window period by period -- week by week at the weekly cadence --
    so the shape of the response over time is visible rather than averaged
    away. Each figure holds three panels, one per dominant land cover, on a
    SHARED x-axis so the strata are directly comparable.

    The inference drawn is whatever the estimator produces at that period:
    CWZ conformal bounds and p-values for ASCM, and nothing claimed where a
    method does not produce per-period inference.
    """
    bench = season_benchmark(cadence, sorted(
        effects.query("damage").date.unique())) if effects is not None else None

    frames = out / "figure4_by_period"
    frames.mkdir(parents=True, exist_ok=True)
    # Clear stale period PNGs first: a rerun with fewer/different periods
    # than a previous run otherwise leaves old period-date pairs behind
    # alongside the new ones (found 2026-11-11: 7 stale files from an
    # earlier window in hls_2y_random_control, duplicate period numbers
    # with conflicting dates).
    for stale in frames.glob("figure4_period*.png"):
        stale.unlink()

    damage = effects[effects.damage].copy() if effects is not None else None
    if damage is None or damage.empty:
        print("  no damage-window periods; figure 4 by period skipped")
        return bench

    present = [x for x in STRATA
               if any(strata.get(s) == x for s in damage.site.unique())]
    periods = sorted(damage.date.unique())
    # One shared x-range across every period and panel, so a later week is not
    # silently rescaled against an earlier one.
    span = float(np.nanpercentile(np.abs(damage[spec["effect"]]), 99.5))
    limits = (-span * 1.1, span * 1.1)

    for index, period in enumerate(periods, 1):
        at = damage[damage.date == period]
        fig, axes = plt.subplots(1, len(present),
                                 figsize=(5.0 * len(present), 8.6),
                                 squeeze=False)
        for ax, stratum in zip(axes[0], present):
            sub = at[[strata.get(s) == stratum for s in at.site]].copy()
            sub = sub.sort_values(spec["effect"]).reset_index(drop=True)
            y = np.arange(len(sub))
            lo = sub.get(spec["period_lower"]) if spec["period_lower"] else None
            hi = sub.get(spec["period_upper"]) if spec["period_upper"] else None
            if lo is not None and hi is not None and lo.notna().any():
                ax.hlines(y, lo, hi, color=BAND, lw=2.2, zorder=1,
                          label=spec["interval"])
            pv = sub.get(spec["period_p"]) if spec["period_p"] else None
            has_p = pv is not None and pv.notna().any()
            sig = (pv < 0.05).fillna(False) if has_p \
                else pd.Series(False, index=sub.index)
            if has_p:
                # Only claim significance where the estimator produces a
                # per-period p. TASC and SHC do not, and an empty "p < .05"
                # entry in the legend would imply they had been tested.
                ax.scatter(sub.loc[sig, spec["effect"]], y[sig.to_numpy()],
                           s=28, color=SIG, edgecolor="black", lw=.4,
                           zorder=3, label="conformal p < .05")
            ax.scatter(sub.loc[~sig, spec["effect"]], y[~sig.to_numpy()], s=28,
                       color=NOTSIG, edgecolor="black", lw=.4, zorder=3,
                       label="site effect" if not has_p
                       else "not significant")
            ax.axvline(0, color=MUTED, lw=1)
            mean = float(sub[spec["effect"]].mean())
            ax.axvline(mean, color=AVG, lw=1.8, ls=(0, (6, 2)),
                       label="panel mean")
            if bench is not None:
                ax.axvline(bench, color="black", lw=1.2, ls=":",
                           label=f"season-matched benchmark {bench:+.3f}")
            step = max(1, len(sub) // 22)
            ax.set_yticks(y[::step])
            ax.set_yticklabels(sub["site"][::step], fontsize=6)
            ax.set_xlim(*limits)
            ax.grid(axis="x", color=GRID, lw=.6)
            ax.set_axisbelow(True)
            ax.set_title(f"{stratum} (n={len(sub)})\nmean {mean:+.4f}"
                         + (f", {int(sig.sum())} with p < .05" if has_p
                            else ", no per-period inference"), fontsize=10)
            ax.set_xlabel(f"{spec['label']} effect (NDVI)", fontsize=9)
        axes[0][0].set_ylabel("Treated site, ordered by effect")
        axes[0][-1].legend(fontsize=7.5, loc="lower right", framealpha=.95)
        stamp = pd.Timestamp(period).date()
        fig.suptitle(
            f"Figure 4 - {spec['label']} {cadence}: period {index} of "
            f"{len(periods)} after Helene, starting {stamp}\n"
            f"one period per figure; panels share an x-axis across all "
            f"{len(periods)} periods", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        fig.savefig(frames / f"figure4_period{index:02d}_{stamp}.png", dpi=160)
        plt.close(fig)

    print(f"  figure 4: {len(periods)} per-period figures in "
          f"{frames.name}/")

    # The window as a whole, kept alongside so the three months can still be
    # read as one number per site.
    fig, ax = plt.subplots(figsize=(10, max(5, len(summary) * 0.13)))
    _caterpillar(ax, summary, spec, strata, bench,
                 f"Figure 4 - {spec['label']} {cadence}: the three months "
                 f"after Helene as one average per site")
    ax.set_xlabel(f"Site-specific {spec['label']} ATT (NDVI), three months")
    ax.set_ylabel("Treated site, ordered by effect")
    ax.legend(fontsize=8, loc="lower right", framealpha=.95)
    fig.tight_layout()
    fig.savefig(out / "figure4_site_att_significance.png", dpi=180)
    plt.close(fig)
    return bench


# --------------------------------------------------------------- effect map
def ramp(value, limit):
    """Signed value -> hex on a loss/neutral/gain diverging ramp."""
    if not np.isfinite(value) or limit <= 0:
        return "#bdbdbd"
    t = max(-1.0, min(1.0, value / limit))
    end = GAIN if t >= 0 else LOSS
    f = abs(t)
    rgb = [int(round(MID[i] + (end[i] - MID[i]) * f)) for i in range(3)]
    return "#%02x%02x%02x" % tuple(rgb)


def _diff(env, treated_site, donor_site, key):
    t, d = env.get(treated_site, {}).get(key), env.get(donor_site, {}).get(key)
    return None if t is None or d is None or pd.isna(t) or pd.isna(d) else round(float(t - d), 2)


def effect_map(method, cadence, spec, summary, strata, out):
    treated_xy, donor_xy = coordinates()
    cover = landcover_of()
    env = environment_of()
    att = summary.set_index("site")[spec["att"]]
    limit = float(np.nanpercentile(np.abs(att.to_numpy(float)), 95)) or 1e-6

    lo = summary.set_index("site").get(spec["lower"])
    hi = summary.set_index("site").get(spec["upper"])
    rmse = summary.set_index("site").get(spec["pre_rmse"])
    assigned_n = pd.read_csv(
        HLS / DESIGN / f"{DESIGN}_matched_pairs.csv",
        dtype={"treatment_id": str, "control_id": str}
    ).groupby("treatment_id")["control_id"].nunique()

    treated = {}
    for site, value in att.items():
        if site not in treated_xy:
            continue
        excludes_zero = False
        if lo is not None and hi is not None and pd.notna(lo.get(site)) \
                and pd.notna(hi.get(site)):
            excludes_zero = bool(lo[site] > 0 or hi[site] < 0)
        cls, pct = cover.get(site, ("unknown", float("nan")))
        e = env.get(site, {})
        treated[site] = dict(xy=treated_xy[site], att=round(float(value), 4),
                             colour=ramp(float(value), limit),
                             solid=excludes_zero,
                             stratum=strata.get(site, "unknown"),
                             lc=cls, lc_pct=pct,
                             rmse=(round(float(rmse[site]), 4)
                                   if rmse is not None and pd.notna(rmse.get(site)) else None),
                             n_assigned=int(assigned_n.get(site, 0)),
                             elevation_m=e.get("elevation_m"), slope_deg=e.get("slope_deg"),
                             temp_c=e.get("temp_c"))

    links, donors = {}, {}
    if spec["weights"]:
        # ASCM: real per-donor fitted weights, so the panel gets a value and
        # sign per donor, and the chart has something to plot.
        weights = pd.read_csv(ROOT / method / cadence / "tables"
                              / f"{DESIGN}_site_donor_weights.csv")
        peak = float(weights["ascm_weight"].abs().max()) or 1e-6
        for site, group in weights.groupby("site"):
            rows = []
            for r in group.itertuples():
                if r.donor not in donor_xy:
                    continue
                share = abs(float(r.ascm_weight)) / peak
                rows.append([r.donor, round(float(r.ascm_weight), 4),
                           _diff(env, site, r.donor, "elevation_m"),
                           _diff(env, site, r.donor, "slope_deg"),
                           _diff(env, site, r.donor, "temp_c")])
                prev = donors.get(r.donor, {}).get("share", 0.0)
                cls, pct = cover.get(r.donor, ("unknown", float("nan")))
                donors[r.donor] = dict(xy=donor_xy[r.donor],
                                       share=round(max(prev, share), 4),
                                       lc=cls, lc_pct=pct)
            links[site] = rows
            if site in treated:
                treated[site]["n_fitted"] = len(rows)
        # Donors the design offered but no site weighted: drawn faint, so the
        # pool's geography is visible rather than implied.
        used = set(donors)
        for donor, xy in donor_xy.items():
            if donor not in used:
                cls, pct = cover.get(donor, ("unknown", float("nan")))
                donors[donor] = dict(xy=xy, share=0.0, lc=cls, lc_pct=pct)
    else:
        # TASC (and any other no-per-donor-weight method): the design's
        # FITTED donor pool for this cadence still exists (PANELS/
        # site_donors_<cadence>.csv), just with no individual weight to
        # report -- a factor-model fit does not produce one. weight is sent
        # as null so the page can say so rather than draw a fake bar.
        donor_pool = pd.read_csv(ROOT / "panels" / f"site_donors_{cadence}.csv",
                                 dtype={"site": str, "donor": str})
        for site, group in donor_pool.groupby("site"):
            rows = []
            for r in group.itertuples():
                if r.donor not in donor_xy:
                    continue
                rows.append([r.donor, None,
                           _diff(env, site, r.donor, "elevation_m"),
                           _diff(env, site, r.donor, "slope_deg"),
                           _diff(env, site, r.donor, "temp_c")])
                if r.donor not in donors:
                    cls, pct = cover.get(r.donor, ("unknown", float("nan")))
                    donors[r.donor] = dict(xy=donor_xy[r.donor], share=0.0, lc=cls, lc_pct=pct)
            links[site] = rows
            if site in treated:
                treated[site]["n_fitted"] = len(rows)

    payload = json.dumps({"treated": treated, "donors": donors,
                          "links": links, "limit": round(limit, 4),
                          "lc_colours": NLCD_COLOURS,
                          "lc_labels": NLCD_LABELS,
                          "has_weights": bool(spec["weights"])},
                         separators=(",", ":"))
    html = (MAP_TEMPLATE
            .replace("/*DATA*/", payload)
            .replace("/*TITLE*/", f"{spec['label']} {cadence} effect map")
            .replace("/*METHOD*/", json.dumps(spec["label"]))
            .replace("/*CADENCE*/", json.dumps(cadence)))
    path = out / f"{method}_{cadence}_effect_map.html"
    path.write_text(html, encoding="utf-8")
    return path


MAP_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>/*TITLE*/</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%}
  #map{width:100%;height:100%;background:#f4f4f2}
  .scale{background:rgba(255,255,255,.92);padding:7px 10px;border-radius:7px;
    font:12px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    color:#1c1c1a;box-shadow:0 1px 4px rgba(0,0,0,.18)}
  .scale b{font-weight:600}
  .bar{height:9px;width:170px;margin:5px 0 2px;border-radius:2px;
    background:linear-gradient(to right,#b2182b,#f7f7f7,#1a9850)}
  .ends{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}
  .lcrow{display:flex;align-items:center;gap:6px;margin-top:3px}
  .sw{width:12px;height:12px;border-radius:2px;flex:none;
      border:1px solid rgba(0,0,0,.25)}
  #panel{position:absolute;top:10px;right:10px;width:320px;max-height:calc(100% - 20px);
    overflow-y:auto;background:rgba(255,255,255,.97);border-radius:8px;
    padding:12px 14px;font:12px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    color:#1c1c1a;box-shadow:0 1px 6px rgba(0,0,0,.25);display:none;z-index:1000}
  #panel h3{margin:0 0 4px;font-size:14px}
  #panel .sub{color:#6b6b66;margin-bottom:8px}
  #panel table{width:100%;border-collapse:collapse;font-size:11px;margin-top:6px}
  #panel th,#panel td{padding:2px 4px;text-align:right;white-space:nowrap}
  #panel th:first-child,#panel td:first-child{text-align:left}
  #panel thead tr{border-bottom:1px solid #dcdad4}
  #panel tbody tr:nth-child(even){background:#f7f6f3}
  #panel .barwrap{display:flex;align-items:center;gap:4px}
  #panel .barfill{height:9px;border-radius:2px}
  #panel .close{position:absolute;top:8px;right:10px;cursor:pointer;color:#6b6b66;
    font-weight:600}
  #panel .note{color:#6b6b66;font-size:10.5px;margin-top:6px}
</style>
</head>
<body>
<div id="map"></div>
<div id="panel"></div>
<script>
const DATA = /*DATA*/;
const METHOD = /*METHOD*/, CADENCE = /*CADENCE*/;

const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/";
const map = L.map("map", {preferCanvas:true}).setView([37.5,-90], 4);
// NLCD land cover as the base: the same classification the site covariates
// and the forest / agriculture / developed strata come from, so a site's
// colour can be read against the cover it sits in.
L.tileLayer.wms("https://www.mrlc.gov/geoserver/mrlc_display/wms", {
  layers:"NLCD_2021_Land_Cover_L48", format:"image/png", transparent:true,
  version:"1.1.1", maxZoom:16, opacity:0.55,
  // Held back from full strength: the effect ramp is red-to-green and NLCD's
  // own palette is too, so the base has to recede for the sites to read.
  attribution:"Land cover: MRLC NLCD 2021"}).addTo(map);
// Place names on top, so the land cover stays readable underneath.
L.tileLayer(ESRI + "World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
  {maxZoom:16, opacity:0.85}).addTo(map);

// A site's own dominant NLCD class, with the share of its footprint. Same
// classification as the basemap beneath it, so the two can be read together.
function lcLabel(d){
  const name = DATA.lc_labels[d.lc] || d.lc || "unknown";
  const swatch = DATA.lc_colours[d.lc] || "#999999";
  const pct = (d.lc_pct === null || isNaN(d.lc_pct)) ? ""
              : ` ${d.lc_pct.toFixed(1)}%`;
  return `<span style="display:inline-block;width:9px;height:9px;` +
         `background:${swatch};border:1px solid #fff;` +
         `vertical-align:middle"></span> ${name}${pct}`;
}

const donorLayer = L.layerGroup().addTo(map);
const linkLayer  = L.layerGroup().addTo(map);
const siteLayer  = L.layerGroup().addTo(map);
const donorMarkers = {};

// Donor style carries the weight: heavier donors are larger and more opaque.
function donorStyle(share){
  // White stroke so a donor reads against whatever land cover it sits on.
  return {radius: 2.6 + 5.8*share, color:"#ffffff", weight: 0.9,
          opacity: 0.45 + 0.5*share, fillColor:"#1f4e79",
          fillOpacity: 0.3 + 0.65*share};
}
for (const id in DATA.donors){
  const d = DATA.donors[id];
  const m = L.circleMarker(d.xy, donorStyle(d.share))
    .bindTooltip(`${id} &middot; weight ${d.share.toFixed(3)}<br>` +
                 `${lcLabel(d)}`, {direction:"top"});
  donorMarkers[id] = m;
  donorLayer.addLayer(m);
}

function siteStyle(t){
  // Always a halo; the ring colour carries whether the interval excludes 0.
  return {radius:8, color: t.solid ? "#1c1c1a" : "#ffffff",
          weight: t.solid ? 2.6 : 2, fillColor:t.colour, fillOpacity:1};
}
for (const id in DATA.treated){
  const t = DATA.treated[id];
  const m = L.circleMarker(t.xy, siteStyle(t)).bindTooltip(
    `${id}<br>${lcLabel(t)}<br>ATT ${t.att>=0?"+":""}${t.att.toFixed(4)}`
    + (t.solid ? " (interval excludes 0)" : ""), {direction:"top"});
  m.on("click", e=>{ L.DomEvent.stop(e); select(id); });
  siteLayer.addLayer(m);
}

let selected = null;
const panelEl = document.getElementById("panel");

// Signed value -> a bar width/colour on a loss(red)/gain(green) scale, same
// convention as the site markers themselves.
function diffBar(value, limit){
  if (value === null || value === undefined || isNaN(value)) return "—";
  const t = Math.max(-1, Math.min(1, value / (limit || 1)));
  const colour = t >= 0 ? "#1a9850" : "#b2182b";
  const width = Math.round(Math.abs(t) * 40);
  return `<span class="barwrap"><span class="barfill" style="width:${width}px;` +
         `background:${colour}"></span>${value>=0?"+":""}${value.toFixed(1)}</span>`;
}

function renderPanel(id){
  const t = DATA.treated[id];
  const rows = (DATA.links[id] || []).slice();
  const hasWeights = DATA.has_weights;
  // Order by |weight| when there is one; otherwise just by donor id, since
  // every row carries the same (null) weight.
  rows.sort((a,b)=> hasWeights ? Math.abs(b[1]) - Math.abs(a[1]) : a[0].localeCompare(b[0]));
  const elevLimit = Math.max(1, ...rows.map(r=>Math.abs(r[2]||0)));
  const slopeLimit = Math.max(1, ...rows.map(r=>Math.abs(r[3]||0)));
  const tempLimit = Math.max(2, ...rows.map(r=>Math.abs(r[4]||0)));
  const peak = hasWeights ? Math.max(1e-6, ...rows.map(r=>Math.abs(r[1]))) : 1;

  const chart = rows.length ? `
    <div style="margin-top:8px">
      <div class="sub" style="margin-bottom:3px">${hasWeights ? "Donor weights (ASCM)" : "Donors (fitted pool)"}</div>
      ${rows.map(r=>{
        const d = DATA.donors[r[0]] || {};
        const swatch = DATA.lc_colours[d.lc] || "#999999";
        const lcName = DATA.lc_labels[d.lc] || d.lc || "unknown";
        const bar = hasWeights ? (() => {
          const w = r[1], frac = Math.abs(w)/peak, px = Math.round(frac*160);
          const colour = w >= 0 ? "#1a9850" : "#b2182b";
          return `<span style="width:${px}px;height:9px;background:${colour};border-radius:2px"></span>` +
            `<span style="font-size:10px">${w>=0?"+":""}${w.toFixed(3)}</span>`;
        })() : `<span style="font-size:10px;color:#6b6b66">${lcName}</span>`;
        return `<div style="display:flex;align-items:center;gap:6px;margin:2px 0">` +
          `<span style="width:10px;height:10px;border-radius:2px;flex:none;` +
          `background:${swatch};border:1px solid rgba(0,0,0,.25)" title="${lcName}"></span>` +
          `<span style="width:46px;font-size:10px">${r[0]}</span>` +
          bar + `</div>`;
      }).join("")}
    </div>` : "";

  const table = rows.length ? `
    <table>
      <thead><tr><th>Donor</th>${hasWeights ? "<th>Weight</th>" : ""}
        <th>&Delta;Elev(m)</th><th>&Delta;Slope(&deg;)</th><th>&Delta;Temp(&deg;C)</th>
        <th>Land cover</th></tr></thead>
      <tbody>${rows.map(r=>`<tr><td>${r[0]}</td>` +
        (hasWeights ? `<td>${r[1]>=0?"+":""}${r[1].toFixed(3)}</td>` : "") +
        `<td>${diffBar(r[2], elevLimit)}</td><td>${diffBar(r[3], slopeLimit)}</td>` +
        `<td>${diffBar(r[4], tempLimit)}</td><td>${lcLabel(DATA.donors[r[0]] || {})}</td></tr>`
      ).join("")}</tbody>
    </table>` : `<div class="note">No donors to show for this site.</div>`;

  panelEl.innerHTML = `
    <span class="close" onclick="reset()">&times;</span>
    <h3>${id}</h3>
    <div class="sub">${lcLabel(t)}</div>
    <div>ATT: <b>${t.att>=0?"+":""}${t.att.toFixed(4)}</b>
      ${t.solid ? " (interval excludes 0)" : ""}</div>
    ${t.rmse !== null && t.rmse !== undefined ? `<div>Pre-treatment RMSE: ${t.rmse.toFixed(4)}</div>` : ""}
    <div>Assigned controls: ${t.n_assigned ?? "—"} &middot; fitted: ${t.n_fitted ?? "—"}</div>
    <div>Site elevation ${t.elevation_m!=null?t.elevation_m.toFixed(0)+" m":"—"} &middot; ` +
      `slope ${t.slope_deg!=null?t.slope_deg.toFixed(1)+"°":"—"} &middot; ` +
      `temp ${t.temp_c!=null?t.temp_c.toFixed(1)+"°C":"—"}</div>
    ${!hasWeights ? '<div class="note">This method (factor-model fit) has no per-donor ' +
      'weight -- donors shown are the fitted pool for this cadence, unweighted.</div>' : ""}
    ${table}
    ${chart}
  `;
  panelEl.style.display = "block";
}

function select(id){
  // Clicking the same site again puts the map back: the links are a toggle,
  // not a mode you have to click the background to leave.
  if (selected === id){ reset(); return; }
  selected = id;
  linkLayer.clearLayers();
  const rows = DATA.links[id] || [];
  for (const did in donorMarkers)
    donorMarkers[did].setStyle({opacity:.08, fillOpacity:.05, radius:2});
  const here = DATA.treated[id].xy;
  const bounds = [here];
  rows.forEach(r=>{
    const d = DATA.donors[r[0]];
    if (!d) return;
    bounds.push(d.xy);
    L.polyline([here, d.xy],
      {color:"#1f4e79", weight:1.2, opacity:.45}).addTo(linkLayer);
    const w = r[1] === null ? 1 : (Math.abs(r[1])/DATA.limit > 1 ? 1 : Math.abs(r[1])/DATA.limit);
    donorMarkers[r[0]].setStyle(
      Object.assign(donorStyle(w),
                    {opacity:1, fillOpacity:.9, color:"#ffffff", weight:1.5}))
      .bringToFront();
  });
  siteLayer.eachLayer(l=>l.setStyle({radius:4, fillOpacity:.35, weight:1}));
  siteLayer.eachLayer(l=>{
    if (l.getTooltip().getContent().startsWith(id + " ")){
      l.setStyle(Object.assign(siteStyle(DATA.treated[id]), {radius:10}));
      l.bringToFront();
    }
  });
  if (bounds.length > 1) map.fitBounds(bounds, {padding:[55,55]});
  renderPanel(id);
}

function reset(){
  selected = null;
  linkLayer.clearLayers();
  panelEl.style.display = "none";
  panelEl.innerHTML = "";
  for (const did in donorMarkers)
    donorMarkers[did].setStyle(donorStyle(DATA.donors[did].share));
  for (const id in DATA.treated) {}
  siteLayer.eachLayer(l=>{
    const key = l.getTooltip().getContent().split(" ")[0];
    l.setStyle(siteStyle(DATA.treated[key]));
  });
  map.setView([37.5,-90], 4);
}
map.on("click", reset);

const legend = L.control({position:"bottomleft"});
legend.onAdd = function(){
  const div = L.DomUtil.create("div", "scale");
  const lim = DATA.limit;
  div.innerHTML =
    `<b>${METHOD} ${CADENCE}</b><br>Treated site: NDVI effect,<br>` +
    `three months after Helene<div class="bar"></div>` +
    `<div class="ends"><span>-${lim.toFixed(3)}</span><span>0</span>` +
    `<span>+${lim.toFixed(3)}</span></div>` +
    `<span style="color:#6b6b66">dark ring = interval excludes 0` +
    (Object.keys(DATA.links).length
      ? `<br>blue = donors, size by weight` : ``) +
    `<br>click a site for its donors, click it again to clear</span>`;
  return div;
};
legend.addTo(map);

// Land-cover key for the basemap. Same classes the site labels report, so a
// marker's stated cover can be checked against the pixels around it.
const cover = L.control({position:"bottomright"});
cover.onAdd = function(){
  const div = L.DomUtil.create("div", "scale");
  const rows = Object.keys(DATA.lc_labels).map(k =>
    `<div class="lcrow"><span class="sw" style="background:` +
    `${DATA.lc_colours[k]}"></span>${DATA.lc_labels[k]}</div>`).join("");
  div.innerHTML = `<b>NLCD 2021 land cover</b>${rows}`;
  L.DomEvent.disableClickPropagation(div);
  return div;
};
cover.addTo(map);
</script>
</body>
</html>
"""



# --------------------------------------------------------------------- SHC
# SHC is fitted by its own self-contained script, which draws a handful of
# per-site PNGs. These are the grouped figures: every site, arranged by
# dominant land cover, the same way ASCM and TASC are presented.


def run(method, cadence):
    out_fig = ROOT / method / cadence / "figures"
    out_map = ROOT / method / cadence / "maps"
    out_fig.mkdir(parents=True, exist_ok=True)

    spec, effects, summary = load(method, cadence)
    strata = strata_of()

    print(f"  {method} {cadence}: {summary.site.nunique()} sites, "
          f"{len(effects)} site-period rows")
    figure1(method, cadence, spec, effects, strata, out_fig)
    figure2(method, cadence, spec, effects, summary, strata, out_fig)
    bench = figure4(method, cadence, spec, summary, strata, out_fig, effects)
    print(f"  season-matched benchmark: "
          f"{bench:+.4f}" if bench is not None else "  benchmark unavailable")

    out_map.mkdir(parents=True, exist_ok=True)
    path = effect_map(method, cadence, spec, summary, strata, out_map)
    print(f"  wrote {path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=sorted(SPEC))
    ap.add_argument("--cadence", required=True,
                    choices=["weekly", "biweekly", "monthly"])
    args = ap.parse_args()
    run(args.method, args.cadence)


if __name__ == "__main__":
    main()
