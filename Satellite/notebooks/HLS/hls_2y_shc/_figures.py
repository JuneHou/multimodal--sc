#!/usr/bin/env python3
"""SHC-only figures for the 2-year treatment-history workflow.

CONSOLIDATED HERE on 2026-11-11, alongside 04_shc_hls_2y.py: SHC has no
geographic donors, so the three per-design pipelines' own `shc_figure1`/
`run_shc` (in their `_figures.py`) produced byte-identical site_summary.csv/
site_paths.csv tables -- the only thing that differed was each pipeline's
figure4 overlaying a "season-matched benchmark" line computed from THAT
design's own donor panel (`PANELS/{DESIGN}_{cadence}_ndvi.csv`, via
`season_benchmark()` in the per-pipeline `_figures.py`). That benchmark
compares a raw, no-synthetic-control calendar difference against SHC's own
historical-block fit -- a legitimate comparison, but tied to one design's
donor pool, which this consolidated, donor-free results folder has no
single correct choice of. The benchmark line is THEREFORE NOT drawn here
(consistent with how scripts/validation/hls_flooding_128/hls_3y_shc/_figures.py handles the
same question for the 3-year version) rather than picking one design's
panel arbitrarily and presenting it as "the" SHC benchmark.

    python3 _figures.py --cadence weekly
    python3 _figures.py --cadence monthly biweekly weekly
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

CALS = Path('/Users/gaoyujuan/REAP Dropbox/Gao yujuan/Virginia Tech/CALS')
ROOT = CALS / 'test results' / 'hls_2y_shc'
LANDCOVER = CALS / 'datasets' / 'flooding' / 'HLS' / 'treatment' / 'site_characteristics' / 'treatment_site_landcover.csv'
EVENT = pd.Timestamp('2024-09-27')
DAMAGE_END = pd.Timestamp('2025-01-01')   # the three months the original request asked about
STRATA = ['forest', 'agriculture', 'developed']
PER_PAGE, NCOLS = 24, 4
LC_ORDER = ['forest', 'agriculture', 'shrub_grass', 'developed', 'wetlands', 'barren', 'water']
WTOL = 1e-4
OBSERVED, PRIMARY, SECONDARY = '#1c1c1a', '#d95f02', '#377eb8'
DAMAGE_FILL, GRID, MUTED = '#9ecae1', '#dcdad4', '#6b6b66'
SIG, NOTSIG, BAND = '#d95f02', '#7f8c8d', '#b9c6d2'
SHC_SPEC = dict(att='att_3month', att_full='att_damage', lower='att_lower',
               upper='att_upper', effect='effect', pvalue='placebo_rank_p',
               period_lower=None, period_upper=None, period_p=None,
               label='SHC', interval='95% interval',
               ptest='historical placebo rank p')


def strata_of():
    lc = pd.read_csv(LANDCOVER)
    return dict(zip(lc['site_id'], lc['dominant']))


def _pages(sites):
    for start in range(0, len(sites), PER_PAGE):
        yield start // PER_PAGE, sites[start:start + PER_PAGE]


def _decorate(ax, dates):
    ax.axvline(EVENT, color=MUTED, lw=1, ls=':')
    ax.axvspan(EVENT, DAMAGE_END, color=DAMAGE_FILL, alpha=0.18, lw=0)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')


def _caterpillar(ax, sub, spec, strata, bench, title, label_counts=True):
    sub = sub.sort_values(spec['att']).reset_index(drop=True)
    y = np.arange(len(sub))
    lo, hi = sub.get(spec['lower']), sub.get(spec['upper'])
    if lo is not None and hi is not None:
        ax.hlines(y, lo, hi, color=BAND, lw=2.2, zorder=1, label=spec['interval'])
    p = sub.get(spec['pvalue']) if spec['pvalue'] else None
    sig = p.lt(0.05) if p is not None else pd.Series(False, index=sub.index)
    ax.scatter(sub.loc[sig, spec['att']], y[sig.to_numpy()], s=26, color=SIG,
              edgecolor='black', lw=.4, zorder=3,
              label=(f"{spec.get('ptest', 'p')} < .05 (n={int(sig.sum())})"
                     if label_counts else f"{spec.get('ptest', 'p')} < .05"))
    ax.scatter(sub.loc[~sig, spec['att']], y[~sig.to_numpy()], s=26, color=NOTSIG,
              edgecolor='black', lw=.4, zorder=3,
              label=(f"not significant (n={int((~sig).sum())})"
                     if label_counts else "not significant"))
    ax.axvline(0, color=MUTED, lw=1)
    mean = sub[spec['att']].mean()
    ax.axvline(mean, color=SECONDARY, lw=1.6,
              label=(f"mean ATT {mean:+.4f}" if label_counts else "mean ATT for this panel"))
    if bench is not None:
        ax.axvline(bench, color='black', lw=1.4, ls='--',
                  label=f"season-matched benchmark {bench:+.4f}")
    step = max(1, len(sub) // 22)
    ax.set_yticks(y[::step])
    ax.set_yticklabels(sub['site'][::step], fontsize=6)
    ax.grid(axis='x', color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10)
    return mean, int(sig.sum())


def shc_figure1(cadence, paths, summary, strata, out, pool='selected'):
    """Observed, counterfactual and effect for every site, by land cover."""
    d_all = paths[paths.pool == pool] if 'pool' in paths else paths
    info = summary.set_index('site')
    for stratum in STRATA:
        sites = sorted(s for s in d_all.site.unique() if strata.get(s) == stratum)
        if not sites:
            continue
        for page, chunk in _pages(sites):
            rows = int(np.ceil(len(chunk) / NCOLS))
            fig, axes = plt.subplots(rows * 2, NCOLS, figsize=(4.0 * NCOLS, 4.0 * rows),
                                     squeeze=False)
            for j, site in enumerate(chunk):
                r, c = divmod(j, NCOLS)
                top, bottom = axes[r * 2][c], axes[r * 2 + 1][c]
                d = d_all[d_all.site == site].sort_values('calendar_date')
                pre, post = d[d.period == 'pre'], d[d.period == 'post']
                for part in (pre, post):
                    top.plot(part.calendar_date, part.observed, color=OBSERVED, lw=1.5)
                    top.plot(part.calendar_date, part.counterfactual, color=PRIMARY, lw=1.3, ls='--')
                top.plot([], [], color=OBSERVED, lw=1.5, label='Observed (this site)')
                top.plot([], [], color=PRIMARY, lw=1.3, ls='--',
                        label='SHC counterfactual (own earlier blocks)')
                _decorate(top, d.calendar_date)
                row = info.loc[site] if site in info.index else None
                title = site
                if row is not None and pd.notna(row.get('att_3month')):
                    title = f"{site}  m={int(row.m)}  ATT {row.att_3month:+.3f}"
                top.set_title(title, fontsize=9)
                bottom.plot(pre.calendar_date, pre.effect, color=MUTED, lw=1.1,
                           label='Pre-period fit residual')
                bottom.plot(post.calendar_date, post.effect, color=SECONDARY, lw=1.8,
                           marker='o', ms=3, label='Effect')
                bottom.axhline(0, color=MUTED, lw=.9)
                _decorate(bottom, d.calendar_date)
            for j in range(len(chunk), rows * NCOLS):
                r, c = divmod(j, NCOLS)
                axes[r * 2][c].axis('off')
                axes[r * 2 + 1][c].axis('off')
            axes[0][0].set_ylabel('NDVI', fontsize=8)
            axes[1][0].set_ylabel('Effect', fontsize=8)
            h = (axes[0][0].get_legend_handles_labels()[0] + axes[1][0].get_legend_handles_labels()[0])
            l = (axes[0][0].get_legend_handles_labels()[1] + axes[1][0].get_legend_handles_labels()[1])
            fig.legend(h, l, loc='lower center', ncol=4, fontsize=9, frameon=False)
            fig.suptitle(
                f"Figure 1 - SHC {cadence}: observed, counterfactual and effect per site\n"
                f"{stratum} sites ({len(sites)} total), on each site's own calendar; "
                f"shading = three months after Helene", fontsize=11)
            fig.tight_layout(rect=(0, 0.03, 1, 0.95))
            suffix = f"_p{page + 1}" if len(sites) > PER_PAGE else ""
            fig.savefig(out / f"figure1_site_effect_{stratum}{suffix}.png", dpi=150)
            plt.close(fig)


def figure4(cadence, summary, strata, out, effects):
    """One figure per period inside the damage window, plus the window-wide
    caterpillar. No season-matched benchmark -- see module docstring."""
    spec = SHC_SPEC
    bench = None
    frames = out / 'figure4_by_period'
    frames.mkdir(parents=True, exist_ok=True)
    # Clear stale period PNGs first: a rerun with fewer/different periods
    # than a previous run otherwise leaves old period-date pairs behind
    # alongside the new ones (found 2026-11-11 in hls_2y_random_control).
    for stale in frames.glob('figure4_period*.png'):
        stale.unlink()

    damage = effects[effects.damage].copy()
    if damage.empty:
        print('  no damage-window periods; figure 4 by period skipped')
        return bench

    present = [x for x in STRATA if any(strata.get(s) == x for s in damage.site.unique())]
    periods = sorted(damage.date.unique())
    span = float(np.nanpercentile(np.abs(damage[spec['effect']]), 99.5))
    limits = (-span * 1.1, span * 1.1)

    for index, period in enumerate(periods, 1):
        at = damage[damage.date == period]
        fig, axes = plt.subplots(1, len(present), figsize=(5.0 * len(present), 8.6), squeeze=False)
        for ax, stratum in zip(axes[0], present):
            sub = at[[strata.get(s) == stratum for s in at.site]].copy()
            sub = sub.sort_values(spec['effect']).reset_index(drop=True)
            y = np.arange(len(sub))
            pv = sub.get(spec['period_p']) if spec['period_p'] else None
            has_p = pv is not None and pv.notna().any()
            sig = (pv < 0.05).fillna(False) if has_p else pd.Series(False, index=sub.index)
            ax.scatter(sub.loc[~sig, spec['effect']], y[~sig.to_numpy()], s=28, color=NOTSIG,
                      edgecolor='black', lw=.4, zorder=3,
                      label='site effect' if not has_p else 'not significant')
            ax.axvline(0, color=MUTED, lw=1)
            mean = float(sub[spec['effect']].mean())
            ax.axvline(mean, color=SECONDARY, lw=1.8, ls=(0, (6, 2)), label='panel mean')
            step = max(1, len(sub) // 22)
            ax.set_yticks(y[::step])
            ax.set_yticklabels(sub['site'][::step], fontsize=6)
            ax.set_xlim(*limits)
            ax.grid(axis='x', color=GRID, lw=.6)
            ax.set_axisbelow(True)
            ax.set_title(f"{stratum} (n={len(sub)})\nmean {mean:+.4f}, no per-period inference",
                        fontsize=10)
            ax.set_xlabel(f"{spec['label']} effect (NDVI)", fontsize=9)
        axes[0][0].set_ylabel('Treated site, ordered by effect')
        axes[0][-1].legend(fontsize=7.5, loc='lower right', framealpha=.95)
        stamp = pd.Timestamp(period).date()
        fig.suptitle(
            f"Figure 4 - SHC {cadence}: period {index} of {len(periods)} after Helene, "
            f"starting {stamp}\none period per figure; panels share an x-axis across all "
            f"{len(periods)} periods", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        fig.savefig(frames / f"figure4_period{index:02d}_{stamp}.png", dpi=160)
        plt.close(fig)
    print(f"  figure 4: {len(periods)} per-period figures in {frames.name}/")

    fig, ax = plt.subplots(figsize=(10, max(5, len(summary) * 0.13)))
    _caterpillar(ax, summary, spec, strata, bench,
                f"Figure 4 - SHC {cadence}: the three months after Helene as one average per site")
    ax.set_xlabel('Site-specific SHC ATT (NDVI), three months')
    ax.set_ylabel('Treated site, ordered by effect')
    ax.legend(fontsize=8, loc='lower right', framealpha=.95)
    fig.tight_layout()
    fig.savefig(out / 'figure4_site_att_significance.png', dpi=180)
    plt.close(fig)
    return bench


def run_shc(cadence):
    out = ROOT / 'shc' / cadence / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    tables = ROOT / 'shc' / cadence / 'tables'
    paths = pd.read_csv(tables / 'site_paths.csv', parse_dates=['calendar_date'])
    summary = pd.read_csv(tables / 'site_summary.csv')
    strata = strata_of()
    print(f"  shc {cadence}: {summary.site.nunique()} sites")

    shc_figure1(cadence, paths, summary, strata, out)

    sel = paths[paths.pool == 'selected'] if 'pool' in paths else paths
    damage = sel[(sel.period == 'post') & (sel.calendar_date < DAMAGE_END)].copy()
    damage['damage'] = True
    damage['date'] = damage['calendar_date']
    if len(damage):
        figure4(cadence, summary, strata, out, damage)
    print(f"  shc {cadence}: grouped figures written")


def shc_donor_selection_figure(cadence='weekly'):
    """Same figure as scripts/validation/hls_flooding_128/hls_3y_shc/_figures.py:
    shc_donor_selection_figure, adapted to this folder's OLD (un-renamed)
    schema: donor period AND weight live together in one site_weights.csv
    here, not split into historical_donor_periods.csv / historical_donor_
    weights.csv, and the date axis comes from site_paths.csv's own
    calendar_date column (this folder has no separate panel file)."""
    tables = ROOT / 'shc' / cadence / 'tables'
    figures = ROOT / 'shc' / cadence / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    weights = pd.read_csv(tables / 'site_weights.csv', dtype={'site': str})
    for col in ('donor_pre_start', 'donor_pre_end', 'donor_pseudopost_start', 'donor_pseudopost_end'):
        weights[col] = pd.to_datetime(weights[col])
    blocks = weights.query("pool == 'selected' and weight > @WTOL").copy()
    if blocks.empty:
        raise ValueError('No selected donor blocks with nonzero weight; nothing to plot')

    paths = pd.read_csv(tables / 'site_paths.csv', usecols=['calendar_date'],
                        parse_dates=['calendar_date'])
    dates = np.sort(paths['calendar_date'].unique())
    n_dates = len(dates)
    step_days = int(np.median(np.diff(dates)) / np.timedelta64(1, 'D')) or 7

    lc = pd.read_csv(LANDCOVER, dtype={'site_id': str}).set_index('site_id')['dominant']
    sites = sorted(blocks['site'].unique(),
                    key=lambda s: (LC_ORDER.index(lc.get(s, 'forest'))
                                   if lc.get(s, 'forest') in LC_ORDER else len(LC_ORDER), s))
    site_row = {s: i for i, s in enumerate(sites)}
    n_sites = len(sites)

    match_w = np.zeros((n_sites, n_dates))
    post_w = np.zeros((n_sites, n_dates))
    for row in blocks.itertuples():
        r = site_row[row.site]
        pre_lo = np.searchsorted(dates, np.datetime64(row.donor_pre_start))
        pre_hi = np.searchsorted(dates, np.datetime64(row.donor_pre_end), side='right')
        post_lo = np.searchsorted(dates, np.datetime64(row.donor_pseudopost_start))
        post_hi = np.searchsorted(dates, np.datetime64(row.donor_pseudopost_end), side='right')
        match_w[r, pre_lo:pre_hi] += row.weight
        post_w[r, post_lo:post_hi] += row.weight

    match_w = np.where(match_w > WTOL, match_w, np.nan)
    post_w = np.where(post_w > WTOL, post_w, np.nan)
    vmax = float(np.nanmax(np.r_[match_w.ravel(), post_w.ravel()]))

    fig_h = max(4.0, 0.16 * n_sites + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    date_edges = np.r_[dates, dates[-1] + np.timedelta64(step_days, 'D')]
    norm = Normalize(vmin=0, vmax=vmax)

    combined_match = np.full((2 * n_sites, n_dates), np.nan)
    combined_post = np.full((2 * n_sites, n_dates), np.nan)
    for i in range(n_sites):
        combined_match[2 * i] = match_w[i]
        combined_post[2 * i + 1] = post_w[i]
    y_edges = np.arange(2 * n_sites + 1) * 0.5
    ax.pcolormesh(date_edges, y_edges, combined_match, cmap='Blues', norm=norm, shading='flat')
    pm = ax.pcolormesh(date_edges, y_edges, combined_post, cmap='Oranges', norm=norm, shading='flat')

    ax.axvline(EVENT, color='black', ls='--', lw=1.2, zorder=5)
    ax.text(EVENT, n_sites + 0.3, '  Hurricane Helene', rotation=90,
           va='bottom', ha='right', fontsize=8)

    ax.set_yticks([(2 * i + 1) * 0.5 for i in range(n_sites)])
    ax.set_yticklabels(sites, fontsize=5.3)
    ax.set_ylim(0, n_sites)

    boundaries = [0]
    labels = []
    cur = lc.get(sites[0], 'unknown')
    for i, s in enumerate(sites):
        lab = lc.get(s, 'unknown')
        if lab != cur:
            boundaries.append(i)
            labels.append(cur)
            cur = lab
    boundaries.append(n_sites)
    labels.append(cur)
    for b in boundaries[1:-1]:
        ax.axhline(b, color='grey', lw=0.6)
    for lo, hi, lab in zip(boundaries[:-1], boundaries[1:], labels):
        ax.text(date_edges[-1] + np.timedelta64(10, 'D'), (lo + hi) / 2, lab,
               fontsize=7, va='center', ha='left')

    ax.set_xlabel('Calendar date')
    ax.set_title(f'SHC selected historical donor blocks -- {cadence} NDVI (2-year history)\n'
                "top half of each row: matching window; bottom half: pseudo-post window; "
                "color = summed selected-pool weight at that date")
    legend = ax.legend(handles=[
        Patch(facecolor=plt.get_cmap('Blues')(0.7), label='Matching window (top half of row)'),
        Patch(facecolor=plt.get_cmap('Oranges')(0.7), label='Pseudo-post window (bottom half of row)'),
        Line2D([0], [0], color='black', ls='--', lw=1.2, label='Hurricane Helene')],
        loc='upper left', bbox_to_anchor=(1.01, 1.08), fontsize=8, frameon=False,
        title='Row halves')
    ax.add_artist(legend)
    cbar = fig.colorbar(pm, ax=ax, pad=0.12, fraction=0.025)
    cbar.set_label('summed donor weight (darker = more weight);\nblue and orange use the same 0-1 scale')
    fig.text(0.01, 0.01,
            'Diagnostic use: concentration near the right edge (recent history) or in a single '
            'season across many sites is worth checking against season_match_feasibility.csv '
            'before treating att_damage as attributable.', fontsize=6.2, color='#883333', wrap=True)
    fig.tight_layout(rect=(0, 0.04, 0.93, 1))
    out = figures / 'figure_shc_selected_historical_periods.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'wrote {out}  ({n_sites} sites, {len(blocks)} selected donor-block rows)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cadence', nargs='+', choices=['monthly', 'biweekly', 'weekly'],
                    default=['monthly', 'biweekly', 'weekly'])
    ap.add_argument('--donor-selection', action='store_true',
                    help='also/instead write figure_shc_selected_historical_periods.png')
    ap.add_argument('--no-grouped-figures', action='store_true')
    args = ap.parse_args()
    for cadence in args.cadence:
        if not args.no_grouped_figures:
            run_shc(cadence)
        if args.donor_selection:
            shc_donor_selection_figure(cadence)


if __name__ == '__main__':
    main()
