"""
Phase 2 figures, emitted as dependency-free SVG.

`matplotlib` is **not installed** in `/data/wang/junh/envs/medrag` (the only env
with the project's numpy/scipy/pandas/torch stack; the `causal` env has
matplotlib but not this project's other pins).  Rather than add a dependency to
a shared environment or shuttle results between interpreters, the two required
figures are written directly as SVG.  They are static, deterministic, and open
in any browser.

Colours are the reference categorical slots 1 (blue `#2a78d6`) and 2 (orange
`#eb6834`), which are documented as clearing the all-pairs colour-vision-
deficiency and normal-vision separation floors in both light and dark modes.
Identity is never carried by colour alone: every figure direct-labels its
groups (with counts where they exist), so all of them are readable in
greyscale.  Each SVG carries a `prefers-color-scheme` block so it is legible on
a dark background too.

Figures
-------
    scatter_sc_vs_lvcf   per-patient paired comparison against the y=x line
    histogram            single-series distribution with a median rule
    sc_trajectory        THE canonical SC plot: target vs synthetic over t,
                         with the pre/post divide marked
    placebo_distribution the permutation distribution a target's statistic is
                         judged against, with an empirical rank p-value

The last two were missing entirely.  Between them they are how a synthetic
control is normally *read*: one says whether the pre-period fit is real, the
other says whether the post-period number is unusual.  Unit-tested on synthetic
arrays in `scripts/smoke_phase2_fixes.py`; no figure in this module has been
generated from MIMIC-derived data since they were added.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

SERIES_1_L, SERIES_1_D = "#2a78d6", "#3987e5"
SERIES_2_L, SERIES_2_D = "#eb6834", "#d95926"

_STYLE = """
  .surface { fill: #fcfcfb; }
  .ink     { fill: #0b0b0b; }
  .ink2    { fill: #52514e; }
  .grid    { stroke: #d9d8d3; stroke-width: 1; }
  .axis    { stroke: #8f8e88; stroke-width: 1; }
  .s1      { fill: %(s1l)s; }
  .s2      { fill: %(s2l)s; }
  .s1s     { stroke: %(s1l)s; }
  .s2s     { stroke: %(s2l)s; }
  .ref     { stroke: #52514e; stroke-width: 2; stroke-dasharray: 6 4; fill: none; }
  text     { font-family: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .title   { font-size: 15px; font-weight: 600; }
  .sub     { font-size: 11.5px; }
  .lab     { font-size: 11px; }
  .tick    { font-size: 10.5px; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink     { fill: #ffffff; }
    .ink2    { fill: #c3c2b7; }
    .grid    { stroke: #3a3a38; }
    .axis    { stroke: #6e6d67; }
    .s1      { fill: %(s1d)s; }
    .s2      { fill: %(s2d)s; }
    .s1s     { stroke: %(s1d)s; }
    .s2s     { stroke: %(s2d)s; }
    .ref     { stroke: #c3c2b7; }
  }
""" % {"s1l": SERIES_1_L, "s1d": SERIES_1_D, "s2l": SERIES_2_L, "s2d": SERIES_2_D}


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min([s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw] or [10]) * mag
    t0 = np.ceil(lo / step) * step
    return [float(t0 + i * step) for i in range(int((hi - t0) / step) + 1)]


def _fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _header(w: int, h: int, title: str, subtitle: str) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img">',
        f"<style>{_STYLE}</style>",
        f'<rect class="surface" width="{w}" height="{h}"/>',
        f'<text class="title ink" x="24" y="30">{html.escape(title)}</text>',
        f'<text class="sub ink2" x="24" y="50">{html.escape(subtitle)}</text>',
    ]


def scatter_sc_vs_lvcf(
    x: Sequence[float],
    y: Sequence[float],
    path: Path,
    title: str,
    subtitle: str = "",
    xlabel: str = "LVCF post-period RMSE",
    ylabel: str = "SC post-period RMSE",
    w: int = 720,
    h: int = 640,
) -> Path:
    """Per-patient scatter with the y = x reference line.

    Form: identity/comparison of two paired measurements, so a scatter against
    the 45-degree line -- the line, not a bar chart, is what makes "who wins"
    readable at a glance.  Points below the line are patients where the
    synthetic control beats last-value-carried-forward.  Colour is redundant
    with position (below vs above the line) and both groups are direct-labelled
    with their counts, so nothing is encoded by colour alone.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L, R, T, B = 88, 28, 76, 66
    pw, ph = w - L - R, h - T - B
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def sx(v):
        return L + (v - lo) / (hi - lo) * pw

    def sy(v):
        return T + ph - (v - lo) / (hi - lo) * ph

    win = y < x
    s = _header(w, h, title, subtitle)
    ticks = _nice_ticks(lo, hi, 5)
    for t in ticks:
        s.append(f'<line class="grid" x1="{L}" y1="{sy(t):.1f}" x2="{L+pw}" y2="{sy(t):.1f}"/>')
        s.append(f'<line class="grid" x1="{sx(t):.1f}" y1="{T}" x2="{sx(t):.1f}" y2="{T+ph}"/>')
        s.append(f'<text class="tick ink2" x="{L-8}" y="{sy(t)+3.5:.1f}" text-anchor="end">{_fmt(t)}</text>')
        s.append(f'<text class="tick ink2" x="{sx(t):.1f}" y="{T+ph+18}" text-anchor="middle">{_fmt(t)}</text>')
    s.append(f'<line class="axis" x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}"/>')
    s.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}"/>')
    s.append(f'<line class="ref" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}"/>')

    for cls, mask in (("s2", ~win), ("s1", win)):
        s.append(f'<g class="{cls}" fill-opacity="0.5">')
        for xi, yi in zip(x[mask], y[mask]):
            s.append(f'<circle cx="{sx(xi):.1f}" cy="{sy(yi):.1f}" r="3.2"/>')
        s.append("</g>")

    nw = int(win.sum())
    s.append(f'<text class="lab ink2" x="{L+pw-10}" y="{T+ph-14}" text-anchor="end">'
             f'above the line: SC worse  (n={len(x)-nw}, {100*(1-win.mean()):.1f}%)</text>')
    s.append(f'<text class="lab ink2" x="{L+14}" y="{T+18}">'
             f'below the line: SC better  (n={nw}, {100*win.mean():.1f}%)</text>')
    s.append(f'<text class="lab ink" x="{L+pw/2:.0f}" y="{h-22}" text-anchor="middle">{html.escape(xlabel)}</text>')
    s.append(f'<text class="lab ink" transform="translate(22,{T+ph/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">{html.escape(ylabel)}</text>')
    s.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s))
    return path


def histogram(
    v: Sequence[float],
    path: Path,
    title: str,
    subtitle: str = "",
    xlabel: str = "",
    ylabel: str = "patients",
    bins: int = 30,
    annotate: Optional[Sequence[Tuple[float, str]]] = None,
    w: int = 720,
    h: int = 460,
) -> Path:
    """Single-series distribution.  One series, so no legend box -- the title
    names it; the median is direct-labelled rather than every bar."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    L, R, T, B = 78, 28, 76, 66
    pw, ph = w - L - R, h - T - B
    cnt, edges = np.histogram(v, bins=bins)
    ymax = max(int(cnt.max()), 1)
    x0, x1 = float(edges[0]), float(edges[-1])
    if x1 <= x0:
        x1 = x0 + 1.0

    def sx(a):
        return L + (a - x0) / (x1 - x0) * pw

    def sy(c):
        return T + ph - (c / ymax) * ph

    s = _header(w, h, title, subtitle)
    for t in _nice_ticks(0, ymax, 4):
        s.append(f'<line class="grid" x1="{L}" y1="{sy(t):.1f}" x2="{L+pw}" y2="{sy(t):.1f}"/>')
        s.append(f'<text class="tick ink2" x="{L-8}" y="{sy(t)+3.5:.1f}" text-anchor="end">{_fmt(t)}</text>')
    # 2px surface gap between adjacent fills
    for i, c in enumerate(cnt):
        if c <= 0:
            continue
        bx, bw = sx(edges[i]), sx(edges[i + 1]) - sx(edges[i])
        s.append(f'<rect class="s1" x="{bx+1:.1f}" y="{sy(c):.1f}" '
                 f'width="{max(bw-2,0.6):.1f}" height="{T+ph-sy(c):.1f}" rx="2"/>')
    for t in _nice_ticks(x0, x1, 6):
        s.append(f'<text class="tick ink2" x="{sx(t):.1f}" y="{T+ph+18}" text-anchor="middle">{_fmt(t)}</text>')
    s.append(f'<line class="axis" x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}"/>')
    med = float(np.median(v))
    s.append(f'<line class="ref" x1="{sx(med):.1f}" y1="{T}" x2="{sx(med):.1f}" y2="{T+ph}"/>')
    s.append(f'<text class="lab ink2" x="{sx(med)+7:.1f}" y="{T+16}">median {_fmt(med)}</text>')
    for av, lab in (annotate or []):
        if x0 <= av <= x1:
            s.append(f'<line class="axis" x1="{sx(av):.1f}" y1="{T}" x2="{sx(av):.1f}" y2="{T+ph}"/>')
            s.append(f'<text class="lab ink2" x="{sx(av)+7:.1f}" y="{T+34}">{html.escape(lab)}</text>')
    s.append(f'<text class="lab ink" x="{L+pw/2:.0f}" y="{h-22}" text-anchor="middle">{html.escape(xlabel)}</text>')
    s.append(f'<text class="lab ink" transform="translate(20,{T+ph/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">{html.escape(ylabel)}</text>')
    s.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s))
    return path


def sc_trajectory(
    target: Sequence[float],
    synthetic: Sequence[float],
    path: Path,
    title: str,
    T0: int,
    subtitle: str = "",
    xlabel: str = "visit index t",
    ylabel: str = "",
    t_values: Optional[Sequence[float]] = None,
    w: int = 760,
    h: int = 460,
) -> Path:
    """**The canonical synthetic-control plot**: target vs synthetic over time.

    Two lines against `t`, with a dashed vertical rule at the pre/post divide
    (between `T0` and `T0+1`).  This is the figure every synthetic-control paper
    leads with, and it was entirely absent from this project: it is how a reader
    judges whether the pre-period fit is *real* -- whether the synthetic line
    actually tracks the target before the divide, rather than being reported as
    a single RMSE number that a two-timestep pre-period can drive to ~0 with
    donors that have nothing to do with the patient.  A reader who cannot see
    the pre-period cannot tell an honest fit from an interpolation artifact.

    Form: two paired series over an ordered index, so lines with markers, not
    bars.  Only two series, so they are direct-labelled at their right-hand end
    instead of carrying a legend box, and the pre/post split is a rule with a
    text label rather than a shaded region (the shading would fight the lines).
    Colour is redundant with the direct labels, so the figure survives
    greyscale and colour-vision deficiency.

    Note on what the scalar is: a state here is a `d`-dimensional vector per
    visit, so the caller must reduce it to one number per `t` before plotting.
    Use a LINEAR functional -- the total code mass `X_t.sum()` is the natural
    one -- because the synthetic control is a linear combination of donors, so
    a linear summary of the synthetic control equals the same combination of
    the donors' summaries and the plotted "synthetic" line is genuinely the
    synthetic control's own trajectory.  A non-linear summary (a norm, a
    cosine) would not have that property and the plot would be misleading.

    Args:
        target:    (T,) scalar summary of the target's state at each t.
        synthetic: (T,) same summary of the synthetic control.  Same length.
        T0:        last PRE-period index.  The rule is drawn at `T0 + 0.5`;
                   `t <= T0` is fit, `t > T0` is forecast.
        t_values:  x positions.  Defaults to `0 .. T-1`.
    """
    y1 = np.asarray(target, dtype=float)
    y2 = np.asarray(synthetic, dtype=float)
    if y1.shape != y2.shape or y1.ndim != 1:
        raise ValueError(
            f"target and synthetic must be 1-D and the same length; got "
            f"{y1.shape} and {y2.shape}")
    if len(y1) == 0:
        raise ValueError("empty trajectory")
    xs = (np.arange(len(y1), dtype=float) if t_values is None
          else np.asarray(t_values, dtype=float))

    L, R, T, B = 84, 132, 76, 66     # extra right margin for the direct labels
    pw, ph = w - L - R, h - T - B
    ylo = float(min(y1.min(), y2.min()))
    yhi = float(max(y1.max(), y2.max()))
    if yhi <= ylo:
        ylo, yhi = ylo - 0.5, yhi + 0.5
    ypad = 0.10 * (yhi - ylo)
    ylo, yhi = ylo - ypad, yhi + ypad
    xlo, xhi = float(xs.min()), float(xs.max())
    if xhi <= xlo:
        xlo, xhi = xlo - 0.5, xhi + 0.5
    xpad = 0.06 * (xhi - xlo)
    xlo, xhi = xlo - xpad, xhi + xpad

    def sx(v):
        return L + (v - xlo) / (xhi - xlo) * pw

    def sy(v):
        return T + ph - (v - ylo) / (yhi - ylo) * ph

    s = _header(w, h, title, subtitle)
    for t in _nice_ticks(ylo, yhi, 5):
        s.append(f'<line class="grid" x1="{L}" y1="{sy(t):.1f}" x2="{L+pw}" y2="{sy(t):.1f}"/>')
        s.append(f'<text class="tick ink2" x="{L-8}" y="{sy(t)+3.5:.1f}" '
                 f'text-anchor="end">{_fmt(t)}</text>')
    for xv in xs:
        s.append(f'<text class="tick ink2" x="{sx(xv):.1f}" y="{T+ph+18}" '
                 f'text-anchor="middle">{_fmt(xv)}</text>')
    s.append(f'<line class="axis" x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}"/>')
    s.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}"/>')

    # The pre/post divide -- the whole point of the figure.
    xd = float(T0) + 0.5
    if xlo <= xd <= xhi:
        s.append(f'<line class="ref" x1="{sx(xd):.1f}" y1="{T}" '
                 f'x2="{sx(xd):.1f}" y2="{T+ph}"/>')
        s.append(f'<text class="lab ink2" x="{sx(xd)-7:.1f}" y="{T-8}" '
                 f'text-anchor="end">← fit on t ≤ {T0}</text>')
        s.append(f'<text class="lab ink2" x="{sx(xd)+7:.1f}" y="{T-8}">'
                 f'forecast t &gt; {T0} →</text>')

    for cls, yy, lab in (("s1s", y1, "target"), ("s2s", y2, "synthetic control")):
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, yy))
        s.append(f'<polyline class="{cls}" fill="none" stroke-width="2.4" '
                 f'stroke-linejoin="round" points="{pts}"/>')
        dot = "s1" if cls == "s1s" else "s2"
        for a, b in zip(xs, yy):
            s.append(f'<circle class="{dot}" cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="3.6"/>')
        # Direct label at the right-hand end instead of a legend box: two
        # series only, so the reader never has to move their eye off the data.
        s.append(f'<text class="lab {dot}" x="{L+pw+10}" '
                 f'y="{sy(yy[-1])+4:.1f}">{html.escape(lab)}</text>')

    s.append(f'<text class="lab ink" x="{L+pw/2:.0f}" y="{h-22}" '
             f'text-anchor="middle">{html.escape(xlabel)}</text>')
    s.append(f'<text class="lab ink" transform="translate(22,{T+ph/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">{html.escape(ylabel)}</text>')
    s.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s))
    return path


def placebo_distribution(
    placebo: Sequence[float],
    observed: float,
    path: Path,
    title: str,
    subtitle: str = "",
    xlabel: str = "",
    ylabel: str = "placebo units",
    bins: int = 30,
    lower_is_better: bool = True,
    observed_label: str = "target",
    w: int = 760,
    h: int = 470,
) -> Path:
    """**The placebo permutation distribution** a target's statistic is judged against.

    The inferential half of the classical synthetic-control method (Abadie et
    al.): the same estimator is applied to units that were never treated, and
    the treated unit's statistic is read off against the resulting distribution.
    Without it, a single fit statistic has no scale -- "post-period RMSE 0.31"
    means nothing until you know what 748 placebo fits produce.  This figure was
    absent from the project entirely, so the placebo fits that WERE computed had
    no picture.

    Form: a histogram of the placebo statistics, with the observed value drawn
    as a solid rule and direct-labelled with its **empirical rank p-value** --
    the fraction of placebo units at least as extreme, computed with the
    standard `(1 + #at-least-as-extreme) / (1 + n)` correction so it can never
    be reported as exactly 0.  One series, so no legend; the marker carries the
    identity.

    Args:
        placebo:  (n,) statistic for each placebo unit.  Non-finite entries are
                  dropped (and counted in the subtitle by the caller).
        observed: the target's statistic, on the SAME scale.
        lower_is_better: direction of "extreme".  True for an error-like
                  statistic (RMSE): the target is unusual if it is unusually
                  LOW.  False for F1 or a ratio.
    """
    v = np.asarray(placebo, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite placebo values")
    obs = float(observed)
    n = int(v.size)
    at_least = int((v <= obs).sum() if lower_is_better else (v >= obs).sum())
    p = (1.0 + at_least) / (1.0 + n)

    L, R, T, B = 78, 28, 76, 78
    pw, ph = w - L - R, h - T - B
    lo_e = float(min(v.min(), obs))
    hi_e = float(max(v.max(), obs))
    if hi_e <= lo_e:
        hi_e = lo_e + 1.0
    pad = 0.03 * (hi_e - lo_e)
    cnt, edges = np.histogram(v, bins=bins, range=(lo_e - pad, hi_e + pad))
    ymax = max(int(cnt.max()), 1)
    x0, x1 = float(edges[0]), float(edges[-1])

    def sx(a):
        return L + (a - x0) / (x1 - x0) * pw

    def sy(c):
        return T + ph - (c / ymax) * ph

    s = _header(w, h, title, subtitle)
    for t in _nice_ticks(0, ymax, 4):
        s.append(f'<line class="grid" x1="{L}" y1="{sy(t):.1f}" x2="{L+pw}" y2="{sy(t):.1f}"/>')
        s.append(f'<text class="tick ink2" x="{L-8}" y="{sy(t)+3.5:.1f}" '
                 f'text-anchor="end">{_fmt(t)}</text>')
    for i, c in enumerate(cnt):
        if c <= 0:
            continue
        bx, bw = sx(edges[i]), sx(edges[i + 1]) - sx(edges[i])
        s.append(f'<rect class="s1" x="{bx+1:.1f}" y="{sy(c):.1f}" '
                 f'width="{max(bw-2,0.6):.1f}" height="{T+ph-sy(c):.1f}" rx="2"/>')
    for t in _nice_ticks(x0, x1, 6):
        s.append(f'<text class="tick ink2" x="{sx(t):.1f}" y="{T+ph+18}" '
                 f'text-anchor="middle">{_fmt(t)}</text>')
    s.append(f'<line class="axis" x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}"/>')

    # The observed statistic: solid, in the second series colour so it never
    # reads as one more bar.
    ox = sx(obs)
    s.append(f'<line class="s2s" x1="{ox:.1f}" y1="{T-6}" x2="{ox:.1f}" '
             f'y2="{T+ph}" stroke-width="2.6"/>')
    anchor = "start" if ox < L + pw * 0.62 else "end"
    dx = 8 if anchor == "start" else -8
    s.append(f'<text class="lab s2" x="{ox+dx:.1f}" y="{T+14}" '
             f'text-anchor="{anchor}">{html.escape(observed_label)} = {_fmt(obs)}</text>')
    s.append(f'<text class="lab ink2" x="{ox+dx:.1f}" y="{T+30}" text-anchor="{anchor}">'
             f'rank p = {p:.4f}  ({at_least}/{n} placebos '
             f'{"≤" if lower_is_better else "≥"} it)</text>')
    med = float(np.median(v))
    s.append(f'<line class="ref" x1="{sx(med):.1f}" y1="{T}" x2="{sx(med):.1f}" y2="{T+ph}"/>')
    s.append(f'<text class="lab ink2" x="{sx(med)+7:.1f}" y="{T+ph-10:.1f}">'
             f'placebo median {_fmt(med)}</text>')

    s.append(f'<text class="lab ink" x="{L+pw/2:.0f}" y="{h-34}" '
             f'text-anchor="middle">{html.escape(xlabel)}</text>')
    s.append(f'<text class="lab ink2" x="{L}" y="{h-14}">'
             f'p = (1 + #placebos at least as extreme) / (1 + n); '
             f'{"lower" if lower_is_better else "higher"} is more extreme</text>')
    s.append(f'<text class="lab ink" transform="translate(20,{T+ph/2:.0f}) rotate(-90)" '
             f'text-anchor="middle">{html.escape(ylabel)}</text>')
    s.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(s))
    return path
