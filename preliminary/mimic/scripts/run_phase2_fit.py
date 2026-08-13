#!/usr/bin/env python
"""
Phase 2 runner: fit the synthetic control, run the baselines, score everything.

    /data/wang/junh/envs/medrag/bin/python scripts/run_phase2_fit.py

Default sweep (this is what produces the headline table):

    encoder/scaling : bag-raw, bag-rowL2                  (the required ablation)
    T0              : 1 (primary), 2 (robustness), 0 (shifted-T0 placebo)
    m (pre-filter)  : 25, 50, 100, all                    (m-sensitivity)
    methods         : sc, sc_random(placebo), lvcf, nn1, topk_mean, cohort_mean,
                      ridge

**MedCPT is no longer configured.**  The state representation for Phase 2 is
now only the 585-dim binary bag-of-codes, in raw and row-L2 forms.  Nothing was
deleted -- `data/derived/states_medcpt.npz` and the Phase 1 code that built it
are untouched and `load_states("medcpt")` still works -- but no configuration
in this runner points at it, so no run scores it.

Everything is deterministic given `--seed`; the only randomness anywhere is the
donor draw for the `sc_random` placebo and the bootstrap/CV seeds, all fixed.

Resumable: every fit is cached under `results/.cache/` keyed by
(panel, m, selection, seed, solver tolerances).  Re-running skips solved
configurations unless `--no-resume`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Pin BLAS to one thread per process BEFORE numpy is imported.  Parallelism here
# is over *targets*, and each target's solve is a small dense Gram/matmul; left
# unpinned, every one of the --jobs worker processes also spawns a full BLAS
# thread pool and the box thrashes (observed: load average 284 on a 128-core
# machine, with the m=all fits making no measurable progress).  One thread per
# worker is both faster and neighbourly on a shared box.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.state import load_states  # noqa: E402
from src.synthctl import baselines as bl  # noqa: E402
from src.synthctl import evaluate as ev  # noqa: E402
from src.synthctl import figures as figs  # noqa: E402
from src.synthctl.fit import (  # noqa: E402
    DEFAULT_M, SLSQP_FTOL, SLSQP_MAXITER, WEIGHT_EPS, Panel, TargetFit,
    convergence_report, fit_all,
)

DERIVED = _PROJECT_ROOT / "data" / "derived"
RESULTS = _PROJECT_ROOT / "results"
FIGURES = _PROJECT_ROOT / "figures"
CACHE = RESULTS / ".cache"
TIMELINE = DERIVED / "timeline_mimic3_mortality.parquet"

#: (encoder, scale, rownorm).  `scale` is `state.load_states`' scaling; `rownorm`
#: is this module's per-visit L2 ablation, which state.py deliberately does not
#: implement (Phase 1 MODULE.md "Scaling": "not implemented ... the most
#: promising and a one-line addition if Phase 2 wants it").
#:
#: `("medcpt", "none", "raw")` was REMOVED here.  The 768-dim dense embedding is
#: not interpretable, so it cannot carry the primary metric (code-set F1 needs a
#: vocabulary) and every conclusion drawn on it would have rested on the
#: secondary proper-scoring numbers alone.  The artifacts and the Phase 1 code
#: that produced them are deliberately left in place; they are simply no longer
#: configured.
CONFIGS: Tuple[Tuple[str, str, str], ...] = (
    ("bag", "none", "raw"),
    ("bag", "none", "rowl2"),
)

#: m=None means "no pre-filter, every eligible donor".
def m_grid_for(T0: int, encoder: str) -> Tuple[Optional[int], ...]:
    """The m-sensitivity grid, which has to be encoder-aware for cost reasons.

    Measured per-target SLSQP cost at T0=1 (single-threaded, 748 donors
    available), and the reason the grid is not uniform:

    | m   | medcpt        | bag            |
    |-----|---------------|----------------|
    | 100 | 0.03 s, 13 it | 0.08 s, 45 it  |
    | 200 | 0.16 s, 13 it | 0.70 s, 58 it  |
    | 400 | 1.32 s, 13 it | 5.64 s, 55 it  |
    | 748 | 9.2 s,  14 it | ~38 s (extrap.)|

    The binary bag design needs **4x the SLSQP iterations** of the dense MedCPT
    one at every m -- its objective is far flatter near the optimum, which is
    the optimisation-side shadow of the tie structure Phase 1 measured (~100
    distinct donor distances across 748 donors). Combined with the O(m^3) inner
    LSQ that makes `m=all` cost ~20 min per bag configuration versus ~5 min for
    MedCPT.

    So: `m=all` is run for MedCPT at T0=1 and for **every** encoder at T0=2
    (only 347 donors there, so it is cheap), and replaced by m=200/400 for the
    bag encoder at T0=1.  m=200 and m=400 are run for MedCPT too, so the two
    sensitivity curves are directly comparable over the shared grid.

    At T0=0 there are 1,795 donors and a single `m=all` solve exceeded 120 s
    even for MedCPT -- ~60 h over the 1,796 targets -- so the shifted-T0 placebo
    is pre-filtered only.

    NOTE: MedCPT is no longer in `CONFIGS`, so the non-bag branch at T0=1 is now
    unreachable from the default sweep.  The measurements above are kept as the
    provenance of the grid's shape (and the `encoder` argument with them), not
    because a MedCPT run is expected.
    """
    if T0 == 0:
        return (25, 50, 100)
    if T0 == 1 and encoder == "bag":
        return (25, 50, 100, 200, 400)
    if T0 == 1:
        return (25, 50, 100, 200, 400, None)
    return (25, 50, 100, None)


# --------------------------------------------------------------------------
# Cohort labels.
# --------------------------------------------------------------------------


def load_labels(ids: Sequence[str], t: int) -> np.ndarray:
    """Mortality label at visit `t` for each subject, from the Phase 0 parquet.

    Phase 0's semantics: the label at `t` is the expire flag of the patient's
    NEXT admission.  Within the >=3-visit cohort that makes the label at t=0 and
    t=1 almost all zero by construction (a patient with a third visit survived
    the first two), so the outcome AUROC is only meaningful at `T0+1 >= 2`.
    """
    df = pd.read_parquet(TIMELINE, columns=["subject_id", "t", "label"])
    df["subject_id"] = df["subject_id"].astype(str)
    lut = {(s, int(tt)): int(l)
           for s, tt, l in df.itertuples(index=False, name=None)}
    return np.array([lut[(str(s), int(t))] for s in ids], dtype=int)


# --------------------------------------------------------------------------
# Fit caching (resumability).
# --------------------------------------------------------------------------


def _cache_path(panel: Panel, m, seed: int, random_donors: bool) -> Path:
    tag = f"{panel.key}_m{'all' if m is None else m}"
    tag += f"_rand{seed}" if random_donors else "_nn"
    return CACHE / f"fits_{tag}.npz"


def _save_fits(path: Path, fits: List[TargetFit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        subject_id=np.array([f.subject_id for f in fits], dtype="U"),
        i=np.array([f.i for f in fits], dtype=np.int32),
        donor_rows=np.stack([f.donor_rows for f in fits]).astype(np.int32),
        w=np.stack([f.w for f in fits]).astype(np.float64),
        pre_rmse=np.array([f.pre_rmse for f in fits]),
        eff_ipr=np.array([f.eff_donors_ipr for f in fits]),
        eff_cnt=np.array([f.eff_donors_count for f in fits], dtype=np.int32),
        max_w=np.array([f.max_weight for f in fits]),
        status=np.array([f.status for f in fits], dtype=np.int32),
        nit=np.array([f.nit for f in fits], dtype=np.int32),
        viol=np.array([f.simplex_violation for f in fits]),
        message=np.array([f.message for f in fits], dtype="U"),
        selection=np.array([f.selection for f in fits], dtype="U"),
        m=np.array([f.m for f in fits], dtype=np.int32),
        # Carried so a cache written with fit_intercept=True cannot silently
        # come back with mu=0 and a forecast that no longer matches the fit.
        mu=np.array([f.mu for f in fits]),
    )


def _load_fits(path: Path, panel: Panel) -> List[TargetFit]:
    z = np.load(path, allow_pickle=False)
    # Caches written before the intercept existed have no `mu` array.  Treat
    # them as mu=0, which is what they were.
    mu = z["mu"] if "mu" in z.files else np.zeros(len(z["i"]))
    out = []
    for k in range(len(z["i"])):
        rows = z["donor_rows"][k]
        w = z["w"][k]
        out.append(TargetFit(
            subject_id=str(z["subject_id"][k]), i=int(z["i"][k]), T0=panel.T0,
            m=int(z["m"][k]), donor_rows=rows, w=w,
            forecast=w @ panel.Y[rows] + float(mu[k]), mu=float(mu[k]),
            pre_rmse=float(z["pre_rmse"][k]),
            eff_donors_ipr=float(z["eff_ipr"][k]),
            eff_donors_count=int(z["eff_cnt"][k]),
            max_weight=float(z["max_w"][k]), status=int(z["status"][k]),
            message=str(z["message"][k]), nit=int(z["nit"][k]),
            simplex_violation=float(z["viol"][k]),
            selection=str(z["selection"][k]),
        ))
    return out


def get_fits(panel: Panel, m, seed: int, random_donors: bool, n_jobs: int,
             resume: bool) -> List[TargetFit]:
    p = _cache_path(panel, m, seed, random_donors)
    if resume and p.exists():
        try:
            fits = _load_fits(p, panel)
            print(f"[fit] {p.name}: resumed {len(fits)} cached fits", flush=True)
            return fits
        except Exception as e:  # corrupt/partial cache -> recompute, do not die
            print(f"[fit] cache {p.name} unreadable ({e}); recomputing", flush=True)
    fits = fit_all(panel, m=m, seed=seed, random_donors=random_donors,
                   n_jobs=n_jobs)
    _save_fits(p, fits)
    return fits


# --------------------------------------------------------------------------
# JSON hygiene.
# --------------------------------------------------------------------------


def jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer, int)) and not isinstance(o, bool):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


# --------------------------------------------------------------------------
# One configuration.
# --------------------------------------------------------------------------


def run_config(
    encoder: str, scale: str, rownorm: str, T0: int, m_grid: Sequence,
    args, out: Dict[str, Any],
) -> None:
    ss = load_states(encoder, derived_dir=DERIVED, scale=scale)
    panel_raw = Panel.build(ss, T0)
    panel = panel_raw.rownorm_l2() if rownorm == "rowl2" else panel_raw
    key = panel.key
    print(f"\n### {key}: n={panel.n} targets, d={panel.d}, "
          f"pre-period dim={panel.P.shape[1]}", flush=True)

    # Ground truth for the interpretable metrics always comes from the RAW
    # binary bag, so the raw / row-L2 ablation arms are scored identically.
    is_bag = encoder == "bag"
    Ybin = panel_raw.Y if is_bag else None
    K = (np.array([int((panel_raw.target_last_state(i) > 0).sum())
                   for i in range(panel_raw.n)]) if is_bag else None)
    vocab = ss.vocab if is_bag else None
    labels = load_labels(panel.ids, T0 + 1) if args.auroc else None

    simple = bl.all_simple_forecasts(panel, k=args.k,
                                     ridge_alpha=args.ridge_alpha)

    for m in m_grid:
        mtag = "all" if m is None else str(m)
        fits = get_fits(panel, m, args.seed, False, args.jobs, args.resume)
        forecasts = dict(simple)
        forecasts["sc"] = np.stack([f.forecast for f in fits])
        fitmap = {"sc": fits}

        # With no pre-filter the "random m donors" placebo IS the estimator
        # (both use the whole pool), so it is skipped rather than duplicated.
        rfits = None
        if m is not None:
            rfits = get_fits(panel, m, args.seed, True, args.jobs, args.resume)
            forecasts["sc_random"] = np.stack([f.forecast for f in rfits])
            fitmap["sc_random"] = rfits

        do_auroc = args.auroc and (args.auroc_all_m or m == args.primary_m)
        t0 = time.time()
        res = ev.evaluate_config(
            forecasts, panel.Y, reference="lvcf", Ybin=Ybin, K=K,
            labels=labels, vocab=vocab, fits=fitmap, do_auroc=do_auroc,
            boot_B=args.boot_B, boot_seed=args.boot_seed,
        )
        res.update(encoder=encoder, scale=scale, rownorm=rownorm, T0=T0,
                   m=mtag, key=key, n_donor_pool=panel.n - 1,
                   convergence={"sc": convergence_report(fits)})
        if rfits is not None:
            res["convergence"]["sc_random"] = convergence_report(rfits)
        print(f"[eval] {key} m={mtag} scored in {time.time()-t0:.1f}s"
              f"{' (+AUROC)' if do_auroc else ''}", flush=True)

        title = (f"{encoder.upper()} / scale={scale} / {rownorm} · T0={T0} "
                 f"(fit t=0..{T0}, forecast t={T0+1}) · m={mtag} · "
                 f"n={panel.n}, donors={panel.n-1}")
        table = ev.format_table(res, title)
        print(table, flush=True)
        out["tables"].append(table)
        out["configs"].append(jsonable(res))

        _emit_target_rows(out, res, forecasts, fitmap, panel, panel_raw,
                          Ybin, K, encoder, scale, rownorm, T0, mtag)
        if m == args.primary_m:
            _emit_weights(out, fitmap, panel, encoder, scale, rownorm, T0, mtag)
            if args.save_forecasts:
                np.savez_compressed(
                    RESULTS / f"phase2_forecasts_{key}_m{mtag}.npz",
                    subject_id=np.array(panel.ids, dtype="U"),
                    truth=panel.Y.astype(np.float32),
                    **{f"fc_{nm}": F.astype(np.float32)
                       for nm, F in forecasts.items()},
                )
            _emit_figures(res, forecasts, fits, panel, encoder, scale, rownorm,
                          T0, mtag)


def _emit_target_rows(out, res, forecasts, fitmap, panel, panel_raw, Ybin, K,
                      encoder, scale, rownorm, T0, mtag) -> None:
    rmse = res["_rmse_rows"]
    for r in res["rows"]:
        nm = r["method"]
        f1 = r.get("_f1_rows")
        ff = fitmap.get(nm)
        cos = ev.cosine_rows(forecasts[nm], panel.Y)
        for j, sid in enumerate(panel.ids):
            rec = {
                "encoder": encoder, "scale": scale, "rownorm": rownorm,
                "T0": T0, "m": mtag, "method": nm, "subject_id": sid,
                "post_rmse": float(rmse[nm][j]),
                "cosine": float(cos[j]),
                "code_f1": float(f1[j]) if f1 is not None else np.nan,
                "K_budget": int(K[j]) if K is not None else -1,
            }
            if ff is not None:
                g = ff[j]
                rec.update(pre_rmse=g.pre_rmse, eff_donors_ipr=g.eff_donors_ipr,
                           eff_donors_count=g.eff_donors_count,
                           max_weight=g.max_weight, solver_status=g.status,
                           solver_nit=g.nit, simplex_violation=g.simplex_violation)
            out["target_rows"].append(rec)


def _emit_weights(out, fitmap, panel, encoder, scale, rownorm, T0, mtag) -> None:
    for nm, ff in fitmap.items():
        for g in ff:
            keep = np.flatnonzero(g.w > WEIGHT_EPS)
            for j in keep:
                out["weight_rows"].append({
                    "encoder": encoder, "scale": scale, "rownorm": rownorm,
                    "T0": T0, "m": mtag, "method": nm,
                    "subject_id": g.subject_id,
                    "donor_subject_id": panel.ids[int(g.donor_rows[j])],
                    "w": float(g.w[j]),
                })


def _emit_figures(res, forecasts, fits, panel, encoder, scale, rownorm, T0,
                  mtag) -> None:
    rmse = res["_rmse_rows"]
    tag = f"{encoder}-{rownorm}-T{T0}-m{mtag}"
    v = next(r for r in res["rows"] if r["method"] == "sc")["vs_ref"]

    # ---- the canonical SC trajectory, for one representative target -------
    # Representative = the target whose SC post-period RMSE is the cohort
    # MEDIAN, chosen deterministically (argsort, stable) so the figure is not a
    # cherry-pick and does not move between runs.  A best-case or worst-case
    # target would both be misleading.
    order = np.argsort(rmse["sc"], kind="stable")
    j = int(order[len(order) // 2])
    g = fits[j]
    # Scalar summary per visit = total code mass, sum over coordinates.  It is
    # LINEAR, so the synthetic control's summary really is the same convex
    # combination of the donors' summaries (see figures.sc_trajectory).
    tgt_blk = panel.P[j].reshape(T0 + 1, panel.d)
    syn_blk = (g.w @ panel.P[g.donor_rows]).reshape(T0 + 1, panel.d) + g.mu
    tgt = np.concatenate([tgt_blk.sum(axis=1), [panel.Y[j].sum()]])
    syn = np.concatenate([syn_blk.sum(axis=1), [float(g.forecast.sum())]])
    figs.sc_trajectory(
        tgt, syn, FIGURES / f"phase2_trajectory_{tag}.svg",
        title=f"Synthetic control vs target — {encoder}/{rownorm}, T0={T0}, m={mtag}",
        T0=T0,
        subtitle=(f"median-RMSE target subject {g.subject_id} (rank "
                  f"{len(order)//2+1}/{len(order)} by SC post RMSE);  "
                  f"pre-period RMSE {g.pre_rmse:.4f}, post {rmse['sc'][j]:.4f};  "
                  f"{g.eff_donors_count} donors above 1e-4"),
        ylabel="total code mass  Σ_k X[t, k]",
    )

    # ---- the placebo distribution the target's statistic is judged against --
    # Classical Abadie construction: the same estimator applied to units that
    # are not the one in question.  The distribution is every OTHER patient's
    # SC post-period RMSE; the marker is this target's.
    others = np.delete(rmse["sc"], j)
    figs.placebo_distribution(
        others, float(rmse["sc"][j]),
        FIGURES / f"phase2_placebo_{tag}.svg",
        title=f"Placebo distribution — {encoder}/{rownorm}, T0={T0}, m={mtag}",
        subtitle=(f"same estimator refit on each of the other {len(others)} "
                  f"patients; marker is subject {g.subject_id}, the target "
                  f"plotted in the trajectory figure"),
        xlabel=f"post-period RMSE at t={T0+1}", ylabel="placebo patients",
        bins=40, lower_is_better=True,
        observed_label=f"subject {g.subject_id}",
    )

    figs.scatter_sc_vs_lvcf(
        rmse["lvcf"], rmse["sc"], FIGURES / f"phase2_sc_vs_lvcf_{tag}.svg",
        title=f"Synthetic control vs LVCF — {encoder}/{rownorm}, T0={T0}, m={mtag}",
        subtitle=(f"per-patient post-period RMSE at t={T0+1};  "
                  f"mean Δ = {v['mean_diff']:+.4f} "
                  f"[{v['lo']:+.4f}, {v['hi']:+.4f}]  ·  "
                  f"SC wins {100*v['win_frac']:.1f}% of patients"),
    )
    figs.histogram(
        [f.eff_donors_ipr for f in fits],
        FIGURES / f"phase2_weight_sparsity_{tag}.svg",
        title=f"Synthetic-control weight sparsity — {encoder}/{rownorm}, T0={T0}, m={mtag}",
        subtitle=(f"effective donor count 1/Σw² per patient;  "
                  f"m={mtag} donors offered, uniform start would give "
                  f"{mtag if mtag!='all' else panel.n-1}"),
        xlabel="effective number of donors  1/Σw²",
        bins=40,
    )
    figs.histogram(
        [f.pre_rmse for f in fits],
        FIGURES / f"phase2_pre_rmse_{tag}.svg",
        title=f"Pre-period fit (DIAGNOSTIC ONLY) — {encoder}/{rownorm}, T0={T0}, m={mtag}",
        subtitle="pre-period RMSE over t=0..%d; near-zero here does NOT imply a good forecast" % T0,
        xlabel="pre-period RMSE", bins=40,
    )


# --------------------------------------------------------------------------
# Cross-configuration summaries.
# --------------------------------------------------------------------------


def summarise(out: Dict[str, Any], args) -> str:
    """Headline / m-sensitivity / robustness blocks."""
    L = []
    cfg = out["configs"]

    def row(c, nm):
        return next((r for r in c["rows"] if r["method"] == nm), None)

    L.append("\n\n" + "#" * 100)
    L.append("# HEADLINE — does SC beat LVCF on the PRIMARY metric,")
    L.append("#            code-set F1 at matched cardinality K_i?")
    L.append("# K_i = the number of codes the target itself had at T0, so every method")
    L.append("# emits exactly K_i codes and LVCF decodes to exactly its own code set.")
    L.append("# Post-period RMSE is shown alongside as the SECONDARY probabilistic view:")
    L.append("# squared error is a proper scoring rule on a binary target, so it rewards a")
    L.append("# calibrated soft forecast over a hard 0/1 one whether or not the soft one")
    L.append("# names better codes.  SC-vs-LVCF on RMSE is not an apples-to-apples test.")
    L.append("#" * 100)
    L.append(f"\nPrimary configuration: T0={args.primary_t0} "
             f"(fit t=0..{args.primary_t0}, forecast t={args.primary_t0+1}), "
             f"m={args.primary_m}, paired bootstrap B={args.boot_B} seed={args.boot_seed}\n")
    hdr = (f"{'encoder/scaling':<22}{'method':<34}{'code F1':<10}"
           f"{'Δ F1 vs LVCF (95% CI)':<36}{'win%':<8}{'verdict (PRIMARY)':<24}"
           f"{'| post RMSE':<13}{'Δ RMSE vs LVCF'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for c in cfg:
        if c["T0"] != args.primary_t0 or c["m"] != str(args.primary_m):
            continue
        name = f"{c['encoder']}/{c['rownorm']}"
        for nm in ("sc", "sc_random", "nn1", "topk_mean", "cohort_mean",
                   "ridge"):
            r = row(c, nm)
            if r is None:
                continue
            v = r["vs_ref"]
            fv = r.get("f1_vs_ref")
            f1m = (r.get("f1") or {}).get("mean")
            if fv is None:
                d, wf, verdict = "—".ljust(36), "—", "(no code-set F1)"
            else:
                d = (f"{fv['mean_diff']:+.4f} [{fv['lo']:+.4f}, "
                     f"{fv['hi']:+.4f}]{fv['sig']}").ljust(36)
                wf = f"{100*fv['win_frac']:.1f}"
                verdict = ("BEATS LVCF on F1" if fv["beats_ref"] else
                           "LOSES to LVCF on F1" if fv["hi"] < 0 else
                           "no difference on F1")
            L.append(f"{name:<22}{bl.LABELS.get(nm, nm)[:33]:<34}"
                     f"{(f'{f1m:.4f}' if f1m is not None else '—'):<10}{d}{wf:<8}"
                     f"{verdict:<24}"
                     f"| {r['rmse']['mean']:<11.4f}"
                     f"{v['mean_diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{v['sig']}")
        r = row(c, "lvcf")
        f1m = (r.get("f1") or {}).get("mean")
        L.append(f"{name:<22}{'LVCF (reference)':<34}"
                 f"{(f'{f1m:.4f}' if f1m is not None else '—'):<10}"
                 f"{'—':<36}{'—':<8}{'(reference)':<24}"
                 f"| {r['rmse']['mean']:<11.4f}—")
        L.append("")

    # ---- shrinkage vs direction ----------------------------------------
    L.append("\n" + "#" * 100)
    L.append("# SHRINKAGE CHECK — RMSE is not scale-free.  Any average of donors sits")
    L.append("# nearer the cohort centroid than a real visit does, so it can win on RMSE")
    L.append("# by predicting a SMALLER vector rather than a better one.  Cosine ignores")
    L.append("# magnitude: better RMSE + worse cosine than LVCF == the win is shrinkage.")
    L.append("#" * 100)
    hdr = (f"{'encoder/scaling':<22}{'T0':<5}{'method':<14}{'RMSE':<10}"
           f"{'Δ RMSE vs LVCF':<18}{'cosine':<10}{'Δ cosine vs LVCF (95% CI)':<34}"
           f"{'|f|/|y|':<9}{'reads as'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for c in cfg:
        if c["m"] != str(args.primary_m):
            continue
        for r in sorted(c["rows"], key=lambda r: r["rmse"]["mean"]):
            nm = r["method"]
            cv = r.get("cosine_vs_ref")
            rv = r.get("vs_ref")
            if nm == "lvcf":
                reads, dc, dr = "(reference)", "—".ljust(34), "—".ljust(18)
            else:
                better_rmse = rv["beats_ref"]
                better_cos = cv["beats_ref"]
                worse_cos = cv["hi"] < 0
                reads = ("better direction AND smaller error" if better_rmse and better_cos
                         else "SHRINKAGE ONLY" if better_rmse and worse_cos
                         else "better RMSE, cosine indistinct" if better_rmse
                         else "no RMSE win")
                dc = f"{cv['mean_diff']:+.4f} [{cv['lo']:+.4f}, {cv['hi']:+.4f}]{cv['sig']}".ljust(34)
                dr = f"{rv['mean_diff']:+.4f}".ljust(18)
            L.append(f"{c['encoder']+'/'+c['rownorm']:<22}{c['T0']:<5}{nm:<14}"
                     f"{r['rmse']['mean']:<10.4f}{dr}{r['cosine']['mean']:<10.4f}{dc}"
                     f"{r['norm_ratio']:<9.3f}{reads}")
        L.append("")

    # ---- is the win just averaging? ------------------------------------
    L.append("\n" + "#" * 100)
    L.append("# IS THE WIN JUST AVERAGING?  SC against references OTHER than LVCF,")
    L.append("# on SECONDARY post-period RMSE.")
    L.append("# Under squared error any averaging shrinks variance, and one observed")
    L.append("# visit is a noisy target — so beating LVCF is NOT evidence that donor")
    L.append("# matching works. These paired comparisons are what test that.  `ridge`")
    L.append("# is the reference that matters most: it is the only one that LEARNS the")
    L.append("# pre-to-post map, so SC losing to it means SC is not buying anything a")
    L.append("# plain supervised regression does not already give.")
    L.append("#" * 100)
    hdr = (f"{'encoder/scaling':<22}{'T0':<5}{'m':<7}{'SC vs':<26}"
           f"{'Δ RMSE (95% CI)':<38}{'win%':<8}{'verdict'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for c in cfg:
        if c["m"] != str(args.primary_m):
            continue
        r = row(c, "sc")
        if r is None:
            continue
        for other in ("lvcf", "topk_mean", "cohort_mean", "sc_random", "ridge"):
            v = (r["vs_ref"] if other == "lvcf" else r.get("vs", {}).get(other))
            if v is None:
                continue
            verdict = ("SC better" if v["beats_ref"] else
                       "SC WORSE" if v["lo"] > 0 else "indistinguishable")
            L.append(f"{c['encoder']+'/'+c['rownorm']:<22}{c['T0']:<5}{c['m']:<7}"
                     f"{other:<26}"
                     f"{v['mean_diff']:+.5f} [{v['lo']:+.5f}, {v['hi']:+.5f}]{v['sig']:<14}"
                     f"{100*v['win_frac']:<8.1f}{verdict}")
        L.append("")

    # ---- how much of the win is free? ----------------------------------
    L.append("#" * 100)
    L.append("# DECOMPOSITION of SC's RMSE advantage over LVCF.")
    L.append("# The cohort mean uses NO patient identity at all, so whatever it earns is")
    L.append("# free.  'free share' = (LVCF - cohort_mean) / (LVCF - SC): the fraction of")
    L.append("# SC's headline win that a single constant vector already delivers.")
    L.append("#" * 100)
    hdr = (f"{'encoder/scaling':<22}{'T0':<5}{'LVCF':<10}{'cohort':<10}{'top-10':<10}"
           f"{'SC':<10}{'total gain':<12}{'free share':<13}{'matching adds'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for c in cfg:
        if c["m"] != str(args.primary_m):
            continue
        g = {nm: (row(c, nm) or {}).get("rmse", {}).get("mean", float("nan"))
             for nm in ("lvcf", "cohort_mean", "topk_mean", "sc")}
        total = g["lvcf"] - g["sc"]
        free = g["lvcf"] - g["cohort_mean"]
        share = free / total if total else float("nan")
        L.append(f"{c['encoder']+'/'+c['rownorm']:<22}{c['T0']:<5}{g['lvcf']:<10.4f}"
                 f"{g['cohort_mean']:<10.4f}{g['topk_mean']:<10.4f}{g['sc']:<10.4f}"
                 f"{total:<12.5f}{100*share:<13.1f}{100*(1-share):.1f}%")
    L.append("")

    # ---- primary metric, in full, plus outcome AUROC --------------------
    L.append("#" * 100)
    L.append("# PRIMARY METRIC IN FULL — code-set F1 at matched cardinality, every method,")
    L.append("# plus outcome AUROC at T0+1.  These are the two numbers a clinician would")
    L.append("# recognise.  Every method decodes to the SAME per-patient budget K_i, so")
    L.append("# this is the comparison that is not decided by the form of the prediction.")
    L.append("# Where the RMSE block above and this block disagree, THIS one is the finding")
    L.append("# and the RMSE difference is about the scoring rule.")
    L.append("#" * 100)
    hdr = (f"{'encoder/scaling':<22}{'T0':<5}{'method':<14}{'code F1':<10}"
           f"{'Δ F1 vs LVCF (95% CI)':<36}{'win%':<8}{'AUROC':<9}{'AUROC 95% CI':<18}{'verdict'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for c in cfg:
        if c["m"] != str(args.primary_m):
            continue
        for r in sorted(c["rows"], key=lambda r: -(r.get("f1") or {}).get("mean", -1)):
            nm = r["method"]
            f1 = r.get("f1")
            au = r.get("auroc_dense") or {}
            if f1 is None and not au:
                continue
            fv = r.get("f1_vs_ref")
            if nm == "lvcf":
                d, verdict = "(reference)".ljust(36), ""
                wf = "—"
            elif fv is None:
                d, verdict, wf = "—".ljust(36), "", "—"
            else:
                d = (f"{fv['mean_diff']:+.4f} [{fv['lo']:+.4f}, "
                     f"{fv['hi']:+.4f}]{fv['sig']}").ljust(36)
                verdict = ("beats LVCF on F1" if fv["beats_ref"]
                           else "LOSES to LVCF on F1" if fv["hi"] < 0
                           else "F1 indistinguishable")
                wf = f"{100*fv['win_frac']:.1f}"
            aus = (f"{au['auroc']:.3f}" if np.isfinite(au.get("auroc", np.nan))
                   else "n/a")
            auci = (f"[{au['lo']:.3f}, {au['hi']:.3f}]{au.get('sig_vs_chance','')}"
                    if np.isfinite(au.get("auroc", np.nan)) else "")
            # Code-set F1 is bag-only (it needs an interpretable vocabulary);
            # print an em dash rather than a bare nan for MedCPT.
            f1s = f"{f1['mean']:.4f}" if f1 else "—"
            L.append(f"{c['encoder']+'/'+c['rownorm']:<22}{c['T0']:<5}{nm:<14}"
                     f"{f1s:<10}{d}{wf:<8}{aus:<9}{auci:<18}{verdict}")
        L.append("")

    # ---- m sensitivity -------------------------------------------------
    L.append("\n" + "#" * 100)
    L.append("# m-SENSITIVITY — does the pre-filter size move the answer?")
    L.append("#" * 100)
    hdr = (f"{'encoder/scaling':<22}{'T0':<5}{'m':<7}{'SC RMSE':<11}"
           f"{'Δ vs LVCF (95% CI)':<34}{'win%':<8}{'eff donors':<12}{'pre-RMSE':<11}"
           f"{'non-conv'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for c in cfg:
        r = row(c, "sc")
        if r is None:
            continue
        v = r["vs_ref"]
        sp = r.get("sparsity", {})
        pre = r.get("pre_rmse_diagnostic", {})
        nc = c.get("convergence", {}).get("sc", {}).get("n_not_converged", 0)
        # Flag loudly: at large m the RAW binary bag design makes SLSQP
        # degenerate, and rows with failures must not be read as clean.
        ncs = f"{nc}/{c['n_targets']}" + ("  <-- READ WITH CARE" if nc else "")
        L.append(f"{c['encoder']+'/'+c['rownorm']:<22}{c['T0']:<5}{c['m']:<7}"
                 f"{r['rmse']['mean']:<11.4f}"
                 f"{v['mean_diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{v['sig']:<10}"
                 f"{100*v['win_frac']:<8.1f}{sp.get('eff_donors_ipr_mean', float('nan')):<12.1f}"
                 f"{pre.get('mean', float('nan')):<11.4f}{ncs}")

    # ---- method ranking across T0 --------------------------------------
    L.append("\n" + "#" * 100)
    L.append("# ROBUSTNESS — method ranking by mean post-period RMSE (best first)")
    L.append("# The ranking must NOT flip between T0=1 and T0=2.  If it does, the")
    L.append("# two-timestep pre-period is carrying the result and it is untrustworthy.")
    L.append("#" * 100)
    for c in cfg:
        if c["m"] != str(args.primary_m):
            continue
        order = sorted(c["rows"], key=lambda r: r["rmse"]["mean"])
        L.append(f"\n{c['encoder']}/{c['rownorm']}  T0={c['T0']}  m={c['m']}  "
                 f"n={c['n_targets']}")
        L.append("   " + "  >  ".join(
            f"{r['method']}({r['rmse']['mean']:.4f})" for r in order))

    # ---- pre-period diagnostic -----------------------------------------
    L.append("\n" + "#" * 100)
    L.append("# PRE-PERIOD OVERFITTING CHECK (diagnostic, not the validation metric)")
    L.append("# Correlation between pre-period fit and post-period accuracy across")
    L.append("# patients.  ~0 means a good pre-period fit tells you nothing.")
    L.append("#" * 100)
    hdr = (f"{'encoder/scaling':<22}{'T0':<5}{'m':<7}{'pre-RMSE mean':<16}"
           f"{'post-RMSE mean':<16}{'pearson r':<12}{'spearman r':<12}{'p(spearman)'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for c in cfg:
        r = row(c, "sc")
        if r is None or "pre_vs_post" not in r:
            continue
        pv = r["pre_vs_post"]
        L.append(f"{c['encoder']+'/'+c['rownorm']:<22}{c['T0']:<5}{c['m']:<7}"
                 f"{pv['pre_rmse_mean']:<16.5f}{pv['post_rmse_mean']:<16.5f}"
                 f"{pv.get('pearson_r', float('nan')):<12.3f}"
                 f"{pv.get('spearman_r', float('nan')):<12.3f}"
                 f"{pv.get('spearman_p', float('nan')):.2e}")

    # ---- solver convergence --------------------------------------------
    L.append("\n" + "#" * 100)
    L.append("# SOLVER CONVERGENCE — every solve is reported; none is dropped")
    L.append("#" * 100)
    hdr = (f"{'encoder/scaling':<22}{'T0':<5}{'m':<7}{'method':<12}{'n':<7}"
           f"{'non-conv':<10}{'nit mean':<11}{'max |simplex viol|'}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for c in cfg:
        for nm, cr in c.get("convergence", {}).items():
            L.append(f"{c['encoder']+'/'+c['rownorm']:<22}{c['T0']:<5}{c['m']:<7}"
                     f"{nm:<12}{cr['n']:<7}{cr['n_not_converged']:<10}"
                     f"{cr['nit_mean']:<11.1f}{cr['max_simplex_violation']:.2e}")
    return "\n".join(L)


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # `medcpt` is intentionally NOT a choice: it is no longer in CONFIGS, and
    # accepting it would silently select zero configurations.
    ap.add_argument("--encoder", choices=["bag", "all"], default="all")
    ap.add_argument("--scale", choices=["none", "global", "zscore"], default=None,
                    help="override state scaling for every config")
    ap.add_argument("--rownorm", choices=["raw", "rowl2", "all"], default="all")
    ap.add_argument("--t0", type=int, action="append", default=None,
                    help="repeatable; default 1, 2, 0")
    ap.add_argument("--m", action="append", default=None,
                    help="repeatable; integer or 'all'. default per-T0 grid")
    ap.add_argument("--primary-m", type=int, default=DEFAULT_M)
    ap.add_argument("--primary-t0", type=int, default=1)
    ap.add_argument("--k", type=int, default=bl.DEFAULT_K,
                    help="k for the top-k unweighted-mean baseline")
    ap.add_argument("--ridge-alpha", type=float, default=bl.DEFAULT_RIDGE_ALPHA,
                    help="L2 penalty for the ridge baseline; fixed a priori, "
                         "NOT tuned (see baselines.ridge)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot-B", type=int, default=ev.BOOT_B)
    ap.add_argument("--boot-seed", type=int, default=ev.BOOT_SEED)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--no-auroc", dest="auroc", action="store_false")
    ap.add_argument("--auroc-all-m", action="store_true",
                    help="outcome AUROC at every m, not just the primary m")
    ap.add_argument("--no-forecasts", dest="save_forecasts", action="store_false")
    ap.add_argument("--out-prefix", default="phase2")
    args = ap.parse_args()

    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    configs = [c for c in CONFIGS
               if (args.encoder in ("all", c[0]))
               and (args.rownorm in ("all", c[2]))]
    if args.scale:
        configs = [(e, args.scale, r) for e, _, r in configs]
    t0s = args.t0 or [1, 2, 0]
    m_over = None
    if args.m:
        m_over = tuple(None if str(x).lower() == "all" else int(x) for x in args.m)

    out: Dict[str, Any] = {"configs": [], "tables": [], "target_rows": [],
                           "weight_rows": []}
    meta = {
        "phase": 2, "run_started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "seed": args.seed, "boot_B": args.boot_B, "boot_seed": args.boot_seed,
        "topk_k": args.k, "ridge_alpha": args.ridge_alpha,
        "primary_metric": "code_f1_matched_cardinality",
        "primary_m": args.primary_m,
        "primary_t0": args.primary_t0,
        "slsqp_maxiter": SLSQP_MAXITER, "slsqp_ftol": SLSQP_FTOL,
        "weight_eps": WEIGHT_EPS, "jobs": args.jobs,
        "configs": [list(c) for c in configs], "t0_grid": t0s,
        "m_grid": {f"T{t}/{e}": [("all" if x is None else x)
                                 for x in (m_over or m_grid_for(t, e))]
                   for t in t0s for e in sorted({c[0] for c in configs})},
        "auroc_folds": ev.AUROC_FOLDS, "auroc_seed": ev.AUROC_SEED,
    }
    import scipy
    meta["scipy"] = scipy.__version__

    for T0 in t0s:
        for enc, scale, rn in configs:
            grid = m_over or m_grid_for(T0, enc)
            run_config(enc, scale, rn, T0, grid, args, out)

    summary = summarise(out, args)
    print(summary, flush=True)

    # ---- persist -------------------------------------------------------
    pref = args.out_prefix
    pd.DataFrame(out["target_rows"]).to_parquet(
        RESULTS / f"{pref}_targets.parquet", index=False)
    if out["weight_rows"]:
        pd.DataFrame(out["weight_rows"]).to_parquet(
            RESULTS / f"{pref}_weights.parquet", index=False)
    (RESULTS / f"{pref}_aggregate.json").write_text(
        json.dumps({"meta": meta, "configs": out["configs"]},
                   indent=2, sort_keys=False))
    meta["runtime_sec"] = round(time.time() - t_start, 1)
    (RESULTS / f"{pref}_headline.txt").write_text(
        "\n".join(out["tables"]) + "\n" + summary + "\n\n"
        + json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {RESULTS}/{pref}_targets.parquet "
          f"({len(out['target_rows'])} rows), {pref}_weights.parquet "
          f"({len(out['weight_rows'])} rows), {pref}_aggregate.json, "
          f"{pref}_headline.txt")
    print(f"figures -> {FIGURES}")
    print(f"total runtime {meta['runtime_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
