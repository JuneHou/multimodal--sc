#!/usr/bin/env python
"""Phase 0 runner: rebuild the corrupted MIMIC-III time axis.

Thin wrapper around `src.datasets.rebuild_timeline.run()`.  Prints the headline
numbers and writes a JSON report next to the parquet.  Exits non-zero if the
validation gate fails, so downstream phases can be scripted off it.

    /data/wang/junh/envs/medrag/bin/python \
        preliminary/mimic/scripts/run_phase0_rebuild.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import rebuild_timeline as rt  # noqa: E402


def _default(o):
    try:
        return str(o)
    except Exception:
        return repr(o)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mimic3-root", type=Path, default=rt.MIMIC3_ROOT)
    p.add_argument("--cohort-json", type=Path, default=rt.COHORT_JSON)
    p.add_argument("--resource-dir", type=Path, default=rt.DEFAULT_RESOURCE_DIR)
    p.add_argument("--output", type=Path, default=rt.DEFAULT_OUTPUT)
    p.add_argument("--kare-pickle", type=Path, default=rt.KARE_SAMPLE_PKL)
    p.add_argument(
        "--cohort-mode",
        choices=rt.COHORT_MODES,
        default=rt.DEFAULT_COHORT_MODE,
        help=(
            "rederived (default): re-run the visit selection on the corrected "
            "axis, which restores the cohort to roughly its pre-fix size -- "
            "9,582 visits / 6,112 subjects / 749 with >=3 visits.  This is "
            "what the parquet on disk was built with and what Phase 1 and 2 "
            "consume.  kare_recovered: re-sort exactly the visits KARE "
            "selected (the literal Phase 0 spec); roughly halves the cohort."
        ),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=rt.DEFAULT_OUTPUT.parent / "phase0_rebuild_report.json",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="run everything but do not write the parquet",
    )
    args = p.parse_args()

    res = rt.run(
        mimic3_root=args.mimic3_root,
        cohort_json=args.cohort_json,
        resource_dir=args.resource_dir,
        output_path=args.output,
        kare_pickle=args.kare_pickle,
        cohort_mode=args.cohort_mode,
        write=not args.no_write,
    )

    report = {
        "passed": res.passed,
        "validation": res.validation,
        "spot_check": res.spot_check,
        "name_drift": res.drift,
        "damage": {k: v for k, v in res.damage.items() if k != "per_patient_tau"},
        "output_path": str(res.output_path) if res.output_path else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, default=_default)

    v, d = res.validation, res.damage
    print("\n" + "=" * 72)
    print("PHASE 0 SUMMARY")
    print("=" * 72)
    print(f"GATE: {'PASS' if res.passed else 'FAIL'}   "
          f"match rate = {100 * v['match_rate']:.4f}%  "
          f"(threshold {100 * rt.MATCH_RATE_GATE:.2f}%)")
    for lvl, desc in (
        ("l1", "hadm sequence vs PyHealth pickle"),
        ("l2", "code sets vs PyHealth pickle"),
        ("l3", "name sets vs published cohort"),
    ):
        r = v.get(lvl)
        if r is None:
            print(f"  {lvl.upper()} {desc:38s} SKIPPED")
        else:
            n = r.get("n_patients") or r.get("n_admissions") or r.get("n_entries")
            print(f"  {lvl.upper()} {desc:38s} {r['n_match']}/{n} = "
                  f"{100 * r['rate']:.4f}%")
    sc = res.spot_check
    print(f"\nspot check subject {sc.get('subject_id')}: "
          f"{'OK' if sc.get('matches') else 'MISMATCH'}  "
          f"recovered={sc.get('recovered_visit_order')}")

    kt = d.get("kendall_tau", {})
    print(f"\nKendall tau (hadm order vs chronological order), "
          f"n={kt.get('n_patients_with_ge2_visits')} multi-visit patients:")
    print(f"  mean={kt.get('mean')}  median={kt.get('median')}  "
          f"std={kt.get('std')}")
    print(f"  frac tau==1 (already correct): {kt.get('frac_tau_eq_1')}")
    print(f"  frac tau<1  (scrambled):       {kt.get('frac_tau_lt_1')}")
    ordr = d.get("ordering", {})
    n_multi = kt.get("n_patients_with_ge2_visits") or 1
    print(f"  patients with any ordering change: "
          f"{ordr.get('n_patients_order_changed')}/{ordr.get('n_patients')} all "
          f"({100 * (ordr.get('frac_patients_order_changed') or 0):.2f}%)  |  "
          f"{ordr.get('n_patients_order_changed_among_multivisit')}/{n_multi} "
          f"multi-visit "
          f"({100 * (ordr.get('n_patients_order_changed_among_multivisit') or 0) / n_multi:.2f}%)")

    pre = d.get("cohort_pre_fix", {})
    post = d.get("cohort_post_fix", {})
    alt = d.get("cohort_post_fix_rederived", {})
    print("\ncohort size (patients with >=3 / >=4 visits):")
    print(f"  pre-fix                 : {pre.get('n_ge3_visits')} / "
          f"{pre.get('n_ge4_visits')}")
    if post:
        print(f"  post-fix [WRITTEN: {args.cohort_mode}] : "
              f"{post.get('n_ge3_visits')} / {post.get('n_ge4_visits')}  "
              f"({post.get('n_visits')} visits / {post.get('n_patients')} patients)")
    if alt:
        print(f"  post-fix [rederived]    : {alt.get('n_ge3_visits')} / "
              f"{alt.get('n_ge4_visits')}  ({alt.get('n_visits')} visits / "
              f"{alt.get('n_patients')} patients)")
    mono_pre = d.get("monotonicity_pre_fix", {})
    mono_post = d.get("monotonicity_post_fix", {})
    print("\nnon-monotone label sequences (a death followed by a survivor):")
    print(f"  pre-fix : {mono_pre.get('n_patients_with_death_then_survivor')}")
    if mono_post:
        print(f"  post-fix: {mono_post.get('n_patients_with_death_then_survivor')}  "
              f"-> {'CLEAN' if mono_post.get('clean') else 'STILL BROKEN'}")
        print(f"  (separately, {mono_post.get('n_patients_with_consecutive_death_flags')}"
              f" patients have consecutive HOSPITAL_EXPIRE_FLAG=1 admissions -- "
              f"a MIMIC-III source anomaly, not an ordering bug)")
    print(f"\nreport: {args.report}")
    print(f"parquet: {res.output_path}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
