#!/usr/bin/env python
"""Phase 1 runner: build per-visit state matrices for latent synthetic control.

Builds both encoders (bag-of-codes, MedCPT), writes the state artifacts to
data/derived/, runs the sanity checks, and writes a JSON report.

    /data/wang/junh/envs/medrag/bin/python \
        preliminary/mimic/scripts/run_phase1_states.py

Exits non-zero if any hard check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def pick_device(requested: str, min_free_mib: int = 8000) -> str:
    """Choose a device, respecting other users of the box.

    GPUs here are contended; MedCPT is 110M params and runs fine on CPU.  Only
    take a GPU if one has real headroom, and take the emptiest one by pinning
    CUDA_VISIBLE_DEVICES (the vendored `embed` only understands the literal
    string 'cuda', so device selection has to happen through the environment).
    """
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()
        free = [(int(l.split(",")[1]), int(l.split(",")[0])) for l in out if l.strip()]
    except Exception as exc:  # pragma: no cover
        print(f"  nvidia-smi unavailable ({exc}); using CPU", flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    free.sort(reverse=True)
    if requested.startswith("cuda") or requested == "auto":
        want = None
        if ":" in requested:
            want = int(requested.split(":")[1])
        best_free, best_idx = free[0] if want is None else (
            dict((i, f) for f, i in free)[want], want
        )
        if requested == "auto" and best_free < min_free_mib:
            print(
                f"  emptiest GPU (cuda:{best_idx}) has only {best_free} MiB free "
                f"(< {min_free_mib}); using CPU",
                flush=True,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
        print(f"  using physical cuda:{best_idx} ({best_free} MiB free)", flush=True)
        return "cuda"
    raise ValueError(requested)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeline", type=Path, default=None)
    p.add_argument("--derived-dir", type=Path, default=None)
    p.add_argument("--min-df", type=int, default=None)
    p.add_argument("--t0", type=int, default=1, help="primary pre-period cutoff")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-smell", type=int, default=5, help="patients in the NN check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-medcpt", action="store_true")
    p.add_argument(
        "--skip-truncation-probe",
        action="store_true",
        help="skip the extra naive-truncation embedding pass used to measure "
             "what chunking buys",
    )
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    device = pick_device(args.device) if not args.skip_medcpt else "cpu"

    import numpy as np
    import pandas as pd

    from src.datasets import state as st

    timeline = args.timeline or st.DEFAULT_TIMELINE
    derived = args.derived_dir or st.DEFAULT_DERIVED
    min_df = st.DEFAULT_MIN_DF if args.min_df is None else args.min_df
    report_path = args.report or Path(derived) / "phase1_states_report.json"
    T0 = args.t0
    rng = np.random.default_rng(args.seed)

    rep: dict = {"config": {
        "timeline": str(timeline), "derived_dir": str(derived), "min_df": min_df,
        "T0": T0, "device": device, "seed": args.seed,
        "model": st.MEDCPT_MODEL, "maxlen": st.MEDCPT_MAXLEN,
        "batch": st.MEDCPT_BATCH, "dtype": str(st.STATE_DTYPE.__name__),
    }}

    print(f"== reading {timeline}")
    df = pd.read_parquet(timeline)
    # Pin row order: canonical id order, then t.  Row order is not load-bearing
    # (StateSet re-sorts by t per subject) but the saved matrix should be
    # byte-reproducible across runs.
    df = (
        df.assign(_k=df["subject_id"].astype(str).map(st._id_key))
        .sort_values(["_k", "t"])
        .drop(columns="_k")
        .reset_index(drop=True)
    )
    counts = df.groupby("subject_id").size()
    rep["cohort"] = {
        "n_visits": int(len(df)), "n_subjects": int(df.subject_id.nunique()),
        "n_ge3_visits": int((counts >= 3).sum()),
        "n_ge4_visits": int((counts >= 4).sum()),
        "label_prevalence": float(df.label.mean()),
    }
    print(f"   {rep['cohort']}")
    ok_cohort = rep["cohort"]["n_ge3_visits"] == 749 and rep["cohort"]["n_ge4_visits"] == 348
    rep["cohort"]["matches_expected_749_348"] = bool(ok_cohort)

    # ---------------- encoder (a): bag of codes ----------------------------
    print("== building bag-of-codes states")
    X_bag, vocab, index = st.build_bag_states(df, min_df=min_df)
    rep["bag"] = {"min_df": min_df, "vocab_size": len(vocab), **st.sparsity(X_bag)}
    print(f"   {rep['bag']}")

    # min_df sensitivity: how much does the vocabulary floor cost?
    X_bag1, vocab1, _ = st.build_bag_states(df, min_df=1)
    rep["bag"]["vocab_size_min_df_1"] = len(vocab1)
    rep["bag"]["codes_dropped_by_min_df"] = len(vocab1) - len(vocab)

    st.save_states("bag", X_bag, index, derived_dir=derived, vocab=vocab,
                   meta={"min_df": min_df, "source": str(timeline),
                         "binary": True, "prefixed_sections": ["c:", "p:", "d:"]})

    print("== leakage check (per-visit vs cumulative)")
    rep["leakage"] = st.validate_no_leakage(df, X_bag, vocab, index)
    print(f"   {rep['leakage']}")

    # ---------------- encoder (b): MedCPT ----------------------------------
    if not args.skip_medcpt:
        print("== building MedCPT states")
        cache = Path(derived) / ".emb_cache" / "medcpt_visits.pkl"
        X_med, index_med, mstats = st.build_medcpt_states(
            df, cache_path=cache, device=device)
        assert index_med.equals(index)
        rep["medcpt"] = {"model": st.MEDCPT_MODEL, **mstats, **st.sparsity(X_med)}
        print(f"   {st.sparsity(X_med)}")
        st.save_states("medcpt", X_med, index, derived_dir=derived,
                       meta={"model": st.MEDCPT_MODEL, "source": str(timeline),
                             "pooling": "CLS", "l2_normalised": True,
                             "chunked_mean_pooled": True,
                             "maxlen": st.MEDCPT_MAXLEN,
                             "token_stats": mstats})

        if not args.skip_truncation_probe:
            print("== truncation probe (what chunking buys)")
            X_tr = st.build_medcpt_truncated(df, cache_path=cache, device=device)
            cos = (X_med * X_tr).sum(axis=1)
            rep["medcpt_truncation"] = {
                "cosine_chunked_vs_truncated_mean": float(cos.mean()),
                "cosine_chunked_vs_truncated_min": float(cos.min()),
                "n_rows_cos_below_0.99": int((cos < 0.99).sum()),
                "n_rows_cos_below_0.95": int((cos < 0.95).sum()),
            }
            print(f"   {rep['medcpt_truncation']}")

    # ---------------- reload through the public API ------------------------
    print("== reloading via public API and exercising the donor interface")
    sets = {"bag": st.load_states("bag", derived_dir=derived)}
    if not args.skip_medcpt:
        sets["medcpt"] = st.load_states("medcpt", derived_dir=derived)

    ss = sets["bag"]
    assert np.array_equal(ss.X, X_bag), "round-trip mismatch for bag states"
    eligible = ss.eligible_subjects(T0)
    rep["donor_pool"] = {
        "T0": T0, "n_eligible": len(eligible),
        "matches_cohort_ge3": len(eligible) == rep["cohort"]["n_ge3_visits"],
    }
    tgt = eligible[0]
    D, ids = ss.donor_pool(T0, exclude=tgt)
    rep["donor_pool"].update({
        "donor_array_shape": list(D.shape),
        "n_donor_ids": len(ids),
        "target_excluded": tgt not in ids,
        "ids_are_id_ordered": ids == st.sorted_ids(ids),
        "pre_period_shape": list(ss.pre_period(tgt, T0).shape),
    })
    Dp, Dn, ids2 = ss.donor_matrix(T0, exclude=tgt)
    rep["donor_pool"]["donor_matrix_shapes"] = [list(Dp.shape), list(Dn.shape)]
    assert ids2 == ids
    # row alignment: D[j, t] must equal the donor's own state at t
    j = len(ids) // 2
    assert np.array_equal(D[j, T0 + 1], ss.state(ids[j], T0 + 1))
    rep["donor_pool"]["row_alignment_verified"] = True
    print(f"   {rep['donor_pool']}")

    # ---------------- scaling sensitivity ----------------------------------
    print("== scaling sensitivity")
    targets = list(rng.choice(eligible, size=min(40, len(eligible)), replace=False))
    rep["scaling"] = {}
    for enc, s0 in sets.items():
        s_glob = st.load_states(enc, derived_dir=derived, scale="global")
        s_z = st.load_states(enc, derived_dir=derived, scale="zscore")
        rep["scaling"][enc] = {
            "none_vs_global_top10": st.neighbour_overlap(s0, s_glob, T0, targets, k=10),
            "none_vs_zscore_top10": st.neighbour_overlap(s0, s_z, T0, targets, k=10),
            "none_vs_zscore_top3": st.neighbour_overlap(s0, s_z, T0, targets, k=3),
            "row_norm_mean_none": s0.meta["scaling_params"]["row_norm_mean"],
            "distance_profile": st.distance_profile(s0, T0, targets, k=10),
        }
        print(f"   {enc}: {rep['scaling'][enc]}")

    # bag min_df sensitivity on neighbour ordering
    ss1 = st.StateSet(encoder="bag", X=X_bag1, index=index, vocab=vocab1)
    rep["bag"]["min_df_1_vs_5_top10_overlap"] = st.neighbour_overlap(
        ss, ss1, T0, targets, k=10)

    if len(sets) == 2:
        rep["cross_encoder_top10_overlap"] = st.neighbour_overlap(
            sets["bag"], sets["medcpt"], T0, targets, k=10)
        print(f"   cross-encoder: {rep['cross_encoder_top10_overlap']}")

    # ---------------- nearest-neighbour smell test -------------------------
    print("== nearest-neighbour smell test")
    lookup = df.set_index(["subject_id", "t"])
    smell = []
    picks = list(rng.choice(eligible, size=min(args.n_smell, len(eligible)),
                            replace=False))
    for s in picks:
        entry = {"target": s, "target_visits": ss.n_visits(s),
                 "target_conditions_t0": st._codes(lookup.loc[(s, 0), "conditions"]),
                 "target_conditions_t1": st._codes(lookup.loc[(s, 1), "conditions"]),
                 "encoders": {}}
        for enc, sset in sets.items():
            nn = st.nearest_donors(sset, s, T0, k=3)
            entry["encoders"][enc] = [
                {"donor": dsid, "dist": round(dd, 4),
                 "conditions_t0": st._codes(lookup.loc[(dsid, 0), "conditions"]),
                 "conditions_t1": st._codes(lookup.loc[(dsid, 1), "conditions"])}
                for dsid, dd in nn
            ]
        smell.append(entry)
    rep["nn_smell_test"] = smell
    for e in smell:
        print(f"\n  TARGET {e['target']} ({e['target_visits']} visits)")
        print(f"    t0: {', '.join(e['target_conditions_t0'][:8])}")
        print(f"    t1: {', '.join(e['target_conditions_t1'][:8])}")
        for enc, nns in e["encoders"].items():
            print(f"    -- {enc} --")
            for n in nns:
                print(f"      {n['donor']} d={n['dist']}")
                print(f"        t0: {', '.join(n['conditions_t0'][:8])}")
                print(f"        t1: {', '.join(n['conditions_t1'][:8])}")

    # ---------------- gate -------------------------------------------------
    checks = {
        "cohort_749_348": ok_cohort,
        "no_cumulative_leakage": rep["leakage"]["t0_rebuilt_in_isolation_matches"],
        "donor_pool_matches_cohort": rep["donor_pool"]["matches_cohort_ge3"],
        "target_excluded_from_donors": rep["donor_pool"]["target_excluded"],
        "donor_ids_ordered": rep["donor_pool"]["ids_are_id_ordered"],
    }
    if not args.skip_medcpt:
        checks["medcpt_dim_768"] = rep["medcpt"]["d"] == 768
        checks["all_chunks_within_maxlen"] = (
            rep["medcpt"]["chunk_tokens_max"] <= st.MEDCPT_MAXLEN)
        checks["medcpt_rows_unit_norm"] = bool(
            np.allclose(np.linalg.norm(sets["medcpt"].X, axis=1), 1.0, atol=1e-4))
    rep["checks"] = checks
    rep["passed"] = all(checks.values())

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    print(f"\n== checks: {checks}")
    print(f"== report -> {report_path}")
    print("== PASS" if rep["passed"] else "== FAIL")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
