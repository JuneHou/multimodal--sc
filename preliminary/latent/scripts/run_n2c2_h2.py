"""H2 -- cross-space weight consistency on n2c2 (pre-registration, design 2).

    "Real patients, T0=1: fit SC in bag space (observable machinery, cosine
    donor schema, m=50) and in each latent space, same donor pool.  Compare
    per-patient weight vectors.  H2 supported if mean weight-cosine exceeds the
    permutation null's 95th percentile AND top-1 donor agreement >= 3x the null
    rate.  Refuted if weight-cosine is within the null's IQR."

The observable space is the addendum's: per-record UMLS concept bags, min-df
>= 5, built by `run_n2c2_concepts.py`, **no negation handling** (the
pre-registered v1 caveat).  Steps here:

    2   bag state matrix, binary, per-record (never cumulative; the mimic
        leakage check is rerun on it)
    3   for each of the 287 eligible targets: donor pool = the 50 nearest by
        cosine on the BAG-space flattened pre-period; `solve_simplex_qp` in
        (a) bag, (b) qwen3-768, (c) biolord-768 over that one pool; compare
        (b) vs (a) and (c) vs (a) against a permuted-target null.

Outputs
    latent/results/n2c2_cross_space.parquet        one row per (target, arm)
    latent/results/n2c2_cross_space_summary.json   H2 metrics, null, verdicts

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_n2c2_h2.py
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_LATENT = Path(__file__).resolve().parents[1]
_PRELIM = _LATENT.parent
for p in (str(_LATENT / "src"), str(_PRELIM / "mimic")):
    if p not in sys.path:
        sys.path.insert(0, p)

import n2c2_concepts as nc  # noqa: E402
from src.synthctl.fit import solve_simplex_qp  # noqa: E402
from src.vendored.stats import boot_continuous  # noqa: E402

DATA = _LATENT / "data"
RESULTS = _LATENT / "results"

T0 = 1
M_CROSS = 50
CROSS_SCHEMA = "cosine"
SEED = 0
BOOT_B = 2000
BOOT_SEED = 0
ARM_FILES = {"qwen3": DATA / "states_n2c2_qwen3.npz",
             "biolord": DATA / "states_n2c2_biolord.npz"}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def cosine_donors(P: np.ndarray, i: int, m: int) -> np.ndarray:
    """The `m` nearest donors to target `i` by cosine on the flattened
    pre-period block, ties broken by row order (canonical `patient_idx`).

    The pre-registration fixes the donor schema as cosine for design 2 and the
    pool as bag-space for every arm, so this is computed once per target on the
    bag block and reused verbatim by the latent fits.
    """
    nrm = np.linalg.norm(P, axis=1)
    denom = nrm * nrm[i]
    sim = np.divide(P @ P[i], denom, out=np.full(P.shape[0], -np.inf),
                    where=denom > 0)
    sim[i] = -np.inf
    return np.argsort(-sim, kind="stable")[:m]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--bags", default="n2c2_concept_bags.parquet",
                    help="bag file under latent/data. Everything else -- "
                         "targets, T0, m, donor schema, seed, null "
                         "construction, bootstrap -- is identical across bag "
                         "versions, which is what makes v1 and v2 comparable")
    args = ap.parse_args()

    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, Any] = {}

    # -- step 2: the bag state matrix -------------------------------------
    bags_df = pd.read_parquet(DATA / args.bags)
    corpus = pd.read_parquet(DATA / "n2c2_corpus.parquet")
    assert np.array_equal(bags_df["patient_idx"].to_numpy(),
                          corpus["patient_idx"].to_numpy())
    assert np.array_equal(bags_df["t"].to_numpy(), corpus["t"].to_numpy())
    bags: List[List[str]] = [list(b) for b in bags_df["cuis"]]

    t0 = time.time()
    X, vocab = nc.build_design_matrix(bags, min_df=nc.MIN_DF)
    leak = nc.validate_no_leakage(bags_df, bags, X, vocab)
    # Concepts per record rise with t (45 -> 64 over t=0..4) even inside the
    # >=3-record cohort.  That is NOT accumulation -- the t=0 rebuild below is
    # bit-identical -- it is that later notes are simply longer; the mean note
    # length by t is reported alongside so the two can be compared.
    leak["mean_n_chars_by_t"] = {
        int(k): float(v) for k, v in
        corpus.assign(nch=corpus["text"].str.len())
              .groupby("t")["nch"].mean().head(6).items()}
    timings["bag_matrix_seconds"] = round(time.time() - t0, 1)
    assert leak["t0_rebuilt_in_isolation_matches"], "bag states leak across t"
    print(f"[bag] X={X.shape} vocab={len(vocab)} leakage-ok="
          f"{leak['t0_rebuilt_in_isolation_matches']}", flush=True)

    ids, P_bag = nc.cohort_flat_blocks(X, bags_df, T0)
    n = len(ids)
    print(f"[bag] {n} eligible targets at T0={T0}, block dim {P_bag.shape[1]}",
          flush=True)

    # -- the two latent arms, row-aligned to the same (patient, t) --------
    lat: Dict[str, np.ndarray] = {}
    for arm, path in ARM_FILES.items():
        z = np.load(path, allow_pickle=True)
        assert np.array_equal(z["patient_idx"], corpus["patient_idx"].to_numpy())
        assert np.array_equal(z["t"], corpus["t"].to_numpy())
        Z = np.ascontiguousarray(z["X"], dtype=np.float64)
        _, lat[arm] = nc.cohort_flat_blocks(Z, bags_df, T0, ids=ids)
        print(f"[latent] {arm}: {Z.shape} -> block {lat[arm].shape}", flush=True)

    # -- step 3: one donor pool per target, three fits ---------------------
    rng = np.random.default_rng(SEED)
    perm = np.array([rng.choice(np.delete(np.arange(n), i)) for i in range(n)])

    W_bag: List[np.ndarray] = []
    W_lat: Dict[str, List[np.ndarray]] = {a: [] for a in ARM_FILES}
    diag: List[Dict[str, Any]] = []
    zero_norm_targets = 0
    t0 = time.time()
    for i in range(n):
        rows = cosine_donors(P_bag, i, M_CROSS)
        assert len(rows) == M_CROSS and i not in set(rows.tolist())
        if np.linalg.norm(P_bag[i]) == 0:
            zero_norm_targets += 1
        sb = solve_simplex_qp(P_bag[rows], P_bag[i])
        W_bag.append(sb.w)
        rec: Dict[str, Any] = {
            "i": i, "patient_idx": int(ids[i]),
            "bag_pre_rmse": float(np.sqrt(sb.sse / P_bag.shape[1])),
            "bag_max_w": float(sb.w.max()),
            "bag_eff_donors": float(1.0 / max(float(sb.w @ sb.w), 1e-300)),
            "bag_eff_donors_count": int((sb.w > 1e-4).sum()),
            "bag_status": int(sb.status),
            "bag_n_concepts_pre": float(P_bag[i].sum()),
        }
        for arm in ARM_FILES:
            sl = solve_simplex_qp(lat[arm][rows], lat[arm][i])
            W_lat[arm].append(sl.w)
            rec[f"{arm}_pre_rmse"] = float(np.sqrt(sl.sse / lat[arm].shape[1]))
            rec[f"{arm}_max_w"] = float(sl.w.max())
            rec[f"{arm}_eff_donors"] = float(1.0 / max(float(sl.w @ sl.w), 1e-300))
            rec[f"{arm}_eff_donors_count"] = int((sl.w > 1e-4).sum())
            rec[f"{arm}_status"] = int(sl.status)
        diag.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} targets in {time.time()-t0:.0f}s", flush=True)
    timings["fit_seconds"] = round(time.time() - t0, 1)
    print(f"[fit] {n} targets x {1+len(ARM_FILES)} spaces in "
          f"{timings['fit_seconds']}s", flush=True)

    rows_out: List[Dict[str, Any]] = []
    for arm in ARM_FILES:
        for i in range(n):
            wb, wl = W_bag[i], W_lat[arm][i]
            wn = W_lat[arm][int(perm[i])]      # permuted-target null
            rows_out.append({
                "arm": arm, "i": i, "patient_idx": int(ids[i]),
                "cos": _cos(wb, wl), "l1": float(np.abs(wb - wl).sum()),
                "top1_agree": bool(int(np.argmax(wb)) == int(np.argmax(wl))),
                "null_i": int(perm[i]),
                "null_cos": _cos(wb, wn),
                "null_l1": float(np.abs(wb - wn).sum()),
                "null_top1_agree": bool(int(np.argmax(wb)) == int(np.argmax(wn))),
                **{k: v for k, v in diag[i].items()
                   if k.startswith(("bag_", arm + "_"))},
            })
    df = pd.DataFrame(rows_out)

    npc = bags_df["n_concepts"].to_numpy().astype(float)
    summary: Dict[str, Any] = {
        "stage": "H2 (cross-space weight consistency), n2c2",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bags_file": str(DATA / args.bags),
        "n_targets": n, "T0": T0, "m": M_CROSS, "donor_schema": CROSS_SCHEMA,
        "donor_pool_space": "bag (UMLS concept bags), identical pool in all "
                            "three spaces",
        "bag_dim": int(P_bag.shape[1]), "bag_vocab": int(len(vocab)),
        "min_df": nc.MIN_DF,
        "seed": SEED, "boot_B": BOOT_B, "boot_seed": BOOT_SEED,
        "negation_handling": (
            "negspacy 1.0.4 NegEx, en_clinical termset (bags v2)"
            if "_v2" in args.bags else
            "NONE (pre-registered v1 caveat, addendum)"),
        "leakage_check": leak,
        "zero_norm_bag_targets": int(zero_norm_targets),
        "concepts_per_record": {
            "mean": float(npc.mean()), "std": float(npc.std(ddof=1)),
            "min": float(npc.min()), "max": float(npc.max()),
            "median": float(np.median(npc)),
            "q25": float(np.quantile(npc, .25)),
            "q75": float(np.quantile(npc, .75)),
            "n_zero": int((npc == 0).sum()),
        },
        "null": ("weights of a randomly chosen OTHER target with the same pool "
                 "size (every pool is m=50), one draw per target at seed 0, "
                 "compared positionally -- which destroys exactly the donor "
                 "correspondence the real comparison relies on"),
        "arms": {},
    }
    for arm in ARM_FILES:
        g = df[df["arm"] == arm]
        c, ncos = g["cos"].to_numpy(), g["null_cos"].to_numpy()
        clo, chi = boot_continuous(c, B=BOOT_B, seed=BOOT_SEED)
        nlo, nhi = boot_continuous(ncos, B=BOOT_B, seed=BOOT_SEED)
        l1lo, l1hi = boot_continuous(g["l1"].to_numpy(), B=BOOT_B, seed=BOOT_SEED)
        p95 = float(np.percentile(ncos, 95))
        q1, q3 = float(np.percentile(ncos, 25)), float(np.percentile(ncos, 75))
        t1, nt1 = float(g["top1_agree"].mean()), float(g["null_top1_agree"].mean())
        sup = bool(float(c.mean()) > p95 and
                   (t1 >= 3 * nt1 if nt1 > 0 else t1 > 0))
        ref = bool(q1 <= float(c.mean()) <= q3)
        summary["arms"][arm] = {
            "mean_weight_cosine": float(c.mean()),
            "weight_cosine_ci": [float(clo), float(chi)],
            "median_weight_cosine": float(np.median(c)),
            "mean_l1": float(g["l1"].mean()), "l1_ci": [float(l1lo), float(l1hi)],
            "top1_agreement": t1,
            "null_mean_cosine": float(ncos.mean()),
            "null_cosine_ci": [float(nlo), float(nhi)],
            "null_cosine_p95": p95,
            "null_cosine_iqr": [q1, q3],
            "null_mean_l1": float(g["null_l1"].mean()),
            "null_top1_agreement": nt1,
            "top1_ratio_vs_null": (t1 / nt1) if nt1 > 0 else float("inf"),
            "top1_threshold_3x_null": 3 * nt1,
            "mean_bag_pre_rmse": float(g["bag_pre_rmse"].mean()),
            "mean_latent_pre_rmse": float(g[f"{arm}_pre_rmse"].mean()),
            "mean_bag_eff_donors_ipr": float(g["bag_eff_donors"].mean()),
            "mean_latent_eff_donors_ipr": float(g[f"{arm}_eff_donors"].mean()),
            "mean_bag_eff_donors_count": float(g["bag_eff_donors_count"].mean()),
            "mean_latent_eff_donors_count": float(
                g[f"{arm}_eff_donors_count"].mean()),
            "mean_bag_max_w": float(g["bag_max_w"].mean()),
            "mean_latent_max_w": float(g[f"{arm}_max_w"].mean()),
            "n_nonconverged_bag": int((g["bag_status"] != 0).sum()),
            "n_nonconverged_latent": int((g[f"{arm}_status"] != 0).sum()),
            "supported": sup, "refuted": ref,
            "verdict": ("supported" if sup else
                        "refuted (within null IQR)" if ref else
                        "neither (above null IQR, criteria not both met)"),
        }
    summary["deviations"] = [{
        "item": "extraction interpreter",
        "pre_registered": "QuickUMLS over Jun's UTS Metathesaurus download",
        "done": ("QuickUMLS 1.4.2 in a dedicated venv "
                 "/data/wang/junh/envs/quickumls (python 3.10.19), driven from "
                 "medrag by subprocess over JSONL"),
        "reason": ("quickumls is absent from medrag and the instruction is that "
                   "nothing may be installed into medrag. The venv carries "
                   "spacy 3.7.5 + en_core_web_sm 3.7.1, quickumls-simstring "
                   "1.1.5.post1, leveldb 0.201 (quickumls/toolbox.py imports it "
                   "unconditionally even for the unqlite backend), unqlite "
                   "1.2.0."),
    }, {
        "item": "QuickUMLS index",
        "pre_registered": "deterministic index",
        "done": ("the PRE-BUILT index at /data/wang/junh/datasets/quickUMLS "
                 "(unqlite backend, ENG) was used unchanged -- no rebuild"),
        "reason": ("probed with a test sentence before any run and found "
                   "compatible with quickumls 1.4.2: congestive heart failure "
                   "C0018802/T047, type 2 diabetes mellitus C0011860/T047, "
                   "colonoscopy C0009378/T060, lisinopril C0065374/T116+T121, "
                   "metformin C0025598/T121, all at similarity 1.0"),
    }, {
        "item": "semantic types",
        "done": ("the 15 TUIs specified for this run: disorders T046-T050, "
                 "T184, T191; procedures T059-T061; drugs/chemicals T116, "
                 "T121, T123, T195, T200"),
        "note": ("the addendum's word 'finding' is carried by T184 (Sign or "
                 "Symptom) in this list; the broad T033 (Finding) is NOT "
                 "included"),
    }, {
        "item": "match similarity floor",
        "done": ("QuickUMLS ran at threshold 0.9 Jaccard; 38 of 101,222 "
                 "returned matches (0.038%) carry a recomputed similarity "
                 "below 0.9 (e.g. 'insulins'->'insulins' 0.83) and were kept"),
        "reason": ("QuickUMLS recomputes similarity after the simstring "
                   "retrieval; filtering them post hoc would be a threshold "
                   "not in the pre-registration"),
    }] + ([{
        "item": "negation",
        "done": ("NegEx (negspacy 1.0.4, en_clinical termset) over the v1 "
                 "QuickUMLS spans; negated matches dropped"),
        "reason": ("the addendum's ONE permitted follow-up, triggered because "
                   "H2 was refuted on the v1 bags. The QuickUMLS matches are "
                   "re-used unchanged -- no re-extraction -- so v1 and v2 "
                   "differ only in which matches survive."),
    }, {
        "item": "known extraction false positive",
        "done": "C0019602 (histidine) dropped in v2",
        "reason": ("all 1,026 v1 matches of that CUI are the pronoun 'his', so "
                   "dropping the CUI and dropping the artifact are the same "
                   "operation on this corpus"),
    }] if "_v2" in args.bags else [{
        "item": "negation",
        "done": "NONE",
        "reason": ("pre-registered v1 caveat (addendum): negated concepts enter "
                   "the bag; a NegEx-filtered v2 is the one permitted follow-up "
                   "before interpreting an H2 failure"),
    }, {
        "item": "known extraction false positive",
        "done": ("the pronoun 'his' matches C0019602 histidine (T116/T121/T123) "
                 "and so appears in most records; it is left in"),
        "reason": ("no stop-list is pre-registered, and a concept present in "
                   "nearly every record is a near-constant column that the "
                   "convex fit cannot use to discriminate donors"),
    }])
    summary["negative_control_random_bags"] = (
        "before the real bags existed, the same runner was executed on "
        "randomly generated bags (400-CUI vocabulary, 5-60 concepts per "
        "record, seed 1): mean weight-cosine 0.313 (qwen3) / 0.295 (biolord) "
        "against null IQRs [0.235, 0.382] / [0.222, 0.368], top-1 agreement "
        "1.4% both arms -- i.e. the machinery returns 'refuted' on noise, and "
        "the pipeline does not manufacture agreement")
    summary["timings"] = timings
    summary["wall_clock_seconds"] = round(time.time() - t_start, 1)

    sfx = args.out_suffix
    df.to_parquet(RESULTS / f"n2c2_cross_space{sfx}.parquet", index=False)
    (RESULTS / f"n2c2_cross_space_summary{sfx}.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary["arms"], indent=2), flush=True)
    print(f"[H2] total {time.time()-t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
