"""
L1 runner: embed `latent/data/rendered_visits.parquet` with both arms.

Writes
    latent/data/states_trial_qwen3_4096.npz   native Qwen3-Embedding-8B output
    latent/data/states_trial_qwen3.npz        MRL-truncated 768-d  (PRIMARY)
    latent/data/states_trial_biolord.npz      BioLORD-2023 768-d
    latent/results/l1_report.json             every pre-registered check

Checks reported: unit norms; byte-determinism of a 100-row re-encode; token
lengths against each encoder's limit; and the neighbour smell test (5 fixed
patients, seed 0, top-3 nearest OTHER-patient visits in each arm, with texts).

    /data/wang/junh/envs/medrag/bin/python latent/scripts/run_embed.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_LATENT = Path(__file__).resolve().parents[1]
_PRELIM = _LATENT.parent
for p in (str(_LATENT / "src"), str(_PRELIM / "mimic"),
          str(_PRELIM / "observable" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _pick_gpu() -> str:
    """Freest GPU by used memory, chosen BEFORE torch is imported."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    rows = [tuple(int(x) for x in ln.split(",")) for ln in out.strip().split("\n")]
    rows.sort(key=lambda r: (r[1], r[0]))
    return str(rows[0][0])


GPU = os.environ.get("CUDA_VISIBLE_DEVICES") or _pick_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import encode as enc  # noqa: E402

IN_PARQUET = _LATENT / "data" / "rendered_visits.parquet"
OUT = {
    "qwen3_4096": _LATENT / "data" / "states_trial_qwen3_4096.npz",
    "qwen3": _LATENT / "data" / "states_trial_qwen3.npz",
    "biolord": _LATENT / "data" / "states_trial_biolord.npz",
}
REPORT = _LATENT / "results" / "l1_report.json"

N_SMELL_PATIENTS = 5
SMELL_SEED = 0
TOPK = 3
BATCH = {"qwen3": 64, "biolord": 256}


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _codes(text: str, header: str) -> set:
    """Parse one section back out of a rendered string (smell-test only)."""
    for part in text.split(". "):
        if part.startswith(header + ": "):
            body = part[len(header) + 2:].rstrip(".")
            return set() if body == "none" else set(body.split("; "))
    return set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def main() -> int:
    t_start = time.time()
    df = pd.read_parquet(IN_PARQUET)
    texts: List[str] = df["text"].tolist()
    pid = df["patient_idx"].to_numpy(dtype=np.int32)
    tt = df["t"].to_numpy(dtype=np.int32)
    n = len(texts)

    import torch

    report: Dict[str, Any] = {
        "stage": "L1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu_visible": GPU,
        "gpu_name": torch.cuda.get_device_name(0),
        "n_rows": n,
        "input_parquet": str(IN_PARQUET),
        "vendored_embed_not_used": ("mimic/src/vendored/embedding.py::embed is "
                                    "CLS-pooling transformers code; Qwen3 is "
                                    "last-token pooled and BioLORD is mean "
                                    "pooled, so sentence-transformers is used "
                                    "directly for both arms"),
        "arms": {},
    }

    raw: Dict[str, np.ndarray] = {}
    for arm in ("qwen3", "biolord"):
        t0 = time.time()
        model = enc.load_encoder(arm, device="cuda")
        tok = model.tokenizer
        lens = np.array([len(x) for x in
                         tok(texts, add_special_tokens=True)["input_ids"]])
        max_seq = int(model.max_seq_length)
        n_trunc = int((lens > max_seq).sum())
        assert n_trunc == 0, (
            f"{arm}: {n_trunc} texts exceed max_seq_length={max_seq}")

        X = enc.encode_texts(model, texts, batch_size=BATCH[arm])
        Xn = enc.l2_normalise(X)
        t_enc = time.time() - t0

        # byte-determinism: encode the same 100 rows twice, in their own call.
        head = texts[:100]
        a = enc.l2_normalise(enc.encode_texts(model, head, batch_size=BATCH[arm],
                                              show_progress=False))
        b = enc.l2_normalise(enc.encode_texts(model, head, batch_size=BATCH[arm],
                                              show_progress=False))
        det_identical = _sha(a) == _sha(b)
        # informational: the same rows as part of the full-corpus pass differ
        # only by padding/batch composition.
        vs_full = float(np.abs(a - Xn[:100]).max())

        norms = np.linalg.norm(Xn, axis=1)
        report["arms"][arm] = {
            "model": enc.QWEN3_REPO if arm == "qwen3" else enc.BIOLORD_REPO,
            "dtype": str(next(model.parameters()).dtype),
            "max_seq_length_used": max_seq,
            "native_dim": int(Xn.shape[1]),
            "batch_size": BATCH[arm],
            "token_length": {"max": int(lens.max()), "mean": float(lens.mean()),
                             "p95": float(np.percentile(lens, 95)),
                             "n_over_max_seq_length": n_trunc},
            "unit_norm": {"min": float(norms.min()), "max": float(norms.max()),
                          "max_abs_dev_from_1": float(np.abs(norms - 1).max())},
            "determinism_100row_reencode_byte_identical": bool(det_identical),
            "determinism_sha256": _sha(a),
            "reencode_vs_full_corpus_pass_max_abs_diff": vs_full,
            "encode_seconds": round(t_enc, 1),
        }
        raw[arm] = Xn
        del model
        torch.cuda.empty_cache()
        print(f"[l1] {arm} done in {t_enc:.1f}s", flush=True)

    # BioLORD's shipped default would have truncated: quantify what was avoided.
    from transformers import AutoTokenizer
    bl_tok = AutoTokenizer.from_pretrained(enc.BIOLORD_REPO)
    bl_lens = np.array([len(x) for x in
                        bl_tok(texts, add_special_tokens=True)["input_ids"]])
    report["arms"]["biolord"]["would_truncate_at_shipped_128"] = int(
        (bl_lens > 128).sum())

    # ---- save ------------------------------------------------------------
    meta_common = {"corpus": str(IN_PARQUET), "n_rows": n,
                   "row_order": "rendered_visits.parquet order (patient asc, t asc)"}
    X_q4096 = raw["qwen3"]
    X_q768 = enc.mrl_truncate(X_q4096, enc.MRL_DIM)
    enc.save_states(OUT["qwen3_4096"], X_q4096, pid, tt,
                    {**meta_common, "arm": "qwen3", "dim": 4096,
                     "model": enc.QWEN3_REPO, "instruction_prefix": None})
    enc.save_states(OUT["qwen3"], X_q768, pid, tt,
                    {**meta_common, "arm": "qwen3", "dim": enc.MRL_DIM,
                     "model": enc.QWEN3_REPO, "instruction_prefix": None,
                     "reduction": "MRL truncation to first 768 dims, re-L2-normalised",
                     "primary": True})
    enc.save_states(OUT["biolord"], raw["biolord"], pid, tt,
                    {**meta_common, "arm": "biolord", "dim": 768,
                     "model": enc.BIOLORD_REPO,
                     "max_seq_length": enc.BIOLORD_MAX_SEQ})
    report["outputs"] = {k: str(v) for k, v in OUT.items()}
    report["mrl_truncation"] = {
        "dim": enc.MRL_DIM,
        "mean_cos_4096_vs_768_neighbour_top1_agreement": None,  # filled below
    }

    # ---- neighbour smell test -------------------------------------------
    spaces = {"qwen3_768": X_q768, "biolord": raw["biolord"],
              "qwen3_4096": X_q4096}
    rng = np.random.default_rng(SMELL_SEED)
    patients = np.sort(rng.choice(np.unique(pid), size=N_SMELL_PATIENTS,
                                  replace=False))
    smell: List[Dict[str, Any]] = []
    for p in patients:
        q_row = int(np.where((pid == p) & (tt == 0))[0][0])
        entry: Dict[str, Any] = {
            "patient_idx": int(p), "t": 0, "row": q_row,
            "query_text": texts[q_row], "arms": {}}
        q_med = _codes(texts[q_row], "Medications")
        q_ae = _codes(texts[q_row], "Adverse events")
        for space, X in spaces.items():
            sims = X @ X[q_row]
            sims[pid == p] = -np.inf          # other patients only
            top = np.argsort(-sims, kind="stable")[:TOPK]
            entry["arms"][space] = [{
                "row": int(r), "patient_idx": int(pid[r]), "t": int(tt[r]),
                "cosine": float(sims[r]), "text": texts[r],
                "med_jaccard": _jaccard(q_med, _codes(texts[r], "Medications")),
                "ae_jaccard": _jaccard(q_ae, _codes(texts[r], "Adverse events")),
            } for r in top]
        smell.append(entry)
    report["neighbour_smell_test"] = {
        "seed": SMELL_SEED, "n_patients": N_SMELL_PATIENTS,
        "query": "each patient's t=0 visit", "top_k": TOPK,
        "patients": [int(p) for p in patients],
        "results": smell,
    }

    # aggregate donor-profile overlap over the whole cohort, so the eyeball
    # verdict has a number behind it: mean top-3 other-patient Jaccard.
    agg: Dict[str, Any] = {}
    sub = np.arange(0, n, 10)                 # every 10th visit: 910 queries
    for space, X in spaces.items():
        S = X[sub] @ X.T
        med_j: List[float] = []
        ae_j: List[float] = []
        cos: List[float] = []
        for k, r0 in enumerate(sub):
            s = S[k].copy()
            s[pid == pid[r0]] = -np.inf
            top = np.argsort(-s, kind="stable")[:TOPK]
            qm, qa = _codes(texts[r0], "Medications"), _codes(texts[r0], "Adverse events")
            for r in top:
                cos.append(float(s[r]))
                jm = _jaccard(qm, _codes(texts[r], "Medications"))
                ja = _jaccard(qa, _codes(texts[r], "Adverse events"))
                if jm == jm:
                    med_j.append(jm)
                if ja == ja:
                    ae_j.append(ja)
        agg[space] = {"n_queries": int(len(sub)),
                      "mean_top3_cosine": float(np.mean(cos)),
                      "mean_top3_med_jaccard": float(np.mean(med_j)),
                      "mean_top3_ae_jaccard": float(np.mean(ae_j))}
    # random-pair baseline for the same Jaccards
    rng2 = np.random.default_rng(SMELL_SEED)
    pairs = rng2.integers(0, n, size=(3000, 2))
    rj_m, rj_a = [], []
    for i, j in pairs:
        if pid[i] == pid[j]:
            continue
        x = _jaccard(_codes(texts[i], "Medications"), _codes(texts[j], "Medications"))
        y = _jaccard(_codes(texts[i], "Adverse events"), _codes(texts[j], "Adverse events"))
        if x == x:
            rj_m.append(x)
        if y == y:
            rj_a.append(y)
    agg["random_pairs_baseline"] = {"n": len(rj_m),
                                    "mean_med_jaccard": float(np.mean(rj_m)),
                                    "mean_ae_jaccard": float(np.mean(rj_a))}
    report["neighbour_profile_overlap"] = agg

    # top-1 neighbour agreement between the 4096-d and the 768-d MRL space
    t1_4096, t1_768 = [], []
    S4 = X_q4096[sub] @ X_q4096.T
    S7 = X_q768[sub] @ X_q768.T
    for k, r0 in enumerate(sub):
        for S, acc in ((S4, t1_4096), (S7, t1_768)):
            s = S[k].copy()
            s[pid == pid[r0]] = -np.inf
            acc.append(int(np.argmax(s)))
    report["mrl_truncation"]["mean_cos_4096_vs_768_neighbour_top1_agreement"] = \
        float(np.mean(np.array(t1_4096) == np.array(t1_768)))
    report["mrl_truncation"]["mean_row_cosine_4096_vs_768"] = None

    report["wall_clock_seconds"] = round(time.time() - t_start, 1)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("neighbour_smell_test",)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
