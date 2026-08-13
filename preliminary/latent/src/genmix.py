"""Shared L3/L4 pieces: mixture construction, decode prompts, judge prompt.

The decoder is **retrieval-conditioned decoding**, the pre-registration's
documented preliminary stand-in for a trained decoder: the donor record texts
and their convex weights are put in an instruction and a chat model writes the
mixed patient's record.  Nothing is trained.

Kept in one place so the generation stage (vLLM) and the analysis stage
(embedding) build the identical mixtures from the identical seeds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

#: per-donor prompt budget, in Qwen2.5 tokens.  The corpus reaches 6,225 tokens
#: per record, so five uncapped donors can exceed the 32k context; 2,048
#: truncates 15.7% of records and keeps the worst-case prompt near 10k tokens.
DONOR_MAX_TOKENS = 2048

#: generation cap (pre-registration: "~800 tokens").
GEN_MAX_TOKENS = 800

TRUNC_MARK = "\n[... record truncated here for prompt length ...]"

DECODE_INSTRUCTION = (
    "You are given {k} real de-identified clinical records, each from a "
    "different patient and each with a mixing weight. The weights sum to 1 and "
    "state how much that record should influence the result.\n\n"
    "{blocks}\n"
    "Write ONE clinical record for a single synthetic patient that is the "
    "weighted blend of the records above. The content -- presenting problem, "
    "history, medications, findings, assessment and plan -- must reflect each "
    "source record in proportion to its weight: the highest-weighted record "
    "should dominate, and lower-weighted records should contribute "
    "proportionally less. Match the register, section structure and clinical "
    "style of the source records. Output only the record text itself, with no "
    "commentary, no headings about weights, and no mention of this "
    "instruction."
)

JUDGE_PROMPT = (
    "You are a cardiologist rating a clinical record. Read the record below "
    "and rate, on a 1-10 scale, the severity and extent of coronary artery "
    "disease evidenced in this record.\n\n"
    "Scale:\n"
    "1  = no evidence of coronary artery disease anywhere in the record\n"
    "2-3 = risk factors only (e.g. hypertension, hyperlipidaemia, diabetes), "
    "no coronary disease documented\n"
    "4-5 = suspected or mild coronary artery disease, or an abnormal stress "
    "test without documented intervention\n"
    "6-7 = documented coronary artery disease: prior myocardial infarction, "
    "angina, or a single revascularisation\n"
    "8-9 = advanced coronary artery disease: multivessel disease, repeat "
    "revascularisation, or coronary artery bypass grafting\n"
    "10 = extensive end-stage coronary disease with ischaemic "
    "cardiomyopathy, heart failure, or refractory angina\n\n"
    "RECORD:\n{record}\n\n"
    "Answer with a single integer from 1 to 10 and nothing else."
)


def truncate_tokens(text: str, tok, max_tokens: int = DONOR_MAX_TOKENS
                    ) -> Tuple[str, bool]:
    """First `max_tokens` tokens of `text`.  Head, because an n2c2 record puts
    the narrative, problem list and medications before the closing boilerplate.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False
    return tok.decode(ids[:max_tokens]) + TRUNC_MARK, True


def decode_prompt(donor_texts: Sequence[str], weights: Sequence[float], tok,
                  max_tokens: int = DONOR_MAX_TOKENS) -> Tuple[str, int]:
    """The retrieval-conditioned decode prompt.  Returns (prompt, n_truncated).

    Donors appear in the order given -- L3 uses the draw order (so donor
    position is independent of weight), L4 uses a fixed [not-met, met] order
    (so alpha is the only thing that varies within a pair).
    """
    blocks: List[str] = []
    n_trunc = 0
    for i, (txt, w) in enumerate(zip(donor_texts, weights), start=1):
        t, was = truncate_tokens(txt, tok, max_tokens)
        n_trunc += int(was)
        blocks.append(f"--- RECORD {i} (weight {float(w):.3f}) ---\n{t}\n")
    return (DECODE_INSTRUCTION.format(k=len(donor_texts),
                                      blocks="\n".join(blocks)), n_trunc)


def parse_judge(text: str) -> float:
    """First integer 1-10 in the judge's reply, else nan (never a fabricated
    default -- the vendored module's rule: if a parse fails, record it)."""
    import re

    m = re.search(r"\b(10|[1-9])\b", text or "")
    return float(m.group(1)) if m else float("nan")


def load_corpus(latent_dir: Path) -> Dict[str, Any]:
    """Corpus rows + the qwen3-768 latents, asserted row-aligned."""
    import pandas as pd
    import sys

    sys.path.insert(0, str(latent_dir / "src"))
    import encode as enc

    df = pd.read_parquet(latent_dir / "data" / "n2c2_corpus.parquet")
    s = enc.load_states(latent_dir / "data" / "states_n2c2_qwen3.npz")
    Z = np.ascontiguousarray(s["X"], dtype=np.float64)
    assert len(df) == Z.shape[0] == 1264, (len(df), Z.shape)
    assert np.array_equal(df["patient_idx"].to_numpy(np.int32), s["patient_idx"])
    assert np.array_equal(df["t"].to_numpy(np.int32), s["t"])
    return {"df": df, "Z": Z, "pid": s["patient_idx"], "t": s["t"]}


# --------------------------------------------------------------------- L3


def l3_mixtures(n_corpus: int, Z: np.ndarray, n_per_family: int = 250,
                K: int = 5, seed: int = 0) -> List[Dict[str, Any]]:
    """The decision-cell mixtures, from L2's own `draw_replicates`.

    Keying is L2's: the stream depends on (seed, K, family) only, so these are
    literally the first `n_per_family` pseudo-targets of the L2 K=5 cells.
    """
    import sys

    import run_weight_recovery as wr  # noqa: F401  (path set by the caller)

    out: List[Dict[str, Any]] = []
    for family in ("dirichlet", "spiky"):
        D, W = wr.draw_replicates(n_corpus, K, family, n_per_family, seed=seed)
        for r in range(n_per_family):
            donors = D[r].astype(int)
            w = W[r].astype(float)
            out.append({
                "mixture_id": f"{family}_{r:04d}",
                "family": family, "rep": int(r), "K": int(K),
                "donor_rows": donors, "weights": w,
                "z_star": w @ Z[donors],
                "top_donor_row": int(donors[int(np.argmax(w))]),
                "top_weight": float(w.max()),
            })
    return out


# --------------------------------------------------------------------- L4

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def l4_pairs(df, Z: np.ndarray, n_pairs: int = 50, seed: int = 0
             ) -> List[Dict[str, Any]]:
    """`n_pairs` (ADVANCED-CAD met, nearest not-met neighbour) pairs on t=0.

    One t=0 record per patient; the neighbour is the nearest not-met patient by
    cosine in the qwen3-768 space.  `alpha` is the weight on the MET (severe)
    record throughout, so severity is expected to increase with alpha.
    """
    t0 = df.index[df["t"] == 0].to_numpy()
    lab = df.loc[t0, "crit_ADVANCED-CAD"].to_numpy()
    met_rows = t0[lab == "met"]
    not_rows = t0[lab == "not met"]
    assert len(met_rows) + len(not_rows) == len(t0)

    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(met_rows, size=n_pairs, replace=False))

    Zn = Z[not_rows]
    pairs: List[Dict[str, Any]] = []
    for r in pick:
        sims = Zn @ Z[r]
        j = int(np.argmax(sims))
        pairs.append({
            "pair_id": f"pair_{int(r):04d}",
            "met_row": int(r), "notmet_row": int(not_rows[j]),
            "met_patient": str(df.loc[r, "patient_id"]),
            "notmet_patient": str(df.loc[not_rows[j], "patient_id"]),
            "neighbour_cosine": float(sims[j]),
        })
    return pairs
