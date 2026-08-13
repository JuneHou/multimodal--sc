"""
L0 -- render one NCT00174655 visit as text, for the latent-recovery preliminary.

The pre-registration (`Docs/latent_prereg_2026-08-11.md`) fixes the string
exactly::

    "Medications: <name>; <name>. Adverse events: <name>; <name>."

with four properties that are NOT free choices and must not drift:

* **names come from `voc.pkl`'s `idx2word`**, per event type;
* **`[PAD]` (index 0) is excluded** -- it encodes "this event type is empty at
  this visit", not a code;
* **`treatment` and `ae_serious` are excluded** -- `treatment` is the trial's
  frozen intervention (matching on it would match on the thing the design holds
  fixed) and `ae_serious` is a severity re-coding of the same adverse events.
  This is the same state definition `observable/src/trial_nct00174655.py` uses
  for the bag, so the text and the bag describe the identical event set;
* **names are alphabetised within each section** -- the pre-registered
  order-sensitivity control.  The raw per-visit code lists are in the order the
  source frames happened to be grouped, and a last-token-pooled encoder is
  order-sensitive, so an unsorted rendering would let ordering noise enter the
  latent.  Sorting is plain lexicographic `sorted()`.

An empty section renders as `"Medications: none."` / `"Adverse events: none."`
rather than being dropped, so every visit produces both sections and the
sentence structure is constant across the corpus.

WHAT THE "NAMES" ACTUALLY ARE -- read before interpreting any latent
--------------------------------------------------------------------
They are **opaque numeric codes**, not English terms.  `voc['medication']`
holds WHO drug codes (`C_CTXWHO`, e.g. `00955301001`) and
`voc['adverse_event']` holds COSTART codes (`C_AECOS`, e.g. `00038`), both
carried through the release's own `process_NCT00174655.ipynb` as strings.  The
PyTrial release ships no code->term dictionary.  So the rendered text a
sentence encoder sees is a list of digit strings, and any "semantic" structure
in the resulting latent is lexical structure over digits, not clinical meaning.
This is a property of the dataset, recorded here rather than worked around: the
pre-registration says "names from voc idx2word", and that is what this module
does.

Public API
----------
    render_visit(rec, names) -> str
    render_corpus(visit, voc) -> pandas.DataFrame  [patient_idx, t, text]
    section_names(voc, key)  -> {code index: name}
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

#: Index 0 of every vocabulary in this release.  Means "empty", never a code.
PAD_INDEX = 0

#: (vocabulary key, field position in `visit[i][j]`, sentence header).  Exactly
#: the two state channels; `treatment` (position 0) and `ae_serious` (position
#: 3) are excluded on purpose -- see the module docstring.
SECTIONS: Tuple[Tuple[str, int, str], ...] = (
    ("medication", 1, "Medications"),
    ("adverse_event", 2, "Adverse events"),
)

#: What an empty section renders as.
EMPTY = "none"


def section_names(voc: Any, key: str) -> Dict[int, str]:
    """`{code index: name}` for one vocabulary, `[PAD]` kept out of the map."""
    idx2word = voc[key].idx2word
    names = {int(i): str(idx2word[i]) for i in idx2word}
    assert names[PAD_INDEX] == "[PAD]", f"{key}[0] is {names[PAD_INDEX]!r}"
    del names[PAD_INDEX]
    assert len(set(names.values())) == len(names), f"duplicate names in {key!r}"
    return names


def render_visit(rec: Sequence[Any], names: Dict[str, Dict[int, str]]) -> str:
    """One visit record -> the pre-registered sentence pair.

    Args:
        rec: `visit[i][j]`, the 6-field list
            `[treatment, medication, adverse_event, ae_serious, stage, ts]`.
        names: `{vocab key: {index: name}}` from `section_names`.

    Returns:
        `"Medications: a; b. Adverse events: c. "` -- one space between the two
        sentences, no trailing space.
    """
    parts: List[str] = []
    for key, pos, header in SECTIONS:
        codes = [int(c) for c in rec[pos] if int(c) != PAD_INDEX]
        # Alphabetised within the section: the pre-registered order control.
        # `sorted(set(...))` because a code repeated inside one visit record is
        # still one event.
        terms = sorted({names[key][c] for c in codes})
        body = "; ".join(terms) if terms else EMPTY
        parts.append(f"{header}: {body}.")
    return " ".join(parts)


def render_corpus(visit: Sequence[Sequence[Sequence[Any]]],
                  voc: Any) -> pd.DataFrame:
    """Render every visit of every patient, in file order.

    `patient_idx` is the patient's index into `visit.pkl` -- the same identity
    `observable/src/trial_nct00174655.py` writes as its `subject_id` string, so
    the text rows and the bag rows are joinable.  `t` is the 0-based visit
    index.  Row order is (patient asc, t asc) and is the order every downstream
    embedding matrix is aligned to.
    """
    names = {key: section_names(voc, key) for key, _p, _h in SECTIONS}
    pid: List[int] = []
    ts: List[int] = []
    txt: List[str] = []
    for i, patient in enumerate(visit):
        for j, rec in enumerate(patient):
            pid.append(i)
            ts.append(j)
            txt.append(render_visit(rec, names))
    df = pd.DataFrame({"patient_idx": pd.array(pid, dtype="int32"),
                       "t": pd.array(ts, dtype="int32"),
                       "text": txt})
    assert not df.duplicated(["patient_idx", "t"]).any()
    return df
