"""
L0 -- parse the n2c2 2018 Track 1 / 2014 i2b2 longitudinal corpus into records.

The dataset swap (Docs/latent_prereg_addendum_2026-08-11.md) replaces the
NCT00174655 rendering step with parsing: the artifact IS the record text, no
rendering.  Loader rules are fixed by the verification pass
(`latent/results/n2c2_verify.json`) and are NOT re-derived here:

* the usable XMLs are ONLY `train-*/train/` (202) and
  `n2c2-t1_gold_standard_test_data-20260811T175332Z-1-001/**/test/` (86).  The
  `...175327Z...` copy is an AppleDouble stub with an empty `test/` directory;
  the abstracts, the MIMICause community annotations and the eval script are
  not corpus.
* records split on `^Record date:` **at line start** -- verified 1,264/1,264
  occurrences are at line start and 1,264/1,264 records parse that way.  The
  asterisk banner lines are NOT a reliable separator (6 patients have more
  banner lines than records).
* records are **sorted by parsed date**; train 112 and 344 are non-chronological
  in file order.  `t` is the 0-based index AFTER sorting.
* dates are surrogate (years 2059-2173, date-shifted de-identification) and are
  used only for ordering.

The banner lines (a line of nothing but `*`) are dropped from record text: they
are a de-identification/formatting separator, not clinical prose.  Nothing else
about the text is altered -- natural prose is embedded as-is (the addendum
retires the concept-order canonicalisation, which applied to rendered bags).

Public API
----------
    corpus_files(root)                 -> [(patient_id, split, path), ...]
    parse_patient(path)                -> dict(text, records, criteria, ...)
    build_corpus(root)                 -> pandas.DataFrame (one row per record)
    CRITERIA                              the 13 tag names
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

TRAIN_GLOB = "train-20260811T180607Z-1-001/train/*.xml"
TEST_GLOB = ("n2c2-t1_gold_standard_test_data-20260811T175332Z-1-001/"
             "n2c2-t1_gold_standard_test_data/test/*.xml")

EXPECTED = {"train": 202, "test": 86}

#: exact record separator, verified 1,264/1,264
RECORD_SPLIT = re.compile(r"^(?=Record date:)", re.M)
RECORD_DATE = re.compile(r"^Record date:\s*(\d{4})-(\d{2})-(\d{2})\s*$", re.M)
BANNER = re.compile(r"^\*+\s*$", re.M)

CRITERIA = ("ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE", "ASP-FOR-MI",
            "CREATININE", "DIETSUPP-2MOS", "DRUG-ABUSE", "ENGLISH", "HBA1C",
            "KETO-1YR", "MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS")
TAG_VALUES = ("met", "not met")


def corpus_files(root: Path) -> List[Tuple[str, str, Path]]:
    """`(patient_id, split, path)` for the 288 usable patient XMLs."""
    root = Path(root)
    out: List[Tuple[str, str, Path]] = []
    for split, pat in (("train", TRAIN_GLOB), ("test", TEST_GLOB)):
        paths = sorted(root.glob(pat))
        # AppleDouble resource forks (`._100.xml`) are byte garbage, never data.
        paths = [p for p in paths if not p.name.startswith("._")]
        if len(paths) != EXPECTED[split]:
            raise AssertionError(
                f"{split}: found {len(paths)} xml, expected {EXPECTED[split]}")
        out.extend((p.stem, split, p) for p in paths)
    return out


def _strip_banner(s: str) -> str:
    return BANNER.sub("", s).strip()


def parse_patient(path: Path) -> Dict[str, Any]:
    """One patient XML -> its records (date-sorted) and its 13 criteria tags."""
    root = ET.parse(str(path)).getroot()
    text = root.find("TEXT").text
    assert text is not None and text.strip(), f"{path}: empty TEXT"

    parts = RECORD_SPLIT.split(text)
    preamble = parts[0]
    assert not preamble.strip(), (
        f"{path}: non-empty text before the first record: {preamble[:80]!r}")
    chunks = parts[1:]
    assert chunks, f"{path}: no `Record date:` marker"

    recs: List[Dict[str, Any]] = []
    for order, ch in enumerate(chunks):
        m = RECORD_DATE.match(ch)
        assert m is not None, f"{path}: unparsable record header {ch[:60]!r}"
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        body = _strip_banner(ch)
        assert body, f"{path}: empty record body at file order {order}"
        recs.append({"file_order": order, "record_date": d, "text": body})

    # stable sort -> ties keep file order
    recs.sort(key=lambda r: r["record_date"])
    dates_ascending_in_file = all(
        recs[i]["file_order"] < recs[i + 1]["file_order"]
        for i in range(len(recs) - 1))

    tags = root.find("TAGS")
    crit = {}
    for name in CRITERIA:
        el = tags.find(name)
        assert el is not None, f"{path}: missing criterion {name}"
        v = el.attrib["met"]
        assert v in TAG_VALUES, f"{path}: {name} has unexpected value {v!r}"
        crit[name] = v

    return {"path": path, "n_records": len(recs), "records": recs,
            "criteria": crit,
            "dates_ascending_in_file_order": dates_ascending_in_file,
            "n_banner_lines": len(BANNER.findall(text))}


def build_corpus(root: Path):
    """One row per record: patient_id, split, t, record_date, text, n_records."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    per_patient: List[Dict[str, Any]] = []
    for pid, split, path in corpus_files(root):
        p = parse_patient(path)
        for t, r in enumerate(p["records"]):
            rows.append({
                "patient_id": pid,
                "patient_idx": int(pid),
                "split": split,
                "t": t,
                "file_order": r["file_order"],
                "record_date": r["record_date"].isoformat(),
                "text": r["text"],
                "n_records": p["n_records"],
                "n_chars": len(r["text"]),
                "n_ws_tokens": len(r["text"].split()),
            })
        per_patient.append({
            "patient_id": pid, "split": split, "file": str(path),
            "n_records": p["n_records"],
            "dates": [r["record_date"].isoformat() for r in p["records"]],
            "dates_ascending_in_file_order": p["dates_ascending_in_file_order"],
            "n_banner_lines": p["n_banner_lines"],
            "criteria": p["criteria"],
        })
    df = pd.DataFrame(rows)
    for name in CRITERIA:
        m = {p["patient_id"]: p["criteria"][name] for p in per_patient}
        df[f"crit_{name}"] = df["patient_id"].map(m)
    df = df.sort_values(["patient_idx", "t"], kind="stable").reset_index(drop=True)
    return df, per_patient
