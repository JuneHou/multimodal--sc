"""Skill-local tests for synthetic-historical-control-band-forecast.

These tests exercise the runnable SHC band-forecast script end to end on the
bundled toy parcel, plus its error paths and the answer-contract validator.
They use only the Python standard library so they run offline.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SKILL = ROOT / "skills" / "statistical-and-machine-learning-foundations-for-science" / "synthetic-historical-control-band-forecast"
SCRIPT = SKILL / "scripts" / "run_shc_band_forecast.py"
VALIDATOR = SKILL / "scripts" / "validate_answer.py"
EXAMPLE = SKILL / "examples" / "toy_parcel_data.csv"
TRUTH = SKILL / "examples" / "toy_parcel_truth.json"
BANDS = ["ndvi", "nir", "red", "swir"]


def run(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd) if cwd else str(ROOT),
        check=check,
        capture_output=True,
        text=True,
        timeout=300,
    )


class ShcBandForecastTests(unittest.TestCase):
    def test_forecast_writes_valid_answer_with_all_bands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "answer.json"
            summary = Path(tmp) / "summary.json"
            completed = run(
                [
                    str(SCRIPT),
                    "--input", str(EXAMPLE),
                    "--out", str(out),
                    "--summary-out", str(summary),
                ]
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["n_bands"], len(BANDS))

            answer = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(set(answer.keys()), {"prediction", "rationale"})
            self.assertEqual(sorted(answer["prediction"].keys()), sorted(BANDS))
            for band, value in answer["prediction"].items():
                self.assertIsInstance(value, (int, float), band)
            self.assertIsInstance(answer["rationale"], str)
            self.assertTrue(answer["rationale"].strip())

            diag = json.loads(summary.read_text(encoding="utf-8"))
            for band in BANDS:
                self.assertIn(band, diag["band_diagnostics"])
                self.assertLess(diag["band_diagnostics"][band]["matching_mse"], 0.1)

    def test_validator_reports_low_error_against_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "answer.json"
            run([str(SCRIPT), "--input", str(EXAMPLE), "--out", str(out)])
            completed = run(
                [
                    str(VALIDATOR),
                    "--answer", str(out),
                    "--expected-bands", ",".join(BANDS),
                    "--truth", str(TRUTH),
                ]
            )
            report = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(report["status"], "valid")
            self.assertLess(report["accuracy"]["mean_abs_error"], 0.05)

    def test_missing_input_fails_with_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "answer.json"
            completed = run(
                [str(SCRIPT), "--input", str(Path(tmp) / "nope.csv"), "--out", str(out)],
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("failed to read", completed.stderr)

    def test_empty_input_fails_with_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "data.csv"
            empty.write_text("band,time,value\n", encoding="utf-8")
            completed = run(
                [str(SCRIPT), "--input", str(empty), "--out", str(Path(tmp) / "a.json")],
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_validator_rejects_bad_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "answer.json"
            bad.write_text(json.dumps({"prediction": {"ndvi": "high"}, "rationale": ""}), encoding="utf-8")
            completed = run([str(VALIDATOR), "--answer", str(bad)], check=False)
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
