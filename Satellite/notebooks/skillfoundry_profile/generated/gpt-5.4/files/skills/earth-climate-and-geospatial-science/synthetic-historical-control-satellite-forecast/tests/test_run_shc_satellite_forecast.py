#!/usr/bin/env python3
"""Tests for the synthetic historical control satellite forecast skill."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SKILL_ROOT / "scripts" / "run_shc_satellite_forecast.py"
VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_answer.py"
EXAMPLE_INPUT = SKILL_ROOT / "examples" / "toy_satellite_history.csv"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module(SCRIPT_PATH, "shc_satellite_forecast")


class SyntheticHistoricalControlSatelliteForecastTests(unittest.TestCase):
    def test_project_to_simplex_respects_constraints(self) -> None:
        weights = MODULE.project_to_simplex([0.8, -0.2, 1.7])
        self.assertAlmostEqual(sum(weights), 1.0, places=8)
        self.assertTrue(all(weight >= 0.0 for weight in weights))

    def test_load_long_history_detects_ragged_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ragged.csv"
            path.write_text(
                "band,time,value\nA,1,1.0\nA,2,2.0\nB,1,3.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                MODULE.load_long_history(path)

    def test_end_to_end_forecast_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            answer_path = Path(tmpdir) / "answer.json"
            details_path = Path(tmpdir) / "details.json"
            validation_path = Path(tmpdir) / "validation.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(EXAMPLE_INPUT),
                    "--out",
                    str(answer_path),
                    "--details-out",
                    str(details_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_PATH),
                    "--input",
                    str(EXAMPLE_INPUT),
                    "--answer",
                    str(answer_path),
                    "--out",
                    str(validation_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            answer = json.loads(answer_path.read_text(encoding="utf-8"))
            details = json.loads(details_path.read_text(encoding="utf-8"))
            validation = json.loads(validation_path.read_text(encoding="utf-8"))

            self.assertTrue(validation["valid"])
            self.assertEqual(sorted(answer["prediction"]), ["NDVI", "NIR", "SWIR"])
            self.assertGreater(len(answer["rationale"]), 80)
            self.assertEqual(details["band_count"], 3)
            self.assertEqual(details["time_count"], 12)
            self.assertAlmostEqual(answer["prediction"]["NDVI"], 0.247981, places=3)
            self.assertAlmostEqual(answer["prediction"]["NIR"], 0.452827, places=3)
            self.assertAlmostEqual(answer["prediction"]["SWIR"], 0.208907, places=3)
            self.assertTrue(
                all(entry["method"] in {"synthetic_historical_control", "ridge_ols_fallback"} for entry in details["per_band"])
            )


if __name__ == "__main__":
    unittest.main()
