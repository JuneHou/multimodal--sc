from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class SyntheticHistoricalControlSatelliteForecastSmokeTests(unittest.TestCase):
    def test_synthetic_historical_control_satellite_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            answer_path = tmp / "answer.json"
            details_path = tmp / "details.json"
            validation_path = tmp / "validation.json"
            subprocess.run(
                [
                    "python3",
                    "skills/earth-climate-and-geospatial-science/synthetic-historical-control-satellite-forecast/scripts/run_shc_satellite_forecast.py",
                    "--input",
                    "skills/earth-climate-and-geospatial-science/synthetic-historical-control-satellite-forecast/examples/toy_satellite_history.csv",
                    "--out",
                    str(answer_path),
                    "--details-out",
                    str(details_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            subprocess.run(
                [
                    "python3",
                    "skills/earth-climate-and-geospatial-science/synthetic-historical-control-satellite-forecast/scripts/validate_answer.py",
                    "--input",
                    "skills/earth-climate-and-geospatial-science/synthetic-historical-control-satellite-forecast/examples/toy_satellite_history.csv",
                    "--answer",
                    str(answer_path),
                    "--out",
                    str(validation_path),
                ],
                cwd=ROOT,
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
            self.assertTrue(all(row["donor_count"] >= 1 for row in details["per_band"]))


if __name__ == "__main__":
    unittest.main()
