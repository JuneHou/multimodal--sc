"""Repository-level smoke test for the synthetic-historical-control-band-forecast skill.

References the skill slug so the skill-suite audit maps this test back to
`skills/statistical-and-machine-learning-foundations-for-science/synthetic-historical-control-band-forecast`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = "skills/statistical-and-machine-learning-foundations-for-science/synthetic-historical-control-band-forecast"
SCRIPT = ROOT / SKILL_PATH / "scripts" / "run_shc_band_forecast.py"
VALIDATOR = ROOT / SKILL_PATH / "scripts" / "validate_answer.py"
EXAMPLE = ROOT / SKILL_PATH / "examples" / "toy_parcel_data.csv"


class ShcBandForecastSmokeTests(unittest.TestCase):
    def test_skill_smoke_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "answer.json"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(EXAMPLE), "--out", str(out)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertIn('"status": "ok"', completed.stdout)
            answer = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(set(answer["prediction"]), {"ndvi", "nir", "red", "swir"})

            validated = subprocess.run(
                [sys.executable, str(VALIDATOR), "--answer", str(out)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertIn('"status": "valid"', validated.stdout)


if __name__ == "__main__":
    unittest.main()
