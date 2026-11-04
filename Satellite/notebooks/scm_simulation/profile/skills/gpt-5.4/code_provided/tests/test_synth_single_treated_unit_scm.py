from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = (
    ROOT
    / "skills"
    / "statistical-and-machine-learning-foundations-for-science"
    / "synth-single-treated-unit-scm"
)
SCRIPT = SKILL_DIR / "scripts" / "run_synth_single_treated_unit_scm.R"
PANEL = SKILL_DIR / "examples" / "toy_tobacco_panel.csv"


class SynthSingleTreatedUnitScmTests(unittest.TestCase):
    def test_describe_panel_reports_expected_shape(self) -> None:
        completed = subprocess.run(
            [
                "Rscript",
                str(SCRIPT),
                "--panel",
                str(PANEL),
                "--treated-periods",
                "2006,2007,2008",
                "--describe-panel",
                "--out",
                "-",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["unit_count"], 5)
        self.assertEqual(payload["donor_count"], 4)
        self.assertEqual(payload["period_count"], 8)
        self.assertTrue(payload["balanced_panel"])
        self.assertEqual(payload["inferred_pre_periods"], ["2001", "2002", "2003", "2004", "2005"])

    def test_scm_run_matches_example_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            answer_path = Path(tmpdir) / "answer.json"
            diagnostics_path = Path(tmpdir) / "diagnostics.json"
            subprocess.run(
                [
                    "Rscript",
                    str(SCRIPT),
                    "--panel",
                    str(PANEL),
                    "--treated-periods",
                    "2006,2007,2008",
                    "--out",
                    str(answer_path),
                    "--diagnostics-out",
                    str(diagnostics_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            answer = json.loads(answer_path.read_text(encoding="utf-8"))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(answer["counterfactual"]["2006"], 15.0189, places=4)
            self.assertAlmostEqual(answer["counterfactual"]["2007"], 15.2242, places=4)
            self.assertAlmostEqual(answer["counterfactual"]["2008"], 15.4294, places=4)
            self.assertAlmostEqual(answer["att"], -2.1909, places=4)
            self.assertIn("Pre-treatment RMSPE", answer["rationale"])
            self.assertLess(diagnostics["pre_rmspe"], 0.02)
            self.assertEqual(diagnostics["top_donors"][0]["unit_id"], "donor_d")
            self.assertEqual(diagnostics["target_periods"], ["2006", "2007", "2008"])


if __name__ == "__main__":
    unittest.main()
