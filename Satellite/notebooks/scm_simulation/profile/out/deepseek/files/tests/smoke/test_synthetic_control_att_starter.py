from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "statistical-and-machine-learning-foundations-for-science" / "synthetic-control-att-starter"
ESTIMATE = SKILL / "scripts" / "estimate_synthetic_control.R"
INSPECT = SKILL / "scripts" / "inspect_panel.py"
VALIDATE = SKILL / "scripts" / "validate_answer.py"
PANEL = SKILL / "examples" / "tobacco_panel.csv"
TREATED_PERIODS = ",".join(str(p) for p in range(1989, 2001))

RScript = shutil.which("Rscript")


def r_ready() -> bool:
    if RScript is None:
        return False
    probe = subprocess.run(
        [RScript, "-e", 'cat(requireNamespace("Synth", quietly=TRUE) && requireNamespace("jsonlite", quietly=TRUE))'],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip().endswith("TRUE")


@unittest.skipUnless(r_ready(), "Rscript with Synth and jsonlite is required")
class SyntheticControlSmokeTests(unittest.TestCase):
    def test_estimated_att_is_near_known_effect(self) -> None:
        self.assertTrue(PANEL.is_file(), f"missing bundled panel: {PANEL}")
        with tempfile.TemporaryDirectory() as tmp:
            answer = Path(tmp) / "answer.json"
            summary = Path(tmp) / "summary.json"
            subprocess.run(
                [
                    RScript,
                    str(ESTIMATE),
                    "--panel",
                    str(PANEL),
                    "--treated-periods",
                    TREATED_PERIODS,
                    "--out",
                    str(answer),
                    "--summary-out",
                    str(summary),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            payload = json.loads(answer.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["counterfactual"]), set(TREATED_PERIODS.split(",")))
            self.assertAlmostEqual(payload["att"], -8.0, delta=1.0)
            self.assertTrue(payload["rationale"].strip())
            self.assertLess(json.loads(summary.read_text(encoding="utf-8"))["pre_treatment_mspe"], 5.0)

    def test_inspector_and_validator_agree_with_fixture(self) -> None:
        completed = subprocess.run(
            ["python3", str(INSPECT), "--panel", str(PANEL)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["donor_count"], 15)

        with tempfile.TemporaryDirectory() as tmp:
            answer = Path(tmp) / "answer.json"
            subprocess.run(
                [
                    RScript,
                    str(ESTIMATE),
                    "--panel",
                    str(PANEL),
                    "--treated-periods",
                    TREATED_PERIODS,
                    "--out",
                    str(answer),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            verdict = subprocess.run(
                [
                    "python3",
                    str(VALIDATE),
                    "--answer",
                    str(answer),
                    "--panel",
                    str(PANEL),
                    "--treated-periods",
                    TREATED_PERIODS,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(json.loads(verdict.stdout)["problems"], [])


if __name__ == "__main__":
    unittest.main()
