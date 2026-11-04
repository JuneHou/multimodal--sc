from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_DIR / "scripts"
MAKE_PANEL = SCRIPTS / "make_example_panel.py"
INSPECT = SCRIPTS / "inspect_panel.py"
ESTIMATE = SCRIPTS / "estimate_synthetic_control.R"
VALIDATE = SCRIPTS / "validate_answer.py"

RScript = shutil.which("Rscript")


def r_available() -> bool:
    if RScript is None:
        return False
    probe = subprocess.run(
        [RScript, "-e", 'cat(requireNamespace("Synth", quietly=TRUE) && requireNamespace("jsonlite", quietly=TRUE))'],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip().endswith("TRUE")


@unittest.skipUnless(r_available(), "Rscript with Synth and jsonlite is required")
class SyntheticControlAttTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.workdir = Path(cls.tmp.name)
        cls.panel = cls.workdir / "tobacco_panel.csv"
        cls.periods = ",".join(str(p) for p in range(1989, 2001))
        subprocess.run(
            ["python3", str(MAKE_PANEL), "--out", str(cls.panel), "--first-treated-period", "1989"],
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_inspector_reports_well_formed_panel(self) -> None:
        completed = subprocess.run(
            ["python3", str(INSPECT), "--panel", str(self.panel)],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["blocking_issues"], [])
        self.assertEqual(report["donor_count"], 15)
        self.assertEqual(report["treated_unit"], "treated")

    def test_inspector_rejects_missing_columns(self) -> None:
        bad = self.workdir / "bad_panel.csv"
        bad.write_text("unit_id,period,foo\n1,1,2\n", encoding="utf-8")
        completed = subprocess.run(
            ["python3", str(INSPECT), "--panel", str(bad)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing", completed.stdout.lower())

    def test_estimator_recovers_known_effect_and_writes_contract(self) -> None:
        answer_path = self.workdir / "answer.json"
        summary_path = self.workdir / "summary.json"
        completed = subprocess.run(
            [
                RScript,
                str(ESTIMATE),
                "--panel",
                str(self.panel),
                "--treated-periods",
                self.periods,
                "--out",
                str(answer_path),
                "--summary-out",
                str(summary_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(stdout["status"], "ok")
        self.assertEqual(stdout["method"], "synth")

        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        self.assertEqual(set(answer["counterfactual"]), set(self.periods.split(",")))
        self.assertAlmostEqual(answer["att"], -8.0, delta=1.0)
        self.assertIsInstance(answer["rationale"], str)
        self.assertTrue(answer["rationale"])

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertLess(summary["pre_treatment_mspe"], 5.0)
        self.assertEqual(len(summary["solution_w"]), 15)

    def test_validator_accepts_and_rejects(self) -> None:
        answer_path = self.workdir / "answer2.json"
        subprocess.run(
            [
                RScript,
                str(ESTIMATE),
                "--panel",
                str(self.panel),
                "--treated-periods",
                self.periods,
                "--out",
                str(answer_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        ok = subprocess.run(
            [
                "python3",
                str(VALIDATE),
                "--answer",
                str(answer_path),
                "--panel",
                str(self.panel),
                "--treated-periods",
                self.periods,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        verdict = json.loads(ok.stdout)
        self.assertEqual(verdict["status"], "ok")
        self.assertEqual(verdict["problems"], [])

        # Corrupt the ATT and confirm the validator fails.
        corrupted = json.loads(answer_path.read_text(encoding="utf-8"))
        corrupted["att"] = 999.0
        bad_path = self.workdir / "bad_answer.json"
        bad_path.write_text(json.dumps(corrupted), encoding="utf-8")
        bad = subprocess.run(
            [
                "python3",
                str(VALIDATE),
                "--answer",
                str(bad_path),
                "--panel",
                str(self.panel),
                "--treated-periods",
                self.periods,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("att", bad.stdout.lower())


class SyntheticControlPanelGeneratorTests(unittest.TestCase):
    def test_generator_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a_panel.csv"
            b = Path(tmp) / "b_panel.csv"
            for out in (a, b):
                subprocess.run(
                    ["python3", str(MAKE_PANEL), "--out", str(out), "--seed", "7"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8"))
            self.assertIn("Z1", a.read_text(encoding="utf-8").splitlines()[0])

    def test_generator_rejects_too_few_donors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "c_panel.csv"
            completed = subprocess.run(
                ["python3", str(MAKE_PANEL), "--out", str(out), "--donors", "1"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("donors", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
