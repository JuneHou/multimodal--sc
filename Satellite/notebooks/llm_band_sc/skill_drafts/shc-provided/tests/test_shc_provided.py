"""Tests for the shc-provided skill. Run from the skill directory:
    python -m unittest discover -s tests -p 'test_*.py'

execution   scripts/fit.py and scripts/show_data.py run on the synthetic example and print what SKILL.md describes
planted     scripts/fit.py recovers the example's planted block weights (0.7 on block 2, 0.3 on block 4) to 1e-6
parity      scripts/fit.py at her settings reproduces lb_shc_parity.csv on three real cells (blocks identical;
            weights, matching MSE and predictions to 1e-8)
guard       data holding P10 or not exactly P01-P09, a data file not named data.csv, and a field naming P10 are refused

tests/fixtures holds the three cells' P01-P09 data as the harness writes it and their rows of lb_shc_parity.csv,
which carry her predictions for P10. That is why tests/ is not mounted in the sandbox.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
FIT = SKILL / "scripts" / "fit.py"
SHOW = SKILL / "scripts" / "show_data.py"
EXAMPLES = SKILL / "examples"
FIXTURES = SKILL / "tests" / "fixtures"
FIT_MARK = "@@FIT@@"
PLANTED = [0.0, 0.7, 0.0, 0.3, 0.0, 0.0]
CELLS = [("sentinel1", "treatment_0015"), ("sentinel2", "treatment_0001"), ("sentinel2", "treatment_0014")]


def run(script, *args, cwd):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(script), *map(str, args)], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=300)


def split_output(stdout):
    """-> (the JSON summary the model sees, the record the harness captures)."""
    shown = [l for l in stdout.splitlines() if not l.startswith(FIT_MARK)]
    marked = [l for l in stdout.splitlines() if l.startswith(FIT_MARK)]
    return json.loads("\n".join(shown)), json.loads(marked[-1][len(FIT_MARK):])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Execution(Base):
    def test_help_documents_every_argument(self):
        r = run(FIT, "--help", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        for flag in ("--data", "--spec", "--out", "--log"):
            self.assertIn(flag, r.stdout)
        r = run(SHOW, "--help", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        for flag in ("--data", "--out"):
            self.assertIn(flag, r.stdout)

    def test_example_run_prints_the_summary(self):
        out = self.tmp / "summary.json"
        r = run(FIT, "--data", EXAMPLES / "data.csv", "--spec", EXAMPLES / "spec.json", "--out", out,
                "--log", self.tmp / "log.jsonl", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        summary, record = split_output(r.stdout)
        for key in ("m", "n", "matching_window", "blocks_available", "blocks_kept", "matching_mse", "prediction"):
            self.assertIn(key, summary)
        self.assertEqual(sorted(summary["prediction"]["values"]), ["VH", "VV", "VV-VH"])
        self.assertEqual(summary["prediction"]["month"], "2021-10")
        self.assertEqual(json.loads(out.read_text()), summary)
        self.assertEqual(len((self.tmp / "log.jsonl").read_text().splitlines()), 1)
        self.assertEqual(record["donor_ids"], [b["block"] for b in summary["blocks_kept"]])

    def test_show_data_prints_nine_months_and_the_target(self):
        out = self.tmp / "data.json"
        r = run(SHOW, "--data", EXAMPLES / "data.csv", "--out", out, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(sum(1 for l in lines if l.startswith("P0")), 9)
        self.assertEqual(lines[-1], "The month to predict is P10 = 2021-10. Its values are withheld.")
        self.assertEqual(len(json.loads(out.read_text())["months"]), 9)


class Planted(Base):
    def test_planted_weights_recovered(self):
        r = run(FIT, "--data", EXAMPLES / "data.csv", "--spec", EXAMPLES / "spec.json", "--log", self.tmp / "log.jsonl", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        summary, record = split_output(r.stdout)
        expected = json.loads((EXAMPLES / "expected_output.json").read_text())
        full = [0.0] * summary["blocks_available"]
        for block, w in zip(record["donor_ids"], record["weights"]):
            full[block - 1] = w
        self.assertEqual(record["donor_ids"], expected["blocks_kept"])
        self.assertLessEqual(max(abs(a - b) for a, b in zip(full, PLANTED)), 1e-6)
        self.assertLessEqual(max(abs(record["predictions"][b] - v) for b, v in expected["prediction"]["values"].items()), 1e-6)


class Parity(Base):
    def test_her_settings_reproduce_parity_on_three_cells(self):
        with open(FIXTURES / "parity.csv", newline="") as f:
            ref = {(r["sensor"], r["site_id"]): r for r in csv.DictReader(f)}
        self.assertEqual(sorted(ref), sorted(CELLS))
        worst = 0.0
        for sensor, site in CELLS:
            r = run(FIT, "--data", FIXTURES / f"{sensor}_{site}" / "data.csv", "--spec", FIXTURES / "her_spec.json",
                    "--log", self.tmp / "log.jsonl", cwd=self.tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            _, record = split_output(r.stdout)
            row = ref[(sensor, site)]
            self.assertEqual(record["donor_ids"], [int(x) for x in row["selected_blocks"].split()], site)
            worst = max(worst,
                        max(abs(a - float(b)) for a, b in zip(record["weights"], row["weights_full"].split())),
                        abs(record["fit_mse"] - float(row["latent_mse"])),
                        max(abs(v - float(row[f"pred_{b}"])) for b, v in record["predictions"].items()))
        self.assertLessEqual(worst, 1e-8)


class Guard(Base):
    def _data(self, name, lines):
        p = self.tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_data_with_p10_is_refused(self):
        rows = (EXAMPLES / "data.csv").read_text().splitlines()
        p = self._data("p10/data.csv", rows + ["VV,P10,2021-10,-10.0"])
        for script, extra in ((FIT, ["--spec", EXAMPLES / "spec.json", "--log", self.tmp / "log.jsonl"]), (SHOW, [])):
            r = run(script, "--data", p, *extra, cwd=self.tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("P10", r.stderr)
            self.assertNotIn("-10.0", r.stdout)

    def test_data_without_nine_months_is_refused(self):
        rows = [l for l in (EXAMPLES / "data.csv").read_text().splitlines() if ",P09," not in l]
        p = self._data("eight/data.csv", rows)
        r = run(FIT, "--data", p, "--spec", EXAMPLES / "spec.json", "--log", self.tmp / "log.jsonl", cwd=self.tmp)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("must hold exactly the nine periods P01-P09", r.stderr)

    def test_data_file_not_named_data_csv_is_refused(self):
        p = self._data("other.csv", (EXAMPLES / "data.csv").read_text().splitlines())
        r = run(FIT, "--data", p, "--spec", EXAMPLES / "spec.json", "--log", self.tmp / "log.jsonl", cwd=self.tmp)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("must be named data.csv", r.stderr)

    def test_field_naming_p10_is_refused(self):
        spec = {**json.loads((EXAMPLES / "spec.json").read_text()), "target": "P10"}
        (self.tmp / "spec.json").write_text(json.dumps(spec))
        r = run(FIT, "--data", EXAMPLES / "data.csv", "--spec", self.tmp / "spec.json", "--log", self.tmp / "log.jsonl", cwd=self.tmp)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refers to P10, which is withheld", r.stderr)
        self.assertFalse((self.tmp / "log.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
