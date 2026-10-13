"""Tests for the shc-written skill. Run from the skill directory:
    python -m unittest discover -s tests -p 'test_*.py'

This skill provides no fitter; the model writes predict.py. The tests therefore check what the skill does provide.
execution   scripts/show_data.py runs on the synthetic example and on three real cells and prints nine months
planted     the example's planted block weights (0.7 on block 2, 0.3 on block 4) are recovered to 1e-6 from
            examples/data.csv by least squares with the weights summing to one over all six blocks, and the
            prediction they imply equals examples/expected_output.json and assets/example_prediction.txt to 1e-6
parity      her notebook code run as predict.py (tests/fixtures/her_predict.py, the program gate SK5 runs) reads
            data.csv and prints one line per band that reproduces lb_shc_parity.csv on three real cells to 1e-8
guard       data holding P10 or not exactly P01-P09 is refused by scripts/show_data.py

tests/fixtures holds the three cells' P01-P09 data as the harness writes it, their rows of lb_shc_parity.csv, which
carry her predictions for P10, and her code. That is why tests/ is not mounted in the sandbox.
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

import numpy as np

SKILL = Path(__file__).resolve().parents[1]
SHOW = SKILL / "scripts" / "show_data.py"
EXAMPLES = SKILL / "examples"
FIXTURES = SKILL / "tests" / "fixtures"
PLANTED = [0.0, 0.7, 0.0, 0.3, 0.0, 0.0]
CELLS = [("sentinel1", "treatment_0015"), ("sentinel2", "treatment_0001"), ("sentinel2", "treatment_0014")]


def run(script, *args, cwd):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(script), *map(str, args)], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=300)


def read_example():
    """-> (bands in file order, {band: [P01..P09 values]})."""
    with open(EXAMPLES / "data.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    bands = list(dict.fromkeys(r["band"] for r in rows))
    series = {b: [float(r["mean_value"]) for r in sorted((r for r in rows if r["band"] == b), key=lambda r: r["period_id"])]
              for b in bands}
    return bands, series


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Execution(Base):
    def test_help_documents_every_argument(self):
        r = run(SHOW, "--help", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        for flag in ("--data", "--out"):
            self.assertIn(flag, r.stdout)

    def test_show_data_on_the_example_and_three_cells(self):
        for data in [EXAMPLES / "data.csv"] + [FIXTURES / f"{s}_{c}" / "data.csv" for s, c in CELLS]:
            out = self.tmp / "data.json"
            r = run(SHOW, "--data", data, "--out", out, cwd=self.tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            lines = r.stdout.splitlines()
            self.assertEqual(sum(1 for l in lines if l.startswith("P0")), 9, data)
            self.assertTrue(lines[-1].startswith("The month to predict is P10 = "), lines[-1])
            self.assertEqual(len(json.loads(out.read_text())["months"]), 9)


class Planted(Base):
    def test_planted_weights_recovered(self):
        bands, series = read_example()
        m, n = 3, 1
        n_blocks = 9 - m - n + 1
        z = np.concatenate([series[b][9 - m:] for b in bands])
        Z = np.column_stack([np.concatenate([series[b][j:j + m] for b in bands]) for j in range(n_blocks)])
        A = np.vstack([Z, np.ones((1, n_blocks))])          # least squares with the weights summing to one
        w = np.linalg.lstsq(A, np.concatenate([z, [1.0]]), rcond=None)[0]
        self.assertGreaterEqual(w.min(), -1e-6)
        self.assertLessEqual(np.max(np.abs(w - PLANTED)), 1e-6)
        pred = {b: float(np.array([series[b][j + m] for j in range(n_blocks)]) @ np.array(PLANTED)) for b in bands}
        expected = json.loads((EXAMPLES / "expected_output.json").read_text())
        self.assertLessEqual(max(abs(pred[b] - v) for b, v in expected["prediction"]["values"].items()), 1e-6)
        printed = dict(l.split() for l in (SKILL / "assets" / "example_prediction.txt").read_text().splitlines())
        self.assertLessEqual(max(abs(pred[b] - float(v)) for b, v in printed.items()), 1e-6)
        self.assertEqual(sorted(printed), sorted(bands))


class Parity(Base):
    def test_her_code_as_predict_py_reproduces_parity_on_three_cells(self):
        with open(FIXTURES / "parity.csv", newline="") as f:
            ref = {(r["sensor"], r["site_id"]): r for r in csv.DictReader(f)}
        self.assertEqual(sorted(ref), sorted(CELLS))
        worst = 0.0
        for sensor, site in CELLS:
            cell = self.tmp / site
            cell.mkdir()
            shutil.copy(FIXTURES / f"{sensor}_{site}" / "data.csv", cell / "data.csv")
            shutil.copy(FIXTURES / "her_predict.py", cell / "predict.py")
            r = run(cell / "predict.py", cwd=cell)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = {l.split()[0]: float(l.split()[1]) for l in r.stdout.splitlines() if l.strip()}
            row = ref[(sensor, site)]
            want = {k[5:]: float(v) for k, v in row.items() if k.startswith("pred_") and v}
            self.assertEqual(sorted(got), sorted(want), site)
            worst = max(worst, max(abs(got[b] - want[b]) for b in want))
        self.assertLessEqual(worst, 1e-8)


class Guard(Base):
    def _data(self, name, lines):
        p = self.tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_data_with_p10_is_refused(self):
        rows = (EXAMPLES / "data.csv").read_text().splitlines()
        r = run(SHOW, "--data", self._data("p10/data.csv", rows + ["VV,P10,2021-10,-10.0"]), cwd=self.tmp)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("P10", r.stderr)
        self.assertNotIn("-10.0", r.stdout)

    def test_data_without_nine_months_is_refused(self):
        rows = [l for l in (EXAMPLES / "data.csv").read_text().splitlines() if ",P09," not in l]
        r = run(SHOW, "--data", self._data("eight/data.csv", rows), cwd=self.tmp)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("must hold exactly the nine periods P01-P09", r.stderr)


if __name__ == "__main__":
    unittest.main()
