from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCANPY_PYTHON = ROOT / "slurm" / "envs" / "scanpy" / "bin" / "python"


class LocalSkillSmokeTests(unittest.TestCase):
    def test_snakemake_layout_check(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                "skills/reproducible-workflows/snakemake-toy-workflow-starter/scripts/check_workflow_layout.py",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn("layout looks complete", completed.stdout)

    def test_snakemake_runtime_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "snakemake-summary.json"
            workspace = Path(tmp_dir) / "workspace"
            subprocess.run(
                [
                    "python3",
                    "skills/reproducible-workflows/snakemake-toy-workflow-starter/scripts/run_snakemake_workflow.py",
                    "--workspace",
                    str(workspace),
                    "--summary-out",
                    str(summary_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=240,
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["snakemake_version"], "9.16.3")
            self.assertEqual(payload["copied_text"], "toy input for snakemake starter")
            self.assertEqual(payload["result_files"], ["copied.txt", "summary.json"])
            self.assertEqual(payload["summary"]["line_count"], 1)

    def test_scanpy_preflight_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "summary.json"
            subprocess.run(
                [
                    "python3",
                    "skills/transcriptomics/scanpy-qc-starter/scripts/preflight_counts.py",
                    "--input",
                    "skills/transcriptomics/scanpy-qc-starter/examples/toy_counts.tsv",
                    "--out",
                    str(output_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["genes"], 4)
            self.assertEqual(payload["cells"], 3)

    def test_scanpy_qc_runtime(self) -> None:
        self.assertTrue(SCANPY_PYTHON.exists(), f"Missing Scanpy interpreter: {SCANPY_PYTHON}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "scanpy-summary.json"
            h5ad_path = Path(tmp_dir) / "toy-counts.h5ad"
            subprocess.run(
                [
                    str(SCANPY_PYTHON),
                    "skills/transcriptomics/scanpy-qc-starter/scripts/run_scanpy_qc.py",
                    "--input",
                    "skills/transcriptomics/scanpy-qc-starter/examples/toy_counts.tsv",
                    "--summary-out",
                    str(summary_path),
                    "--h5ad-out",
                    str(h5ad_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["genes"], 4)
            self.assertEqual(payload["cells"], 3)
            self.assertGreater(payload["max_pct_counts_mt"], 0.0)
            self.assertEqual(len(payload["per_cell"]), 3)
            self.assertTrue(h5ad_path.exists())

    def test_scanpy_ranked_genes_runtime(self) -> None:
        self.assertTrue(SCANPY_PYTHON.exists(), f"Missing Scanpy interpreter: {SCANPY_PYTHON}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "ranked-genes.json"
            subprocess.run(
                [
                    str(SCANPY_PYTHON),
                    "skills/transcriptomics/scanpy-ranked-genes-starter/scripts/run_scanpy_ranked_genes.py",
                    "--input",
                    "skills/transcriptomics/scanpy-ranked-genes-starter/examples/toy_counts.tsv",
                    "--groups",
                    "skills/transcriptomics/scanpy-ranked-genes-starter/examples/toy_groups.tsv",
                    "--top-n",
                    "2",
                    "--out",
                    str(out_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["groups"], ["group_a", "group_b"])
            self.assertEqual(payload["top_markers_by_group"]["group_a"][0]["gene"], "GeneA")
            self.assertEqual(payload["top_markers_by_group"]["group_b"][0]["gene"], "GeneC")

    def test_synth_single_treated_unit_scm_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            answer_path = Path(tmp_dir) / "answer.json"
            diagnostics_path = Path(tmp_dir) / "diagnostics.json"
            subprocess.run(
                [
                    "Rscript",
                    "skills/statistical-and-machine-learning-foundations-for-science/synth-single-treated-unit-scm/scripts/run_synth_single_treated_unit_scm.R",
                    "--panel",
                    "skills/statistical-and-machine-learning-foundations-for-science/synth-single-treated-unit-scm/examples/toy_tobacco_panel.csv",
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
            payload = json.loads(answer_path.read_text(encoding="utf-8"))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload["counterfactual"].keys()), ["2006", "2007", "2008"])
            self.assertLess(payload["att"], -1.0)
            self.assertLess(diagnostics["pre_rmspe"], 0.02)


if __name__ == "__main__":
    unittest.main()
