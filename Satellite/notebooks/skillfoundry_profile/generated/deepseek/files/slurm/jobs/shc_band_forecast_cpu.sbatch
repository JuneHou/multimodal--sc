#!/bin/bash
#SBATCH --job-name=shc-band-forecast
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Synthetic Historical Control (SHC) band-forecast exercise path.
#
# Runs the deterministic toy parcel through the SHC forecast script, validates
# the answer contract, and writes a JSON report under slurm/reports/. Submit
# from the repository root:
#
#   sbatch slurm/jobs/shc_band_forecast_cpu.sbatch
#
# The runner scripts are pure standard library, so no environment modules are
# required. Override PYTHON if your cluster needs a specific interpreter.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SKILL_DIR="skills/statistical-and-machine-learning-foundations-for-science/synthetic-historical-control-band-forecast"
OUT_DIR="slurm/reports/shc-band-forecast-${SLURM_JOB_ID:-local}"
mkdir -p "${OUT_DIR}"

echo "hostname: $(hostname)"
echo "started:  $(date --iso-8601=seconds)"

"${PYTHON}" "${SKILL_DIR}/scripts/run_shc_band_forecast.py" \
  --input "${SKILL_DIR}/examples/toy_parcel_data.csv" \
  --out "${OUT_DIR}/answer.json" \
  --summary-out "${OUT_DIR}/summary.json" \
  > "${OUT_DIR}/stdout.json"

"${PYTHON}" "${SKILL_DIR}/scripts/validate_answer.py" \
  --answer "${OUT_DIR}/answer.json" \
  --expected-bands ndvi,nir,red,swir \
  --truth "${SKILL_DIR}/examples/toy_parcel_truth.json" \
  > "${OUT_DIR}/validation.json"

echo "finished: $(date --iso-8601=seconds)"
echo "report:   ${OUT_DIR}"
