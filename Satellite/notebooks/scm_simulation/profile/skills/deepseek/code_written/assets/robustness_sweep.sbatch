#!/bin/bash
#SBATCH --job-name=sc-robustness
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-2

# Slurm exercise path for the synthetic-control skill: run the robustness ladder
# (default, lag-augmented, genoud) as a job array over one treated panel.
#
# Submit from the directory holding the panel (portable, no absolute paths):
#   sbatch --export=ALL,PANEL=tobacco_panel.csv,TREATED_PERIODS=1989-2000 \
#     <skill_dir>/assets/robustness_sweep.sbatch
#
# Requires PANEL and TREATED_PERIODS in the environment. TREATED_PERIODS may be a
# comma list ("1989,1990,1991") or a range ("1989-2000"). The estimator script
# must be on the path via ESTIMATOR (default: scripts/estimate_synthetic_control.R
# relative to the submit directory).

set -euo pipefail

: "${PANEL:?set PANEL to the panel CSV path}"
: "${TREATED_PERIODS:?set TREATED_PERIODS to a comma list or start-end range}"
ESTIMATOR="${ESTIMATOR:-scripts/estimate_synthetic_control.R}"

if [[ ! -f "${ESTIMATOR}" ]]; then
  echo "ERROR: estimator not found at ${ESTIMATOR}; set ESTIMATOR to its path." >&2
  exit 2
fi

if [[ "${TREATED_PERIODS}" == *-* && "${TREATED_PERIODS}" != *,* ]]; then
  start="${TREATED_PERIODS%-*}"; end="${TREATED_PERIODS#*-}"
  periods="$(seq -s, "${start}" "${end}")"
else
  periods="${TREATED_PERIODS}"
fi

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) extra=(--out "sc_default_${SLURM_JOB_ID}.json") ;;
  1) extra=(--lag-periods 1987,1988 --out "sc_lag_${SLURM_JOB_ID}.json") ;;
  2) extra=(--genoud --out "sc_genoud_${SLURM_JOB_ID}.json") ;;
  *) echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

echo "spec=${SLURM_ARRAY_TASK_ID} panel=${PANEL} periods=${periods}"
Rscript "${ESTIMATOR}" --panel "${PANEL}" --treated-periods "${periods}" \
  --summary-out "sc_spec${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}_summary.json" "${extra[@]}"
