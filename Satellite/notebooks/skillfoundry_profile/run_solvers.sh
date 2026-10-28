#!/usr/bin/env bash
# The solver models on both generated skills, both formats, one dataset at a time. Run by Jun in Jun's own terminal
# with the two keys exported:
#   bash run_solvers.sh monthly     the 27 monthly cells through sf_run.py
#   bash run_solvers.sh daily       the 52 daily cells through sf_daily.py
# Four models per generator and format (sf_run.DEFAULT_MODELS): three through ARC, gpt-5.5 through OpenAI.
# The solver's Python is the interpreter that starts the runner, so it is the satellite Python of row G8d.
# A stopped run resumes where it stopped when the same command is run again. Everything printed also goes to
# runs/solvers_<dataset>.log.
set -euo pipefail
DATASET="${1:?usage: bash run_solvers.sh monthly|daily}"
case "$DATASET" in
  monthly) RUNNER=sf_run.py ;;
  daily)   RUNNER=sf_daily.py ;;
  *) echo "unknown dataset: $DATASET (monthly or daily)"; exit 1 ;;
esac
: "${ARC_LLM_API_KEY:?export ARC_LLM_API_KEY first}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY first}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/data/wang/junh/envs/satellite/bin/python
cd "$HERE"
mkdir -p runs
LOG="runs/solvers_${DATASET}.log"
echo "solvers $DATASET | $RUNNER | $($PY --version 2>&1) | $(date -Is)" | tee -a "$LOG"
for g in gpt-5.4 deepseek; do
  for f in code_provided code_written; do
    "$PY" "$RUNNER" --generator "$g" --format "$f" 2>&1 | tee -a "$LOG"
  done
done
echo "finished $DATASET $(date -Is)" | tee -a "$LOG"
