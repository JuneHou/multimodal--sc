#!/usr/bin/env bash
# The four solver models on one SCM skill, both cases (dgp1, dgp2), through the pipeline's scs_run.py.
# Run by Jun in Jun's own terminal with ARC_LLM_API_KEY and OPENAI_API_KEY exported:
#   bash run_solvers.sh configure          skills/scm-configure
#   bash run_solvers.sh design             skills/scm-design
#   bash run_solvers.sh profile-gpt-5.4    profile/skills/gpt-5.4/code_provided  (after scs_profile.py collect)
#   bash run_solvers.sh profile-deepseek   profile/skills/deepseek/code_provided (after scs_profile.py collect)
# Models, token caps, effort and step cap are scs_run.py's: gpt-oss-120b, GLM-5.3 and DeepSeek-V4.1-Flash through
# ARC, gpt-5.5 through OpenAI, low effort, 12 steps. The solver's Python is the satellite Python, as in the earlier
# solver runs. A stopped run resumes where it stopped when the same command is run again. Records go to
# runs/<label>/, and everything printed also goes to runs/solvers.log.
set -euo pipefail
ARM="${1:?usage: bash run_solvers.sh configure|design|profile-gpt-5.4|profile-deepseek}"
HERE="$(cd "$(dirname "$0")" && pwd)"
case "$ARM" in
  configure)        SKILL="$HERE/skills/scm-configure";                  LABEL=simulation ;;
  design)           SKILL="$HERE/skills/scm-design";                     LABEL=simulation ;;
  profile-gpt-5.4)  SKILL="$HERE/profile/skills/gpt-5.4/code_provided";  LABEL=simulation_profile-gpt-5.4 ;;
  profile-deepseek) SKILL="$HERE/profile/skills/deepseek/code_provided"; LABEL=simulation_profile-deepseek ;;
  *) echo "unknown arm: $ARM"; exit 1 ;;
esac
[ -d "$SKILL" ] || { echo "no skill folder at $SKILL"; exit 1; }
: "${ARC_LLM_API_KEY:?export ARC_LLM_API_KEY first}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY first}"
PY=/data/wang/junh/envs/satellite/bin/python
mkdir -p "$HERE/runs"
LOG="$HERE/runs/solvers.log"
cd "$HERE/../sc_skills"
echo "solvers $ARM | $SKILL | $($PY --version 2>&1) | $(date -Is)" | tee -a "$LOG"
"$PY" scs_run.py --method ../../scm_simulation --use simulation --skill "$SKILL" --label "$LABEL" 2>&1 | tee -a "$LOG"
echo "finished $ARM $(date -Is)" | tee -a "$LOG"
