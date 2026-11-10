#!/usr/bin/env bash
# The four solver models on one SCM skill, both cases (dgp1, dgp2) of one noise level, through the pipeline's scs_run.py.
# Run by Jun in Jun's own terminal with ARC_LLM_API_KEY and OPENAI_API_KEY exported:
#   bash run_solvers.sh <format> [sigma] [cases]
#     format   configure         skills/scm-configure                      records in scm-configure/sigma_<s>/
#              design            skills/scm-design                         records in scm-design/sigma_<s>/
#              profile-gpt-5.4   profile/skills/gpt-5.4/code_provided      records in skill_profile/gpt-5.4/sigma_<s>/
#              profile-deepseek  profile/skills/deepseek/code_provided     records in skill_profile/deepseek/sigma_<s>/
#              four-design       four_design/skills/scm-four-design        records in scm-four-design/sigma_<s>/
#     sigma    1.0 (default), 0.5 or 0.2: the collaborator's noise level, data/simulation/simulation_data/sigma_<s>/
#     cases    run only the first N cases of the level (scs_run --limit), for a first check; all cases when absent
#   MODELS="gpt-oss-120b GLM-5.3 DeepSeek-V4.1-Flash-thinking-low" bash run_solvers.sh ...   runs only the named models
#     (scs_run --models); all four when MODELS is unset. Each key is required only when a named model needs it.
# Models, effort and step cap are scs_run.py's: gpt-oss-120b, GLM-5.3 and DeepSeek-V4.1-Flash through ARC, gpt-5.5
# through OpenAI, low effort, 12 steps, no reply-length limit (every ARC call streamed). The solver's Python is the
# satellite Python, as in the earlier solver runs. A stopped run resumes where it stopped when the same command is run
# again. Everything printed also goes to solvers.log.
set -euo pipefail
ARM="${1:?usage: bash run_solvers.sh configure|design|profile-gpt-5.4|profile-deepseek|four-design [1.0|0.5|0.2] [cases]}"
SIGMA="${2:-1.0}"
CASES="${3:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
case "$SIGMA" in
  1.0) USE=simulation ;;
  0.5|0.2) USE="sigma_$SIGMA" ;;
  *) echo "unknown sigma: $SIGMA (1.0, 0.5 or 0.2)"; exit 1 ;;
esac
case "$ARM" in
  configure)        SKILL="$HERE/skills/scm-configure";                  OUT="$HERE/scm-configure/sigma_$SIGMA" ;;
  design)           SKILL="$HERE/skills/scm-design";                     OUT="$HERE/scm-design/sigma_$SIGMA" ;;
  profile-gpt-5.4)  SKILL="$HERE/profile/skills/gpt-5.4/code_provided";  OUT="$HERE/skill_profile/gpt-5.4/sigma_$SIGMA" ;;
  profile-deepseek) SKILL="$HERE/profile/skills/deepseek/code_provided"; OUT="$HERE/skill_profile/deepseek/sigma_$SIGMA" ;;
  four-design)      SKILL="$HERE/four_design/skills/scm-four-design";   OUT="$HERE/scm-four-design/sigma_$SIGMA" ;;
  *) echo "unknown format: $ARM"; exit 1 ;;
esac
[ -d "$SKILL" ] || { echo "no skill folder at $SKILL"; exit 1; }
MODELS="${MODELS:-gpt-oss-120b GLM-5.3 DeepSeek-V4.1-Flash-thinking-low openai:gpt-5.5}"
for m in $MODELS; do
  case "$m" in
    openai:*) : "${OPENAI_API_KEY:?export OPENAI_API_KEY first (model $m)}" ;;
    *)        : "${ARC_LLM_API_KEY:?export ARC_LLM_API_KEY first (model $m)}" ;;
  esac
done
PY=/data/wang/junh/envs/satellite/bin/python
LOG="$HERE/solvers.log"
cd "$HERE/../sc_skills"
echo "solvers $ARM sigma $SIGMA cases ${CASES/#0/all} models $MODELS | $SKILL | $($PY --version 2>&1) | $(date -Is)" | tee -a "$LOG"
# shellcheck disable=SC2086
"$PY" scs_run.py --method ../../scm_simulation --use "$USE" --skill "$SKILL" --out "$OUT" --limit "$CASES" \
  --models $MODELS 2>&1 | tee -a "$LOG"
echo "finished $ARM sigma $SIGMA $(date -Is)" | tee -a "$LOG"
