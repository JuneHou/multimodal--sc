#!/usr/bin/env bash
# One SkillFoundry generator on the SCM request, through the pipeline's scs_box.sh with this experiment's paths filled
# in, and the solvers' R environment in the box so the generator can run and test R code. Run by Jun in Jun's own
# terminal:
#   bash run_generator.sh gpt-5.4      needs OPENAI_API_KEY exported
#   bash run_generator.sh deepseek     needs OPENROUTER_API_KEY exported
#   bash run_generator.sh probe <gpt-5.4|deepseek>   the box check, no key and no model call
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOX="$HERE/../sc_skills/scs_box.sh"
P="$HERE/profile"
export SCS_BOX_RUNTIME=/data/wang/junh/envs/rsynth      # R 4.3.3 with Synth 1.1-10 and jsonlite, as the solvers have
if [ "${1:-}" = probe ]; then
  G="${2:?usage: bash run_generator.sh probe gpt-5.4|deepseek}"
  exec bash "$BOX" probe "$G" "$P/work/$G" "$P/codex_home/$G" "$HERE/../../data" "$HOME"
fi
G="${1:?usage: bash run_generator.sh gpt-5.4|deepseek}"
case "$G" in gpt-5.4|deepseek) ;; *) echo "unknown generator: $G (gpt-5.4 or deepseek)"; exit 1 ;; esac
exec bash "$BOX" run "$G" "$P/work/$G" "$P/codex_home/$G" "$P/out/$G" "$P/request.md" "scm-$G" "Causal inference"
