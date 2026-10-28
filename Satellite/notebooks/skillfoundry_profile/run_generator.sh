#!/usr/bin/env bash
# SkillFoundry's design-skill command for one generator, in that generator's working copy (plan rows G1, G1r, G1e,
# G4, G4b, G6, G10, G11, G15, G15r, G15t, G17, G19). Run by Jun in Jun's own terminal with the key exported:
#   bash run_generator.sh gpt-5.4      needs OPENAI_API_KEY      (passed to Codex as CODEX_API_KEY)
#   bash run_generator.sh deepseek     needs OPENROUTER_API_KEY  (read by the openrouter provider in codex_home/config.toml)
#   bash run_generator.sh --probe <g>  checks the box below with no model call and no key
# GPT-5.4 at high effort is their routing default for design-skill, so the gpt-5.4 run passes no model flag.
# A run takes up to an hour (their stage timeout); everything printed also goes to generated/<generator>/run.log.
set -euo pipefail
MODE=run
if [ "${1:-}" = "--probe" ]; then MODE=probe; shift; fi
G="${1:?usage: bash run_generator.sh [--probe] gpt-5.4|deepseek}"
case "$G" in gpt-5.4|deepseek) ;; *) echo "unknown generator: $G (gpt-5.4 or deepseek)"; exit 1 ;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV=/data/wang/junh/envs/skillfoundry
WORK="$HERE/work/$G"
OUT="$HERE/generated/$G"
CHOME="$HERE/codex_home/$G"            # one Codex home per generator, so neither run can read the other's sessions

[ -d "$WORK/.git" ] || { echo "no working copy at $WORK"; exit 1; }
[ "$(git -C "$WORK" rev-parse --short HEAD)" = 4a8cc0c ] || { echo "$WORK is not at commit 4a8cc0c"; exit 1; }
[ -f "$WORK/paper/ssrn-4995085.pdf" ] || { echo "the SHC paper is missing from $WORK/paper/"; exit 1; }
mkdir -p "$CHOME"
cp -f "$HERE/codex_home/config.toml" "$CHOME/config.toml"

# G15r: the generator reads only its working copy (the repository and the paper), its own Codex home, the generator
# environment and system files, and reaches the network. The month-only DeepSeek run of 2026-10-13 read the held-out
# P10 values in llm_band_sc/lb_features_monthly.csv and the collaborator's per-cell results, because Codex's
# workspace-write sandbox limits writes only. /tmp and HOME are empty and private; the environment is cleared.
BOX=(bwrap --die-with-parent --new-session --unshare-all --share-net --clearenv
     --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/sbin /sbin
     --ro-bind /etc /etc --ro-bind /run/systemd/resolve /run/systemd/resolve
     --proc /proc --dev /dev --tmpfs /tmp
     --ro-bind "$ENV" "$ENV"
     --bind "$WORK" "$WORK"
     --bind "$CHOME" "$CHOME"
     --dir /home/generator
     --setenv HOME /home/generator --setenv CODEX_HOME "$CHOME" --setenv PATH "$ENV/bin:/usr/bin:/bin"
     --setenv PYTHONNOUSERSITE 1 --setenv LANG C.UTF-8 --setenv TERM "${TERM:-dumb}"
     --chdir "$WORK")
# use_legacy_landlock: Codex 0.116.0 runs commands through bubblewrap with --argv0, which this host's bubblewrap 0.6.1
# lacks, so every command failed (first gpt-5.4 run, 2026-10-13). Its Landlock sandbox gives the same workspace-write
# policy: writes inside the working copy, none outside, network as set below.
COMMON=(--full-auto
        --codex-config-override 'features.use_legacy_landlock=true'
        --codex-config-override 'sandbox_workspace_write.network_access=true'
        --codex-config-override 'web_search="live"')

if [ "$MODE" = probe ]; then
  OTHER="$HERE/work/$([ "$G" = gpt-5.4 ] && echo deepseek || echo gpt-5.4)"
  "${BOX[@]}" bash -c '
    seen() { [ -e "$1" ] && echo "visible" || echo "absent"; }
    echo "experiment README: $(seen "'"$HERE"'/README.md")"
    echo "simple-skill folder: $(seen "'"$HERE"'/../llm_band_sc")"
    echo "collaborator data: $(seen "'"$HERE"'/../../data")"
    echo "other working copy: $(seen "'"$OTHER"'")"
    echo "home directory: $(seen /home/grads)"
    echo "tmp entries: $(ls -A /tmp | wc -l)"
    echo "paper: $([ -r paper/ssrn-4995085.pdf ] && echo readable || echo missing)"
    echo "env names with KEY: $(env | cut -d= -f1 | grep -c KEY)"
    echo "tools: $(codex --version) | $(python --version 2>&1) | $(git --version) | $(pdftotext -v 2>&1 | head -1)"
    python scripts/sciskill_framework.py --help > /dev/null && echo "framework: starts"
    python - <<PY
import urllib.request, urllib.error
try:
    urllib.request.urlopen("https://api.openai.com/v1/models", timeout=20)
    print("network: reachable")
except urllib.error.HTTPError as e:
    print(f"network: reachable (HTTP {e.code})")
except Exception as e:
    print(f"network: failed ({e})")
PY
    codex sandbox linux --full-auto -c features.use_legacy_landlock=true -- bash -c \
      "touch .probe_in && rm .probe_in && echo codex sandbox write inside: ok; touch /home/generator/x 2>/dev/null && echo codex sandbox write outside: ALLOWED || echo codex sandbox write outside: denied"
  '
  exit 0
fi

[ -z "$(git -C "$WORK" status --porcelain)" ] || { echo "$WORK has changes: this generator has run already"; exit 1; }
[ ! -e "$OUT/run.log" ] || { echo "$OUT/run.log exists: this generator has run already"; exit 1; }
case "$G" in
  gpt-5.4)  : "${OPENAI_API_KEY:?export OPENAI_API_KEY first}"
            KEYENV=(--setenv CODEX_API_KEY "$OPENAI_API_KEY")
            MODEL=() ;;
  deepseek) : "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"
            KEYENV=(--setenv OPENROUTER_API_KEY "$OPENROUTER_API_KEY")
            MODEL=(--model deepseek/deepseek-v4.1-flash --codex-config-override 'model_provider="openrouter"') ;;
esac
mkdir -p "$OUT"
echo "generator $G | codex $("$ENV/bin/codex" --version) | python $("$ENV/bin/python" --version 2>&1) | $(date -Is)" | tee "$OUT/run.log"
"${BOX[@]}" "${KEYENV[@]}" python scripts/sciskill_framework.py "${MODEL[@]}" "${COMMON[@]}" design-skill \
  --prompt "$(cat "$HERE/prompts/design_task.md")" \
  --focus-term "Causal inference" \
  --verification-mode standard \
  --label "shc-$G" 2>&1 | tee -a "$OUT/run.log"
echo "finished $(date -Is); next: python sf_collect.py $G" | tee -a "$OUT/run.log"
