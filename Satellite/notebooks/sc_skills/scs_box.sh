#!/usr/bin/env bash
# scs_box.sh — SkillFoundry's design-skill command for one generator, inside the generator's box (profile step 3).
# The box and the command are skillfoundry_profile/run_generator.sh's, with the working copy, Codex home, output
# folder, request file, label and focus term given as arguments, so it serves any method package (scs_profile.py
# prepares the working copy and writes the request). Run by Jun in Jun's own terminal with the key exported:
#   bash scs_box.sh run   <gpt-5.4|deepseek> <work> <codex home> <out folder> <request file> <label> <focus term>
#       gpt-5.4 needs OPENAI_API_KEY (passed to Codex as CODEX_API_KEY); deepseek needs OPENROUTER_API_KEY
#   bash scs_box.sh probe <gpt-5.4|deepseek> <work> <codex home> <path that must be absent>...
#       checks the box with no model call and no key
# The generator reads only its working copy (the repository, the paper in paper/ and the reference code in code/), its
# own Codex home, the generator environment and system files, and reaches the network. /tmp and HOME are empty.
set -euo pipefail
MODE="${1:?usage: bash scs_box.sh run|probe <generator> <work> <codex home> ...}"
G="${2:?generator}"
WORK="$(cd "${3:?work}" && pwd)"
CHOME="$(mkdir -p "${4:?codex home}" && cd "$4" && pwd)"
case "$G" in gpt-5.4|deepseek) ;; *) echo "unknown generator: $G (gpt-5.4 or deepseek)"; exit 1 ;; esac
ENV=/data/wang/junh/envs/skillfoundry
[ -d "$WORK/.git" ] || { echo "no working copy at $WORK"; exit 1; }
[ "$(git -C "$WORK" rev-parse --short HEAD)" = 4a8cc0c ] || { echo "$WORK is not at commit 4a8cc0c"; exit 1; }
ls "$WORK"/paper/*.pdf > /dev/null 2>&1 || { echo "no paper in $WORK/paper/"; exit 1; }
[ -f "$CHOME/config.toml" ] || { echo "no config.toml in $CHOME"; exit 1; }

# SCS_BOX_RUNTIME, when set, is a language runtime the method's code needs (an R installation with the package),
# bound read-only and placed on PATH after the generator environment, so the generator can run and test that code.
RUNTIME="${SCS_BOX_RUNTIME:-}"
RT_BIND=(); RT_PATH=""
if [ -n "$RUNTIME" ]; then
  [ -d "$RUNTIME/bin" ] || { echo "no runtime at $RUNTIME"; exit 1; }
  RT_BIND=(--ro-bind "$RUNTIME" "$RUNTIME"); RT_PATH=":$RUNTIME/bin"
fi
BOX=(bwrap --die-with-parent --new-session --unshare-all --share-net --clearenv
     --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/sbin /sbin
     --ro-bind /etc /etc --ro-bind /run/systemd/resolve /run/systemd/resolve
     --proc /proc --dev /dev --tmpfs /tmp
     --ro-bind "$ENV" "$ENV"
     ${RT_BIND[@]+"${RT_BIND[@]}"}
     --bind "$WORK" "$WORK"
     --bind "$CHOME" "$CHOME"
     --dir /home/generator
     --setenv HOME /home/generator --setenv CODEX_HOME "$CHOME" --setenv PATH "$ENV/bin$RT_PATH:/usr/bin:/bin"
     --setenv PYTHONNOUSERSITE 1 --setenv LANG C.UTF-8 --setenv TERM "${TERM:-dumb}"
     --chdir "$WORK")
# use_legacy_landlock: Codex 0.116.0 runs commands through bubblewrap with --argv0, which this host's bubblewrap 0.6.1
# lacks; its Landlock sandbox gives the same workspace-write policy (run_generator.sh).
COMMON=(--full-auto
        --codex-config-override 'features.use_legacy_landlock=true'
        --codex-config-override 'sandbox_workspace_write.network_access=true'
        --codex-config-override 'web_search="live"')

if [ "$MODE" = probe ]; then
  shift 4
  ABSENT=""
  for p in "$@"; do ABSENT="$ABSENT $(printf '%q' "$p")"; done
  "${BOX[@]}" bash -c '
    for p in '"$ABSENT"'; do [ -e "$p" ] && echo "VISIBLE: $p" || echo "absent: $p"; done
    echo "tmp entries: $(ls -A /tmp | wc -l)"
    echo "paper: $(ls paper/*.pdf > /dev/null 2>&1 && echo readable || echo missing)"
    echo "code: $([ -d code ] && echo "$(find code -type f | wc -l) files" || echo none)"
    echo "env names with KEY: $(env | cut -d= -f1 | grep -c KEY)"
    echo "tools: $(codex --version) | $(python --version 2>&1) | $(git --version)"
    python scripts/sciskill_framework.py --help > /dev/null && echo "framework: starts"
    if command -v Rscript > /dev/null; then
      echo "runtime: $(Rscript -e '\''cat(R.version.string, "| Synth", as.character(packageVersion("Synth")), "| jsonlite", as.character(packageVersion("jsonlite")))'\'' 2>&1 | tail -1)"
    else
      echo "runtime: no Rscript"
    fi
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
    if command -v Rscript > /dev/null; then
      codex sandbox linux --full-auto -c features.use_legacy_landlock=true -- bash -c \
        "Rscript -e '\''suppressPackageStartupMessages({library(Synth); library(jsonlite)}); cat(\"codex sandbox runs R: ok\\n\")'\'' 2>&1 | tail -1"
    fi
  '
  exit 0
fi

[ "$MODE" = run ] || { echo "mode is run or probe"; exit 1; }
OUT="$(mkdir -p "${5:?out folder}" && cd "$5" && pwd)"
REQUEST="${6:?request file}"; LABEL="${7:?label}"; FOCUS="${8:?focus term}"
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
echo "generator $G | codex $("$ENV/bin/codex" --version) | python $("$ENV/bin/python" --version 2>&1) | $(date -Is)" | tee "$OUT/run.log"
"${BOX[@]}" "${KEYENV[@]}" python scripts/sciskill_framework.py "${MODEL[@]}" "${COMMON[@]}" design-skill \
  --prompt "$(cat "$REQUEST")" \
  --focus-term "$FOCUS" \
  --verification-mode standard \
  --label "$LABEL" 2>&1 | tee -a "$OUT/run.log"
echo "finished $(date -Is); next: python scs_profile.py collect ..." | tee -a "$OUT/run.log"
