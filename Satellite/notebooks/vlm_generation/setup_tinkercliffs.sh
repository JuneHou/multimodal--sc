#!/bin/bash
# One-time setup on the tinkercliffs LOGIN node (internet access): env, BAGEL repo, model weights.
# Run:  bash setup_tinkercliffs.sh        (idempotent; re-run to resume downloads)
set -euo pipefail
module load Miniconda3
PROJ="/projects/slmreasoning/junh"
ENV_PATH="$PROJ/envs/vlmgen"
export HF_HOME="$PROJ/hf_cache"
mkdir -p "$PROJ" "$HF_HOME" "$PROJ/hf_models"
if [ ! -x "$ENV_PATH/bin/python" ]; then
  conda create -y -p "$ENV_PATH" -c conda-forge --override-channels python=3.10
fi
PIP="$ENV_PATH/bin/pip"
"$PIP" install --upgrade pip
"$PIP" install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu124   # BAGEL requirements pin 2.5.1
"$PIP" install -U "diffusers>=0.36" transformers tokenizers accelerate "safetensors>=0.8" pillow huggingface_hub numpy pandas
# BAGEL inference code in its OWN env (its requirements pin transformers 4.49 / safetensors 0.4, too old for diffusers)
if [ ! -d "$PROJ/Bagel" ]; then git clone https://github.com/ByteDance-Seed/Bagel.git "$PROJ/Bagel"; fi
BENV="$PROJ/envs/bagel"
if [ ! -x "$BENV/bin/python" ]; then conda create -y -p "$BENV" -c conda-forge --override-channels python=3.10; fi
"$BENV/bin/pip" install --upgrade pip
"$BENV/bin/pip" install -r "$PROJ/Bagel/requirements.txt" || echo "BAGEL requirements: some pins failed, check manually"
"$BENV/bin/pip" install flash_attn==2.5.8 --no-build-isolation || echo "flash_attn not installed (optional)"
"$BENV/bin/pip" install pillow huggingface_hub
# weights: check the shared model store first, else download
ls -d /common/data/models/*Qwen-Image-Edit* /common/data/models/*BAGEL* 2>/dev/null || true
"$ENV_PATH/bin/huggingface-cli" download Qwen/Qwen-Image-Edit-2511
"$BENV/bin/huggingface-cli" download ByteDance-Seed/BAGEL-7B-MoT --local-dir "$PROJ/hf_models/BAGEL-7B-MoT"
"$ENV_PATH/bin/python" - <<'PY'
import diffusers, transformers, torch
from diffusers import QwenImageEditPlusPipeline
print("ok: diffusers", diffusers.__version__, "transformers", transformers.__version__, "torch", torch.__version__)
PY
echo "setup done"
