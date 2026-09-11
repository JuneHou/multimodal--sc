#!/bin/bash
# tinkercliffs job: MODEL=qwen|bagel LAYOUT=persite|sheet [ENV=bagel] sbatch vg_generate.sbatch [--limit 4] [--seed 0]
#SBATCH --job-name=vg
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err
set -euo pipefail
module load Miniconda3
MODEL="${MODEL:-qwen}"
LAYOUT="${LAYOUT:-persite}"
PROJ="/projects/slmreasoning/junh"
ENV_PATH="$PROJ/envs/${ENV:-vlmgen}"
PYTHON="$ENV_PATH/bin/python"
REPO_DIR="$PROJ/vlm_generation"
export HF_HOME="$PROJ/hf_cache" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export BAGEL_REPO="$PROJ/Bagel"
cd "$REPO_DIR"
echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"; echo "Host: $(hostname)"; echo "PWD: $(pwd)"; "$PYTHON" -V
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MP=""
if [ "$MODEL" = "bagel" ] && [ -d "$PROJ/hf_models/BAGEL-7B-MoT" ]; then MP="--model-path $PROJ/hf_models/BAGEL-7B-MoT"; fi
"$PYTHON" -u vg_generate.py --model "$MODEL" --layout "$LAYOUT" $MP "$@"
EC=$?
echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"; echo "Exit code: $EC"; exit $EC
