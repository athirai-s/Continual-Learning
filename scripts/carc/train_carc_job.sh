#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=jieyuz_1727
#SBATCH --output=train_carc_%j.log

# Usage:
#   sbatch scripts/carc/train_carc_job.sh full_ft
#   sbatch scripts/carc/train_carc_job.sh lora
#   sbatch scripts/carc/train_carc_job.sh smf
#   sbatch scripts/carc/train_carc_job.sh casm

set -euo pipefail

METHOD="${1:?Usage: sbatch scripts/carc/train_carc_job.sh <full_ft|lora|smf|casm>}"

module purge
module load gcc/12.3.0 cuda/12.4.1

cd /project2/jieyuz_1727/Continual-Learning
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
source /project2/jieyuz_1727/Continual-Learning/venv/bin/activate

python -u scripts/carc/train_carc.py --method "$METHOD"
