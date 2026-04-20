#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=jieyuz_1727
#SBATCH --output=train_carc_synth_%j.log

# Usage:
#   sbatch train_carc_synthetic_job.sh full_ft
#   sbatch train_carc_synthetic_job.sh lora
#   sbatch train_carc_synthetic_job.sh smf
#   sbatch train_carc_synthetic_job.sh casm

set -euo pipefail

METHOD="${1:?Usage: sbatch train_carc_synthetic_job.sh <full_ft|lora|smf|casm>}"

module purge
module load gcc/12.3.0 cuda/12.4.1 python/3.11.9

cd /project2/jieyuz_1727/Continual-Learning
source /scratch1/ramyakri/cl_venv/bin/activate

python -u train_carc_synthetic.py --method "$METHOD"
