#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=jieyuz_1727
#SBATCH --output=step3_eval_%j.log

module purge
module load gcc/12.3.0 cuda/12.4.1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /project2/jieyuz_1727/Continual-Learning
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
source /project2/jieyuz_1727/Continual-Learning/venv/bin/activate

python -u scripts/experiments/run_step3_eval.py
