#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=jieyuz_1727
#SBATCH --output=pretrain_p1_%j.log

module purge
module load gcc/12.3.0 cuda/12.4.1

# Prevent OpenBLAS/numpy from spawning too many threads (hangs dataset loading)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /project2/jieyuz_1727/Continual-Learning
source /project2/jieyuz_1727/Continual-Learning/venv/bin/activate

python -u pretrain_period1.py
