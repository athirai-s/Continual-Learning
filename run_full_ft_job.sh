#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=jieyuz_1727
#SBATCH --output=full_ft_3b_%j.log

set -euo pipefail

module purge
module load gcc/12.3.0 cuda/12.4.1

cd /project2/jieyuz_1727/Continual-Learning
source /project2/jieyuz_1727/Continual-Learning/venv/bin/activate

python run_full_ft.py
