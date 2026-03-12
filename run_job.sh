#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=jieyuz_1727
#SBATCH --output=train_%j.log

module load gcc/12.3.0
module load cuda/12.4.1

cd /project2/jieyuz_1727/Continual-Learning
source venv/bin/activate

python 3B_train.py
