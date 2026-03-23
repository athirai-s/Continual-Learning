#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=jieyuz_1727
#SBATCH --output=train_%j.log

module load gcc/12.3.0
module load cuda/12.4.1

cd /project2/jieyuz_1727/Continual-Learning
source venv/bin/activate

MODEL_ID="/scratch1/ashanmug/models/Llama-3.2-3B-Instruct"
RUN_ID="slurm_${SLURM_JOB_ID:-manual}"

python main.py \
  --mode real \
  --model-name "$MODEL_ID" \
  --dataset-name temporal_wiki \
  --run-id "$RUN_ID" \
  --checkpoint-dir checkpoints
