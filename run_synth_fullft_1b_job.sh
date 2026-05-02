#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=jieyuz_1727
#SBATCH --output=synth_fullft_1b_%j.log

module purge
module load gcc/12.3.0 cuda/12.4.1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd /project2/jieyuz_1727/Continual-Learning
source /project2/jieyuz_1727/Continual-Learning/venv/bin/activate

python -u train_carc_synthetic.py \
    --method full_ft \
    --model /scratch1/ashanmug/models/Llama-3.2-1B-Instruct \
    --run-id step2_synth_fullft_1b \
    --checkpoint-dir /scratch1/ashanmug/checkpoints
