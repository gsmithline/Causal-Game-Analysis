#!/bin/bash
#SBATCH --job-name=psro_bargain
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=psro_%j.log

module load python cuda
source ~/venv/bin/activate  # adjust to your venv path

cd ~/Causal-Game-Analysis

# Install nashpy if not present (for Nash equilibrium solving)
pip install nashpy --quiet

python -m training.psro.train \
    --iterations 20 \
    --num-envs 4096 \
    --episodes-per-br 10000 \
    --episodes-per-eval 2000 \
    --lr 1e-3 \
    --save-dir ./checkpoints/psro \
    --wandb-project bargaining-psro \
    --wandb-run-name "psro_$(date +%Y%m%d_%H%M%S)" \
    --log-interval 1
