#!/bin/bash
#SBATCH --job-name=ppo_bargain
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ppo_%j.log

module load python cuda
source ~/venv/bin/activate  # adjust to your venv path

cd ~/Causal-Game-Analysis

python -m training.ppo.train \
    --num-envs 4096 \
    --iterations 200 \
    --episodes-per-iter 2000 \
    --lr 1e-3 \
    --save-dir ./checkpoints/ppo \
    --wandb-project bargaining-ppo \
    --wandb-run-name "ppo_mlp_$(date +%Y%m%d_%H%M%S)" \
    --log-interval 10
