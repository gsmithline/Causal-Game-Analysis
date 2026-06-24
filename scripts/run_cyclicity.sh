#!/bin/bash
#SBATCH --job-name=curb_cyclicity
#SBATCH --account=wellman98
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gsmithl/Causal-Game-Analysis/logs/cyclicity_%j.log
#SBATCH --error=/home/gsmithl/Causal-Game-Analysis/logs/cyclicity_%j.err

source /home/gsmithl/.venvs/causal/bin/activate
cd /home/gsmithl/Causal-Game-Analysis

# Download bargaining data from HuggingFace if not present
# Compute Nash clustering + CURB coverage for all games up to 1100 strategies
python3 scripts/curb_coverage_vs_cyclicity.py \
    --hf-data \
    --max-n-clustering 1100 \
    --max-n-curb 1100 \
    --outdir results/ \
    --spinning-tops-pkl data/spinning_top_payoffs.pkl
