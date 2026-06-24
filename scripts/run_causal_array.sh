#!/bin/bash
#SBATCH --job-name=causal_egta
#SBATCH --account=wellman98
#SBATCH --partition=standard
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gsmithl/Causal-Game-Analysis/logs/causal_%A_%a.log
#SBATCH --error=/home/gsmithl/Causal-Game-Analysis/logs/causal_%A_%a.err

source /home/gsmithl/.venvs/causal/bin/activate
cd /home/gsmithl/Causal-Game-Analysis

python3 -m src.iterative_game_analysis.causal_analysis_runner \
    --domain bargaining \
    --n-bootstrap 20 \
    --solvers mene maxent_cce \
    --max-workers 8 \
    --mc-curb-budget 200 \
    --seed $SLURM_ARRAY_TASK_ID \
    --output data/analysis/causal_bargaining_chunk_${SLURM_ARRAY_TASK_ID}
