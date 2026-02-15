#!/bin/bash
#SBATCH --job-name=nfsp_20k_iter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nfsp_test_%j.out
#SBATCH --error=logs/nfsp_test_%j.err
#SBATCH --account=wellman98
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsmithl@umich.edu
#SBATCH --chdir=/home/gsmithl/Causal-Game-Analysis

set -euo pipefail
mkdir -p logs

echo "=============================================="
echo "NFSP Training - Bargaining Game (~4 hours)"
echo "=============================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Nodes  : $SLURM_NODELIST"
echo "=============================================="

# Load modules
module load python3.11-anaconda
module load cuda

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch_env

# Ensure we use only conda environment packages
export PYTHONNOUSERSITE=1
unset PYTHONPATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Weights & Biases configuration
export WANDB_ENTITY="gsmithl-university-of-michigan"
export WANDB_PROJECT="Iterative-Meta-Game-Analysis"
export WANDB_RUN_NAME="nfsp_test_${SLURM_JOB_ID}"
export WANDB_API_KEY="9e4245617ee0b64c178395bb5d6eaffb3815a69b"

nvidia-smi -L || true

# Install dependencies
echo "Installing dependencies..."
"$CONDA_PREFIX/bin/pip" install torch numpy matplotlib typing_extensions wandb -q

# Build CUDA extension
echo "Building CUDA extension..."
"$CONDA_PREFIX/bin/pip" install -e ./simulator --no-build-isolation -q
echo "CUDA extension ready."

# Hyperparameters for ~4 hour run
NUM_ENVS=2048
ITERATIONS=20000
EPISODES_PER_ITER=1000
ETA=0.3
EPSILON=0.06
Q_LR=3e-4
POLICY_LR=3e-4
BATCH_SIZE=256
SEED=42
LOG_INTERVAL=100
EXPLOITABILITY_INTERVAL=250
SAVE_INTERVAL=2000       # Checkpoint every 500 iterations (~10 checkpoints)
SAVE_DIR="./checkpoints/nfsp_test_${SLURM_JOB_ID}"

mkdir -p "$SAVE_DIR"

echo ""
echo "NFSP Training Configuration (~4 hours):"
echo "  Architecture:     MLP"
echo "  Environments:     $NUM_ENVS"
echo "  Iterations:       $ITERATIONS"
echo "  Episodes/iter:    $EPISODES_PER_ITER"
echo "  Eta (BR prob):    $ETA"
echo "  Epsilon:          $EPSILON"
echo "  Q Learning rate:  $Q_LR"
echo "  Policy LR:        $POLICY_LR"
echo "  Batch size:       $BATCH_SIZE"
echo "  Save interval:    $SAVE_INTERVAL"
echo "  Seed:             $SEED"
echo "  Save directory:   $SAVE_DIR"
echo "=============================================="
echo ""

"$CONDA_PREFIX/bin/python" -u -m training.nfsp.train \
    --num-envs $NUM_ENVS \
    --iterations $ITERATIONS \
    --episodes-per-iter $EPISODES_PER_ITER \
    --eta $ETA \
    --epsilon $EPSILON \
    --q-lr $Q_LR \
    --policy-lr $POLICY_LR \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --save-dir "$SAVE_DIR" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --log-interval $LOG_INTERVAL \
    --exploitability-interval $EXPLOITABILITY_INTERVAL \
    --save-interval $SAVE_INTERVAL

echo "NFSP Training complete!"
