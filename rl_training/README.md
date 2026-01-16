# RL Training Framework

A comprehensive reinforcement learning training framework for the CUDA-accelerated bargaining game environment.

## Overview

This framework provides multiple state-of-the-art algorithms for training agents in the bargaining game:

| Algorithm | Type | Description |
|-----------|------|-------------|
| **PPO** | Self-play | Proximal Policy Optimization with shared policy |
| **NFSP** | Self-play | Neural Fictitious Self-Play |
| **Sampled CFR** | Equilibrium | Deep Counterfactual Regret Minimization |
| **PSRO** | Population | Policy Space Response Oracles |
| **Ex²PSRO** | Population + Welfare | PSRO with welfare-focused exploration |
| **MAPPO** | Self-play | Multi-Agent PPO with centralized critic |
| **FCP** | Population | Fictitious Co-Play |
| **IS-MCTS** | Search | Information-Set Monte Carlo Tree Search |

## Quick Start

### 1. Install Dependencies

```bash
# Install the package
pip install -e .

# For CUDA environment (requires NVIDIA GPU)
cd cuda_bargain && pip install -e .
```

### 2. Train an Agent

```bash
# Train PPO (simplest)
python scripts/train_ppo_bargain.py --num-envs 4096 --total-timesteps 5000000

# Train with Weights & Biases logging
python scripts/train_ppo_bargain.py --wandb --wandb-project my-project
```

### 3. View Results

```bash
# List all training runs
python scripts/view_results.py --list

# Show leaderboard
python scripts/view_results.py --leaderboard
```

## Directory Structure

```
rl_training/
├── algorithms/          # Algorithm implementations
│   ├── ppo_bargain/     # PPO self-play
│   ├── nfsp_bargain/    # Neural Fictitious Self-Play
│   ├── sampled_cfr/     # Deep CFR
│   ├── psro/            # Policy Space Response Oracles
│   ├── ex2psro/         # Ex²PSRO (welfare-focused)
│   ├── mappo/           # Multi-Agent PPO
│   ├── fcp/             # Fictitious Co-Play
│   └── is_mcts/         # Information-Set MCTS
├── baselines/           # Simple baseline policies
├── core/                # Base classes and registry
├── envs/                # Environment wrappers
├── networks/            # Neural network architectures
└── utils/               # Logging, checkpointing, results
```

## Algorithms

### PPO (Proximal Policy Optimization)

Self-play training with a shared policy network for both players.

```bash
python scripts/train_ppo_bargain.py \
    --num-envs 4096 \
    --total-timesteps 10000000 \
    --lr 3e-4 \
    --network transformer \
    --wandb
```

**Key hyperparameters:**
- `--lr`: Learning rate (default: 3e-4)
- `--rollout-steps`: Steps per rollout (default: 64)
- `--network`: Network type - `mlp` or `transformer`

### NFSP (Neural Fictitious Self-Play)

Combines DQN best-response learning with supervised average policy learning.

```bash
python scripts/train_nfsp_bargain.py \
    --num-envs 4096 \
    --total-timesteps 5000000 \
    --eta 0.1 \
    --wandb
```

**Key hyperparameters:**
- `--eta`: Probability of best-response mode (default: 0.1)

### Sampled CFR (Deep Counterfactual Regret Minimization)

Neural network approximation of CFR for large games.

```bash
python scripts/train_sampled_cfr.py \
    --num-envs 4096 \
    --iterations 1000 \
    --lr 1e-3 \
    --wandb
```

### PSRO (Policy Space Response Oracles)

Iteratively builds a population by computing Nash equilibria and training best responses.

```bash
python scripts/train_psro.py \
    --psro-iterations 20 \
    --br-training-steps 100000 \
    --nash-solver replicator \
    --wandb
```

**Key hyperparameters:**
- `--psro-iterations`: Number of PSRO iterations
- `--br-training-steps`: Training steps for each best response
- `--nash-solver`: `replicator` or `fictitious_play`
- `--max-policies`: Maximum policies per player

### Ex²PSRO (Explicit Exploration PSRO)

Extends PSRO to find high-welfare equilibria by:
1. Creating exploration policies that imitate high-welfare behavior
2. Regularizing best response training toward the exploration policy
3. Biasing equilibrium selection toward prosocial outcomes

Based on "Explicit Exploration for High-Welfare Equilibria in Game-Theoretic Multiagent Reinforcement Learning" (OpenReview 2025).

```bash
python scripts/train_ex2psro.py \
    --psro-iterations 20 \
    --br-training-steps 100000 \
    --welfare-fn utilitarian \
    --kl-coef 0.1 \
    --exploration-top-k 3 \
    --wandb
```

**Key hyperparameters:**
- `--welfare-fn`: Welfare function (`utilitarian`, `nash`, `egalitarian`)
- `--kl-coef`: KL regularization coefficient toward exploration policy
- `--exploration-top-k`: Number of top welfare policies to imitate
- `--use-adaptive-kl`: Automatically adjust KL coefficient
- `--imitation-epochs`: Epochs to train exploration policy

### MAPPO (Multi-Agent PPO)

Centralized training with decentralized execution - separate actors, shared critic.

```bash
python scripts/train_mappo.py \
    --num-envs 4096 \
    --total-timesteps 10000000 \
    --share-actor \
    --wandb
```

**Key hyperparameters:**
- `--share-actor`: Share actor weights between players
- `--centralized-critic`: Use centralized critic (default: True)

### FCP (Fictitious Co-Play)

Trains against historical policy snapshots for robustness.

```bash
python scripts/train_fcp.py \
    --num-envs 4096 \
    --total-timesteps 10000000 \
    --population-size 10 \
    --snapshot-interval 10000 \
    --wandb
```

**Key hyperparameters:**
- `--population-size`: Size of historical policy pool
- `--snapshot-interval`: Steps between snapshots
- `--prioritized`: Prioritize recent policies

### IS-MCTS (Search Enhancement)

Improve any trained policy with Monte Carlo Tree Search at test time.

```bash
python scripts/evaluate_with_search.py \
    --checkpoint checkpoints/ppo_bargain/best.pt \
    --num-simulations 100 \
    --use-gumbel
```

## Hyperparameter Sweeps

### Using W&B Sweeps

```bash
# Create a sweep
wandb sweep sweeps/sweep_ppo.yaml

# Run agents (can run on multiple machines)
wandb agent <sweep_id>
```

### Using the Custom Sweep Script

```bash
# Grid search
python scripts/hyperparameter_sweep.py --algorithm ppo --method grid

# Random search with 50 trials
python scripts/hyperparameter_sweep.py --algorithm mappo --method random --num-trials 50

# Bayesian optimization (requires optuna)
python scripts/hyperparameter_sweep.py --algorithm fcp --method bayes --num-trials 100

# Compare all algorithms
python scripts/hyperparameter_sweep.py --compare-algorithms --seeds 42 123 456
```

## Results Management

Training results are automatically saved to `results/`:

```
results/
├── index.json                    # Master index
├── ppo/
│   └── run_20240115_143022_seed42/
│       ├── metadata.json         # Run metadata
│       ├── config.json           # Hyperparameters
│       ├── metrics.json          # Training history
│       ├── final_eval.json       # Final evaluation
│       ├── policy.pt             # Trained weights
│       └── checkpoints/          # Intermediate saves
```

### Viewing Results

```bash
# List all runs
python scripts/view_results.py --list

# Filter by algorithm
python scripts/view_results.py --list --algorithm ppo

# Show run details
python scripts/view_results.py --show <run_id>

# Compare runs
python scripts/view_results.py --compare run1 run2 run3

# Find best run
python scripts/view_results.py --best ppo

# Leaderboard
python scripts/view_results.py --leaderboard

# Export to CSV
python scripts/view_results.py --export results.csv
```

### Loading Trained Policies

```python
from rl_training.utils import ResultsManager

manager = ResultsManager("results")

# Get best PPO run
best = manager.get_best_run("ppo")

# Load policy weights
policy_state = manager.load_policy(best["run_id"])

# Use in your code
policy.load_state_dict(policy_state["policy_state_dict"])
```

## LLM Policies

Evaluate LLMs as negotiators in the bargaining game:

```python
from rl_training.llm_policies import OpenAIPolicy

# Create LLM policy
policy = OpenAIPolicy(
    model_name="gpt-5.2-pro",  # or "o3", "gpt-4o", etc.
    temperature=0.7,
    verbose=True,
)

# Use like any other policy
action = policy.get_action(obs, action_mask)
```

### Supported Models

| Model | Type | Best For |
|-------|------|----------|
| `gpt-5.2-pro` | Standard | Best overall capability |
| `gpt-5.2-thinking` | Reasoning | Complex strategic decisions |
| `o3` | Reasoning | Deep strategic reasoning |
| `gpt-4o` | Standard | Fast, previous gen baseline |

### Cross-Play Evaluation

Run full cross-play evaluation between LLMs, RL agents, and baselines:

```bash
python scripts/evaluate_crossplay.py \
    --policies gpt-5.2-pro o3 gpt-4o \
    --rl-checkpoints checkpoints/ppo/best.pt \
    --baselines random greedy fair_split \
    --num-games 500 \
    --output results/crossplay_matrix.json
```

This generates a payoff matrix suitable for meta-game analysis.

## Baselines

Simple baseline policies for comparison:

```python
from rl_training.baselines import (
    RandomPolicy,      # Uniform random over valid actions
    GreedyPolicy,      # Myopic best response
    AlwaysWalkPolicy,  # Always takes outside option
    FairSplitPolicy,   # Approximates Nash bargaining
)
```

Evaluate baselines:

```bash
python scripts/evaluate_baselines.py --num-games 10000
```

## Network Architectures

### MLP Policy

Simple feedforward network:
- Input: 92-dim observation
- Hidden: 256 → 256 (configurable)
- Output: 82 actions + value head

### Transformer Policy

Attention-based architecture:
- Tokenizes observation into 5 tokens (values, outside, offer, state, mask)
- Transformer encoder layers
- Better for capturing complex dependencies

Select with `--network transformer` or `--network mlp`.

## Logging

### Console Logging

Enabled by default. Shows smoothed metrics every N steps.

### Weights & Biases

```bash
# Enable with --wandb flag
python scripts/train_ppo_bargain.py --wandb --wandb-project my-project

# Optional arguments
--wandb-name "experiment-1"
--wandb-tags ppo baseline
```

### TensorBoard

```python
from rl_training.utils import create_logger

logger = create_logger(
    console=True,
    tensorboard_dir="runs/experiment",
)
```

## Environment Details

The bargaining game environment:
- **Players**: 2
- **Items**: 3 types with quantities [7, 4, 1]
- **Actions**: 82 (80 offers + accept + walk away)
- **Observations**: 92-dimensional
- **CUDA-accelerated**: Supports thousands of parallel games

## Troubleshooting

### CUDA Not Available

The CUDA environment requires an NVIDIA GPU. Check with:

```python
import torch
print(torch.cuda.is_available())
```

### Out of Memory

Reduce `--num-envs` or `--minibatch-size`.

### Training Not Converging

- Try different learning rates
- Increase `--total-timesteps`
- Try different network architectures
- Check entropy coefficient (higher = more exploration)

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{metagame2024,
  title={A Meta-Game Evaluation Framework for Deep Multiagent Reinforcement Learning},
  booktitle={IJCAI},
  year={2024}
}
```
