# Training Scripts

Command-line scripts for training, evaluation, and result management.

## Training Scripts

### Single Algorithm Training

| Script | Algorithm | Description |
|--------|-----------|-------------|
| `train_ppo_bargain.py` | PPO | Self-play with shared policy |
| `train_nfsp_bargain.py` | NFSP | Neural Fictitious Self-Play |
| `train_sampled_cfr.py` | Deep CFR | Sampled Counterfactual Regret |
| `train_psro.py` | PSRO | Policy Space Response Oracles |
| `train_mappo.py` | MAPPO | Multi-Agent PPO |
| `train_fcp.py` | FCP | Fictitious Co-Play |

### Basic Usage

```bash
# PPO Training
python scripts/train_ppo_bargain.py \
    --num-envs 4096 \
    --total-timesteps 10000000 \
    --lr 3e-4 \
    --seed 42

# With Weights & Biases logging
python scripts/train_ppo_bargain.py \
    --num-envs 4096 \
    --total-timesteps 10000000 \
    --wandb \
    --wandb-project causal-bargain \
    --wandb-name "ppo-experiment-1"
```

### Common Arguments

All training scripts share these arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--num-envs` | Parallel environments | 4096 |
| `--seed` | Random seed | 42 |
| `--cuda-device` | GPU device ID | 0 |
| `--lr` | Learning rate | 3e-4 |
| `--checkpoint-dir` | Save directory | `checkpoints/<algo>` |
| `--log-interval` | Log every N updates | 10 |
| `--wandb` | Enable W&B logging | False |
| `--wandb-project` | W&B project name | causal-bargain |
| `--wandb-name` | W&B run name | Auto |
| `--wandb-tags` | W&B tags | [algo, bargain] |

---

## Algorithm-Specific Arguments

### PPO (`train_ppo_bargain.py`)

```bash
python scripts/train_ppo_bargain.py \
    --total-timesteps 10000000 \
    --rollout-steps 64 \
    --network transformer  # or mlp
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--total-timesteps` | Total training steps | 10,000,000 |
| `--rollout-steps` | Steps per rollout | 64 |
| `--network` | Network type (`mlp`/`transformer`) | transformer |

### NFSP (`train_nfsp_bargain.py`)

```bash
python scripts/train_nfsp_bargain.py \
    --total-timesteps 5000000 \
    --eta 0.1
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--total-timesteps` | Total training steps | 5,000,000 |
| `--eta` | Best-response probability | 0.1 |

### Sampled CFR (`train_sampled_cfr.py`)

```bash
python scripts/train_sampled_cfr.py \
    --iterations 1000 \
    --lr 1e-3
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--iterations` | CFR iterations | 1000 |

### PSRO (`train_psro.py`)

```bash
python scripts/train_psro.py \
    --psro-iterations 20 \
    --br-training-steps 100000 \
    --nash-solver replicator \
    --max-policies 20
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--psro-iterations` | PSRO iterations | 20 |
| `--br-training-steps` | Steps per best response | 100,000 |
| `--nash-solver` | Nash solver (`replicator`/`fictitious_play`) | replicator |
| `--max-policies` | Max policies per player | 20 |
| `--num-eval-games` | Games for payoff estimation | 1000 |

### MAPPO (`train_mappo.py`)

```bash
python scripts/train_mappo.py \
    --total-timesteps 10000000 \
    --share-actor \
    --centralized-critic
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--total-timesteps` | Total training steps | 10,000,000 |
| `--rollout-steps` | Steps per rollout | 64 |
| `--share-actor` | Share actor weights | False |
| `--centralized-critic` | Use centralized critic | True |

### FCP (`train_fcp.py`)

```bash
python scripts/train_fcp.py \
    --total-timesteps 10000000 \
    --population-size 10 \
    --snapshot-interval 10000 \
    --prioritized
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--total-timesteps` | Total training steps | 10,000,000 |
| `--population-size` | Historical policy pool size | 10 |
| `--snapshot-interval` | Steps between snapshots | 10,000 |
| `--prioritized` | Prioritize recent policies | False |

---

## Evaluation Scripts

### Baseline Evaluation (`evaluate_baselines.py`)

Evaluate simple baseline policies against each other.

```bash
python scripts/evaluate_baselines.py --num-games 10000 --cuda-device 0
```

### Search Enhancement (`evaluate_with_search.py`)

Evaluate trained policies with IS-MCTS search improvement.

```bash
python scripts/evaluate_with_search.py \
    --checkpoint checkpoints/ppo_bargain/best.pt \
    --network mlp \
    --num-simulations 100 \
    --use-gumbel
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Policy checkpoint path | Required |
| `--network` | Network type | mlp |
| `--num-games` | Evaluation games | 1000 |
| `--num-simulations` | MCTS simulations per move | 100 |
| `--use-gumbel` | Use Gumbel search variant | False |
| `--c-puct` | Exploration constant | 1.5 |

---

## Sweep Scripts

### Unified Sweep Runner (`run_sweep.py`)

Run a single training trial with automatic result saving.

```bash
python scripts/run_sweep.py \
    --algorithm ppo \
    --seed 42 \
    --wandb
```

### Hyperparameter Sweep (`hyperparameter_sweep.py`)

Run hyperparameter sweeps across algorithms.

```bash
# Grid search
python scripts/hyperparameter_sweep.py \
    --algorithm ppo \
    --method grid

# Random search
python scripts/hyperparameter_sweep.py \
    --algorithm mappo \
    --method random \
    --num-trials 50

# Bayesian optimization
python scripts/hyperparameter_sweep.py \
    --algorithm fcp \
    --method bayes \
    --num-trials 100

# Compare all algorithms
python scripts/hyperparameter_sweep.py \
    --compare-algorithms \
    --seeds 42 123 456 789 1337

# Dry run (print commands only)
python scripts/hyperparameter_sweep.py \
    --algorithm ppo \
    --method grid \
    --dry-run
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--algorithm` | Algorithm to sweep | Required |
| `--method` | Search method (`grid`/`random`/`bayes`) | random |
| `--num-trials` | Trials for random/bayes | 20 |
| `--compare-algorithms` | Compare all algorithms | False |
| `--algorithms` | Algorithms to compare | All |
| `--seeds` | Seeds for comparison | [42, 123, 456] |
| `--output-dir` | Results output directory | sweep_results |
| `--dry-run` | Print commands only | False |

---

## Results Management (`view_results.py`)

View, compare, and export training results.

### List Runs

```bash
# All runs
python scripts/view_results.py --list

# Filter by algorithm
python scripts/view_results.py --list --algorithm ppo

# Filter by status
python scripts/view_results.py --list --status completed
```

### View Run Details

```bash
python scripts/view_results.py --show ppo_20240115_143022_seed42
```

### Compare Runs

```bash
python scripts/view_results.py --compare run_id_1 run_id_2 run_id_3
```

### Find Best Run

```bash
# Best by total reward
python scripts/view_results.py --best ppo

# Best by specific metric
python scripts/view_results.py --best ppo --metric final_metrics.final_reward_p0
```

### Leaderboard

```bash
python scripts/view_results.py --leaderboard
```

### Export Results

```bash
# Export to CSV
python scripts/view_results.py --export results.csv --format csv

# Export to JSON
python scripts/view_results.py --export results.json --format json

# Export specific algorithm
python scripts/view_results.py --export ppo_results.csv --algorithm ppo
```

### Load and Evaluate

```bash
python scripts/view_results.py \
    --load ppo_20240115_143022_seed42 \
    --evaluate \
    --eval-games 5000
```

---

## Example Workflows

### 1. Quick Training Run

```bash
# Train PPO for 1M steps
python scripts/train_ppo_bargain.py \
    --num-envs 2048 \
    --total-timesteps 1000000 \
    --seed 42

# View results
python scripts/view_results.py --list --algorithm ppo
```

### 2. Full Experiment with W&B

```bash
# Run training with logging
python scripts/train_ppo_bargain.py \
    --num-envs 4096 \
    --total-timesteps 10000000 \
    --wandb \
    --wandb-project bargain-experiments \
    --wandb-name "ppo-baseline-v1" \
    --wandb-tags ppo baseline production

# Check W&B dashboard for live metrics
```

### 3. Hyperparameter Search

```bash
# Run grid search
python scripts/hyperparameter_sweep.py \
    --algorithm ppo \
    --method grid \
    --wandb

# Check best configuration
python scripts/view_results.py --best ppo
```

### 4. Algorithm Comparison

```bash
# Train all algorithms with multiple seeds
python scripts/hyperparameter_sweep.py \
    --compare-algorithms \
    --seeds 42 123 456 789 1337 \
    --wandb

# View leaderboard
python scripts/view_results.py --leaderboard

# Export for paper
python scripts/view_results.py --export comparison.csv
```

### 5. Policy Evaluation with Search

```bash
# Train base policy
python scripts/train_ppo_bargain.py \
    --num-envs 4096 \
    --total-timesteps 5000000

# Evaluate with and without search
python scripts/evaluate_with_search.py \
    --checkpoint checkpoints/ppo_bargain/latest.pt \
    --num-simulations 100
```

---

## Output Locations

| Output | Location |
|--------|----------|
| Checkpoints | `checkpoints/<algorithm>/` |
| Organized results | `results/<algorithm>/run_<timestamp>_seed<N>/` |
| Sweep results | `sweep_results/` |
| W&B logs | Weights & Biases cloud |
