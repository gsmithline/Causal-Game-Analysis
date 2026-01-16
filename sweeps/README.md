# Hyperparameter Sweeps

Configuration files and tools for hyperparameter optimization.

## Overview

Two sweep approaches are available:

1. **W&B Sweeps** - YAML configs for Weights & Biases sweep orchestration
2. **Python Script** - Standalone sweep runner with grid/random/Bayesian search

## W&B Sweep Configs

### Available Configs

| File | Algorithm | Search Space |
|------|-----------|--------------|
| `sweep_ppo.yaml` | PPO | lr, rollout_steps, network, num_envs |
| `sweep_nfsp.yaml` | NFSP | eta, num_envs |
| `sweep_sampled_cfr.yaml` | Sampled CFR | lr, iterations, num_envs |
| `sweep_psro.yaml` | PSRO | lr, br_training_steps, nash_solver, max_policies |
| `sweep_mappo.yaml` | MAPPO | lr, rollout_steps, share_actor, num_envs |
| `sweep_fcp.yaml` | FCP | lr, population_size, snapshot_interval, prioritized |
| `sweep_all_algorithms.yaml` | All | Algorithm comparison across seeds |

### Usage

```bash
# 1. Create a sweep (returns sweep_id)
wandb sweep sweeps/sweep_ppo.yaml

# 2. Run agents (can run multiple on different machines)
wandb agent <your-entity>/<project>/<sweep_id>

# Example:
wandb agent myteam/causal-bargain-sweeps/abc123
```

### Running Multiple Agents

For parallel sweeps across machines:

```bash
# Machine 1
wandb agent <sweep_id>

# Machine 2
wandb agent <sweep_id>

# Machine 3
wandb agent <sweep_id>
```

Each agent picks up hyperparameter configurations from the central sweep coordinator.

### Sweep Methods

The YAML configs support three search methods:

| Method | Description | Best For |
|--------|-------------|----------|
| `bayes` | Bayesian optimization | Finding optimal hyperparameters |
| `random` | Random sampling | Broad exploration |
| `grid` | Exhaustive grid search | Small search spaces |

Change method in YAML:
```yaml
method: bayes  # or random, grid
```

---

## Python Sweep Script

For sweeps without W&B or with custom logic.

### Grid Search

```bash
python scripts/hyperparameter_sweep.py \
    --algorithm ppo \
    --method grid
```

### Random Search

```bash
python scripts/hyperparameter_sweep.py \
    --algorithm mappo \
    --method random \
    --num-trials 50
```

### Bayesian Optimization

Requires `optuna`:
```bash
pip install optuna

python scripts/hyperparameter_sweep.py \
    --algorithm fcp \
    --method bayes \
    --num-trials 100
```

### Algorithm Comparison

Compare all algorithms with consistent evaluation:

```bash
python scripts/hyperparameter_sweep.py \
    --compare-algorithms \
    --seeds 42 123 456 789 1337
```

### With W&B Logging

```bash
python scripts/hyperparameter_sweep.py \
    --algorithm ppo \
    --method random \
    --num-trials 30 \
    --wandb \
    --wandb-project my-sweeps
```

### Dry Run

Preview commands without executing:

```bash
python scripts/hyperparameter_sweep.py \
    --algorithm ppo \
    --method grid \
    --dry-run
```

---

## Search Spaces

### PPO
```yaml
lr: [1e-5, 1e-3]         # log-uniform
rollout_steps: [32, 64, 128, 256]
network: [mlp, transformer]
num_envs: [1024, 2048, 4096]
clip_eps: [0.1, 0.3]     # uniform
entropy_coef: [0.001, 0.1]  # log-uniform
```

### NFSP
```yaml
eta: [0.05, 0.5]         # uniform
num_envs: [1024, 2048, 4096]
```

### Sampled CFR
```yaml
lr: [1e-4, 1e-2]         # log-uniform
iterations: [500, 1000, 2000]
num_envs: [1024, 2048, 4096]
```

### PSRO
```yaml
lr: [1e-5, 1e-3]         # log-uniform
br_training_steps: [50000, 100000, 200000]
nash_solver: [replicator, fictitious_play]
max_policies: [10, 20, 30]
num_eval_games: [500, 1000, 2000]
```

### MAPPO
```yaml
lr: [1e-5, 1e-3]         # log-uniform
rollout_steps: [32, 64, 128]
share_actor: [true, false]
num_envs: [1024, 2048, 4096]
```

### FCP
```yaml
lr: [1e-5, 1e-3]         # log-uniform
population_size: [5, 10, 20]
snapshot_interval: [5000, 10000, 20000]
prioritized: [true, false]
num_envs: [1024, 2048, 4096]
```

---

## Customizing Sweeps

### Modifying W&B Configs

Edit the YAML files to change:

**Search space:**
```yaml
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-6      # Minimum value
    max: 1e-2      # Maximum value

  network:
    values: ["mlp", "transformer"]  # Categorical choices
```

**Optimization target:**
```yaml
metric:
  name: avg_reward_p1    # Metric to optimize
  goal: maximize         # or minimize
```

**Fixed parameters:**
```yaml
command:
  - ${env}
  - python
  - ${program}
  - --total-timesteps
  - "5000000"           # Fixed value
  - ${args}             # Sweep parameters
```

### Modifying Python Search Spaces

Edit `scripts/hyperparameter_sweep.py`:

```python
SEARCH_SPACES = {
    "ppo": {
        "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
        "rollout_steps": {"type": "categorical", "choices": [32, 64, 128]},
        # Add new parameters here
    },
}
```

---

## Viewing Sweep Results

### W&B Dashboard

After running W&B sweeps, view results at:
- `https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>`

Features:
- Parallel coordinates plot
- Parameter importance
- Correlation analysis
- Best runs table

### Local Results

```bash
# List all sweep runs
python scripts/view_results.py --list

# Find best configuration
python scripts/view_results.py --best ppo --metric final_metrics.final_total_reward

# Export for analysis
python scripts/view_results.py --export sweep_results.csv
```

---

## Example Workflows

### 1. Quick Hyperparameter Tuning

```bash
# Run 20 random trials
python scripts/hyperparameter_sweep.py \
    --algorithm ppo \
    --method random \
    --num-trials 20

# Check best
python scripts/view_results.py --best ppo
```

### 2. Full W&B Sweep Pipeline

```bash
# Create sweep
wandb sweep sweeps/sweep_ppo.yaml
# Output: Created sweep with ID: abc123

# Run agents (start multiple terminals)
wandb agent myteam/causal-bargain-sweeps/abc123

# Monitor at wandb.ai
```

### 3. Multi-Seed Algorithm Comparison

```bash
# Train all algorithms with 5 seeds each
python scripts/hyperparameter_sweep.py \
    --compare-algorithms \
    --seeds 42 123 456 789 1337 \
    --wandb

# Generate comparison table
python scripts/view_results.py --leaderboard
python scripts/view_results.py --export algorithm_comparison.csv
```

### 4. Bayesian Optimization for Best Performance

```bash
# Install optuna
pip install optuna

# Run Bayesian optimization
python scripts/hyperparameter_sweep.py \
    --algorithm ppo \
    --method bayes \
    --num-trials 100 \
    --wandb

# Get best hyperparameters
python scripts/view_results.py --best ppo
```

---

## Tips

1. **Start with random search** - Faster than grid, often finds good configs
2. **Use Bayesian for final tuning** - More sample-efficient for fine-tuning
3. **Run multiple seeds** - Reduces variance in algorithm comparison
4. **Monitor with W&B** - Real-time visualization and early stopping
5. **Save all results** - Results manager keeps everything organized
