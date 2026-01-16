# Causal Meta-Game Analysis

A framework for analyzing multi-agent systems using Structural Causal Models (SCM) and do-calculus. This framework provides rigorous causal semantics for evaluating policies in meta-games across three levels of analysis.

## Installation

Requires Python 3.10+. Install using [uv](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/your-username/Causal-Game-Analysis.git
cd Causal-Game-Analysis

# Sync dependencies (creates venv automatically)
uv sync

# For development (includes testing and linting tools)
uv sync --extra dev
```

## Quick Start

### Basic Usage

```python
import pandas as pd
import numpy as np
from causal_game_analysis import (
    MetaGame,
    Bootstrap,
    level1_analysis,
    ecosystem_lift,
    shapley_value,
)

# Create or load cross-play data
# Format: DataFrame with columns (policy_i, policy_j, outcome)
df = pd.DataFrame([
    {"policy_i": "alice", "policy_j": "bob", "outcome": 0.8},
    {"policy_i": "alice", "policy_j": "carol", "outcome": 0.6},
    {"policy_i": "bob", "policy_j": "alice", "outcome": 0.7},
    # ... more cross-play results
])

# Build meta-game from raw data
game = MetaGame.from_dataframe(df)

# Compute equilibrium
sigma = game.solve("mene")  # Max-entropy Nash equilibrium
print("Equilibrium:", dict(zip(game.policies, sigma)))
```

### Level 1: Partner Lift Analysis

```python
# Evaluate a candidate policy against a baseline
result = level1_analysis(
    metagame=game,
    baseline_policies=["alice", "bob"],
    candidate="carol",
    solver="mene"
)

print("Partner Lift per incumbent:", result["per_incumbent"])
print("Uniform average:", result["uniform_avg"])
print("Equilibrium-weighted average:", result["equilibrium_avg"])
print("Worst-case:", result["min"])
print("Best-case:", result["max"])
```

### Level 2: Ecosystem Lift Analysis

```python
# Measure ecosystem impact with re-equilibration
result = ecosystem_lift(
    metagame=game,
    baseline_policies=["alice", "bob"],
    candidate="carol",
    solver="mene",
    welfare_fn="utilitarian"  # or "nash", "egalitarian"
)

print("Ecosystem lift:", result["delta_eco"])
print("Entry mass:", result["entry_mass"])
print("Equilibrium shift:", result["equilibrium_shift"])
print("Incumbent value shifts:", result["incumbent_shifts"])
```

### Level 3: Shapley Attribution

```python
# Compute Shapley values for ecosystem attribution
def value_fn(policies):
    if len(policies) < 2:
        return 0.0
    sub_game = game.subset(policies)
    sigma = sub_game.solve("mene")
    return sub_game.welfare(sigma, "utilitarian")

shapley = shapley_value(game.policies, value_fn)
print("Shapley values:", shapley)

# Or Banzhaf values
from causal_game_analysis import banzhaf_value
banzhaf = banzhaf_value(game.policies, value_fn)
```

### Bootstrap for Uncertainty Quantification

```python
# Bootstrap resampling for confidence intervals
bootstrap = Bootstrap(df, n_samples=1000, seed=42)

# Run any analysis on bootstrap samples
def analyze(g):
    return level1_analysis(g, ["alice", "bob"], "carol")["uniform_avg"]

results = bootstrap.run(analyze, progress=True)

# Get confidence interval
lower, median, upper = Bootstrap.confidence_interval(results, alpha=0.05)
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
```

### EF1 Fairness Analysis (for Bargaining)

```python
from causal_game_analysis import ef1_frequency_matrix, aggregate_ef1_between_groups

# If your data includes EF1 indicator column
df_with_ef1 = pd.DataFrame([
    {"policy_i": "gpt4", "policy_j": "claude", "outcome": 0.8, "ef1": 1},
    {"policy_i": "gpt4", "policy_j": "llama", "outcome": 0.6, "ef1": 0},
    # ...
])

# EF1 frequency matrix
ef1_matrix, policies = ef1_frequency_matrix(df_with_ef1)

# Compare EF1 between groups
ef1_stats = aggregate_ef1_between_groups(
    df_with_ef1,
    group_a=["gpt4", "claude"],
    group_b=["llama", "mistral"]
)
print("EF1 frequency (Group A vs B):", ef1_stats["a_vs_b"])
print("EF1 frequency (within Group A):", ef1_stats["within_a"])
```

### Direct Matrix Construction

```python
# If you already have a payoff matrix
payoff_matrix = np.array([
    [0.5, 0.0, 1.0],  # rock
    [1.0, 0.5, 0.0],  # paper
    [0.0, 1.0, 0.5],  # scissors
])

game = MetaGame(
    policies=["rock", "paper", "scissors"],
    payoff_matrix=payoff_matrix
)
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `MetaGame` | Empirical meta-game representation with payoff matrix |
| `Bootstrap` | Bootstrap resampling for uncertainty quantification |

### Analysis Functions

| Function | Level | Description |
|----------|-------|-------------|
| `level1_analysis()` | 1 | Partner Lift (no re-equilibration) |
| `ecosystem_lift()` | 2 | Ecosystem Lift (with re-equilibration) |
| `shapley_value()` | 3 | Shapley attribution values |
| `banzhaf_value()` | 3 | Banzhaf attribution values |

### Fairness Metrics

| Function | Description |
|----------|-------------|
| `ef1_frequency()` | EF1 frequency per policy pair |
| `ef1_frequency_matrix()` | EF1 frequency as matrix |
| `aggregate_ef1_between_groups()` | Compare EF1 between policy groups |

### Solvers

| Solver | Description |
|--------|-------------|
| `"mene"` | Maximum Entropy Nash Equilibrium (MILP-based) |
| `"uniform"` | Uniform distribution (baseline) |

---

## Framework Overview

### Game-Theoretic Foundations

We model multi-agent evaluation as an empirical game Ĝ = (N, (Sᵢ), (ûᵢ)) where:

- **N** — Set of player roles (typically N = {1, 2} for pairwise evaluation)
- **Sᵢ** — Strategy set for player i (the set of available policies)
- **ûᵢ** — Estimated utility function ûᵢ : ∏ⱼ∈N Sⱼ → ℝ
- **σᵢ** — Mixed strategy for player i, where σᵢ ∈ Δ(Sᵢ)
- **σ₋ᵢ** — Strategy profile of all players except i

### Strategy Restriction

Following empirical game-theoretic analysis, we consider restricted strategy sets. Let S↓X denote restriction to Xᵢ ⊆ Sᵢ. The restricted empirical game is:

```
Ĝ_{S↓X} = (N, (Xᵢ), (ûᵢ))
```

In our framework:
- **X** represents the **baseline library** of policies
- **X⁺ = X ∪ {sⱼ}** represents the library after adding candidate sⱼ

### Equilibrium and Regret

A mixed strategy profile σ* is a **Nash equilibrium** if:

```
σ*ᵢ ∈ brᵢ(σ*₋ᵢ)   ∀i ∈ N
```

where brᵢ(σ₋ᵢ) = argmax over σ'ᵢ ∈ Δ(Sᵢ) of uᵢ(σ'ᵢ, σ₋ᵢ).

Player i's **regret** in profile σ:

```
ρᴳᵢ(σ) = max_{s'ᵢ ∈ Sᵢ} uᵢ(s'ᵢ, σ₋ᵢ) − uᵢ(σᵢ, σ₋ᵢ)
```

The **minimum regret constrained profile** (MRCP) for a restricted game:

```
MRCP(G_{S↓X}) = argmin_{σ ∈ Δ(X)} Σᵢ∈N ρᴳᵢ(σ)
```

---

## Three Levels of Analysis

### Level 1: Interaction-Level (No Re-Equilibration)

Level 1 measures direct interaction effects without ecosystem adaptation. Fix a baseline library X and its equilibrium σ_X. For each incumbent strategy sᵢ ∈ X, compare outcomes against candidate sⱼ versus typical equilibrium partners.

**Baseline expected utility** for incumbent sᵢ ∈ X:

```
U_X(sᵢ) := Σ_{s ∈ X} σ_X(s) · u(sᵢ, s)
```

**Partner Lift** (strategy-specific):

```
PL₁(sᵢ; sⱼ | X) := u(sᵢ, sⱼ) − U_X(sᵢ)
```

**Aggregations:**

| Aggregation | Description |
|-------------|-------------|
| Uniform average | (1/\|X\|) Σ PL₁(sᵢ; sⱼ \| X) over all sᵢ ∈ X |
| Equilibrium-weighted | Σ σ_X(sᵢ) · PL₁(sᵢ; sⱼ \| X) over all sᵢ ∈ X |
| Worst-case | min over sᵢ ∈ X of PL₁(sᵢ; sⱼ \| X) |
| Best-case | max over sᵢ ∈ X of PL₁(sᵢ; sⱼ \| X) |

---

### Level 2: Ecosystem-Level (Re-Equilibration)

Level 2 measures ecosystem effects with strategic adaptation. Expand the strategy set to X⁺ = X ∪ {sⱼ}, compute new equilibrium σ_{X⁺}, and compare welfare.

**Welfare function** over profile σ in game G:

```
W(σ, G) = f((uᵢ(σ))_{i ∈ N})
```

Common choices: utilitarian (Σᵢ uᵢ), Nash product (Πᵢ uᵢ), egalitarian (minᵢ uᵢ).

**Ecosystem lift:**

```
Δ_eco(sⱼ | X) := W(σ_{X⁺}, G_{S↓X⁺}) − W(σ_X, G_{S↓X})
```

**Incumbent value shift** under re-equilibration:

```
Δ_inc(sᵢ; sⱼ | X) := U_{X⁺}(sᵢ) − U_X(sᵢ)
```

**Equilibrium diagnostics:**

| Metric | Description |
|--------|-------------|
| Equilibrium shift | ‖σ_{X⁺} − σ_X‖₁ (L1 norm, restricted to X) |
| Entry mass | σ_{X⁺}(sⱼ) |

---

### Level 3: Ecosystem Attribution (Shapley/Banzhaf)

Level 3 assigns credit across sub-ecosystems using cooperative game theory. Define value function:

```
v(X) := W(σ_X, G_{S↓X})
```

**Shapley value:**

```
φ(s) := (1/|S|!) Σ_{orderings ≺} [v(Pred_≺(s) ∪ {s}) − v(Pred_≺(s))]
```

**Banzhaf value:**

```
β(s) := (1/2^{|S|−1}) Σ_{X ⊆ S∖{s}} [v(X ∪ {s}) − v(X)]
```

---

## Key Distinctions

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| Strategy set | Fixed X | Expanded X⁺ |
| Equilibrium | Baseline σ_X unchanged | Recomputed σ_{X⁺} |
| Interpretation | Partner quality | Ecosystem impact |

---

## Solver Sensitivity

Equilibrium selection affects all metrics. For solver 𝒮:

```
W_𝒮(X) := W(𝒮(G_{S↓X}), G_{S↓X})
```

Solver sensitivity:

```
Sens(𝒮₁, 𝒮₂; X) := W_{𝒮₁}(X) − W_{𝒮₂}(X)
```

---

## RL Training Framework

This repository also includes a comprehensive RL training framework for the CUDA-accelerated bargaining game environment.

### Quick Start

```bash
# Install CUDA environment (requires NVIDIA GPU)
cd cuda_bargain && pip install -e .

# Train a PPO agent
python scripts/train_ppo_bargain.py --num-envs 4096 --total-timesteps 5000000

# View results
python scripts/view_results.py --list
```

### Available Algorithms

| Algorithm | Type | Command |
|-----------|------|---------|
| **PPO** | Self-play | `python scripts/train_ppo_bargain.py` |
| **NFSP** | Self-play | `python scripts/train_nfsp_bargain.py` |
| **Sampled CFR** | Equilibrium | `python scripts/train_sampled_cfr.py` |
| **PSRO** | Population | `python scripts/train_psro.py` |
| **MAPPO** | Self-play | `python scripts/train_mappo.py` |
| **FCP** | Population | `python scripts/train_fcp.py` |

### Training with Logging

```bash
# With Weights & Biases
python scripts/train_ppo_bargain.py \
    --num-envs 4096 \
    --total-timesteps 10000000 \
    --wandb \
    --wandb-project causal-bargain
```

### Hyperparameter Sweeps

```bash
# Using W&B Sweeps
wandb sweep sweeps/sweep_ppo.yaml
wandb agent <sweep_id>

# Using Python script
python scripts/hyperparameter_sweep.py --algorithm ppo --method random --num-trials 50
```

### Results Management

All training results are automatically saved with:
- Trained neural network weights
- Full hyperparameter configuration
- Training metrics history
- Final evaluation results

```bash
# View all runs
python scripts/view_results.py --list

# Compare algorithms
python scripts/view_results.py --leaderboard

# Export to CSV
python scripts/view_results.py --export results.csv
```

### Documentation

- **[RL Training Guide](rl_training/README.md)** - Full documentation for training algorithms
- **[Scripts Reference](scripts/README.md)** - Command-line script usage
- **[Sweeps Guide](sweeps/README.md)** - Hyperparameter optimization

---

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/causal_game_analysis

# Lint code
uv run ruff check .

# Type check
uv run mypy src/
```

## License

MIT
