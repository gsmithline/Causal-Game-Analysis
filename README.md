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

We model multi-agent evaluation as an empirical game $\hat{G} = (N, (S_i), (\hat{u}_i))$ where:

| Symbol | Description |
|--------|-------------|
| $N$ | Set of player roles (typically $N = \{1, 2\}$ for pairwise evaluation) |
| $S_i$ | Strategy set for player $i$ (the set of available policies) |
| $\hat{u}_i : \prod_{j \in N} S_j \to \mathbb{R}$ | Estimated utility function for player $i$ |
| $\sigma_i \in \Delta(S_i)$ | Mixed strategy for player $i$ (distribution over policies) |
| $\sigma_{-i}$ | Strategy profile of all players except $i$ |

### Strategy Restriction

Following empirical game-theoretic analysis, we consider restricted strategy sets. Let $S \downarrow X$ denote restriction to $X_i \subseteq S_i$. The restricted empirical game is:

$$\hat{G}_{S \downarrow X} = (N, (X_i), (\hat{u}_i))$$

In our framework:
- $X$ represents the **baseline library** of policies
- $X^+ = X \cup \{s_j\}$ represents the library after adding candidate $s_j$

### Equilibrium and Regret

A mixed strategy profile $\sigma^*$ is a **Nash equilibrium** if:

$$\sigma^*_i \in \text{br}_i(\sigma^*_{-i}) \quad \forall i \in N$$

where $\text{br}_i(\sigma_{-i}) = \arg\max_{\sigma'_i \in \Delta(S_i)} u_i(\sigma'_i, \sigma_{-i})$.

Player $i$'s **regret** in profile $\sigma$:

$$\rho^G_i(\sigma) = \max_{s'_i \in S_i} u_i(s'_i, \sigma_{-i}) - u_i(\sigma_i, \sigma_{-i})$$

The **minimum regret constrained profile** (MRCP) for a restricted game:

$$\text{MRCP}(G_{S \downarrow X}) = \arg\min_{\sigma \in \Delta(X)} \sum_{i \in N} \rho^G_i(\sigma)$$

---

## Three Levels of Analysis

### Level 1: Interaction-Level (No Re-Equilibration)

Level 1 measures direct interaction effects without ecosystem adaptation. Fix a baseline library $X$ and its equilibrium $\sigma_X$. For each incumbent strategy $s_i \in X$, compare outcomes against candidate $s_j$ versus typical equilibrium partners.

**Baseline expected utility** for incumbent $s_i \in X$:

$$U_X(s_i) := \sum_{s \in X} \sigma_X(s) \cdot u(s_i, s)$$

**Partner Lift** (strategy-specific):

$$\text{PL}_1(s_i; s_j \mid X) := u(s_i, s_j) - U_X(s_i)$$

**Aggregations:**

| Aggregation | Formula |
|-------------|---------|
| Uniform average | $\overline{\text{PL}}_1^{\text{unif}}(s_j \mid X) := \frac{1}{|X|} \sum_{s_i \in X} \text{PL}_1(s_i; s_j \mid X)$ |
| Equilibrium-weighted | $\overline{\text{PL}}_1^{\sigma}(s_j \mid X) := \sum_{s_i \in X} \sigma_X(s_i) \cdot \text{PL}_1(s_i; s_j \mid X)$ |
| Worst-case | $\text{PL}_1^{\min}(s_j \mid X) := \min_{s_i \in X} \text{PL}_1(s_i; s_j \mid X)$ |
| Best-case | $\text{PL}_1^{\max}(s_j \mid X) := \max_{s_i \in X} \text{PL}_1(s_i; s_j \mid X)$ |

---

### Level 2: Ecosystem-Level (Re-Equilibration)

Level 2 measures ecosystem effects with strategic adaptation. Expand the strategy set to $X^+ = X \cup \{s_j\}$, compute new equilibrium $\sigma_{X^+}$, and compare welfare.

**Welfare function** over profile $\sigma$ in game $G$:

$$W(\sigma, G) = f\big((u_i(\sigma))_{i \in N}\big)$$

Common choices: utilitarian ($\sum_i u_i$), Nash product ($\prod_i u_i$), egalitarian ($\min_i u_i$).

**Ecosystem lift:**

$$\Delta_{\text{eco}}(s_j \mid X) := W(\sigma_{X^+}, G_{S \downarrow X^+}) - W(\sigma_X, G_{S \downarrow X})$$

**Incumbent value shift** under re-equilibration:

$$\Delta_{\text{inc}}(s_i; s_j \mid X) := U_{X^+}(s_i) - U_X(s_i)$$

**Equilibrium diagnostics:**

| Metric | Formula |
|--------|---------|
| Equilibrium shift | $\|\sigma_{X^+} - \sigma_X\|_1$ (restricted to $X$) |
| Entry mass | $\sigma_{X^+}(s_j)$ |

---

### Level 3: Ecosystem Attribution (Shapley/Banzhaf)

Level 3 assigns credit across sub-ecosystems using cooperative game theory. Define value function $v(X) := W(\sigma_X, G_{S \downarrow X})$.

**Shapley value:**

$$\phi(s) := \frac{1}{|S|!} \sum_{\text{orderings } \prec} \big[v(\text{Pred}_\prec(s) \cup \{s\}) - v(\text{Pred}_\prec(s))\big]$$

**Banzhaf value:**

$$\beta(s) := \frac{1}{2^{|S|-1}} \sum_{X \subseteq S \setminus \{s\}} \big[v(X \cup \{s\}) - v(X)\big]$$

---

## Key Distinctions

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| Strategy set | Fixed $X$ | Expanded $X^+$ |
| Equilibrium | Baseline $\sigma_X$ unchanged | Recomputed $\sigma_{X^+}$ |
| Interpretation | Partner quality | Ecosystem impact |

---

## Solver Sensitivity

Equilibrium selection affects all metrics. For solver $\mathcal{S}$:

$$W_{\mathcal{S}}(X) := W\big(\mathcal{S}(G_{S \downarrow X}), G_{S \downarrow X}\big)$$

Solver sensitivity:

$$\text{Sens}(\mathcal{S}_1, \mathcal{S}_2; X) := W_{\mathcal{S}_1}(X) - W_{\mathcal{S}_2}(X)$$

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
