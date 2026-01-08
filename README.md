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

### Objects and Notation

| Symbol | Description |
|--------|-------------|
| $\Pi = \{\pi_1, \ldots, \pi_n\}$ | Universe of policies |
| $B \subseteq \Pi$ | Baseline (reduced) library |
| $\pi_j \in \Pi \setminus B$ | Candidate policy (not in baseline) |
| $\pi_i \in B$ | Incumbent policy (in baseline) |
| $Y \in \mathbb{R}$ | Scalar evaluation outcome |

### Pairwise Cross-Play Outcome

For any ordered pair $(\pi, \pi')$, the true pairwise expected outcome is:

$$\mu(\pi, \pi') := \mathbb{E}\big[Y \mid \text{do}(A=\pi, B=\pi')\big]$$

where $A$ and $B$ denote the two player roles. In practice, $\mu(\pi, \pi')$ is estimated empirically from cross-play.

### Causal Pipeline

The framework defines a structural causal model over the following variables:

$$L \rightarrow M_L \rightarrow \sigma_L \rightarrow W_L$$

| Variable | Description |
|----------|-------------|
| $L$ | Available library (set-valued, $S \subseteq \Pi$) |
| $M_L$ | Empirical meta-game induced by library $L$ |
| $\sigma_L$ | Meta-solution (equilibrium mixture) from solver $\mathcal{S}$ |
| $W_L$ | Ecosystem scalar value |

**Structural equations** (given a fixed solver $\mathcal{S}$ and evaluation functional $Y_{\text{eco}}$):

$$\begin{aligned}
M_L &= g(L) \\
\sigma_L &= \mathcal{S}(M_L) \\
W_L &= Y_{\text{eco}}(\sigma_L, M_L)
\end{aligned}$$

The key causal estimand is:

$$W(S) := \mathbb{E}\big[W_L \mid \text{do}(L=S)\big] = Y_{\text{eco}}(\sigma_S, M_S)$$

---

## Three Levels of Analysis

### Level 1: Interaction-Level (No Re-Equilibration)

Level 1 measures the direct interaction effect of a candidate policy $\pi_j$ without letting the ecosystem adapt. You fix a baseline reduced library $B$ and its equilibrium mixture $\sigma_B$. Then, for each incumbent $\pi_i \in B$, you compare the outcome of the pair $(\pi_i, \pi_j)$ to $\pi_i$'s baseline expectation when facing a "typical" partner drawn from $\sigma_B$. This answers: if we drop $\pi_j$ into the existing world as a partner, does it help or hurt incumbents relative to what they normally face at equilibrium?

**Baseline equilibrium interaction value** for incumbent $\pi_i \in B$:

$$U_B(\pi_i) := \mathbb{E}_{\pi \sim \sigma_B}\big[\mu(\pi_i, \pi)\big] = \sum_{\pi \in B} \sigma_B(\pi) \, \mu(\pi_i, \pi)$$

**Partner Lift** (incumbent-specific):

$$\text{PL}_1(\pi_i; \pi_j \mid B) := \mu(\pi_i, \pi_j) - U_B(\pi_i)$$

This answers: "If incumbent $\pi_i$ faces candidate $\pi_j$ instead of a typical equilibrium partner, what is the expected change in outcome?"

**Aggregations:**

| Aggregation | Formula |
|-------------|---------|
| Uniform average | $\overline{\text{PL}}\_1^{\text{unif}}(\pi\_j \mid B) := \frac{1}{\lvert B \rvert} \sum\_{\pi\_i \in B} \text{PL}\_1(\pi\_i; \pi\_j \mid B)$ |
| Equilibrium-weighted | $\overline{\text{PL}}\_1^{\sigma}(\pi\_j \mid B) := \sum\_{\pi\_i \in B} \sigma\_B(\pi\_i) \, \text{PL}\_1(\pi\_i; \pi\_j \mid B)$ |
| Worst-case | $\text{PL}\_1^{\min}(\pi\_j \mid B) := \min\_{\pi\_i \in B} \text{PL}\_1(\pi\_i; \pi\_j \mid B)$ |
| Best-case | $\text{PL}\_1^{\max}(\pi\_j \mid B) := \max\_{\pi\_i \in B} \text{PL}\_1(\pi\_i; \pi\_j \mid B)$ |

---

### Level 2: Ecosystem-Level (Re-Equilibration)

Level 2 measures the ecosystem effect of making $\pi_j$ available to the system and allowing strategic adaptation. You intervene on the library by adding $\pi_j$ to $B$ to form $B^+ = B \cup \{\pi_j\}$, re-estimate/assemble the meta-game on $B^+$, re-solve equilibrium to get $\sigma_{B^+}$, and then compare the resulting ecosystem value $W(B^+)$ to $W(B)$. This answers: does $\pi_j$'s presence change equilibrium behavior and improve (or degrade) the equilibrium-selected outcome of the whole population?

Let $B^+ := B \cup \{\pi_j\}$. The **ecosystem lift** is:

$$\Delta_{\text{eco}}(\pi_j \mid B) := W(B^+) - W(B) = \mathbb{E}\big[W_L \mid \text{do}(L=B^+)\big] - \mathbb{E}\big[W_L \mid \text{do}(L=B)\big]$$

**Incumbent value shift** under re-equilibration:

$$\Delta_{\text{inc}}(\pi_i; \pi_j \mid B) := V_{B^+}(\pi_i) - V_B(\pi_i)$$

where $V_S(\pi_i) := \sum_{\pi \in S} \sigma_S(\pi) \, \mu(\pi_i, \pi)$.

**Equilibrium diagnostics:**

| Metric | Formula |
|--------|---------|
| Equilibrium shift | $\Delta_\sigma(\pi_j \mid B) := \lVert \sigma_{B^+} - \sigma_B \rVert_1$ |
| Entry mass | $\text{EntryMass}(\pi_j \mid B) := \sigma_{B^+}(\pi_j)$ |

---

### Level 3: Ecosystem Attribution (Shapley/Banzhaf)

Level 3 assigns synergy-aware credit for ecosystem outcomes across many possible reduced ecosystems, instead of relying on a single baseline $B$. You treat the ecosystem value $W(S)$ as a cooperative-game value function $v(S)$ over sub-libraries $S \subseteq \Pi$, and compute Shapley or Banzhaf values for each policy. This answers: on average across many "possible worlds" (different sub-libraries), how much marginal ecosystem value does each policy contribute, accounting for complementarities and redundancy among policies?

Define the value function:

$$v(S) := W(S) = \mathbb{E}\big[W_L \mid \text{do}(L=S)\big]$$

**Banzhaf value:**

$$\beta(\pi) := \mathbb{E}_{S \subseteq \Pi \setminus \{\pi\}}\big[v(S \cup \{\pi\}) - v(S)\big]$$

**Shapley value:**

$$\phi(\pi) := \mathbb{E}_{\text{uniform } \prec}\big[v(\text{Pred}_\prec(\pi) \cup \{\pi\}) - v(\text{Pred}_\prec(\pi))\big]$$

where $\text{Pred}_\prec(\pi)$ denotes the set of policies preceding $\pi$ in ordering $\prec$.

---

## Key Distinctions

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| Intervention | $\text{do}(L=B)$ fixed | $\text{do}(L=B^+)$ vs $\text{do}(L=B)$ |
| Equilibrium | Baseline $\sigma_B$ unchanged | Recomputed $\sigma_{B^+}$ |
| Interpretation | Partner quality | Ecosystem impact |

---

## Solver Sensitivity

All interventional values depend on the solver choice. Make this explicit:

$$W_{\mathcal{S}}(S) := Y_{\text{eco}}\big(\mathcal{S}(M_S), M_S\big)$$

Solver sensitivity:

$$\text{Sens}(\mathcal{S}_1, \mathcal{S}_2; S) := W_{\mathcal{S}_1}(S) - W_{\mathcal{S}_2}(S)$$

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
