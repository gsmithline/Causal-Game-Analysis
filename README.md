# Iterative Meta-Game Analysis

A framework for empirical game-theoretic analysis (EGTA) of multi-agent systems. This framework provides rigorous evaluation of policies in meta-games across three levels of analysis, with bootstrap uncertainty quantification.

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
from iterative_game_analysis import (
    MetaGame,
    Bootstrap,
    level1_analysis,
    ecosystem_lift,
    shapley_value,
)

# Create or load bargaining data
# Format: DataFrame with columns for both players' payoffs, BATNAs, and EF1
df = pd.DataFrame([
    {"policy_i": "alice", "policy_j": "bob",
     "payoff_i": 80, "payoff_j": 75,
     "batna_i": 50, "batna_j": 45, "ef1": 1},
    {"policy_i": "alice", "policy_j": "carol",
     "payoff_i": 60, "payoff_j": 70,
     "batna_i": 50, "batna_j": 40, "ef1": 1},
    {"policy_i": "bob", "policy_j": "alice",
     "payoff_i": 75, "payoff_j": 80,
     "batna_i": 45, "batna_j": 50, "ef1": 1},
    # ... more bargaining instances
])

# Build meta-game from raw data (uses payoff_i for payoff matrix)
game = MetaGame.from_dataframe(df, outcome_col="payoff_i")

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
from iterative_game_analysis import banzhaf_value
banzhaf = banzhaf_value(game.policies, value_fn)
```

### Bootstrap for Full Analysis

```python
# Bargaining data format: one row per negotiation instance
df = pd.DataFrame([
    {"policy_i": "gpt4", "policy_j": "claude",
     "payoff_i": 85, "payoff_j": 90,
     "batna_i": 50, "batna_j": 55, "ef1": 1},
    {"policy_i": "gpt4", "policy_j": "llama",
     "payoff_i": 70, "payoff_j": 65,
     "batna_i": 50, "batna_j": 45, "ef1": 0},
    # ... more bargaining instances
])

# Bootstrap with full L1/L2/L3 analysis
bootstrap = Bootstrap(
    df, n_samples=1000, seed=42,
    payoff_i_col="payoff_i", payoff_j_col="payoff_j",
    batna_i_col="batna_i", batna_j_col="batna_j",
    ef1_col="ef1",
)

# Run complete analysis on each bootstrap sample
results = bootstrap.run_full_analysis(
    include_l3=True,
    l3_method="both",  # Shapley and Banzhaf
    progress=True,
)

# Each result contains: l1, l2, l3, matrices, full_game
# Access per-agent leave-one-out metrics
for agent in bootstrap.policies:
    l1 = results[0]["l1"][agent]
    l2 = results[0]["l2"][agent]
    print(f"{agent}: L1 lift={l1['uniform_avg']:.3f}, L2 delta_eco={l2['delta_eco']['uw']:.3f}")

# Welfare metrics (UW, NW, NW+) computed at equilibrium
print("Full game welfare:", results[0]["full_game"]["welfare"])
print("Full game EF1:", results[0]["full_game"]["ef1"])

# Get confidence intervals across bootstrap samples
l1_lifts = [r["l1"]["gpt4"]["uniform_avg"] for r in results]
lower, median, upper = Bootstrap.confidence_interval(l1_lifts, alpha=0.05)
print(f"95% CI for GPT4 L1 lift: [{lower:.3f}, {upper:.3f}]")
```

### EF1 Fairness Analysis (for Bargaining)

```python
from iterative_game_analysis import ef1_frequency_matrix, aggregate_ef1_between_groups

# EF1 frequency matrix from bargaining data
ef1_matrix, policies = ef1_frequency_matrix(df)

# Compare EF1 between policy groups (e.g., LLM providers)
ef1_stats = aggregate_ef1_between_groups(
    df,
    group_a=["gpt4", "claude"],
    group_b=["llama", "mistral"]
)
print("EF1 frequency (Group A vs B):", ef1_stats["a_vs_b"])
print("EF1 frequency (within Group A):", ef1_stats["within_a"])

# EF1 is also computed at equilibrium in run_full_analysis()
results = bootstrap.run_full_analysis(include_l3=False)
print("EF1 at equilibrium:", results[0]["full_game"]["ef1"])
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
| `Bootstrap` | Bootstrap resampling with full L1/L2/L3 analysis |

### Bootstrap Methods

| Method | Description |
|--------|-------------|
| `run_full_analysis()` | Run complete L1/L2/L3 analysis on each bootstrap sample |
| `run()` | Run custom analysis function on each bootstrap sample |
| `sample()` | Generate one bootstrap sample (stratified by policy pair) |
| `confidence_interval()` | Compute percentile CI from bootstrap distribution |

### Analysis Functions

| Function | Level | Description |
|----------|-------|-------------|
| `level1_analysis()` | 1 | Partner Lift (no re-equilibration) |
| `ecosystem_lift()` | 2 | Ecosystem Lift (with re-equilibration) |
| `shapley_value()` | 3 | Shapley attribution values |
| `banzhaf_value()` | 3 | Banzhaf attribution values |

### Welfare Metrics

| Metric | Description |
|--------|-------------|
| UW | Utilitarian Welfare (sum of payoffs) |
| NW | Nash Welfare (geometric mean of payoffs) |
| NW+ | Nash Welfare on advantages (payoff - BATNA) |

---

## Metrics Reference (from LLM Meta-Game Paper)

### Welfare Functions

**Utilitarian Welfare (UW)**

The sum of players' payoffs:

```
UW := u₁ + u₂
```

Maximizing UW finds the most efficient outcome in terms of total value created.

**Nash Welfare (NW)**

The geometric mean of players' payoffs:

```
NW := (u₁ · u₂)^(1/2)
```

Nash welfare balances efficiency and fairness by giving weight to both players' outcomes. It is maximized when gains are distributed more equally.

**Nash Welfare on Advantages (NW+)**

Nash welfare computed on *advantages* (surplus above BATNA):

```
NW+ := (u₁⁺ · u₂⁺)^(1/2)

where u_i⁺ = max{0, uᵢ − bᵢ}
```

Here `bᵢ` is player i's BATNA (Best Alternative to Negotiated Agreement). NW+ measures the geometric mean of gains *above* each player's outside option, accounting for the possibility that a player may receive less than their BATNA.

### Fairness Metric

**EF1 (Envy-Free up to One Item)**

An allocation is **envy-free** if each player values their own bundle at least as much as the other's. Since envy-free allocations may not exist, we use EF1 as a relaxation:

An allocation is **EF1** if, for each player i, there exists an item in the other player's bundle which, if removed, would eliminate envy:

```
vᵢ · a₋ᵢ − vᵢ · aᵢ ≤ max_{k: a₋ᵢ,ₖ > 0} vᵢ,ₖ    for i ∈ {1, 2}
```

The **EF1 frequency** is the fraction of bargaining instances ending in ACCEPT that produce EF1 allocations.

### Individual Effectiveness

**Regret**

For a symmetric two-player game, the regret of strategy π at Nash equilibrium σ* is:

```
Regret(π) := u(σ*) − u(π, σ*₋ᵢ)
```

This measures how much worse strategy π performs compared to the equilibrium value when facing equilibrium opponents. Lower regret indicates a strategy closer to best-responding.

### Equilibrium Selection

**Maximum Entropy Nash Equilibrium (MENE)**

When multiple Nash equilibria exist, we select the one maximizing Shannon entropy:

```
σ* = argmax_{σ ∈ NE(G)} [−σ · ln(σ)]
```

This provides a unique, well-defined equilibrium that spreads probability mass across strategies when indifferent, avoiding arbitrary selection among equivalent equilibria.

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

### Notation Reference

| Symbol | Description |
|--------|-------------|
| G = (N, (Sᵢ), (uᵢ)) | Normal-form game with players N, strategy sets Sᵢ, utilities uᵢ |
| Ĝ | Empirical game (payoffs estimated via simulation/data) |
| S | Full strategy universe (all available strategies/policies) |
| X ⊆ S | Baseline restricted strategy set (library) |
| sⱼ ∈ S ∖ X | Candidate strategy not yet in X |
| sᵢ ∈ X | Incumbent strategy in X |
| S↓X | Restriction operator (per-player: Xᵢ ⊆ Sᵢ) |
| Ĝ_{S↓X} | Restricted empirical game induced by X |
| σ ∈ Δ(X) | Mixed strategy profile over X |
| σ_X | Equilibrium mixture computed on Ĝ_{S↓X} |
| S | Meta-strategy solver (MSS): Ĝ_{S↓X} ↦ σ_X |
| BRᵢ(σ₋ᵢ) | Best-response correspondence for player i |
| ρᵢᴳ(σ) | Regret of player i at profile σ (in game G) |
| Φ | A metric functional, Φ(Ĝ_{S↓X}, σ_X) (Regret, Utility, NW, UW, EF1, etc.) |

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

### EGTA Pipeline: Restrict → Solve → Evaluate

1. **Restricted-game construction:**
   ```
   Ĝ_{S↓X} := g(X)
   ```
   where g(·) denotes assembling/estimating the restricted payoff table on X.

2. **Solve for an equilibrium/solution mixture:**
   ```
   σ_X := S(Ĝ_{S↓X})
   ```

3. **Restricted game value functional:**
   ```
   W(X) := Φ(Ĝ_{S↓X}, σ_X)
   ```
   an outcome induced by the solution concept and the game (e.g., welfare at equilibrium, fairness at equilibrium, exploitability).

4. **Make solver dependence explicit:**
   ```
   W_S(X) := Φ(Ĝ_{S↓X}, S(Ĝ_{S↓X}))
   ```
   Define a solution concept S (MENE, affinity entropy, etc.).

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

### Level 1: Direct Interaction Effect (No Re-Equilibration)

Level 1 measures direct interaction effects without ecosystem adaptation. Fix a baseline library X and its equilibrium σ_X. For each incumbent strategy sᵢ ∈ X, compare outcomes against candidate sⱼ versus typical equilibrium partners.

**Pairwise outcome** for strategy pair (sᵢ, s₋ᵢ):

```
m(sᵢ, s₋ᵢ) := E[Z | (sᵢ, s₋ᵢ)]
```

This is an empirical average for some specified metric Z (e.g., payoff, welfare, fairness).

**Baseline equilibrium interaction value** for incumbent sᵢ ∈ X:

```
V_X(sᵢ) := m(sᵢ, σ_{X₋ᵢ}) := E_{s₋ᵢ ~ σ_{X₋ᵢ}}[m(sᵢ, s₋ᵢ)]
```

**Partner Lift** (direct, no adaptation):

```
PL₁(sᵢ; sⱼ | X) := m(sᵢ, sⱼ) − V_X(sᵢ)
```

Interpretation: If incumbent sᵢ faces sⱼ instead of the equilibrium from Ĝ_{S↓X}, how much does the expected outcome change?

**Aggregations:**

| Aggregation | Description |
|-------------|-------------|
| Uniform average | (1/\|X\|) Σ PL₁(sᵢ; sⱼ \| X) over all sᵢ ∈ X |
| Equilibrium-weighted | Σ σ_X(sᵢ) · PL₁(sᵢ; sⱼ \| X) over all sᵢ ∈ X |
| Worst-case | min over sᵢ ∈ X of PL₁(sᵢ; sⱼ \| X) |
| Best-case | max over sᵢ ∈ X of PL₁(sᵢ; sⱼ \| X) |

---

### Level 2: Restricted Game Change + Re-Equilibration

Level 2 measures ecosystem effects with strategic adaptation by adding sⱼ to the restricted game and re-solving.

**Expanded restricted game:**

```
X⁺ := X ∪ {sⱼ}
```

Re-estimate/assemble Ĝ_{S↓X⁺}, re-solve for σ_{X⁺}, then evaluate W(X⁺).

**Restricted game value functional:**

```
W(X) := Φ(Ĝ_{S↓X}, σ_X)
```

where Φ is a metric functional (welfare, fairness, etc.) evaluated at equilibrium.

**Impact of adding sⱼ to X (ecosystem lift):**

```
ΔW(sⱼ | X) := W(X⁺) − W(X)
```

**Incumbent value shift** under re-equilibration:

```
Δ_inc(sᵢ; sⱼ | X) := V_{X⁺}(sᵢ) − V_X(sᵢ)
```

**Equilibrium diagnostics:**

| Metric | Formula |
|--------|---------|
| Equilibrium shift | Δσ(sⱼ \| X) := ‖σ_{X⁺} − σ_X‖₁ |
| Entry mass | EntryMass(sⱼ \| X) := σ_{X⁺}(sⱼ) |

---

### Level 3: Synergy-Aware Credit Attribution

Level 3 assigns credit across sub-ecosystems using cooperative game theory, averaging over many possible restricted games.

**Value function** over sub-libraries X ⊆ S:

```
v(X) := W(X) = Φ(Ĝ_{S↓X}, σ_X), where σ_X = S(Ĝ_{S↓X})
```

**Banzhaf value:**

```
β(s) := E_{X ⊆ S∖{s}}[v(X ∪ {s}) − v(X)]
     = (1/2^{|S|−1}) Σ_{X ⊆ S∖{s}} [v(X ∪ {s}) − v(X)]
```

**Shapley value:**

```
φ(s) := E_≺[v(Pred_≺(s) ∪ {s}) − v(Pred_≺(s))]
     = (1/|S|!) Σ_≺ [v(Pred_≺(s) ∪ {s}) − v(Pred_≺(s))]
```

where ≺ is a uniform random ordering of S, and Pred_≺(s) denotes the set of strategies preceding s under ≺.

---

## Key Distinctions

| Aspect | Level 1 | Level 2 | Level 3 |
|--------|---------|---------|---------|
| Strategy set | Fixed X | Expanded X⁺ | All subsets of S |
| Equilibrium | σ_X held fixed | Recomputed σ_{X⁺} | Recomputed per subset |
| Interpretation | Partner quality (direct effect) | Restricted game impact (with adaptation) | Synergy-aware attribution |

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

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/iterative_game_analysis

# Lint code
uv run ruff check .

# Type check
uv run mypy src/
```

## License

MIT
