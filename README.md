# Causal EGTA: Counterfactual Analysis of Multi-Agent Ecosystems

### Motivation

Multi-agent ecosystems are increasingly evaluated through empirical game-theoretic analysis (EGTA), where agents interact in a metagame and equilibrium analysis summarizes the ecosystem's strategic behavior. However, standard equilibrium analysis only reveals what happens at the equilibrium — not which agents causally drive it. Nash equilibrium is known to violate independence of irrelevant alternatives (IIA): removing a strategy outside the equilibrium support can change the equilibrium entirely. This means that adding or removing a strategy — even one outside the equilibrium support — can shift which equilibrium is selected and alter its welfare, fairness, or cooperation properties. A strategy that does not rank highest on any individual metric may nonetheless drive higher cooperation or welfare at equilibrium through its strategic interactions with other agents. Standard EGTA has no tools to detect these effects. Measuring cooperation and social welfare in multi-agent systems has been a longstanding goal, with various metrics proposed for quantifying agent behavior (e.g., cooperation rates, social welfare, fairness indices). We contribute an alternate framing: rather than measuring cooperation as a property of individual agents, we measure each agent's *causal contribution* to cooperation and welfare at equilibrium — asking not "how cooperative is this agent?" but "how much does this agent's presence cause the equilibrium to be more cooperative?"

As multi-agent competitions scale — from Kaggle's [Game Arena](https://www.kaggle.com/game-arena) where LLMs compete in Werewolf, Poker, and Chess, to PSRO-trained leagues like AlphaStar — the need for causal evaluation grows: practitioners need to know not just which agents win, but which agents' presence shapes the ecosystem's equilibrium behavior.

### Framework

We introduce a counterfactual credit assignment framework for empirical metagames that measures each agent's causal contribution to equilibrium outcomes via leave-one-out analysis and Harsanyi interaction dividends, with paired bootstrap inference for statistical significance. We address the challenge of multiple equilibria — which renders LOO effects solver-dependent — through CURB welfare intervals that bound the range of equilibrium outcomes across stable basins, separating strategic uncertainty from statistical uncertainty. We extend the meta-game evaluation framework of Zun Li et al. and support multiple solution concepts (MENE, maxent CCE, max affinity entropy).

### Key Findings

We evaluate across multiple domains — a multi-agent bargaining game with RL and LLM agents, and iterated matrix games (Prisoner's Dilemma, Hawk-Dove) — and find that individual performance diverges from ecosystem contribution, that solution concept choice qualitatively changes which agents are identified as important, and that agents outside the equilibrium support can have the largest causal effects — empirically confirming that IIA violations in metagames are not merely theoretical but have substantial consequences for multi-agent evaluation.

### Connection to Agent Importance in MARL

Recent work on explainable multi-agent importance (EMAI, Xu et al. 2024) measures agent importance through counterfactual reasoning: randomize an agent's actions and measure the reward change. Our framework applies a similar counterfactual logic at the metagame level — but instead of randomizing actions uniformly, we evaluate outcomes at equilibrium under a chosen solution concept. This grounds the counterfactual in strategic reasoning: the importance of an agent is measured by how its presence or absence affects the equilibrium behavior of the ecosystem, not just average performance.

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

---

## Framework Overview

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

### Notation Reference

| Symbol | Description |
|--------|-------------|
| G = (N, (Sᵢ), (uᵢ)) | Normal-form game with players N, strategy sets Sᵢ, utilities uᵢ |
| Ĝ | Empirical game (payoffs estimated via simulation/data) |
| S | Full strategy universe (all available strategies/policies) |
| X ⊆ S | Baseline restricted strategy set (library) |
| S↓X | Restriction operator (per-player: Xᵢ ⊆ Sᵢ) |
| Ĝ_{S↓X} | Restricted empirical game induced by X |
| σ ∈ Δ(X) | Mixed strategy profile over X |
| σ_X | Equilibrium mixture computed on Ĝ_{S↓X} |
| S | Meta-strategy solver (MSS): Ĝ_{S↓X} ↦ σ_X |
| BRᵢ(σ₋ᵢ) | Best-response correspondence for player i |
| ρᵢᴳ(σ) | Regret of player i at profile σ (in game G) |
| Φ | A metric functional, Φ(Ĝ_{S↓X}, σ_X) (Regret, Utility, NW, UW, EF1, etc.) |

---

## Counterfactual Analysis

### LOO (Leave-One-Out) Effect

The causal effect of agent sᵢ on equilibrium welfare:

```
ΔW(sᵢ | Ĝ_{S↓X}) := W_S(Ĝ_{S↓X}) − W_S(Ĝ_{S↓X\{sᵢ}})
```

Remove agent sᵢ from the strategy set, re-solve for equilibrium on the restricted game, and measure the welfare change. Positive means the agent's presence helps; negative means it hurts.

### Harsanyi Interaction Dividends

Pairwise interaction effects between agents sᵢ and sⱼ:

```
Δ²W(sᵢ, sⱼ | Ĝ_{S↓X}) := W_S(Ĝ_{S↓X}) − W_S(Ĝ_{S↓X\{sᵢ}}) − W_S(Ĝ_{S↓X\{sⱼ}}) + W_S(Ĝ_{S↓X\{sᵢ,sⱼ}})
```

- **Δ² > 0**: complementary — together they contribute more than the sum of parts
- **Δ² < 0**: substitutes — individually helpful but redundant together
- **Δ² ≈ 0**: independent — their effects don't interact

### Synergy Index (Emergence Detection)

The Synergy Index (cf. MACIE, Weinberg 2025) measures whether the multi-agent ecosystem exhibits emergence — collective performance at equilibrium exceeding the sum of individual contributions:

```
SI = (W(X) − Σᵢ W({sᵢ})) / max(W(X), Σᵢ W({sᵢ}))
```

Where W(X) is welfare at the full-game equilibrium and W({sᵢ}) is agent sᵢ's self-play payoff (M[i,i]). In a metagame, self-play represents the outcome when only that agent exists in the ecosystem.

- **SI > 0**: positive emergence — the ecosystem at equilibrium produces more welfare than the sum of individual agents in isolation. Strategic diversity creates value.
- **SI < 0**: negative emergence — agents interfere with each other. The ecosystem at equilibrium is worse than agents operating independently.
- **SI ≈ 0**: no emergence — collective welfare is approximately the sum of parts.

The Synergy Index complements LOO effects (individual importance) and Harsanyi dividends (pairwise interactions) by providing a single system-level measure of whether strategic interaction creates or destroys value.

### On Direct/Indirect Effect Decomposition

In causal mediation analysis (cf. Weighted Möbius Score, Jiang & Steinert-Threlkeld 2023), total effects can be decomposed into direct effects (the removed agent's payoffs no longer contribute) and indirect/mediated effects (the removal changes other agents' equilibrium behavior). However, in a strategic context this decomposition is not cleanly defined: there is no principled way to "hold the equilibrium fixed" while removing a strategy, since the equilibrium is itself a function of the strategy set. For out-of-support agents the entire LOO effect is indirect (their payoffs never entered σᵀMσ), while for in-support agents the two effects are entangled. We therefore report total LOO effects and use structural analysis (BR graph changes, CURB set stability) to explain the mechanism behind each effect, rather than attempting a numerical direct/indirect split.

### Paired Bootstrap Inference

One bootstrap resample of the raw data induces ALL subgames:

```
resample raw data → build full N×N matrices → slice rows/cols for each subgame
```

Because the full game and every restricted game come from the SAME resample, comparisons are PAIRED across bootstrap index b:

```
d_b = W(full, b) − W(LOO_i, b)
```

95% CIs on the paired differences determine statistical significance.

---

## Solution Concepts

| Solver | Description |
|--------|-------------|
| MENE | Maximum Entropy Nash Equilibrium — unique NE maximizing Shannon entropy |
| Maxent CCE | Maximum Entropy Coarse Correlated Equilibrium — unique CCE maximizing entropy over the CE polytope (via Polarix) |
| Max Affinity Entropy | Nash equilibrium with affinity entropy regularization (via Polarix) |

### Solver Sensitivity

Equilibrium selection affects all metrics:

```
Sens(S₁, S₂; X) := W_{S₁}(X) − W_{S₂}(X)
```

Different solution concepts can identify different agents as causally important. MENE may concentrate on one set of agents while maxent CCE favors another — this divergence is itself an important finding.

---

## CURB Sets and Welfare Intervals

### CURB Sets

**Closed Under Rational Behavior (Basu & Weibull, 1991).** A non-empty subset C ⊆ S is CURB if:

```
CBR(C) ⊆ C
```

Every Nash equilibrium is guaranteed to live in some CURB set. The full CURB lattice characterizes the space of stable equilibria.

### Welfare Intervals via CURB

For each CURB set, solve for equilibrium and compute welfare. Report the interval across all CURB sets:

```
[W_min, W_max] = [min_{C ∈ CURB} W(σ*_C), max_{C ∈ CURB} W(σ*_C)]
```

This separates:
- **Strategic uncertainty**: the interval width (which equilibrium basin)
- **Statistical uncertainty**: bootstrap CIs on the interval bounds (payoff estimation noise)

### CURB-Conditional LOO

LOO effects computed within each CURB set:

```
ΔW(sᵢ | C) := W(σ*_C) − W(σ*_{C\{sᵢ}})
```

An agent may be essential in one CURB set but irrelevant in another.

### CURB-Level Interaction Effects

Instead of pairwise Harsanyi dividends between individual agents, we can compute interaction effects between a CURB set and the rest of the game:

```
Δ²W(C, X\C) = W(X) − W(C) − W(X\C) + W(∅)
```

Where W(C) is welfare at equilibrium within the CURB set, and W(X\C) is welfare at equilibrium of everything outside it. This measures whether the CURB set and the non-CURB strategies create value together or independently:

- **Positive**: the CURB set and outside strategies are complementary — they create value together that neither group achieves alone
- **Negative**: they interfere — the CURB set is better off without the outsiders, or vice versa
- **Zero**: they are independent ecosystems

This also addresses the problem of wide individual LOO CIs for interchangeable agents (e.g., the RL trio {ppo, psro, mappo}). Individual LOO effects have wide CIs because removing one just shifts weight to the others, but a group-level LOO — removing the entire CURB set — produces a clean, significant effect. CURB sets provide a principled, game-theoretically motivated grouping for this aggregation.

### Finding CURB Sets via LP

For each candidate strategy πᵢ, check if it's a best response to some mixture over C:

```
Find σⱼ ≥ 0 for j ∈ C,  Σⱼ σⱼ = 1
subject to:  Σⱼ σⱼ · u(πᵢ, πⱼ) ≥ Σⱼ σⱼ · u(πₖ, πⱼ)   ∀ πₖ ∈ S
```

If feasible for all πᵢ ∈ C and infeasible for all πᵢ ∉ C, then C is CURB.

---

## Welfare Metrics

### Bargaining Domain

| Metric | Description |
|--------|-------------|
| UW | Utilitarian Welfare: u₁ + u₂ |
| NW | Nash Welfare: (u₁ · u₂)^(1/2) |
| NW+ | Nash Welfare on advantages: (u₁⁺ · u₂⁺)^(1/2) where u⁺ = max{0, u − BATNA} |
| EF1 | Envy-Free up to One Item frequency |
| Regret | u(σ*) − u(π, σ*₋ᵢ) — deviation incentive at equilibrium |

### Iterated Matrix Game Domains (PD, Hawk-Dove)

| Metric | Description |
|--------|-------------|
| Payoff | Average per-round payoff (evaluated at equilibrium) |
| Cooperation | Cooperation rate (evaluated at equilibrium) |
| NW | Geometric mean of both players' payoffs per matchup |

---

## Evaluation Domains

### 1. Multi-Agent Bargaining (from Zun Li et al.)

RL agents (PPO, PSRO, MAPPO) and LLM agents (OpenAI o1 variants) negotiate over item allocations. The empirical metagame has 11+ strategies with rich welfare/fairness structure.

### 2. Iterated Prisoner's Dilemma

Strategies from the [Axelrod Python library](https://github.com/Axelrod-project/Axelrod) play repeated PD with noise. Noise differentiates robust vs fragile cooperators and creates non-trivial equilibrium structure.

### 3. Iterated Hawk-Dove

Same Axelrod strategies under Hawk-Dove payoffs (anti-coordination). Different game structure reveals different causal patterns — e.g., Cooperator helps welfare in Hawk-Dove but hurts in PD.

---

## Game-Theoretic Foundations

### Equilibrium and Regret

A mixed strategy profile σ* is a **Nash equilibrium** if:

```
σ*ᵢ ∈ brᵢ(σ*₋ᵢ)   ∀i ∈ N
```

Player i's **regret** in profile σ:

```
ρᴳᵢ(σ) = max_{s'ᵢ ∈ Sᵢ} uᵢ(s'ᵢ, σ₋ᵢ) − uᵢ(σᵢ, σ₋ᵢ)
```

### MENE

When multiple Nash equilibria exist, select the one maximizing Shannon entropy:

```
σ* = argmax_{σ ∈ NE(G)} [−σ · ln(σ)]
```

---

## Key References

- Basu & Weibull (1991). "Strategy Subsets Closed Under Rational Behavior." *Economics Letters* 36(2).
- Kline & Tamer (2024). "Counterfactual Analysis in Empirical Games." *arXiv:2410.12731*.
- Wellman (2024). "Empirical Game-Theoretic Analysis: A Survey." *JAIR*.
- Xu et al. (2024). "EMAI: Explainable Multi-Agent Importance." *arXiv:2412.15619*.
- Zun Li et al. "LLM Meta-Game." (base evaluation framework)
- Hurkens (1995). "Learning by Forgetful Players." *GEB* 11.
- Young (1993). "The Evolution of Conventions." *Econometrica* 61.

---

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/iterative_game_analysis

# Lint code
uv run ruff check .
```

## License

MIT
