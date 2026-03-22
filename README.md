# Causal EGTA: Counterfactual Analysis in Empirical Metagames

## Motivation

Metagame analysis, a specific approach within empirical game-theoretic analysis (EGTA), evaluates multi-agent systems by constructing a metagame and computing equilibria over agent populations, then ranking and evaluating each agent with respect to that equilibrium. This analysis focuses on happens at the selected equilibrium.

The set of Nash equilibria satisfies contraction consistency but not expansion consistency with respect to strategy removal. If an equilibrium exists in the full game and its support strategies are retained, it persists in the restricted game (contraction). However, removing a strategy can *create* new equilibria that did not exist before, by eliminating a profitable deviation that was previously destabilizing other profiles (expansion failure). This means that adding or removing a strategy from the metagame, even one outside the equilibrium support, can change the set of equilibria. Furthermore, equilibrium *selection* is not preserved: the equilibrium chosen by a given solver can change because the selection landscape (e.g., entropy over the equilibrium set) shifts when a strategy is removed.

In an evaluation scenario this matters for a few reasons. First, a strategy that does not rank highest on any individual metric may nonetheless drive higher cooperation or welfare at equilibrium through its interactions with other agents. Second, using game-theoretic analysis to rank and evaluate agents depends on the equilibrium selected, and agents can rank very differently under different equilibrium selections. Equilibrium selection is a well-studied open problem in game theory, and we make no claims of solving it here. Rather, we propose a add on to meta-game evaluation that aims to account for multiple restricted games, multiple equilibria, and the causal role of individual agents/strategies across these settings. By examining how welfare and cooperation change as agents are added or removed from the metagame, and by examining outcomes across strategically stable subsets (CURB sets), we provide evaluations that are informed by the structure of the metagame's equilibrium landscape rather than dependent on a single selection.

As LLMs are increasingly deployed as autonomous agents in strategic settings, understanding how their presence affects cooperation and welfare in multi-agent populations is of growing importance. As multi-agent competitions scale, from Kaggle's [Game Arena](https://www.kaggle.com/game-arena) where LLMs compete in social deduction games, to PSRO-trained leagues, practitioners are looking to study not just which agents are competitively strong, but which agents' presence shapes the equilibrium. We contribute a methodology that measures each agent's *causal contribution* to cooperation and welfare at equilibrium, asking not "how cooperative is this agent?" but "how much does this agent's presence cause the equilibrium to be more cooperative?"

### Connection to Agent Importance in MARL

Recent work on explainable multi-agent importance (EMAI, Xu et al. 2024) measures agent importance through counterfactual reasoning: randomize an agent's actions and measure the reward change. Our framework applies a similar counterfactual logic at the metagame level, but instead of randomizing actions uniformly, we evaluate outcomes at equilibrium under a chosen solution concept. This grounds the counterfactual in strategic reasoning: the importance of an agent is measured by how its presence or absence affects the equilibrium, not just average performance.

---

## Three Levels of Analysis

### Level 1: Equilibrium Analysis (no counterfactuals)

Standard EGTA. Solve for equilibrium on the full game and evaluate each agent's performance:

```
σ*_X = S(Ĝ_{S↓X})
V_i(X) = M[i,:] @ σ*_X          (agent i's payoff at equilibrium)
W(X) = σ*_Xᵀ M σ*_X             (aggregate welfare at equilibrium)
```

This tells you what happens at the equilibrium and how each agent performs. No counterfactuals. Reports individual metrics (regret, payoff, fairness) at the selected equilibrium.

### Level 2: Full-Game LOO (counterfactual, single solver)

Remove agent sᵢ from the full game, re-solve for equilibrium, and measure the welfare change:

```
ΔW(sᵢ | X) = W(σ*_X) - W(σ*_{X\{sᵢ}})
```

Where both equilibria are found by the same solver S (e.g., MENE). Paired bootstrap provides confidence intervals: one resample of the raw data induces all subgames (full, LOO, LTO), so comparisons are paired across bootstrap index b:

```
d_b = W(full, b) - W(LOO_i, b)
```

This measures each agent's causal contribution to equilibrium welfare. However, results are solver-dependent: different solution concepts can identify different agents as important.

**Harsanyi Interaction Dividends** extend LOO to pairwise interactions:

```
Δ²W(sᵢ, sⱼ | X) = W(σ*_X) - W(σ*_{X\{sᵢ}}) - W(σ*_{X\{sⱼ}}) + W(σ*_{X\{sᵢ,sⱼ}})
```

- Δ² > 0: complementary (together they contribute more than the sum of parts)
- Δ² < 0: substitutes (individually helpful but redundant together)
- Δ² ≈ 0: independent

**Solution concept comparison**: We support multiple solvers (MENE, maxent CCE, max affinity entropy) and compare the causal attributions across them. Solver sensitivity is quantified as:

```
Sens(S₁, S₂; X) = W_{S₁}(X) - W_{S₂}(X)
```

### Level 3: CURB-Conditional LOO (LOO on strategically coherent restricted games)

A central challenge is that games with multiple equilibria render Level 2 LOO effects dependent on equilibrium selection. Level 3 addresses this by performing LOO on CURB sets (Closed Under Rational Behavior, Basu & Weibull 1991), strategically self-reinforcing subsets of the strategy space where every Nash equilibrium is guaranteed to live.

Rather than LOO on the full game with a single solver, Level 3 performs LOO within each CURB set. This is LOO on restricted games, where the restricted games are chosen to be strategically coherent (CURB) rather than arbitrary.

**Algorithm (per bootstrap sample b):**

1. **Find CURB sets** of the bootstrapped payoff matrix M^(b) using Klimm & Weibull (2009), plus MC sampling of random CURB closures for non-minimal CURB sets.

2. **Solve NE within each CURB set** C on the restricted game:

```
σ*_C = S(Ĝ_{S↓C})
```

3. **Evaluate welfare on the restricted game:**

```
W_k(C, b) = σ*_Cᵀ M_k^(b)[C, C] σ*_C
```

4. **Welfare interval for bootstrap b:**

```
[W_min^(b), W_max^(b)] = [min_C W(C, b), max_C W(C, b)]
```

5. **LOO within each CURB set:** For each agent sᵢ ∈ C (skipping singleton CURB sets), solve NE on the restricted game C \ {sᵢ} and compute the delta:

```
Δ(sᵢ | C, b) = W(σ*_C, M^(b)[C, C]) - W(σ*_{C\{sᵢ}}, M^(b)[C\{sᵢ}, C\{sᵢ}])
```

Agents not in C have no effect on C's equilibrium.

6. **Interval per bootstrap:** For each agent sᵢ, take min and max delta across all CURB sets containing sᵢ:

```
min_Δ^(b)(sᵢ) = min_{C : sᵢ ∈ C, |C| ≥ 2} Δ(sᵢ | C, b)
max_Δ^(b)(sᵢ) = max_{C : sᵢ ∈ C, |C| ≥ 2} Δ(sᵢ | C, b)
```

7. **Across B bootstraps**, report mean and 95% CI on the min and max delta distributions.

**Classification:**

| Condition | Interpretation |
|-----------|---------------|
| CI lower of min_Δ > 0 | Robustly helpful: agent improves welfare in every CURB set it belongs to |
| CI upper of max_Δ < 0 | Robustly harmful: agent hurts welfare in every CURB set it belongs to |
| min_Δ < 0 < max_Δ | CURB-set-dependent: effect varies across strategically coherent restricted games |

**Why Level 3 is more principled than Level 2:**

- Level 2 uses one solver on the full game. The LOO effect depends on which equilibrium the solver picks (IIA violation).
- Level 3 performs LOO across multiple strategically coherent restricted games (CURB sets) and reports the range of effects.
- Level 2 can declare an agent "significantly helpful" when it is only helpful in one CURB set (the one the solver happened to pick).
- Level 3 reveals whether the effect is robust across CURB sets or varies across them.
- The full game is always a CURB set, so the Level 2 full-game LOO is one entry in the Level 3 interval. Level 3 strictly generalizes Level 2.

---

## Additional Analysis Tools

### Synergy Index (Emergence Detection)

The Synergy Index (cf. MACIE, Weinberg 2025) measures whether the multi-agent system exhibits emergence:

```
SI = (W(X) - Σᵢ W({sᵢ})) / max(W(X), Σᵢ W({sᵢ}))
```

Where W({sᵢ}) is agent sᵢ's self-play payoff. SI > 0 indicates positive emergence (strategic diversity creates value), SI < 0 indicates interference.

### CURB-Level Interaction Effects

Interaction effects between a CURB set and the rest of the game:

```
Δ²W(C, X\C) = W(X) - W(C) - W(X\C) + W(∅)
```

This addresses the problem of wide individual LOO CIs for interchangeable agents. Removing the entire CURB set (group-level LOO) produces a clean, significant effect where individual LOO cannot distinguish within the group.

### On Direct/Indirect Effect Decomposition

In causal mediation analysis (cf. Weighted Mobius Score, Jiang & Steinert-Threlkeld 2023), total effects can be decomposed into direct and indirect effects. However, in a strategic context this decomposition is not cleanly defined: there is no principled way to "hold the equilibrium fixed" while removing a strategy, since the equilibrium is itself a function of the strategy set. For out-of-support agents the entire LOO effect is indirect, while for in-support agents the two effects are entangled. We therefore report total LOO effects and use structural analysis (CURB set stability, best-response graph changes) to explain the mechanism behind each effect.

---

## CURB Set Computation

### Definition

A non-empty subset C ⊆ S is **Closed Under Rational Behavior (CURB)** if CBR(C) ⊆ S, where CBR(C) is the set of all strategies that are a best response to some mixture over C.

### Klimm & Weibull (2009) Algorithm

We implement the two-step algorithm from Klimm & Weibull for finding all minimal CURB sets:

**Algorithm 1:** Find all minimal wCURB (weak CURB) configurations via the pure best-reply graph on strategy profiles. For each profile s, compute the set of reachable profiles P(s), then iteratively expand until closed under pure best responses.

**Algorithm 2:** Promote wCURB candidates to CURB via LP feasibility checks. Maintains a family of candidates, picks the size-minimal candidate, checks all strategies outside it via LP. If a violator is found (a strategy that is a BR to some mixture over the candidate), adds it to all candidates in the family and restarts. If no violator, the candidate is confirmed CURB.

For two-player games, CURB = sCURB (strong CURB), so the algorithm finds all minimal CURB sets.

### LP Feasibility Check

For each candidate strategy πᵢ, check if it is a best response to some mixture over C:

```
Find σⱼ ≥ 0 for j ∈ C,  Σⱼ σⱼ = 1
subject to:  Σⱼ σⱼ · u(πᵢ, πⱼ) ≥ Σⱼ σⱼ · u(πₖ, πⱼ)   ∀ πₖ ∈ S
```

If feasible, then πᵢ ∈ CBR(C).

---

## Solution Concepts

| Solver | Description |
|--------|-------------|
| MENE | Maximum Entropy Nash Equilibrium: unique NE maximizing Shannon entropy |
| Maxent CCE | Maximum Entropy Coarse Correlated Equilibrium: unique CCE maximizing entropy over the CE polytope (via Polarix) |
| Max Affinity Entropy | Nash equilibrium with affinity entropy regularization (via Polarix). Uses a similarity kernel to weight entropy by strategic diversity. |

---

## Welfare Metrics

### Bargaining Domain

| Metric | Description |
|--------|-------------|
| UW | Utilitarian Welfare: u₁ + u₂ |
| NW | Nash Welfare: (u₁ · u₂)^(1/2) |
| NW+ | Nash Welfare on advantages: (u₁⁺ · u₂⁺)^(1/2) where u⁺ = max{0, u - BATNA} |
| EF1 | Envy-Free up to One Item frequency |
| Regret | u(σ*) - u(π, σ*₋ᵢ): deviation incentive at equilibrium |

### Iterated Matrix Game Domains (PD, Hawk-Dove)

| Metric | Description |
|--------|-------------|
| Payoff | Average per-round payoff (evaluated at equilibrium) |
| Cooperation | Cooperation rate (evaluated at equilibrium) |
| NW | Geometric mean of both players' payoffs per matchup |

---

## Evaluation Domains

### 1. Multi-Agent Bargaining (from Zun Li et al.)

RL agents (PPO, PSRO, MAPPO) and LLM agents (OpenAI o1 variants) negotiate over item allocations. The empirical metagame has 13 strategies with rich welfare/fairness structure.

### 2. Iterated Prisoner's Dilemma

Strategies from the [Axelrod Python library](https://github.com/Axelrod-project/Axelrod) play repeated PD with noise. Noise differentiates robust vs fragile cooperators and creates non-trivial equilibrium structure.

### 3. Iterated Hawk-Dove

Same Axelrod strategies under Hawk-Dove payoffs (anti-coordination). Different game structure reveals different causal patterns across domains.

---

## Notation Reference

| Symbol | Description |
|--------|-------------|
| G = (N, (Sᵢ), (uᵢ)) | Normal-form game with players N, strategy sets Sᵢ, utilities uᵢ |
| Ĝ | Empirical game (payoffs estimated via simulation/data) |
| S | Full strategy universe |
| X ⊆ S | Restricted strategy set |
| S↓X | Restriction operator |
| Ĝ_{S↓X} | Restricted empirical game induced by X |
| σ ∈ Δ(X) | Mixed strategy profile over X |
| σ*_X | Equilibrium mixture computed on Ĝ_{S↓X} |
| S | Meta-strategy solver (MSS): Ĝ_{S↓X} ↦ σ_X |
| BRᵢ(σ₋ᵢ) | Best-response correspondence for player i |
| CBR(C) | Conditional best response set of C |
| ρᵢᴳ(σ) | Regret of player i at profile σ |
| Φ | Metric functional (UW, NW, EF1, etc.) |
| W_S(X) | Welfare at equilibrium: Φ(Ĝ_{S↓X}, S(Ĝ_{S↓X})) |

---

## Key References

- Basu & Weibull (1991). "Strategy Subsets Closed Under Rational Behavior." *Economics Letters* 36(2).
- Klimm & Weibull (2009). "Finding all minimal CURB sets." *HAL-00442118*.
- Benisch, Davis & Sandholm (2010). "Algorithms for CURB Sets." *JAIR* 38.
- Kline & Tamer (2024). "Counterfactual Analysis in Empirical Games." *arXiv:2410.12731*.
- Wellman (2024). "Empirical Game-Theoretic Analysis: A Survey." *JAIR*.
- Xu et al. (2024). "EMAI: Explainable Multi-Agent Importance." *arXiv:2412.15619*.
- Jiang & Steinert-Threlkeld (2023). "Weighted Mobius Score." *arXiv:2305.09204*.
- Weinberg (2025). "MACIE: Multi-Agent Causal Intelligence Explainer." *SSRN*.
- Smithline, Mascioli, Chakraborty & Wellman (2025). "Measuring Competition and Cooperation in LLM Bargaining."
- Li & Wellman (2024). "Meta-Game Evaluation of Agents." *IJCAI*.
- Hurkens (1995). "Learning by Forgetful Players." *GEB* 11.
- Young (1993). "The Evolution of Conventions." *Econometrica* 61.

---

## Installation

Requires Python 3.10+. Install using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/your-username/Causal-Game-Analysis.git
cd Causal-Game-Analysis
uv sync
```

## Development

```bash
uv run pytest
uv run pytest --cov=src/iterative_game_analysis
uv run ruff check .
```

## License

MIT
