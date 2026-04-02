# Counterfactual Analysis in Empirical Metagames

## Motivation

The number of AI agents deployed in shared environments is growing. As these populations grow and agents increasingly interact strategically, understanding the impact of each agent on the broader dynamics, not just its individual performance, becomes a way to evaluate and manage them.

We frame this as evaluation for strategy set curation: if you control which strategies are available to players, each inclusion changes the game, which changes the equilibrium, which changes the outcome for everyone. Standard metagame analysis evaluates how each strategy performs at equilibrium. We measure the impact of including each strategy, asking not "how good is this strategy?" but "what is the impact of including this strategy on the equilibrium outcome?"

A strategy can be individually strong but making it available leads to a worse equilibrium for the system. Conversely, a strategy that is never played at equilibrium can still shape which equilibrium forms by its presence in the strategy set. Equilibrium is a nonlinear function of the strategy set, so the impact of including a strategy is generally not predictable from its performance within the equilibrium. This can reveal structural roles invisible to standard analysis: which strategies are holding the current equilibrium in place, which would shift it to a qualitatively different one, and which are substitutable.

### Worked Example (Bargaining Domain, Average Game)

The MENE equilibrium of the average game is 79% PPO and 21% PSRO. All other agents have zero support. One might expect only removing these two agents would shift the equilibrium.

**Removing 5.2_low** (0% support, rank 1 in individual welfare):
- Equilibrium shifts: 79% PPO / 21% PSRO → 68% PPO / 30% 5.4_medium / 2% PSRO
- Welfare improves across all metrics: UW +10.87, NW +7.60, NW+ +4.41, EF1 +4.2%, EF1+ +7.7%
- The magnitude of impact is larger than removing PPO, the 79% agent
- 5.2_low is never played, but its presence as the best response to 5.4_medium prevents 5.4_medium from entering the equilibrium

**Removing PPO** (79% support):
- Equilibrium shifts: 79% PPO / 21% PSRO → 80% MAPPO / 20% PSRO
- Near-zero welfare impact because MAPPO is a functional copy of PPO
- The dominant agent is entirely substitutable

**Removing PSRO** (21% support, ranks 7th-10th across welfare/fairness metrics):
- Equilibrium shifts: 79% PPO / 21% PSRO → 44% 5.4_medium / 32% 5.2_low / 17% 5.2_medium / 7% 5.2_none
- Regime change from RL-dominated to LLM-dominated equilibrium
- Largest welfare impact of any agent: UW +39.08, NW +24.44, EF1 +16.8%
- PSRO's presence forces a competitive equilibrium

**CURB-conditional LOO** on the same game (23 CURB sets, 3 minimal singletons: {ppo}, {psro}, {mappo}):

| Agent | n_curbs | UW [min, max] | NW [min, max] | NW+ [min, max] | EF1 [min, max] | EF1+ [min, max] | Classification |
|-------|---------|---------------|---------------|----------------|----------------|-----------------|----------------|
| ppo | 16 | [+6.26, +6.26] | [+4.00, +4.00] | [+2.04, +2.04] | [+0.041, +0.041] | [+0.010, +0.010] | Robustly helpful |
| psro | 20 | [-45.34, +6.67] | [-28.44, +3.96] | [-13.95, +3.43] | [-0.209, +0.038] | [-0.284, +0.009] | CURB-dependent |
| mappo | 20 | [-3.57, 0.00] | [-0.72, 0.00] | [-0.36, 0.00] | [-0.063, 0.00] | [-0.015, 0.00] | Robustly harmful |
| 5.2_low | 20 | [-46.92, 0.00] | [-28.63, 0.00] | [-4.41, 0.00] | [-0.197, 0.00] | [-0.077, 0.00] | Robustly harmful |

Although 5.2_low has zero support in the full-game equilibrium, it has the largest worst-case CURB LOO magnitude (-46.92 UW) of any agent. PPO's effect is identical across all 16 CURB sets it appears in, completely robust. PSRO is the only agent whose effect is CURB-dependent, ranging from -45.34 to +6.67 depending on the strategic context.

### Independence of Irrelevant Strategies

This shift is motivated by a fundamental property of Nash equilibrium under strategy removal. The Nash equilibrium correspondence satisfies independence of irrelevant strategies (IIS) (Peleg & Tijs, 1996; Ray, 2000) in the contraction direction: removing a strategy outside the equilibrium support preserves existing equilibria, as it only eliminates potential deviations. However, IIS only guarantees preservation of existing equilibria, not the absence of new ones: removing a strategy can create new equilibria by eliminating a profitable deviation that was previously destabilizing other profiles. Furthermore, equilibrium selection is not preserved: the equilibrium chosen by a given solver can change because the selection landscape shifts when a strategy is removed. This means that adding or removing a strategy from the metagame, even one with zero equilibrium weight, can change both which equilibria exist and which one is selected.

Equilibrium selection is a well-studied open problem in game theory. We propose an add-on to metagame evaluation that accounts for multiple restricted games, multiple equilibria, and the counterfactual role of individual agents across these settings. By examining how welfare and cooperation change as agents are added or removed from the strategy set, and by examining outcomes across strategically stable subsets (CURB sets), we provide evaluations that are informed by the structure of the equilibrium landscape rather than dependent on a single selection.

### Connection to Agent Importance in MARL

Recent work on explainable multi-agent importance (EMAI, Xu et al. 2024) measures agent importance through counterfactual reasoning: randomize an agent's actions and measure the reward change. Our framework applies a similar counterfactual logic at the metagame level, but instead of randomizing actions uniformly, we evaluate outcomes at equilibrium under a chosen solution concept. This grounds the counterfactual in strategic reasoning: the importance of an agent is measured by how its presence or absence affects the equilibrium, not just average performance.

### Connection to Equilibrium-Based Rating Methods

Recent work on equilibrium-based evaluation includes clone-invariant deviation ratings (Marris et al. 2025), which rate strategies robustly under CCE, and within-equilibrium marginal decompositions (Liu et al. 2025), which attribute ratings to co-player contributions at a fixed equilibrium. Both operate within a single equilibrium and focus on competitive rating. Our approach is complementary: we perform interventional counterfactual analysis (strategy removal with re-equilibration) across multiple equilibria (via CURB sets), measuring counterfactual contributions to welfare and cooperation rather than competitive rankings. This captures effects that within-equilibrium methods cannot detect, such as agents outside the equilibrium support that reshape which equilibrium exists.

---

## Three Levels of Analysis

All three levels operate on the same bootstrapped empirical game per sample. One bootstrap resample of the raw game data induces the full game, all LOO restricted games, all LTO restricted games, and all CURB restricted games from the same resampled payoff matrix. Because every comparison within a bootstrap sample comes from the same resample, all differences are paired, canceling the dominant noise source and enabling tight confidence intervals.

### Level 1: Equilibrium Analysis (no counterfactuals)

Standard EGTA. Solve for equilibrium on the bootstrapped game and evaluate each agent's performance:

```
σ*_X = S(Ĝ_{S↓X})
V_i(X) = M[i,:] @ σ*_X          (agent i's payoff at equilibrium)
W(X) = σ*_Xᵀ M σ*_X             (aggregate welfare at equilibrium)
```

This tells you what happens at the equilibrium and how each agent performs. No counterfactuals. Reports individual metrics (regret, payoff, fairness, cooperation) at the selected equilibrium. Answers: "how does each agent perform at equilibrium?"

### Level 2: Full-Game LOO (counterfactual, single solver)

For each bootstrap sample, remove agent sᵢ from the same bootstrapped game, re-solve for equilibrium, and measure the welfare change. All full-game and restricted-game solves use the same resampled payoff matrix, so LOO differences are paired across bootstrap index b:

```
ΔW(sᵢ | X) = W(σ*_X) - W(σ*_{X\{sᵢ}})
d_b = W(full, b) - W(LOO_i, b)
```

This measures each agent's counterfactual contribution to equilibrium welfare. However, results are solver-dependent: different solution concepts can identify different agents as important. Answers: "which agents counterfactually drive the equilibrium?"

**Harsanyi Interaction Dividends** extend LOO to pairwise interactions, computed on the same bootstrap sample (full, LOO, and LTO all from one resample):

```
Δ²W(sᵢ, sⱼ | X) = W(σ*_X) - W(σ*_{X\{sᵢ}}) - W(σ*_{X\{sⱼ}}) + W(σ*_{X\{sᵢ,sⱼ}})
```

- Δ² > 0: complementary (together they contribute more than the sum of parts)
- Δ² < 0: substitutes (individually helpful but redundant together)
- Δ² ≈ 0: independent

**Solution concept comparison**: We support multiple solvers (MENE, maxent CCE, max affinity entropy) and compare the counterfactual attributions across them to distinguish solver-dependent from robust findings. Solver sensitivity is quantified as:

```
Sens(S₁, S₂; X) = W_{S₁}(X) - W_{S₂}(X)
```

### Level 3: CURB-Conditional LOO (LOO on strategically coherent restricted games)

A central challenge is that games with multiple equilibria render Level 2 LOO effects dependent on equilibrium selection. Level 3 addresses this by performing LOO on CURB sets (Closed Under Rational Behavior, Basu & Weibull 1991), strategically self-reinforcing subsets of the strategy space where every Nash equilibrium is guaranteed to live.

Rather than LOO on the full game with a single solver, Level 3 performs LOO within each CURB set. This is LOO on restricted games, where the restricted games are chosen to be strategically coherent (CURB) rather than arbitrary. As with Levels 1 and 2, all CURB set computation, NE solving, and LOO evaluation are performed on the same bootstrapped payoff matrix per sample, maintaining the paired structure throughout. Answers: "is this agent's contribution robust across strategically coherent restricted games?"

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

#### CURB Coverage and the Prep Set Fallback

A known limitation of CURB sets is that in some games, even the smallest minimal CURB set spans nearly the entire strategy space (Ritzberger & Weibull 1995; Balkenborg, Hofbauer & Kuzmics 2013). When this occurs, Level 3 conditioning degenerates: LOO within a CURB set that contains all strategies is equivalent to Level 2 full-game LOO, and the CURB decomposition adds no information.

We quantify this with a **CURB coverage ratio**:

```
ρ(G) = |C_min| / |S|
```

where C_min is the largest minimal CURB set and S is the full strategy set. When ρ ≈ 1, Level 3 is uninformative. When ρ is small, the game admits a rich CURB decomposition and Level 3 provides genuine insight beyond Level 2.

This is connected to game structure: highly cyclic games (where every strategy is a best response to some mixture) tend to have large CURB closures because the best-response graph is dense. Transitive games, where strategies are roughly linearly ordered by strength, tend to have many small CURB sets.

**Prep sets as a finer alternative.** When CURB sets are too coarse, *preparation sets* (Voorneveld 2004) provide a strictly finer decomposition. A prep set is a minimal non-empty set P ⊆ S closed under *some* best response: for every mixture over P, at least one best response remains in P. CURB requires *all* best responses to stay in the set.

Formally, for a generalized best reply correspondence τ ∈ T^PS (Balkenborg et al. 2013):

- **τ-CURB set**: R is τ-CURB if τ(Θ(R)) ⊇ Θ(R) — closed under all τ-best replies
- **τ-prep set**: R is τ-prep if for all x ∈ Θ(R) and all players i, τᵢ(x) ∩ Δ(Rᵢ) ≠ ∅ — at least one τ-best reply stays in R

Key relationships (Balkenborg et al. 2013, Lemmas 2 & 4):

1. Every τ-prep set is contained in some τ-CURB set (prep ⊆ CURB)
2. The smaller τ is in the lattice of generalized BR correspondences, the more τ-CURB sets and the fewer τ-prep sets exist
3. The finest decomposition uses σ (the refined BR correspondence): minimal σ-CURB sets are exactly persistent retracts (Kalai & Samet 1984), and every minimally asymptotically stable face (MASF) must be a tight σ-prep set (Theorem 4)

**Equilibrium validity.** This distinction has a key operational consequence for counterfactual analysis:

- **CURB set equilibria require no deviation check.** Because CURB is closed under *all* best responses, no strategy outside C can be a profitable deviation against any NE of the restricted game G[C]. Any NE of G[C] is automatically a NE of the full game G. This is why Level 3 LOO within CURB sets produces valid counterfactuals without additional verification.

- **Prep set equilibria require a deviation check.** Because prep sets are only closed under *some* best response, strategies outside P may be profitable deviations against a NE of G[P]. Formally: if σ* is a NE of G[P], one must verify that for all sⱼ ∈ S \ P, u(sⱼ, σ*₋ᵢ) ≤ u(σ*ᵢ, σ*₋ᵢ) for all players i. If any outside strategy is a profitable deviation, the restricted equilibrium is not a NE of the full game, and the LOO counterfactual based on it may not be strategically meaningful.

For the purposes of counterfactual analysis:

- **CURB sets** (standard β-CURB) are the default for Level 3 — they are computationally tractable via the Klimm & Weibull algorithm and guarantee equilibrium validity without deviation checks
- **Prep sets** are the fallback when CURB coverage is high — they can decompose a game that CURB cannot, but each restricted equilibrium must be validated against the full strategy set before computing LOO deltas
- The coverage ratio ρ serves as a diagnostic: report it alongside Level 3 results to indicate how informative the CURB conditioning is for a given game

No new theoretical results are needed here; the lattice relationships and containment properties are established in the cited literature. Our contribution is empirical: measuring ρ across games and showing when Level 3 adds value.

---

## Additional Analysis Tools 

### Synergy Index (Emergence Detection) I'm not gonna use these, just ideas for new metrics

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

Strategies from the [Axelrod Python library](https://github.com/Axelrod-project/Axelrod) play repeated PD.

### 3. Iterated Hawk-Dove

Same Axelrod strategies under Hawk-Dove payoffs (anti-coordination). Different game structure reveals different counterfactual patterns across domains.

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
| M̂_k | Estimated matrix for metric k (e.g., UW, NW, coop, EF1) |
| W_{S,k}(X) | Welfare at equilibrium: σ*_Xᵀ M̂_k[X,X] σ*_X where σ*_X = S(Ĝ_{S↓X}) |

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
- Smithline, Mascioli, Chakraborty & Wellman (2025). "Measuring Competition and Cooperation in LLM Bargaining: An Empirical Meta-Game Analysis."
- Li & Wellman (2024). "A meta-game evaluation framework for deep multiagent reinforcement learning." *IJCAI*.
- Hurkens (1995). "Learning by Forgetful Players." *GEB* 11.
- Young (1993). "The Evolution of Conventions." *Econometrica* 61.

---

## Data

The bargaining game crossplay data is available on HuggingFace: https://huggingface.co/datasets/Gsmith43/causal-game-crossplay

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
