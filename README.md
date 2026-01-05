# Causal Meta-Game Analysis

A framework for analyzing multi-agent systems using Structural Causal Models (SCM) and do-calculus. This framework provides rigorous causal semantics for evaluating policies in meta-games across three levels of analysis.

## Installation

Requires Python 3.10+. Install using [uv](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/your-username/Causal-Game-Analysis.git
cd Causal-Game-Analysis

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# For development (includes testing and linting tools)
uv pip install -e ".[dev]"
```

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

Measures how a candidate $\pi_j$ performs as a partner compared to the baseline equilibrium mixture, **without** allowing the ecosystem to re-equilibrate.

**Baseline equilibrium interaction value** for incumbent $\pi_i \in B$:

$$U_B(\pi_i) := \mathbb{E}_{\pi \sim \sigma_B}\big[\mu(\pi_i, \pi)\big] = \sum_{\pi \in B} \sigma_B(\pi) \, \mu(\pi_i, \pi)$$

**Partner Lift** (incumbent-specific):

$$\text{PL}_1(\pi_i; \pi_j \mid B) := \mu(\pi_i, \pi_j) - U_B(\pi_i)$$

This answers: *"If incumbent $\pi_i$ faces candidate $\pi_j$ instead of a typical equilibrium partner, what is the expected change in outcome?"*

**Aggregations:**

| Aggregation | Formula |
|-------------|---------|
| Uniform average | $\overline{\text{PL}}_1^{\text{unif}}(\pi_j \mid B) := \frac{1}{\|B\|} \sum_{\pi_i \in B} \text{PL}_1(\pi_i; \pi_j \mid B)$ |
| Equilibrium-weighted | $\overline{\text{PL}}_1^{\sigma}(\pi_j \mid B) := \sum_{\pi_i \in B} \sigma_B(\pi_i) \, \text{PL}_1(\pi_i; \pi_j \mid B)$ |
| Worst-case | $\text{PL}_1^{\min}(\pi_j \mid B) := \min_{\pi_i \in B} \text{PL}_1(\pi_i; \pi_j \mid B)$ |
| Best-case | $\text{PL}_1^{\max}(\pi_j \mid B) := \max_{\pi_i \in B} \text{PL}_1(\pi_i; \pi_j \mid B)$ |

---

### Level 2: Ecosystem-Level (Re-Equilibration)

Measures the causal effect of adding $\pi_j$ to the library, **allowing** strategic adaptation via re-equilibration.

Let $B^+ := B \cup \{\pi_j\}$. The **ecosystem lift** is:

$$\Delta_{\text{eco}}(\pi_j \mid B) := W(B^+) - W(B) = \mathbb{E}\big[W_L \mid \text{do}(L=B^+)\big] - \mathbb{E}\big[W_L \mid \text{do}(L=B)\big]$$

**Incumbent value shift** under re-equilibration:

$$\Delta_{\text{inc}}(\pi_i; \pi_j \mid B) := V_{B^+}(\pi_i) - V_B(\pi_i)$$

where $V_S(\pi_i) := \sum_{\pi \in S} \sigma_S(\pi) \, \mu(\pi_i, \pi)$.

**Equilibrium diagnostics:**

| Metric | Formula |
|--------|---------|
| Equilibrium shift | $\Delta_\sigma(\pi_j \mid B) := \|\sigma_{B^+} - \sigma_B\|_1$ |
| Entry mass | $\text{EntryMass}(\pi_j \mid B) := \sigma_{B^+}(\pi_j)$ |

---

### Level 3: Ecosystem Attribution (Shapley/Banzhaf)

Attributes ecosystem value to individual policies using cooperative game theory.

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
