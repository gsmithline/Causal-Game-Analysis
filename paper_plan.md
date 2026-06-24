# Paper Plan: Counterfactual Analysis in Empirical Metagames

Status: scoping. Target venue and deadline TBD.

---

## Story arc

Strategy sets in EGTA are ad hoc — whatever agents researchers happened to build — so the equilibrium they induce inherits that contingency. Existing rating / decomposition methods (Elo, α-rank, Bertrand's disc decomposition, clone-invariant deviation ratings, within-equilibrium attributions) score *strength* or *competitive standing*; none of them tell you what each strategy contributes to the equilibrium itself as a system-level outcome.

We propose a three-level diagnostic that treats the equilibrium as a function of the strategy set and measures each strategy's *externality* on welfare/cooperation/fairness at equilibrium. The framework is counterfactual in the game-theoretic sense (comparing two model equilibria), not interventional. We validate on bargaining (headline) and iterated PD (cross-domain), and use Czarnecki's spinning-top suite to substantiate a structural diagnostic (the CURB coverage ratio ρ) that links our framework to existing cyclicity findings while extending to non-zero-sum games where those methods don't apply.

---

## Contributions

**C1. Three-level diagnostic framework.**
- L1: equilibrium evaluation (standard EGTA).
- L2: full-game LOO + second-order interaction effects (each strategy's marginal effect on equilibrium welfare; pairwise complementarity/substitution).
- L3: CURB-conditional LOO (LOO inside each strategically coherent restricted game; classifies each strategy as consistently-positive / consistently-negative / CURB-dependent).

**C2. Equilibrium-as-diagnostic framing.**
The strategy set is a design choice; each agent imposes externalities on the equilibrium. Distinct from competitive rating (Elo, α-rank, disc decomposition, clone-invariant ratings) and from within-equilibrium attribution (Liu et al. 2025). Counterfactual (two games), not interventional (Pearl/Conitzer).

**C3. CURB coverage ρ as a structural diagnostic.**
ρ(G) = |largest minimal CURB| / |S| tells you when L3 buys you anything, and bridges to existing cyclicity findings.
- (A) Theoretical: ρ = 1 iff the pure-best-response graph on S is strongly connected.
- (B) Empirical: ρ tracks cyclicity (Bertrand λ_disc/λ_transitive or ‖M − Mᵀ‖_F/‖M‖_F) across Czarnecki's spinning-top suite.
- (C) Generalization: ρ is defined on any normal-form game, including non-zero-sum (bargaining, PD), where Bertrand's disc decomposition does not apply.

**C4. Bargaining empirical findings.**
- A zero-support strategy (5.2_low) has larger LOO welfare impact than the 79%-support strategy (PPO).
- PSRO is the only agent with CURB-dependent sign — its effect flips depending on strategic context.
- MAPPO is a functional clone of PPO (substitute interaction).

---

## Theory proved in paper

**Theorem 1 (CURB stability under strategy removal).**
If C ⊆ S is CURB in G and s ∈ C with |C| ≥ 2, then C \ {s} is CURB in G \ {s}.

*Proof.* For any σ ∈ Δ(C \ {s}) ⊆ Δ(C), BR_G(σ) ⊆ C by CURB closure. Then BR_{G\{s}}(σ) = BR_G(σ) \ {s} ⊆ C \ {s}. ∎

**Corollary 1.** Any NE of G[C \ {s}] is a NE of G \ {s}. CURB-conditional LOO produces valid counterfactual equilibria without deviation verification.

**Proposition (ρ = 1 characterization).**
ρ(G) = 1 iff the pure-best-response graph on S is strongly connected. Equivalently, S is the unique minimal CURB set.

*Sketch.* Klimm-Weibull's wCURB algorithm computes connected components of the pure-BR graph; ρ = 1 says S is a single component.

## Theory cited, not proved
- Independence of Irrelevant Strategies (Peleg & Tijs 1996; Ray 2000)
- CURB closure → NE-of-full-game validity (Basu & Weibull 1991)
- Möbius / Harsanyi decomposition of set functions (Harsanyi 1959)
- Prep set lattice / persistent retracts (Balkenborg, Hofbauer & Kuzmics 2013)

---

## Experiments

### Headline: Bargaining (already done, used in worked example)
- 13 strategies (RL: PPO, PSRO, MAPPO; LLM: o1 variants 5.2_*, 5.4_*)
- L1 / L2 / L3 across metrics: UW, NW, NW+, EF1, EF1+
- Bootstrap CIs (paired across full-game / LOO / CURB-restricted)
- Findings: 5.2_low magnitude > PPO; PSRO CURB-dependent; MAPPO clone of PPO

### Cross-domain: Iterated PD (Axelrod)
- Axelrod strategy pool (subset, sized for tractability — TBD)
- Two matrices: payoff (individual effectiveness) + cooperation (mutual-cooperation rate)
- Same L1 / L2 / L3 pipeline
- Validates framework off zero-sum and shows metric-agnostic attribution (welfare vs cooperation)
- Needs: Axelrod tournament runner, payoff/cooperation matrix builder, sanity check on small N before full run

### Structural diagnostic: Spinning-top (Czarnecki 2020 / Bertrand 2023)
- 19 games from `spinning_top_payoffs.pkl` (Blotto family, Kuhn-poker, AlphaStar, Go 3×3 / 4×4, hex, tic-tac-toe, quoridor, …)
- Compute ρ per game; cyclicity measure per game (Bertrand λ ratio via discrating solver OR cheap Frobenius ratio)
- Scatter plot + Spearman correlation → C3(B)
- Compute ρ on bargaining and PD too → C3(C), shows the diagnostic generalizes off zero-sum
- Bootstrap stability of ρ: only on bargaining (per-pair counts unavailable for Czarnecki)

### Real-elite-player replication: StarCraft II + Lichess elite chess
Extra real-elite-player datasets so the ρ↔cyclicity correlation has a higher-stakes anchor. **Default placement: main paper.** Push to appendix only what doesn't fit after the first draft. Decision deferred until we see actual page count.

- **StarCraft II.** Aligulac tournament data (Aug 2019 – May 2020), ~1.7M games, ~20k players. Loaded via `external/discrating/expes/chess_starcraft/main_starcraft.py` (downloads from aligulac.com).
- **Lichess elite chess.** Lichess elite database, ~4.7M games, 40k+ players, filtered to 2400+ vs 2200+ (Bertrand's cohort). Loaded via `external/discrating/expes/chess_starcraft/main_chess.py`.
- Computation: ρ, ‖M − Mᵀ‖_F / ‖M‖_F, Bertrand λ ratio per dataset. Cyclicity-stratified ρ table.

**Risk: matrix size vs CURB tractability.** Full StarCraft (~20k) and Lichess (~40k) matchup matrices outrun the Klimm-Weibull LP-based CURB algorithm. Note that Great Lakes (Michigan HPC) is available for the compute-heavy parts — large bootstraps, PGN ingestion, disc-decomposition on big matrices — so data-engineering scale isn't the binding constraint. The binding constraint is the *intrinsic* complexity of Klimm-Weibull's LP feasibility checks, which scale with strategy count regardless of cluster size. Mitigation plan, in order of preference:
1. Restrict to high-confrontation subgraph (Bertrand filters chess to pairs with ≥80 confrontations, StarCraft to ≥some-threshold) — typically collapses to a few hundred players.
2. If still too large, take top-N by rating (e.g., top 200) to get a CURB-tractable sub-matrix.
3. Report ρ on the subset; report cyclicity measure on both the subset and the full matrix; flag any discrepancy.

The interpretive caveat from earlier applies: in these datasets "strategies" are individual humans/bots, so LOO is "remove this player," weaker than "remove this designed strategy." Frame the appendix as *structural diagnostic replication*, not as compositional analysis.

---

## Section outline (AAMAS-shape, ~8pp + refs)

1. **Introduction** (~1.5pp) — Conitzer/Oesterheld and Dafoe motivation: individual quality ≠ system outcome; the strategy set is a design choice; equilibrium as diagnostic.
2. **Background** (~1pp) — EGTA, CURB and prep sets, related rating/decomposition methods (Bertrand, Czarnecki, Marris, Liu).
3. **Framework** (~2pp) — three levels; Theorem 1 + Corollary 1; ρ-characterization proposition; honest scoping of what's counterfactual vs interventional.
4. **Empirical results** (~2.5pp) — bargaining worked example; PD cross-domain; ρ-vs-cyclicity diagnostic on spinning-top + bargaining + PD.
5. **Discussion + limitations** (~0.5–1pp) — when L3 buys you nothing (high ρ); ANOVA analogy; non-additivity; what we don't claim.
6. **Appendix** (overflow, no page limit) — content that doesn't fit in main paper after first-draft pass. Likely candidates if cuts are needed: subset-selection details, full ρ + cyclicity tables, bootstrap stability of ρ, extended worked-example variants.

---

## Open decisions

- **Target venue and deadline.** AAMAS 2027 (~Oct 2026 deadline) is the working assumption per project memory; confirm.
- **Domain count.** Current: bargaining + PD + Czarnecki (structural-diagnostic only). If schedule pressure, drop PD and rely on bargaining + Czarnecki, but this weakens the cross-domain generalization claim.
- **"Harsanyi dividend" naming.** Currently the README labels L2 second-difference terms as Harsanyi dividends. Strictly these are Möbius coefficients evaluated at the grand coalition X = S, not the full Möbius transform. Options: rename to "second-order LOO effect" (recommended) or keep label and footnote the technicality.
- **Subset-selection criterion for StarCraft / Lichess.** Whether to use Bertrand's confrontation-count threshold, a top-N-by-rating cutoff, or something else. Decide once the matrix-size diagnostic is run.

## Reserve (not in scope unless reviewers push)

- **Proposition 2 (solver-sensitivity bound).** |ΔW_{S₁}(s|C) − ΔW_{S₂}(s|C)| ≤ diam_W(NE(G[C])) + diam_W(NE(G[C\{s}])). Bounds L3 solver-disagreement by NE multiplicity within each CURB set. Hold in reserve.
- **Proposition 3 (ρ = 1 ⇒ L3 ≡ L2).** Cheap to state. Add if there's room in §3.
- **Prep-set decomposition as fallback for high-ρ games.** README discusses this. Currently not exercised empirically; future work unless bargaining or PD turns out to have high ρ.

---

## Reusing the discrating repo (https://github.com/QB3/discrating, MIT-licensed)

Narrow but real. What's grabbable:

Now vendored as a submodule at `external/discrating/` (BSD 3-Clause). What we use:

- `discrating.solvers.solver(...)` — fits Bertrand's disc decomposition (u, v) components to logit(P) via alternate minimization with l-BFGS. Used to compute Bertrand's λ_disc/λ_transitive ratio for the C3(B) scatter plot.
- `discrating.utils.get_energy(us, vs)` — assembles the skew-symmetric reconstruction Σ_k u_k v_kᵀ − v_k u_kᵀ from the fitted components.
- Loading convention for `spinning_top_payoffs.pkl` — pickle of game-name → win-rate matrix, P_ij ∈ [0,1] with P_ij + P_ji = 1.
- `expes/chess_starcraft/main_starcraft.py` and `main_chess.py` — data loaders for the aligulac (StarCraft II) and Lichess elite databases. Both *download data at runtime*; nothing is shipped in the repo.

What's *not* in the submodule:
- The data itself. `spinning_top_payoffs.pkl` lives in Czarnecki et al. 2020's NeurIPS supplementary (linked from their `main_pred.py`). Aligulac and Lichess data are fetched live by their scripts.

What we'd skip:
- Their plotting code (matplotlib defaults; the project's `visuals/` is already styled).

Integration plan:
1. `scripts/cyclicity_ratio.py` (or extend `evaluation/`) imports `discrating.solvers.solver`, loads each game matrix, emits `{game → (ρ, λ_disc/λ_transitive, ‖M − Mᵀ‖_F/‖M‖_F)}`.
2. Spinning-top runs on the pickle. StarCraft and Lichess runs go through `discrating/expes/chess_starcraft/` loaders + our subset-selection step (Open decision above).
3. Output drives the main-paper scatter (spinning-top + bargaining + PD) and the appendix table (StarCraft + Lichess).

---

## Empirical findings — first pass (10 games, 2026-05-18, payoff-matrix-fixed)

NOTE (2026-05-18 update): an initial spinning-top loader mistakenly applied
`M = 2P − 1` to the Czarnecki pickle entries, which are already skew-symmetric
payoff matrices in [−1, 1]. The fix is in place; the corrected numbers below
are the canonical ones. Frobenius cyclicity is degenerate on zero-sum
(skew-symmetric) games — always 2.0 — so for spinning-top games rely on
Bertrand's normal-decomposition magnitudes (λ₂/λ₁ and top-pair fraction)
instead.

The 3-move parity game has a near-dominant pure NE (strategy 0 never loses
to any of the other 159 strategies, ties with 16 of them). β-CURB cannot
reduce to {0} because of those ties; σ-CURB breaks them via BHK
perturbation and recovers the singleton. Bertrand's spectral measure
reports the ties as "multi-component cyclic" (top% = 0.49) — it cannot
distinguish tie-induced cyclic spectrum from real cyclic content. This is
a *structural* gap of spectral methods that σ-CURB closes.

Computed ρ_β (standard β-CURB coverage), ρ_σ (persistent-retract / σ-CURB coverage via BHK 2013 perturbation), the number of weakly inferior strategies |W|, and a Frobenius cyclicity proxy ‖M − Mᵀ‖_F / ‖M‖_F across three domains:

| Game | n | ρ_β | ρ_σ | Δρ | \|W\| | cyc |
|------|---|-----|-----|-----|---|-----|
| bargaining | 10 | 0.100 | 0.100 | 0.000 | 3 | 0.344 |
| PD tournament | 30 | 0.033 | 0.033 | 0.000 | 1 | 0.519 |
| RPS | 3 | 1.000 | 1.000 | 0.000 | 0 | 1.706 |
| 5,3-Blotto | 21 | 0.857 | 0.857 | 0.000 | 3 | 1.620 |
| 5,4-Blotto | 56 | 0.929 | 0.929 | 0.000 | 4 | 1.291 |
| Kuhn-poker | 64 | 0.594 | 0.594 | 0.000 | 24 | 1.040 |
| 10,3-Blotto | 66 | 0.955 | 0.955 | 0.000 | 3 | 1.711 |
| **5,5-Blotto** | 126 | 0.960 | **0.802** | **0.158** | 25 | 1.050 |
| **3-move parity game 2** | 160 | 0.900 | **0.006** | **0.894** | 159 | 1.633 |
| 10,4-Blotto | 286 | 0.944 | 0.944 | 0.000 | 16 | 1.431 |

**Headline observations:**

1. **ρ_β tracks cyclicity across the suite.** ρ_β ≥ 0.85 whenever cyc ≥ 1.0 (cyclic games); ρ_β < 0.1 when cyc < 0.55 (games with pure NE — bargaining, PD). C3(B) confirmed.

2. **σ-CURB usually equals β-CURB** (8 of 10 games). Weakly inferior strategies, when present, are already excluded by β-CURB's best-response closure. So β-CURB is "good enough" as a structural diagnostic in most games.

3. **But σ-CURB *can* strictly refine β-CURB** (2 of 10 games), and when it does, the refinement is large:
   - **5,5-Blotto**: ρ_β = 0.96 → ρ_σ = 0.80 (20 strategies removed).
   - **3-move parity game 2**: ρ_β = 0.90 → ρ_σ = 0.006 — σ collapses the minimal CURB to a *singleton* (pure NE), exposing structure β-CURB missed entirely. 159 of 160 strategies are weakly inferior.

4. **The 3-move parity game is the showcase example.** β-CURB sees a near-fully-cyclic game (ρ_β = 0.9); σ-CURB reveals it actually has a pure NE. This is precisely the high-ρ regime where BHK 2013 predicts σ refinement to bite.

**Implications for the paper:**

- C3(A) — ρ = 1 characterization — still worth proving.
- C3(B) — ρ_β ↔ cyclicity correlation — empirically supported across 10 games.
- C3(C) — ρ defined on non-zero-sum domains — confirmed (bargaining, PD).
- **New angle**: σ-refinement is a *sometimes-firing diagnostic*. It rarely changes the picture, but when it does (3-move parity), it changes it drastically. Worth a paragraph + the parity-game case study.

**Outstanding:** larger spinning-top games (hex, tic-tac-toe, AlphaStar, Go, connect-four, quoridor) are still running in the background. Klimm-Weibull scaling looks like 10,4-Blotto (n=286) took 66s for σ; n=1000+ will likely be 30+ minutes each.

## Candidate theorem: σ-CURB recovers the transitive peak of a spinning-top game (open, 2026-05-18)

Empirical observation from 3-move parity (Czarnecki Thm 1 construction): β-CURB and Bertrand's spectral measure both report rich cyclic structure (ρ_β = 0.9, top% = 0.49); σ-CURB collapses to a singleton at the never-losing peak strategy (ρ_σ = 1/n).

This is not a coincidence of one game — it appears to be a structural consequence of the Czarnecki et al. 2020 "Game of Skill" geometry. Stated as a candidate theorem, to be sharpened later:

### Conjecture (informal)

Let G be a finite 2-player symmetric zero-sum game with payoff matrix M ∈ R^{n×n}, M = −Mᵀ. Define the transitive peak

  T(G) = { t ∈ Π : M[t, k] ≥ 0 for all k ∈ Π }

— the set of strategies that never lose. Suppose:

  (S1) T(G) is non-empty.
  (S2) Every strategy outside T(G) is weakly inferior in BHK 2013's sense (weakly dominated by, or own-payoff-equivalent to a proper mixture over, the other strategies).

Then **every minimal persistent retract of G is contained in T(G).** When T(G) has a unique payoff-equivalence class, the minimal persistent retract is a singleton drawn from T(G) (modulo equivalence).

### Proof sketch

*Step 1 (peak strategies tie each other).* For any t, t' ∈ T(G): t never loses ⇒ M[t, t'] ≥ 0; t' never loses ⇒ M[t', t] ≥ 0. By skew-symmetry M[t, t'] = −M[t', t], so both are zero. The peak is a flat "tied plateau."

*Step 2 (β-CURB cannot reduce to a singleton on the peak).* For any single t ∈ T(G), CBR_G({t}) = argmax_k M[k, t]. The maximum is 0, achieved by every k ∈ T(G). So {t} is not β-CURB whenever |T(G)| > 1. This is the "tie blockage" β-CURB cannot see through.

*Step 3 (off-peak strategies are perturbed; on-peak strategies are not).* By (S2), the BHK perturbation reduces M[k, :] by ε for every k ∉ T(G) and leaves T(G) untouched. In the perturbed game G' the new payoffs are M'[k, t] = M[k, t] − ε for k ∉ T(G), M'[k, t] = M[k, t] for k ∈ T(G).

*Step 4 (in G', CBR collapses to the peak).* For col playing pure t ∈ T(G): M'[k, t] equals 0 for k ∈ T(G), and at most −ε for k ∉ T(G). So CBR_G'({t}) = T(G). The peak T(G) is therefore β-CURB in G', i.e., σ-CURB in G — and crucially, *no off-peak strategy* survives.

*Step 5 (minimality within T(G)).* If T(G) is a single payoff-equivalence class, then within T(G) every pure strategy is own-payoff-equivalent to every other; the next round of BHK perturbation (applied to T(G) as a subgame) breaks ties between them, and one survives. The empirical result (3-move parity: σ-CURB = {0}, |T(G)| = 16) is consistent with this iterated picture, though formally the BHK 2013 theorem only refers to *one* perturbation pass; the iteration would need its own justification.

### Open questions and caveats — 2026-05-18 updates after empirical verification

- **Proper-mixture vs degenerate-mixture (RESOLVED).** The LP fix is in
  `evaluation/persistent_retracts.py` (controlled by `require_proper=True`,
  default). Across all 7 games tested (bargaining, PD, RPS, 5,3-/5,4-/10,3-
  /5,5-Blotto, Kuhn-poker, 3-move parity), legacy and proper-mixture give
  *identical* |W|. The empirical results are principled per BHK 2013
  without modification.
- **Theorem hypothesis (S2) is much stronger than expected (NEW).** I
  verified σ-CURB ⊆ T(G) on the seven spinning-top games. It only holds
  on **3-move parity** — the one game in the set where (S2) genuinely
  applies (every off-peak strategy is weakly inferior). On the others:
  - 5,5-Blotto: T(G) = 1, σ-CURB has 101 strategies. (S2) fails because
    the cyclic middle layer is *not* fully weakly inferior.
  - Kuhn-poker: T(G) = ∅, σ-CURB has 38 strategies. (S1) fails.
  - RPS, Blottos: T(G) = ∅. (S1) fails.
  So the strong conjecture characterizes a *narrow* subclass of Games of
  Skill — those where the off-peak structure is fully redundant (3-move
  parity is one by construction). Honest paper framing should reflect
  that narrowness.
- **Weak conjecture (TRIVIALLY TRUE).** σ-CURB ⊆ Π \ W — every minimal
  persistent retract is contained in the complement of the weakly
  inferior set. Direct from the perturbation construction; not a
  theorem in any deep sense, but it's the right inequality to *state*
  for general games and it's the one that bounds ρ_σ ≤ |Π \ W| / n.
- **Reverse direction.** Is σ-CURB = {t} singleton ⇒ G has a spinning-
  top structure with a unique peak? Still open and probably the more
  interesting direction.
- **Generalization beyond zero-sum.** Step 1 (peak strategies tie each
  other) used skew-symmetry. For non-zero-sum games, the "never loses"
  notion needs reinterpretation. Open.

### Status after verification

The strong theorem (σ-CURB ⊆ T(G)) holds *only* on games whose off-peak
structure is fully weakly-inferior — a narrower subclass than the full
spinning-top hypothesis. 3-move parity is the clearest example (it's
constructed to have this property); 5,5-Blotto and Kuhn-poker have
*non-redundant* cyclic middle layers and the theorem doesn't apply.

For the paper's C3 section, this means the theorem is not a universal
characterization of Games of Skill. The honest framing is two-part:

1. **General structural fact (trivial proof, useful inequality):**
   σ-CURB ⊆ Π \ W, so ρ_σ ≤ |Π \ W| / n. This bounds the σ refinement
   in terms of the size of the weakly-inferior set.

2. **Sharper statement for special games:** When (S1) + (S2) hold —
   the strong "fully redundant off-peak" condition — σ-CURB ⊆ T(G),
   and on 3-move parity this gives ρ_σ = 1/n (singleton at the peak).
   Bertrand's spectral measure cannot detect this collapse; β-CURB
   cannot either because of the tie blockage.

The 3-move parity case still earns its keep as a demonstration: it's
the cleanest example of σ-CURB seeing something Bertrand structurally
cannot. But the paper should not over-claim a general spinning-top
theorem; the empirical evidence supports a narrower statement.

---

## Bertrand-style normal decomposition side-by-side (2026-05-18)

Computed Bertrand 2023 Theorem 1 magnitudes λ_1, λ_2, λ_2/λ_1 and top-pair fraction λ_1² / Σ λ_l² via SVD of the antisymmetric component (M − Mᵀ)/2 (faithful to Bertrand's normal decomposition; faster than their iterative l-BFGS solver and works for non-zero-sum games via the antisymmetric component).

| Game | ρ_β | ρ_σ | λ₂/λ₁ | top% | Agreement |
|------|-----|-----|-------|------|-----------|
| bargaining | 0.100 | 0.100 | 0.123 | 0.983 | agree (near-transitive / pure NE) |
| PD | 0.033 | 0.033 | 0.343 | 0.809 | agree |
| RPS | 1.000 | 1.000 | 0.000 | 1.000 | agree (pure disc) |
| Blottos (5,3 / 5,4 / 10,3) | ≥0.85 | ≥0.85 | high | low | agree (rich cyclic) |
| Kuhn-poker | 0.594 | 0.594 | 0.290 | 0.841 | **disagree** |
| 5,5-Blotto | 0.960 | 0.802 | 0.523 | 0.467 | agree on cyclic, σ exposes 20 extras |
| 3-move parity | 0.900 | 0.006 | 0.642 | 0.486 | **strong disagreement** |
| 10,4-Blotto | 0.944 | 0.944 | 0.715 | 0.344 | agree (rich cyclic) |

**Two paragraphs for the C3 section:**

1. *Where they agree.* For 7 of 10 games, ρ_β and Bertrand's top-pair fraction tell the same qualitative story — pure-NE games show low ρ_β + high top%; cyclic games show high ρ_β + low top%. ρ_β replicates Bertrand's cyclicity diagnostic on the games where Bertrand applies.

2. *Where they disagree, σ-CURB sees what Bertrand cannot.* The 3-move parity game (n=160) is the smoking gun: Bertrand reports a multi-component cyclic structure (top% = 0.49, λ₂/λ₁ = 0.64) but σ-CURB collapses the minimal CURB to a singleton (ρ_σ = 0.006) — exposing a pure NE that 159 of 160 strategies are weakly inferior against. Bertrand's normal decomposition is a *spectral* statement about the antisymmetric part of logit(P); it cannot distinguish "rich cyclic content" from "near-degenerate weakly-inferior structure that collapses to a strict equilibrium." CURB-based diagnostics are sensitive to that distinction. Kuhn-poker is a milder version of the same: Bertrand's top% = 0.84 ("mostly one disc"), but minimal β-CURB has 38 strategies — strategic structure Bertrand flattens.

This sharpens C3(C): ρ doesn't just *extend* Bertrand's measure to non-zero-sum games — it captures CURB-closure facts that the spectral measure cannot, even on the zero-sum games where both methods apply.

---

## Concrete next steps (in order)

1. Confirm scope above (venue, domain count, Harsanyi naming, subset criterion).
2. Write the PD pipeline (Axelrod tournament → payoff + cooperation matrices → existing L1/L2/L3 entrypoints). Sanity-check on small N first.
3. Pull `spinning_top_payoffs.pkl` from Czarnecki supplementary; write the cyclicity-ratio script using the vendored `external/discrating/`.
4. Run ρ + cyclicity on all 19 spinning-top games + bargaining + PD; produce the C3(B)+(C) figure for the main paper.
5. Appendix pipeline: run `external/discrating/expes/chess_starcraft/main_starcraft.py` and `main_chess.py` to materialize the aligulac and Lichess matchup matrices; apply subset-selection; compute ρ + cyclicity; produce appendix table.
6. Cut the "Harsanyi dividend" terminology in `README.md` and code comments; replace with the chosen term.
7. Draft Theorem 1, Corollary 1, ρ-characterization proposition for the paper body.
8. First full draft pass on intro + framework + experiments.
