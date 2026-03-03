"""Investigate the EF1 Banzhaf paradox.

Raw game data shows agents frequently achieve EF1 (~50% for strong agents,
~58-60% NW+), yet Walk/Tough dominate small-coalition equilibria in the
Banzhaf value computation, dragging coalition EF1 values down.

This script loads bootstrap sample 0, examines coalition details, and
compares equilibrium-weighted EF1 vs uniform-weighted EF1 vs raw empirical
EF1 rates.
"""

import pickle
import sys
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------
# 1.  Load the bootstrap results
# -----------------------------------------------------------------------
pkl_path = Path("/Users/gabesmithline/Desktop/Causal-Game-Analysis/data/analysis/iterative_analysis_results.pkl")
print(f"Loading {pkl_path} ...")
with open(pkl_path, "rb") as f:
    results = pickle.load(f)

raw = results["raw"]          # list of per-bootstrap-sample dicts
config = results["config"]
strategy_names = config["strategy_names"]
print(f"Loaded {len(raw)} bootstrap samples.")
print(f"Strategies: {strategy_names}")

# Convenience mappings
idx_of = {name: i for i, name in enumerate(strategy_names)}

# -----------------------------------------------------------------------
# 2.  Pick sample 0 and pull out the matrices and L3 coalition details
# -----------------------------------------------------------------------
sample = raw[0]
matrices = sample["matrices"]       # dict of numpy matrices
ef1_matrix = matrices["ef1"]        # NaN where no accept occurred
payoff_matrix = matrices["payoff"]
nw_matrix = matrices["nw"]
nw_plus_matrix = matrices["nw_plus"]

full_sigma = sample["full_game"]["sigma"]
print(f"\n{'='*70}")
print("FULL GAME EQUILIBRIUM (sample 0)")
print(f"{'='*70}")
for i, name in enumerate(strategy_names):
    print(f"  {name:25s}: {full_sigma[i]:.6f}")

# -----------------------------------------------------------------------
# 3.  Display the raw EF1 matrix (before nan->0)
# -----------------------------------------------------------------------
print(f"\n{'='*70}")
print("RAW EF1 MATRIX (NaN = no accepts => undefined)")
print(f"{'='*70}")
header = f"{'':25s}" + "".join(f"{n:>12s}" for n in strategy_names)
print(header)
for i, ni in enumerate(strategy_names):
    row = f"{ni:25s}"
    for j in range(len(strategy_names)):
        v = ef1_matrix[i, j]
        if np.isnan(v):
            row += f"{'NaN':>12s}"
        else:
            row += f"{v:12.4f}"
    print(row)

# -----------------------------------------------------------------------
# 4.  For each target coalition, compute:
#     (a) Equilibrium sigma from the payoff subgame
#     (b) v(S) using equilibrium weights (sigma^T * ef1_clean * sigma)
#     (c) v(S) using uniform weights (1/n ... 1/n)
# -----------------------------------------------------------------------
# We need the solver to re-compute equilibria for subgames.
sys.path.insert(0, str(Path("/Users/gabesmithline/Desktop/Causal-Game-Analysis")))
from src.iterative_game_analysis.metagame import MetaGame

# Build a metagame from this sample's payoff matrix
metagame = MetaGame(
    policies=strategy_names,
    payoff_matrix=payoff_matrix,
)

COALITIONS = {
    "{PPO, PSRO}":                ["ppo", "psro"],
    "{PPO, PSRO, NFSP}":         ["ppo", "psro", "nfsp"],
    "{PPO, Walk}":                ["ppo", "walk"],
    "{PPO, Tough}":               ["ppo", "tough"],
    "{PSRO, Walk}":               ["psro", "walk"],
    "{PSRO, Tough}":              ["psro", "tough"],
    "{PPO, PSRO, Walk}":          ["ppo", "psro", "walk"],
    "{PPO, PSRO, Tough}":         ["ppo", "psro", "tough"],
    "{PPO, PSRO, Walk, Tough}":   ["ppo", "psro", "walk", "tough"],
    "{Walk, Tough}":              ["walk", "tough"],
    "{Walk}":                     ["walk"],
    "{Tough}":                    ["tough"],
    "FULL (all 8)":               list(strategy_names),
}

print(f"\n{'='*70}")
print("COALITION VALUE ANALYSIS  (EF1)")
print("For each coalition S, show equilibrium sigma, v_eq(S), v_unif(S)")
print(f"{'='*70}")

for label, members in COALITIONS.items():
    indices = [idx_of[p] for p in members]
    n = len(members)

    # Sub-matrices
    sub_payoff = payoff_matrix[np.ix_(indices, indices)]
    sub_ef1_raw = ef1_matrix[np.ix_(indices, indices)]
    sub_ef1 = np.nan_to_num(sub_ef1_raw, nan=0.0)

    # Equilibrium on payoff subgame
    sub_game = MetaGame(members, sub_payoff)
    try:
        sigma = sub_game.solve("mene")
    except Exception as e:
        print(f"\n  {label}: solver failed ({e})")
        continue

    v_eq = float(sigma @ sub_ef1 @ sigma)

    # Uniform weights
    u = np.ones(n) / n
    v_unif = float(u @ sub_ef1 @ u)

    # Also compute: average of diagonal (self-play EF1) and average of
    # all off-diagonal (cross-play EF1)
    diag_vals = [sub_ef1_raw[i, i] for i in range(n)]
    off_diag_vals = [sub_ef1_raw[i, j] for i in range(n) for j in range(n) if i != j]
    avg_diag = np.nanmean(diag_vals) if any(not np.isnan(v) for v in diag_vals) else float("nan")
    avg_offdiag = np.nanmean(off_diag_vals) if any(not np.isnan(v) for v in off_diag_vals) else float("nan")

    # Count NaN cells
    nan_count = int(np.isnan(sub_ef1_raw).sum())
    total_cells = n * n

    print(f"\n  {label}")
    print(f"    Members: {members}")
    print(f"    Equilibrium sigma:")
    for i_m, m_name in enumerate(members):
        print(f"      {m_name:25s}: {sigma[i_m]:.6f}")
    print(f"    v_eq(S)  [sigma^T * EF1 * sigma]  = {v_eq:.6f}")
    print(f"    v_unif(S) [uniform weights]        = {v_unif:.6f}")
    print(f"    Gap (v_unif - v_eq)                = {v_unif - v_eq:.6f}")
    print(f"    NaN cells in EF1 sub-matrix: {nan_count}/{total_cells}")
    print(f"    Avg self-play EF1 (diagonal)       = {avg_diag:.4f}")
    if n > 1:
        print(f"    Avg cross-play EF1 (off-diag)      = {avg_offdiag:.4f}")

    # Show the EF1 sub-matrix
    print(f"    EF1 sub-matrix (NaN->0 for eq computation):")
    hdr = "      " + "".join(f"{m:>12s}" for m in members)
    print(hdr)
    for i_m, m_name in enumerate(members):
        row_str = f"      {m_name:>12s}"
        for j_m in range(n):
            raw_v = sub_ef1_raw[i_m, j_m]
            if np.isnan(raw_v):
                row_str += f"{'NaN(->0)':>12s}"
            else:
                row_str += f"{raw_v:12.4f}"
        print(row_str)


# -----------------------------------------------------------------------
# 5.  Also do the same for NW+ to show the pattern is EF1-specific
# -----------------------------------------------------------------------
print(f"\n{'='*70}")
print("COALITION VALUE ANALYSIS  (NW+)")
print(f"{'='*70}")

KEY_COALITIONS = {
    "{PPO, PSRO}":                ["ppo", "psro"],
    "{PPO, Walk}":                ["ppo", "walk"],
    "{PPO, Tough}":               ["ppo", "tough"],
    "{PPO, PSRO, Walk}":          ["ppo", "psro", "walk"],
    "{PPO, PSRO, Tough}":         ["ppo", "psro", "tough"],
    "FULL (all 8)":               list(strategy_names),
}

for label, members in KEY_COALITIONS.items():
    indices = [idx_of[p] for p in members]
    n = len(members)
    sub_payoff = payoff_matrix[np.ix_(indices, indices)]
    sub_nw_plus = nw_plus_matrix[np.ix_(indices, indices)]
    sub_nw_plus_clean = np.nan_to_num(sub_nw_plus, nan=0.0)

    sub_game = MetaGame(members, sub_payoff)
    try:
        sigma = sub_game.solve("mene")
    except Exception:
        continue

    v_eq = float(sigma @ sub_nw_plus_clean @ sigma)
    u = np.ones(n) / n
    v_unif = float(u @ sub_nw_plus_clean @ u)

    print(f"\n  {label}")
    print(f"    Equilibrium sigma:")
    for i_m, m_name in enumerate(members):
        print(f"      {m_name:25s}: {sigma[i_m]:.6f}")
    print(f"    v_eq(S)  [NW+, equilibrium]   = {v_eq:.4f}")
    print(f"    v_unif(S) [NW+, uniform]       = {v_unif:.4f}")
    print(f"    Gap (v_unif - v_eq)            = {v_unif - v_eq:.4f}")


# -----------------------------------------------------------------------
# 6.  Load raw crossplay data and compute empirical EF1 rates
# -----------------------------------------------------------------------
print(f"\n{'='*70}")
print("EMPIRICAL EF1 RATES FROM RAW CROSSPLAY DATA")
print(f"{'='*70}")

from src.iterative_game_analysis.full_analysis import load_crossplay_to_dataframe

crossplay_dir = Path("/Users/gabesmithline/Desktop/Causal-Game-Analysis/data/crossplay")
print("\nLoading raw crossplay data...")
df = load_crossplay_to_dataframe(
    crossplay_dir,
    strategy_names,
    raw_utility=True,
)

# Compute pairwise empirical EF1 rates
print(f"\n--- Empirical EF1 rates (accepted games only, NaN excluded) ---")
pairs_of_interest = [
    ("ppo", "psro"),
    ("ppo", "walk"),
    ("ppo", "tough"),
    ("psro", "walk"),
    ("psro", "tough"),
    ("walk", "tough"),
    ("ppo", "ppo"),
    ("psro", "psro"),
    ("walk", "walk"),
    ("tough", "tough"),
    ("ppo", "nfsp"),
    ("psro", "nfsp"),
    ("ppo", "mappo"),
    ("ppo", "soft"),
    ("ppo", "openai_5.2_none"),
]

for pi, pj in pairs_of_interest:
    mask = (df["policy_i"] == pi) & (df["policy_j"] == pj)
    subset = df.loc[mask, "ef1"]
    n_total = len(subset)
    n_valid = subset.notna().sum()
    if n_valid > 0:
        ef1_rate = subset.mean()  # NaN excluded by default
        print(f"  {pi:25s} vs {pj:25s}: EF1={ef1_rate:.4f}  (n_accept={n_valid}, n_total={n_total})")
    else:
        print(f"  {pi:25s} vs {pj:25s}: NO ACCEPTS  (n_total={n_total})")

# Also compute: what fraction of games end in acceptance?
print(f"\n--- Accept rate (fraction of games with non-NaN EF1) ---")
for name in strategy_names:
    mask_i = df["policy_i"] == name
    total_as_i = mask_i.sum()
    accept_as_i = df.loc[mask_i, "ef1"].notna().sum()
    if total_as_i > 0:
        print(f"  {name:25s} as policy_i: {accept_as_i}/{total_as_i} = {accept_as_i/total_as_i:.4f}")

# -----------------------------------------------------------------------
# 7.  The key comparison: equilibrium-weighted vs empirical for specific
#     coalitions
# -----------------------------------------------------------------------
print(f"\n{'='*70}")
print("KEY COMPARISON: EQUILIBRIUM v(S) vs EMPIRICAL EF1")
print("Shows the disconnect between what equilibrium predicts and what")
print("agents actually achieve")
print(f"{'='*70}")

def empirical_ef1_for_coalition(df, members):
    """Average pairwise empirical EF1 across all pairs in coalition."""
    rates = []
    for pi in members:
        for pj in members:
            mask = (df["policy_i"] == pi) & (df["policy_j"] == pj)
            vals = df.loc[mask, "ef1"]
            valid = vals.dropna()
            if len(valid) > 0:
                rates.append(valid.mean())
    if rates:
        return np.mean(rates)
    return float("nan")

COMPARE_COALITIONS = {
    "{PPO, PSRO}":            ["ppo", "psro"],
    "{PPO, Walk}":            ["ppo", "walk"],
    "{PPO, Tough}":           ["ppo", "tough"],
    "{PPO, PSRO, Walk}":      ["ppo", "psro", "walk"],
    "{PPO, PSRO, Tough}":     ["ppo", "psro", "tough"],
    "{PPO, PSRO, NFSP}":      ["ppo", "psro", "nfsp"],
    "{Walk, Tough}":          ["walk", "tough"],
    "FULL (all 8)":           list(strategy_names),
}

print(f"\n  {'Coalition':<30s} {'v_eq(EF1)':>12s} {'v_unif(EF1)':>12s} {'Empirical':>12s} {'eq_gap':>12s}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

for label, members in COMPARE_COALITIONS.items():
    indices = [idx_of[p] for p in members]
    n = len(members)
    sub_payoff = payoff_matrix[np.ix_(indices, indices)]
    sub_ef1_raw = ef1_matrix[np.ix_(indices, indices)]
    sub_ef1 = np.nan_to_num(sub_ef1_raw, nan=0.0)

    sub_game = MetaGame(members, sub_payoff)
    try:
        sigma = sub_game.solve("mene")
    except Exception:
        continue

    v_eq = float(sigma @ sub_ef1 @ sigma)
    u = np.ones(n) / n
    v_unif = float(u @ sub_ef1 @ u)
    emp = empirical_ef1_for_coalition(df, members)
    gap = emp - v_eq if not np.isnan(emp) else float("nan")

    print(f"  {label:<30s} {v_eq:12.4f} {v_unif:12.4f} {emp:12.4f} {gap:12.4f}")


# -----------------------------------------------------------------------
# 8.  Why does Walk/Tough dominate? Show payoff matrix for small coalitions
# -----------------------------------------------------------------------
print(f"\n{'='*70}")
print("WHY WALK/TOUGH DOMINATE: PAYOFF MATRIX ANALYSIS")
print("In the payoff metagame, Walk/Tough exploit cooperators.")
print(f"{'='*70}")

for label, members in [
    ("{PPO, Walk}", ["ppo", "walk"]),
    ("{PPO, Tough}", ["ppo", "tough"]),
    ("{PSRO, Walk}", ["psro", "walk"]),
    ("{PPO, PSRO, Walk}", ["ppo", "psro", "walk"]),
    ("{PPO, PSRO, Tough}", ["ppo", "psro", "tough"]),
]:
    indices = [idx_of[p] for p in members]
    n = len(members)
    sub_payoff = payoff_matrix[np.ix_(indices, indices)]
    sub_ef1_raw = ef1_matrix[np.ix_(indices, indices)]

    sub_game = MetaGame(members, sub_payoff)
    try:
        sigma = sub_game.solve("mene")
    except Exception:
        continue

    print(f"\n  {label}")
    print(f"    Payoff sub-matrix:")
    hdr = "      " + "".join(f"{m:>12s}" for m in members)
    print(hdr)
    for i_m, m_name in enumerate(members):
        row_str = f"      {m_name:>12s}"
        for j_m in range(n):
            row_str += f"{sub_payoff[i_m, j_m]:12.2f}"
        print(row_str)
    print(f"    Equilibrium: {dict(zip(members, [f'{s:.4f}' for s in sigma]))}")
    print(f"    Interpretation: Walk/Tough get high payoff when others try to")
    print(f"    cooperate (because Walk gets BATNA which can be high, Tough")
    print(f"    sometimes forces favorable deals). The NE puts weight on them.")


# -----------------------------------------------------------------------
# 9.  Summary of the paradox
# -----------------------------------------------------------------------
print(f"\n{'='*70}")
print("SUMMARY: THE EF1 BANZHAF PARADOX")
print(f"{'='*70}")
print("""
The disconnect arises from two interacting mechanisms:

1. NaN-to-0 conflation: Walk never accepts deals, so all EF1 cells
   involving Walk are NaN (undefined). When computing v(S) = sigma^T *
   EF1 * sigma, NaN is replaced with 0. This means Walk is treated as
   having *zero* fairness, not "undefined" fairness.

2. Payoff-matrix equilibrium selection: The coalition's equilibrium is
   computed on the PAYOFF matrix (not the EF1 matrix). In small coalitions
   containing Walk or Tough + a cooperator, Walk/Tough receive substantial
   equilibrium weight because they exploit cooperators (Walk gets BATNA,
   Tough forces favorable deals). When this equilibrium is then applied to
   the EF1 matrix, the heavy Walk/Tough weight multiplies their 0-valued
   EF1 rows, dragging v(S) toward 0.

The result: v({PPO, Walk}) ~ 0 even though PPO achieves ~50% EF1 in its
games. The Banzhaf marginal contribution of PPO to coalitions containing
Walk is near-zero, because Walk dominates the equilibrium and has EF1=0.

Possible fixes:
- Renormalize EF1 to exclude NaN cells in the sigma^T * M * sigma computation
- Use a separate equilibrium concept for EF1 (e.g., solve on EF1 matrix)
- Weight Banzhaf coalitions by "relevance" (e.g., skip or downweight
  coalitions where >50% of EF1 cells are NaN)
""")
