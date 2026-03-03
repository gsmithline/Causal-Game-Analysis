"""
Analysis: Why does NFSP have highest EF1 but not highest NW/NW+?

Grounded in the actual computation from original_paper_analysis.py:

  nw_plus_matrix[i,j] = mean over ALL games of sqrt(max(raw_adv_i,0) * max(raw_adv_j,0))
    → walks: adv = 0, contributes 0
    → accepts below BATNA: clipped adv = 0, contributes 0
    → displayed as: (nw_plus_matrix @ sigma) / MAX_NW_PLUS * 100  (MAX_NW_PLUS=81.7)

  ef1_matrix[i,j] = (# EF1 accepts) / (# accepts)
    → walks: excluded from both numerator and denominator
    → displayed as: (ef1_matrix @ sigma) * 100

Both metrics only have signal on accepted games. But nw_plus_matrix averages
over all games (walks dilute with 0s), while ef1_matrix only averages over accepts.
"""

import numpy as np
import orjson
from pathlib import Path

ITEM_QUANTITIES = np.array([7, 4, 1])
CROSSPLAY_DIR = Path(__file__).parent / "data" / "crossplay"
MAX_NW_PLUS = 81.7  # From original_paper_analysis.py
MAX_NW = 378.7
MAX_UW = 805.9

STRATEGIES = ["walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro", "openai_5.2_none"]
FOCUS = ["nfsp", "ppo", "psro", "mappo"]


def load_json(path):
    with open(path, 'rb') as f:
        return orjson.loads(f.read())


def is_ef1(u1, u2, a1, a2):
    u1, u2, a1, a2 = np.array(u1), np.array(u2), np.array(a1), np.array(a2)
    p1_ok = np.sum(u1 * a1) >= np.sum(u1 * a2) - np.max(np.where(a2 > 0, u1, 0))
    p2_ok = np.sum(u2 * a2) >= np.sum(u2 * a1) - np.max(np.where(a1 > 0, u2, 0))
    return p1_ok and p2_ok


def load_matchup(si, sj):
    path = CROSSPLAY_DIR / f"{si}_p1_vs_{sj}_p2" / "games.json"
    if not path.exists():
        return None
    data = load_json(path)
    results = []
    for g in data["games"]:
        o = g["outcome"]
        u1 = np.array(o["utilities_p1"])
        u2 = np.array(o["utilities_p2"])
        max_p1 = np.sum(u1 * ITEM_QUANTITIES)
        max_p2 = np.sum(u2 * ITEM_QUANTITIES)
        raw_p1 = o["payoff_p1"] * max_p1
        raw_p2 = o["payoff_p2"] * max_p2
        raw_b1 = o["batna_p1"] * max_p1
        raw_b2 = o["batna_p2"] * max_p2
        accept = o["result"] == "accept"
        ef1 = False
        if accept and o.get("allocation_p1") and o.get("allocation_p2"):
            ef1 = is_ef1(o["utilities_p1"], o["utilities_p2"],
                         o["allocation_p1"], o["allocation_p2"])
        adv1 = max(raw_p1 - raw_b1, 0)
        adv2 = max(raw_p2 - raw_b2, 0)
        results.append({
            "accept": accept, "ef1": ef1,
            "nw_plus": np.sqrt(adv1 * adv2),
            "nw": np.sqrt(max(raw_p1, 0) * max(raw_p2, 0)),
            "uw": raw_p1 + raw_p2,
            "adv1": adv1, "adv2": adv2,
            "raw_p1": raw_p1, "raw_p2": raw_p2,
            "raw_b1": raw_b1, "raw_b2": raw_b2,
        })
    return results


# ======================================================================
# Build matrices exactly as original_paper_analysis.py does (no bootstrap)
# ======================================================================
print("Loading all matchups...")

matchup_data = {}
for si in STRATEGIES:
    for sj in STRATEGIES:
        games = load_matchup(si, sj)
        if games:
            matchup_data[(si, sj)] = games

n = len(STRATEGIES)
idx = {s: i for i, s in enumerate(STRATEGIES)}

# Build matrices matching original_paper_analysis.py
nwp_matrix = np.zeros((n, n))       # nw_plus_matrix: mean over ALL games (raw utility)
ef1_matrix = np.full((n, n), np.nan)  # ef1_matrix: fraction of accepts that are EF1
nw_matrix = np.zeros((n, n))
uw_matrix = np.zeros((n, n))
accept_rate_matrix = np.zeros((n, n))

# Also build accepts-only NW+ and NW matrices for comparison
nwp_acc_matrix = np.full((n, n), np.nan)
nw_acc_matrix = np.full((n, n), np.nan)

for (si, sj), games in matchup_data.items():
    i, j = idx[si], idx[sj]
    accepts = [g for g in games if g["accept"]]
    n_total = len(games)
    n_acc = len(accepts)

    accept_rate_matrix[i, j] = n_acc / n_total if n_total > 0 else 0

    # Exactly as original_paper_analysis.py: mean over ALL games
    nwp_matrix[i, j] = np.mean([g["nw_plus"] for g in games])
    nw_matrix[i, j] = np.mean([g["nw"] for g in games])
    uw_matrix[i, j] = np.mean([g["uw"] for g in games])

    if n_acc > 0:
        ef1_matrix[i, j] = sum(1 for g in accepts if g["ef1"]) / n_acc
        nwp_acc_matrix[i, j] = np.mean([g["nw_plus"] for g in accepts])
        nw_acc_matrix[i, j] = np.mean([g["nw"] for g in accepts])

# Symmetrize (matching original_paper_analysis.py)
nwp_sym = 0.5 * (nwp_matrix + nwp_matrix.T)
nw_sym = 0.5 * (nw_matrix + nw_matrix.T)
uw_sym = 0.5 * (uw_matrix + uw_matrix.T)
ef1_sym = 0.5 * (np.nan_to_num(ef1_matrix, nan=0) + np.nan_to_num(ef1_matrix, nan=0).T)
nwp_acc_sym = 0.5 * (np.nan_to_num(nwp_acc_matrix, nan=0) + np.nan_to_num(nwp_acc_matrix, nan=0).T)
nw_acc_sym = 0.5 * (np.nan_to_num(nw_acc_matrix, nan=0) + np.nan_to_num(nw_acc_matrix, nan=0).T)
acc_sym = 0.5 * (accept_rate_matrix + accept_rate_matrix.T)

# Approximate equilibrium from bootstrap
sigma = np.zeros(n)
eq_weights = {"walk": 0.011, "tough": 0.0, "nfsp": 0.009, "mappo": 0.187,
              "soft": 0.0, "ppo": 0.432, "psro": 0.335, "openai_5.2_none": 0.026}
for s, w in eq_weights.items():
    sigma[idx[s]] = w
sigma /= sigma.sum()


# ======================================================================
# TABLE 1: Reproduce the bootstrap results (point estimate, no resampling)
# to verify our matrices match
# ======================================================================
print("\n" + "=" * 100)
print("TABLE 1: Verify matrices match bootstrap results (point estimates)")
print("=" * 100)

print(f"\n  {'Agent':<16s} {'NW%':>8s} {'UW%':>8s} {'NW+%':>8s} {'EF1%':>8s}")
print(f"  {'-'*50}")

for agent in STRATEGIES:
    i = idx[agent]
    nw_val = (nw_sym[i] @ sigma) / MAX_NW * 100
    uw_val = (uw_sym[i] @ sigma) / MAX_UW * 100
    nwp_val = (nwp_sym[i] @ sigma) / MAX_NW_PLUS * 100
    ef1_val = (ef1_sym[i] @ sigma) * 100
    print(f"  {agent:<16s} {nw_val:>7.2f}% {uw_val:>7.2f}% {nwp_val:>7.2f}% {ef1_val:>7.2f}%")


# ======================================================================
# TABLE 2: The core comparison — how the matrices differ
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 2: How the matrices are built (the denominator difference)")
print()
print("  nw_plus_matrix[i,j] = mean(sqrt(max(adv_i,0)*max(adv_j,0))) over ALL games")
print("  ef1_matrix[i,j]     = count(EF1 accepts) / count(accepts)")
print()
print("  Walk games:          NW+ numerator += 0, denominator += 1  (dilutes)")
print("                       EF1: excluded entirely")
print("  Below-BATNA accepts: NW+ numerator += 0, denominator += 1  (dilutes)")
print("                       EF1: denominator += 1                  (dilutes)")
print("=" * 100)


# ======================================================================
# TABLE 3: The decomposition — per-agent at equilibrium
# nw_plus_matrix[i,j] = accept_rate[i,j] * mean_nwp_on_accepts[i,j]
# So: NW+(agent) = Σ_j σ_j * accept_rate[i,j] * NW+_per_accept[i,j]
# vs: EF1(agent) = Σ_j σ_j * EF1_per_accept[i,j]
# The accept_rate factor is only in NW+, not in EF1.
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 3: Decomposition at equilibrium")
print()
print("  NW+(agent)  = Σ_j σ_j × accept_rate[i,j] × NW+_per_accept[i,j]")
print("  EF1(agent)  = Σ_j σ_j × EF1_per_accept[i,j]")
print()
print("  NW+ has an extra accept_rate multiplier that EF1 does not.")
print("=" * 100)

print(f"\n  {'Agent':<16s} {'AccRate':>8s}  {'NW+/MAX%':>9s} {'NW+_acc':>10s}  {'EF1%':>8s}")
print(f"  {'':16s} {'vs eq':>8s}  {'(all gms)':>9s} {'(raw,acc)':>10s}  {'(acc only)':>8s}")
print(f"  {'-'*60}")

for agent in STRATEGIES:
    i = idx[agent]
    acc_rate = acc_sym[i] @ sigma
    nwp_val = (nwp_sym[i] @ sigma) / MAX_NW_PLUS * 100
    nwp_acc_raw = nwp_acc_sym[i] @ sigma  # raw utility, NOT normalized by MAX_NW_PLUS
    ef1_val = (ef1_sym[i] @ sigma) * 100
    print(f"  {agent:<16s} {acc_rate*100:>7.1f}%  {nwp_val:>8.2f}% {nwp_acc_raw:>10.1f}  {ef1_val:>7.2f}%")


# ======================================================================
# TABLE 4: Per-matchup detail for focus agents vs equilibrium opponents
# Show the raw matrix entries that feed into the equilibrium calculation
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 4: Raw symmetrized matrix entries vs equilibrium-heavy opponents")
print("         (ppo=43%, psro=34%, mappo=19% of σ)")
print("=" * 100)

eq_opps = ["ppo", "psro", "mappo"]
for agent in FOCUS:
    print(f"\n  {agent.upper()}:")
    print(f"  {'vs':>12s} {'σ_opp':>6s}  {'acc_rate':>9s} {'ef1[i,j]':>9s} {'nwp/MAX%':>9s} "
          f"{'nwp_acc(raw)':>13s} {'0-adv%|acc':>11s}")
    print(f"  {'-'*75}")

    for opp in eq_opps:
        i, j = idx[agent], idx[opp]
        # Use symmetrized values
        acc_rate = acc_sym[i, j]
        ef1_val = ef1_sym[i, j]
        nwp_val = nwp_sym[i, j]
        nwp_acc_val = nwp_acc_sym[i, j]
        w = eq_weights[opp]

        # Compute 0-advantage rate on accepted games
        g1 = matchup_data.get((agent, opp), [])
        g2 = matchup_data.get((opp, agent), [])
        all_g = g1 + g2
        accepts = [g for g in all_g if g["accept"]]
        zero_adv = sum(1 for g in accepts if g["adv1"] == 0 or g["adv2"] == 0) / max(len(accepts), 1)

        print(f"  vs {opp:<8s} {w:>5.3f}  {acc_rate*100:>8.1f}% {ef1_val*100:>8.1f}% "
              f"{nwp_val/MAX_NW_PLUS*100:>8.2f}% {nwp_acc_val:>13.1f} "
              f"{zero_adv*100:>10.1f}%")


# ======================================================================
# TABLE 5: Below-BATNA accepts — NW+ clips to 0 but counted in EF1 denominator
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 5: Below-BATNA accepted deals (adv clipped to 0 → NW+ = 0)")
print("         These count in EF1 denominator but add 0 to NW+ numerator")
print("=" * 100)

print(f"\n  {'Agent':<12s} {'#Accepts':>10s} {'%NW+=0|acc':>12s} {'%EF1|NW+=0':>12s} "
      f"{'%EF1|NW+>0':>12s} {'AvgNW+|NW+>0':>13s}")
print(f"  {'-'*75}")

for agent in FOCUS:
    all_accepts = []
    for opp in STRATEGIES:
        for si, sj in [(agent, opp), (opp, agent)]:
            if si == sj == agent:
                continue
            games = matchup_data.get((si, sj))
            if games is None:
                continue
            all_accepts.extend(g for g in games if g["accept"])

    n_acc = len(all_accepts)
    nwp_zero = [g for g in all_accepts if g["nw_plus"] == 0]
    nwp_pos = [g for g in all_accepts if g["nw_plus"] > 0]
    n_zero = len(nwp_zero)
    n_pos = len(nwp_pos)

    ef1_in_zero = sum(1 for g in nwp_zero if g["ef1"]) / max(n_zero, 1) * 100
    ef1_in_pos = sum(1 for g in nwp_pos if g["ef1"]) / max(n_pos, 1) * 100
    avg_nwp_pos = np.mean([g["nw_plus"] for g in nwp_pos]) if nwp_pos else 0

    print(f"  {agent:<12s} {n_acc:>10,d} {n_zero/n_acc*100:>11.1f}% {ef1_in_zero:>11.1f}% "
          f"{ef1_in_pos:>11.1f}% {avg_nwp_pos:>12.1f}")


# ======================================================================
# TABLE 6: Rankings — same metric, different denominators
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 6: Rankings — how denominator choice affects ranking")
print("=" * 100)

agents_data = []
for agent in STRATEGIES:
    i = idx[agent]
    agents_data.append({
        "name": agent,
        "ef1": (ef1_sym[i] @ sigma) * 100,
        "nwp": (nwp_sym[i] @ sigma) / MAX_NW_PLUS * 100,
        "nwp_acc": nwp_acc_sym[i] @ sigma,  # raw utility, no /MAX normalization
        "acc_rate": (acc_sym[i] @ sigma) * 100,
    })

def rank_col(data, key, label, fmt="{:>7.2f}%"):
    ranked = sorted(data, key=lambda x: -x[key])
    return [(d["name"], fmt.format(d[key]), r+1) for r, d in enumerate(ranked)]

# Side by side
print(f"\n  {'#':>3s}  {'EF1 (acc denom)':>20s}  {'NW+/MAX (all denom)':>22s}  {'NW+_acc (raw)':>22s}  {'Accept rate':>14s}")
print(f"  {'-'*85}")

ef1_r = rank_col(agents_data, "ef1", "EF1")
nwp_r = rank_col(agents_data, "nwp", "NW+")
nwp_acc_r = rank_col(agents_data, "nwp_acc", "NW+ acc", fmt="{:>7.1f}")
acc_r = rank_col(agents_data, "acc_rate", "AccRate")

for rank in range(n):
    e = ef1_r[rank]
    p = nwp_r[rank]
    pa = nwp_acc_r[rank]
    a = acc_r[rank]
    print(f"  #{rank+1:>1d}  {e[0]:<10s} {e[1]:>8s}  {p[0]:<12s} {p[1]:>8s}  "
          f"{pa[0]:<12s} {pa[1]:>8s}  {a[0]:<8s} {a[1]:>5s}")


# ======================================================================
# TABLE 7: Summary — the two factors
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 7: Why NFSP has highest EF1 but not highest NW+")
print("=" * 100)
print("""
  NW+(agent) / MAX_NW+  =  Σ_j σ_j × accept_rate[i,j] × mean_NW+_on_accepts[i,j] / MAX_NW+
  EF1(agent)            =  Σ_j σ_j × EF1_rate_on_accepts[i,j]

  Two factors cause the divergence:

  1. ACCEPT RATE GAP: NFSP walks more vs the equilibrium mix.
     NW+ formula has an accept_rate multiplier; EF1 does not.
     Walking adds 0 to the NW+ numerator while increasing the denominator.

  2. CLIPPING: In accepted games where one side got below their BATNA,
     max(advantage, 0) clips to 0, making NW+ = 0 for that game.
     These games still count in the EF1 denominator but contribute nothing to NW+.
     NFSP has FEWER of these bad deals (37.5% vs 44-49% for others),
     which is why it ranks #1 in NW+_per_accept. But this advantage
     is overwhelmed by the accept rate gap.
""")

# Show the math for NFSP vs PPO
i_nfsp = idx["nfsp"]
i_ppo = idx["ppo"]

nfsp_acc = acc_sym[i_nfsp] @ sigma
ppo_acc = acc_sym[i_ppo] @ sigma
nfsp_nwp_acc = nwp_acc_sym[i_nfsp] @ sigma    # raw utility
ppo_nwp_acc = nwp_acc_sym[i_ppo] @ sigma      # raw utility
nfsp_nwp = nwp_sym[i_nfsp] @ sigma / MAX_NW_PLUS * 100
ppo_nwp = nwp_sym[i_ppo] @ sigma / MAX_NW_PLUS * 100
nfsp_ef1 = ef1_sym[i_nfsp] @ sigma * 100
ppo_ef1 = ef1_sym[i_ppo] @ sigma * 100

print(f"  NFSP vs PPO comparison:")
print(f"  {'':22s} {'NFSP':>10s} {'PPO':>10s} {'NFSP wins?':>12s}")
print(f"  {'-'*57}")
print(f"  {'Accept rate vs eq':<22s} {nfsp_acc*100:>9.1f}% {ppo_acc*100:>9.1f}% {'':>12s}")
print(f"  {'NW+ per accept (raw)':<22s} {nfsp_nwp_acc:>10.1f} {ppo_nwp_acc:>10.1f} {'YES':>12s}")
print(f"  {'NW+/MAX overall':<22s} {nfsp_nwp:>9.2f}% {ppo_nwp:>9.2f}% {'no':>12s}")
print(f"  {'EF1 per accept':<22s} {nfsp_ef1:>9.2f}% {ppo_ef1:>9.2f}% {'YES':>12s}")
print(f"\n  NFSP makes better deals when it deals (higher raw NW+/accept, higher EF1).")
print(f"  But PPO deals more often ({ppo_acc*100:.0f}% vs {nfsp_acc*100:.0f}%), and the accept_rate")
print(f"  multiplier in the NW+ formula means volume wins.")


# ======================================================================
# Helper: decode action to allocation
# ======================================================================
def decode_action_to_alloc(action):
    """Decode action index to (p1_alloc, p2_alloc). Returns None if not a counteroffer."""
    if action >= 80:
        return None
    item2 = action % 2
    rem = action // 2
    item1 = rem % 5
    item0 = rem // 5
    p2_alloc = np.array([item0, item1, item2])
    p1_alloc = ITEM_QUANTITIES - p2_alloc
    return p1_alloc, p2_alloc


def load_raw_games(si, sj):
    """Load raw game data including actions."""
    path = CROSSPLAY_DIR / f"{si}_p1_vs_{sj}_p2" / "games.json"
    if not path.exists():
        return None
    return load_json(path)["games"]


# ======================================================================
# TABLE 8: NFSP EF1 frequency (of outcomes) vs each opponent
# EF1 freq = EF1 accepts / accepts (standard paper definition)
# ======================================================================
# ======================================================================
# TABLE 7b: Accept rate per agent — (# accepts) / (# games) vs each opponent
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 7b: Accept rate per agent vs each opponent")
print("          Accept rate = (# games ending in accept) / (# total games)")
print("=" * 100)

print(f"\n  {'Agent':<16s}", end="")
for opp in STRATEGIES:
    print(f" {opp[:7]:>8s}", end="")
print(f" {'Overall':>9s}")
print(f"  {'-'*(16 + 9*len(STRATEGIES) + 9)}")

for agent in STRATEGIES:
    print(f"  {agent:<16s}", end="")
    total_acc, total_games = 0, 0
    for opp in STRATEGIES:
        # Symmetrize: average accept rate across both role assignments
        g1 = matchup_data.get((agent, opp), [])
        g2 = matchup_data.get((opp, agent), [])
        n1 = len(g1)
        n2 = len(g2)
        acc1 = sum(1 for g in g1 if g["accept"])
        acc2 = sum(1 for g in g2 if g["accept"])
        if n1 + n2 > 0:
            rate = (acc1 + acc2) / (n1 + n2)
            total_acc += acc1 + acc2
            total_games += n1 + n2
        else:
            rate = 0
        print(f" {rate*100:>7.1f}%", end="")
    overall = total_acc / total_games if total_games > 0 else 0
    print(f" {overall*100:>8.1f}%")


print("\n\n" + "=" * 100)
print("TABLE 8: NFSP EF1 frequency of outcomes vs each opponent")
print("         EF1 freq = (# EF1 accepts) / (# accepts)  [paper definition]")
print("=" * 100)

print(f"\n  {'Opponent':<16s} {'as P1':>10s} {'as P2':>10s} {'Symmetrized':>12s} "
      f"{'#Acc(P1)':>10s} {'#Acc(P2)':>10s}")
print(f"  {'-'*70}")

for opp in STRATEGIES:
    if opp == "nfsp":
        continue
    # NFSP as P1
    games_p1 = matchup_data.get(("nfsp", opp), [])
    acc_p1 = [g for g in games_p1 if g["accept"]]
    ef1_p1 = sum(1 for g in acc_p1 if g["ef1"]) / max(len(acc_p1), 1)
    # NFSP as P2
    games_p2 = matchup_data.get((opp, "nfsp"), [])
    acc_p2 = [g for g in games_p2 if g["accept"]]
    ef1_p2 = sum(1 for g in acc_p2 if g["ef1"]) / max(len(acc_p2), 1)
    # Symmetrized
    ef1_sym_val = 0.5 * (ef1_p1 + ef1_p2)
    print(f"  {opp:<16s} {ef1_p1*100:>9.1f}% {ef1_p2*100:>9.1f}% {ef1_sym_val*100:>11.1f}% "
          f"{len(acc_p1):>10,d} {len(acc_p2):>10,d}")

# Self-play: same EF1 outcome for both, but show both columns
games_self = matchup_data.get(("nfsp", "nfsp"), [])
acc_self = [g for g in games_self if g["accept"]]
ef1_self = sum(1 for g in acc_self if g["ef1"]) / max(len(acc_self), 1)
print(f"  {'nfsp (self)':<16s} {ef1_self*100:>9.1f}% {ef1_self*100:>9.1f}% {ef1_self*100:>11.1f}% "
      f"{len(acc_self):>10,d} {len(acc_self):>10,d}")


# ======================================================================
# TABLE 9: NFSP EF1 frequency of all offers made by NFSP
# For every counteroffer NFSP makes, decode the allocation and check EF1
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 9: NFSP EF1 frequency of all offers (counteroffers) vs each opponent")
print("         EF1 offer freq = (# EF1 offers by NFSP) / (# offers by NFSP)")
print("=" * 100)

print(f"\n  {'Opponent':<16s} {'as P1':>10s} {'as P2':>10s} {'Symmetrized':>12s} "
      f"{'#Offers(P1)':>12s} {'#Offers(P2)':>12s}")
print(f"  {'-'*75}")

for opp in STRATEGIES:
    if opp == "nfsp":
        continue

    # NFSP as P1: NFSP acts on even turns (0, 2, 4)
    raw_p1 = load_raw_games("nfsp", opp)
    n_offers_p1, n_ef1_p1 = 0, 0
    if raw_p1:
        for g in raw_p1:
            u1 = g["outcome"]["utilities_p1"]
            u2 = g["outcome"]["utilities_p2"]
            for turn_idx, action in enumerate(g["actions"]):
                if turn_idx % 2 == 0 and action < 80:  # NFSP's turn as P1
                    alloc = decode_action_to_alloc(action)
                    if alloc:
                        n_offers_p1 += 1
                        if is_ef1(u1, u2, alloc[0], alloc[1]):
                            n_ef1_p1 += 1

    # NFSP as P2: NFSP acts on odd turns (1, 3, 5)
    raw_p2 = load_raw_games(opp, "nfsp")
    n_offers_p2, n_ef1_p2 = 0, 0
    if raw_p2:
        for g in raw_p2:
            u1 = g["outcome"]["utilities_p1"]
            u2 = g["outcome"]["utilities_p2"]
            for turn_idx, action in enumerate(g["actions"]):
                if turn_idx % 2 == 1 and action < 80:  # NFSP's turn as P2
                    alloc = decode_action_to_alloc(action)
                    if alloc:
                        n_offers_p2 += 1
                        if is_ef1(u1, u2, alloc[0], alloc[1]):
                            n_ef1_p2 += 1

    ef1_rate_p1 = n_ef1_p1 / max(n_offers_p1, 1)
    ef1_rate_p2 = n_ef1_p2 / max(n_offers_p2, 1)
    ef1_rate_sym = 0.5 * (ef1_rate_p1 + ef1_rate_p2)

    print(f"  {opp:<16s} {ef1_rate_p1*100:>9.1f}% {ef1_rate_p2*100:>9.1f}% {ef1_rate_sym*100:>11.1f}% "
          f"{n_offers_p1:>12,d} {n_offers_p2:>12,d}")

# Self-play: split by P1 (even turns) and P2 (odd turns)
raw_self = load_raw_games("nfsp", "nfsp")
n_offers_self_p1, n_ef1_self_p1 = 0, 0
n_offers_self_p2, n_ef1_self_p2 = 0, 0
if raw_self:
    for g in raw_self:
        u1 = g["outcome"]["utilities_p1"]
        u2 = g["outcome"]["utilities_p2"]
        for turn_idx, action in enumerate(g["actions"]):
            if action < 80:
                alloc = decode_action_to_alloc(action)
                if alloc:
                    if turn_idx % 2 == 0:  # NFSP as P1
                        n_offers_self_p1 += 1
                        if is_ef1(u1, u2, alloc[0], alloc[1]):
                            n_ef1_self_p1 += 1
                    else:  # NFSP as P2
                        n_offers_self_p2 += 1
                        if is_ef1(u1, u2, alloc[0], alloc[1]):
                            n_ef1_self_p2 += 1
ef1_rate_self_p1 = n_ef1_self_p1 / max(n_offers_self_p1, 1)
ef1_rate_self_p2 = n_ef1_self_p2 / max(n_offers_self_p2, 1)
ef1_rate_self_sym = 0.5 * (ef1_rate_self_p1 + ef1_rate_self_p2)
print(f"  {'nfsp (self)':<16s} {ef1_rate_self_p1*100:>9.1f}% {ef1_rate_self_p2*100:>9.1f}% {ef1_rate_self_sym*100:>11.1f}% "
      f"{n_offers_self_p1:>12,d} {n_offers_self_p2:>12,d}")


# ======================================================================
# TABLE 10: Raw NW and NW+ conditioned on accepts only (raw pooled)
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 10: Raw NW and NW+ conditioned on accepts only (pooled across all opponents)")
print("          Not normalized by MAX_NW or MAX_NW_PLUS")
print("=" * 100)

print(f"\n  {'Agent':<16s} {'NW(acc)':>10s} {'NW(all)':>10s} {'NW+(acc)':>10s} {'NW+(all)':>10s} {'#Accepts':>10s} {'#Games':>10s}")
print(f"  {'-'*78}")

for agent in STRATEGIES:
    all_games = []
    for opp in STRATEGIES:
        for si, sj in [(agent, opp), (opp, agent)]:
            if si == sj == agent:
                games = matchup_data.get((si, sj), [])
                if games:
                    all_games.extend(games)
                continue
            games = matchup_data.get((si, sj))
            if games:
                all_games.extend(games)

    accepts = [g for g in all_games if g["accept"]]
    n_total = len(all_games)
    n_acc = len(accepts)

    nw_all = np.mean([g["nw"] for g in all_games]) if all_games else 0
    nwp_all = np.mean([g["nw_plus"] for g in all_games]) if all_games else 0
    nw_acc = np.mean([g["nw"] for g in accepts]) if accepts else 0
    nwp_acc = np.mean([g["nw_plus"] for g in accepts]) if accepts else 0

    print(f"  {agent:<16s} {nw_acc:>10.1f} {nw_all:>10.1f} {nwp_acc:>10.1f} {nwp_all:>10.1f} {n_acc:>10,d} {n_total:>10,d}")


# ======================================================================
# TABLE 11: Per-agent EF1 frequency — outcomes and offers (all agents)
# ======================================================================
print("\n\n" + "=" * 100)
print("TABLE 11: Per-agent EF1 frequency — outcomes vs offers (all agents)")
print("          Outcome EF1 = EF1 accepts / accepts")
print("          Offer EF1   = EF1 offers / total offers by agent")
print("=" * 100)

print(f"\n  {'Agent':<16s} {'EF1 Outcome':>12s} {'EF1 Offers':>12s} {'#Accepts':>10s} {'#Offers':>10s}")
print(f"  {'-'*62}")

for agent in STRATEGIES:
    # Outcome EF1: pool all matchups for this agent
    total_acc, total_ef1_acc = 0, 0
    # Offer EF1: pool all offers by this agent
    total_offers, total_ef1_offers = 0, 0

    for opp in STRATEGIES:
        # Agent as P1
        games_p1 = matchup_data.get((agent, opp), [])
        acc_p1 = [g for g in games_p1 if g["accept"]]
        total_acc += len(acc_p1)
        total_ef1_acc += sum(1 for g in acc_p1 if g["ef1"])

        raw_p1 = load_raw_games(agent, opp)
        if raw_p1:
            for g in raw_p1:
                u1 = g["outcome"]["utilities_p1"]
                u2 = g["outcome"]["utilities_p2"]
                for turn_idx, action in enumerate(g["actions"]):
                    if turn_idx % 2 == 0 and action < 80:
                        alloc = decode_action_to_alloc(action)
                        if alloc:
                            total_offers += 1
                            if is_ef1(u1, u2, alloc[0], alloc[1]):
                                total_ef1_offers += 1

        # Agent as P2
        games_p2 = matchup_data.get((opp, agent), [])
        acc_p2 = [g for g in games_p2 if g["accept"]]
        total_acc += len(acc_p2)
        total_ef1_acc += sum(1 for g in acc_p2 if g["ef1"])

        raw_p2 = load_raw_games(opp, agent)
        if raw_p2:
            for g in raw_p2:
                u1 = g["outcome"]["utilities_p1"]
                u2 = g["outcome"]["utilities_p2"]
                for turn_idx, action in enumerate(g["actions"]):
                    if turn_idx % 2 == 1 and action < 80:
                        alloc = decode_action_to_alloc(action)
                        if alloc:
                            total_offers += 1
                            if is_ef1(u1, u2, alloc[0], alloc[1]):
                                total_ef1_offers += 1

    ef1_outcome = total_ef1_acc / max(total_acc, 1)
    ef1_offer = total_ef1_offers / max(total_offers, 1)
    print(f"  {agent:<16s} {ef1_outcome*100:>11.1f}% {ef1_offer*100:>11.1f}% {total_acc:>10,d} {total_offers:>10,d}")
