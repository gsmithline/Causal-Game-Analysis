"""Check if BATNA values are the same across matchups involving walk."""
import orjson
import numpy as np
from pathlib import Path

CROSSPLAY = Path("data/crossplay")

def load_games(si, sj):
    p = CROSSPLAY / f"{si}_p1_vs_{sj}_p2" / "games.json"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return orjson.loads(f.read())["games"]

ITEM_QUANTITIES = np.array([7, 4, 1])

# Check: are the game settings (utilities) the same across matchups?
# Compare first 5 games from walk_vs_walk, walk_vs_nfsp, nfsp_vs_walk
matchups = [
    ("walk", "walk"),
    ("walk", "nfsp"),
    ("nfsp", "walk"),
    ("nfsp", "nfsp"),
]

print("=== First 3 games: utilities and BATNAs across matchups ===\n")
for si, sj in matchups:
    games = load_games(si, sj)
    print(f"{si}_p1_vs_{sj}_p2: {len(games)} games")
    for i in range(3):
        g = games[i]["outcome"]
        u1 = g["utilities_p1"]
        u2 = g["utilities_p2"]
        b1 = g["batna_p1"]
        b2 = g["batna_p2"]
        max1 = sum(a*b for a,b in zip(u1, ITEM_QUANTITIES))
        max2 = sum(a*b for a,b in zip(u2, ITEM_QUANTITIES))
        print(f"  game {i}: u1={u1}, u2={u2}, batna_p1={b1:.4f} (raw={b1*max1:.2f}), batna_p2={b2:.4f} (raw={b2*max2:.2f}), result={g['result']}")
    print()

# Compare mean BATNA across all walk matchups (before bootstrap)
print("=== Mean raw BATNA for P1 across walk matchups (no bootstrap) ===\n")
strategies = ["walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro", "openai_5.2_none"]

for sj in strategies:
    games = load_games("walk", sj)
    if games is None:
        continue
    batnas = []
    for g in games:
        o = g["outcome"]
        u1 = np.array(o["utilities_p1"])
        max1 = np.sum(u1 * ITEM_QUANTITIES)
        batnas.append(o["batna_p1"] * max1)
    print(f"  walk_p1 vs {sj:16s}_p2: mean_batna_p1 = {np.mean(batnas):.6f}, n={len(batnas)}")

print()
print("=== Mean raw BATNA for P2 across walk matchups ===\n")
for si in strategies:
    games = load_games(si, "walk")
    if games is None:
        continue
    batnas = []
    for g in games:
        o = g["outcome"]
        u2 = np.array(o["utilities_p2"])
        max2 = np.sum(u2 * ITEM_QUANTITIES)
        batnas.append(o["batna_p2"] * max2)
    print(f"  {si:16s}_p1 vs walk_p2: mean_batna_p2 = {np.mean(batnas):.6f}, n={len(batnas)}")

# Also check: walk's actual payoff (should = BATNA when walk walks)
print()
print("=== Walk's actual payoff vs BATNA (should be equal) ===\n")
for sj in strategies:
    games = load_games("walk", sj)
    if games is None:
        continue
    payoffs = []
    batnas = []
    for g in games:
        o = g["outcome"]
        u1 = np.array(o["utilities_p1"])
        max1 = np.sum(u1 * ITEM_QUANTITIES)
        payoffs.append(o["payoff_p1"] * max1)
        batnas.append(o["batna_p1"] * max1)
    print(f"  walk vs {sj:16s}: mean_payoff={np.mean(payoffs):.6f}, mean_batna={np.mean(batnas):.6f}, diff={np.mean(payoffs)-np.mean(batnas):.6f}")
