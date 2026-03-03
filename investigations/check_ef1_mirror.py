"""Check EF1 for every game in ef1_bargainer vs ef1_bargainer."""
import numpy as np
import orjson
from pathlib import Path

ITEM_QUANTITIES = np.array([7, 4, 1])

def is_ef1_single(utilities, my_alloc, opp_alloc):
    """Check EF1 from one player's perspective. Returns (bool, own_val, opp_val, max_removal)."""
    own_val = np.dot(utilities, my_alloc)
    opp_val = np.dot(utilities, opp_alloc)
    max_removal = np.max(np.where(np.array(opp_alloc) > 0, utilities, 0))
    is_ef1 = own_val >= opp_val - max_removal
    return bool(is_ef1), float(own_val), float(opp_val), float(max_removal)

def exists_ef1_beating_batnas(u1, u2, batna1, batna2):
    """Check if any allocation is EF1 and strictly beats both BATNAs."""
    for a0 in range(8):
        for a1_ in range(5):
            for a2_ in range(2):
                a1_alloc = np.array([a0, a1_, a2_], dtype=float)
                a2_alloc = ITEM_QUANTITIES - a1_alloc
                pay1 = np.dot(u1, a1_alloc)
                pay2 = np.dot(u2, a2_alloc)
                if pay1 <= batna1 - 1e-4 or pay2 <= batna2 - 1e-4:
                    continue
                p1_opp = np.dot(u1, a2_alloc)
                p1_rem = np.max(np.where(a2_alloc > 0, u1, 0))
                if pay1 < p1_opp - p1_rem:
                    continue
                p2_opp = np.dot(u2, a1_alloc)
                p2_rem = np.max(np.where(a1_alloc > 0, u2, 0))
                if pay2 < p2_opp - p2_rem:
                    continue
                return True
    return False


games_path = Path("data/crossplay/ef1_bargainer_p1_vs_ef1_bargainer_p2/games.json")
with open(games_path, 'rb') as f:
    data = orjson.loads(f.read())

games = data["games"]
n_accept = 0
n_walk = 0
n_ef1_both = 0
n_ef1_p1_only = 0
n_ef1_p2_only = 0
n_ef1_neither = 0
failures = []

# BATNA counters
n_both_beat_batna = 0
n_p1_below = 0
n_p2_below = 0
n_both_below = 0

# EF1+ counters
n_rational = 0
n_ef1plus = 0
n_rational_ef1_fail = 0
n_rational_batna_fail = 0
n_rational_both_fail = 0

for game in games:
    outcome = game["outcome"]
    if outcome["result"] != "accept":
        n_walk += 1
        continue

    n_accept += 1
    u1 = np.array(outcome["utilities_p1"], dtype=float)
    u2 = np.array(outcome["utilities_p2"], dtype=float)
    a1 = np.array(outcome["allocation_p1"], dtype=float)
    a2 = np.array(outcome["allocation_p2"], dtype=float)

    max1 = np.dot(u1, ITEM_QUANTITIES)
    max2 = np.dot(u2, ITEM_QUANTITIES)
    raw_pay1 = outcome["payoff_p1"] * max1
    raw_pay2 = outcome["payoff_p2"] * max2
    raw_batna1 = outcome["batna_p1"] * max1
    raw_batna2 = outcome["batna_p2"] * max2

    # Check allocation sums to item quantities
    alloc_sum = a1 + a2
    if not np.allclose(alloc_sum, ITEM_QUANTITIES):
        print(f"  WARNING game {game['game_id']}: allocation doesn't sum to quantities: {a1} + {a2} = {alloc_sum}")

    p1_ok, p1_own, p1_opp, p1_rem = is_ef1_single(u1, a1, a2)
    p2_ok, p2_own, p2_opp, p2_rem = is_ef1_single(u2, a2, a1)

    ef1_both = p1_ok and p2_ok
    if ef1_both:
        n_ef1_both += 1
    elif p1_ok and not p2_ok:
        n_ef1_p1_only += 1
        failures.append(("P2_fail", game["game_id"], u1, u2, a1, a2, p2_own, p2_opp, p2_rem))
    elif not p1_ok and p2_ok:
        n_ef1_p2_only += 1
        failures.append(("P1_fail", game["game_id"], u1, u2, a1, a2, p1_own, p1_opp, p1_rem))
    else:
        n_ef1_neither += 1
        failures.append(("both_fail", game["game_id"], u1, u2, a1, a2, p1_own, p1_opp, p1_rem))

    # BATNA check (with float tolerance)
    p1_beats = raw_pay1 > raw_batna1 - 1e-4
    p2_beats = raw_pay2 > raw_batna2 - 1e-4
    both_beat = p1_beats and p2_beats

    if both_beat:
        n_both_beat_batna += 1
    elif not p1_beats and not p2_beats:
        n_both_below += 1
    elif not p1_beats:
        n_p1_below += 1
    else:
        n_p2_below += 1

    # EF1+ rational check
    rational = exists_ef1_beating_batnas(u1, u2, raw_batna1, raw_batna2)
    if rational:
        n_rational += 1
        if ef1_both and both_beat:
            n_ef1plus += 1
        elif not ef1_both and not both_beat:
            n_rational_both_fail += 1
        elif not ef1_both:
            n_rational_ef1_fail += 1
        else:
            n_rational_batna_fail += 1

print(f"Total games: {len(games)}")
print(f"  Walks: {n_walk}")
print(f"  Accepts: {n_accept}")
print(f"  Walk rate: {n_walk/len(games)*100:.1f}%")
print()
print(f"Among accepted games:")
print(f"  Bilateral EF1: {n_ef1_both} ({n_ef1_both/n_accept*100:.2f}%)")
print(f"  P1 EF1 only:   {n_ef1_p1_only} ({n_ef1_p1_only/n_accept*100:.2f}%)")
print(f"  P2 EF1 only:   {n_ef1_p2_only} ({n_ef1_p2_only/n_accept*100:.2f}%)")
print(f"  Neither EF1:    {n_ef1_neither} ({n_ef1_neither/n_accept*100:.2f}%)")

print(f"\nBATNA breakdown (among {n_accept} accepts, strict >):")
print(f"  Both beat BATNA:  {n_both_beat_batna:6d} ({n_both_beat_batna/n_accept*100:.2f}%)")
print(f"  P1 below only:    {n_p1_below:6d} ({n_p1_below/n_accept*100:.2f}%)")
print(f"  P2 below only:    {n_p2_below:6d} ({n_p2_below/n_accept*100:.2f}%)")
print(f"  Both below:       {n_both_below:6d} ({n_both_below/n_accept*100:.2f}%)")

print(f"\nEF1+ breakdown:")
print(f"  Rational games (denom):  {n_rational:6d} ({n_rational/n_accept*100:.2f}% of accepts)")
if n_rational > 0:
    print(f"  EF1+ passes (numer):     {n_ef1plus:6d} ({n_ef1plus/n_rational*100:.2f}% of rational)")
    print(f"  Failures in rational games:")
    print(f"    EF1 fail only:   {n_rational_ef1_fail:6d} ({n_rational_ef1_fail/n_rational*100:.2f}%)")
    print(f"    BATNA fail only: {n_rational_batna_fail:6d} ({n_rational_batna_fail/n_rational*100:.2f}%)")
    print(f"    Both fail:       {n_rational_both_fail:6d} ({n_rational_both_fail/n_rational*100:.2f}%)")

if failures:
    print(f"\n--- First 10 EF1 failures ---")
    for f in failures[:10]:
        kind, gid, u1, u2, a1, a2, own, opp, rem = f
        print(f"\nGame {gid} [{kind}]:")
        print(f"  P1 utilities: {u1}, P2 utilities: {u2}")
        print(f"  P1 allocation: {a1}, P2 allocation: {a2}")
        print(f"  Failing player: own_val={own:.1f}, opp_val={opp:.1f}, max_removal={rem:.1f}")
        print(f"  Check: {own:.1f} >= {opp:.1f} - {rem:.1f} = {opp - rem:.1f}? {own >= opp - rem}")

# Print details of BATNA failures
print(f"\n--- BATNA failure details ---")
batna_fail_count = 0
for game in games:
    outcome = game["outcome"]
    if outcome["result"] != "accept":
        continue
    u1 = np.array(outcome["utilities_p1"], dtype=float)
    u2 = np.array(outcome["utilities_p2"], dtype=float)
    a1 = np.array(outcome["allocation_p1"], dtype=float)
    a2 = np.array(outcome["allocation_p2"], dtype=float)
    max1 = np.dot(u1, ITEM_QUANTITIES)
    max2 = np.dot(u2, ITEM_QUANTITIES)
    raw_pay1 = outcome["payoff_p1"] * max1
    raw_pay2 = outcome["payoff_p2"] * max2
    raw_batna1 = outcome["batna_p1"] * max1
    raw_batna2 = outcome["batna_p2"] * max2

    if not (raw_pay1 > raw_batna1 and raw_pay2 > raw_batna2):
        batna_fail_count += 1
        # Compute what the agent would have seen
        agent_val_p1 = np.dot(u1, a1)
        agent_batna_p1 = outcome["batna_p1"] * max1
        actions = game.get("actions", [])
        last_action = actions[-1] if actions else None
        who_accepted = "P2 accepted" if last_action == 80 else f"last action={last_action}"
        print(f"\nGame {game['game_id']}:")
        print(f"  Actions: {actions} ({who_accepted})")
        print(f"  P1 utils: {u1}, P2 utils: {u2}")
        print(f"  P1 alloc: {a1}, P2 alloc: {a2}")
        print(f"  raw_pay1={raw_pay1:.6f}, raw_batna1={raw_batna1:.6f}, diff={raw_pay1-raw_batna1:.10f}")
        print(f"  raw_pay2={raw_pay2:.6f}, raw_batna2={raw_batna2:.6f}, diff={raw_pay2-raw_batna2:.10f}")
        print(f"  payoff_p1 (normalized)={outcome['payoff_p1']:.10f}, batna_p1={outcome['batna_p1']:.10f}")
        print(f"  Agent's view: val={agent_val_p1:.6f}, batna={agent_batna_p1:.6f}, diff={agent_val_p1-agent_batna_p1:.10f}")
