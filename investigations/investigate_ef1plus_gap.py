"""Investigate why ef1_bargainer has EF1 ~96% but EF1+ ~72%."""
import numpy as np
import orjson
from pathlib import Path

ITEM_QUANTITIES = np.array([7, 4, 1])

def is_ef1_bilateral(u1, u2, a1, a2):
    """Check bilateral EF1. Returns (both, p1_ok, p2_ok)."""
    p1_own = np.dot(u1, a1)
    p1_opp = np.dot(u1, a2)
    p1_rem = np.max(np.where(a2 > 0, u1, 0))
    p1_ok = p1_own >= p1_opp - p1_rem

    p2_own = np.dot(u2, a2)
    p2_opp = np.dot(u2, a1)
    p2_rem = np.max(np.where(a1 > 0, u2, 0))
    p2_ok = p2_own >= p2_opp - p2_rem

    return p1_ok and p2_ok, p1_ok, p2_ok

def exists_ef1_beating_batnas(u1, u2, batna1, batna2):
    """Check if any allocation is EF1 and beats both BATNAs."""
    for a0 in range(8):
        for a1_ in range(5):
            for a2_ in range(2):
                a1_alloc = np.array([a0, a1_, a2_], dtype=float)
                a2_alloc = ITEM_QUANTITIES - a1_alloc

                pay1 = np.dot(u1, a1_alloc)
                pay2 = np.dot(u2, a2_alloc)
                if pay1 <= batna1 or pay2 <= batna2:
                    continue

                # Check bilateral EF1
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

def load_json(path):
    with open(path, 'rb') as f:
        return orjson.loads(f.read())

crossplay_dir = Path("data/crossplay")

# Get all matchups involving ef1_bargainer
strategies = ["walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro", "ef1_bargainer"]

for strat in strategies:
    for role in ["p1", "p2"]:
        if role == "p1":
            matchup = f"ef1_bargainer_p1_vs_{strat}_p2"
            ef1_role = "P1"
            opp = strat
        else:
            matchup = f"{strat}_p1_vs_ef1_bargainer_p2"
            ef1_role = "P2"
            opp = strat

        games_path = crossplay_dir / matchup / "games.json"
        if not games_path.exists():
            continue

        data = load_json(games_path)
        games = data["games"]

        n_total = len(games)
        n_accept = 0
        n_walk = 0

        # EF1 counters
        n_ef1_both = 0
        n_ef1_p1_only = 0
        n_ef1_p2_only = 0
        n_ef1_neither = 0

        # BATNA counters (among accepts)
        n_both_beat_batna = 0
        n_p1_below_batna = 0
        n_p2_below_batna = 0
        n_both_below_batna = 0

        # EF1+ components
        n_rational = 0  # EF1+ denominator: accept & ef1_rational_exists
        n_ef1plus = 0   # EF1+ numerator: accept & ef1 & beats_both & rational

        # Failure breakdown among rational accepted games
        n_rational_ef1_fail = 0        # rational accept, but not bilateral EF1
        n_rational_batna_fail = 0      # rational accept, bilateral EF1, but BATNA fail
        n_rational_both_fail = 0       # rational accept, both fail

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

            # EF1 check
            ef1_both, ef1_p1, ef1_p2 = is_ef1_bilateral(u1, u2, a1, a2)
            if ef1_both:
                n_ef1_both += 1
            elif ef1_p1:
                n_ef1_p1_only += 1
            elif ef1_p2:
                n_ef1_p2_only += 1
            else:
                n_ef1_neither += 1

            # BATNA check (strict >)
            p1_beats = raw_pay1 > raw_batna1
            p2_beats = raw_pay2 > raw_batna2
            both_beat = p1_beats and p2_beats

            if both_beat:
                n_both_beat_batna += 1
            elif not p1_beats and not p2_beats:
                n_both_below_batna += 1
            elif not p1_beats:
                n_p1_below_batna += 1
            else:
                n_p2_below_batna += 1

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

        print(f"\n{'='*70}")
        print(f"{matchup}  (ef1_bargainer is {ef1_role}, opponent is {opp})")
        print(f"{'='*70}")
        print(f"Total: {n_total}, Accepts: {n_accept}, Walks: {n_walk} ({n_walk/n_total*100:.1f}%)")

        if n_accept == 0:
            print("  No accepted games.")
            continue

        print(f"\nEF1 breakdown (among {n_accept} accepts):")
        print(f"  Bilateral EF1:  {n_ef1_both:6d} ({n_ef1_both/n_accept*100:6.2f}%)")
        print(f"  P1 EF1 only:    {n_ef1_p1_only:6d} ({n_ef1_p1_only/n_accept*100:6.2f}%)")
        print(f"  P2 EF1 only:    {n_ef1_p2_only:6d} ({n_ef1_p2_only/n_accept*100:6.2f}%)")
        print(f"  Neither EF1:    {n_ef1_neither:6d} ({n_ef1_neither/n_accept*100:6.2f}%)")

        print(f"\nBATNA breakdown (among {n_accept} accepts, strict >):")
        print(f"  Both beat BATNA:   {n_both_beat_batna:6d} ({n_both_beat_batna/n_accept*100:6.2f}%)")
        print(f"  P1 below only:     {n_p1_below_batna:6d} ({n_p1_below_batna/n_accept*100:6.2f}%)")
        print(f"  P2 below only:     {n_p2_below_batna:6d} ({n_p2_below_batna/n_accept*100:6.2f}%)")
        print(f"  Both below:        {n_both_below_batna:6d} ({n_both_below_batna/n_accept*100:6.2f}%)")

        print(f"\nEF1+ (among {n_accept} accepts):")
        print(f"  Rational games (denominator): {n_rational:6d} ({n_rational/n_accept*100:6.2f}%)")
        if n_rational > 0:
            print(f"  EF1+ passes (numerator):      {n_ef1plus:6d} ({n_ef1plus/n_rational*100:6.2f}% of rational)")
            print(f"  Failures in rational games:")
            print(f"    EF1 fail only:    {n_rational_ef1_fail:6d} ({n_rational_ef1_fail/n_rational*100:6.2f}%)")
            print(f"    BATNA fail only:  {n_rational_batna_fail:6d} ({n_rational_batna_fail/n_rational*100:6.2f}%)")
            print(f"    Both fail:        {n_rational_both_fail:6d} ({n_rational_both_fail/n_rational*100:6.2f}%)")
