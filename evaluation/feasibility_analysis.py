#!/usr/bin/env python3
"""
Compute theoretical deal feasibility rate.

For each randomly generated game, brute-force all 80 possible allocations
and check if ANY allocation beats both players' BATNAs simultaneously.

This gives the ceiling on accept rate — the best possible deal rate
with perfect play.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from simulator.python.bargain_env_cpu import (
    ITEM_QUANTITIES, NUM_ITEM_TYPES,
    NUM_COUNTEROFFER_ACTIONS, MIN_VALUE, MAX_VALUE,
)


def decode_action(action: int) -> np.ndarray:
    """Decode action index to offer array (items going to P2)."""
    offer = np.zeros(NUM_ITEM_TYPES, dtype=np.int8)
    offer[2] = action % 2
    action //= 2
    offer[1] = action % 5
    offer[0] = action // 5
    return offer


def precompute_allocations():
    """Precompute all 80 possible allocations."""
    allocations_p2 = np.zeros((NUM_COUNTEROFFER_ACTIONS, NUM_ITEM_TYPES), dtype=np.float32)
    allocations_p1 = np.zeros((NUM_COUNTEROFFER_ACTIONS, NUM_ITEM_TYPES), dtype=np.float32)

    for a in range(NUM_COUNTEROFFER_ACTIONS):
        offer = decode_action(a)
        allocations_p2[a] = offer
        allocations_p1[a] = ITEM_QUANTITIES - offer

    return allocations_p1, allocations_p2


def analyze_feasibility(num_samples: int = 100_000, seed: int = 42):
    """Analyze deal feasibility across random game settings."""
    rng = np.random.default_rng(seed)
    alloc_p1, alloc_p2 = precompute_allocations()

    # Generate game settings in bulk
    values = rng.integers(MIN_VALUE, MAX_VALUE + 1,
                          size=(num_samples, 2, NUM_ITEM_TYPES)).astype(np.float32)

    max_vals = np.sum(values * ITEM_QUANTITIES, axis=2)  # (N, 2)

    outside_options = np.zeros((num_samples, 2), dtype=np.float32)
    for p in range(2):
        outside_options[:, p] = rng.integers(
            1, max_vals[:, p].astype(np.int64) + 1
        ).astype(np.float32)

    norm_batnas = outside_options / max_vals  # (N, 2)

    # For each game, check all 80 allocations
    p1_deal_values = values[:, 0, :] @ alloc_p1.T  # (N, 80)
    p2_deal_values = values[:, 1, :] @ alloc_p2.T  # (N, 80)

    p1_deal_norm = p1_deal_values / max_vals[:, 0:1]
    p2_deal_norm = p2_deal_values / max_vals[:, 1:2]

    p1_beats = p1_deal_norm >= norm_batnas[:, 0:1]
    p2_beats = p2_deal_norm >= norm_batnas[:, 1:2]
    both_beat = p1_beats & p2_beats

    feasible = both_beat.any(axis=1)
    feasibility_rate = feasible.mean()

    uw_per_alloc = p1_deal_norm + p2_deal_norm
    uw_feasible = np.where(both_beat, uw_per_alloc, -np.inf)
    best_uw = uw_feasible.max(axis=1)

    surplus = (p1_deal_norm - norm_batnas[:, 0:1]) + (p2_deal_norm - norm_batnas[:, 1:2])
    surplus_feasible = np.where(both_beat, surplus, -np.inf)
    best_surplus = surplus_feasible.max(axis=1)

    # ---- Results ----
    print("=" * 60)
    print(f"DEAL FEASIBILITY ANALYSIS ({num_samples:,} games)")
    print("=" * 60)

    print(f"\nOverall feasibility rate: {feasibility_rate:.1%}")
    print(f"  Feasible games: {feasible.sum():,} / {num_samples:,}")
    print(f"  Infeasible games: {(~feasible).sum():,} / {num_samples:,}")

    print(f"\n--- Feasibility by BATNA ranges ---")
    batna_bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    for lo1, hi1 in batna_bins:
        for lo2, hi2 in batna_bins:
            mask = ((norm_batnas[:, 0] >= lo1) & (norm_batnas[:, 0] < hi1) &
                    (norm_batnas[:, 1] >= lo2) & (norm_batnas[:, 1] < hi2))
            n = mask.sum()
            if n > 0:
                rate = feasible[mask].mean()
                print(f"  P1 [{lo1:.2f},{hi1:.2f}) x P2 [{lo2:.2f},{hi2:.2f}): "
                      f"{rate:.1%}  (n={n:,})")

    print(f"\n--- Feasibility by max(BATNA) ---")
    max_batna = norm_batnas.max(axis=1)
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.6), (0.6, 0.7),
                   (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        mask = (max_batna >= lo) & (max_batna < hi)
        n = mask.sum()
        if n > 0:
            rate = feasible[mask].mean()
            print(f"  max BATNA [{lo:.2f},{hi:.2f}): {rate:.1%}  (n={n:,})")

    feasible_surplus = best_surplus[feasible]
    feasible_uw = best_uw[feasible]
    print(f"\n--- Among feasible games ---")
    print(f"  Best surplus mean: {feasible_surplus.mean():.3f}")
    print(f"  Best surplus median: {np.median(feasible_surplus):.3f}")
    print(f"  Best UW mean: {feasible_uw.mean():.3f}")
    print(f"  Best UW median: {np.median(feasible_uw):.3f}")

    num_feasible_allocs = both_beat[feasible].sum(axis=1)
    print(f"  Feasible allocations per game: mean={num_feasible_allocs.mean():.1f}, "
          f"median={np.median(num_feasible_allocs):.0f}")

    infeasible_mask = ~feasible
    if infeasible_mask.any():
        near_miss = best_surplus[infeasible_mask]
        print(f"\n--- Among infeasible games ---")
        print(f"  Best near-miss surplus mean: {near_miss.mean():.3f}")
        print(f"  Games within 0.05 of feasible: {(near_miss > -0.05).sum():,} "
              f"({(near_miss > -0.05).mean():.1%})")

    print(f"\n--- BATNA distribution ---")
    print(f"  Mean: P1={norm_batnas[:, 0].mean():.3f}, P2={norm_batnas[:, 1].mean():.3f}")
    print(f"  Sum > 1.0: {(norm_batnas.sum(axis=1) > 1.0).mean():.1%}")

    print(f"\n--- Comparison to actual crossplay ---")
    print(f"  Theoretical ceiling: {feasibility_rate:.1%}")
    print(f"  Actual accept rates:")
    print(f"    MAPPO: 42.2%")
    print(f"    PPO:   40.2%")
    print(f"    PSRO:  38.1%")
    print(f"    NFSP:  30.7%")
    gap = feasibility_rate - 0.402
    print(f"  Gap (ceiling - best actual): {gap:.1%}")


if __name__ == "__main__":
    analyze_feasibility(num_samples=100_000, seed=42)
