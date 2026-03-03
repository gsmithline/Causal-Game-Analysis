"""Print summary tables from PSIE bootstrap results.

Usage:
    python evaluation/print_psie_tables.py
    python evaluation/print_psie_tables.py --input data/analysis/psie_results.pkl
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def ci(arr):
    """Return mean, 2.5th, 97.5th percentile."""
    a = np.array(arr)
    return np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5)


def print_tables(path):
    with open(path, "rb") as f:
        d = pickle.load(f)

    strategy_names = d["config"]["strategy_names"]
    analyses = d["bootstrap"]["analyses"]
    n_bs = len(analyses)

    # =================================================================
    # TABLE 1: Basin Membership
    # =================================================================
    print("=" * 80)
    print("TABLE 1: Basin Membership (% of bootstrap samples)")
    print("=" * 80)
    print(f"  {'Seed':20s} {'PPO-only':>10s} {'PSRO-only':>10s} {'Both':>10s} {'Neither':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for seed_idx, name in enumerate(strategy_names):
        ppo_only = psro_only = both = neither = 0
        for a in analyses:
            t = a["terminal_support_per_seed"][seed_idx]
            has_ppo = "ppo" in t
            has_psro = "psro" in t
            if has_ppo and has_psro:
                both += 1
            elif has_ppo:
                ppo_only += 1
            elif has_psro:
                psro_only += 1
            else:
                neither += 1
        print(
            f"  {name:20s} {ppo_only/n_bs:>9.0%} {psro_only/n_bs:>10.0%} "
            f"{both/n_bs:>9.0%} {neither/n_bs:>10.0%}"
        )

    # =================================================================
    # TABLE 2: Terminal Welfare by Seed
    # =================================================================
    print()
    print("=" * 80)
    print("TABLE 2: Terminal Welfare by Seed (mean [95% CI])")
    print("=" * 80)
    print(f"  {'Seed':20s} {'UW':>22s} {'NW':>22s} {'EF1':>22s}")
    print(f"  {'-'*20} {'-'*22} {'-'*22} {'-'*22}")
    for seed_idx, name in enumerate(strategy_names):
        welfares = {"uw": [], "nw": [], "ef1": []}
        for a in analyses:
            m = a["traces"][seed_idx]["trace"][-1]["metrics"]
            for k in welfares:
                welfares[k].append(m[k])
        parts = []
        for k in ["uw", "nw", "ef1"]:
            mean, lo, hi = ci(welfares[k])
            if k == "ef1":
                parts.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                parts.append(f"{mean:.1f} [{lo:.1f}, {hi:.1f}]")
        print(f"  {name:20s} {parts[0]:>22s} {parts[1]:>22s} {parts[2]:>22s}")

    # =================================================================
    # TABLE 3: Seed Entry Mass at Terminal
    # =================================================================
    print()
    print("=" * 80)
    print("TABLE 3: Seed Entry Mass at Terminal")
    print("=" * 80)
    print(
        f"  {'Seed':20s} {'% nonzero':>10s} {'Mean mass':>10s} "
        f"{'Mean|>0':>10s} {'[95% CI]':>20s}"
    )
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")
    for seed_idx, name in enumerate(strategy_names):
        masses = []
        for a in analyses:
            terminal = a["traces"][seed_idx]["trace"][-1]
            io = terminal.get("in_out")
            if io is not None:
                masses.append(io["entry_mass"])
            else:
                masses.append(1.0)
        arr = np.array(masses)
        nonzero = np.sum(arr > 1e-6)
        mean, lo, hi = ci(masses)
        if nonzero > 0:
            mean_nz = np.mean(arr[arr > 1e-6])
            print(
                f"  {name:20s} {nonzero/n_bs:>9.0%} {mean:>10.3f} {mean_nz:>10.3f} "
                f"[{lo:.3f}, {hi:.3f}]"
            )
        else:
            print(
                f"  {name:20s} {0:>9.0%} {mean:>10.3f} {'---':>10s} "
                f"[{lo:.3f}, {hi:.3f}]"
            )

    # =================================================================
    # TABLE 4: Step 1 Marginal Contribution
    # =================================================================
    print()
    print("=" * 80)
    print("TABLE 4: Step 1 Marginal Contribution (in/out delta, mean [95% CI])")
    print("=" * 80)
    print(
        f"  {'Seed':20s} {'dUW':>22s} {'dNW':>22s} "
        f"{'dEF1':>24s} {'mass>0':>8s}"
    )
    print(f"  {'-'*20} {'-'*22} {'-'*22} {'-'*24} {'-'*8}")
    for seed_idx, name in enumerate(strategy_names):
        deltas = {"uw": [], "nw": [], "ef1": []}
        entry_masses = []
        for a in analyses:
            trace = a["traces"][seed_idx]["trace"]
            if len(trace) > 1:
                io = trace[1].get("in_out")
                if io is not None:
                    for k in deltas:
                        deltas[k].append(io["delta"][k])
                    entry_masses.append(io["entry_mass"])
        if deltas["uw"]:
            n = len(deltas["uw"])
            nonzero = sum(1 for x in entry_masses if x > 1e-6)
            parts = []
            for k in ["uw", "nw"]:
                mean, lo, hi = ci(deltas[k])
                parts.append(f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            mean, lo, hi = ci(deltas["ef1"])
            parts.append(f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            print(
                f"  {name:20s} {parts[0]:>22s} {parts[1]:>22s} "
                f"{parts[2]:>24s} {nonzero:>3d}/{n:<3d}"
            )
        else:
            print(f"  {name:20s} {'(singleton)':>22s}")

    # =================================================================
    # TABLE 4b: Terminal Marginal Contribution
    # =================================================================
    print()
    print("=" * 80)
    print("TABLE 4b: Terminal Marginal Contribution (in/out delta, mean [95% CI])")
    print("=" * 80)
    print(
        f"  {'Seed':20s} {'dUW':>22s} {'dNW':>22s} "
        f"{'dEF1':>24s} {'mass>0':>8s}"
    )
    print(f"  {'-'*20} {'-'*22} {'-'*22} {'-'*24} {'-'*8}")
    for seed_idx, name in enumerate(strategy_names):
        deltas = {"uw": [], "nw": [], "ef1": []}
        entry_masses = []
        n_singleton = 0
        for a in analyses:
            trace = a["traces"][seed_idx]["trace"]
            terminal = trace[-1]
            io = terminal.get("in_out")
            if io is not None:
                for k in deltas:
                    deltas[k].append(io["delta"][k])
                entry_masses.append(io["entry_mass"])
            else:
                n_singleton += 1
        if deltas["uw"]:
            n = len(deltas["uw"])
            nonzero = sum(1 for x in entry_masses if x > 1e-6)
            parts = []
            for k in ["uw", "nw"]:
                mean, lo, hi = ci(deltas[k])
                parts.append(f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            mean, lo, hi = ci(deltas["ef1"])
            parts.append(f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            suffix = f"{nonzero:>3d}/{n:<3d}"
            if n_singleton > 0:
                suffix += f" (+{n_singleton} single)"
            print(
                f"  {name:20s} {parts[0]:>22s} {parts[1]:>22s} "
                f"{parts[2]:>24s} {suffix}"
            )
        else:
            print(f"  {name:20s} {'(always singleton)':>22s}")

    # =================================================================
    # TABLE 5: Most Common Terminal per Seed
    # =================================================================
    print()
    print("=" * 80)
    print("TABLE 5: Most Common Terminal Support per Seed")
    print("=" * 80)
    for seed_idx, name in enumerate(strategy_names):
        counter = Counter()
        for a in analyses:
            counter[a["terminal_support_per_seed"][seed_idx]] += 1
        tops = counter.most_common(3)
        first = True
        for support, count in tops:
            label = name if first else ""
            support_str = ", ".join(support)
            print(
                f"  {label:20s} {{{support_str:50s}}} "
                f"{count:>3d}/{n_bs} = {count/n_bs:>4.0%}"
            )
            first = False

    # =================================================================
    # TABLE 6: Trajectory Crossings
    # =================================================================
    print()
    print("=" * 80)
    print("TABLE 6: Trajectory Crossing Pairs")
    print("=" * 80)
    pair_counter = Counter()
    for a in analyses:
        for c in a["crossings"]:
            seeds = [v[0] for v in c["visitors"]]
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    pair = tuple(sorted([seeds[i], seeds[j]]))
                    pair_counter[pair] += 1
    crossing_rate = sum(1 for a in analyses if a["crossings"]) / n_bs
    print(f"  Samples with any crossing: {crossing_rate:.0%}")
    print()
    print(f"  {'Seed A':>20s}   {'Seed B':>20s}   {'Freq':>12s}")
    print(f"  {'-'*20}   {'-'*20}   {'-'*12}")
    for (a, b), count in pair_counter.most_common(10):
        print(f"  {a:>20s}   {b:>20s}   {count:>3d}/{n_bs} = {count/n_bs:>3.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print PSIE summary tables")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to psie_results.pkl",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    path = Path(args.input) if args.input else (
        base_dir / "data" / "analysis" / "psie_results.pkl"
    )
    print_tables(path)
