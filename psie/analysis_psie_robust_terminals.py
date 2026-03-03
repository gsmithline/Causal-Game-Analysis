"""PSIE robust terminal support analysis.

For each seed, collect terminal supports across all 1000 bootstrap samples,
filter to those appearing >= 20% of the time, and analyze patterns.
"""

import pickle
from collections import Counter, defaultdict

# ── Load data ──────────────────────────────────────────────────────────────────
with open("data/analysis/psie_results.pkl", "rb") as f:
    data = pickle.load(f)

strategy_names = data["config"]["strategy_names"]
analyses = data["bootstrap"]["analyses"]
n_bootstrap = len(analyses)
n_seeds = len(strategy_names)
THRESHOLD = 0.20

print(f"Strategies ({n_seeds}): {strategy_names}")
print(f"Bootstrap samples: {n_bootstrap}")
print(f"Frequency threshold: {THRESHOLD:.0%}")

# ── Reconstruct per-seed terminal supports for every bootstrap sample ─────────
# entry_order[name][seed_idx] is the step the strategy entered (None = never entered)
# A strategy is in the terminal support of seed i iff entry_order[name][i] is not None.

# seed_idx -> list of terminal support tuples (one per bootstrap sample)
seed_terminals = defaultdict(list)

for a in analyses:
    for seed_idx in range(n_seeds):
        terminal = []
        for name in strategy_names:
            if a["entry_order"][name][seed_idx] is not None:
                terminal.append(name)
        seed_terminals[seed_idx].append(tuple(sorted(terminal)))

# ── For each seed, count frequencies and filter >= 20% ────────────────────────
print("\n" + "=" * 75)
print("SECTION 1: ROBUST TERMINAL SUPPORTS PER SEED (>= 20% frequency)")
print("=" * 75)

seed_robust = {}  # seed_idx -> {terminal_tuple: frequency}
for seed_idx in range(n_seeds):
    counter = Counter(seed_terminals[seed_idx])
    robust = {}
    for terminal, count in counter.most_common():
        freq = count / n_bootstrap
        if freq >= THRESHOLD:
            robust[terminal] = freq
    seed_robust[seed_idx] = robust

    seed_name = strategy_names[seed_idx]
    n_distinct = len(counter)
    print(f"\nSeed: {seed_name}  ({n_distinct} distinct terminals across {n_bootstrap} samples)")
    if robust:
        for terminal, freq in sorted(robust.items(), key=lambda x: -x[1]):
            print(f"  {freq:6.1%}  {{{', '.join(terminal)}}}")
    else:
        print("  (no terminal exceeds 20% threshold)")
        # Show top 3 anyway
        for terminal, count in counter.most_common(3):
            print(f"  {count/n_bootstrap:6.1%}  {{{', '.join(terminal)}}}  (below threshold)")

# ── Summary statistics ─────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("SECTION 2: SUMMARY")
print("=" * 75)

seeds_with_robust = [i for i in range(n_seeds) if seed_robust[i]]
print(f"\nSeeds with at least one terminal >= 20%: {len(seeds_with_robust)}/{n_seeds}")
for i in seeds_with_robust:
    print(f"  {strategy_names[i]}: {len(seed_robust[i])} robust terminal(s)")

seeds_without = [i for i in range(n_seeds) if not seed_robust[i]]
if seeds_without:
    print(f"\nSeeds with NO terminal >= 20%: {len(seeds_without)}")
    for i in seeds_without:
        print(f"  {strategy_names[i]}")

# ── Collect all robust terminals ──────────────────────────────────────────────
print("\n" + "=" * 75)
print("SECTION 3: ROBUST TERMINAL CATALOG")
print("=" * 75)

all_robust_terminals = set()
for robust in seed_robust.values():
    all_robust_terminals.update(robust.keys())

print(f"\nTotal distinct robust terminals: {len(all_robust_terminals)}")
# For each robust terminal, which seeds produce it and at what frequency?
terminal_seed_map = defaultdict(dict)  # terminal -> {seed_name: freq}
for seed_idx in range(n_seeds):
    for terminal, freq in seed_robust[seed_idx].items():
        terminal_seed_map[terminal][strategy_names[seed_idx]] = freq

for terminal in sorted(all_robust_terminals, key=lambda t: -max(terminal_seed_map[t].values())):
    seeds_info = terminal_seed_map[terminal]
    print(f"\n  {{{', '.join(terminal)}}}  (size {len(terminal)})")
    for seed_name, freq in sorted(seeds_info.items(), key=lambda x: -x[1]):
        print(f"    seed={seed_name:20s}  freq={freq:.1%}")

# ── Basin structure: which seeds share robust terminals? ──────────────────────
print("\n" + "=" * 75)
print("SECTION 4: BASIN STRUCTURE (shared robust terminals)")
print("=" * 75)

for terminal in sorted(all_robust_terminals, key=lambda t: (-len(terminal_seed_map[t]), t)):
    seeds_info = terminal_seed_map[terminal]
    if len(seeds_info) > 1:
        print(f"\n  {{{', '.join(terminal)}}}")
        print(f"    Shared by {len(seeds_info)} seeds: {list(seeds_info.keys())}")
        print(f"    Frequencies: {', '.join(f'{s}={f:.1%}' for s, f in sorted(seeds_info.items(), key=lambda x: -x[1]))}")

# Check for seeds that always go to the same robust terminal
print("\n  Seed stability (seeds with a single dominant terminal >= 50%):")
for seed_idx in range(n_seeds):
    robust = seed_robust[seed_idx]
    if robust:
        top_terminal, top_freq = max(robust.items(), key=lambda x: x[1])
        if top_freq >= 0.50:
            print(f"    {strategy_names[seed_idx]:20s}: {top_freq:.1%} -> {{{', '.join(top_terminal)}}}")

# ── Strategy importance ───────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("SECTION 5: STRATEGY IMPORTANCE IN ROBUST TERMINALS")
print("=" * 75)

# How often does each strategy appear in robust terminals (weighted by frequency)?
# Two measures: (a) appears in how many distinct robust terminals, (b) frequency-weighted
strategy_in_robust_count = Counter()
strategy_in_robust_weighted = defaultdict(float)

for terminal in all_robust_terminals:
    for name in terminal:
        strategy_in_robust_count[name] += 1
    # Weighted: average frequency across seeds that produce this terminal
    avg_freq = sum(terminal_seed_map[terminal].values()) / n_seeds
    for name in terminal:
        strategy_in_robust_weighted[name] += avg_freq

print(f"\n  {'Strategy':20s}  {'In # robust terminals':>22s}  {'Freq-weighted importance':>24s}")
print(f"  {'-'*20}  {'-'*22}  {'-'*24}")
for name in strategy_names:
    count = strategy_in_robust_count.get(name, 0)
    weighted = strategy_in_robust_weighted.get(name, 0.0)
    total = len(all_robust_terminals)
    print(f"  {name:20s}  {count:>3d}/{total:<3d} ({count/total if total else 0:.0%}){' ':>10s}  {weighted:.3f}")

# ── Entry dynamics for robust terminals ───────────────────────────────────────
print("\n" + "=" * 75)
print("SECTION 6: ENTRY DYNAMICS FOR ROBUST TERMINALS")
print("=" * 75)

# For each robust terminal, reconstruct the typical expansion path
# We look at the entry_order data: for the seeds that produce this terminal,
# what is the typical step order?

for terminal in sorted(all_robust_terminals, key=lambda t: (-max(terminal_seed_map[t].values()), t)):
    seeds_info = terminal_seed_map[terminal]
    print(f"\n  Terminal: {{{', '.join(terminal)}}}")
    print(f"  Produced by seeds: {list(seeds_info.keys())}")

    # For each seed that produces this terminal, get the entry order from
    # bootstrap samples where that seed actually produces this terminal
    for seed_name, freq in sorted(seeds_info.items(), key=lambda x: -x[1]):
        seed_idx = strategy_names.index(seed_name)

        # Collect entry orders from samples where this seed -> this terminal
        entry_sequences = []
        for b_idx, a in enumerate(analyses):
            if seed_terminals[seed_idx][b_idx] == terminal:
                # Get entry order for this seed in this sample
                entry_steps = {}
                for name in strategy_names:
                    step = a["entry_order"][name][seed_idx]
                    if step is not None:
                        entry_steps[name] = step
                # Sort by step
                seq = sorted(entry_steps.items(), key=lambda x: x[1])
                entry_sequences.append(tuple(name for name, _ in seq))

        if entry_sequences:
            seq_counter = Counter(entry_sequences)
            most_common_seq, seq_count = seq_counter.most_common(1)[0]
            print(f"    From seed={seed_name} ({freq:.1%}):")
            print(f"      Most common path ({seq_count}/{len(entry_sequences)} = {seq_count/len(entry_sequences):.0%}): "
                  f"{' -> '.join(most_common_seq)}")
            # Show top 3 if diverse
            if len(seq_counter) > 1:
                print(f"      ({len(seq_counter)} distinct paths)")
                for seq, cnt in seq_counter.most_common(3):
                    print(f"        {cnt/len(entry_sequences):5.1%}  {' -> '.join(seq)}")

# ── Final summary table ──────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("SECTION 7: FINAL SUMMARY TABLE")
print("=" * 75)

print(f"\n  {'Seed':<20s} | {'# Robust':>8s} | {'Top Terminal':>45s} | {'Top Freq':>8s}")
print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*45}-+-{'-'*8}")
for seed_idx in range(n_seeds):
    seed_name = strategy_names[seed_idx]
    robust = seed_robust[seed_idx]
    if robust:
        top_terminal, top_freq = max(robust.items(), key=lambda x: x[1])
        top_str = "{" + ", ".join(top_terminal) + "}"
        if len(top_str) > 45:
            top_str = top_str[:42] + "..."
        print(f"  {seed_name:<20s} | {len(robust):>8d} | {top_str:>45s} | {top_freq:>7.1%}")
    else:
        print(f"  {seed_name:<20s} | {0:>8d} | {'(none above 20%)':>45s} | {'':>8s}")
