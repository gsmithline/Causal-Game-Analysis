"""
PSIE Results Pattern Analysis
Loads data/analysis/psie_results.pkl and produces a detailed report covering:
  1. Basin Structure
  2. Strategy Importance
  3. Expansion Dynamics
  4. Sensitivity / Robustness
"""

import pickle
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

# ── Load data ──────────────────────────────────────────────────────────────────
with open("data/analysis/psie_results.pkl", "rb") as f:
    data = pickle.load(f)

NAMES = data["config"]["strategy_names"]
N = len(NAMES)
name2idx = {n: i for i, n in enumerate(NAMES)}
traces = data["point_estimate"]["traces"]
pe_analysis = data["point_estimate"]["analysis"]
boot_analyses = data["bootstrap"]["analyses"]
agg = data["bootstrap"]["aggregated"]
N_BOOT = agg["n_samples"]

# ── Helper: reconstruct per-seed terminal support from an analysis dict ────────
def per_seed_terminals(analysis):
    """Return list of tuples (one per seed) of sorted strategy names in terminal."""
    terminals = []
    for seed_idx in range(N):
        members = []
        for strat_name in NAMES:
            if analysis["entry_order"][strat_name][seed_idx] is not None:
                members.append(strat_name)
        terminals.append(tuple(sorted(members)))
    return terminals


# ══════════════════════════════════════════════════════════════════════════════
#  1.  BASIN STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  SECTION 1: BASIN STRUCTURE")
print("=" * 80)

# 1a. Point-estimate terminal supports per seed
print("\n--- 1a. Point-Estimate Terminal Supports per Seed ---\n")
pe_terminals = []
for t in traces:
    ts = tuple(sorted(t["terminal_support"]))
    pe_terminals.append(ts)
    print(f"  Seed {t['seed_idx']:>2} ({t['seed']:<18s}): {', '.join(ts)}")

# 1b. Distinct terminal supports (point estimate)
unique_pe = sorted(set(pe_terminals))
print(f"\n  Number of distinct terminal supports (point estimate): {len(unique_pe)}")
for i, sup in enumerate(unique_pe):
    seeds_in = [NAMES[j] for j in range(N) if pe_terminals[j] == sup]
    print(f"    Basin {i+1}: {{{', '.join(sup)}}}  <-- seeds: {seeds_in}")

# 1c. Bootstrap: most common terminal per seed
print("\n--- 1b. Bootstrap: Most Common Terminal per Seed ---\n")
boot_seed_terminals = defaultdict(list)  # seed_idx -> list of terminal tuples
for ba in boot_analyses:
    terminals = per_seed_terminals(ba)
    for si in range(N):
        boot_seed_terminals[si].append(terminals[si])

for si in range(N):
    ctr = Counter(boot_seed_terminals[si])
    top5 = ctr.most_common(5)
    total = sum(ctr.values())
    print(f"  Seed {si} ({NAMES[si]}):")
    for sup, cnt in top5:
        print(f"    {cnt:>5}/{total}  ({100*cnt/total:5.1f}%)  {{{', '.join(sup)}}}")
    if len(ctr) > 5:
        print(f"    ... and {len(ctr)-5} more distinct terminals")
    print()

# 1d. Group seeds by most-common terminal
print("--- 1c. Seed Grouping by Most Common Terminal ---\n")
seed_modal_terminal = {}
for si in range(N):
    seed_modal_terminal[si] = Counter(boot_seed_terminals[si]).most_common(1)[0][0]

# Group
basin_groups = defaultdict(list)
for si in range(N):
    basin_groups[seed_modal_terminal[si]].append(NAMES[si])

for sup, seeds in sorted(basin_groups.items(), key=lambda x: -len(x[1])):
    print(f"  {{{', '.join(sup)}}} <-- seeds: {seeds}")
print(f"\n  Total distinct modal basins: {len(basin_groups)}")

# 1e. Co-occurrence matrix: how often two seeds end at the same terminal (bootstrap)
print("\n--- 1d. Seed-to-Seed Co-occurrence Matrix (% same terminal, bootstrap) ---\n")
cooccurrence = np.zeros((N, N))
for b_idx in range(N_BOOT):
    terminals = per_seed_terminals(boot_analyses[b_idx])
    for i in range(N):
        for j in range(i, N):
            if terminals[i] == terminals[j]:
                cooccurrence[i][j] += 1
                if i != j:
                    cooccurrence[j][i] += 1

cooccurrence_pct = 100 * cooccurrence / N_BOOT

# Print header
hdr = "".join(f"{NAMES[j][:6]:>8}" for j in range(N))
print(f"  {'':>18}{hdr}")
for i in range(N):
    row = "".join(f"{cooccurrence_pct[i][j]:7.1f}%" for j in range(N))
    print(f"  {NAMES[i]:<18}{row}")

# Highlight notable pairs (>20% co-occurrence, excluding diagonal)
print("\n  Notable co-occurring seed pairs (>20% same terminal):")
for i in range(N):
    for j in range(i + 1, N):
        if cooccurrence_pct[i][j] > 20:
            print(f"    {NAMES[i]} & {NAMES[j]}: {cooccurrence_pct[i][j]:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  2.  STRATEGY IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  SECTION 2: STRATEGY IMPORTANCE")
print("=" * 80)

# 2a. Point estimate: frequency in terminal supports
print("\n--- 2a. Point-Estimate: Strategy Frequency in Terminal Supports ---\n")
pe_freq = Counter()
for ts in pe_terminals:
    for s in ts:
        pe_freq[s] += 1
for name in NAMES:
    print(f"  {name:<20s}: appears in {pe_freq.get(name,0):>2}/{N} terminal supports")

# 2b. Bootstrap: per-strategy frequency across all seeds and samples
print("\n--- 2b. Bootstrap: Strategy Frequency in Terminal Supports ---\n")
print(f"  (Out of {N} seeds x {N_BOOT} bootstrap samples = {N*N_BOOT} total seed-runs)\n")

strat_boot_freq = Counter()
strat_boot_freq_per_seed = defaultdict(lambda: np.zeros(N_BOOT))
for b_idx, ba in enumerate(boot_analyses):
    terminals = per_seed_terminals(ba)
    for si in range(N):
        for s in terminals[si]:
            strat_boot_freq[s] += 1
            strat_boot_freq_per_seed[s][b_idx] += 1

total_runs = N * N_BOOT
print(f"  {'Strategy':<20s} {'Total':>7} {'%':>7}  {'Mean/sample':>12}  {'Never-in':>10}  {'Always-in':>10}")
for name in sorted(NAMES, key=lambda n: -strat_boot_freq[n]):
    freq = strat_boot_freq[name]
    pct = 100 * freq / total_runs
    mean_per = freq / N_BOOT
    nev = agg["never_entered_freq"].get(name, 0)
    alw = agg["always_entered_freq"].get(name, 0)
    print(f"  {name:<20s} {freq:>7} {pct:>6.1f}%  {mean_per:>11.2f}  {nev:>9.1%}  {alw:>9.1%}")

# 2c. Core vs peripheral
print("\n--- 2c. Core vs Peripheral Strategies ---\n")
print("  Core (appear in >60% of terminal supports across all seed-runs):")
for name in sorted(NAMES, key=lambda n: -strat_boot_freq[n]):
    pct = 100 * strat_boot_freq[name] / total_runs
    if pct > 60:
        print(f"    {name}: {pct:.1f}%")

print("  Peripheral (appear in <20% of terminal supports):")
for name in sorted(NAMES, key=lambda n: strat_boot_freq[n]):
    pct = 100 * strat_boot_freq[name] / total_runs
    if pct < 20:
        print(f"    {name}: {pct:.1f}%")

# 2d. Per-strategy: in how many distinct terminal support sets does it appear?
print("\n--- 2d. Strategy Appearance in Distinct Terminal Support Sets (bootstrap) ---\n")
all_distinct_terminals = set()
strat_in_terminals = defaultdict(set)
for b_idx, ba in enumerate(boot_analyses):
    terminals = per_seed_terminals(ba)
    for si in range(N):
        ts = terminals[si]
        all_distinct_terminals.add(ts)
        for s in ts:
            strat_in_terminals[s].add(ts)

print(f"  Total distinct terminal supports observed: {len(all_distinct_terminals)}\n")
for name in sorted(NAMES, key=lambda n: -len(strat_in_terminals[n])):
    print(f"  {name:<20s}: appears in {len(strat_in_terminals[name]):>3} distinct terminal sets")


# ══════════════════════════════════════════════════════════════════════════════
#  3.  EXPANSION DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  SECTION 3: EXPANSION DYNAMICS")
print("=" * 80)

# 3a. Average entry step per strategy (aggregated data)
print("\n--- 3a. Average Entry Step per Strategy (across all seeds, bootstrap) ---\n")
print(f"  {'Strategy':<20s} {'Mean Step':>10} {'Std':>8} {'CI Lower':>10} {'CI Upper':>10}")
entry_data = agg["entry_step"]
for name in sorted(NAMES, key=lambda n: entry_data[n]["mean"]):
    d = entry_data[name]
    print(f"  {name:<20s} {d['mean']:>10.3f} {d['std']:>8.3f} {d['ci_lower']:>10.2f} {d['ci_upper']:>10.2f}")

# 3b. Entry order: build average rank per strategy across seeds/bootstrap
# For each bootstrap sample and each seed, rank strategies by entry step (None = not entered)
print("\n--- 3b. Average Entry Rank (1=first entrant, higher=later; None=not entered) ---\n")
print("  Note: Rank computed only when strategy actually enters. Mean rank across all entries.\n")

entry_ranks = defaultdict(list)  # strategy -> list of ranks when it entered
for ba in boot_analyses:
    for si in range(N):
        # Get entry steps for this seed
        entries = {}
        for sname in NAMES:
            step = ba["entry_order"][sname][si]
            if step is not None:
                entries[sname] = step
        # Rank them
        sorted_entries = sorted(entries.items(), key=lambda x: x[1])
        for rank, (sname, step) in enumerate(sorted_entries, 1):
            entry_ranks[sname].append(rank)

print(f"  {'Strategy':<20s} {'Mean Rank':>10} {'Median':>8} {'Times Entered':>15} {'% Entered':>10}")
for name in sorted(NAMES, key=lambda n: np.mean(entry_ranks[n]) if entry_ranks[n] else 999):
    ranks = entry_ranks[name]
    if ranks:
        print(f"  {name:<20s} {np.mean(ranks):>10.2f} {np.median(ranks):>8.1f} {len(ranks):>15} {100*len(ranks)/(N*N_BOOT):>9.1f}%")
    else:
        print(f"  {name:<20s} {'N/A':>10} {'N/A':>8} {0:>15} {0:>9.1f}%")

# 3c. Consistent early vs late entrants
print("\n--- 3c. Early vs Late Entrants ---\n")
early_thresh = 1.5
late_thresh = 3.0
print(f"  Early entrants (mean rank < {early_thresh}):")
for name in sorted(NAMES, key=lambda n: np.mean(entry_ranks[n]) if entry_ranks[n] else 999):
    if entry_ranks[name] and np.mean(entry_ranks[name]) < early_thresh:
        print(f"    {name}: mean rank {np.mean(entry_ranks[name]):.2f}")
print(f"  Late entrants (mean rank > {late_thresh}):")
for name in sorted(NAMES, key=lambda n: np.mean(entry_ranks[n]) if entry_ranks[n] else 999):
    if entry_ranks[name] and np.mean(entry_ranks[name]) > late_thresh:
        print(f"    {name}: mean rank {np.mean(entry_ranks[name]):.2f}")

# 3d. Path length variability per seed
print("\n--- 3d. Path Length Statistics per Seed ---\n")
print(f"  {'Seed':<20s} {'Mean':>6} {'Std':>6} {'CI Low':>8} {'CI High':>8} {'CV':>8}")
path_data = agg["path_lengths_per_seed"]
for name in NAMES:
    d = path_data[name]
    cv = d["std"] / d["mean"] if d["mean"] > 0 else 0
    print(f"  {name:<20s} {d['mean']:>6.2f} {d['std']:>6.2f} {d['ci_lower']:>8.1f} {d['ci_upper']:>8.1f} {cv:>7.2f}")

# 3e. Detailed entry-order per seed (point estimate)
print("\n--- 3e. Point-Estimate Expansion Paths ---\n")
for t in traces:
    print(f"  Seed: {t['seed']}")
    for step in t["trace"]:
        sup_names = step["support_names"]
        br = step.get("best_responder", "N/A")
        max_dev = step.get("max_deviation_gain", 0)
        print(f"    Step {step['support'].index(step['support'][0]) if len(step['support'])==1 else ''}"
              f" support={sup_names}, eq_value={step['eq_value']:.2f}, "
              f"best_resp={br} (gain={max_dev:.4f})")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  4.  SENSITIVITY / ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  SECTION 4: SENSITIVITY / ROBUSTNESS")
print("=" * 80)

# 4a. Terminal support stability per seed
print("\n--- 4a. Terminal Support Stability per Seed ---\n")
print(f"  {'Seed':<20s} {'Modal %':>8} {'#Distinct':>10} {'Entropy':>9}")
seed_stability = {}
for si in range(N):
    ctr = Counter(boot_seed_terminals[si])
    total = sum(ctr.values())
    modal_freq = ctr.most_common(1)[0][1]
    modal_pct = 100 * modal_freq / total
    n_distinct = len(ctr)
    # Shannon entropy
    probs = np.array([c / total for c in ctr.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    seed_stability[si] = {"modal_pct": modal_pct, "n_distinct": n_distinct, "entropy": entropy}
    print(f"  {NAMES[si]:<20s} {modal_pct:>7.1f}% {n_distinct:>10} {entropy:>9.3f}")

# Most/least stable
most_stable = max(seed_stability, key=lambda s: seed_stability[s]["modal_pct"])
least_stable = min(seed_stability, key=lambda s: seed_stability[s]["modal_pct"])
print(f"\n  Most stable seed:  {NAMES[most_stable]} ({seed_stability[most_stable]['modal_pct']:.1f}% modal)")
print(f"  Least stable seed: {NAMES[least_stable]} ({seed_stability[least_stable]['modal_pct']:.1f}% modal)")

# 4b. Path length variation per seed
print("\n--- 4b. Path Length Coefficient of Variation per Seed ---\n")
# Collect raw path lengths per seed from bootstrap
boot_path_lengths = defaultdict(list)
for ba in boot_analyses:
    for si in range(N):
        boot_path_lengths[si].append(ba["path_lengths"][si])

print(f"  {'Seed':<20s} {'Mean':>6} {'Std':>6} {'Min':>5} {'Max':>5} {'CV':>7}")
for si in range(N):
    pls = boot_path_lengths[si]
    mn, sd = np.mean(pls), np.std(pls)
    cv = sd / mn if mn > 0 else 0
    print(f"  {NAMES[si]:<20s} {mn:>6.2f} {sd:>6.2f} {min(pls):>5} {max(pls):>5} {cv:>6.2f}")

# 4c. Fragile patterns - terminal supports that appear in <5% of bootstrap samples
print("\n--- 4c. Fragile Terminal Supports (appear in <5% of seed-runs) ---\n")
all_terminal_freq = Counter()
for b_idx, ba in enumerate(boot_analyses):
    terminals = per_seed_terminals(ba)
    for si in range(N):
        all_terminal_freq[terminals[si]] += 1

total_seedruns = N * N_BOOT
fragile = [(sup, cnt) for sup, cnt in all_terminal_freq.items()
           if cnt / total_seedruns < 0.05]
fragile.sort(key=lambda x: -x[1])
print(f"  Total distinct terminals: {len(all_terminal_freq)}")
print(f"  Fragile terminals (<5% of seed-runs): {len(fragile)}\n")
for sup, cnt in fragile[:20]:
    pct = 100 * cnt / total_seedruns
    print(f"    {pct:>5.2f}%  ({cnt:>5} / {total_seedruns})  {{{', '.join(sup)}}}")
if len(fragile) > 20:
    print(f"    ... and {len(fragile) - 20} more")

# 4d. Robust patterns - terminals that appear in >5% of seed-runs
print("\n--- 4d. Robust Terminal Supports (appear in >=5% of seed-runs) ---\n")
robust = [(sup, cnt) for sup, cnt in all_terminal_freq.items()
          if cnt / total_seedruns >= 0.05]
robust.sort(key=lambda x: -x[1])
for sup, cnt in robust:
    pct = 100 * cnt / total_seedruns
    # Which seeds contribute?
    seed_contribs = []
    for si in range(N):
        seed_cnt = sum(1 for t in boot_seed_terminals[si] if t == sup)
        if seed_cnt > 0:
            seed_contribs.append(f"{NAMES[si]}({100*seed_cnt/N_BOOT:.0f}%)")
    print(f"  {pct:>5.1f}%  {{{', '.join(sup)}}}")
    print(f"         Seeds: {', '.join(seed_contribs)}")

# 4e. Bootstrap convergence rate
print(f"\n--- 4e. Overall Convergence Rate ---\n")
print(f"  Rate at which all seeds converge to same terminal: {agg['convergence_rate']*100:.1f}%")
all_converge_count = sum(1 for ba in boot_analyses if ba["all_converge_same"])
print(f"  Bootstrap samples where all seeds match: {all_converge_count}/{N_BOOT}")

# 4f. Terminal regret statistics
print("\n--- 4f. Terminal Regret Statistics per Seed ---\n")
print(f"  Regret = Nash value - expected utility (positive = no profitable deviation)\n")
print(f"  {'Seed':<20s} {'Mean':>10} {'Std':>10} {'CI Low':>10} {'CI High':>10}")
for name in NAMES:
    d = agg["terminal_regrets_per_seed"][name]
    print(f"  {name:<20s} {d['mean']:>10.2f} {d['std']:>10.2f} {d['ci_lower']:>10.2f} {d['ci_upper']:>10.2f}")

print("\n" + "=" * 80)
print("  ANALYSIS COMPLETE")
print("=" * 80)
