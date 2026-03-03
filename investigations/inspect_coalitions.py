"""Inspect equilibria before/after Tough and Walk are added to specific coalitions."""

import pickle
import numpy as np
from pathlib import Path

results_path = Path("data/analysis/iterative_analysis_results.pkl")
with open(results_path, "rb") as f:
    results = pickle.load(f)

raw_samples = results["raw"]

# Coalitions to inspect (without the agent, then agent is added)
tough_coalitions = [
    ["nfsp", "walk"],
    ["nfsp", "soft", "walk"],
    ["openai_5.2_none", "walk"],
    ["psro", "walk"],
    ["nfsp", "openai_5.2_none", "walk"],
]

walk_coalitions = [
    ["nfsp", "tough"],
    ["soft", "tough"],
    ["nfsp", "soft", "tough"],
    ["openai_5.2_none", "tough"],
    ["psro", "tough"],
]


def fmt_sigma(sigma_dict):
    """Format equilibrium dict as string, showing only strategies with > 0.001 support."""
    if not sigma_dict:
        return "(empty game)"
    parts = []
    for k, v in sorted(sigma_dict.items(), key=lambda x: -x[1]):
        if v > 0.001:
            parts.append(f"{k}={v:.4f}")
    return ", ".join(parts) if parts else "(all < 0.001)"


def find_record(details, policy, coalition_without_sorted):
    """Find the coalition detail record matching policy and S."""
    for r in details:
        if r["policy"] == policy and sorted(r["coalition_without"]) == coalition_without_sorted:
            return r
    return None


def print_coalition_analysis(agent, coalitions_without):
    print(f"\n{'='*80}")
    print(f"  AGENT: {agent.upper()}")
    print(f"{'='*80}")

    for S in coalitions_without:
        S_sorted = sorted(S)
        S_with = sorted(S + [agent])
        print(f"\n--- S = {{{', '.join(S_sorted)}}} → S∪{{{agent}}} = {{{', '.join(S_with)}}} ---")

        # Collect across bootstrap samples
        support_count = 0
        total = 0
        sample_results = []

        for b_idx, sample in enumerate(raw_samples):
            details = sample.get("l3", {}).get("coalition_details", [])
            rec = find_record(details, agent, S_sorted)
            if rec is None:
                continue
            total += 1

            sigma_without = rec["sigma_without"]
            sigma_with = rec["sigma_with"]
            marginals = rec["marginals"]

            agent_support = sigma_with.get(agent, 0.0)
            if agent_support > 0.001:
                support_count += 1

            sample_results.append({
                "b_idx": b_idx,
                "sigma_without": sigma_without,
                "sigma_with": sigma_with,
                "agent_support": agent_support,
                "nw_plus_marginal": marginals.get("nw_plus", 0),
                "uw_marginal": marginals.get("uw", 0),
            })

        if total == 0:
            print("  No records found!")
            continue

        print(f"  {agent} in support: {support_count}/{total} samples")

        # Show a few representative examples
        # 1. First sample where agent gets support
        support_examples = [r for r in sample_results if r["agent_support"] > 0.001]
        no_support_examples = [r for r in sample_results if r["agent_support"] <= 0.001]

        if support_examples:
            ex = support_examples[0]
            print(f"\n  Example (bootstrap {ex['b_idx']}) — {agent} IN support:")
            print(f"    Eq WITHOUT {agent}: {fmt_sigma(ex['sigma_without'])}")
            print(f"    Eq WITH    {agent}: {fmt_sigma(ex['sigma_with'])}")
            print(f"    {agent} support: {ex['agent_support']:.4f}")
            print(f"    NW+ marginal: {ex['nw_plus_marginal']:.4f}, UW marginal: {ex['uw_marginal']:.4f}")

        if no_support_examples:
            ex = no_support_examples[0]
            print(f"\n  Example (bootstrap {ex['b_idx']}) — {agent} NOT in support:")
            print(f"    Eq WITHOUT {agent}: {fmt_sigma(ex['sigma_without'])}")
            print(f"    Eq WITH    {agent}: {fmt_sigma(ex['sigma_with'])}")
            print(f"    {agent} support: {ex['agent_support']:.4f}")
            print(f"    NW+ marginal: {ex['nw_plus_marginal']:.4f}, UW marginal: {ex['uw_marginal']:.4f}")

        # Summary stats
        nw_plus_margs = [r["nw_plus_marginal"] for r in sample_results]
        uw_margs = [r["uw_marginal"] for r in sample_results]
        print(f"\n  NW+ marginal: mean={np.mean(nw_plus_margs):.4f}, "
              f"min={np.min(nw_plus_margs):.4f}, max={np.max(nw_plus_margs):.4f}")
        print(f"  UW  marginal: mean={np.mean(uw_margs):.4f}, "
              f"min={np.min(uw_margs):.4f}, max={np.max(uw_margs):.4f}")

        # Show all unique equilibria patterns (without agent)
        eq_without_patterns = {}
        eq_with_patterns = {}
        for r in sample_results:
            key_w = fmt_sigma(r["sigma_without"])
            key_wt = fmt_sigma(r["sigma_with"])
            eq_without_patterns[key_w] = eq_without_patterns.get(key_w, 0) + 1
            eq_with_patterns[key_wt] = eq_with_patterns.get(key_wt, 0) + 1

        print(f"\n  Unique eq WITHOUT {agent} ({len(eq_without_patterns)} patterns):")
        for pat, cnt in sorted(eq_without_patterns.items(), key=lambda x: -x[1]):
            print(f"    [{cnt:3d}x] {pat}")

        print(f"\n  Unique eq WITH {agent} ({len(eq_with_patterns)} patterns):")
        for pat, cnt in sorted(eq_with_patterns.items(), key=lambda x: -x[1])[:10]:
            print(f"    [{cnt:3d}x] {pat}")
        if len(eq_with_patterns) > 10:
            print(f"    ... and {len(eq_with_patterns) - 10} more patterns")


print_coalition_analysis("tough", tough_coalitions)
print_coalition_analysis("walk", walk_coalitions)
