"""
L3 Coalition Deep Dive Analysis.

Dissects the 5.12M coalition records (1000 bootstraps × 5120 records) to
reveal *which subgames* drive Shapley/Banzhaf results.

Five analyses, all computed in a single pass over the data:
  1. Marginal by Coalition Size — size-resolved attribution curves
  2. Top/Bottom Coalitions — the specific subgames driving each agent
  3. Pairwise Synergy Matrix — who helps/hurts whom
  4. Equilibrium Displacement — who steals weight from whom
  5. PPO vs PSRO Puzzle — why PPO has high Shapley but PSRO has near-zero Banzhaf

Usage:
    python visuals/l3_coalition_deep_dive.py [--sample-only] [--welfare uw nw nw_plus ef1]
"""

import argparse
import math
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# Reuse constants from visualize_analysis.py
from visualize_analysis import (
    STRATEGY_ORDER,
    DISPLAY_NAMES,
    STRATEGY_COLORS,
    L3_NORM,
    POS_COLOR,
    NEG_COLOR,
    DEFAULT_DPI,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WF_TITLES = {
    "uw": "Utilitarian Welfare",
    "nw": "Nash Welfare",
    "nw_plus": "NW+ Welfare",
    "ef1": "EF1 Frequency",
}


def _dn(strategy: str) -> str:
    return DISPLAY_NAMES.get(strategy, strategy)


def _norm(val, wf_key):
    """Normalize a raw marginal to % of max (or pp for EF1)."""
    return val / L3_NORM[wf_key] * 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_raw(pkl_path: str | Path):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    return results["raw"]


# ---------------------------------------------------------------------------
# Single-pass accumulation
# ---------------------------------------------------------------------------


def accumulate(raw_samples, welfare_keys):
    """Single pass over all bootstrap × coalition records.

    Returns a dict of accumulator arrays for all 5 analyses.
    """
    agents = STRATEGY_ORDER
    n_agents = len(agents)
    n_wf = len(welfare_keys)
    n_boot = len(raw_samples)
    n_sizes = 10  # coalition sizes 0..9

    agent_to_idx = {a: i for i, a in enumerate(agents)}
    wf_to_idx = {w: i for i, w in enumerate(welfare_keys)}

    # --- Analysis 1: marginal by coalition size ---
    # Per-bootstrap sums so we can compute CI bands
    size_sums = np.zeros((n_boot, n_agents, n_wf, n_sizes))
    size_counts = np.zeros((n_boot, n_agents, n_sizes))

    # --- Analysis 2: top/bottom coalitions ---
    # Build coalition index: (policy_idx, sorted_tuple_of_S) -> int
    # We'll populate this on the fly during the first bootstrap sample
    coal_to_idx = {}
    coal_keys = []  # list of (policy, tuple(sorted(S)))
    coal_sums = None  # deferred allocation
    coal_boot_count = 0  # number of bootstrap samples processed

    # --- Analysis 3: pairwise synergy ---
    syn_present_sum = np.zeros((n_agents, n_wf, n_agents))
    syn_present_cnt = np.zeros((n_agents, n_agents))
    syn_absent_sum = np.zeros((n_agents, n_wf, n_agents))
    syn_absent_cnt = np.zeros((n_agents, n_agents))

    # --- Analysis 4: equilibrium displacement ---
    disp_sum = np.zeros((n_agents, n_agents))
    disp_cnt = np.zeros((n_agents, n_agents))

    # --- Analysis 5: PPO vs PSRO ---
    psro_nw_all = []  # all PSRO NW marginals (flat list)

    print(f"Accumulating over {n_boot} bootstrap samples × 5120 records …")

    for b_idx, sample in enumerate(raw_samples):
        if (b_idx + 1) % 100 == 0 or b_idx == 0:
            print(f"  sample {b_idx + 1}/{n_boot}")

        details = sample["l3"]["coalition_details"]

        for r in details:
            policy = r["policy"]
            S = r["coalition_without"]
            size_s = len(S)
            members = set(S)

            p_idx = agent_to_idx[policy]

            # Marginal values for selected welfare keys
            margs = [r["marginals"][w] for w in welfare_keys]

            # --- A1: size marginals ---
            for w_idx, m in enumerate(margs):
                size_sums[b_idx, p_idx, w_idx, size_s] += m
            size_counts[b_idx, p_idx, size_s] += 1

            # --- A2: coalition-level marginals ---
            coal_key = (policy, tuple(sorted(S)))
            if coal_key not in coal_to_idx:
                coal_to_idx[coal_key] = len(coal_keys)
                coal_keys.append(coal_key)
            c_idx = coal_to_idx[coal_key]

            # Allocate / extend coal_sums lazily
            if coal_sums is None:
                coal_sums = np.zeros((6000, n_wf))  # slightly over 5120
                coal_cnt = np.zeros(6000)
            if c_idx >= coal_sums.shape[0]:
                # Shouldn't happen (5120 coalitions) but be safe
                extra = np.zeros((1000, n_wf))
                coal_sums = np.concatenate([coal_sums, extra], axis=0)
                coal_cnt = np.concatenate([coal_cnt, np.zeros(1000)])

            for w_idx, m in enumerate(margs):
                coal_sums[c_idx, w_idx] += m
            coal_cnt[c_idx] += 1

            # --- A3: pairwise synergy ---
            others = set(agents) - {policy}
            for partner in others:
                partner_idx = agent_to_idx[partner]
                if partner in members:
                    for w_idx, m in enumerate(margs):
                        syn_present_sum[p_idx, w_idx, partner_idx] += m
                    syn_present_cnt[p_idx, partner_idx] += 1
                else:
                    for w_idx, m in enumerate(margs):
                        syn_absent_sum[p_idx, w_idx, partner_idx] += m
                    syn_absent_cnt[p_idx, partner_idx] += 1

            # --- A4: equilibrium displacement ---
            sigma_w = r["sigma_without"]
            sigma_wt = r["sigma_with"]
            for member in members:
                m_idx = agent_to_idx[member]
                old_wt = sigma_w.get(member, 0.0)
                new_wt = sigma_wt.get(member, 0.0)
                disp_sum[p_idx, m_idx] += (new_wt - old_wt)
                disp_cnt[p_idx, m_idx] += 1

            # --- A5: PSRO NW marginals ---
            if policy == "psro" and "nw" in r["marginals"]:
                psro_nw_all.append(r["marginals"]["nw"])

    # Trim coal arrays
    n_coal = len(coal_keys)
    coal_sums = coal_sums[:n_coal]
    coal_cnt = coal_cnt[:n_coal]

    print(f"  Done. {n_coal} unique coalitions indexed.")

    return {
        "agents": agents,
        "welfare_keys": welfare_keys,
        "n_boot": n_boot,
        "agent_to_idx": agent_to_idx,
        "wf_to_idx": wf_to_idx,
        # A1
        "size_sums": size_sums,
        "size_counts": size_counts,
        # A2
        "coal_keys": coal_keys,
        "coal_sums": coal_sums,
        "coal_cnt": coal_cnt,
        # A3
        "syn_present_sum": syn_present_sum,
        "syn_present_cnt": syn_present_cnt,
        "syn_absent_sum": syn_absent_sum,
        "syn_absent_cnt": syn_absent_cnt,
        # A4
        "disp_sum": disp_sum,
        "disp_cnt": disp_cnt,
        # A5
        "psro_nw_all": np.array(psro_nw_all),
    }


# ---------------------------------------------------------------------------
# Analysis 1: Marginal by Coalition Size
# ---------------------------------------------------------------------------


def analysis1_marginal_by_size(acc, fig_dir):
    """Plot mean marginal at each coalition size, one figure per welfare fn."""
    agents = acc["agents"]
    welfare_keys = acc["welfare_keys"]
    n_boot = acc["n_boot"]
    agent_to_idx = acc["agent_to_idx"]
    wf_to_idx = acc["wf_to_idx"]

    size_sums = acc["size_sums"]   # (n_boot, n_agents, n_wf, 10)
    size_counts = acc["size_counts"]  # (n_boot, n_agents, 10)

    sizes = np.arange(10)

    for wf in welfare_keys:
        w_idx = wf_to_idx[wf]
        fig, ax = plt.subplots(figsize=(10, 6))

        for agent in agents:
            p_idx = agent_to_idx[agent]

            # Per-bootstrap mean marginal at each size
            # size_counts[b, p, s] is the count; same across wf
            cnts = size_counts[:, p_idx, :]  # (n_boot, 10)
            sums = size_sums[:, p_idx, w_idx, :]  # (n_boot, 10)

            # Avoid division by zero (size 0 has only 1 coalition)
            with np.errstate(divide="ignore", invalid="ignore"):
                per_boot_mean = np.where(cnts > 0, sums / cnts, np.nan)

            # Normalize to %
            per_boot_mean = per_boot_mean / L3_NORM[wf] * 100

            mean_vals = np.nanmean(per_boot_mean, axis=0)
            ci_lo = np.nanpercentile(per_boot_mean, 2.5, axis=0)
            ci_hi = np.nanpercentile(per_boot_mean, 97.5, axis=0)

            color = STRATEGY_COLORS[agent]
            ax.plot(sizes, mean_vals, "o-", color=color, label=_dn(agent),
                    markersize=4, linewidth=1.5)
            ax.fill_between(sizes, ci_lo, ci_hi, color=color, alpha=0.12)

        ax.set_xlabel("Coalition Size |S| (without agent)")
        ax.set_ylabel(f"Mean Marginal ({'pp' if wf == 'ef1' else '% of max'})")
        ax.set_title(f"Marginal by Coalition Size — {WF_TITLES[wf]}")
        ax.set_xticks(sizes)
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"a1_marginal_by_size_{wf}.png"),
                    dpi=DEFAULT_DPI)
        plt.close(fig)

    # Print summary table
    print("\n=== Analysis 1: Mean Marginal by Coalition Size ===")
    for wf in welfare_keys:
        w_idx = wf_to_idx[wf]
        print(f"\n  {WF_TITLES[wf]} ({'pp' if wf == 'ef1' else '% of max'}):")
        header = f"  {'Agent':<18s}" + "".join(f"{'s=' + str(s):>8s}" for s in range(10))
        print(header)
        for agent in agents:
            p_idx = agent_to_idx[agent]
            cnts = size_counts[:, p_idx, :]
            sums = size_sums[:, p_idx, w_idx, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                pbm = np.where(cnts > 0, sums / cnts, np.nan)
            pbm = pbm / L3_NORM[wf] * 100
            means = np.nanmean(pbm, axis=0)
            row = f"  {_dn(agent):<18s}" + "".join(f"{m:8.2f}" for m in means)
            print(row)


# ---------------------------------------------------------------------------
# Analysis 2: Top/Bottom Coalitions
# ---------------------------------------------------------------------------


def analysis2_top_bottom_coalitions(acc, welfare_keys_subset=None):
    """Print the 5 coalitions with highest/lowest mean marginal per agent."""
    agents = acc["agents"]
    welfare_keys = welfare_keys_subset or acc["welfare_keys"]
    coal_keys = acc["coal_keys"]
    coal_sums = acc["coal_sums"]
    coal_cnt = acc["coal_cnt"]
    wf_to_idx = acc["wf_to_idx"]

    # Compute mean marginals per coalition
    with np.errstate(divide="ignore", invalid="ignore"):
        coal_means = np.where(coal_cnt[:, None] > 0,
                              coal_sums / coal_cnt[:, None], 0)

    print("\n=== Analysis 2: Top/Bottom 5 Coalitions ===")

    for wf in welfare_keys:
        if wf not in wf_to_idx:
            continue
        w_idx = wf_to_idx[wf]
        unit = "pp" if wf == "ef1" else "% of max"

        print(f"\n{'─' * 80}")
        print(f"  {WF_TITLES[wf]} ({unit})")
        print(f"{'─' * 80}")

        for agent in agents:
            # Find all coalitions for this agent
            indices = [i for i, (p, _) in enumerate(coal_keys) if p == agent]
            if not indices:
                continue

            vals = [(i, _norm(coal_means[i, w_idx], wf)) for i in indices]
            vals.sort(key=lambda x: x[1], reverse=True)

            top5 = vals[:5]
            bot5 = vals[-5:]

            print(f"\n  {_dn(agent)}:")
            print(f"    TOP 5:")
            for idx, val in top5:
                _, S_tuple = coal_keys[idx]
                members_str = ", ".join(_dn(m) for m in S_tuple) if S_tuple else "(empty)"
                print(f"      {val:+7.2f} {unit}  S = {{{members_str}}}")

            print(f"    BOTTOM 5:")
            for idx, val in bot5:
                _, S_tuple = coal_keys[idx]
                members_str = ", ".join(_dn(m) for m in S_tuple) if S_tuple else "(empty)"
                print(f"      {val:+7.2f} {unit}  S = {{{members_str}}}")


# ---------------------------------------------------------------------------
# Analysis 3: Pairwise Synergy Matrix
# ---------------------------------------------------------------------------


def analysis3_pairwise_synergy(acc, fig_dir):
    """Plot 10×10 synergy heatmap per welfare fn."""
    agents = acc["agents"]
    welfare_keys = acc["welfare_keys"]
    n_agents = len(agents)
    agent_to_idx = acc["agent_to_idx"]
    wf_to_idx = acc["wf_to_idx"]

    syn_pres_sum = acc["syn_present_sum"]
    syn_pres_cnt = acc["syn_present_cnt"]
    syn_abs_sum = acc["syn_absent_sum"]
    syn_abs_cnt = acc["syn_absent_cnt"]

    labels = [_dn(a) for a in agents]

    for wf in welfare_keys:
        w_idx = wf_to_idx[wf]
        unit = "pp" if wf == "ef1" else "% of max"

        mat = np.full((n_agents, n_agents), np.nan)
        for t in range(n_agents):
            for p in range(n_agents):
                if t == p:
                    continue
                cnt_pres = syn_pres_cnt[t, p]
                cnt_abs = syn_abs_cnt[t, p]
                if cnt_pres > 0 and cnt_abs > 0:
                    mean_pres = syn_pres_sum[t, w_idx, p] / cnt_pres
                    mean_abs = syn_abs_sum[t, w_idx, p] / cnt_abs
                    mat[t, p] = _norm(mean_pres - mean_abs, wf)

        mask = np.isnan(mat)
        vmax = np.nanmax(np.abs(mat[~mask])) if not np.all(mask) else 1

        fig, ax = plt.subplots(figsize=(10, 8.5))
        sns.heatmap(mat, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax,
                    cbar_kws={"label": f"Synergy ({unit})"})
        ax.set_xlabel("Partner P (present/absent in S)")
        ax.set_ylabel("Target T (whose marginal is measured)")
        ax.set_title(
            f"Pairwise Synergy — {WF_TITLES[wf]}\n"
            f"Cell = E[marginal(T) | P∈S] − E[marginal(T) | P∉S]"
        )
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"a3_synergy_{wf}.png"),
                    dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    print("\n=== Analysis 3: Pairwise Synergy Heatmaps saved ===")


# ---------------------------------------------------------------------------
# Analysis 4: Equilibrium Displacement
# ---------------------------------------------------------------------------


def analysis4_equilibrium_displacement(acc, fig_dir):
    """Plot 10×10 heatmap: rows=joiner, cols=displaced member."""
    agents = acc["agents"]
    n_agents = len(agents)
    disp_sum = acc["disp_sum"]
    disp_cnt = acc["disp_cnt"]

    labels = [_dn(a) for a in agents]

    mat = np.full((n_agents, n_agents), np.nan)
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            if disp_cnt[i, j] > 0:
                mat[i, j] = disp_sum[i, j] / disp_cnt[i, j]

    mask = np.isnan(mat)
    vmax = np.nanmax(np.abs(mat[~mask])) if not np.all(mask) else 0.01

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(mat, mask=mask, annot=True, fmt=".4f",
                cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean Δ eq. weight"})
    ax.set_xlabel("Existing Member (whose weight changes)")
    ax.set_ylabel("Joiner (agent entering the coalition)")
    ax.set_title(
        "Equilibrium Displacement\n"
        "Cell = mean change in member's eq. weight when joiner enters"
    )

    fig.text(
        0.5, -0.02,
        "Red (+): member gains weight  |  Blue (−): member loses weight",
        ha="center", fontsize=9, style="italic",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "a4_equilibrium_displacement.png"),
                dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Analysis 4: Equilibrium Displacement heatmap saved ===")


# ---------------------------------------------------------------------------
# Analysis 5: PPO vs PSRO Puzzle
# ---------------------------------------------------------------------------


def analysis5_ppo_psro_puzzle(acc, fig_dir):
    """Investigate why PPO has high Shapley but PSRO has near-zero Banzhaf."""
    agents = acc["agents"]
    agent_to_idx = acc["agent_to_idx"]
    wf_to_idx = acc["wf_to_idx"]
    n_boot = acc["n_boot"]

    size_sums = acc["size_sums"]
    size_counts = acc["size_counts"]
    psro_nw_all = acc["psro_nw_all"]

    # Only proceed if NW is in welfare keys
    if "nw" not in wf_to_idx:
        print("\n=== Analysis 5: Skipped (NW not in welfare keys) ===")
        return

    w_idx = wf_to_idx["nw"]
    n = len(agents)  # 10 players
    sizes = np.arange(10)

    # Shapley weight function: s!(n-s-1)!/n!
    shapley_weights = np.array([
        math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
        for s in range(10)
    ])

    # --- Figure 1: PPO vs PSRO marginal-by-size with Shapley weights overlay ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    for agent, style in [("ppo", "o-"), ("psro", "s--")]:
        p_idx = agent_to_idx[agent]
        cnts = size_counts[:, p_idx, :]
        sums = size_sums[:, p_idx, w_idx, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            pbm = np.where(cnts > 0, sums / cnts, np.nan)
        pbm = pbm / L3_NORM["nw"] * 100

        mean_vals = np.nanmean(pbm, axis=0)
        ci_lo = np.nanpercentile(pbm, 2.5, axis=0)
        ci_hi = np.nanpercentile(pbm, 97.5, axis=0)

        color = STRATEGY_COLORS[agent]
        ax1.plot(sizes, mean_vals, style, color=color, label=f"{_dn(agent)} marginal",
                 markersize=6, linewidth=2)
        ax1.fill_between(sizes, ci_lo, ci_hi, color=color, alpha=0.15)

    ax1.set_xlabel("Coalition Size |S|")
    ax1.set_ylabel("Mean NW Marginal (% of max)", color="black")
    ax1.set_xticks(sizes)
    ax1.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)

    # Overlay Shapley weights on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(sizes, shapley_weights, alpha=0.2, color="grey",
            label="Shapley weight", width=0.6)
    ax2.set_ylabel("Shapley Weight s!(n−s−1)!/n!", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title("PPO vs PSRO: NW Marginal by Coalition Size + Shapley Weights")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "a5_ppo_psro_size_overlay.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)

    # --- Figure 2: PSRO NW marginal histogram ---
    if len(psro_nw_all) > 0:
        normed = psro_nw_all / L3_NORM["nw"] * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(normed, bins=80, color=STRATEGY_COLORS["psro"],
                edgecolor="black", linewidth=0.3, alpha=0.8)
        ax.axvline(0, color="black", lw=1.2, ls="--")
        median_val = np.median(normed)
        mean_val = np.mean(normed)
        ax.axvline(mean_val, color=POS_COLOR, lw=1.5, ls=":",
                   label=f"Mean = {mean_val:.2f}")
        ax.axvline(median_val, color=NEG_COLOR, lw=1.5, ls=":",
                   label=f"Median = {median_val:.2f}")
        ax.set_xlabel("NW Marginal (% of max)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"PSRO NW Marginal Distribution (all {len(normed):,} coalition records)"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "a5_psro_nw_histogram.png"),
                    dpi=DEFAULT_DPI)
        plt.close(fig)

    print("\n=== Analysis 5: PPO vs PSRO figures saved ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="L3 Coalition Deep Dive Analysis"
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Use only 1 bootstrap sample for quick iteration",
    )
    parser.add_argument(
        "--pkl-path", type=str,
        default="data/analysis/iterative_analysis_results.pkl",
        help="Path to results pkl",
    )
    parser.add_argument(
        "--fig-dir", type=str,
        default="data/analysis/figures/l3_deep_dive",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--welfare", nargs="+",
        default=["uw", "nw", "nw_plus", "ef1"],
        help="Welfare functions to analyze",
    )
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    print("Loading data …")
    raw = load_raw(args.pkl_path)
    if args.sample_only:
        raw = raw[:1]
        print("  --sample-only: using 1 bootstrap sample")
    print(f"  {len(raw)} bootstrap samples loaded")

    acc = accumulate(raw, args.welfare)

    analysis1_marginal_by_size(acc, args.fig_dir)
    analysis2_top_bottom_coalitions(acc)
    analysis3_pairwise_synergy(acc, args.fig_dir)
    analysis4_equilibrium_displacement(acc, args.fig_dir)
    analysis5_ppo_psro_puzzle(acc, args.fig_dir)

    n_figs = len(args.welfare) * 2 + 1 + 2  # A1 + A3 + A4 + A5
    print(f"\nDone. ~{n_figs} figures saved to {args.fig_dir}/")


if __name__ == "__main__":
    main()
