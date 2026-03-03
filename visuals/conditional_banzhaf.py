"""Conditional Banzhaf Analysis

Splits Banzhaf marginal contributions by meaningful conditions
on the coalition composition, producing conditioned beeswarm plots
and paired statistical tests (Wilcoxon signed-rank) for each
strategy x welfare metric x condition.

Output saved to data/analysis/banzhaf_conditions/
"""

import json
import pickle
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Constants (match visualize_analysis.py)
# ---------------------------------------------------------------------------

MAX_UW = 805.9
MAX_NW = 378.7
MAX_NW_PLUS = 81.7

L3_NORM = {
    "uw": MAX_UW,
    "nw": MAX_NW,
    "nw_plus": MAX_NW_PLUS,
    "ef1": 1.0,
    "ef1_plus": 1.0,
}

DISPLAY_NAMES = {
    "walk": "Walk",
    "tough": "Tough",
    "nfsp": "NFSP",
    "mappo": "MAPPO",
    "soft": "Soft",
    "ppo": "PPO",
    "psro": "PSRO",
    "openai_5.2_none": "OpenAI-5.2",
    "openai_5.2_low": "OpenAI-5.2-Low",
    "ef1_bargainer": "EF1",
}

WF_TITLES = {
    "uw": "Utilitarian Welfare",
    "nw": "Nash Welfare",
    "nw_plus": "NW+ Welfare",
    "ef1": "EF1 Frequency",
    "ef1_plus": "EF1+ Frequency",
}

COND_TRUE_COLOR = "#d62728"   # red — condition met
COND_FALSE_COLOR = "#1f77b4"  # blue — condition not met
DEFAULT_DPI = 200

WEAK_AGENTS = {"walk", "tough", "soft"}
RL_AGENTS = {"ppo", "psro", "nfsp", "mappo"}
LLM_AGENTS = {"openai_5.2_none", "openai_5.2_low"}

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------


def _make_presence_fn(target):
    """Return a condition function checking if target is in coalition_without."""
    def fn(record):
        return target in record["coalition_without"]
    return fn


def _make_group_fn(group):
    """Return a condition function checking if any member of group is present."""
    def fn(record):
        return bool(group & set(record["coalition_without"]))
    return fn


def cond_small_coalition(record):
    return len(record["coalition_without"]) <= 3


def cond_singleton(record):
    return len(record["coalition_without"]) == 0


CONDITIONS = {
    # --- Strategy presence ---
    "ppo_present": {
        "fn": _make_presence_fn("ppo"),
        "true_label": "PPO ∈ X",
        "false_label": "PPO ∉ X",
        "title_suffix": "conditioned on PPO presence",
        "exclude_policies": {"ppo"},
    },
    "psro_present": {
        "fn": _make_presence_fn("psro"),
        "true_label": "PSRO ∈ X",
        "false_label": "PSRO ∉ X",
        "title_suffix": "conditioned on PSRO presence",
        "exclude_policies": {"psro"},
    },
    "openai52low_present": {
        "fn": _make_presence_fn("openai_5.2_low"),
        "true_label": "5.2-Low ∈ X",
        "false_label": "5.2-Low ∉ X",
        "title_suffix": "conditioned on OpenAI-5.2-Low presence",
        "exclude_policies": {"openai_5.2_low"},
    },
    "nfsp_present": {
        "fn": _make_presence_fn("nfsp"),
        "true_label": "NFSP ∈ X",
        "false_label": "NFSP ∉ X",
        "title_suffix": "conditioned on NFSP presence",
        "exclude_policies": {"nfsp"},
    },
    "ef1_present": {
        "fn": _make_presence_fn("ef1_bargainer"),
        "true_label": "EF1 ∈ X",
        "false_label": "EF1 ∉ X",
        "title_suffix": "conditioned on EF1 presence",
        "exclude_policies": {"ef1_bargainer"},
    },
    # --- Group presence ---
    "weak_present": {
        "fn": _make_group_fn(WEAK_AGENTS),
        "true_label": "Weak ∈ X",
        "false_label": "Weak ∉ X",
        "title_suffix": "conditioned on weak agent presence",
        "exclude_policies": WEAK_AGENTS,
    },
    "rl_present": {
        "fn": _make_group_fn(RL_AGENTS),
        "true_label": "RL ∈ X",
        "false_label": "RL ∉ X",
        "title_suffix": "conditioned on RL agent presence",
        "exclude_policies": RL_AGENTS,
    },
    "llm_present": {
        "fn": _make_group_fn(LLM_AGENTS),
        "true_label": "LLM ∈ X",
        "false_label": "LLM ∉ X",
        "title_suffix": "conditioned on LLM agent presence",
        "exclude_policies": LLM_AGENTS,
    },
    # --- Restricted game structure ---
    "small_coalition": {
        "fn": cond_small_coalition,
        "true_label": "|X| ≤ 3",
        "false_label": "|X| > 3",
        "title_suffix": "conditioned on restricted game size",
        "exclude_policies": set(),
    },
    "singleton": {
        "fn": cond_singleton,
        "true_label": "|X| = 0",
        "false_label": "|X| > 0",
        "title_suffix": "conditioned on singleton restricted game",
        "exclude_policies": set(),
    },
}

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_conditional_banzhaf(raw, condition_fn, wf_keys, exclude_policies=None):
    """Compute per-bootstrap conditional Banzhaf values.

    For each bootstrap sample, split coalition_details by condition,
    then average marginals within each group to get a conditional
    Banzhaf value per strategy.

    Returns:
        dict with keys 'true' and 'false', each mapping to
        {wf_key: {strategy: [values across bootstrap samples]}}
    """
    if exclude_policies is None:
        exclude_policies = set()

    result = {
        "true": {wk: {} for wk in wf_keys},
        "false": {wk: {} for wk in wf_keys},
    }

    for sample in raw:
        groups = {"true": {}, "false": {}}
        for r in sample["l3"]["coalition_details"]:
            p = r["policy"]
            if p in exclude_policies:
                continue
            group = "true" if condition_fn(r) else "false"
            if p not in groups[group]:
                groups[group][p] = {wk: [] for wk in wf_keys}
            for wk in wf_keys:
                groups[group][p][wk].append(r["marginals"][wk])

        for group in ("true", "false"):
            for p, wk_marginals in groups[group].items():
                for wk in wf_keys:
                    if wk_marginals[wk]:
                        val = np.mean(wk_marginals[wk])
                        result[group][wk].setdefault(p, []).append(val)

    return result


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------


def compute_pvalues(cond_data, wf_keys):
    """Two-sided bootstrap p-value for conditional Banzhaf differences.

    For each strategy x welfare metric, computes the per-bootstrap-sample
    difference (true - false), then reports the two-sided p-value as:
        p = 2 * min(fraction ≥ 0, fraction ≤ 0)
    i.e. the proportion of bootstrap samples inconsistent with the
    observed sign of the mean difference.

    Returns:
        dict: {wf_key: {strategy: {pvalue, mean_diff, mean_true, mean_false,
               significant_005, significant_001, n_samples}}}
    """
    results = {}
    for wk in wf_keys:
        results[wk] = {}
        true_strats = set(cond_data["true"][wk].keys())
        false_strats = set(cond_data["false"][wk].keys())
        common = true_strats & false_strats

        for s in sorted(common):
            arr_t = np.array(cond_data["true"][wk][s])
            arr_f = np.array(cond_data["false"][wk][s])
            n = min(len(arr_t), len(arr_f))
            if n < 10:
                continue
            arr_t = arr_t[:n]
            arr_f = arr_f[:n]
            diff = arr_t - arr_f
            mean_diff = float(np.mean(diff))

            # Two-sided bootstrap p-value
            frac_pos = np.mean(diff >= 0)
            frac_neg = np.mean(diff <= 0)
            pval = float(2.0 * min(frac_pos, frac_neg))
            pval = min(pval, 1.0)  # cap at 1

            results[wk][s] = {
                "pvalue": pval,
                "mean_true": float(np.mean(arr_t)),
                "mean_false": float(np.mean(arr_f)),
                "mean_diff": mean_diff,
                "significant_005": pval < 0.05,
                "significant_001": pval < 0.01,
                "n_samples": n,
            }

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_conditional_beeswarm(cond_data, cond_cfg, wf_keys, save_path,
                              pval_results=None):
    """Plot a conditioned beeswarm figure with optional significance markers."""
    n_panels = len(wf_keys)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    true_label = cond_cfg["true_label"]
    false_label = cond_cfg["false_label"]

    for ax, wk in zip(axes, wf_keys):
        true_vals = cond_data["true"][wk]
        false_vals = cond_data["false"][wk]
        div = L3_NORM[wk]
        scale = 100.0

        all_strats = sorted(
            set(true_vals.keys()) | set(false_vals.keys()),
            key=lambda s: np.mean(
                np.abs(np.array(true_vals.get(s, [0])) / div * scale)
            ) + np.mean(
                np.abs(np.array(false_vals.get(s, [0])) / div * scale)
            ),
            reverse=True,
        )

        y_positions = np.arange(len(all_strats))
        rng = np.random.default_rng(42)

        for i, s in enumerate(all_strats):
            if s in true_vals and true_vals[s]:
                vals_t = np.array(true_vals[s]) / div * scale
                jitter_t = rng.uniform(-0.12, 0.12, size=len(vals_t))
                ax.scatter(
                    vals_t, i - 0.15 + jitter_t,
                    c=COND_TRUE_COLOR, s=30, alpha=0.6,
                    edgecolors="black", linewidths=0.2, zorder=3,
                )
                ax.scatter(
                    np.mean(vals_t), i - 0.15,
                    c=COND_TRUE_COLOR, s=120, marker="D",
                    edgecolors="black", linewidths=1.0, zorder=5,
                )

            if s in false_vals and false_vals[s]:
                vals_f = np.array(false_vals[s]) / div * scale
                jitter_f = rng.uniform(-0.12, 0.12, size=len(vals_f))
                ax.scatter(
                    vals_f, i + 0.15 + jitter_f,
                    c=COND_FALSE_COLOR, s=30, alpha=0.6,
                    edgecolors="black", linewidths=0.2, zorder=3,
                )
                ax.scatter(
                    np.mean(vals_f), i + 0.15,
                    c=COND_FALSE_COLOR, s=120, marker="D",
                    edgecolors="black", linewidths=1.0, zorder=5,
                )

        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([DISPLAY_NAMES.get(s, s) for s in all_strats])
        ax.invert_yaxis()

        unit = "pp" if wk in ("ef1", "ef1_plus") else "% of max"
        ax.set_title(
            f"Conditional Banzhaf – {WF_TITLES[wk]} ({cond_cfg['title_suffix']})"
        )
        ax.set_xlabel(f"Banzhaf Value ({unit})")

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COND_TRUE_COLOR,
               markersize=10, markeredgecolor="black", label=true_label),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COND_FALSE_COLOR,
               markersize=10, markeredgecolor="black", label=false_label),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decomposition bar chart (Figure 5-style)
# ---------------------------------------------------------------------------

# Conditions to decompose by, and their display colors
DECOMP_CONDITIONS = [
    ("openai52low_present", "#e41a1c", "5.2-Low ∈ X", "5.2-Low ∉ X"),
    ("weak_present",        "#377eb8", "Weak ∈ X",    "Weak ∉ X"),
    ("small_coalition",     "#4daf4a", "|X| ≤ 3",     "|X| > 3"),
]


def plot_decomposition_bar(all_cond_data, wf_key, save_path):
    """Stacked horizontal bar decomposing each agent's Banzhaf by condition.

    For each condition, shows paired bars (true / false) for every agent,
    sorted by unconditional Banzhaf. Inspired by Figure 5 of
    'Re-evaluating Open-ended Evaluation of LLMs' (Liu et al., ICLR 2025).
    """
    div = L3_NORM[wf_key]
    scale = 100.0
    n_conds = len(DECOMP_CONDITIONS)

    # Collect mean conditional values per agent per condition
    # Also compute unconditional mean for sorting
    all_agents = set()
    cond_means = {}  # (cond_name, group) -> {agent: mean_val}
    for cond_name, _, _, _ in DECOMP_CONDITIONS:
        for group in ("true", "false"):
            key = (cond_name, group)
            cond_means[key] = {}
            agents_in = all_cond_data[cond_name][group][wf_key]
            for agent, vals in agents_in.items():
                mean_v = np.mean(vals) / div * scale
                cond_means[key][agent] = mean_v
                all_agents.add(agent)

    # Sort agents by average of all conditional means (proxy for unconditional)
    def sort_key(agent):
        vals = []
        for cond_name, _, _, _ in DECOMP_CONDITIONS:
            for group in ("true", "false"):
                v = cond_means.get((cond_name, group), {}).get(agent, 0)
                vals.append(v)
        return np.mean(vals)

    agents_sorted = sorted(all_agents, key=sort_key, reverse=True)
    n_agents = len(agents_sorted)

    # Layout: one row of subplots, one per condition
    fig, axes = plt.subplots(1, n_conds, figsize=(5.5 * n_conds, 0.55 * n_agents + 1.2),
                             sharey=True)
    if n_conds == 1:
        axes = [axes]

    bar_height = 0.35
    y_pos = np.arange(n_agents)

    for ax, (cond_name, color, true_label, false_label) in zip(axes, DECOMP_CONDITIONS):
        true_vals = [cond_means.get((cond_name, "true"), {}).get(a, 0)
                     for a in agents_sorted]
        false_vals = [cond_means.get((cond_name, "false"), {}).get(a, 0)
                      for a in agents_sorted]

        # Lighter shade for "false" condition
        import matplotlib.colors as mcolors
        light_color = mcolors.to_rgba(color, alpha=0.35)
        dark_color = mcolors.to_rgba(color, alpha=0.90)

        ax.barh(y_pos - bar_height / 2, true_vals, height=bar_height,
                color=dark_color, edgecolor="black", linewidth=0.4,
                label=true_label, zorder=3)
        ax.barh(y_pos + bar_height / 2, false_vals, height=bar_height,
                color=light_color, edgecolor="black", linewidth=0.4,
                label=false_label, zorder=3)

        ax.axvline(0, color="black", lw=0.8, ls="-")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([DISPLAY_NAMES.get(a, a) for a in agents_sorted])
        ax.invert_yaxis()

        unit = "pp" if wf_key in ("ef1", "ef1_plus") else "% of max"
        ax.set_xlabel(f"Banzhaf ({unit})", fontsize=10)
        ax.set_title(f"{true_label}  vs  {false_label}", fontsize=11,
                     fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.grid(axis="x", alpha=0.25, zorder=0)

    fig.suptitle(f"Conditional Banzhaf Decomposition — {WF_TITLES[wf_key]}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def write_summary(all_results, save_dir):
    """Write a human-readable summary of significant findings."""
    lines = [
        "Conditional Banzhaf Analysis — Significance Summary",
        "=" * 55,
        "",
        "Two-sided bootstrap p-value on paired samples (n=1000).",
        "p = 2 * min(frac(diff >= 0), frac(diff <= 0))",
        "* p < 0.05   ** p < 0.01   *** p < 0.001",
        "",
    ]

    for cond_name, data in all_results.items():
        cond_cfg = CONDITIONS[cond_name]
        lines.append(f"\n{'─' * 60}")
        lines.append(f"Condition: {cond_cfg['true_label']} vs {cond_cfg['false_label']}")
        lines.append(f"Excluded: {cond_cfg['exclude_policies'] or 'none'}")
        lines.append(f"{'─' * 60}")

        pvals = data["pvalues"]
        for wk in pvals:
            lines.append(f"\n  {WF_TITLES[wk]}:")
            unit = "pp" if wk in ("ef1", "ef1_plus") else "% of max"
            div = L3_NORM[wk]
            scale = 100.0

            entries = []
            for s, r in sorted(pvals[wk].items(),
                                key=lambda x: abs(x[1]["mean_diff"]),
                                reverse=True):
                mt = r["mean_true"] / div * scale
                mf = r["mean_false"] / div * scale
                md = r["mean_diff"] / div * scale
                pv = r["pvalue"]
                if pv < 0.001:
                    sig = "***"
                elif pv < 0.01:
                    sig = "** "
                elif pv < 0.05:
                    sig = "*  "
                else:
                    sig = "   "
                dn = DISPLAY_NAMES.get(s, s)
                entries.append(
                    f"    {sig} {dn:>16s}  "
                    f"true={mt:+7.2f}  false={mf:+7.2f}  "
                    f"diff={md:+7.2f} {unit}  p={pv:.4f}"
                )
            lines.extend(entries)

    summary_path = os.path.join(save_dir, "significance_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    base = Path(__file__).resolve().parent.parent
    pkl_path = base / "data" / "analysis" / "iterative_analysis_results.pkl"
    save_dir = base / "data" / "analysis" / "banzhaf_conditions"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    raw = results["raw"]
    print(f"  Loaded {len(raw)} bootstrap samples")

    sample0_details = raw[0]["l3"]["coalition_details"]
    wf_keys = [wk for wk in L3_NORM if wk in sample0_details[0]["marginals"]]
    print(f"  Welfare keys: {wf_keys}")

    all_results = {}

    for cond_name, cond_cfg in CONDITIONS.items():
        print(f"\n{'=' * 50}")
        print(f"Condition: {cond_name}")
        exclude = cond_cfg.get("exclude_policies", set())
        if exclude:
            print(f"  Excluding: {exclude}")

        # Compute conditional Banzhaf
        cond_data = compute_conditional_banzhaf(
            raw, cond_cfg["fn"], wf_keys, exclude_policies=exclude
        )

        # Statistical tests
        pval_results = compute_pvalues(cond_data, wf_keys)

        # Print significant results
        for wk in wf_keys:
            for s, r in sorted(pval_results[wk].items(),
                                key=lambda x: x[1]["pvalue"]):
                div = L3_NORM[wk]
                scale = 100.0
                mt = r["mean_true"] / div * scale
                mf = r["mean_false"] / div * scale
                sig = ""
                if r["pvalue"] < 0.001:
                    sig = "***"
                elif r["pvalue"] < 0.01:
                    sig = "**"
                elif r["pvalue"] < 0.05:
                    sig = "*"
                dn = DISPLAY_NAMES.get(s, s)
                print(
                    f"  {dn:>16s} | {WF_TITLES[wk]:>20s} | "
                    f"true={mt:+7.2f}  false={mf:+7.2f}  "
                    f"p={r['pvalue']:.4f} {sig}"
                )

        # Plot
        fig_path = str(save_dir / f"fig_cbanzhaf_{cond_name}.png")
        plot_conditional_beeswarm(
            cond_data, cond_cfg, wf_keys, fig_path, pval_results
        )
        print(f"  Saved {fig_path}")

        # Save raw p-value results as JSON
        json_path = str(save_dir / f"pvalues_{cond_name}.json")
        # Convert for JSON serialization (numpy types → Python natives)
        pval_json = {}
        for wk in pval_results:
            pval_json[wk] = {}
            for s, r in pval_results[wk].items():
                pval_json[wk][DISPLAY_NAMES.get(s, s)] = {
                    k: bool(v) if isinstance(v, (np.bool_,)) else
                       float(v) if isinstance(v, (np.floating,)) else
                       int(v) if isinstance(v, (np.integer,)) else v
                    for k, v in r.items()
                }
        with open(json_path, "w") as f:
            json.dump(pval_json, f, indent=2)

        all_results[cond_name] = {
            "pvalues": pval_results,
            "cond_data": cond_data,
        }

    # Write overall summary
    write_summary(all_results, str(save_dir))

    # Decomposition bar charts for key welfare metrics
    decomp_cond_data = {
        cond_name: all_results[cond_name]["cond_data"]
        for cond_name, _, _, _ in DECOMP_CONDITIONS
        if cond_name in all_results
    }
    for wk in ["nw_plus", "ef1"]:
        fig_path = str(save_dir / f"fig_decomposition_{wk}.png")
        plot_decomposition_bar(decomp_cond_data, wk, fig_path)
        print(f"  Saved {fig_path}")

    print(f"\nAll results saved to {save_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
