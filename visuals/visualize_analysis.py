"""
Visualize iterative game-analysis results.

Loads aggregated results from JSON (and raw per-bootstrap data from pkl)
and generates PNG figures covering equilibrium, regret, per-agent values,
L1 partner lift (payoff + EF1), L2 ecosystem impact (welfare + EF1),
L3 Banzhaf/Shapley attribution bar charts (Utilitarian/Nash/NW+/EF1),
EF1 fairness, and Banzhaf/Shapley beeswarm plots.

All welfare metrics are normalized by global maximum constants from the
game distribution (matching the original paper presentation).

Usage:
    python visuals/visualize_analysis.py
"""

import json
import pickle
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

STRATEGY_ORDER = [
    "walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro",
    "openai_5.2_none", "openai_5.2_low", "ef1_bargainer"
]

DISPLAY_NAMES = {
    "walk": "walk",
    "tough": "tough",
    "nfsp": "nfsp",
    "mappo": "mappo",
    "soft": "soft",
    "ppo": "ppo",
    "psro": "psro",
    "openai_5.2_none": "5.2-none",
    "openai_5.2_low": "5.2-low",
    "openai_5.2_medium": "5.2-med",
    "openai_5.4_low": "5.4-low",
    "openai_5.4_medium": "5.4-med",
    "ef1_bargainer": "ef1",
    "aspiration": "aspire",
}

_TAB10 = plt.cm.tab10.colors
STRATEGY_COLORS = {s: _TAB10[i] for i, s in enumerate(STRATEGY_ORDER)}

POS_COLOR = "#d62728"  # SHAP-style red for positive
NEG_COLOR = "#1f77b4"  # SHAP-style blue for negative

DEFAULT_DPI = 200
DEFAULT_FIGSIZE = (10, 6)

# Normalization constants: expected maximum welfare over the game distribution
# (from evaluation/original_paper_analysis.py)
MAX_UW = 805.9    # Expected max utilitarian welfare (sum of raw payoffs)
MAX_NW = 378.7    # Expected max Nash welfare
MAX_NW_PLUS = 81.7  # Expected max NW on advantages

# Maps welfare key -> (divisor, display label)
WELFARE_NORM = {
    "uw": (MAX_UW, "Utilitarian"),
    "nw": (MAX_NW, "Nash"),
    "nw_plus": (MAX_NW_PLUS, "NW+"),
}

# Maps per-agent-value key -> normalization divisor
PAV_NORM = {
    "payoff": MAX_UW,
    "nw": MAX_NW,
    "nw_plus": MAX_NW_PLUS,
}

# Maps L3 suffix -> normalization divisor
L3_NORM = {
    "uw": MAX_UW,
    "nw": MAX_NW,
    "nw_plus": MAX_NW_PLUS,
    "ef1": 1.0,  # EF1 is already a frequency, scale ×100 for %
    "ef1_plus": 1.0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dn(strategy: str) -> str:
    """Return display name for a strategy."""
    return DISPLAY_NAMES.get(strategy, strategy)


def _norm_pct(val, divisor):
    """Normalize a raw value to percentage of max."""
    return val / divisor * 100


def _yerr(means, ci_lo, ci_hi):
    """Compute clamped error bars from means and CI bounds."""
    lo = [max(0, m - l) for m, l in zip(means, ci_lo)]
    hi = [max(0, h - m) for m, h in zip(means, ci_hi)]
    return [lo, hi]


def load_results(base_dir: str | Path | None = None):
    """Load aggregated JSON and raw pkl results.

    Returns (aggregated_dict, raw_list, config_dict).
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "data" / "analysis"
    base_dir = Path(base_dir)

    with open(base_dir / "iterative_analysis_results.json") as f:
        js = json.load(f)

    with open(base_dir / "iterative_analysis_results.pkl", "rb") as f:
        pk = pickle.load(f)

    agg = js["aggregated"]
    raw = pk["raw"]
    config = js.get("config", pk.get("config", {}))
    return agg, raw, config


# ---------------------------------------------------------------------------
# Figure 1 – Equilibrium Distribution
# ---------------------------------------------------------------------------


def plot_equilibrium_distribution(agg, save_dir):
    strategies = agg["strategy_names"]
    eq_mean = agg["full_game"]["equilibrium"]["mean"]
    eq_std = agg["full_game"]["equilibrium"]["std"]
    support_freq = agg["full_game"]["support_frequency"]

    n = len(strategies)
    x = np.arange(n)
    colors = [STRATEGY_COLORS[s] for s in strategies]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.bar(x, eq_mean, yerr=eq_std, capsize=4, color=colors,
           edgecolor="black", linewidth=0.5)

    for i, s in enumerate(strategies):
        freq = support_freq.get(s, 0)
        if freq > 0:
            ax.text(i, eq_mean[i] + eq_std[i] + 0.01,
                    f"{freq:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0 / n, ls="--", color="grey", lw=1, label=f"Uniform (1/{n})")
    ax.set_xticks(x)
    ax.set_xticklabels([_dn(s) for s in strategies], rotation=30, ha="right")
    ax.set_ylabel("Equilibrium Weight")
    ax.set_title("Equilibrium Distribution (MENE)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_equilibrium_distribution.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 – Strategy Regret
# ---------------------------------------------------------------------------


def plot_strategy_regret(agg, save_dir):
    regret = agg["full_game"]["regret"]
    strategies = list(regret.keys())

    strategies = sorted(strategies, key=lambda s: regret[s]["mean"])
    means = [regret[s]["mean"] for s in strategies]
    ci_lo = [regret[s]["ci_lower"] for s in strategies]
    ci_hi = [regret[s]["ci_upper"] for s in strategies]
    y = np.arange(len(strategies))
    colors = [STRATEGY_COLORS[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y, means, xerr=_yerr(means, ci_lo, ci_hi), capsize=3,
            color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_dn(s) for s in strategies])
    ax.set_xlabel("Regret (raw utility)")
    ax.set_title("Strategy Regret")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig2_strategy_regret.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 – Per-Agent Values (Payoff / NW / NW+) — normalized %
# ---------------------------------------------------------------------------


def plot_per_agent_values(agg, save_dir):
    pav = agg["full_game"]["per_agent_values"]
    strategies = agg["strategy_names"]
    metrics = ["payoff", "nw", "nw_plus"]
    metric_labels = ["Utilitarian", "Nash", "NW+"]
    metric_colors = ["#4c72b0", "#55a868", "#c44e52"]

    n = len(strategies)
    width = 0.25
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, (met, label, col) in enumerate(zip(metrics, metric_labels, metric_colors)):
        div = PAV_NORM[met]
        means = [_norm_pct(pav[met][s]["mean"], div) for s in strategies]
        ci_lo = [_norm_pct(pav[met][s]["ci_lower"], div) for s in strategies]
        ci_hi = [_norm_pct(pav[met][s]["ci_upper"], div) for s in strategies]
        ax.bar(x + j * width, means, width, yerr=_yerr(means, ci_lo, ci_hi),
               capsize=3, label=label, color=col, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x + width)
    ax.set_xticklabels([_dn(s) for s in strategies], rotation=30, ha="right")
    ax.set_ylabel("% of Maximum Welfare")
    ax.set_title("Per-Agent Welfare at Equilibrium (normalized)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_per_agent_values.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 – L1 Partner Lift Heatmaps (one per metric)
# ---------------------------------------------------------------------------

# Config for each L1 heatmap: (metric_key, norm_divisor, metric_label, unit, filename)
_L1_HEATMAP_CFG = [
    ("payoff",  MAX_UW,     "Payoff",        "% of max UW",  "fig4a_l1_payoff_heatmap.png"),
    ("nw",      MAX_NW,     "Nash Welfare",  "% of max NW",  "fig4b_l1_nw_heatmap.png"),
    ("nw_plus", MAX_NW_PLUS,"NW+ Welfare",   "% of max NW+", "fig4c_l1_nw_plus_heatmap.png"),
    ("ef1",     1.0,        "EF1 Frequency", "pp",            "fig4d_l1_ef1_heatmap.png"),
]


def _plot_l1_heatmap(agg, save_dir, metric_key, norm_div, metric_label, unit, filename):
    """Generic L1 partner-lift heatmap for any per-incumbent metric.

    Each cell (row=candidate, col=incumbent) shows:
        M[incumbent, candidate]  −  M[incumbent, :] @ σ_B

    where σ_B is the equilibrium of the restricted game without the candidate.

    Positive (red): incumbent gets MORE from the candidate than from its
        baseline expected value → candidate is a good partner.
    Negative (blue): incumbent gets LESS → candidate is a bad partner.
    """
    strategies = agg["strategy_names"]
    l1 = agg["l1"]
    n = len(strategies)

    mat = np.full((n, n), np.nan)
    for i, cand in enumerate(strategies):
        metric_data = l1[cand]["per_incumbent"].get(metric_key, {})
        for j, inc in enumerate(strategies):
            if inc in metric_data:
                raw = metric_data[inc]["mean"]
                if metric_key == "ef1":
                    mat[i, j] = raw * 100
                else:
                    mat[i, j] = _norm_pct(raw, norm_div)

    labels_x = [_dn(s) for s in strategies]
    labels_y = [_dn(s) for s in strategies]
    mask = np.isnan(mat)
    vmax = max(np.nanmax(np.abs(mat[~mask])), 1e-6) if not np.all(mask) else 1

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(mat, mask=mask, annot=True, fmt=".1f",
                cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                xticklabels=labels_x, yticklabels=labels_y,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": f"Partner Lift ({unit})"})

    ax.set_xlabel("Incumbent (whose value is measured)", fontsize=11)
    ax.set_ylabel("Candidate (paired against incumbent)", fontsize=11)
    ax.set_title(
        f"L1 Partner Lift: {metric_label}\n"
        f"Cell = M[incumbent, candidate] − E[incumbent | eq. without candidate]",
        fontsize=12,
    )

    # Annotation below the plot explaining the color scale
    fig.text(
        0.5, -0.02,
        "Red (+): candidate is a better partner than baseline  |  "
        "Blue (−): candidate is a worse partner than baseline",
        ha="center", fontsize=9, style="italic",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_l1_heatmaps(agg, save_dir):
    """Generate all L1 heatmaps (payoff, nw, nw_plus, ef1)."""
    for metric_key, norm_div, metric_label, unit, filename in _L1_HEATMAP_CFG:
        _plot_l1_heatmap(agg, save_dir, metric_key, norm_div,
                         metric_label, unit, filename)


# ---------------------------------------------------------------------------
# Figure 5 – L2 Ecosystem Impact (normalized welfare + EF1)
# ---------------------------------------------------------------------------


def plot_l2_ecosystem_impact(agg, save_dir):
    l2 = agg["l2"]
    strategies = agg["strategy_names"]

    in_support = [s for s in strategies
                  if l2[s]["entry_mass"]["mean"] > 1e-6]

    if not in_support:
        in_support = strategies
        note = "(No strategy has entry mass > 1e-6; showing all)"
    else:
        omitted = [_dn(s) for s in strategies if s not in in_support]
        note = f"Out-of-support strategies omitted: {', '.join(omitted)}" if omitted else ""

    n = len(in_support)
    x = np.arange(n)
    labels = [_dn(s) for s in in_support]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Left panel: delta_eco by welfare type (normalized %)
    welfare_keys = ["uw", "nw", "nw_plus"]
    welfare_labels = ["Utilitarian", "Nash", "NW+"]
    welfare_colors = ["#4c72b0", "#55a868", "#c44e52"]
    width = 0.25

    for j, (wk, wl, wc) in enumerate(zip(welfare_keys, welfare_labels, welfare_colors)):
        div = WELFARE_NORM[wk][0]
        means = [_norm_pct(l2[s]["delta_eco"][wk]["mean"], div) for s in in_support]
        ci_lo = [_norm_pct(l2[s]["delta_eco"][wk]["ci_lower"], div) for s in in_support]
        ci_hi = [_norm_pct(l2[s]["delta_eco"][wk]["ci_upper"], div) for s in in_support]
        ax1.bar(x + j * width, means, width, yerr=_yerr(means, ci_lo, ci_hi),
                capsize=3, label=wl, color=wc, edgecolor="black", linewidth=0.4)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_ylabel("Delta Welfare (% of max)")
    ax1.set_title("L2: Ecosystem Welfare Impact")
    ax1.legend(fontsize=8)

    # Middle panel: entry_mass and equilibrium_shift
    em_means = [l2[s]["entry_mass"]["mean"] for s in in_support]
    em_ci_lo = [l2[s]["entry_mass"]["ci_lower"] for s in in_support]
    em_ci_hi = [l2[s]["entry_mass"]["ci_upper"] for s in in_support]
    es_means = [l2[s]["equilibrium_shift"]["mean"] for s in in_support]
    es_ci_lo = [l2[s]["equilibrium_shift"]["ci_lower"] for s in in_support]
    es_ci_hi = [l2[s]["equilibrium_shift"]["ci_upper"] for s in in_support]

    w2 = 0.35
    ax2.bar(x - w2 / 2, em_means, w2, yerr=_yerr(em_means, em_ci_lo, em_ci_hi),
            capsize=3, label="Entry Mass", color="#4c72b0",
            edgecolor="black", linewidth=0.4)
    ax2.bar(x + w2 / 2, es_means, w2, yerr=_yerr(es_means, es_ci_lo, es_ci_hi),
            capsize=3, label="Eq. Shift", color="#dd8452",
            edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Value")
    ax2.set_title("L2: Entry Mass & Equilibrium Shift")
    ax2.legend(fontsize=8)

    # Right panel: EF1 lift (percentage points)
    ef1_means = [l2[s]["ef1_lift"]["mean"] * 100 for s in in_support]
    ef1_ci_lo = [l2[s]["ef1_lift"]["ci_lower"] * 100 for s in in_support]
    ef1_ci_hi = [l2[s]["ef1_lift"]["ci_upper"] * 100 for s in in_support]
    colors_ef1 = [STRATEGY_COLORS[s] for s in in_support]

    ax3.bar(x, ef1_means, 0.5, yerr=_yerr(ef1_means, ef1_ci_lo, ef1_ci_hi),
            capsize=3, color=colors_ef1, edgecolor="black", linewidth=0.4)
    ax3.axhline(0, color="black", lw=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=30, ha="right")
    ax3.set_ylabel("Delta EF1 Frequency (pp)")
    ax3.set_title("L2: EF1 Frequency Lift")

    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_l2_ecosystem_impact.png"),
                dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic L3 attribution bar chart (SHAP-style)
# ---------------------------------------------------------------------------


def _plot_l3_bar(agg, save_dir, method, filename):
    """Plot SHAP-style horizontal bar chart for Shapley or Banzhaf.

    Produces N subplots: one per welfare key found (uw, nw, nw_plus, ef1).
    Values for uw/nw/nw_plus are normalized to % of max welfare.
    EF1 values are scaled ×100 (percentage points).
    """
    l3 = agg["l3"]
    total_value = l3.get("total_value", {})

    # Discover available keys for this method
    all_wf_keys = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]
    wf_titles = {
        "uw": "Utilitarian Welfare",
        "nw": "Nash Welfare",
        "nw_plus": "NW+ Welfare",
        "ef1": "EF1 Frequency",
        "ef1_plus": "EF1+ Frequency",
    }
    available = [wk for wk in all_wf_keys if f"{method}_{wk}" in l3]
    if not available:
        return

    n_panels = len(available)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, wk in zip(axes, available):
        key = f"{method}_{wk}"
        data = l3[key]
        strategies = list(data.keys())

        div = L3_NORM[wk]
        scale = 100.0  # show as %

        strategies = sorted(strategies, key=lambda s: abs(data[s]["mean"] / div),
                            reverse=True)
        means = [data[s]["mean"] / div * scale for s in strategies]
        ci_lo = [data[s]["ci_lower"] / div * scale for s in strategies]
        ci_hi = [data[s]["ci_upper"] / div * scale for s in strategies]
        bar_colors = [POS_COLOR if m >= 0 else NEG_COLOR for m in means]
        y = np.arange(len(strategies))
        ax.barh(y, means, xerr=_yerr(means, ci_lo, ci_hi), capsize=3,
                color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([_dn(s) for s in strategies])
        ax.invert_yaxis()

        # Total value annotation
        tv = total_value.get(wk, {})
        if isinstance(tv, dict) and "mean" in tv:
            tv_pct = tv["mean"] / div * scale
            tv_str = f"  (total = {tv_pct:.1f}%)"
        else:
            tv_str = ""
        label = method.capitalize()
        unit = "pp" if wk in ("ef1", "ef1_plus") else "% of max"
        ax.set_title(f"{label} – {wf_titles[wk]}{tv_str}")
        ax.set_xlabel(f"{label} Value ({unit})")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_l3_banzhaf_bar(agg, save_dir):
    _plot_l3_bar(agg, save_dir, "banzhaf", "fig6_l3_banzhaf_attribution.png")


def plot_l3_shapley_bar(agg, save_dir):
    _plot_l3_bar(agg, save_dir, "shapley", "fig9_l3_shapley_attribution.png")


# ---------------------------------------------------------------------------
# Figure 7 – EF1 Fairness
# ---------------------------------------------------------------------------


def plot_ef1_fairness(agg, save_dir):
    pav = agg["full_game"]["per_agent_values"]
    ef1_data = pav["ef1"]
    strategies = agg["strategy_names"]
    full_ef1 = agg["full_game"]["ef1"]

    means = [ef1_data[s]["mean"] * 100 for s in strategies]
    ci_lo = [ef1_data[s]["ci_lower"] * 100 for s in strategies]
    ci_hi = [ef1_data[s]["ci_upper"] * 100 for s in strategies]
    full_ef1_pct = full_ef1["mean"] * 100

    x = np.arange(len(strategies))
    colors = [STRATEGY_COLORS[s] for s in strategies]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.bar(x, means, yerr=_yerr(means, ci_lo, ci_hi), capsize=4,
           color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(full_ef1_pct, ls="--", color="grey", lw=1,
               label=f"Full-game EF1 = {full_ef1_pct:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([_dn(s) for s in strategies], rotation=30, ha="right")
    ax.set_ylabel("EF1 Frequency (%)")
    ax.set_title("EF1 Frequency per Strategy at Equilibrium")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig7_ef1_fairness.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic L3 beeswarm plot
# ---------------------------------------------------------------------------


def _plot_l3_beeswarm(agg, raw, save_dir, method, filename):
    """Plot beeswarm for Shapley or Banzhaf, one subplot per welfare key.

    Values are normalized the same way as the bar charts.
    """
    all_wf_keys = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]
    wf_titles = {
        "uw": "Utilitarian Welfare",
        "nw": "Nash Welfare",
        "nw_plus": "NW+ Welfare",
        "ef1": "EF1 Frequency",
        "ef1_plus": "EF1+ Frequency",
    }
    available = [wk for wk in all_wf_keys if f"{method}_{wk}" in raw[0]["l3"]]
    if not available:
        return

    # Collect per-bootstrap values
    per_key_data = {}
    for wk in available:
        key = f"{method}_{wk}"
        strat_vals = {}
        for sample in raw:
            for s, v in sample["l3"][key].items():
                strat_vals.setdefault(s, []).append(v)
        per_key_data[wk] = strat_vals

    n_panels = len(available)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, wk in zip(axes, available):
        strat_vals = per_key_data[wk]
        strategies = list(strat_vals.keys())
        div = L3_NORM[wk]
        scale = 100.0

        strategies = sorted(strategies,
                            key=lambda s: np.mean(np.abs(np.array(strat_vals[s]) / div * scale)),
                            reverse=True)

        y_positions = np.arange(len(strategies))
        for i, s in enumerate(strategies):
            vals = np.array(strat_vals[s]) / div * scale
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(vals))
            dot_colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
            ax.scatter(vals, i + jitter, c=dot_colors, s=40, alpha=0.7,
                       edgecolors="black", linewidths=0.3, zorder=3)

        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([_dn(s) for s in strategies])
        ax.invert_yaxis()
        label = method.capitalize()
        unit = "pp" if wk in ("ef1", "ef1_plus") else "% of max"
        ax.set_title(f"{label} Beeswarm – {wf_titles[wk]}")
        ax.set_xlabel(f"{label} Value ({unit})")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_l3_banzhaf_beeswarm(agg, raw, save_dir):
    _plot_l3_beeswarm(agg, raw, save_dir, "banzhaf",
                      "fig8_l3_banzhaf_beeswarm.png")


def plot_l3_banzhaf_beeswarm_no_singleton(agg, raw, save_dir):
    """Beeswarm recomputed from coalition_details, excluding the singleton
    (empty coalition S=∅) so that the trivial v({i})-v(∅) term is removed."""
    all_wf_keys = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]
    wf_titles = {
        "uw": "Utilitarian Welfare",
        "nw": "Nash Welfare",
        "nw_plus": "NW+ Welfare",
        "ef1": "EF1 Frequency",
        "ef1_plus": "EF1+ Frequency",
    }

    # Check which welfare keys are available in coalition_details
    sample0_details = raw[0]["l3"].get("coalition_details", [])
    if not sample0_details:
        return
    available_wf = [wk for wk in all_wf_keys if wk in sample0_details[0]["marginals"]]
    if not available_wf:
        return

    # Recompute per-bootstrap Banzhaf excluding singleton
    per_key_data = {wk: {} for wk in available_wf}
    for sample in raw:
        # Group marginals by policy, excluding empty coalition
        policy_marginals = {}  # policy -> {wk: [marginals]}
        for r in sample["l3"]["coalition_details"]:
            if not r["coalition_without"]:  # skip singleton (S = ∅)
                continue
            p = r["policy"]
            if p not in policy_marginals:
                policy_marginals[p] = {wk: [] for wk in available_wf}
            for wk in available_wf:
                policy_marginals[p][wk].append(r["marginals"][wk])

        for p, wk_marginals in policy_marginals.items():
            for wk in available_wf:
                banzhaf = np.mean(wk_marginals[wk])
                per_key_data[wk].setdefault(p, []).append(banzhaf)

    n_panels = len(available_wf)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, wk in zip(axes, available_wf):
        strat_vals = per_key_data[wk]
        strategies = list(strat_vals.keys())
        div = L3_NORM[wk]
        scale = 100.0

        strategies = sorted(
            strategies,
            key=lambda s: np.mean(np.abs(np.array(strat_vals[s]) / div * scale)),
            reverse=True,
        )

        y_positions = np.arange(len(strategies))
        for i, s in enumerate(strategies):
            vals = np.array(strat_vals[s]) / div * scale
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(vals))
            dot_colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
            ax.scatter(vals, i + jitter, c=dot_colors, s=40, alpha=0.7,
                       edgecolors="black", linewidths=0.3, zorder=3)

        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([_dn(s) for s in strategies])
        ax.invert_yaxis()
        unit = "pp" if wk in ("ef1", "ef1_plus") else "% of max"
        ax.set_title(f"Banzhaf Beeswarm (excl. singleton) – {wf_titles[wk]}")
        ax.set_xlabel(f"Banzhaf Value ({unit})")

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, "fig8b_l3_banzhaf_beeswarm_no_singleton.png"),
        dpi=DEFAULT_DPI,
    )
    plt.close(fig)


def plot_l3_shapley_beeswarm(agg, raw, save_dir):
    _plot_l3_beeswarm(agg, raw, save_dir, "shapley",
                      "fig10_l3_shapley_beeswarm.png")


# ---------------------------------------------------------------------------
# Matrix key -> welfare key mapping
# ---------------------------------------------------------------------------

_MATRIX_TO_WF = {
    "payoff": ("uw", "Utilitarian Welfare", MAX_UW),
    "nw":     ("nw", "Nash Welfare",        MAX_NW),
    "nw_plus":("nw_plus", "NW+ Welfare",    MAX_NW_PLUS),
    "ef1":    ("ef1", "EF1 Frequency",      1.0),
}


# ---------------------------------------------------------------------------
# Image Plot Variant 1 – Cell-Level Equilibrium Contributions
# ---------------------------------------------------------------------------


def plot_cell_contributions(raw, config, save_dir):
    """For each welfare metric, show raw matrix + cell contribution (σ_i·M_ij·σ_j).

    Generates one 2-panel figure per metric.
    """
    strategies = config["strategy_names"]
    labels = [_dn(s) for s in strategies]
    n = len(strategies)
    n_boot = len(raw)

    for mat_key, (wf_key, wf_label, norm_div) in _MATRIX_TO_WF.items():
        # Average matrix and sigma across bootstrap samples
        matrices = []
        sigmas = []
        for sample in raw:
            M = np.array(sample["matrices"][mat_key], dtype=float)
            sigma = np.array(sample["full_game"]["sigma"], dtype=float)
            matrices.append(M)
            sigmas.append(sigma)

        avg_M = np.nanmean(matrices, axis=0)
        avg_sigma = np.mean(sigmas, axis=0)

        # Cell contributions: σ_i · M_ij · σ_j
        contrib = np.outer(avg_sigma, avg_sigma) * avg_M

        # Normalize
        if wf_key == "ef1":
            raw_display = avg_M * 100  # show as %
            contrib_display = contrib * 100
            raw_cbar = f"{wf_label} (%)"
            contrib_cbar = "Contribution (pp)"
        else:
            raw_display = avg_M / norm_div * 100
            contrib_display = contrib / norm_div * 100
            raw_cbar = f"{wf_label} (% of max)"
            contrib_cbar = "Contribution (% of max)"

        mask = np.isnan(raw_display)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Left: raw matrix
        vmax_raw = np.nanmax(np.abs(raw_display[~mask])) if not np.all(mask) else 1
        sns.heatmap(raw_display, mask=mask, annot=True, fmt=".1f",
                    cmap="YlOrRd", xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax1,
                    cbar_kws={"label": raw_cbar})
        ax1.set_title(f"Raw {wf_label} Matrix")
        ax1.set_xlabel("Strategy (column)")
        ax1.set_ylabel("Strategy (row)")

        # Right: cell contributions
        contrib_masked = np.where(mask, np.nan, contrib_display)
        vmax_c = np.nanmax(np.abs(contrib_masked[~np.isnan(contrib_masked)])) if not np.all(np.isnan(contrib_masked)) else 1
        # Use sequential colormap since contributions are non-negative
        sns.heatmap(contrib_display, mask=mask, annot=True, fmt=".2f",
                    cmap="Reds", xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax2,
                    cbar_kws={"label": contrib_cbar})
        ax2.set_title(f"Cell Contributions at Equilibrium (σ_i · M_ij · σ_j)")
        ax2.set_xlabel("Strategy (column)")
        ax2.set_ylabel("Strategy (row)")

        fig.suptitle(f"{wf_label}: Raw Matrix vs Equilibrium Contributions", fontsize=13, y=1.01)
        fig.tight_layout()
        fname = f"fig11_{wf_key}_cell_contributions.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Image Plot Variant 2 – Matrix with Shapley/Banzhaf Margins
# ---------------------------------------------------------------------------


def plot_matrix_with_margins(agg, raw, config, save_dir):
    """For each welfare metric, show raw matrix with Shapley bars on top,
    Banzhaf bars on the right, and equilibrium weights at bottom.

    Generates one figure per metric.
    """
    strategies = config["strategy_names"]
    labels = [_dn(s) for s in strategies]
    n = len(strategies)
    l3 = agg["l3"]

    # Average matrices and sigma across bootstrap samples
    avg_matrices = {}
    avg_sigma = np.mean([np.array(s["full_game"]["sigma"]) for s in raw], axis=0)
    for mat_key in _MATRIX_TO_WF:
        avg_matrices[mat_key] = np.nanmean(
            [np.array(s["matrices"][mat_key], dtype=float) for s in raw], axis=0
        )

    for mat_key, (wf_key, wf_label, norm_div) in _MATRIX_TO_WF.items():
        shap_key = f"shapley_{wf_key}"
        banz_key = f"banzhaf_{wf_key}"
        if shap_key not in l3 or banz_key not in l3:
            continue

        avg_M = avg_matrices[mat_key]

        # Get Shapley and Banzhaf values (normalized)
        shap_vals = [l3[shap_key][s]["mean"] for s in strategies]
        banz_vals = [l3[banz_key][s]["mean"] for s in strategies]

        if wf_key == "ef1":
            mat_display = avg_M * 100
            shap_display = np.array([v * 100 for v in shap_vals])
            banz_display = np.array([v * 100 for v in banz_vals])
            mat_cbar = f"{wf_label} (%)"
            margin_unit = "pp"
        else:
            mat_display = avg_M / norm_div * 100
            shap_display = np.array([v / norm_div * 100 for v in shap_vals])
            banz_display = np.array([v / norm_div * 100 for v in banz_vals])
            mat_cbar = f"{wf_label} (% of max)"
            margin_unit = "% of max"

        mask = np.isnan(mat_display)

        # --- Layout: 3 rows × 3 cols via gridspec ---
        #   [top-left: Shapley bars] [top-right: empty      ]
        #   [mid-left: heatmap     ] [mid-right: Banzhaf bars]
        #   [bot-left: sigma bars  ] [bot-right: empty       ]
        fig = plt.figure(figsize=(14, 11))
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[1.2, 6, 1],
            width_ratios=[6, 1.5],
            hspace=0.08, wspace=0.08,
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        ax_bottom = fig.add_subplot(gs[2, 0])

        # -- Main heatmap (no built-in colorbar; we add our own) --
        sns.heatmap(
            mat_display, mask=mask, annot=True, fmt=".1f",
            cmap="YlOrRd", xticklabels=False, yticklabels=labels,
            linewidths=0.5, ax=ax_main, cbar=False,
        )
        ax_main.set_ylabel("")
        ax_main.tick_params(axis="y", labelsize=10)

        # Colorbar: thin horizontal bar below the heatmap title area
        sm = plt.cm.ScalarMappable(
            cmap="YlOrRd",
            norm=plt.Normalize(vmin=np.nanmin(mat_display), vmax=np.nanmax(mat_display)),
        )
        cbar_ax = fig.add_axes([0.12, 0.02, 0.45, 0.015])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(mat_cbar, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # -- Top: Shapley bars (aligned to heatmap columns) --
        x = np.arange(n) + 0.5  # center on heatmap cells
        shap_colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in shap_display]
        ax_top.bar(x, shap_display, width=0.7, color=shap_colors,
                   edgecolor="black", linewidth=0.4)
        ax_top.axhline(0, color="black", lw=0.6)
        ax_top.set_xlim(0, n)
        ax_top.set_xticks([])
        ax_top.set_ylabel(f"Shapley\n({margin_unit})", fontsize=9)
        ax_top.set_title(f"{wf_label}: Matrix + Attribution Margins", fontsize=13)
        ax_top.spines["bottom"].set_visible(False)

        # -- Right: Banzhaf bars (aligned to heatmap rows) --
        y = np.arange(n) + 0.5
        banz_colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in banz_display]
        ax_right.barh(y, banz_display, height=0.7, color=banz_colors,
                      edgecolor="black", linewidth=0.4)
        ax_right.axvline(0, color="black", lw=0.6)
        ax_right.set_ylim(n, 0)  # match heatmap y-direction (top to bottom)
        ax_right.set_yticks([])
        ax_right.set_xlabel(f"Banzhaf\n({margin_unit})", fontsize=9)
        ax_right.spines["left"].set_visible(False)

        # -- Bottom: equilibrium weights --
        sigma_colors = [STRATEGY_COLORS[s] for s in strategies]
        ax_bottom.bar(x, avg_sigma, width=0.7, color=sigma_colors,
                      edgecolor="black", linewidth=0.4)
        ax_bottom.set_xlim(0, n)
        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
        ax_bottom.set_ylabel("σ (eq.)", fontsize=9)
        ax_bottom.set_ylim(bottom=0)
        ax_bottom.spines["top"].set_visible(False)

        fname = f"fig12_{wf_key}_matrix_margins.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Decision Plot – Cumulative Shapley Paths
# ---------------------------------------------------------------------------


def plot_shapley_decision(raw, config, save_dir):
    """SHAP-style decision plot: cumulative Shapley contributions.

    Each bootstrap sample is a separate line tracing from 0 to total welfare.
    Strategies on Y-axis sorted by mean |Shapley value|.
    One figure per welfare metric.
    """
    strategies = config["strategy_names"]
    n_strats = len(strategies)

    all_wf_keys = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]
    wf_titles = {
        "uw": "Utilitarian Welfare",
        "nw": "Nash Welfare",
        "nw_plus": "NW+ Welfare",
        "ef1": "EF1 Frequency",
        "ef1_plus": "EF1+ Frequency",
    }

    available = [wk for wk in all_wf_keys if f"shapley_{wk}" in raw[0]["l3"]]
    if not available:
        return

    for wf_key in available:
        shap_key = f"shapley_{wf_key}"
        div = L3_NORM[wf_key]
        scale = 100.0
        unit = "% of max" if wf_key != "ef1" else "pp"

        # Collect per-bootstrap Shapley vectors
        all_shap = []  # list of dicts
        all_totals = []
        for sample in raw:
            sv = {s: sample["l3"][shap_key][s] / div * scale for s in strategies}
            tv = sample["l3"]["total_value"][wf_key] / div * scale
            all_shap.append(sv)
            all_totals.append(tv)

        # Sort strategies by mean |Shapley value| (most important at top)
        mean_abs = {s: np.mean([abs(sv[s]) for sv in all_shap]) for s in strategies}
        sorted_strats = sorted(strategies, key=lambda s: mean_abs[s])  # ascending (bottom=smallest)

        fig, ax = plt.subplots(figsize=(10, 7))

        n_boot = len(raw)
        cmap = plt.cm.viridis
        line_colors = [cmap(i / max(n_boot - 1, 1)) for i in range(n_boot)]

        for b, sv in enumerate(all_shap):
            # Cumulative path: start from 0, add each strategy's contribution
            cumulative = [0.0]
            for s in sorted_strats:
                cumulative.append(cumulative[-1] + sv[s])

            # Y positions: one per strategy + base
            y_pos = np.arange(n_strats + 1)
            ax.plot(cumulative, y_pos, marker="o", markersize=4,
                    color=line_colors[b], alpha=0.7, linewidth=1.5,
                    label=f"Bootstrap {b+1}" if n_boot <= 10 else None)

            # Mark final value
            ax.plot(cumulative[-1], y_pos[-1], "D", color=line_colors[b],
                    markersize=6, zorder=5)

        # Y-axis labels
        y_labels = ["Base (0)"] + [_dn(s) for s in sorted_strats]
        ax.set_yticks(np.arange(n_strats + 1))
        ax.set_yticklabels(y_labels)
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)

        # Mean total line
        mean_total = np.mean(all_totals)
        ax.axvline(mean_total, color="grey", lw=1.5, ls=":",
                   label=f"Mean total = {mean_total:.1f}{unit}")

        ax.set_xlabel(f"Cumulative Shapley Value ({unit})")
        ax.set_title(f"Decision Plot – {wf_titles[wf_key]}")
        if n_boot <= 10:
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.legend(fontsize=8, loc="lower right", ncol=2)

        fig.tight_layout()
        fname = f"fig13_{wf_key}_decision_plot.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=DEFAULT_DPI)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_all_figures(data_dir=None, fig_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "analysis"
    if fig_dir is None:
        fig_dir = Path(data_dir) / "figures"

    os.makedirs(fig_dir, exist_ok=True)
    fig_dir = str(fig_dir)

    agg, raw, config = load_results(data_dir)

    figs = [
        ("fig1_equilibrium_distribution",    lambda: plot_equilibrium_distribution(agg, fig_dir)),
        ("fig2_strategy_regret",             lambda: plot_strategy_regret(agg, fig_dir)),
        ("fig3_per_agent_values",            lambda: plot_per_agent_values(agg, fig_dir)),
        ("fig4_l1_heatmaps (×4)",            lambda: plot_l1_heatmaps(agg, fig_dir)),
        ("fig5_l2_ecosystem_impact",         lambda: plot_l2_ecosystem_impact(agg, fig_dir)),
        ("fig6_l3_banzhaf_attribution",      lambda: plot_l3_banzhaf_bar(agg, fig_dir)),
        ("fig7_ef1_fairness",                lambda: plot_ef1_fairness(agg, fig_dir)),
        ("fig8_l3_banzhaf_beeswarm",         lambda: plot_l3_banzhaf_beeswarm(agg, raw, fig_dir)),
        ("fig8b_l3_banzhaf_beeswarm_no_sing", lambda: plot_l3_banzhaf_beeswarm_no_singleton(agg, raw, fig_dir)),
        ("fig9_l3_shapley_attribution",      lambda: plot_l3_shapley_bar(agg, fig_dir)),
        ("fig10_l3_shapley_beeswarm",        lambda: plot_l3_shapley_beeswarm(agg, raw, fig_dir)),
        ("fig11_cell_contributions (×4)",    lambda: plot_cell_contributions(raw, config, fig_dir)),
        ("fig12_matrix_margins (×4)",        lambda: plot_matrix_with_margins(agg, raw, config, fig_dir)),
        ("fig13_decision_plots (×4)",        lambda: plot_shapley_decision(raw, config, fig_dir)),
    ]

    n = len(figs)
    print(f"Generating figures …")
    for i, (name, fn) in enumerate(figs, 1):
        fn()
        print(f"  [{i:2d}/{n}] {name}")

    print(f"\nAll {n} figures saved to {fig_dir}/")


if __name__ == "__main__":
    generate_all_figures()
