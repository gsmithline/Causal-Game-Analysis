"""
L2.5 Best-Response Graph Causal Analysis.

Constructs epsilon-best-response graphs from the payoff matrix and applies
the do-operator (add/remove an agent) to make causal claims about how each
agent reshapes the strategic ecology: which cycles it creates/destroys,
which agents it makes strategically relevant/irrelevant, and how the
interaction topology changes.

Usage:
    python visuals/l25_br_graph_analysis.py [--sample-only] [--eps 0.01] [--top-k 3]
"""

import argparse
import pickle
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import networkx as nx
from scipy.linalg import eigvals, eig

# Reuse style constants from the main visualization module
from visualize_analysis import (
    STRATEGY_ORDER, DISPLAY_NAMES, STRATEGY_COLORS,
    DEFAULT_DPI, DEFAULT_FIGSIZE, _dn,
)

# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def br_graph(payoff_matrix, policies, mode="eps", eps=0.01, top_k=None):
    """Build an epsilon-best-response directed graph.

    Edge i -> j means j is an epsilon-best-response to opponent i:
    when the opponent plays pure strategy i, j is within eps of the
    best response payoff.

    Args:
        payoff_matrix: (n, n) array where M[j, i] is payoff for policy j
            against opponent policy i.
        policies: list of policy names (length n).
        mode: "eps" for epsilon-ball or "top_k" for top-k BRs.
        eps: epsilon tolerance as fraction of payoff range (used if mode="eps").
        top_k: number of top BRs to include (used if mode="top_k").

    Returns:
        networkx.DiGraph with policies as nodes.
    """
    M = np.nan_to_num(payoff_matrix, nan=0.0)
    n = len(policies)
    G = nx.DiGraph()
    G.add_nodes_from(policies)

    if mode == "eps":
        payoff_range = M.max() - M.min()
        eps_abs = eps * payoff_range if payoff_range > 0 else 0.0

        for i in range(n):
            # Column i: payoffs for all policies against opponent i
            col = M[:, i]
            best = col.max()
            for j in range(n):
                if col[j] >= best - eps_abs:
                    G.add_edge(policies[i], policies[j])

    elif mode == "top_k":
        k = top_k if top_k is not None else 1
        for i in range(n):
            col = M[:, i]
            top_indices = np.argsort(col)[::-1][:k]
            for j in top_indices:
                G.add_edge(policies[i], policies[j])

    return G


# ---------------------------------------------------------------------------
# Structural Metrics (single graph)
# ---------------------------------------------------------------------------


def graph_metrics(G):
    """Compute structural metrics for a single directed graph.

    Returns:
        dict with keys: n_edges, lscc_size, scc_entropy, frac_nontrivial,
        n_sink_sccs, spectral_radius.
    """
    n = G.number_of_nodes()

    # SCC decomposition
    sccs = list(nx.strongly_connected_components(G))
    scc_sizes = [len(c) for c in sccs]

    # Largest SCC
    lscc_size = max(scc_sizes) if scc_sizes else 0

    # SCC entropy (entropy of size distribution)
    total = sum(scc_sizes)
    if total > 0:
        probs = np.array(scc_sizes) / total
        scc_entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))
    else:
        scc_entropy = 0.0

    # Fraction of nodes in nontrivial SCCs (size >= 2)
    nontrivial_nodes = sum(s for s in scc_sizes if s >= 2)
    frac_nontrivial = nontrivial_nodes / n if n > 0 else 0.0

    # Sink SCCs: SCCs with no outgoing edges to other SCCs
    condensation = nx.condensation(G)
    n_sink_sccs = sum(1 for node in condensation.nodes()
                      if condensation.out_degree(node) == 0)

    # Spectral radius of adjacency matrix
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(float)
    if A.size > 0:
        eigs = eigvals(A)
        spectral_radius = float(np.max(np.abs(eigs)))
    else:
        spectral_radius = 0.0

    return {
        "n_edges": G.number_of_edges(),
        "lscc_size": lscc_size,
        "scc_entropy": scc_entropy,
        "frac_nontrivial": frac_nontrivial,
        "n_sink_sccs": n_sink_sccs,
        "spectral_radius": spectral_radius,
    }


# ---------------------------------------------------------------------------
# Structural Shift Metrics (before/after comparison)
# ---------------------------------------------------------------------------


def graph_shift_metrics(G_before, G_after):
    """Compute structural change metrics between two graphs.

    Args:
        G_before: baseline graph (without candidate).
        G_after: treated graph (with candidate).

    Returns:
        dict with delta metrics and edge churn.
    """
    m_before = graph_metrics(G_before)
    m_after = graph_metrics(G_after)

    edges_before = set(G_before.edges())
    edges_after = set(G_after.edges())
    edges_added = edges_after - edges_before
    edges_removed = edges_before - edges_after
    edge_churn = len(edges_added | edges_removed) / (len(edges_before) + 1e-9)

    return {
        "edge_churn": edge_churn,
        "edges_added": edges_added,
        "edges_removed": edges_removed,
        "n_edges_added": len(edges_added),
        "n_edges_removed": len(edges_removed),
        "delta_lscc": m_after["lscc_size"] - m_before["lscc_size"],
        "delta_scc_entropy": m_after["scc_entropy"] - m_before["scc_entropy"],
        "delta_frac_nontrivial": m_after["frac_nontrivial"] - m_before["frac_nontrivial"],
        "delta_n_sink_sccs": m_after["n_sink_sccs"] - m_before["n_sink_sccs"],
        "delta_spectral_radius": m_after["spectral_radius"] - m_before["spectral_radius"],
        "metrics_before": m_before,
        "metrics_after": m_after,
    }


# ---------------------------------------------------------------------------
# Markov Chain Best-Response Dynamics
# ---------------------------------------------------------------------------


def transition_matrix(G, policies):
    """Build row-stochastic transition matrix from BR graph.

    T[i,j] = 1/|BR(i)| if j is a successor of i, else 0.
    Absorbing state if i has no successors: T[i,i] = 1.
    """
    n = len(policies)
    pol_to_idx = {p: k for k, p in enumerate(policies)}
    T = np.zeros((n, n))
    for i, p in enumerate(policies):
        succs = list(G.successors(p))
        if succs:
            for s in succs:
                T[i, pol_to_idx[s]] += 1.0
            T[i] /= T[i].sum()
        else:
            T[i, i] = 1.0  # absorbing
    return T


def classify_chain(T):
    """Classify states of a Markov chain into recurrent and transient.

    Returns dict with recurrent_classes, transient_states, is_irreducible.
    """
    n = T.shape[0]
    # Build DiGraph from positive entries
    D = nx.DiGraph()
    D.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if T[i, j] > 0:
                D.add_edge(i, j)

    # SCC decomposition; sink SCCs = recurrent classes
    cond = nx.condensation(D)
    sccs = list(nx.strongly_connected_components(D))
    # Map each node to its SCC index in the condensation
    node_to_scc = {}
    for scc_idx in cond.nodes():
        for node in cond.nodes[scc_idx]["members"]:
            node_to_scc[node] = scc_idx

    recurrent_classes = []
    transient_states = []
    for scc_idx in cond.nodes():
        members = sorted(cond.nodes[scc_idx]["members"])
        if cond.out_degree(scc_idx) == 0:
            recurrent_classes.append(members)
        else:
            transient_states.extend(members)

    transient_states.sort()
    is_irreducible = len(recurrent_classes) == 1 and len(transient_states) == 0

    return {
        "recurrent_classes": recurrent_classes,
        "transient_states": transient_states,
        "is_irreducible": is_irreducible,
    }


def stationary_distribution(T, cl):
    """Compute stationary distribution of the Markov chain.

    For each recurrent class: solve pi @ T_R = pi via left eigenvector.
    Transient states get pi = 0.
    When multiple recurrent classes: weight by absorption probabilities
    from a uniform start over transient states.

    Returns dict with pi (n,), pi_per_class, absorption_probs.
    """
    n = T.shape[0]
    rec_classes = cl["recurrent_classes"]
    transient = cl["transient_states"]

    # Stationary dist within each recurrent class
    pi_per_class = []
    for rc in rec_classes:
        idx = np.array(rc)
        T_R = T[np.ix_(idx, idx)]
        # Left eigenvector: pi @ T_R = pi => T_R^T @ pi^T = pi^T
        eigenvalues, eigenvectors = eig(T_R.T)
        # Find eigenvector for eigenvalue closest to 1
        close_to_one = np.argmin(np.abs(eigenvalues - 1.0))
        pi_rc = np.real(eigenvectors[:, close_to_one])
        pi_rc = np.clip(pi_rc, 0, None)
        if pi_rc.sum() > 0:
            pi_rc /= pi_rc.sum()
        else:
            pi_rc = np.ones(len(rc)) / len(rc)
        pi_per_class.append(pi_rc)

    # Absorption probabilities
    absorption_probs = None
    if len(rec_classes) == 1:
        weights = np.array([1.0])
    elif len(transient) == 0:
        # All states recurrent, multiple classes — uniform weight
        sizes = np.array([len(rc) for rc in rec_classes], dtype=float)
        weights = sizes / sizes.sum()
    else:
        # Fundamental matrix: N = (I - Q)^{-1}
        t_idx = np.array(transient)
        Q = T[np.ix_(t_idx, t_idx)]
        I_Q = np.eye(len(transient)) - Q
        try:
            N = np.linalg.solve(I_Q, np.eye(len(transient)))
        except np.linalg.LinAlgError:
            N = np.linalg.pinv(I_Q)

        # R matrix: transitions from transient to recurrent states
        # Absorption prob into class k = sum of N @ R columns for class k
        n_classes = len(rec_classes)
        class_abs = np.zeros((len(transient), n_classes))
        for k, rc in enumerate(rec_classes):
            r_idx = np.array(rc)
            R_k = T[np.ix_(t_idx, r_idx)]
            class_abs[:, k] = (N @ R_k).sum(axis=1)

        absorption_probs = class_abs
        # Uniform start over transient states
        mean_abs = class_abs.mean(axis=0)
        mean_abs = np.clip(mean_abs, 0, None)
        if mean_abs.sum() > 0:
            weights = mean_abs / mean_abs.sum()
        else:
            weights = np.ones(n_classes) / n_classes

    # Assemble full pi
    pi = np.zeros(n)
    for k, rc in enumerate(rec_classes):
        for local_i, global_i in enumerate(rc):
            pi[global_i] = weights[k] * pi_per_class[k][local_i]

    # Numerical guard
    pi = np.clip(pi, 0, None)
    if pi.sum() > 0:
        pi /= pi.sum()

    return {"pi": pi, "pi_per_class": pi_per_class, "absorption_probs": absorption_probs}


def spectral_gap(T):
    """Compute spectral gap: 1 - |lambda_2|.

    Larger gap = faster mixing = more volatile meta.
    """
    eigenvalues = eigvals(T)
    mags = np.sort(np.abs(eigenvalues))[::-1]
    if len(mags) < 2:
        return 1.0
    return float(1.0 - mags[1])


def mean_first_passage_times(T, cl, pi_vec):
    """Compute mean first passage times within recurrent classes.

    Within each recurrent class: Z = (I - T_R + Pi_R)^{-1},
    MFPT[i,j] = (Z[j,j] - Z[i,j]) / pi[j].
    Cross-class and transient: NaN.

    Returns dict with mfpt_matrix (n,n) and absorption_times.
    """
    n = T.shape[0]
    mfpt = np.full((n, n), np.nan)
    rec_classes = cl["recurrent_classes"]
    transient = cl["transient_states"]

    for rc in rec_classes:
        idx = np.array(rc)
        T_R = T[np.ix_(idx, idx)]
        pi_rc = pi_vec[idx]

        if pi_rc.sum() == 0:
            continue

        pi_rc_norm = pi_rc / pi_rc.sum()
        Pi_R = np.tile(pi_rc_norm, (len(rc), 1))  # each row = pi_rc

        A = np.eye(len(rc)) - T_R + Pi_R
        try:
            Z = np.linalg.solve(A, np.eye(len(rc)))
        except np.linalg.LinAlgError:
            Z = np.linalg.pinv(A)

        for li, gi in enumerate(rc):
            for lj, gj in enumerate(rc):
                if pi_rc_norm[lj] > 1e-15:
                    mfpt[gi, gj] = (Z[lj, lj] - Z[li, lj]) / pi_rc_norm[lj]

    # Absorption times for transient states
    absorption_times = None
    if transient:
        t_idx = np.array(transient)
        Q = T[np.ix_(t_idx, t_idx)]
        I_Q = np.eye(len(transient)) - Q
        try:
            N = np.linalg.solve(I_Q, np.eye(len(transient)))
        except np.linalg.LinAlgError:
            N = np.linalg.pinv(I_Q)
        absorption_times = N.sum(axis=1)

    return {"mfpt_matrix": mfpt, "absorption_times": absorption_times}


def markov_metrics(G, policies):
    """Convenience wrapper: compute all Markov chain metrics for a BR graph."""
    T = transition_matrix(G, policies)
    cl = classify_chain(T)
    sd = stationary_distribution(T, cl)
    sg = spectral_gap(T)
    mfpt = mean_first_passage_times(T, cl, sd["pi"])

    recurrent_mask = np.zeros(len(policies), dtype=bool)
    for rc in cl["recurrent_classes"]:
        for i in rc:
            recurrent_mask[i] = True

    return {
        "T": T,
        "pi": sd["pi"],
        "spectral_gap": sg,
        "classify": cl,
        "mfpt_matrix": mfpt["mfpt_matrix"],
        "absorption_times": mfpt["absorption_times"],
        "recurrent_mask": recurrent_mask,
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_raw(pkl_path):
    """Load raw bootstrap results from pkl file.

    Returns:
        (raw_list, config_dict, policies)
    """
    with open(pkl_path, "rb") as f:
        pk = pickle.load(f)

    raw = pk["raw"]
    config = pk.get("config", {})
    policies = config.get("strategy_names", STRATEGY_ORDER)
    return raw, config, policies


# ---------------------------------------------------------------------------
# Bootstrap Loop + do-Inclusion
# ---------------------------------------------------------------------------


def run_bootstrap_analysis(raw, policies, mode="eps", eps=0.01, top_k=None,
                           sample_only=False):
    """Run BR graph analysis across bootstrap samples.

    For each sample:
    - Build full-game graph
    - For each candidate: build baseline (without) and treated (with) graphs
    - Compute structural shift metrics

    Args:
        raw: list of bootstrap sample dicts.
        policies: list of policy names.
        mode: "eps" or "top_k".
        eps: epsilon tolerance.
        top_k: number of top BRs.
        sample_only: if True, use only 1 sample.

    Returns:
        dict with:
        - full_game_metrics: list of per-sample graph_metrics dicts
        - edge_probs: (n, n) array of P(edge i->j) across bootstraps
        - candidate_deltas: {candidate: {metric: [per-sample values]}}
        - candidate_edge_diffs: {candidate: {sample: {edges_added, edges_removed}}}
        - lscc_before_after: {candidate: {"before": [...], "after": [...]}}
        - policies: list of policy names
    """
    n_samples = 1 if sample_only else len(raw)
    n = len(policies)

    # Accumulators
    full_game_metrics_list = []
    edge_matrix = np.zeros((n, n, n_samples))  # edge existence per sample

    metric_keys = [
        "edge_churn", "delta_lscc", "delta_scc_entropy",
        "delta_frac_nontrivial", "delta_n_sink_sccs", "delta_spectral_radius",
    ]
    candidate_deltas = {
        p: {k: [] for k in metric_keys} for p in policies
    }
    candidate_edge_diffs = {p: [] for p in policies}
    lscc_before_after = {p: {"before": [], "after": []} for p in policies}

    # Markov chain accumulators
    full_pi = np.zeros((n, n_samples))
    full_recurrent = np.zeros((n, n_samples))
    full_spectral_gap = []
    full_sigma = np.zeros((n, n_samples))
    full_mfpt = np.full((n, n, n_samples), np.nan)
    candidate_delta_pi = {p: np.zeros((n, n_samples)) for p in policies}
    candidate_sg_before = {p: [] for p in policies}
    candidate_sg_after = {p: [] for p in policies}

    pol_to_idx = {p: i for i, p in enumerate(policies)}

    for b in range(n_samples):
        payoff = np.array(raw[b]["matrices"]["payoff"], dtype=float)

        # Full-game graph
        G_full = br_graph(payoff, policies, mode=mode, eps=eps, top_k=top_k)
        fm = graph_metrics(G_full)
        full_game_metrics_list.append(fm)

        # Record edge existence
        for u, v in G_full.edges():
            i, j = pol_to_idx[u], pol_to_idx[v]
            edge_matrix[i, j, b] = 1.0

        # Markov chain metrics for full game
        mk_full = markov_metrics(G_full, policies)
        full_pi[:, b] = mk_full["pi"]
        full_recurrent[:, b] = mk_full["recurrent_mask"].astype(float)
        full_spectral_gap.append(mk_full["spectral_gap"])
        full_mfpt[:, :, b] = mk_full["mfpt_matrix"]

        # MENE equilibrium from bootstrap sample
        sigma = raw[b].get("full_game", {}).get("sigma", None)
        if sigma is not None:
            full_sigma[:, b] = np.array(sigma, dtype=float)[:n]

        # Per-candidate do-inclusion
        for cand in policies:
            cand_idx = pol_to_idx[cand]
            baseline_pols = [p for p in policies if p != cand]
            baseline_idx = [pol_to_idx[p] for p in baseline_pols]

            # Baseline: submatrix without candidate
            sub_payoff = payoff[np.ix_(baseline_idx, baseline_idx)]
            G_baseline = br_graph(sub_payoff, baseline_pols, mode=mode,
                                  eps=eps, top_k=top_k)

            # Treated: full graph (already built)
            shift = graph_shift_metrics(G_baseline, G_full)

            for k in metric_keys:
                candidate_deltas[cand][k].append(shift[k])

            candidate_edge_diffs[cand].append({
                "edges_added": shift["edges_added"],
                "edges_removed": shift["edges_removed"],
            })

            lscc_before_after[cand]["before"].append(
                shift["metrics_before"]["lscc_size"]
            )
            lscc_before_after[cand]["after"].append(
                shift["metrics_after"]["lscc_size"]
            )

            # Markov chain: baseline metrics and delta
            mk_base = markov_metrics(G_baseline, baseline_pols)
            # Pad baseline pi to full size (missing candidate gets 0)
            pi_base_full = np.zeros(n)
            for local_i, p in enumerate(baseline_pols):
                pi_base_full[pol_to_idx[p]] = mk_base["pi"][local_i]
            candidate_delta_pi[cand][:, b] = mk_full["pi"] - pi_base_full
            candidate_sg_before[cand].append(mk_base["spectral_gap"])
            candidate_sg_after[cand].append(mk_full["spectral_gap"])

    # Edge probability matrix
    edge_probs = edge_matrix.mean(axis=2)

    return {
        "full_game_metrics": full_game_metrics_list,
        "edge_probs": edge_probs,
        "candidate_deltas": candidate_deltas,
        "candidate_edge_diffs": candidate_edge_diffs,
        "lscc_before_after": lscc_before_after,
        "policies": policies,
        "n_samples": n_samples,
        "markov": {
            "full_pi": full_pi,
            "full_recurrent": full_recurrent,
            "full_spectral_gap": np.array(full_spectral_gap),
            "full_sigma": full_sigma,
            "full_mfpt": full_mfpt,
            "candidate_delta_pi": candidate_delta_pi,
            "candidate_sg_before": candidate_sg_before,
            "candidate_sg_after": candidate_sg_after,
        },
    }


# ---------------------------------------------------------------------------
# Aggregation Helpers
# ---------------------------------------------------------------------------


def aggregate_deltas(candidate_deltas, n_samples):
    """Compute mean, 95% CI, and P(delta > 0) for each candidate/metric.

    Returns:
        {candidate: {metric: {"mean", "ci_lo", "ci_hi", "p_positive"}}}
    """
    result = {}
    for cand, metrics in candidate_deltas.items():
        result[cand] = {}
        for metric, vals in metrics.items():
            arr = np.array(vals)
            result[cand][metric] = {
                "mean": float(np.mean(arr)),
                "ci_lo": float(np.percentile(arr, 2.5)),
                "ci_hi": float(np.percentile(arr, 97.5)),
                "p_positive": float(np.mean(arr > 0)),
            }
    return result


# ---------------------------------------------------------------------------
# Visualization: A1 — Full-Game BR Graph Structure
# ---------------------------------------------------------------------------


def plot_edge_probability_heatmap(edge_probs, policies, save_dir):
    """Figure 1: Edge-existence probability heatmap (10x10)."""
    labels = [_dn(p) for p in policies]
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(edge_probs, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={"label": "P(edge exists)"})
    ax.set_xlabel("Target (best-response)", fontsize=11)
    ax.set_ylabel("Source (opponent)", fontsize=11)
    ax.set_title("L2.5: Edge-Existence Probability\n"
                 "(i -> j if j is eps-BR to opponent i)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_edge_probability_heatmap.png"),
                dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mean_graph(edge_probs, policies, save_dir, threshold=0.1):
    """Figure 2: Mean graph visualization with edge width proportional to P(edge)."""
    n = len(policies)
    G = nx.DiGraph()
    G.add_nodes_from(policies)

    edge_weights = []
    for i in range(n):
        for j in range(n):
            if edge_probs[i, j] >= threshold:
                G.add_edge(policies[i], policies[j], weight=edge_probs[i, j])
                edge_weights.append(edge_probs[i, j])

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Draw nodes
    node_colors = [STRATEGY_COLORS.get(p, "#999999") for p in policies]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=800, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={p: _dn(p) for p in policies},
                            font_size=8, font_weight="bold")

    # Draw edges with width proportional to probability
    if edge_weights:
        widths = [G[u][v]["weight"] * 4 for u, v in G.edges()]
        alphas = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=widths, alpha=0.7,
            edge_color="gray", arrows=True, arrowsize=15,
            connectionstyle="arc3,rad=0.1",
        )

    ax.set_title("L2.5: Mean Best-Response Graph\n"
                 f"(edges with P >= {threshold})", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig2_mean_br_graph.png"),
                dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: A2 — do-Inclusion Structural Effects
# ---------------------------------------------------------------------------


def plot_structural_deltas(agg_deltas, policies, save_dir):
    """Figure 3: Grouped bar chart of LSCC delta, SCC entropy delta,
    spectral radius delta per candidate."""
    metrics_to_plot = ["delta_lscc", "delta_scc_entropy", "delta_spectral_radius"]
    metric_labels = ["LSCC Size", "SCC Entropy", "Spectral Radius"]
    metric_colors = ["#4c72b0", "#55a868", "#c44e52"]

    n = len(policies)
    x = np.arange(n)
    width = 0.25
    labels = [_dn(p) for p in policies]

    fig, ax = plt.subplots(figsize=(14, 6))
    for j, (mk, ml, mc) in enumerate(zip(metrics_to_plot, metric_labels,
                                         metric_colors)):
        means = [agg_deltas[p][mk]["mean"] for p in policies]
        ci_lo = [agg_deltas[p][mk]["ci_lo"] for p in policies]
        ci_hi = [agg_deltas[p][mk]["ci_hi"] for p in policies]
        yerr_lo = [max(0, m - lo) for m, lo in zip(means, ci_lo)]
        yerr_hi = [max(0, hi - m) for m, hi in zip(means, ci_hi)]
        ax.bar(x + j * width, means, width, yerr=[yerr_lo, yerr_hi],
               capsize=3, label=ml, color=mc, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Delta (treated - baseline)")
    ax.set_title("L2.5: Structural Effects of do-Inclusion")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_structural_deltas.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_edge_churn(agg_deltas, policies, save_dir):
    """Figure 4: Edge churn per candidate."""
    labels = [_dn(p) for p in policies]
    means = [agg_deltas[p]["edge_churn"]["mean"] for p in policies]
    ci_lo = [agg_deltas[p]["edge_churn"]["ci_lo"] for p in policies]
    ci_hi = [agg_deltas[p]["edge_churn"]["ci_hi"] for p in policies]
    yerr_lo = [max(0, m - lo) for m, lo in zip(means, ci_lo)]
    yerr_hi = [max(0, hi - m) for m, hi in zip(means, ci_hi)]
    colors = [STRATEGY_COLORS.get(p, "#999999") for p in policies]

    x = np.arange(len(policies))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, means, yerr=[yerr_lo, yerr_hi], capsize=4,
           color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Edge Churn (|E+ ∪ E-| / |E_baseline|)")
    ax.set_title("L2.5: Edge Churn per Candidate (do-Inclusion)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig4_edge_churn.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: A3 — Edge-Level Causal Effects
# ---------------------------------------------------------------------------


def plot_edge_diff_heatmap(candidate_edge_diffs, policies, save_dir,
                           selected_candidate=None):
    """Figure 5: Per-candidate edge-diff heatmap for a selected candidate."""
    if selected_candidate is None:
        # Pick the candidate with the most total edge changes
        total_changes = {}
        for cand, samples in candidate_edge_diffs.items():
            total = sum(len(s["edges_added"]) + len(s["edges_removed"])
                        for s in samples)
            total_changes[cand] = total
        selected_candidate = max(total_changes, key=total_changes.get)

    n = len(policies)
    pol_to_idx = {p: i for i, p in enumerate(policies)}
    n_samples = len(candidate_edge_diffs[selected_candidate])

    # Build addition/removal frequency matrices
    add_freq = np.zeros((n, n))
    rem_freq = np.zeros((n, n))

    for sample in candidate_edge_diffs[selected_candidate]:
        for u, v in sample["edges_added"]:
            if u in pol_to_idx and v in pol_to_idx:
                add_freq[pol_to_idx[u], pol_to_idx[v]] += 1
            # edges involving the candidate itself — skip indices not in
            # pol_to_idx (they're new edges from/to the candidate which
            # always exist in the treated graph but not baseline)
        for u, v in sample["edges_removed"]:
            if u in pol_to_idx and v in pol_to_idx:
                rem_freq[pol_to_idx[u], pol_to_idx[v]] += 1

    add_freq /= n_samples
    rem_freq /= n_samples

    # Net diff: positive = edge added, negative = edge removed
    diff = add_freq - rem_freq

    labels = [_dn(p) for p in policies]
    vmax = max(np.abs(diff).max(), 0.01)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(diff, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-vmax, vmax=vmax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "P(added) - P(removed)"})
    ax.set_xlabel("Target", fontsize=11)
    ax.set_ylabel("Source", fontsize=11)
    ax.set_title(f"L2.5: Edge Diff When Adding {_dn(selected_candidate)}\n"
                 f"(red = edge created, blue = edge destroyed)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_edge_diff_heatmap.png"),
                dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: A4 — Nontransitivity / Cycle Analysis
# ---------------------------------------------------------------------------


def plot_lscc_violin(lscc_before_after, policies, save_dir):
    """Figure 6: LSCC size distribution (violin) — baseline vs treated per candidate."""
    labels = [_dn(p) for p in policies]
    n = len(policies)

    fig, ax = plt.subplots(figsize=(14, 6))
    positions_before = np.arange(n) * 2
    positions_after = np.arange(n) * 2 + 0.6

    data_before = [lscc_before_after[p]["before"] for p in policies]
    data_after = [lscc_before_after[p]["after"] for p in policies]

    vp1 = ax.violinplot(data_before, positions=positions_before,
                        showmeans=True, showmedians=False, widths=0.5)
    vp2 = ax.violinplot(data_after, positions=positions_after,
                        showmeans=True, showmedians=False, widths=0.5)

    # Color the violins
    for body in vp1["bodies"]:
        body.set_facecolor("#1f77b4")
        body.set_alpha(0.7)
    for part in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if part in vp1:
            vp1[part].set_color("#1f77b4")

    for body in vp2["bodies"]:
        body.set_facecolor("#d62728")
        body.set_alpha(0.7)
    for part in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if part in vp2:
            vp2[part].set_color("#d62728")

    ax.set_xticks(np.arange(n) * 2 + 0.3)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("LSCC Size")
    ax.set_title("L2.5: LSCC Size — Baseline (blue) vs Treated (red)")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1f77b4", alpha=0.7, label="Baseline (without candidate)"),
                       Patch(facecolor="#d62728", alpha=0.7, label="Treated (full game)")],
              fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig6_lscc_violin.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: Markov Chain Best-Response Dynamics
# ---------------------------------------------------------------------------


def plot_stationary_vs_mene(markov, policies, save_dir):
    """Figure 7: Stationary distribution pi vs MENE sigma per strategy."""
    full_pi = markov["full_pi"]       # (n, n_samples)
    full_sigma = markov["full_sigma"]  # (n, n_samples)
    n = len(policies)
    labels = [_dn(p) for p in policies]

    pi_mean = full_pi.mean(axis=1)
    pi_lo = np.percentile(full_pi, 2.5, axis=1)
    pi_hi = np.percentile(full_pi, 97.5, axis=1)
    sigma_mean = full_sigma.mean(axis=1)

    x = np.arange(n)
    width = 0.5
    colors = [STRATEGY_COLORS.get(p, "#999999") for p in policies]

    fig, ax = plt.subplots(figsize=(12, 6))
    yerr_lo = np.clip(pi_mean - pi_lo, 0, None)
    yerr_hi = np.clip(pi_hi - pi_mean, 0, None)
    ax.bar(x, pi_mean, width, yerr=[yerr_lo, yerr_hi], capsize=3,
           color=colors, edgecolor="black", linewidth=0.5, label="Stationary π")
    ax.scatter(x, sigma_mean, color="black", marker="D", s=60, zorder=5,
               label="MENE σ")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("L2.5: Stationary Distribution π vs MENE Equilibrium σ")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig7_stationary_vs_mene.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_recurrence_probability(markov, policies, save_dir):
    """Figure 8: P(recurrent) per strategy across bootstrap samples."""
    full_recurrent = markov["full_recurrent"]  # (n, n_samples)
    labels = [_dn(p) for p in policies]
    p_rec = full_recurrent.mean(axis=1)
    colors = [STRATEGY_COLORS.get(p, "#999999") for p in policies]

    x = np.arange(len(policies))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, p_rec, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("P(Recurrent)")
    ax.set_ylim(0, 1.05)
    ax.set_title("L2.5: Recurrence Probability per Strategy")
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig8_recurrence_probability.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_delta_pi_heatmap(markov, policies, save_dir):
    """Figure 9: Heatmap of mean delta-pi when each candidate is added."""
    candidate_delta_pi = markov["candidate_delta_pi"]
    n = len(policies)
    labels = [_dn(p) for p in policies]

    # Build (n x n) matrix: row = candidate added, col = strategy affected
    delta_mat = np.zeros((n, n))
    for i, cand in enumerate(policies):
        delta_mat[i, :] = candidate_delta_pi[cand].mean(axis=1)

    vmax = max(np.abs(delta_mat).max(), 0.001)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(delta_mat, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-vmax, vmax=vmax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean Δπ"})
    ax.set_xlabel("Strategy Affected", fontsize=11)
    ax.set_ylabel("Candidate Added", fontsize=11)
    ax.set_title("L2.5: Causal Effect on Stationary Distribution\n"
                 "(Δπ when candidate is added)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig9_delta_pi_heatmap.png"),
                dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_gap_comparison(markov, policies, save_dir):
    """Figure 10: Baseline vs treated spectral gap per candidate."""
    sg_before = markov["candidate_sg_before"]
    sg_after = markov["candidate_sg_after"]
    n = len(policies)
    labels = [_dn(p) for p in policies]

    x = np.arange(n)
    width = 0.35

    before_mean = np.array([np.mean(sg_before[p]) for p in policies])
    before_lo = np.array([np.percentile(sg_before[p], 2.5) for p in policies])
    before_hi = np.array([np.percentile(sg_before[p], 97.5) for p in policies])
    after_mean = np.array([np.mean(sg_after[p]) for p in policies])
    after_lo = np.array([np.percentile(sg_after[p], 2.5) for p in policies])
    after_hi = np.array([np.percentile(sg_after[p], 97.5) for p in policies])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, before_mean, width,
           yerr=[np.clip(before_mean - before_lo, 0, None),
                 np.clip(before_hi - before_mean, 0, None)],
           capsize=3, color="#1f77b4", edgecolor="black", linewidth=0.4,
           label="Baseline (without candidate)")
    ax.bar(x + width / 2, after_mean, width,
           yerr=[np.clip(after_mean - after_lo, 0, None),
                 np.clip(after_hi - after_mean, 0, None)],
           capsize=3, color="#d62728", edgecolor="black", linewidth=0.4,
           label="Treated (full game)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Spectral Gap (1 - |λ₂|)")
    ax.set_title("L2.5: Spectral Gap — Baseline vs Treated")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig10_spectral_gap_comparison.png"),
                dpi=DEFAULT_DPI)
    plt.close(fig)


def plot_mfpt_heatmap(markov, policies, save_dir):
    """Figure 11: Mean first passage time heatmap (full game, bootstrap mean)."""
    full_mfpt = markov["full_mfpt"]  # (n, n, n_samples)
    labels = [_dn(p) for p in policies]
    n = len(policies)

    # Compute nanmean across samples; mask cells with >50% NaN
    nan_frac = np.isnan(full_mfpt).mean(axis=2)
    mfpt_mean = np.nanmean(full_mfpt, axis=2)
    mfpt_mean[nan_frac > 0.5] = np.nan

    # For display, replace NaN with empty annotation
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if np.isnan(mfpt_mean[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{mfpt_mean[i, j]:.1f}"

    mask = np.isnan(mfpt_mean)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(mfpt_mean, annot=annot, fmt="", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax, mask=mask,
                cbar_kws={"label": "Mean First Passage Time (steps)"})
    ax.set_xlabel("Target Strategy", fontsize=11)
    ax.set_ylabel("Source Strategy", fontsize=11)
    ax.set_title("L2.5: Mean First Passage Times\n"
                 "(within recurrent classes, NaN = cross-class/transient)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig11_mfpt_heatmap.png"),
                dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Print Summaries
# ---------------------------------------------------------------------------


def print_full_game_summary(full_game_metrics, n_samples):
    """Print SCC decomposition, sink SCCs, spectral radius."""
    keys = ["n_edges", "lscc_size", "scc_entropy", "frac_nontrivial",
            "n_sink_sccs", "spectral_radius"]

    print("\n" + "=" * 60)
    print("A1: Full-Game BR Graph Summary")
    print("=" * 60)
    print(f"{'Metric':<25} {'Mean':>8} {'CI_lo':>8} {'CI_hi':>8}")
    print("-" * 60)
    for k in keys:
        vals = [m[k] for m in full_game_metrics]
        arr = np.array(vals)
        print(f"{k:<25} {np.mean(arr):8.2f} {np.percentile(arr, 2.5):8.2f} "
              f"{np.percentile(arr, 97.5):8.2f}")


def print_candidate_summary(agg_deltas, policies):
    """Print per-candidate structural metric deltas."""
    metric_keys = [
        "edge_churn", "delta_lscc", "delta_scc_entropy",
        "delta_frac_nontrivial", "delta_n_sink_sccs", "delta_spectral_radius",
    ]

    print("\n" + "=" * 80)
    print("A2: do-Inclusion Structural Effects (per candidate)")
    print("=" * 80)

    for mk in metric_keys:
        print(f"\n  {mk}:")
        print(f"  {'Candidate':<20} {'Mean':>8} {'CI_lo':>8} {'CI_hi':>8} {'P(>0)':>8}")
        print("  " + "-" * 56)
        for p in policies:
            d = agg_deltas[p][mk]
            print(f"  {_dn(p):<20} {d['mean']:8.3f} {d['ci_lo']:8.3f} "
                  f"{d['ci_hi']:8.3f} {d['p_positive']:8.3f}")


def print_top_edge_changes(candidate_edge_diffs, policies):
    """Print top-5 most-changed edges per candidate."""
    print("\n" + "=" * 60)
    print("A3: Top Edge Changes per Candidate")
    print("=" * 60)

    for cand in policies:
        add_counts = {}
        rem_counts = {}
        n_samples = len(candidate_edge_diffs[cand])

        for sample in candidate_edge_diffs[cand]:
            for e in sample["edges_added"]:
                add_counts[e] = add_counts.get(e, 0) + 1
            for e in sample["edges_removed"]:
                rem_counts[e] = rem_counts.get(e, 0) + 1

        # Top 5 additions
        top_add = sorted(add_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_rem = sorted(rem_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"\n  {_dn(cand)}:")
        if top_add:
            print(f"    Added:   ", end="")
            print(", ".join(f"{_dn(u)}->{_dn(v)} ({c/n_samples:.0%})"
                            for (u, v), c in top_add))
        if top_rem:
            print(f"    Removed: ", end="")
            print(", ".join(f"{_dn(u)}->{_dn(v)} ({c/n_samples:.0%})"
                            for (u, v), c in top_rem))


def print_cycle_analysis(lscc_before_after, policies):
    """Print which agents create vs destroy cycles."""
    print("\n" + "=" * 60)
    print("A4: Cycle Creation / Destruction")
    print("=" * 60)
    print(f"{'Candidate':<20} {'LSCC_before':>12} {'LSCC_after':>12} {'Delta':>8} {'Creates?':>10}")
    print("-" * 65)

    for p in policies:
        before = np.mean(lscc_before_after[p]["before"])
        after = np.mean(lscc_before_after[p]["after"])
        delta = after - before
        creates = "CREATES" if delta > 0.5 else ("DESTROYS" if delta < -0.5 else "neutral")
        print(f"{_dn(p):<20} {before:12.1f} {after:12.1f} {delta:8.1f} {creates:>10}")


def print_markov_summary(markov, policies):
    """Print Markov chain best-response dynamics summary."""
    full_pi = markov["full_pi"]
    full_sigma = markov["full_sigma"]
    full_recurrent = markov["full_recurrent"]
    sg = markov["full_spectral_gap"]
    full_mfpt = markov["full_mfpt"]

    print("\n" + "=" * 70)
    print("A5: Markov Chain Best-Response Dynamics")
    print("=" * 70)

    # Stationary distribution vs MENE
    print(f"\n  Stationary Distribution π vs MENE σ:")
    print(f"  {'Strategy':<20} {'π (mean)':>10} {'π (CI)':>16} {'σ (mean)':>10}")
    print("  " + "-" * 60)
    for i, p in enumerate(policies):
        pi_m = full_pi[i].mean()
        pi_lo = np.percentile(full_pi[i], 2.5)
        pi_hi = np.percentile(full_pi[i], 97.5)
        sig_m = full_sigma[i].mean()
        print(f"  {_dn(p):<20} {pi_m:10.4f} [{pi_lo:6.4f},{pi_hi:6.4f}] {sig_m:10.4f}")

    # Recurrence probabilities
    print(f"\n  Recurrence Probabilities:")
    print(f"  {'Strategy':<20} {'P(recurrent)':>14}")
    print("  " + "-" * 36)
    for i, p in enumerate(policies):
        print(f"  {_dn(p):<20} {full_recurrent[i].mean():14.3f}")

    # Spectral gap stats
    print(f"\n  Spectral Gap: mean={sg.mean():.4f}, "
          f"CI=[{np.percentile(sg, 2.5):.4f}, {np.percentile(sg, 97.5):.4f}]")

    # MFPT excerpt (within recurrent, nanmean)
    nan_frac = np.isnan(full_mfpt).mean(axis=2)
    mfpt_mean = np.nanmean(full_mfpt, axis=2)
    mfpt_mean[nan_frac > 0.5] = np.nan

    # Show non-NaN off-diagonal entries
    n = len(policies)
    entries = []
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(mfpt_mean[i, j]):
                entries.append((i, j, mfpt_mean[i, j]))
    entries.sort(key=lambda x: x[2])

    if entries:
        print(f"\n  MFPT Excerpt (top 5 shortest / longest):")
        for i, j, v in entries[:5]:
            print(f"    {_dn(policies[i]):>15} → {_dn(policies[j]):<15} {v:8.1f} steps")
        if len(entries) > 5:
            print("    ...")
            for i, j, v in entries[-3:]:
                print(f"    {_dn(policies[i]):>15} → {_dn(policies[j]):<15} {v:8.1f} steps")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="L2.5 Best-Response Graph Causal Analysis"
    )
    parser.add_argument("--sample-only", action="store_true",
                        help="Use 1 bootstrap sample for quick iteration")
    parser.add_argument("--pkl-path", type=str,
                        default="data/analysis/iterative_analysis_results.pkl",
                        help="Path to results pkl")
    parser.add_argument("--fig-dir", type=str,
                        default="data/analysis/figures/l25_br_graph",
                        help="Output directory for figures")
    parser.add_argument("--eps", type=float, default=0.01,
                        help="Epsilon tolerance as fraction of payoff range")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k BRs (alternative to eps mode)")
    parser.add_argument("--mode", type=str, default="eps",
                        choices=["eps", "top_k"],
                        help="BR graph mode: eps (default) or top_k")

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    pkl_path = project_root / args.pkl_path
    fig_dir = project_root / args.fig_dir

    os.makedirs(fig_dir, exist_ok=True)
    fig_dir = str(fig_dir)

    print(f"Loading data from {pkl_path} ...")
    raw, config, policies = load_raw(pkl_path)
    n_total = len(raw)
    print(f"  {n_total} bootstrap samples, {len(policies)} policies")

    # Run analysis
    print(f"\nRunning BR graph analysis (mode={args.mode}, "
          f"eps={args.eps}, top_k={args.top_k}, "
          f"samples={'1 (sample-only)' if args.sample_only else n_total}) ...")

    results = run_bootstrap_analysis(
        raw, policies,
        mode=args.mode, eps=args.eps, top_k=args.top_k,
        sample_only=args.sample_only,
    )

    # Aggregate deltas
    agg_deltas = aggregate_deltas(results["candidate_deltas"],
                                  results["n_samples"])

    # Print summaries
    print_full_game_summary(results["full_game_metrics"], results["n_samples"])
    print_candidate_summary(agg_deltas, policies)
    print_top_edge_changes(results["candidate_edge_diffs"], policies)
    print_cycle_analysis(results["lscc_before_after"], policies)
    print_markov_summary(results["markov"], policies)

    # Generate figures
    print(f"\nGenerating figures in {fig_dir} ...")

    plot_edge_probability_heatmap(results["edge_probs"], policies, fig_dir)
    print("  [1/8] Edge probability heatmap")

    plot_mean_graph(results["edge_probs"], policies, fig_dir)
    print("  [2/8] Mean BR graph")

    plot_structural_deltas(agg_deltas, policies, fig_dir)
    print("  [3/8] Structural deltas")

    plot_edge_churn(agg_deltas, policies, fig_dir)
    print("  [4/8] Edge churn")

    plot_edge_diff_heatmap(results["candidate_edge_diffs"], policies, fig_dir)
    print("  [5/8] Edge diff heatmap")

    plot_lscc_violin(results["lscc_before_after"], policies, fig_dir)
    print("  [6/8] LSCC violin")

    plot_stationary_vs_mene(results["markov"], policies, fig_dir)
    print("  [7/8] Stationary vs MENE")

    plot_delta_pi_heatmap(results["markov"], policies, fig_dir)
    print("  [8/8] Delta-pi heatmap")

    print(f"\nDone. 8 figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
