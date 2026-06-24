"""
Nash clustering decomposition from Czarnecki et al. (2020),
"Real-World Games Look Like Spinning Tops."

Decomposes a symmetric zero-sum game into transitive layers (clusters)
by iteratively finding the max-entropy Nash equilibrium support and
removing it.

The cyclicity of a game is measured by the fraction of payoff variance
that lives within clusters (cyclic) vs between clusters (transitive).
"""

import numpy as np
from typing import List, Tuple

from .metagame import MetaGame


def nash_clustering(payoff_matrix: np.ndarray, solver: str = "mene",
                    support_threshold: float = 1e-6) -> List[np.ndarray]:
    """Compute Nash clustering of a symmetric zero-sum game.

    Algorithm (Definition 3, Czarnecki et al. 2020):
        1. Find max-entropy Nash equilibrium on remaining strategies
        2. Extract support as a cluster
        3. Remove support strategies, repeat until empty

    Args:
        payoff_matrix: (n, n) symmetric game payoff matrix where
            M[i,j] is player 1's payoff when playing strategy i vs j.
        solver: Equilibrium solver to use (default "mene").
        support_threshold: Minimum weight to count as in support.

    Returns:
        List of clusters, each an array of strategy indices.
        Clusters are ordered from strongest (first) to weakest (last).
    """
    n = payoff_matrix.shape[0]
    remaining = np.arange(n)
    clusters = []

    while len(remaining) > 0:
        # Restrict game to remaining strategies
        M_sub = payoff_matrix[np.ix_(remaining, remaining)]

        if len(remaining) == 1:
            clusters.append(remaining.copy())
            break

        # Solve for max-entropy NE on the restricted game
        names = [str(i) for i in remaining]
        mg = MetaGame(policies=names, payoff_matrix=M_sub)

        try:
            sigma = mg.solve(solver)
        except Exception:
            # If solver fails, put all remaining in one cluster
            clusters.append(remaining.copy())
            break

        # Extract support
        support_mask = sigma >= support_threshold
        if not np.any(support_mask):
            # Fallback: take the strategy with highest weight
            support_mask[np.argmax(sigma)] = True

        support_idx = remaining[support_mask]
        clusters.append(support_idx)

        # Remove support from remaining
        remaining = remaining[~support_mask]

    return clusters


def cyclicity_from_clustering(payoff_matrix: np.ndarray,
                              clusters: List[np.ndarray]) -> float:
    """Compute cyclicity measure from Nash clustering.

    Decomposes total payoff variance into within-cluster (cyclic) and
    between-cluster (transitive) components.

    cyclicity = ||M_within||_F^2 / ||M||_F^2

    where M_within is the block-diagonal part of M restricted to clusters.

    Args:
        payoff_matrix: (n, n) payoff matrix.
        clusters: List of cluster index arrays from nash_clustering().

    Returns:
        Cyclicity in [0, 1]. 0 = perfectly transitive, 1 = all cyclic.
    """
    total_var = np.sum(payoff_matrix ** 2)
    if total_var == 0:
        return 0.0

    within_var = 0.0
    for cluster in clusters:
        if len(cluster) > 1:
            M_block = payoff_matrix[np.ix_(cluster, cluster)]
            within_var += np.sum(M_block ** 2)

    return within_var / total_var


def nash_clustering_profile(payoff_matrix: np.ndarray,
                            clusters: List[np.ndarray]) -> List[dict]:
    """Compute the game profile (spinning top shape) from Nash clustering.

    For each cluster, computes:
        - size: number of strategies in the cluster
        - mean_winrate: average win rate of cluster strategies against
          all strategies below them in the ordering
        - within_variance: Frobenius norm of within-cluster payoffs

    This corresponds to the left panel of Table 1 in the paper.

    Args:
        payoff_matrix: (n, n) payoff matrix.
        clusters: List of cluster index arrays from nash_clustering().

    Returns:
        List of dicts with cluster statistics, ordered strongest to weakest.
    """
    profile = []
    all_below = np.array([], dtype=int)

    for i, cluster in enumerate(clusters):
        M_within = payoff_matrix[np.ix_(cluster, cluster)]

        # Win rate against all clusters below
        if len(all_below) > 0:
            M_vs_below = payoff_matrix[np.ix_(cluster, all_below)]
            mean_winrate = np.mean(M_vs_below)
        else:
            mean_winrate = 0.0

        profile.append({
            "cluster_id": i,
            "size": len(cluster),
            "indices": cluster,
            "mean_winrate_vs_below": mean_winrate,
            "within_frobenius": np.linalg.norm(M_within, "fro"),
        })

        all_below = np.concatenate([all_below, cluster])

    return profile


def compute_game_cyclicity(payoff_matrix: np.ndarray, solver: str = "mene",
                           support_threshold: float = 1e-6) -> dict:
    """One-shot function: cluster the game and return cyclicity stats.

    Args:
        payoff_matrix: (n, n) payoff matrix.
        solver: Equilibrium solver (default "mene").
        support_threshold: Min weight for support membership.

    Returns:
        Dict with keys:
            - clusters: list of index arrays
            - num_clusters: number of clusters
            - cluster_sizes: list of cluster sizes
            - cyclicity: fraction of variance within clusters
            - max_cluster_size: size of largest cluster
            - max_cluster_ratio: largest cluster / n
    """
    clusters = nash_clustering(payoff_matrix, solver=solver,
                               support_threshold=support_threshold)

    n = payoff_matrix.shape[0]
    sizes = [len(c) for c in clusters]
    cyc = cyclicity_from_clustering(payoff_matrix, clusters)

    return {
        "clusters": clusters,
        "num_clusters": len(clusters),
        "cluster_sizes": sizes,
        "cyclicity": cyc,
        "max_cluster_size": max(sizes),
        "max_cluster_ratio": max(sizes) / n,
    }
