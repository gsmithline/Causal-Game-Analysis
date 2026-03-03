"""Bootstrap resampling for uncertainty quantification."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
import pandas as pd
from tqdm import tqdm
from iterative_game_analysis.metagame import MetaGame
from iterative_game_analysis.utils import compute_regret

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")
import itertools

from iterative_game_analysis.analysis import shapley_value, banzhaf_value
from iterative_game_analysis.utils import l1_norm


def _solve_coalition(args):
    """Worker: solve equilibrium for one coalition. Top-level for pickling."""
    coalition_key, policy_subset, sub_payoff, solver = args
    sub_game = MetaGame(policy_subset, sub_payoff)
    sigma = sub_game.solve(solver)
    return coalition_key, sigma


class Bootstrap(Generic[T]):
    """Bootstrap resampling for meta-game analysis.

    Provides uncertainty quantification by resampling raw cross-play data
    and computing statistics across bootstrap samples.

    Attributes:
        df: Raw cross-play data (one row per bargaining instance).
        n_samples: Number of bootstrap samples to generate.
        policy_i_col: Column name for row policy.
        policy_j_col: Column name for column policy.
        payoff_i_col: Column name for policy_i's payoff.
        payoff_j_col: Column name for policy_j's payoff.
        batna_i_col: Column name for policy_i's BATNA.
        batna_j_col: Column name for policy_j's BATNA.
        ef1_col: Column name for EF1 indicator.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
        policy_i_col: str = "policy_i",
        policy_j_col: str = "policy_j",
        payoff_i_col: str = "payoff_i",
        payoff_j_col: str = "payoff_j",
        batna_i_col: str = "batna_i",
        batna_j_col: str = "batna_j",
        ef1_col: str = "ef1",
        policies: list[str] | None = None,
        seed: int | None = None,
    ):
        """Initialize Bootstrap.

        Args:
            df: DataFrame with raw cross-play outcomes (one row per bargaining instance).
            n_samples: Number of bootstrap samples.
            policy_i_col: Column for row policy.
            policy_j_col: Column for column policy.
            payoff_i_col: Column for policy_i's payoff.
            payoff_j_col: Column for policy_j's payoff.
            batna_i_col: Column for policy_i's BATNA.
            batna_j_col: Column for policy_j's BATNA.
            ef1_col: Column for EF1 indicator (1 if allocation was EF1).
            policies: Explicit list of policies (if None, inferred from data).
            seed: Random seed for reproducibility.
        """
        self.df = df
        self.n_samples = n_samples
        self.policy_i_col = policy_i_col
        self.policy_j_col = policy_j_col
        self.payoff_i_col = payoff_i_col
        self.payoff_j_col = payoff_j_col
        self.batna_i_col = batna_i_col
        self.batna_j_col = batna_j_col
        self.ef1_col = ef1_col
        self.policies = policies
        self._rng = np.random.default_rng(seed)

        # Infer policies if not provided
        if self.policies is None:
            all_policies = set(df[policy_i_col].unique()) | set(df[policy_j_col].unique())
            self.policies = sorted(all_policies)

    def sample(self) -> pd.DataFrame:
        """Generate one bootstrap sample by resampling with replacement.

        Resampling is stratified by (policy_i, policy_j) pairs to maintain
        the structure of cross-play.

        Returns:
            Resampled DataFrame.
        """
        # Group by pair and resample within each group
        resampled_dfs = []
        for _, group in self.df.groupby([self.policy_i_col, self.policy_j_col]):
            n = len(group)
            indices = self._rng.choice(n, size=n, replace=True)
            resampled_dfs.append(group.iloc[indices])

        return pd.concat(resampled_dfs, ignore_index=True)

    def sample_metagame(self) -> MetaGame:
        """Generate a MetaGame from one bootstrap sample.

        Returns:
            MetaGame built from resampled data.
        """
        resampled_df = self.sample()
        return MetaGame.from_dataframe(
            resampled_df,
            policy_i_col=self.policy_i_col,
            policy_j_col=self.policy_j_col,
            outcome_col=self.payoff_i_col,
            policies=self.policies,
        )

    def run(self, fn: Callable[[MetaGame], T], progress: bool = False) -> list[T]:
        """Run an analysis function on each bootstrap sample.

        Args:
            fn: Function that takes a MetaGame and returns analysis result.
            progress: If True, print progress updates.

        Returns:
            List of results from each bootstrap sample.
        """
        results = []
        for i in range(self.n_samples):
            if progress and (i + 1) % 100 == 0:
                print(f"Bootstrap sample {i + 1}/{self.n_samples}")
            metagame = self.sample_metagame()
            try:
                result = fn(metagame)
                results.append(result)
            except Exception as e:
                # Skip failed samples (e.g., solver failures)
                if progress:
                    print(f"Sample {i + 1} failed: {e}")
        return results

    @staticmethod
    def confidence_interval(
        values: list[float], alpha: float = 0.05
    ) -> tuple[float, float, float]:
        """Compute percentile confidence interval from bootstrap distribution.

        Args:
            values: Bootstrap sample values.
            alpha: Significance level (default 0.05 for 95% CI).

        Returns:
            Tuple of (lower, median, upper) bounds.
        """
        arr = np.array(values)
        lower = float(np.percentile(arr, 100 * alpha / 2))
        median = float(np.percentile(arr, 50))
        upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        return lower, median, upper

    @staticmethod
    def bootstrap_mean_ci(
        values: list[float], alpha: float = 0.05
    ) -> dict[str, float]:
        """Compute mean and confidence interval from bootstrap samples.

        Args:
            values: Bootstrap sample values.
            alpha: Significance level.

        Returns:
            Dict with 'mean', 'std', 'lower', 'upper' keys.
        """
        arr = np.array(values)
        lower, median, upper = Bootstrap.confidence_interval(values, alpha)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": median,
            "lower": lower,
            "upper": upper,
        }

    def run_full_analysis(
        self,
        solver: str = "mene",
        include_l3: bool = True,
        l3_method: str = "both",
        l3_exact: bool = True,
        l3_n_samples: int = 1000,
        progress: bool = True,
        n_workers: int = 1,
    ) -> list[dict]:
        """Run full L1/L2/L3 analysis on each bootstrap sample.

        For each bootstrap sample, performs leave-one-out analysis for every
        agent, computing all three levels of causal meta-game metrics.

        Structure per bootstrap sample:
        - L1 (per agent): Partner lift metrics when agent is the candidate
        - L2 (per agent): Ecosystem lift when agent is added back
        - L3 (full game): Shapley/Banzhaf attribution over all agents
        - EF1: Fairness frequency matrix and equilibrium-weighted EF1

        All welfare functions (UW, NW, NW+) are computed for each level.
        Regret is computed within each restricted and full game.

        Args:
            solver: Equilibrium solver to use (default "mene").
            include_l3: Whether to compute Level 3 attribution (expensive).
            l3_method: Attribution method - "shapley", "banzhaf", or "both".
            l3_exact: If True, compute exact Shapley/Banzhaf (expensive).
                If False, use Monte Carlo approximation.
            l3_n_samples: Number of samples for Monte Carlo L3 approximation.
            progress: If True, show tqdm progress bar.
            n_workers: Number of parallel workers for L3 coalition solves.

        Returns:
            List of dicts, one per bootstrap sample, each containing:
            - "l1": Dict mapping each agent to their L1 metrics
            - "l2": Dict mapping each agent to their L2 metrics
            - "l3": Dict with Shapley/Banzhaf attributions (if include_l3)
            - "matrices": All metric matrices (payoff, nw, nw_plus, ef1)
            - "full_game": Full game equilibrium, regret, and welfare metrics

        Raises:
            RuntimeError: If solver fails on any bootstrap sample.
        """
        results = []
        iterator = range(self.n_samples)
        if progress:
            iterator = tqdm(iterator, desc="Bootstrap samples")

        for _ in iterator:
            sample_result = self._analyze_single_sample(
                solver=solver,
                include_l3=include_l3,
                l3_method=l3_method,
                l3_exact=l3_exact,
                l3_n_samples=l3_n_samples,
                n_workers=n_workers,
            )
            results.append(sample_result)

        return results

    def _build_all_matrices(
        self, df: pd.DataFrame, policies: list[str]
    ) -> dict[str, NDArray[np.floating]]:
        """Build all metric matrices from resampled DataFrame.

        Constructs matrices from the bargaining instance data:
        - payoff: Expected payoff for policy_i (used for equilibrium + UW)
        - nw: Per-instance Nash welfare, averaged per pair
        - nw_plus: Per-instance NW on advantages, averaged per pair
        - ef1: EF1 frequency per pair
        - ef1_plus: EF1+ frequency per pair (rational games only)

        Args:
            df: Resampled DataFrame with bargaining instances.
            policies: List of policies (defines matrix ordering).

        Returns:
            Dict with 'payoff', 'nw', 'nw_plus', 'ef1', 'ef1_plus' matrices.
        """
        n = len(policies)
        policy_to_idx = {p: i for i, p in enumerate(policies)}

        # Initialize matrices
        payoff_matrix = np.full((n, n), np.nan)
        nw_matrix = np.full((n, n), np.nan)
        nw_plus_matrix = np.full((n, n), np.nan)
        ef1_matrix = np.full((n, n), np.nan)
        ef1_plus_matrix = np.full((n, n), np.nan)
        counts_matrix = np.zeros((n, n), dtype=np.int64)

        df = df.copy()

        df["_nw"] = np.sqrt(
            np.maximum(df[self.payoff_i_col], 0) *
            np.maximum(df[self.payoff_j_col], 0)
        )


        df["_adv_i"] = np.maximum(0, df[self.payoff_i_col] - df[self.batna_i_col])
        df["_adv_j"] = np.maximum(0, df[self.payoff_j_col] - df[self.batna_j_col])

        df["_nw_plus"] = np.sqrt(df["_adv_i"] * df["_adv_j"])

        has_ef1_plus = "ef1_plus" in df.columns
        agg_dict = {
            "payoff": (self.payoff_i_col, "mean"),
            "nw": ("_nw", "mean"),
            "nw_plus": ("_nw_plus", "mean"),
            "ef1": (self.ef1_col, "mean"),
            "count": (self.payoff_i_col, "count"),
        }
        if has_ef1_plus:
            agg_dict["ef1_plus"] = ("ef1_plus", "mean")

        grouped = df.groupby([self.policy_i_col, self.policy_j_col]).agg(**agg_dict)

        for (pi, pj), row in grouped.iterrows():
            if pi in policy_to_idx and pj in policy_to_idx:
                i, j = policy_to_idx[pi], policy_to_idx[pj]
                payoff_matrix[i, j] = row["payoff"]
                nw_matrix[i, j] = row["nw"]
                nw_plus_matrix[i, j] = row["nw_plus"]
                ef1_matrix[i, j] = row["ef1"]
                counts_matrix[i, j] = row["count"]
                if has_ef1_plus:
                    ef1_plus_matrix[i, j] = row["ef1_plus"]

        return {
            "payoff": payoff_matrix,
            "nw": nw_matrix,
            "nw_plus": nw_plus_matrix,
            "ef1": ef1_matrix,
            "ef1_plus": ef1_plus_matrix,
            "counts": counts_matrix,
        }

    def _compute_welfare_all(
        self,
        sigma: NDArray[np.floating],
        matrices: dict[str, NDArray[np.floating]],
    ) -> dict[str, float]:
        """Compute all welfare metrics (UW, NW, NW+) at equilibrium.

        Each metric is computed as σᵀ × matrix × σ using the appropriate matrix.

        Args:
            sigma: Equilibrium mixture.
            matrices: Dict with 'payoff', 'nw', 'nw_plus' matrices.

        Returns:
            Dict with 'uw', 'nw', 'nw_plus' keys.
        """
        # payoff/nw/nw_plus have no NaN in practice, so nan_to_num is harmless
        payoff = np.nan_to_num(matrices["payoff"], nan=0.0)
        nw = np.nan_to_num(matrices["nw"], nan=0.0)
        nw_plus = np.nan_to_num(matrices["nw_plus"], nan=0.0)

        # UW: expected payoff at equilibrium (σᵀ × payoff × σ)
        uw = float(sigma @ payoff @ sigma)

        # NW: expected Nash welfare at equilibrium (σᵀ × nw × σ)
        nw_val = float(sigma @ nw @ sigma)

        # NW+: expected NW on advantages at equilibrium (σᵀ × nw_plus × σ)
        nw_plus_val = float(sigma @ nw_plus @ sigma)

        return {"uw": uw, "nw": nw_val, "nw_plus": nw_plus_val}

    def _compute_ef1_at_equilibrium(
        self, ef1_matrix: NDArray[np.floating], sigma: NDArray[np.floating]
    ) -> float:
        """Compute expected EF1 frequency at equilibrium.

        EF1_eq = σ^T * EF1_matrix * σ

        Args:
            ef1_matrix: EF1 frequency matrix.
            sigma: Equilibrium mixture.

        Returns:
            Expected EF1 frequency weighted by equilibrium.
        """
        # NOTE: NaN→0 is lossy for EF1. NaN means "no accepts occurred"
        # (undefined), not "0% EF1" (worst fairness). Strategies like Walk
        # that never accept have all-NaN EF1 rows/columns. Treating NaN as 0
        # conflates "undefined" with "worst fairness." A proper fix would use
        # a renormalized weighted average excluding NaN cells:
        #   EF1_eq = Σ(σ_i·EF1_ij·σ_j | defined) / Σ(σ_i·σ_j | defined).
        # In practice this is acceptable because Walk (the only all-NaN
        # strategy) receives near-zero equilibrium weight (~1e-15), so the
        # NaN→0 contribution is negligible in equilibrium-weighted metrics.
        # Tough/Soft have low but defined EF1 for most matchups.
        ef1_clean = np.nan_to_num(ef1_matrix, nan=0.0)
        return float(sigma @ ef1_clean @ sigma)

    def _subset_matrices(
        self,
        matrices: dict[str, NDArray[np.floating]],
        indices: list[int],
    ) -> dict[str, NDArray[np.floating]]:
        """Subset all matrices to given indices.

        Args:
            matrices: Dict of matrices to subset.
            indices: Indices to keep.

        Returns:
            Dict of subsetted matrices.
        """
        return {
            key: matrix[np.ix_(indices, indices)]
            for key, matrix in matrices.items()
        }

    def _banzhaf_with_coalitions(
        self,
        policies: list[str],
        wf_matrices: dict[str, NDArray[np.floating]],
        equilibrium_cache: dict[frozenset, NDArray[np.floating]],
        value_table: dict[str, dict[frozenset, float]],
    ) -> dict:
        """Compute exact Banzhaf values while logging per-coalition details.

        For each policy and each coalition S (subset of others), records:
        - The coalition members
        - The equilibrium over S and S+{policy}
        - The marginal contribution v(S+{policy}) - v(S) for each welfare fn

        Args:
            policies: List of all policy names.
            wf_matrices: Dict mapping welfare name to its matrix.
            equilibrium_cache: Pre-computed equilibria per coalition.
            value_table: Pre-computed values: wf_name -> {coalition -> float}.

        Returns:
            Dict with:
            - "banzhaf_values": {wf: {policy: float}} averaged Banzhaf values
            - "coalitions": list of coalition records
        """
        n = len(policies)
        coalition_records = []
        banzhaf_sums = {wf: {p: 0.0 for p in policies} for wf in wf_matrices}

        for policy in policies:
            others = [p for p in policies if p != policy]
            for r in range(len(others) + 1):
                for subset in itertools.combinations(others, r):
                    S = sorted(subset)
                    S_with = sorted(S + [policy])
                    key_s = frozenset(S)
                    key_sw = frozenset(S_with)

                    # Value for S (without policy)
                    if len(S) == 0:
                        v_without = {wf: 0.0 for wf in wf_matrices}
                        sigma_without_dict = {}
                    else:
                        sigma_s = equilibrium_cache[key_s]
                        sigma_without_dict = {
                            p: float(sigma_s[i]) for i, p in enumerate(S)
                        }
                        v_without = {
                            wf: value_table[wf][key_s] for wf in wf_matrices
                        }

                    # Value for S + {policy}
                    sigma_sw = equilibrium_cache[key_sw]
                    sigma_with_dict = {
                        p: float(sigma_sw[i]) for i, p in enumerate(S_with)
                    }
                    v_with = {
                        wf: value_table[wf][key_sw] for wf in wf_matrices
                    }

                    # Marginal contributions
                    marginals = {
                        wf: v_with[wf] - v_without[wf] for wf in wf_matrices
                    }
                    for wf in wf_matrices:
                        banzhaf_sums[wf][policy] += marginals[wf]

                    coalition_records.append({
                        "policy": policy,
                        "coalition_without": S,
                        "coalition_with": S_with,
                        "sigma_without": sigma_without_dict,
                        "sigma_with": sigma_with_dict,
                        "marginals": marginals,
                    })

        # Average over 2^(n-1) coalitions
        n_coalitions = 2 ** (n - 1)
        banzhaf_values = {
            wf: {p: banzhaf_sums[wf][p] / n_coalitions for p in policies}
            for wf in wf_matrices
        }

        return {
            "banzhaf_values": banzhaf_values,
            "coalitions": coalition_records,
        }
    
    def get_cached_equilibrium(self,
                                policy_subset: list[str],
                                metagame: MetaGame,
                                equilibrium_cache: dict[frozenset, NDArray[np.floating]],
                                matrices: dict[str, NDArray[np.floating]],
                                solver: str,
                                ) -> NDArray[np.floating]:
        """Get equilibrium for a coalition, using cache.

        Always sorts policy_subset to ensure consistent ordering
        between cache writes and reads (sigma indices match matrix indices).
        """
        policy_subset = sorted(policy_subset)
        key = frozenset(policy_subset)
        if key not in equilibrium_cache:
            indices = [metagame.policy_index(p) for p in policy_subset]
            sub_payoff = matrices["payoff"][np.ix_(indices, indices)]
            sub_game = MetaGame(policy_subset, sub_payoff)
            equilibrium_cache[key] = sub_game.solve(solver)
        return equilibrium_cache[key]

    def _precompute_all_equilibria(
        self,
        policies: list[str],
        matrices: dict[str, "NDArray[np.floating]"],
        metagame: MetaGame,
        solver: str,
        n_workers: int,
    ) -> dict[frozenset, "NDArray[np.floating]"]:
        """Pre-compute equilibria for all 2^N non-empty coalitions.

        Args:
            policies: List of all policy names.
            matrices: All metric matrices (needs 'payoff' for subgames).
            metagame: Full MetaGame (for policy_index lookups).
            solver: Equilibrium solver name.
            n_workers: Number of parallel workers (1 = sequential).

        Returns:
            Dict mapping frozenset(coalition) -> equilibrium sigma.
        """
        # Enumerate all non-empty subsets
        tasks = []
        for r in range(1, len(policies) + 1):
            for subset in itertools.combinations(policies, r):
                policy_subset = sorted(subset)
                key = frozenset(policy_subset)
                indices = [metagame.policy_index(p) for p in policy_subset]
                sub_payoff = matrices["payoff"][np.ix_(indices, indices)]
                tasks.append((key, policy_subset, sub_payoff, solver))

        cache: dict[frozenset, NDArray[np.floating]] = {}

        if n_workers <= 1:
            # Sequential fallback
            for task in tasks:
                key, sigma = _solve_coalition(task)
                cache[key] = sigma
        else:
            from multiprocessing import Pool
            with Pool(n_workers) as pool:
                for key, sigma in pool.imap_unordered(_solve_coalition, tasks):
                    cache[key] = sigma

        return cache

    def _compute_l3_with_cache(
        self,
        metagame: MetaGame,
        matrices: dict[str, NDArray[np.floating]],
        policies: list[str],
        solver: str,
        l3_method: str,
        l3_exact: bool,
        l3_n_samples: int,
        n_workers: int = 1,
    ) -> dict:
        """Compute L3 attribution with cached equilibrium solves.

        Optimizes by caching equilibrium solutions per coalition, since
        the equilibrium depends only on the payoff matrix, not the welfare
        function. This avoids redundant solves across UW, NW, and NW+.

        Args:
            metagame: Full meta-game.
            matrices: All metric matrices.
            policies: List of policy names.
            solver: Equilibrium solver name.
            l3_method: "shapley", "banzhaf", or "both".
            l3_exact: Whether to use exact computation.
            l3_n_samples: Number of Monte Carlo samples if not exact.
            n_workers: Number of parallel workers for coalition solves.

        Returns:
            Dict with Shapley/Banzhaf values for each welfare function.
        """

        # Pre-compute all coalition equilibria (parallel if n_workers > 1)
        equilibrium_cache = self._precompute_all_equilibria(
            policies, matrices, metagame, solver, n_workers,
        )

        # Map welfare function names to their matrices
        wf_matrices = {
            "uw": matrices["payoff"],
            "nw": matrices["nw"],
            "nw_plus": matrices["nw_plus"],
            "ef1": matrices["ef1"],
            "ef1_plus": matrices["ef1_plus"],
        }

        # Pre-compute value table: wf_name -> {coalition_key -> float}
        # 1024 coalitions × 5 welfare fns = 5120 entries, computed once
        value_table: dict[str, dict[frozenset, float]] = {}
        for wf_name, wf_matrix in wf_matrices.items():
            wf_values: dict[frozenset, float] = {}
            for key, sigma in equilibrium_cache.items():
                policy_subset = sorted(key)
                indices = [metagame.policy_index(p) for p in policy_subset]
                sub = wf_matrix[np.ix_(indices, indices)]
                sub_clean = np.nan_to_num(sub, nan=0.0)
                wf_values[key] = float(sigma @ sub_clean @ sigma)
            value_table[wf_name] = wf_values

        def make_cached_value_fn(wf_name: str):
            """Create a value function backed by the pre-computed table."""
            wf_values = value_table[wf_name]
            def value_fn(policy_subset: list[str]) -> float:
                if len(policy_subset) == 0:
                    return 0.0
                return wf_values[frozenset(policy_subset)]
            return value_fn

        l3_result = {}
        n_mc = None if l3_exact else l3_n_samples

        # Compute Shapley for all welfare functions (pure dict lookups)
        if l3_method in ["shapley", "both"]:
            for wf in wf_matrices:
                value_fn = make_cached_value_fn(wf)
                l3_result[f"shapley_{wf}"] = shapley_value(policies, value_fn, n_mc)

        # Compute Banzhaf for all welfare functions with coalition logging
        if l3_method in ["banzhaf", "both"]:
            coalition_log = self._banzhaf_with_coalitions(
                policies=policies,
                wf_matrices=wf_matrices,
                equilibrium_cache=equilibrium_cache,
                value_table=value_table,
            )
            for wf in wf_matrices:
                l3_result[f"banzhaf_{wf}"] = coalition_log["banzhaf_values"][wf]
            l3_result["coalition_details"] = coalition_log["coalitions"]

        return l3_result

    def _analyze_single_sample(
        self,
        solver: str,
        include_l3: bool,
        l3_method: str,
        l3_exact: bool,
        l3_n_samples: int,
        n_workers: int = 1,
    ) -> dict:
        """Run full analysis on a single bootstrap sample.

        This is the core method that performs L1, L2, L3, and EF1 analysis
        for one resampled meta-game.
        """

        # 1. Resample and build all matrices
        resampled_df = self.sample()
        policies = self.policies
        matrices = self._build_all_matrices(resampled_df, policies)

        # 2. Build metagame from payoff matrix (for equilibrium computation)
        metagame = MetaGame(
            policies=policies,
            payoff_matrix=matrices["payoff"],
            counts_matrix=matrices["counts"],
        )

        # All metric matrices for multi-metric analysis
        # NOTE: payoff/nw/nw_plus have no NaN so nan_to_num is harmless.
        # EF1 has NaN for matchups with no accepts (e.g. Walk's entire
        # row/column). NaN→0 is lossy but acceptable since Walk gets
        # near-zero equilibrium weight — see _compute_ef1_at_equilibrium.
        metric_matrices = {
            "payoff": np.nan_to_num(matrices["payoff"], nan=0.0),
            "nw": np.nan_to_num(matrices["nw"], nan=0.0),
            "nw_plus": np.nan_to_num(matrices["nw_plus"], nan=0.0),
            "ef1": np.nan_to_num(matrices["ef1"], nan=0.0),
            "ef1_plus": np.nan_to_num(matrices["ef1_plus"], nan=0.0),
        }

        # 3. Full game analysis
        sigma_full = metagame.solve(solver)
        regret_full, nash_value_full, expected_utils_full = compute_regret(
            sigma_full, metagame.payoff_matrix
        )
        welfare_full = self._compute_welfare_all(sigma_full, matrices)
        ef1_full = self._compute_ef1_at_equilibrium(matrices["ef1"], sigma_full)
        ef1_plus_full = self._compute_ef1_at_equilibrium(matrices["ef1_plus"], sigma_full)

        per_agent_values_full = {}
        for metric_name, M in metric_matrices.items():
            per_agent_values_full[metric_name] = {
                p: float(M[i] @ sigma_full)
                for i, p in enumerate(policies)
            }

        full_game_result = {
            "sigma": sigma_full,
            "regret": {p: float(regret_full[i]) for i, p in enumerate(policies)},
            "welfare": welfare_full,
            "ef1": ef1_full,
            "ef1_plus": ef1_plus_full,
            "nash_value": nash_value_full,
            "per_agent_values": per_agent_values_full,
        }

        # 4. Leave-one-out analysis (L1 and L2)
        l1_results = {}
        l2_results = {}

        for candidate in policies:
            candidate_idx = metagame.policy_index(candidate)
            baseline_policies = [p for p in policies if p != candidate]
            baseline_indices = [metagame.policy_index(p) for p in baseline_policies]

            # Subset matrices for baseline game
            baseline_matrices = self._subset_matrices(matrices, baseline_indices)
            baseline_game = MetaGame(
                policies=baseline_policies,
                payoff_matrix=baseline_matrices["payoff"],
                counts_matrix=baseline_matrices["counts"],
            )

            # Compute baseline equilibrium (shared by L1 and L2)
            sigma_B = baseline_game.solve(solver)

            # Regret in baseline game
            regret_B, nash_value_B, _ = compute_regret(
                sigma_B, baseline_game.payoff_matrix
            )

            #welfare in baseline game (using subsetted matrices)
            welfare_B = self._compute_welfare_all(sigma_B, baseline_matrices)

            #EF1 in baseline game
            ef1_B = self._compute_ef1_at_equilibrium(baseline_matrices["ef1"], sigma_B)
            ef1_plus_B = self._compute_ef1_at_equilibrium(baseline_matrices["ef1_plus"], sigma_B)

            #L1: Partner lift for each incumbent, all metrics ---
            #expand sigma_B to full game indices
            sigma_B_full = np.zeros(metagame.n_policies)
            for i, p in enumerate(baseline_policies):
                sigma_B_full[metagame.policy_index(p)] = sigma_B[i]

            sigma_B_dict = {p: sigma_B[i] for i, p in enumerate(baseline_policies)}

            #compute partner lift for every metric matrix
            per_incumbent_lift = {}  # metric -> {incumbent -> lift}
            lift_aggregations = {}   # metric -> {uniform_avg, eq_avg, min, max}
            per_agent_values_B = {}  # metric -> {agent -> value at baseline eq}

            for metric_name, M in metric_matrices.items():
                inc_idx = candidate_idx
                per_metric_lift = {}
                per_metric_values_B = {}

                for incumbent in baseline_policies:
                    i_idx = metagame.policy_index(incumbent)
                    pairwise = float(M[i_idx, inc_idx])
                    baseline_val = float(M[i_idx] @ sigma_B_full)
                    per_metric_lift[incumbent] = pairwise - baseline_val
                    per_metric_values_B[incumbent] = baseline_val

                lifts = list(per_metric_lift.values())
                per_incumbent_lift[metric_name] = per_metric_lift
                per_agent_values_B[metric_name] = per_metric_values_B
                lift_aggregations[metric_name] = {
                    "uniform_avg": float(np.mean(lifts)),
                    "equilibrium_avg": sum(
                        sigma_B_dict[p] * per_metric_lift[p]
                        for p in baseline_policies
                    ),
                    "min": float(np.min(lifts)),
                    "max": float(np.max(lifts)),
                }

            l1_results[candidate] = {
                "per_incumbent": per_incumbent_lift,
                "aggregations": lift_aggregations,
                "per_agent_values_B": per_agent_values_B,
                "sigma_B": sigma_B,
                "regret_B": {p: float(regret_B[i]) for i, p in enumerate(baseline_policies)},
                "welfare_B": welfare_B,
                "ef1_B": ef1_B,
                "ef1_plus_B": ef1_plus_B,
            }

            # --- L2: Ecosystem lift ---
            # delta_eco = W(full) - W(baseline) for each welfare function
            delta_eco = {
                wf: welfare_full[wf] - welfare_B[wf] for wf in ["uw", "nw", "nw_plus"]
            }

            #entry mass: equilibrium weight of candidate in full game
            entry_mass = float(sigma_full[candidate_idx])

            #equilibrium shift: compare sigma_B to sigma_full restricted to baseline
            sigma_full_restricted = np.array([
                sigma_full[metagame.policy_index(p)] for p in baseline_policies
            ])
            equilibrium_shift = l1_norm(sigma_full_restricted, sigma_B)

            #incumbent value shifts for every metric
            incumbent_shifts = {}  #metric -> {incumbent -> shift}
            per_agent_values_full = {}  #metric -> {incumbent -> value at full eq}
            for metric_name, M in metric_matrices.items():
                per_metric_shifts = {}
                per_metric_values_full = {}
                for inc in baseline_policies:
                    i_full = metagame.policy_index(inc)
                    i_base = baseline_game.policy_index(inc)
                    M_base = metric_matrices[metric_name][
                        np.ix_(baseline_indices, baseline_indices)
                    ]
                    V_B = float(M_base[i_base] @ sigma_B)
                    V_full = float(M[i_full] @ sigma_full)
                    per_metric_shifts[inc] = V_full - V_B
                    per_metric_values_full[inc] = V_full
                incumbent_shifts[metric_name] = per_metric_shifts
                per_agent_values_full[metric_name] = per_metric_values_full

            l2_results[candidate] = {
                "delta_eco": delta_eco,
                "W_B": welfare_B,
                "W_full": welfare_full,
                "entry_mass": entry_mass,
                "equilibrium_shift": equilibrium_shift,
                "incumbent_shifts": incumbent_shifts,
                "per_agent_values_full": per_agent_values_full,
                "ef1_lift": ef1_full - ef1_B,
                "ef1_plus_lift": ef1_plus_full - ef1_plus_B,
            }

        # 5. Level 3: Attribution (if requested)
        l3_result = None
        if include_l3:
            l3_result = self._compute_l3_with_cache(
                metagame=metagame,
                matrices=matrices,
                policies=policies,
                solver=solver,
                l3_method=l3_method,
                l3_exact=l3_exact,
                l3_n_samples=l3_n_samples,
                n_workers=n_workers,
            )
            l3_result["total_value"] = {
                wf: welfare_full[wf] for wf in ["uw", "nw", "nw_plus"]
            }
            l3_result["total_value"]["ef1"] = ef1_full
            l3_result["total_value"]["ef1_plus"] = ef1_plus_full

        return {
            "l1": l1_results,
            "l2": l2_results,
            "l3": l3_result,
            "matrices": matrices,
            "full_game": full_game_result,
        }
