"""Bootstrap resampling for uncertainty quantification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from causal_game_analysis.metagame import MetaGame
from causal_game_analysis.utils import compute_regret

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")


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
        l3_exact: bool = False,
        l3_n_samples: int = 1000,
        progress: bool = True,
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
            )
            results.append(sample_result)

        return results

    def _build_all_matrices(
        self, df: pd.DataFrame, policies: list[str]
    ) -> dict[str, NDArray[np.floating]]:
        """Build all metric matrices from resampled DataFrame.

        Constructs four matrices from the bargaining instance data:
        - payoff: Expected payoff for policy_i (used for equilibrium + UW)
        - nw: Per-instance Nash welfare, averaged per pair
        - nw_plus: Per-instance NW on advantages, averaged per pair
        - ef1: EF1 frequency per pair

        Args:
            df: Resampled DataFrame with bargaining instances.
            policies: List of policies (defines matrix ordering).

        Returns:
            Dict with 'payoff', 'nw', 'nw_plus', 'ef1' matrices.
        """
        n = len(policies)
        policy_to_idx = {p: i for i, p in enumerate(policies)}

        # Initialize matrices
        payoff_matrix = np.full((n, n), np.nan)
        nw_matrix = np.full((n, n), np.nan)
        nw_plus_matrix = np.full((n, n), np.nan)
        ef1_matrix = np.full((n, n), np.nan)
        counts_matrix = np.zeros((n, n), dtype=np.int64)

        # Compute per-instance metrics
        df = df.copy()

        # Nash welfare: sqrt(payoff_i * payoff_j)
        df["_nw"] = np.sqrt(
            np.maximum(df[self.payoff_i_col], 0) *
            np.maximum(df[self.payoff_j_col], 0)
        )

        # Advantages: max(0, payoff - batna)
        df["_adv_i"] = np.maximum(0, df[self.payoff_i_col] - df[self.batna_i_col])
        df["_adv_j"] = np.maximum(0, df[self.payoff_j_col] - df[self.batna_j_col])

        # NW+: sqrt(advantage_i * advantage_j)
        df["_nw_plus"] = np.sqrt(df["_adv_i"] * df["_adv_j"])

        # Aggregate by policy pair
        grouped = df.groupby([self.policy_i_col, self.policy_j_col]).agg(
            payoff=(self.payoff_i_col, "mean"),
            nw=("_nw", "mean"),
            nw_plus=("_nw_plus", "mean"),
            ef1=(self.ef1_col, "mean"),
            count=(self.payoff_i_col, "count"),
        )

        for (pi, pj), row in grouped.iterrows():
            if pi in policy_to_idx and pj in policy_to_idx:
                i, j = policy_to_idx[pi], policy_to_idx[pj]
                payoff_matrix[i, j] = row["payoff"]
                nw_matrix[i, j] = row["nw"]
                nw_plus_matrix[i, j] = row["nw_plus"]
                ef1_matrix[i, j] = row["ef1"]
                counts_matrix[i, j] = row["count"]

        return {
            "payoff": payoff_matrix,
            "nw": nw_matrix,
            "nw_plus": nw_plus_matrix,
            "ef1": ef1_matrix,
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
        # Handle NaN by treating as 0 for matrix multiplication
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
        # Handle NaN values by treating them as 0
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

    def _compute_l3_with_cache(
        self,
        metagame: MetaGame,
        matrices: dict[str, NDArray[np.floating]],
        policies: list[str],
        solver: str,
        l3_method: str,
        l3_exact: bool,
        l3_n_samples: int,
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

        Returns:
            Dict with Shapley/Banzhaf values for each welfare function.
        """
        from causal_game_analysis.analysis import (
            shapley_value,
            banzhaf_value,
        )

        # Cache: coalition (frozenset) -> equilibrium sigma
        equilibrium_cache: dict[frozenset, NDArray[np.floating]] = {}

        def get_cached_equilibrium(policy_subset: list[str]) -> NDArray[np.floating]:
            """Get equilibrium for a coalition, using cache."""
            key = frozenset(policy_subset)
            if key not in equilibrium_cache:
                indices = [metagame.policy_index(p) for p in policy_subset]
                sub_payoff = matrices["payoff"][np.ix_(indices, indices)]
                sub_game = MetaGame(policy_subset, sub_payoff)
                equilibrium_cache[key] = sub_game.solve(solver)
            return equilibrium_cache[key]

        def make_cached_value_fn(wf_matrix: NDArray[np.floating]):
            """Create a value function that uses cached equilibria."""
            def value_fn(policy_subset: list[str]) -> float:
                if len(policy_subset) == 0:
                    return 0.0
                sigma_sub = get_cached_equilibrium(policy_subset)
                indices = [metagame.policy_index(p) for p in policy_subset]
                sub_matrix = wf_matrix[np.ix_(indices, indices)]
                sub_matrix_clean = np.nan_to_num(sub_matrix, nan=0.0)
                return float(sigma_sub @ sub_matrix_clean @ sigma_sub)
            return value_fn

        l3_result = {}
        n_mc = None if l3_exact else l3_n_samples

        # Map welfare function names to their matrices
        wf_matrices = {
            "uw": matrices["payoff"],
            "nw": matrices["nw"],
            "nw_plus": matrices["nw_plus"],
        }

        # Compute Shapley for all welfare functions (shares cache)
        if l3_method in ["shapley", "both"]:
            for wf, wf_matrix in wf_matrices.items():
                value_fn = make_cached_value_fn(wf_matrix)
                l3_result[f"shapley_{wf}"] = shapley_value(policies, value_fn, n_mc)

        # Compute Banzhaf for all welfare functions (shares cache)
        if l3_method in ["banzhaf", "both"]:
            for wf, wf_matrix in wf_matrices.items():
                value_fn = make_cached_value_fn(wf_matrix)
                l3_result[f"banzhaf_{wf}"] = banzhaf_value(policies, value_fn, n_mc)

        return l3_result

    def _analyze_single_sample(
        self,
        solver: str,
        include_l3: bool,
        l3_method: str,
        l3_exact: bool,
        l3_n_samples: int,
    ) -> dict:
        """Run full analysis on a single bootstrap sample.

        This is the core method that performs L1, L2, L3, and EF1 analysis
        for one resampled meta-game.
        """
        from causal_game_analysis.analysis import (
            shapley_value,
            banzhaf_value,
        )
        from causal_game_analysis.utils import l1_norm

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

        # 3. Full game analysis
        sigma_full = metagame.solve(solver)
        regret_full, nash_value_full, expected_utils_full = compute_regret(
            sigma_full, metagame.payoff_matrix
        )
        welfare_full = self._compute_welfare_all(sigma_full, matrices)
        ef1_full = self._compute_ef1_at_equilibrium(matrices["ef1"], sigma_full)

        full_game_result = {
            "sigma": sigma_full,
            "regret": {p: float(regret_full[i]) for i, p in enumerate(policies)},
            "welfare": welfare_full,
            "ef1": ef1_full,
            "nash_value": nash_value_full,
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

            # Welfare in baseline game (using subsetted matrices)
            welfare_B = self._compute_welfare_all(sigma_B, baseline_matrices)

            # EF1 in baseline game
            ef1_B = self._compute_ef1_at_equilibrium(baseline_matrices["ef1"], sigma_B)

            # --- L1: Partner lift for each incumbent ---
            # Expand sigma_B to full game indices for baseline_value computation
            sigma_B_full = np.zeros(metagame.n_policies)
            for i, p in enumerate(baseline_policies):
                sigma_B_full[metagame.policy_index(p)] = sigma_B[i]

            per_incumbent_lift = {}
            for incumbent in baseline_policies:
                # μ(incumbent, candidate) - U_B(incumbent)
                pairwise = metagame.pairwise_payoff(incumbent, candidate)
                baseline_val = metagame.expected_value(incumbent, sigma_B_full)
                per_incumbent_lift[incumbent] = pairwise - baseline_val

            lifts = list(per_incumbent_lift.values())
            sigma_B_dict = {p: sigma_B[i] for i, p in enumerate(baseline_policies)}

            l1_results[candidate] = {
                "per_incumbent": per_incumbent_lift,
                "uniform_avg": float(np.mean(lifts)),
                "equilibrium_avg": sum(
                    sigma_B_dict[p] * per_incumbent_lift[p] for p in baseline_policies
                ),
                "min": float(np.min(lifts)),
                "max": float(np.max(lifts)),
                "sigma_B": sigma_B,
                "regret_B": {p: float(regret_B[i]) for i, p in enumerate(baseline_policies)},
                "welfare_B": welfare_B,
                "ef1_B": ef1_B,
            }

            # --- L2: Ecosystem lift ---
            # delta_eco = W(full) - W(baseline) for each welfare function
            delta_eco = {
                wf: welfare_full[wf] - welfare_B[wf] for wf in ["uw", "nw", "nw_plus"]
            }

            # Entry mass: equilibrium weight of candidate in full game
            entry_mass = float(sigma_full[candidate_idx])

            # Equilibrium shift: compare sigma_B to sigma_full restricted to baseline
            sigma_full_restricted = np.array([
                sigma_full[metagame.policy_index(p)] for p in baseline_policies
            ])
            equilibrium_shift = l1_norm(sigma_full_restricted, sigma_B)

            # Incumbent value shifts
            incumbent_shifts = {}
            for i, p in enumerate(baseline_policies):
                V_B = baseline_game.expected_value(p, sigma_B)
                V_full = metagame.expected_value(p, sigma_full)
                incumbent_shifts[p] = V_full - V_B

            l2_results[candidate] = {
                "delta_eco": delta_eco,
                "W_B": welfare_B,
                "W_full": welfare_full,
                "entry_mass": entry_mass,
                "equilibrium_shift": equilibrium_shift,
                "incumbent_shifts": incumbent_shifts,
                "ef1_lift": ef1_full - ef1_B,
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
            )
            l3_result["total_value"] = {
                wf: welfare_full[wf] for wf in ["uw", "nw", "nw_plus"]
            }

        return {
            "l1": l1_results,
            "l2": l2_results,
            "l3": l3_result,
            "matrices": matrices,
            "full_game": full_game_result,
        }
