"""Bootstrap resampling for uncertainty quantification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
import pandas as pd

from causal_game_analysis.metagame import MetaGame

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")


class Bootstrap(Generic[T]):
    """Bootstrap resampling for meta-game analysis.

    Provides uncertainty quantification by resampling raw cross-play data
    and computing statistics across bootstrap samples.

    Attributes:
        df: Raw cross-play data.
        n_samples: Number of bootstrap samples to generate.
        policy_i_col: Column name for row policy.
        policy_j_col: Column name for column policy.
        outcome_col: Column name for outcome.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
        policy_i_col: str = "policy_i",
        policy_j_col: str = "policy_j",
        outcome_col: str = "outcome",
        policies: list[str] | None = None,
        seed: int | None = None,
    ):
        """Initialize Bootstrap.

        Args:
            df: DataFrame with raw cross-play outcomes.
            n_samples: Number of bootstrap samples.
            policy_i_col: Column for row policy.
            policy_j_col: Column for column policy.
            outcome_col: Column for outcome value.
            policies: Explicit list of policies (if None, inferred from data).
            seed: Random seed for reproducibility.
        """
        self.df = df
        self.n_samples = n_samples
        self.policy_i_col = policy_i_col
        self.policy_j_col = policy_j_col
        self.outcome_col = outcome_col
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
            outcome_col=self.outcome_col,
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
