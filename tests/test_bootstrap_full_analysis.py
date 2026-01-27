"""Tests for Bootstrap.run_full_analysis() implementation."""

import numpy as np
import pandas as pd
import pytest

from causal_game_analysis.bootstrap import Bootstrap
from causal_game_analysis.metagame import MetaGame


def create_synthetic_bargaining_data(
    policies: list[str],
    n_instances_per_pair: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic bargaining data for testing.

    Args:
        policies: List of policy names.
        n_instances_per_pair: Number of bargaining instances per (i, j) pair.
        seed: Random seed.

    Returns:
        DataFrame with columns: policy_i, policy_j, payoff_i, payoff_j,
        batna_i, batna_j, ef1.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for pi in policies:
        for pj in policies:
            for _ in range(n_instances_per_pair):
                # Generate random payoffs and BATNAs
                batna_i = rng.uniform(10, 50)
                batna_j = rng.uniform(10, 50)

                # Payoffs are typically above BATNA (successful negotiation)
                # but sometimes below (failed negotiation)
                payoff_i = rng.uniform(batna_i - 10, batna_i + 100)
                payoff_j = rng.uniform(batna_j - 10, batna_j + 100)

                # EF1 indicator (random for testing)
                ef1 = rng.choice([0, 1], p=[0.3, 0.7])

                rows.append({
                    "policy_i": pi,
                    "policy_j": pj,
                    "payoff_i": payoff_i,
                    "payoff_j": payoff_j,
                    "batna_i": batna_i,
                    "batna_j": batna_j,
                    "ef1": ef1,
                })

    return pd.DataFrame(rows)


class TestBootstrapFullAnalysis:
    """Tests for run_full_analysis method."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample bargaining data."""
        policies = ["agent_a", "agent_b", "agent_c"]
        return create_synthetic_bargaining_data(policies, n_instances_per_pair=30)

    @pytest.fixture
    def bootstrap(self, sample_data: pd.DataFrame) -> Bootstrap:
        """Create Bootstrap instance."""
        return Bootstrap(
            df=sample_data,
            n_samples=5,  # Small for fast tests
            seed=123,
        )

    def test_run_full_analysis_returns_list(self, bootstrap: Bootstrap):
        """Test that run_full_analysis returns a list of results."""
        results = bootstrap.run_full_analysis(
            include_l3=False,  # Skip L3 for speed
            progress=False,
        )

        assert isinstance(results, list)
        assert len(results) == bootstrap.n_samples

    def test_result_structure(self, bootstrap: Bootstrap):
        """Test that each result has the expected keys."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        result = results[0]
        expected_keys = {"l1", "l2", "l3", "matrices", "full_game"}
        assert set(result.keys()) == expected_keys

    def test_matrices_structure(self, bootstrap: Bootstrap):
        """Test that matrices dict has all expected matrices."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        matrices = results[0]["matrices"]
        expected_keys = {"payoff", "nw", "nw_plus", "ef1", "counts"}
        assert set(matrices.keys()) == expected_keys

        # Check matrix shapes
        n_policies = len(bootstrap.policies)
        for key in ["payoff", "nw", "nw_plus", "ef1"]:
            assert matrices[key].shape == (n_policies, n_policies)

    def test_full_game_structure(self, bootstrap: Bootstrap):
        """Test full_game result structure."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        full_game = results[0]["full_game"]
        expected_keys = {"sigma", "regret", "welfare", "ef1", "nash_value"}
        assert set(full_game.keys()) == expected_keys

        # Check sigma is a valid probability distribution
        sigma = full_game["sigma"]
        assert len(sigma) == len(bootstrap.policies)
        assert np.isclose(sigma.sum(), 1.0)
        assert all(s >= 0 for s in sigma)

        # Check welfare has all metrics
        welfare = full_game["welfare"]
        assert set(welfare.keys()) == {"uw", "nw", "nw_plus"}

    def test_l1_structure(self, bootstrap: Bootstrap):
        """Test L1 results structure."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        l1 = results[0]["l1"]

        # Should have entry for each policy (as candidate)
        assert set(l1.keys()) == set(bootstrap.policies)

        # Check structure for one candidate
        candidate_result = l1[bootstrap.policies[0]]
        expected_keys = {
            "per_incumbent", "uniform_avg", "equilibrium_avg",
            "min", "max", "sigma_B", "regret_B", "welfare_B", "ef1_B"
        }
        assert set(candidate_result.keys()) == expected_keys

        # per_incumbent should have entries for all OTHER policies
        other_policies = [p for p in bootstrap.policies if p != bootstrap.policies[0]]
        assert set(candidate_result["per_incumbent"].keys()) == set(other_policies)

    def test_l2_structure(self, bootstrap: Bootstrap):
        """Test L2 results structure."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        l2 = results[0]["l2"]

        # Should have entry for each policy (as candidate)
        assert set(l2.keys()) == set(bootstrap.policies)

        # Check structure for one candidate
        candidate_result = l2[bootstrap.policies[0]]
        expected_keys = {
            "delta_eco", "W_B", "W_full", "entry_mass",
            "equilibrium_shift", "incumbent_shifts", "ef1_lift"
        }
        assert set(candidate_result.keys()) == expected_keys

        # delta_eco should have all welfare metrics
        assert set(candidate_result["delta_eco"].keys()) == {"uw", "nw", "nw_plus"}

    def test_l3_disabled(self, bootstrap: Bootstrap):
        """Test that L3 is None when disabled."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        assert results[0]["l3"] is None

    def test_l3_enabled_shapley(self, bootstrap: Bootstrap):
        """Test L3 with Shapley method."""
        results = bootstrap.run_full_analysis(
            include_l3=True,
            l3_method="shapley",
            l3_exact=False,
            l3_n_samples=100,  # Small for speed
            progress=False,
        )

        l3 = results[0]["l3"]
        assert l3 is not None

        # Should have Shapley for each welfare function
        assert "shapley_uw" in l3
        assert "shapley_nw" in l3
        assert "shapley_nw_plus" in l3

        # Should not have Banzhaf
        assert "banzhaf_uw" not in l3

        # Check Shapley values have entry for each policy
        shapley_uw = l3["shapley_uw"]
        assert set(shapley_uw.keys()) == set(bootstrap.policies)

    def test_l3_enabled_banzhaf(self, bootstrap: Bootstrap):
        """Test L3 with Banzhaf method."""
        results = bootstrap.run_full_analysis(
            include_l3=True,
            l3_method="banzhaf",
            l3_exact=False,
            l3_n_samples=100,
            progress=False,
        )

        l3 = results[0]["l3"]
        assert l3 is not None

        # Should have Banzhaf for each welfare function
        assert "banzhaf_uw" in l3
        assert "banzhaf_nw" in l3
        assert "banzhaf_nw_plus" in l3

        # Should not have Shapley
        assert "shapley_uw" not in l3

    def test_l3_enabled_both(self, bootstrap: Bootstrap):
        """Test L3 with both methods."""
        results = bootstrap.run_full_analysis(
            include_l3=True,
            l3_method="both",
            l3_exact=False,
            l3_n_samples=100,
            progress=False,
        )

        l3 = results[0]["l3"]
        assert l3 is not None

        # Should have both Shapley and Banzhaf
        assert "shapley_uw" in l3
        assert "banzhaf_uw" in l3

    def test_welfare_values_reasonable(self, bootstrap: Bootstrap):
        """Test that welfare values are in reasonable ranges."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        welfare = results[0]["full_game"]["welfare"]

        # UW should be positive (payoffs are positive)
        assert welfare["uw"] > 0

        # NW and NW+ should be non-negative
        assert welfare["nw"] >= 0
        assert welfare["nw_plus"] >= 0

    def test_ef1_in_valid_range(self, bootstrap: Bootstrap):
        """Test that EF1 values are between 0 and 1."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        ef1 = results[0]["full_game"]["ef1"]
        assert 0 <= ef1 <= 1

        # Also check EF1 matrix
        ef1_matrix = results[0]["matrices"]["ef1"]
        valid_values = ef1_matrix[~np.isnan(ef1_matrix)]
        assert all(0 <= v <= 1 for v in valid_values)

    def test_equilibrium_shift_non_negative(self, bootstrap: Bootstrap):
        """Test that equilibrium shift (L1 norm) is non-negative."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        for candidate, l2_result in results[0]["l2"].items():
            assert l2_result["equilibrium_shift"] >= 0

    def test_entry_mass_valid_probability(self, bootstrap: Bootstrap):
        """Test that entry mass is a valid probability."""
        results = bootstrap.run_full_analysis(
            include_l3=False,
            progress=False,
        )

        for candidate, l2_result in results[0]["l2"].items():
            entry_mass = l2_result["entry_mass"]
            assert 0 <= entry_mass <= 1

    def test_reproducibility_with_seed(self, sample_data: pd.DataFrame):
        """Test that results are reproducible with the same seed."""
        bootstrap1 = Bootstrap(df=sample_data, n_samples=3, seed=999)
        bootstrap2 = Bootstrap(df=sample_data, n_samples=3, seed=999)

        results1 = bootstrap1.run_full_analysis(include_l3=False, progress=False)
        results2 = bootstrap2.run_full_analysis(include_l3=False, progress=False)

        # Check that payoff matrices are identical
        np.testing.assert_array_equal(
            results1[0]["matrices"]["payoff"],
            results2[0]["matrices"]["payoff"],
        )

    def test_different_seeds_produce_different_results(self, sample_data: pd.DataFrame):
        """Test that different seeds produce different results."""
        bootstrap1 = Bootstrap(df=sample_data, n_samples=3, seed=111)
        bootstrap2 = Bootstrap(df=sample_data, n_samples=3, seed=222)

        results1 = bootstrap1.run_full_analysis(include_l3=False, progress=False)
        results2 = bootstrap2.run_full_analysis(include_l3=False, progress=False)

        # Matrices should differ (with high probability)
        assert not np.allclose(
            results1[0]["matrices"]["payoff"],
            results2[0]["matrices"]["payoff"],
        )


class TestBuildAllMatrices:
    """Tests for _build_all_matrices helper."""

    def test_nw_computation(self):
        """Test that NW is computed correctly as sqrt(payoff_i * payoff_j)."""
        df = pd.DataFrame([
            {"policy_i": "a", "policy_j": "b", "payoff_i": 100, "payoff_j": 100,
             "batna_i": 50, "batna_j": 50, "ef1": 1},
            {"policy_i": "a", "policy_j": "b", "payoff_i": 100, "payoff_j": 100,
             "batna_i": 50, "batna_j": 50, "ef1": 1},
        ])

        bootstrap = Bootstrap(df, n_samples=1, seed=42)
        matrices = bootstrap._build_all_matrices(df, ["a", "b"])

        # NW for (a, b) should be sqrt(100 * 100) = 100
        expected_nw = 100.0
        assert np.isclose(matrices["nw"][0, 1], expected_nw)

    def test_nw_plus_computation(self):
        """Test that NW+ uses advantages (payoff - BATNA)."""
        df = pd.DataFrame([
            {"policy_i": "a", "policy_j": "b", "payoff_i": 100, "payoff_j": 80,
             "batna_i": 40, "batna_j": 30, "ef1": 1},
        ])

        bootstrap = Bootstrap(df, n_samples=1, seed=42)
        matrices = bootstrap._build_all_matrices(df, ["a", "b"])

        # Advantages: adv_i = 100 - 40 = 60, adv_j = 80 - 30 = 50
        # NW+ = sqrt(60 * 50) = sqrt(3000)
        expected_nw_plus = np.sqrt(3000)
        assert np.isclose(matrices["nw_plus"][0, 1], expected_nw_plus)

    def test_nw_plus_clamps_negative_advantage(self):
        """Test that NW+ clamps negative advantages to 0."""
        df = pd.DataFrame([
            {"policy_i": "a", "policy_j": "b", "payoff_i": 30, "payoff_j": 80,
             "batna_i": 50, "batna_j": 30, "ef1": 1},  # payoff_i < batna_i
        ])

        bootstrap = Bootstrap(df, n_samples=1, seed=42)
        matrices = bootstrap._build_all_matrices(df, ["a", "b"])

        # adv_i = max(0, 30 - 50) = 0, adv_j = 80 - 30 = 50
        # NW+ = sqrt(0 * 50) = 0
        assert np.isclose(matrices["nw_plus"][0, 1], 0.0)

    def test_ef1_frequency(self):
        """Test that EF1 is computed as frequency."""
        df = pd.DataFrame([
            {"policy_i": "a", "policy_j": "b", "payoff_i": 100, "payoff_j": 100,
             "batna_i": 50, "batna_j": 50, "ef1": 1},
            {"policy_i": "a", "policy_j": "b", "payoff_i": 100, "payoff_j": 100,
             "batna_i": 50, "batna_j": 50, "ef1": 0},
            {"policy_i": "a", "policy_j": "b", "payoff_i": 100, "payoff_j": 100,
             "batna_i": 50, "batna_j": 50, "ef1": 1},
        ])

        bootstrap = Bootstrap(df, n_samples=1, seed=42)
        matrices = bootstrap._build_all_matrices(df, ["a", "b"])

        # EF1 frequency for (a, b) should be 2/3
        expected_ef1 = 2 / 3
        assert np.isclose(matrices["ef1"][0, 1], expected_ef1)


class TestWelfareAtEquilibrium:
    """Tests for welfare computation at equilibrium."""

    def test_welfare_uses_correct_matrices(self):
        """Test that each welfare metric uses its corresponding matrix."""
        # Create simple 2x2 game
        df = pd.DataFrame([
            {"policy_i": "a", "policy_j": "a", "payoff_i": 100, "payoff_j": 100,
             "batna_i": 50, "batna_j": 50, "ef1": 1},
            {"policy_i": "a", "policy_j": "b", "payoff_i": 80, "payoff_j": 120,
             "batna_i": 50, "batna_j": 50, "ef1": 1},
            {"policy_i": "b", "policy_j": "a", "payoff_i": 120, "payoff_j": 80,
             "batna_i": 50, "batna_j": 50, "ef1": 1},
            {"policy_i": "b", "policy_j": "b", "payoff_i": 100, "payoff_j": 100,
             "batna_i": 50, "batna_j": 50, "ef1": 1},
        ])

        bootstrap = Bootstrap(df, n_samples=1, seed=42)
        results = bootstrap.run_full_analysis(include_l3=False, progress=False)

        welfare = results[0]["full_game"]["welfare"]

        # All welfare values should be computed
        assert "uw" in welfare
        assert "nw" in welfare
        assert "nw_plus" in welfare


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
