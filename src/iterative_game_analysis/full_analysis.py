"""Full iterative meta-game analysis pipeline.

Provides data loading, bootstrap L1/L2/L3 analysis, result aggregation,
and persistence. This is the main entry point for running a complete
analysis on crossplay data.

Typical usage:

    from .full_analysis import (
        load_crossplay_to_dataframe,
        run_full_pipeline,
        save_results,
        load_results,
    )

    df = load_crossplay_to_dataframe(crossplay_dir, strategy_names)
    results = run_full_pipeline(df, strategy_names, num_bootstrap=1000)
    save_results(results, output_path)

    # Later: reload for further analysis
    results = load_results(output_path)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .bootstrap import Bootstrap

# Item quantities for the bargaining game
ITEM_QUANTITIES = np.array([7, 4, 1])

# Normalization constants: expected max welfare over the game distribution
MAX_UW = 805.9
MAX_NW = 378.7
MAX_NW_PLUS = 81.7


def _is_ef1_vectorized(
    utilities_p1: np.ndarray,
    utilities_p2: np.ndarray,
    allocation_p1: np.ndarray,
    allocation_p2: np.ndarray,
) -> np.ndarray:
    """Vectorized EF1 check for multiple games."""
    p1_own = np.sum(utilities_p1 * allocation_p1, axis=1)
    p1_other = np.sum(utilities_p1 * allocation_p2, axis=1)
    p1_max_item = np.max(np.where(allocation_p2 > 0, utilities_p1, 0), axis=1)
    p1_ef1 = p1_own >= p1_other - p1_max_item

    p2_own = np.sum(utilities_p2 * allocation_p2, axis=1)
    p2_other = np.sum(utilities_p2 * allocation_p1, axis=1)
    p2_max_item = np.max(np.where(allocation_p1 > 0, utilities_p2, 0), axis=1)
    p2_ef1 = p2_own >= p2_other - p2_max_item

    return p1_ef1 & p2_ef1


def _exists_ef1_beating_batnas(
    utilities_p1: np.ndarray,
    utilities_p2: np.ndarray,
    raw_batna_p1: np.ndarray,
    raw_batna_p2: np.ndarray,
) -> np.ndarray:
    """Check if an EF1 allocation exists that beats both BATNAs.

    Enumerates all possible allocations for items = [7, 4, 1].

    Args:
        utilities_p1: (n_games, 3) per-item values for P1.
        utilities_p2: (n_games, 3) per-item values for P2.
        raw_batna_p1: (n_games,) P1's BATNA in raw utility space.
        raw_batna_p2: (n_games,) P2's BATNA in raw utility space.

    Returns:
        (n_games,) boolean — True if a valid allocation exists.
    """
    items = ITEM_QUANTITIES  # [7, 4, 1]

    all_allocs = np.array([
        (a, b, c)
        for a in range(items[0] + 1)
        for b in range(items[1] + 1)
        for c in range(items[2] + 1)
    ])
    p2_allocs = items - all_allocs

    p1_payoffs = utilities_p1 @ all_allocs.T
    p2_payoffs = utilities_p2 @ p2_allocs.T

    beats_batna = (
        (p1_payoffs > raw_batna_p1[:, None])
        & (p2_payoffs > raw_batna_p2[:, None])
    )

    # EF1 check for P1
    p1_other_vals = utilities_p1 @ p2_allocs.T
    p2_has_items = p2_allocs > 0
    p1_vals_masked = np.where(
        p2_has_items[None, :, :], utilities_p1[:, None, :], 0
    )
    p1_max_remove = np.max(p1_vals_masked, axis=2)
    p1_ef1 = p1_payoffs >= p1_other_vals - p1_max_remove

    # EF1 check for P2
    p2_other_vals = utilities_p2 @ all_allocs.T
    p1_has_items = all_allocs > 0
    p2_vals_masked = np.where(
        p1_has_items[None, :, :], utilities_p2[:, None, :], 0
    )
    p2_max_remove = np.max(p2_vals_masked, axis=2)
    p2_ef1 = p2_payoffs >= p2_other_vals - p2_max_remove

    valid = p1_ef1 & p2_ef1 & beats_batna
    return np.any(valid, axis=1)


def load_crossplay_to_dataframe(
    crossplay_dir: Path | str,
    strategy_names: List[str],
    raw_utility: bool = True,
) -> pd.DataFrame:
    """Load crossplay game data into a DataFrame for Bootstrap analysis.

    Reads JSON game files from the crossplay directory and produces a flat
    DataFrame with one row per game per role direction (two rows per game
    for role-symmetrization).

    Args:
        crossplay_dir: Path to directory containing matchup subdirectories.
            Expected structure: {strat_i}_p1_vs_{strat_j}_p2/games.json
        strategy_names: List of strategy names to include.
        raw_utility: If True, convert normalized payoffs to raw utility
            space (payoff * max_possible_value). Default True.

    Returns:
        DataFrame with columns: policy_i, policy_j, payoff_i, payoff_j,
        batna_i, batna_j, ef1, ef1_plus. EF1 is NaN for non-accept games.
        EF1+ is NaN for non-accept or non-rational games.
    """
    crossplay_dir = Path(crossplay_dir)

    try:
        import orjson

        def _load_json(path):
            with open(path, "rb") as f:
                return orjson.loads(f.read())
    except ImportError:
        def _load_json(path):
            with open(path, "r") as f:
                return json.load(f)

    dfs = []

    for strat_i in strategy_names:
        for strat_j in strategy_names:
            games_path = crossplay_dir / f"{strat_i}_p1_vs_{strat_j}_p2" / "games.json"

            if not games_path.exists():
                print(f"  Warning: No games found for {strat_i} vs {strat_j}")
                continue

            print(f"  Loading {strat_i} vs {strat_j}...")
            data = _load_json(games_path)
            games = data["games"]
            n_games = len(games)

            # Pre-allocate arrays
            payoff_p1 = np.zeros(n_games)
            payoff_p2 = np.zeros(n_games)
            batna_p1 = np.zeros(n_games)
            batna_p2 = np.zeros(n_games)
            utilities_p1 = np.zeros((n_games, 3))
            utilities_p2 = np.zeros((n_games, 3))
            ef1 = np.full(n_games, np.nan)

            # Track accepted games with allocation data for EF1
            accept_indices = []
            alloc_p1_list = []
            alloc_p2_list = []
            util_p1_list = []
            util_p2_list = []

            for idx, game in enumerate(games):
                outcome = game["outcome"]
                payoff_p1[idx] = outcome["payoff_p1"]
                payoff_p2[idx] = outcome["payoff_p2"]
                batna_p1[idx] = outcome["batna_p1"]
                batna_p2[idx] = outcome["batna_p2"]

                if outcome.get("utilities_p1"):
                    utilities_p1[idx] = outcome["utilities_p1"]
                    utilities_p2[idx] = outcome["utilities_p2"]

                if outcome["result"] == "accept":
                    if outcome.get("utilities_p1") and outcome.get("allocation_p1"):
                        accept_indices.append(idx)
                        util_p1_list.append(outcome["utilities_p1"])
                        util_p2_list.append(outcome["utilities_p2"])
                        alloc_p1_list.append(outcome["allocation_p1"])
                        alloc_p2_list.append(outcome["allocation_p2"])

            # Compute EF1 vectorized for accepted games with allocations
            if accept_indices:
                ef1_results = _is_ef1_vectorized(
                    np.array(util_p1_list),
                    np.array(util_p2_list),
                    np.array(alloc_p1_list),
                    np.array(alloc_p2_list),
                )
                ef1[accept_indices] = ef1_results.astype(float)

            # Raw utility space (always needed for EF1+)
            max_p1 = np.sum(utilities_p1 * ITEM_QUANTITIES, axis=1)
            max_p2 = np.sum(utilities_p2 * ITEM_QUANTITIES, axis=1)
            raw_pay_p1 = payoff_p1 * max_p1
            raw_pay_p2 = payoff_p2 * max_p2
            raw_bat_p1 = batna_p1 * max_p1
            raw_bat_p2 = batna_p2 * max_p2

            # EF1+: outcome beats both BATNAs AND a rational EF1 allocation exists
            outcome_beats = (raw_pay_p1 > raw_bat_p1) & (raw_pay_p2 > raw_bat_p2)
            ef1_rational_exists = _exists_ef1_beating_batnas(
                utilities_p1, utilities_p2, raw_bat_p1, raw_bat_p2,
            )
            # ef1_plus: NaN for non-accept or non-rational games,
            # 1 if ef1 & outcome beats BATNAs, 0 otherwise
            ef1_plus = np.full(n_games, np.nan)
            is_accept = np.zeros(n_games, dtype=bool)
            is_accept[accept_indices] = True
            rational_accept = is_accept & ef1_rational_exists
            ef1_plus[rational_accept] = (
                ef1[rational_accept].astype(bool) & outcome_beats[rational_accept]
            ).astype(float)

            # Convert to raw utility space if requested
            if raw_utility:
                pay_p1 = raw_pay_p1
                pay_p2 = raw_pay_p2
                bat_p1 = raw_bat_p1
                bat_p2 = raw_bat_p2
            else:
                pay_p1 = payoff_p1
                pay_p2 = payoff_p2
                bat_p1 = batna_p1
                bat_p2 = batna_p2

            # Two rows per game for role-symmetrization
            forward = pd.DataFrame({
                "policy_i": strat_i,
                "policy_j": strat_j,
                "payoff_i": pay_p1,
                "payoff_j": pay_p2,
                "batna_i": bat_p1,
                "batna_j": bat_p2,
                "ef1": ef1,
                "ef1_plus": ef1_plus,
            })
            reverse = pd.DataFrame({
                "policy_i": strat_j,
                "policy_j": strat_i,
                "payoff_i": pay_p2,
                "payoff_j": pay_p1,
                "batna_i": bat_p2,
                "batna_j": bat_p1,
                "ef1": ef1,
                "ef1_plus": ef1_plus,
            })
            dfs.append(forward)
            dfs.append(reverse)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(df)} total rows ({len(df) // 2} games x 2 directions)")
    return df


def _summarize(samples: List[float]) -> Dict:
    """Compute mean, std, 95% CI from bootstrap samples."""
    arr = np.array(samples)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
    }


def aggregate_results(
    results: List[dict],
    strategy_names: List[str],
) -> Dict:
    """Aggregate raw bootstrap results into summary statistics.

    Computes mean, std, and 95% CI for every metric across bootstrap
    samples.

    Args:
        results: Raw per-sample results from Bootstrap.run_full_analysis().
        strategy_names: Ordered list of strategy names.

    Returns:
        Dict with aggregated metrics for full_game, l1, l2, l3.
    """
    n_bootstrap = len(results)

    METRICS = ["payoff", "nw", "nw_plus", "ef1", "ef1_plus"]

    # --- Full game ---
    sigma_samples = [r["full_game"]["sigma"].tolist() for r in results]
    support_counts = np.zeros(len(strategy_names), dtype=int)
    for r in results:
        support_counts += (r["full_game"]["sigma"] > 1e-2).astype(int)

    full_game = {
        "equilibrium": {
            "mean": np.mean(sigma_samples, axis=0).tolist(),
            "std": np.std(sigma_samples, axis=0).tolist(),
        },
        "support_frequency": {
            name: float(support_counts[i] / n_bootstrap * 100)
            for i, name in enumerate(strategy_names)
        },
        "welfare": {},
        "ef1": _summarize([r["full_game"]["ef1"] for r in results]),
        "ef1_plus": _summarize([r["full_game"]["ef1_plus"] for r in results]),
        "regret": {},
        "per_agent_values": {},
    }
    for wf in ["uw", "nw", "nw_plus"]:
        full_game["welfare"][wf] = _summarize(
            [r["full_game"]["welfare"][wf] for r in results]
        )
    for name in strategy_names:
        full_game["regret"][name] = _summarize(
            [r["full_game"]["regret"][name] for r in results]
        )
    # Per-agent values at equilibrium for every metric
    for metric in METRICS:
        full_game["per_agent_values"][metric] = {}
        for name in strategy_names:
            full_game["per_agent_values"][metric][name] = _summarize(
                [r["full_game"]["per_agent_values"][metric][name] for r in results]
            )

    # --- L1: Partner lift per agent, all metrics ---
    l1 = {}
    for candidate in strategy_names:
        l1[candidate] = {
            "aggregations": {},
            "per_incumbent": {},
            "welfare_B": {},
            "ef1_B": _summarize(
                [r["l1"][candidate]["ef1_B"] for r in results]
            ),
            "ef1_plus_B": _summarize(
                [r["l1"][candidate]["ef1_plus_B"] for r in results]
            ),
        }
        # Per-metric aggregations (uniform_avg, eq_avg, min, max)
        for metric in METRICS:
            l1[candidate]["aggregations"][metric] = {}
            for agg_key in ["uniform_avg", "equilibrium_avg", "min", "max"]:
                l1[candidate]["aggregations"][metric][agg_key] = _summarize(
                    [r["l1"][candidate]["aggregations"][metric][agg_key]
                     for r in results]
                )
        # Per-metric per-incumbent lift
        incumbents = [p for p in strategy_names if p != candidate]
        for metric in METRICS:
            l1[candidate]["per_incumbent"][metric] = {}
            for inc in incumbents:
                l1[candidate]["per_incumbent"][metric][inc] = _summarize(
                    [r["l1"][candidate]["per_incumbent"][metric][inc]
                     for r in results]
                )
        for wf in ["uw", "nw", "nw_plus"]:
            l1[candidate]["welfare_B"][wf] = _summarize(
                [r["l1"][candidate]["welfare_B"][wf] for r in results]
            )

    # --- L2: Ecosystem lift per agent, all metrics ---
    l2 = {}
    for candidate in strategy_names:
        l2[candidate] = {
            "delta_eco": {},
            "entry_mass": _summarize(
                [r["l2"][candidate]["entry_mass"] for r in results]
            ),
            "equilibrium_shift": _summarize(
                [r["l2"][candidate]["equilibrium_shift"] for r in results]
            ),
            "ef1_lift": _summarize(
                [r["l2"][candidate]["ef1_lift"] for r in results]
            ),
            "ef1_plus_lift": _summarize(
                [r["l2"][candidate]["ef1_plus_lift"] for r in results]
            ),
            "incumbent_shifts": {},
        }
        for wf in ["uw", "nw", "nw_plus"]:
            l2[candidate]["delta_eco"][wf] = _summarize(
                [r["l2"][candidate]["delta_eco"][wf] for r in results]
            )
        # Per-metric incumbent shifts
        incumbents = [p for p in strategy_names if p != candidate]
        for metric in METRICS:
            l2[candidate]["incumbent_shifts"][metric] = {}
            for inc in incumbents:
                l2[candidate]["incumbent_shifts"][metric][inc] = _summarize(
                    [r["l2"][candidate]["incumbent_shifts"][metric][inc]
                     for r in results]
                )

    # --- L3: Shapley / Banzhaf ---
    l3 = None
    if results[0].get("l3") is not None:
        l3 = {}
        for key in results[0]["l3"]:
            if key == "total_value":
                l3["total_value"] = {}
                for wf in results[0]["l3"]["total_value"]:
                    l3["total_value"][wf] = _summarize(
                        [r["l3"]["total_value"][wf] for r in results]
                    )
            elif key == "coalition_details":
                # Raw per-coalition data — preserved in pickle, skip aggregation
                continue
            else:
                l3[key] = {}
                for name in strategy_names:
                    l3[key][name] = _summarize(
                        [r["l3"][key][name] for r in results]
                    )

    return {
        "strategy_names": strategy_names,
        "num_bootstrap": n_bootstrap,
        "full_game": full_game,
        "l1": l1,
        "l2": l2,
        "l3": l3,
    }


def run_full_pipeline(
    df: pd.DataFrame,
    strategy_names: List[str],
    num_bootstrap: int = 1000,
    seed: int = 42,
    solver: str = "mene",
    include_l3: bool = True,
    l3_method: str = "both",
    n_workers: int = 1,
) -> Dict:
    """Run the full iterative meta-game analysis pipeline.

    Takes a pre-loaded DataFrame (from load_crossplay_to_dataframe or any
    compatible source) and runs bootstrap L1/L2/L3 analysis.

    Args:
        df: DataFrame with columns policy_i, policy_j, payoff_i, payoff_j,
            batna_i, batna_j, ef1.
        strategy_names: Ordered list of strategy names.
        num_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.
        solver: Equilibrium solver ("mene" or "uniform").
        include_l3: Whether to compute Level 3 attribution.
        l3_method: "shapley", "banzhaf", or "both".
        n_workers: Number of parallel workers for L3 coalition solves.

    Returns:
        Dict with keys:
        - "raw": List of per-sample results (full L1/L2/L3 data)
        - "aggregated": Summary statistics (mean, std, CI)
        - "config": Analysis configuration parameters
    """
    print(f"Creating Bootstrap with {num_bootstrap} samples...")
    bootstrap = Bootstrap(
        df=df,
        n_samples=num_bootstrap,
        seed=seed,
        policies=strategy_names,
    )

    print(f"Running full L1/L2/L3 analysis (solver={solver})...")
    raw_results = bootstrap.run_full_analysis(
        solver=solver,
        include_l3=include_l3,
        l3_method=l3_method,
        progress=True,
        n_workers=n_workers,
    )

    print(f"Aggregating {len(raw_results)} bootstrap samples...")
    aggregated = aggregate_results(raw_results, strategy_names)

    return {
        "raw": raw_results,
        "aggregated": aggregated,
        "config": {
            "strategy_names": strategy_names,
            "num_bootstrap": num_bootstrap,
            "seed": seed,
            "solver": solver,
            "include_l3": include_l3,
            "l3_method": l3_method,
        },
    }


def _make_serializable(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


def save_results(results: Dict, output_path: Path | str) -> None:
    """Save full analysis results (raw + aggregated) to disk.

    Saves two files:
    - {output_path}.pkl: Full raw results (preserves numpy arrays)
    - {output_path}.json: Aggregated summary (human-readable)

    Args:
        results: Output from run_full_pipeline().
        output_path: Base path (without extension). Both .pkl and .json
            will be created.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full raw results as pickle (preserves numpy arrays)
    pkl_path = output_path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved raw results to {pkl_path}")

    json_path = output_path.with_suffix(".json")
    json_data = _make_serializable({
        "config": results["config"],
        "aggregated": results["aggregated"],
    })
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved aggregated results to {json_path}")


def load_results(output_path: Path | str) -> Dict:
    """Load full analysis results from disk.

    Args:
        output_path: Base path (without extension). Loads from .pkl file.

    Returns:
        Full results dict with "raw", "aggregated", and "config" keys.
    """
    pkl_path = Path(output_path).with_suffix(".pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def print_results(agg: Dict, raw_utility: bool = True) -> None:
    """Pretty-print aggregated iterative analysis results.

    Args:
        agg: Aggregated results dict (from aggregate_results or
            results["aggregated"]).
        raw_utility: Whether payoffs are in raw utility space (affects
            normalization constants for display).
    """
    names = agg["strategy_names"]

    print("\n" + "=" * 70)
    print("ITERATIVE META-GAME ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Strategies: {names}")
    print(f"Bootstrap samples: {agg['num_bootstrap']}")

    fg = agg["full_game"]
    print("\n--- Equilibrium Distribution ---")
    eq = fg["equilibrium"]
    for i, name in enumerate(names):
        print(f"  {name:20s}: {eq['mean'][i]:.4f} +/- {eq['std'][i]:.4f}")

    print("\n--- Support Frequency % ---")
    for name in names:
        print(f"  {name:20s}: {fg['support_frequency'][name]:.1f}%")

    print("\n--- Per-Agent Regret (95% CI) ---")
    for name in names:
        r = fg["regret"][name]
        print(f"  {name:20s}: {r['mean']:.2f} +/- {r['std']:.2f}  "
              f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")

    if raw_utility:
        divs = {"uw": MAX_UW, "nw": MAX_NW, "nw_plus": MAX_NW_PLUS}
    else:
        divs = {"uw": 2.0, "nw": 1.0, "nw_plus": 1.0}

    for wf, label in [("uw", "UW"), ("nw", "NW"), ("nw_plus", "NW+")]:
        d = divs[wf]
        m = fg["welfare"][wf]
        print(f"\n--- Full Game {label} % (95% CI, div={d}) ---")
        print(f"  {'ecosystem':20s}: {m['mean']/d*100:.2f} +/- {m['std']/d*100:.2f}  "
              f"[{m['ci_lower']/d*100:.2f}, {m['ci_upper']/d*100:.2f}]")

    e = fg["ef1"]
    print(f"\n--- Full Game EF1 % (95% CI) ---")
    print(f"  {'ecosystem':20s}: {e['mean']*100:.2f} +/- {e['std']*100:.2f}  "
          f"[{e['ci_lower']*100:.2f}, {e['ci_upper']*100:.2f}]")

    ep = fg["ef1_plus"]
    print(f"\n--- Full Game EF1+ % (95% CI, rational games only) ---")
    print(f"  {'ecosystem':20s}: {ep['mean']*100:.2f} +/- {ep['std']*100:.2f}  "
          f"[{ep['ci_lower']*100:.2f}, {ep['ci_upper']*100:.2f}]")

    # --- L1: Partner Lift (all metrics) ---
    print("\n" + "=" * 70)
    print("LEVEL 1: PARTNER LIFT (leave-one-out, no re-equilibration)")
    print("=" * 70)

    metric_labels = [("payoff", "Payoff"), ("nw", "NW"), ("nw_plus", "NW+"), ("ef1", "EF1"), ("ef1_plus", "EF1+")]

    for candidate in names:
        l1 = agg["l1"][candidate]
        print(f"\n  Candidate (removed): {candidate}")
        for metric, mlabel in metric_labels:
            aggs = l1["aggregations"][metric]
            print(f"    [{mlabel}]")
            for agg_name, key in [
                ("Uniform avg", "uniform_avg"),
                ("Eq-weighted avg", "equilibrium_avg"),
                ("Worst-case", "min"),
                ("Best-case", "max"),
            ]:
                m = aggs[key]
                print(f"      {agg_name:20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
                      f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

   
    print("\n" + "=" * 70)
    print("LEVEL 2: ECOSYSTEM LIFT (leave-one-out, with re-equilibration)")
    print("=" * 70)
    for candidate in names:
        l2 = agg["l2"][candidate]
        print(f"\n  Candidate (removed then re-added): {candidate}")

        for wf, label in [("uw", "UW"), ("nw", "NW"), ("nw_plus", "NW+")]:
            m = l2["delta_eco"][wf]
            print(f"    delta_eco({label:3s})     : {m['mean']:.4f} +/- {m['std']:.4f}  "
                  f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

        m = l2["entry_mass"]
        print(f"    {'entry_mass':20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
              f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

        m = l2["equilibrium_shift"]
        print(f"    {'eq_shift (L1 norm)':20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
              f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

        m = l2["ef1_lift"]
        print(f"    {'ef1_lift':20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
              f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

        m = l2["ef1_plus_lift"]
        print(f"    {'ef1_plus_lift':20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
              f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

        # Per-metric incumbent shifts
        incumbents = [p for p in names if p != candidate]
        for metric, mlabel in metric_labels:
            print(f"    Incumbent shifts [{mlabel}]:")
            for inc in incumbents:
                m = l2["incumbent_shifts"][metric][inc]
                print(f"      {inc:20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
                      f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

    # --- L3: Attribution ---
    if agg["l3"] is not None:
        print("\n" + "=" * 70)
        print("LEVEL 3: ATTRIBUTION (Shapley / Banzhaf)")
        print("=" * 70)

        for key in sorted(agg["l3"].keys()):
            if key == "total_value":
                continue
            print(f"\n  --- {key} ---")
            for name in names:
                m = agg["l3"][key][name]
                print(f"    {name:20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
                      f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

        if "total_value" in agg["l3"]:
            print(f"\n  --- Total Ecosystem Value ---")
            for wf in agg["l3"]["total_value"]:
                m = agg["l3"]["total_value"][wf]
                print(f"    {wf:20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
                      f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")