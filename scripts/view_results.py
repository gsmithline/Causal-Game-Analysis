#!/usr/bin/env python3
"""
View and compare training results.

Usage:
    # List all completed runs
    python scripts/view_results.py --list

    # List runs for specific algorithm
    python scripts/view_results.py --list --algorithm ppo

    # Show details of a specific run
    python scripts/view_results.py --show ppo_20240115_143022_seed42

    # Compare multiple runs
    python scripts/view_results.py --compare ppo_run1 mappo_run2 psro_run3

    # Find best run for an algorithm
    python scripts/view_results.py --best ppo

    # Export results to CSV
    python scripts/view_results.py --export results.csv

    # Load and use a trained policy
    python scripts/view_results.py --load ppo_20240115_143022_seed42 --evaluate
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import torch

from rl_training.utils.results_manager import ResultsManager
from rl_training.envs.bargain_wrapper import BargainEnvWrapper
from rl_training.networks.bargain_mlp import BargainMLP
from rl_training.networks.transformer_policy import TransformerPolicyNetwork


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_run_summary(run: dict):
    """Print a formatted summary of a run."""
    print(f"\n{'='*60}")
    print(f"Run ID: {run.get('run_id', 'N/A')}")
    print(f"{'='*60}")

    print(f"\n📋 Basic Info:")
    print(f"   Algorithm:    {run.get('algorithm', 'N/A')}")
    print(f"   Environment:  {run.get('environment', 'bargain')}")
    print(f"   Seed:         {run.get('seed', 'N/A')}")
    print(f"   Status:       {run.get('status', 'N/A')}")

    if run.get('training_time_seconds'):
        print(f"   Training Time: {format_duration(run['training_time_seconds'])}")

    if run.get('hyperparameters'):
        print(f"\n⚙️  Hyperparameters:")
        for key, value in run['hyperparameters'].items():
            print(f"   {key}: {value}")

    if run.get('final_metrics'):
        print(f"\n📊 Final Metrics:")
        for key, value in run['final_metrics'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

    if run.get('best_metrics'):
        print(f"\n🏆 Best Metrics (during training):")
        for key, value in run['best_metrics'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

    print(f"\n📁 Path: {run.get('path', 'N/A')}")


def print_comparison_table(runs: list):
    """Print a comparison table of multiple runs."""
    if not runs:
        print("No runs to compare.")
        return

    # Collect all metrics
    all_metrics = set()
    for run in runs:
        if run.get('final_metrics'):
            all_metrics.update(run['final_metrics'].keys())

    # Header
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")

    # Print header row
    header = f"{'Run ID':<35} {'Algorithm':<12} {'Seed':<6}"
    for metric in sorted(all_metrics):
        header += f" {metric[:15]:<15}"
    print(header)
    print("-" * 100)

    # Print each run
    for run in runs:
        row = f"{run.get('run_id', 'N/A')[:35]:<35} "
        row += f"{run.get('algorithm', 'N/A'):<12} "
        row += f"{run.get('seed', 'N/A'):<6}"

        for metric in sorted(all_metrics):
            value = run.get('final_metrics', {}).get(metric, 'N/A')
            if isinstance(value, float):
                row += f" {value:<15.4f}"
            else:
                row += f" {str(value):<15}"
        print(row)

    print("=" * 100)


def print_leaderboard(manager: ResultsManager, metric: str = "final_metrics.final_total_reward"):
    """Print a leaderboard across all algorithms."""
    algorithms = ["ppo", "nfsp", "sampled_cfr", "psro", "mappo", "fcp"]

    print(f"\n{'='*70}")
    print(f"LEADERBOARD (by {metric})")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Algorithm':<15} {'Run ID':<30} {'Score':<12}")
    print("-" * 70)

    all_best = []
    for algo in algorithms:
        best = manager.get_best_run(algo, metric=metric)
        if best:
            # Extract metric value
            parts = metric.split(".")
            value = best
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            if value is not None:
                all_best.append((algo, best, value))

    # Sort by score
    all_best.sort(key=lambda x: x[2], reverse=True)

    for rank, (algo, run, score) in enumerate(all_best, 1):
        print(f"{rank:<6} {algo:<15} {run['run_id'][:30]:<30} {score:<12.4f}")

    print("=" * 70)


def evaluate_loaded_policy(
    policy_state: dict,
    algorithm: str,
    hyperparameters: dict,
    num_games: int = 1000,
    cuda_device: int = 0,
):
    """Evaluate a loaded policy."""
    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"

    # Determine network type and create
    network_type = hyperparameters.get("network_type", "mlp")

    if network_type == "transformer":
        policy = TransformerPolicyNetwork(
            obs_dim=BargainEnvWrapper.OBS_DIM,
            num_actions=BargainEnvWrapper.NUM_ACTIONS,
        )
    else:
        policy = BargainMLP(
            obs_dim=BargainEnvWrapper.OBS_DIM,
            num_actions=BargainEnvWrapper.NUM_ACTIONS,
        )

    # Load weights
    if "policy_state_dict" in policy_state:
        policy.load_state_dict(policy_state["policy_state_dict"])
    else:
        # Try to load directly (might be the full checkpoint)
        try:
            policy.load_state_dict(policy_state)
        except:
            print("Warning: Could not load policy weights directly.")
            return None

    policy = policy.to(device)
    policy.eval()

    # Create environment
    env = BargainEnvWrapper(
        num_envs=min(num_games, 1024),
        self_play=True,
        device=cuda_device if torch.cuda.is_available() else 0,
        seed=99999,
    )

    total_rewards = [0.0, 0.0]
    games_completed = 0

    obs, info = env.reset()
    action_mask = info['action_mask']

    print(f"\nEvaluating policy over {num_games} games...")

    while games_completed < num_games:
        with torch.no_grad():
            actions, _, _ = policy.get_action(obs, action_mask)

        result = env.step(actions.int())
        dones = result.terminated

        if dones.any():
            rewards = result.rewards[dones].cpu().numpy()
            total_rewards[0] += rewards[:, 0].sum()
            total_rewards[1] += rewards[:, 1].sum()
            games_completed += dones.sum().item()
            env.auto_reset()

        obs = result.observations
        action_mask = result.info['action_mask']

    env.close()

    avg_rewards = [r / num_games for r in total_rewards]
    print(f"\nEvaluation Results ({num_games} games):")
    print(f"   Player 0 avg reward: {avg_rewards[0]:.4f}")
    print(f"   Player 1 avg reward: {avg_rewards[1]:.4f}")
    print(f"   Total avg reward:    {sum(avg_rewards):.4f}")

    return avg_rewards


def main():
    parser = argparse.ArgumentParser(description="View and compare training results")

    # Actions
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--show", type=str, help="Show details of a specific run")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare multiple runs")
    parser.add_argument("--best", type=str, help="Find best run for an algorithm")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")
    parser.add_argument("--export", type=str, help="Export results to file")
    parser.add_argument("--load", type=str, help="Load a trained policy")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate loaded policy")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup failed runs")

    # Filters
    parser.add_argument("--algorithm", type=str, help="Filter by algorithm")
    parser.add_argument("--status", type=str, help="Filter by status")
    parser.add_argument("--metric", type=str, default="final_metrics.final_total_reward",
                       help="Metric for comparison/best")

    # Options
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "json"],
                       help="Export format")

    args = parser.parse_args()

    manager = ResultsManager(args.results_dir)

    if args.list:
        runs = manager.list_runs(algorithm=args.algorithm, status=args.status)
        if not runs:
            print("No runs found.")
            return

        print(f"\n{'='*80}")
        print(f"TRAINING RUNS ({len(runs)} total)")
        print(f"{'='*80}")
        print(f"{'Run ID':<40} {'Algorithm':<12} {'Status':<12} {'Seed':<6}")
        print("-" * 80)

        for run in runs:
            print(f"{run['run_id'][:40]:<40} {run.get('algorithm', 'N/A'):<12} "
                  f"{run.get('status', 'N/A'):<12} {run.get('seed', 'N/A'):<6}")

        print("=" * 80)

    elif args.show:
        run = manager.get_run(args.show)
        if run:
            run_data = run.metadata.to_dict()
            run_data["path"] = str(run.run_dir)
            print_run_summary(run_data)
        else:
            print(f"Run not found: {args.show}")

    elif args.compare:
        runs = []
        for run_id in args.compare:
            run = manager.get_run(run_id)
            if run:
                run_data = run.metadata.to_dict()
                run_data["run_id"] = run_id
                runs.append(run_data)
            else:
                print(f"Warning: Run not found: {run_id}")

        print_comparison_table(runs)

    elif args.best:
        best = manager.get_best_run(args.best, metric=args.metric)
        if best:
            print(f"\nBest run for {args.best} (by {args.metric}):")
            print_run_summary(best)
        else:
            print(f"No completed runs found for algorithm: {args.best}")

    elif args.leaderboard:
        print_leaderboard(manager, metric=args.metric)

    elif args.export:
        manager.export_results(args.export, algorithm=args.algorithm, format=args.format)
        print(f"Results exported to: {args.export}")

    elif args.load:
        policy_state = manager.load_policy(args.load)
        if policy_state is None:
            print(f"Could not load policy for run: {args.load}")
            return

        run = manager.get_run(args.load)
        if run is None:
            print(f"Run metadata not found: {args.load}")
            return

        print(f"Loaded policy from: {args.load}")
        print(f"Algorithm: {run.metadata.algorithm}")

        if args.evaluate:
            evaluate_loaded_policy(
                policy_state,
                run.metadata.algorithm,
                run.metadata.hyperparameters,
                num_games=args.eval_games,
                cuda_device=args.cuda_device,
            )

    elif args.cleanup:
        print("Cleaning up failed/abandoned runs...")
        manager.cleanup_failed_runs(delete=False)
        print("Cleanup complete. Use --cleanup with caution to delete files.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
