#!/usr/bin/env python3
"""
Unified hyperparameter sweep script.

Supports grid search, random search, and Bayesian optimization.
Can run with or without W&B.

Usage:
    # Grid search over PPO hyperparameters
    python scripts/hyperparameter_sweep.py --algorithm ppo --method grid

    # Random search with 50 trials
    python scripts/hyperparameter_sweep.py --algorithm mappo --method random --num-trials 50

    # Compare all algorithms
    python scripts/hyperparameter_sweep.py --compare-algorithms --seeds 42 123 456

    # With W&B tracking
    python scripts/hyperparameter_sweep.py --algorithm ppo --method bayes --wandb
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# Hyperparameter search spaces for each algorithm
SEARCH_SPACES = {
    "ppo": {
        "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
        "rollout_steps": {"type": "categorical", "choices": [32, 64, 128, 256]},
        "network": {"type": "categorical", "choices": ["mlp", "transformer"]},
        "num_envs": {"type": "categorical", "choices": [1024, 2048, 4096]},
        "clip_eps": {"type": "uniform", "low": 0.1, "high": 0.3},
        "entropy_coef": {"type": "log_uniform", "low": 0.001, "high": 0.1},
    },
    "nfsp": {
        "eta": {"type": "uniform", "low": 0.05, "high": 0.5},
        "num_envs": {"type": "categorical", "choices": [1024, 2048, 4096]},
    },
    "sampled_cfr": {
        "lr": {"type": "log_uniform", "low": 1e-4, "high": 1e-2},
        "iterations": {"type": "categorical", "choices": [500, 1000, 2000]},
        "num_envs": {"type": "categorical", "choices": [1024, 2048, 4096]},
    },
    "psro": {
        "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
        "br_training_steps": {"type": "categorical", "choices": [50000, 100000, 200000]},
        "nash_solver": {"type": "categorical", "choices": ["replicator", "fictitious_play"]},
        "max_policies": {"type": "categorical", "choices": [10, 20, 30]},
        "num_eval_games": {"type": "categorical", "choices": [500, 1000, 2000]},
    },
    "mappo": {
        "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
        "rollout_steps": {"type": "categorical", "choices": [32, 64, 128]},
        "share_actor": {"type": "categorical", "choices": [True, False]},
        "num_envs": {"type": "categorical", "choices": [1024, 2048, 4096]},
    },
    "fcp": {
        "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
        "population_size": {"type": "categorical", "choices": [5, 10, 20]},
        "snapshot_interval": {"type": "categorical", "choices": [5000, 10000, 20000]},
        "prioritized": {"type": "categorical", "choices": [True, False]},
        "num_envs": {"type": "categorical", "choices": [1024, 2048, 4096]},
    },
}

# Map parameter names to CLI argument names
PARAM_TO_CLI = {
    "lr": "--lr",
    "rollout_steps": "--rollout-steps",
    "network": "--network",
    "num_envs": "--num-envs",
    "clip_eps": "--clip-eps",
    "entropy_coef": "--entropy-coef",
    "eta": "--eta",
    "iterations": "--iterations",
    "br_training_steps": "--br-training-steps",
    "nash_solver": "--nash-solver",
    "max_policies": "--max-policies",
    "num_eval_games": "--num-eval-games",
    "share_actor": "--share-actor",
    "population_size": "--population-size",
    "snapshot_interval": "--snapshot-interval",
    "prioritized": "--prioritized",
}

# Algorithm script mapping
ALGORITHM_SCRIPTS = {
    "ppo": "scripts/train_ppo_bargain.py",
    "nfsp": "scripts/train_nfsp_bargain.py",
    "sampled_cfr": "scripts/train_sampled_cfr.py",
    "psro": "scripts/train_psro.py",
    "mappo": "scripts/train_mappo.py",
    "fcp": "scripts/train_fcp.py",
}


def sample_param(space: Dict[str, Any]) -> Any:
    """Sample a parameter value from its search space."""
    if space["type"] == "uniform":
        return random.uniform(space["low"], space["high"])
    elif space["type"] == "log_uniform":
        log_low = np.log(space["low"])
        log_high = np.log(space["high"])
        return np.exp(random.uniform(log_low, log_high))
    elif space["type"] == "categorical":
        return random.choice(space["choices"])
    elif space["type"] == "int":
        return random.randint(space["low"], space["high"])
    else:
        raise ValueError(f"Unknown parameter type: {space['type']}")


def generate_grid(search_space: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """Generate all combinations for grid search."""
    param_names = list(search_space.keys())
    param_values = []

    for name in param_names:
        space = search_space[name]
        if space["type"] == "categorical":
            param_values.append(space["choices"])
        elif space["type"] in ["uniform", "log_uniform"]:
            # Discretize continuous params for grid search
            if space["type"] == "log_uniform":
                values = np.logspace(
                    np.log10(space["low"]),
                    np.log10(space["high"]),
                    num=5
                ).tolist()
            else:
                values = np.linspace(space["low"], space["high"], num=5).tolist()
            param_values.append(values)
        else:
            param_values.append([space.get("default", space["low"])])

    configs = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        configs.append(config)

    return configs


def generate_random(search_space: Dict[str, Dict], num_trials: int) -> List[Dict[str, Any]]:
    """Generate random hyperparameter configurations."""
    configs = []
    for _ in range(num_trials):
        config = {}
        for name, space in search_space.items():
            config[name] = sample_param(space)
        configs.append(config)
    return configs


def build_command(
    algorithm: str,
    params: Dict[str, Any],
    seed: int,
    cuda_device: int,
    wandb: bool,
    wandb_project: str,
    total_timesteps: Optional[int] = None,
) -> List[str]:
    """Build command line for training script."""
    script = ALGORITHM_SCRIPTS[algorithm]

    cmd = [sys.executable, script]

    # Add hyperparameters
    for param, value in params.items():
        cli_arg = PARAM_TO_CLI.get(param)
        if cli_arg:
            if isinstance(value, bool):
                if value:
                    cmd.append(cli_arg)
            else:
                cmd.extend([cli_arg, str(value)])

    # Add common args
    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--cuda-device", str(cuda_device)])

    if total_timesteps:
        if algorithm == "sampled_cfr":
            cmd.extend(["--iterations", str(total_timesteps)])
        elif algorithm == "psro":
            cmd.extend(["--psro-iterations", str(total_timesteps)])
        else:
            cmd.extend(["--total-timesteps", str(total_timesteps)])

    if wandb:
        cmd.append("--wandb")
        cmd.extend(["--wandb-project", wandb_project])
        cmd.extend(["--wandb-tags", algorithm, "sweep"])

    return cmd


def run_trial(
    algorithm: str,
    params: Dict[str, Any],
    seed: int,
    cuda_device: int,
    wandb: bool,
    wandb_project: str,
    total_timesteps: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single training trial."""
    cmd = build_command(
        algorithm, params, seed, cuda_device, wandb, wandb_project, total_timesteps
    )

    print(f"\n{'='*60}")
    print(f"Trial: {algorithm} | Seed: {seed}")
    print(f"Params: {params}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    if dry_run:
        return {"status": "dry_run", "params": params}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 4,  # 4 hour timeout
        )

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "params": params,
            "stdout": result.stdout[-5000:] if result.stdout else "",  # Last 5k chars
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "params": params}
    except Exception as e:
        return {"status": "error", "error": str(e), "params": params}


def run_optuna_sweep(
    algorithm: str,
    num_trials: int,
    seed: int,
    cuda_device: int,
    wandb: bool,
    wandb_project: str,
    total_timesteps: Optional[int] = None,
):
    """Run Bayesian optimization with Optuna."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to random search.")
        return None

    search_space = SEARCH_SPACES[algorithm]

    def objective(trial):
        params = {}
        for name, space in search_space.items():
            if space["type"] == "uniform":
                params[name] = trial.suggest_float(name, space["low"], space["high"])
            elif space["type"] == "log_uniform":
                params[name] = trial.suggest_float(name, space["low"], space["high"], log=True)
            elif space["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, space["choices"])
            elif space["type"] == "int":
                params[name] = trial.suggest_int(name, space["low"], space["high"])

        result = run_trial(
            algorithm, params, seed, cuda_device, wandb, wandb_project, total_timesteps
        )

        if result["status"] != "success":
            return float("-inf")

        # Parse final reward from stdout (this is a simplified extraction)
        # In practice, you might want to read from a results file
        return 0.0  # Placeholder - would parse actual metric

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)

    return study


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep runner")

    # Sweep configuration
    parser.add_argument("--algorithm", type=str, choices=list(SEARCH_SPACES.keys()),
                       help="Algorithm to sweep")
    parser.add_argument("--method", type=str, default="random",
                       choices=["grid", "random", "bayes"],
                       help="Search method")
    parser.add_argument("--num-trials", type=int, default=20,
                       help="Number of trials for random/bayes search")

    # Multi-algorithm comparison
    parser.add_argument("--compare-algorithms", action="store_true",
                       help="Compare all algorithms with default hyperparams")
    parser.add_argument("--algorithms", type=str, nargs="+",
                       default=list(SEARCH_SPACES.keys()),
                       help="Algorithms to compare")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Seeds for comparison")

    # Training settings
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=None,
                       help="Override total timesteps")

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="causal-bargain-sweeps")

    # Output
    parser.add_argument("--output-dir", type=str, default="sweep_results")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without running")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    if args.compare_algorithms:
        # Compare all algorithms with multiple seeds
        print("=" * 60)
        print("ALGORITHM COMPARISON SWEEP")
        print("=" * 60)
        print(f"Algorithms: {args.algorithms}")
        print(f"Seeds: {args.seeds}")
        print()

        for algorithm in args.algorithms:
            for seed in args.seeds:
                result = run_trial(
                    algorithm=algorithm,
                    params={},  # Use defaults
                    seed=seed,
                    cuda_device=args.cuda_device,
                    wandb=args.wandb,
                    wandb_project=args.wandb_project,
                    total_timesteps=args.total_timesteps,
                    dry_run=args.dry_run,
                )
                result["algorithm"] = algorithm
                result["seed"] = seed
                results.append(result)

    else:
        # Single algorithm hyperparameter sweep
        if not args.algorithm:
            parser.error("--algorithm required for single-algorithm sweep")

        print("=" * 60)
        print(f"HYPERPARAMETER SWEEP: {args.algorithm.upper()}")
        print(f"Method: {args.method}")
        print("=" * 60)

        search_space = SEARCH_SPACES[args.algorithm]

        if args.method == "grid":
            configs = generate_grid(search_space)
            print(f"Grid search: {len(configs)} configurations")
        elif args.method == "random":
            configs = generate_random(search_space, args.num_trials)
            print(f"Random search: {args.num_trials} trials")
        elif args.method == "bayes":
            if OPTUNA_AVAILABLE:
                study = run_optuna_sweep(
                    args.algorithm,
                    args.num_trials,
                    seed=42,
                    cuda_device=args.cuda_device,
                    wandb=args.wandb,
                    wandb_project=args.wandb_project,
                    total_timesteps=args.total_timesteps,
                )
                if study:
                    print(f"\nBest params: {study.best_params}")
                    print(f"Best value: {study.best_value}")
                configs = []
            else:
                print("Optuna not available, falling back to random search")
                configs = generate_random(search_space, args.num_trials)

        for i, params in enumerate(configs):
            print(f"\nTrial {i+1}/{len(configs)}")
            result = run_trial(
                algorithm=args.algorithm,
                params=params,
                seed=42,
                cuda_device=args.cuda_device,
                wandb=args.wandb,
                wandb_project=args.wandb_project,
                total_timesteps=args.total_timesteps,
                dry_run=args.dry_run,
            )
            result["trial"] = i
            result["algorithm"] = args.algorithm
            results.append(result)

    # Save results
    results_file = output_dir / f"sweep_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"Results saved to: {results_file}")
    print("=" * 60)

    # Summary
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    print(f"Successful: {successful}, Failed: {failed}, Total: {len(results)}")


if __name__ == "__main__":
    main()
