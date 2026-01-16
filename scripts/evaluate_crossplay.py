#!/usr/bin/env python3
"""
Cross-play evaluation script for bargaining game.

Evaluates all pairwise combinations of policies (LLM, RL, heuristic)
and generates a payoff matrix for meta-game analysis.

Usage:
    # Evaluate LLMs against each other
    python scripts/evaluate_crossplay.py \
        --policies gpt-5.2-pro gpt-4o o3 \
        --num-games 100

    # Evaluate LLMs against trained RL agents
    python scripts/evaluate_crossplay.py \
        --policies gpt-5.2-pro \
        --rl-checkpoints checkpoints/ppo/best.pt checkpoints/nfsp/best.pt \
        --num-games 100

    # Include heuristic baselines
    python scripts/evaluate_crossplay.py \
        --policies gpt-5.2-pro o3 \
        --baselines random greedy fair_split \
        --num-games 100

    # Full evaluation matrix
    python scripts/evaluate_crossplay.py \
        --policies gpt-5.2-pro gpt-5.2-thinking o3 gpt-4o \
        --baselines random greedy fair_split always_walk \
        --rl-checkpoints checkpoints/ppo/best.pt \
        --num-games 500 \
        --output results/crossplay_matrix.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_policy(
    policy_spec: str,
    device: str = "cuda:0",
) -> Tuple[Any, str]:
    """
    Load a policy from specification.

    Args:
        policy_spec: Policy specification:
            - LLM model name (e.g., "gpt-5.2-pro", "o3")
            - "random", "greedy", "fair_split", "always_walk" for baselines
            - Path to checkpoint file for RL policies
        device: Device for RL policies

    Returns:
        (policy, name): Policy object and display name
    """
    # Check if it's a checkpoint path
    if policy_spec.endswith(".pt") or policy_spec.endswith(".pth"):
        # Load RL checkpoint
        from rl_training.networks.bargain_mlp import BargainMLP

        checkpoint = torch.load(policy_spec, map_location=device)

        # Determine network architecture
        if "config" in checkpoint:
            config = checkpoint["config"]
            hidden_dims = config.get("hidden_dims", (256, 256))
        else:
            hidden_dims = (256, 256)

        policy = BargainMLP(
            obs_dim=92,
            num_actions=82,
            hidden_dims=hidden_dims,
        ).to(device)

        # Load weights
        if "policy_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["policy_state_dict"])
        elif "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            policy.load_state_dict(checkpoint)

        policy.eval()
        name = Path(policy_spec).stem
        return policy, f"rl_{name}"

    # Check if it's a baseline
    baselines = {
        "random": ("rl_training.baselines", "RandomPolicy"),
        "greedy": ("rl_training.baselines", "GreedyPolicy"),
        "fair_split": ("rl_training.baselines", "FairSplitPolicy"),
        "always_walk": ("rl_training.baselines", "AlwaysWalkPolicy"),
    }

    if policy_spec.lower() in baselines:
        module_name, class_name = baselines[policy_spec.lower()]
        module = __import__(module_name, fromlist=[class_name])
        policy_class = getattr(module, class_name)
        return policy_class(), policy_spec.lower()

    # Assume it's an LLM model name
    from rl_training.llm_policies import OpenAIPolicy, OPENAI_MODELS

    if policy_spec in OPENAI_MODELS or policy_spec.startswith("gpt-") or policy_spec.startswith("o"):
        policy = OpenAIPolicy(model_name=policy_spec, verbose=False)
        return policy, f"llm_{policy_spec}"

    raise ValueError(f"Unknown policy specification: {policy_spec}")


def run_single_game(
    env,
    policy_p0,
    policy_p1,
    device: str = "cuda:0",
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Run a single game between two policies.

    Args:
        env: Bargaining environment (single instance)
        policy_p0: Player 0's policy
        policy_p1: Player 1's policy
        device: Device for tensors

    Returns:
        (reward_p0, reward_p1, info): Final rewards and game info
    """
    obs, info = env.reset()
    action_mask = info['action_mask']
    done = False
    game_log = []

    # Reset LLM conversation histories if applicable
    if hasattr(policy_p0, 'reset_conversation'):
        policy_p0.reset_conversation()
    if hasattr(policy_p1, 'reset_conversation'):
        policy_p1.reset_conversation()

    while not done:
        current_player = env.get_current_player()[0].item()

        # Get action from appropriate policy
        if current_player == 0:
            action = policy_p0.get_action(obs, action_mask)
        else:
            action = policy_p1.get_action(obs, action_mask)

        # Handle different action formats
        if isinstance(action, torch.Tensor):
            action_int = action[0].item() if action.dim() > 0 else action.item()
        else:
            action_int = int(action)

        # Log action
        game_log.append({
            "player": current_player,
            "action": action_int,
        })

        # Step environment
        result = env.step(torch.tensor([action_int], dtype=torch.int32, device=device))

        obs = result.observations
        action_mask = result.info['action_mask']
        done = result.terminated[0].item() or result.truncated[0].item()

        if done:
            rewards = result.rewards[0].cpu().numpy()
            return rewards[0], rewards[1], {"log": game_log}

    return 0.0, 0.0, {"log": game_log}


def evaluate_matchup(
    policy_p0,
    policy_p1,
    num_games: int = 100,
    device: str = "cuda:0",
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a matchup between two policies over multiple games.

    Args:
        policy_p0: Player 0's policy
        policy_p1: Player 1's policy
        num_games: Number of games to play
        device: CUDA device
        show_progress: Show progress bar

    Returns:
        Dictionary with evaluation results
    """
    from rl_training.envs.bargain_wrapper import BargainEnvWrapper

    # Create single-instance environment for LLM compatibility
    env = BargainEnvWrapper(num_envs=1, self_play=True, device=int(device.split(":")[-1]))

    rewards_p0 = []
    rewards_p1 = []
    game_results = []

    iterator = range(num_games)
    if show_progress:
        iterator = tqdm(iterator, desc="Games", leave=False)

    for _ in iterator:
        r0, r1, info = run_single_game(env, policy_p0, policy_p1, device)
        rewards_p0.append(r0)
        rewards_p1.append(r1)
        game_results.append({
            "reward_p0": r0,
            "reward_p1": r1,
            "welfare": r0 + r1,
        })

    env.close()

    return {
        "mean_reward_p0": np.mean(rewards_p0),
        "mean_reward_p1": np.mean(rewards_p1),
        "std_reward_p0": np.std(rewards_p0),
        "std_reward_p1": np.std(rewards_p1),
        "mean_welfare": np.mean([r["welfare"] for r in game_results]),
        "num_games": num_games,
    }


def build_crossplay_matrix(
    policy_specs: List[str],
    num_games: int = 100,
    device: str = "cuda:0",
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Build full cross-play payoff matrix.

    Args:
        policy_specs: List of policy specifications
        num_games: Games per matchup
        device: CUDA device

    Returns:
        (payoff_matrix, policy_names, detailed_results)
        payoff_matrix: [n, n, 2] array where [i, j, :] = (pi reward, pj reward)
    """
    n = len(policy_specs)
    payoff_matrix = np.zeros((n, n, 2))
    policy_names = []
    detailed_results = {}

    # Load all policies
    print("Loading policies...")
    policies = []
    for spec in tqdm(policy_specs):
        policy, name = load_policy(spec, device)
        policies.append(policy)
        policy_names.append(name)

    # Evaluate all matchups
    print(f"\nEvaluating {n * n} matchups ({num_games} games each)...")
    total_matchups = n * n
    pbar = tqdm(total=total_matchups, desc="Matchups")

    for i in range(n):
        for j in range(n):
            matchup_key = f"{policy_names[i]}_vs_{policy_names[j]}"

            result = evaluate_matchup(
                policies[i],
                policies[j],
                num_games=num_games,
                device=device,
                show_progress=False,
            )

            payoff_matrix[i, j, 0] = result["mean_reward_p0"]
            payoff_matrix[i, j, 1] = result["mean_reward_p1"]
            detailed_results[matchup_key] = result

            pbar.update(1)
            pbar.set_postfix({
                "last": f"{policy_names[i][:8]} vs {policy_names[j][:8]}",
                "welfare": f"{result['mean_welfare']:.2f}",
            })

    pbar.close()

    return payoff_matrix, policy_names, detailed_results


def print_results_table(
    payoff_matrix: np.ndarray,
    policy_names: List[str],
):
    """Print formatted results table."""
    n = len(policy_names)

    # Truncate names for display
    display_names = [name[:12] for name in policy_names]
    max_name_len = max(len(n) for n in display_names)

    print("\n" + "=" * 60)
    print("CROSS-PLAY RESULTS (Player 0 rewards)")
    print("=" * 60)

    # Header
    header = " " * (max_name_len + 2)
    for name in display_names:
        header += f"{name:>12}"
    print(header)
    print("-" * len(header))

    # Rows
    for i in range(n):
        row = f"{display_names[i]:<{max_name_len}}  "
        for j in range(n):
            row += f"{payoff_matrix[i, j, 0]:>12.2f}"
        print(row)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for i, name in enumerate(policy_names):
        avg_as_p0 = payoff_matrix[i, :, 0].mean()
        avg_as_p1 = payoff_matrix[:, i, 1].mean()
        avg_overall = (avg_as_p0 + avg_as_p1) / 2
        print(f"{name[:20]:<20} Avg as P0: {avg_as_p0:.2f}, Avg as P1: {avg_as_p1:.2f}, Overall: {avg_overall:.2f}")

    # Welfare matrix
    print("\n" + "=" * 60)
    print("WELFARE (sum of both player rewards)")
    print("=" * 60)

    welfare_matrix = payoff_matrix[:, :, 0] + payoff_matrix[:, :, 1]
    header = " " * (max_name_len + 2)
    for name in display_names:
        header += f"{name:>12}"
    print(header)
    print("-" * len(header))

    for i in range(n):
        row = f"{display_names[i]:<{max_name_len}}  "
        for j in range(n):
            row += f"{welfare_matrix[i, j]:>12.2f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Cross-play evaluation for bargaining game")

    # Policy selection
    parser.add_argument("--policies", type=str, nargs="+", default=[],
                       help="LLM model names (e.g., gpt-5.2-pro, o3)")
    parser.add_argument("--baselines", type=str, nargs="+", default=[],
                       help="Baseline policies (random, greedy, fair_split, always_walk)")
    parser.add_argument("--rl-checkpoints", type=str, nargs="+", default=[],
                       help="Paths to RL policy checkpoints")

    # Evaluation settings
    parser.add_argument("--num-games", type=int, default=100,
                       help="Number of games per matchup")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="CUDA device")

    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--output-csv", type=str, default=None,
                       help="Output CSV file path for payoff matrix")

    args = parser.parse_args()

    # Combine all policy specs
    all_specs = args.policies + args.baselines + args.rl_checkpoints

    if not all_specs:
        print("Error: No policies specified. Use --policies, --baselines, or --rl-checkpoints")
        sys.exit(1)

    print("=" * 60)
    print("CROSS-PLAY EVALUATION")
    print("=" * 60)
    print(f"Policies: {len(all_specs)}")
    print(f"Games per matchup: {args.num_games}")
    print(f"Total games: {len(all_specs) ** 2 * args.num_games:,}")
    print()

    # Run evaluation
    payoff_matrix, policy_names, detailed_results = build_crossplay_matrix(
        all_specs,
        num_games=args.num_games,
        device=args.device,
    )

    # Print results
    print_results_table(payoff_matrix, policy_names)

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "policy_names": policy_names,
            "payoff_matrix": payoff_matrix.tolist(),
            "num_games": args.num_games,
            "detailed_results": detailed_results,
        }

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    if args.output_csv:
        import pandas as pd
        # Create DataFrames for each player's perspective
        df_p0 = pd.DataFrame(payoff_matrix[:, :, 0], index=policy_names, columns=policy_names)
        df_p0.to_csv(args.output_csv.replace(".csv", "_p0.csv"))

        df_p1 = pd.DataFrame(payoff_matrix[:, :, 1], index=policy_names, columns=policy_names)
        df_p1.to_csv(args.output_csv.replace(".csv", "_p1.csv"))

        welfare = payoff_matrix[:, :, 0] + payoff_matrix[:, :, 1]
        df_welfare = pd.DataFrame(welfare, index=policy_names, columns=policy_names)
        df_welfare.to_csv(args.output_csv.replace(".csv", "_welfare.csv"))

        print(f"CSV files saved to: {args.output_csv.replace('.csv', '_*.csv')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
