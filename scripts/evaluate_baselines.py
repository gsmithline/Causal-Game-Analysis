#!/usr/bin/env python3
"""
Evaluate baseline policies against each other.

Usage:
    python scripts/evaluate_baselines.py --num-games 10000
"""

import argparse
import torch
import numpy as np
from typing import Dict, Any

from rl_training.envs.bargain_wrapper import BargainEnvWrapper
from rl_training.baselines import (
    RandomPolicy,
    GreedyPolicy,
    AlwaysWalkPolicy,
    FairSplitPolicy,
)


def evaluate_matchup(
    policy1,
    policy2,
    num_games: int = 10000,
    cuda_device: int = 0,
) -> Dict[str, float]:
    """
    Evaluate two policies against each other.

    Args:
        policy1: Policy for player 1
        policy2: Policy for player 2
        num_games: Number of games to play
        cuda_device: CUDA device ID

    Returns:
        Statistics dictionary
    """
    env = BargainEnvWrapper(
        num_envs=num_games,
        self_play=True,
        device=cuda_device,
        seed=42,
    )

    obs, info = env.reset()
    action_mask = info['action_mask']

    total_rewards_p1 = []
    total_rewards_p2 = []
    accept_count = 0
    walk_count = 0
    games_completed = 0

    max_steps = 10

    for _ in range(max_steps):
        current_player = env.get_current_player()

        # Get actions for each player
        actions = torch.zeros(num_games, dtype=torch.long, device=f"cuda:{cuda_device}")

        p1_mask = current_player == 0
        p2_mask = current_player == 1

        if p1_mask.any():
            p1_actions = policy1.get_action(obs[p1_mask], action_mask[p1_mask])
            actions[p1_mask] = p1_actions

        if p2_mask.any():
            p2_actions = policy2.get_action(obs[p2_mask], action_mask[p2_mask])
            actions[p2_mask] = p2_actions

        # Track action types
        accept_count += (actions == BargainEnvWrapper.ACTION_ACCEPT).sum().item()
        walk_count += (actions == BargainEnvWrapper.ACTION_WALK).sum().item()

        result = env.step(actions.int())
        dones = result.terminated

        if dones.any():
            done_indices = dones.nonzero().squeeze(-1)
            for idx in done_indices:
                i = idx.item()
                total_rewards_p1.append(result.rewards[i, 0].item())
                total_rewards_p2.append(result.rewards[i, 1].item())
                games_completed += 1

            env.auto_reset()

        if games_completed >= num_games:
            break

        obs = result.observations
        action_mask = result.info['action_mask']

    env.close()

    # Calculate statistics
    avg_reward_p1 = np.mean(total_rewards_p1) if total_rewards_p1 else 0
    avg_reward_p2 = np.mean(total_rewards_p2) if total_rewards_p2 else 0
    total_actions = accept_count + walk_count + (games_completed * 2)  # Approximate

    return {
        "p1_avg_reward": avg_reward_p1,
        "p2_avg_reward": avg_reward_p2,
        "total_reward": avg_reward_p1 + avg_reward_p2,
        "games_completed": games_completed,
        "accept_rate": accept_count / max(1, total_actions),
        "walk_rate": walk_count / max(1, total_actions),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline policies")
    parser.add_argument("--num-games", type=int, default=10000, help="Games per matchup")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID")
    args = parser.parse_args()

    print("=" * 70)
    print("BASELINE POLICY EVALUATION - CUDA BARGAINING GAME")
    print("=" * 70)
    print(f"Games per matchup: {args.num_games:,}")
    print()

    # Create policies
    policies = {
        "random": RandomPolicy(),
        "greedy": GreedyPolicy(),
        "always_walk": AlwaysWalkPolicy(),
        "fair_split": FairSplitPolicy(),
    }

    # Round-robin evaluation
    results = {}

    for name1, policy1 in policies.items():
        for name2, policy2 in policies.items():
            matchup_name = f"{name1} vs {name2}"
            print(f"Evaluating: {matchup_name}...", end=" ")

            stats = evaluate_matchup(
                policy1, policy2,
                num_games=args.num_games,
                cuda_device=args.cuda_device,
            )

            results[matchup_name] = stats
            print(f"P1: {stats['p1_avg_reward']:.3f}, P2: {stats['p2_avg_reward']:.3f}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Matchup':<30} {'P1 Reward':>12} {'P2 Reward':>12} {'Total':>10}")
    print("-" * 70)

    for matchup, stats in results.items():
        print(f"{matchup:<30} {stats['p1_avg_reward']:>12.3f} {stats['p2_avg_reward']:>12.3f} {stats['total_reward']:>10.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
