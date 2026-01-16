#!/usr/bin/env python3
"""
Evaluate trained policies with IS-MCTS search enhancement.

Usage:
    python scripts/evaluate_with_search.py --checkpoint checkpoints/ppo_bargain/best.pt --num-simulations 100
"""

import argparse
import torch
import numpy as np

from rl_training.algorithms.is_mcts import ISMCTSConfig, SearchEnhancedPolicy
from rl_training.envs.bargain_wrapper import BargainEnvWrapper
from rl_training.networks.bargain_mlp import BargainMLP
from rl_training.networks.transformer_policy import TransformerPolicyNetwork


def load_policy(checkpoint_path: str, network_type: str, device: str) -> torch.nn.Module:
    """Load a trained policy from checkpoint."""
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    else:
        policy.load_state_dict(checkpoint)

    return policy.to(device)


def evaluate_policy(
    policy,
    env: BargainEnvWrapper,
    num_games: int,
    use_search: bool,
    search_config: ISMCTSConfig = None,
    device: str = "cuda:0",
):
    """Evaluate a policy (optionally with search)."""
    if use_search and search_config:
        policy = SearchEnhancedPolicy(policy, search_config, device)

    total_rewards = [0.0, 0.0]
    games_completed = 0

    obs, info = env.reset()
    action_mask = info['action_mask']

    while games_completed < num_games:
        current_player = env.get_current_player()

        # For search, we need to evaluate one at a time
        if use_search:
            actions = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            for i in range(min(env.num_envs, num_games - games_completed)):
                with torch.no_grad():
                    action, _, _ = policy.get_action(
                        obs[i],
                        action_mask[i],
                        player=current_player[i].item(),
                        use_search=True,
                    )
                actions[i] = action
        else:
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

    avg_rewards = [r / num_games for r in total_rewards]
    return avg_rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate with IS-MCTS search")

    parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path")
    parser.add_argument("--network", type=str, default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--num-games", type=int, default=1000, help="Games to evaluate")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID")

    # Search parameters
    parser.add_argument("--num-simulations", type=int, default=100, help="MCTS simulations")
    parser.add_argument("--use-gumbel", action="store_true", help="Use Gumbel search")
    parser.add_argument("--c-puct", type=float, default=1.5, help="Exploration constant")

    args = parser.parse_args()

    device = f"cuda:{args.cuda_device}"

    print("=" * 60)
    print("POLICY EVALUATION WITH IS-MCTS SEARCH")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Network: {args.network}")
    print(f"Games: {args.num_games}")
    print(f"MCTS simulations: {args.num_simulations}")
    print(f"Gumbel search: {args.use_gumbel}")
    print()

    # Load policy
    policy = load_policy(args.checkpoint, args.network, device)

    # Create environment
    env = BargainEnvWrapper(
        num_envs=32,  # Small batch for search
        self_play=True,
        device=args.cuda_device,
        seed=42,
    )

    # Search config
    search_config = ISMCTSConfig(
        num_simulations=args.num_simulations,
        use_gumbel=args.use_gumbel,
        c_puct=args.c_puct,
    )

    # Evaluate without search
    print("Evaluating without search...")
    no_search_rewards = evaluate_policy(
        policy, env, args.num_games, use_search=False, device=device
    )
    print(f"  P0 avg reward: {no_search_rewards[0]:.4f}")
    print(f"  P1 avg reward: {no_search_rewards[1]:.4f}")
    print(f"  Total: {sum(no_search_rewards):.4f}")
    print()

    # Evaluate with search
    print(f"Evaluating with search ({args.num_simulations} simulations)...")
    search_rewards = evaluate_policy(
        policy, env, args.num_games, use_search=True,
        search_config=search_config, device=device
    )
    print(f"  P0 avg reward: {search_rewards[0]:.4f}")
    print(f"  P1 avg reward: {search_rewards[1]:.4f}")
    print(f"  Total: {sum(search_rewards):.4f}")
    print()

    # Improvement
    improvement = sum(search_rewards) - sum(no_search_rewards)
    print(f"Search improvement: {improvement:+.4f}")

    env.close()


if __name__ == "__main__":
    main()
