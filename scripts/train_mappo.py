#!/usr/bin/env python3
"""
Train MAPPO (Multi-Agent PPO) on CUDA bargaining game.

Usage:
    python scripts/train_mappo.py --num-envs 4096 --total-timesteps 10000000

With Weights & Biases:
    python scripts/train_mappo.py --wandb --wandb-project causal-bargain
"""

import argparse

from rl_training.algorithms.mappo import MAPPOConfig, MAPPOTrainer
from rl_training.utils.logging import create_logger


def main():
    parser = argparse.ArgumentParser(description="Train MAPPO on bargaining game")

    # Training parameters
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rollout-steps", type=int, default=64, help="Steps per rollout")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mappo")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N updates")

    # MAPPO-specific
    parser.add_argument("--share-actor", action="store_true", help="Share actor weights")
    parser.add_argument("--centralized-critic", action="store_true", default=True,
                       help="Use centralized critic")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="causal-bargain", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=["mappo", "bargain"], help="W&B tags")

    args = parser.parse_args()

    print("=" * 60)
    print("MAPPO TRAINING - CUDA BARGAINING GAME")
    print("=" * 60)
    print(f"Environments: {args.num_envs:,}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Shared actor: {args.share_actor}")
    print(f"Centralized critic: {args.centralized_critic}")
    print(f"CUDA device: {args.cuda_device}")
    print(f"Weights & Biases: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    config = MAPPOConfig(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        cuda_device=args.cuda_device,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        share_actor=args.share_actor,
        use_centralized_critic=args.centralized_critic,
    )

    # Create logger
    logger = create_logger(
        console=True,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_name=args.wandb_name,
        wandb_config=config.to_dict(),
        wandb_tags=args.wandb_tags,
        log_interval=args.log_interval,
    )

    trainer = MAPPOTrainer(config, logger=logger)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint()
    finally:
        logger.close()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
