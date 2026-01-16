#!/usr/bin/env python3
"""
Train Sampled CFR / Deep CFR on CUDA bargaining game.

Usage:
    python scripts/train_sampled_cfr.py --num-envs 4096 --iterations 1000

With Weights & Biases:
    python scripts/train_sampled_cfr.py --wandb --wandb-project causal-bargain
"""

import argparse

from rl_training.algorithms.sampled_cfr import SampledCFRConfig, SampledCFRTrainer
from rl_training.utils.logging import create_logger


def main():
    parser = argparse.ArgumentParser(description="Train Sampled CFR on bargaining game")

    # Training parameters
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--iterations", type=int, default=1000, help="CFR iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/sampled_cfr")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="causal-bargain", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=["cfr", "bargain"], help="W&B tags")

    args = parser.parse_args()

    print("=" * 60)
    print("SAMPLED CFR / DEEP CFR - CUDA BARGAINING GAME")
    print("=" * 60)
    print(f"Environments: {args.num_envs:,}")
    print(f"Iterations: {args.iterations:,}")
    print(f"Learning rate: {args.lr}")
    print(f"CUDA device: {args.cuda_device}")
    print(f"Weights & Biases: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    config = SampledCFRConfig(
        num_envs=args.num_envs,
        total_timesteps=args.iterations,  # Iterations for CFR
        seed=args.seed,
        cuda_device=args.cuda_device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
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

    trainer = SampledCFRTrainer(config, logger=logger)

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
