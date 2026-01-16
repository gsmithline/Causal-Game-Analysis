#!/usr/bin/env python3
"""
Train NFSP on CUDA bargaining game.

Usage:
    python scripts/train_nfsp_bargain.py --num-envs 4096 --total-timesteps 5000000

With Weights & Biases:
    python scripts/train_nfsp_bargain.py --wandb --wandb-project causal-bargain
"""

import argparse

from rl_training.algorithms.nfsp_bargain import NFSPBargainConfig, NFSPBargainTrainer
from rl_training.utils.logging import create_logger


def main():
    parser = argparse.ArgumentParser(description="Train NFSP on bargaining game")

    # Training parameters
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--eta", type=float, default=0.1, help="Anticipatory parameter")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/nfsp_bargain")
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N updates")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="causal-bargain", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=["nfsp", "bargain"], help="W&B tags")

    args = parser.parse_args()

    print("=" * 60)
    print("NFSP TRAINING - CUDA BARGAINING GAME")
    print("=" * 60)
    print(f"Environments: {args.num_envs:,}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Eta (BR probability): {args.eta}")
    print(f"CUDA device: {args.cuda_device}")
    print(f"Weights & Biases: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    config = NFSPBargainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        cuda_device=args.cuda_device,
        eta=args.eta,
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

    trainer = NFSPBargainTrainer(config, logger=logger)

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
