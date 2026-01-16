#!/usr/bin/env python3
"""
Train PSRO (Policy Space Response Oracles) on CUDA bargaining game.

Usage:
    python scripts/train_psro.py --psro-iterations 20 --num-envs 4096

With Weights & Biases:
    python scripts/train_psro.py --wandb --wandb-project causal-bargain
"""

import argparse

from rl_training.algorithms.psro import PSROConfig, PSROTrainer
from rl_training.utils.logging import create_logger


def main():
    parser = argparse.ArgumentParser(description="Train PSRO on bargaining game")

    # Training parameters
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--psro-iterations", type=int, default=20, help="PSRO iterations")
    parser.add_argument("--br-training-steps", type=int, default=100000, help="Steps per BR")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/psro")
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N PSRO iterations")

    # PSRO-specific
    parser.add_argument("--max-policies", type=int, default=20, help="Max policies per player")
    parser.add_argument("--num-eval-games", type=int, default=1000, help="Games for payoff estimation")
    parser.add_argument("--nash-solver", type=str, default="replicator",
                       choices=["replicator", "fictitious_play"], help="Nash solver")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="causal-bargain", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=["psro", "bargain"], help="W&B tags")

    args = parser.parse_args()

    print("=" * 60)
    print("PSRO TRAINING - CUDA BARGAINING GAME")
    print("=" * 60)
    print(f"Environments: {args.num_envs:,}")
    print(f"PSRO iterations: {args.psro_iterations}")
    print(f"BR training steps: {args.br_training_steps:,}")
    print(f"Max policies: {args.max_policies}")
    print(f"Nash solver: {args.nash_solver}")
    print(f"CUDA device: {args.cuda_device}")
    print(f"Weights & Biases: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    config = PSROConfig(
        num_envs=args.num_envs,
        total_timesteps=args.psro_iterations,  # PSRO iterations
        seed=args.seed,
        cuda_device=args.cuda_device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        max_policies=args.max_policies,
        num_eval_games=args.num_eval_games,
        nash_solver=args.nash_solver,
        br_training_steps=args.br_training_steps,
        psro_iterations=args.psro_iterations,
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

    trainer = PSROTrainer(config, logger=logger)

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
