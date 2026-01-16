#!/usr/bin/env python3
"""
Train Ex²PSRO (Explicit Exploration PSRO) on CUDA bargaining game.

Ex²PSRO extends PSRO to find high-welfare equilibria by:
1. Creating exploration policies that imitate high-welfare behavior
2. Regularizing best response training toward the exploration policy
3. Biasing equilibrium selection toward prosocial outcomes

Based on "Explicit Exploration for High-Welfare Equilibria in
Game-Theoretic Multiagent Reinforcement Learning" (OpenReview 2025).

Usage:
    python scripts/train_ex2psro.py --psro-iterations 20 --num-envs 4096

With Weights & Biases:
    python scripts/train_ex2psro.py --wandb --wandb-project causal-bargain

With higher welfare focus:
    python scripts/train_ex2psro.py --kl-coef 0.2 --exploration-top-k 5
"""

import argparse

from rl_training.algorithms.ex2psro import Ex2PSROConfig, Ex2PSROTrainer
from rl_training.utils.logging import create_logger


def main():
    parser = argparse.ArgumentParser(description="Train Ex²PSRO on bargaining game")

    # Training parameters
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--psro-iterations", type=int, default=20, help="PSRO iterations")
    parser.add_argument("--br-training-steps", type=int, default=100000, help="Steps per BR")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ex2psro")
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N PSRO iterations")

    # PSRO-specific
    parser.add_argument("--max-policies", type=int, default=20, help="Max policies per player")
    parser.add_argument("--num-eval-games", type=int, default=1000, help="Games for payoff estimation")
    parser.add_argument("--nash-solver", type=str, default="replicator",
                       choices=["replicator", "fictitious_play"], help="Nash solver")

    # Ex²PSRO-specific: Welfare settings
    parser.add_argument("--welfare-fn", type=str, default="utilitarian",
                       choices=["utilitarian", "nash", "egalitarian"],
                       help="Welfare function for ranking policies")
    parser.add_argument("--exploration-top-k", type=int, default=3,
                       help="Top-K highest welfare policies for exploration")
    parser.add_argument("--exploration-temperature", type=float, default=1.0,
                       help="Softmax temperature for welfare-weighted sampling")
    parser.add_argument("--use-welfare-weighted-sampling", action="store_true", default=True,
                       help="Weight by welfare vs uniform over top-k")

    # Ex²PSRO-specific: KL regularization
    parser.add_argument("--kl-coef", type=float, default=0.1,
                       help="Coefficient for KL divergence regularization")
    parser.add_argument("--kl-target", type=float, default=0.01,
                       help="Target KL divergence for adaptive KL")
    parser.add_argument("--use-adaptive-kl", action="store_true", default=True,
                       help="Adapt kl_coef to target KL")
    parser.add_argument("--no-adaptive-kl", action="store_false", dest="use_adaptive_kl",
                       help="Disable adaptive KL")

    # Ex²PSRO-specific: Imitation learning
    parser.add_argument("--imitation-epochs", type=int, default=5,
                       help="Epochs to train exploration policy")
    parser.add_argument("--imitation-lr", type=float, default=1e-3,
                       help="Learning rate for imitation learning")
    parser.add_argument("--imitation-data-size", type=int, default=10000,
                       help="Trajectories to collect for imitation")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="causal-bargain", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=["ex2psro", "bargain", "welfare"],
                       help="W&B tags")

    args = parser.parse_args()

    print("=" * 60)
    print("Ex²PSRO TRAINING - CUDA BARGAINING GAME")
    print("=" * 60)
    print(f"Environments: {args.num_envs:,}")
    print(f"PSRO iterations: {args.psro_iterations}")
    print(f"BR training steps: {args.br_training_steps:,}")
    print(f"Max policies: {args.max_policies}")
    print(f"Nash solver: {args.nash_solver}")
    print()
    print("Ex²PSRO Parameters:")
    print(f"  Welfare function: {args.welfare_fn}")
    print(f"  Exploration top-k: {args.exploration_top_k}")
    print(f"  KL coefficient: {args.kl_coef}")
    print(f"  Adaptive KL: {args.use_adaptive_kl}")
    print(f"  Imitation epochs: {args.imitation_epochs}")
    print()
    print(f"CUDA device: {args.cuda_device}")
    print(f"Weights & Biases: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    config = Ex2PSROConfig(
        num_envs=args.num_envs,
        total_timesteps=args.psro_iterations,  # PSRO iterations
        seed=args.seed,
        cuda_device=args.cuda_device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        # PSRO settings
        max_policies=args.max_policies,
        num_eval_games=args.num_eval_games,
        nash_solver=args.nash_solver,
        br_training_steps=args.br_training_steps,
        psro_iterations=args.psro_iterations,
        # Ex²PSRO welfare settings
        welfare_fn=args.welfare_fn,
        exploration_top_k=args.exploration_top_k,
        exploration_temperature=args.exploration_temperature,
        use_welfare_weighted_sampling=args.use_welfare_weighted_sampling,
        # Ex²PSRO KL regularization
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        use_adaptive_kl=args.use_adaptive_kl,
        # Ex²PSRO imitation learning
        imitation_epochs=args.imitation_epochs,
        imitation_lr=args.imitation_lr,
        imitation_data_size=args.imitation_data_size,
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

    trainer = Ex2PSROTrainer(config, logger=logger)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint()
    finally:
        logger.close()

    # Print welfare improvement summary
    welfare_history = trainer.get_welfare_history()
    if len(welfare_history) > 1:
        print("\nWelfare Improvement Summary:")
        print(f"  Initial welfare: {welfare_history[0]:.4f}")
        print(f"  Final welfare: {welfare_history[-1]:.4f}")
        print(f"  Total improvement: {welfare_history[-1] - welfare_history[0]:.4f}")
        print(f"  Improvement ratio: {welfare_history[-1] / (welfare_history[0] + 1e-8):.2f}x")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
