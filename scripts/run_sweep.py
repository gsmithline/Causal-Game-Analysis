#!/usr/bin/env python3
"""
Unified sweep runner for algorithm comparison.

Used by sweep_all_algorithms.yaml to compare different algorithms
with consistent evaluation.

Usage:
    python scripts/run_sweep.py --algorithm ppo --seed 42
    python scripts/run_sweep.py --algorithm psro --seed 42 --wandb

Results are automatically saved to the results/ directory with:
- Trained policy weights
- Training metrics history
- Final evaluation results
- Full configuration
"""

import argparse
import time
import torch
import numpy as np
from typing import Dict, Any

# Import all trainers and configs
from rl_training.algorithms.ppo_bargain import PPOBargainConfig, PPOBargainTrainer
from rl_training.algorithms.nfsp_bargain import NFSPBargainConfig, NFSPBargainTrainer
from rl_training.algorithms.sampled_cfr import SampledCFRConfig, SampledCFRTrainer
from rl_training.algorithms.psro import PSROConfig, PSROTrainer
from rl_training.algorithms.mappo import MAPPOConfig, MAPPOTrainer
from rl_training.algorithms.fcp import FCPConfig, FCPTrainer
from rl_training.utils.logging import create_logger
from rl_training.utils.results_manager import ResultsManager
from rl_training.envs.bargain_wrapper import BargainEnvWrapper


# Default hyperparameters for each algorithm (tuned defaults)
ALGORITHM_DEFAULTS = {
    "ppo": {
        "config_class": PPOBargainConfig,
        "trainer_class": PPOBargainTrainer,
        "total_timesteps": 5_000_000,
        "num_envs": 4096,
        "lr": 3e-4,
        "rollout_steps": 64,
        "network_type": "mlp",
    },
    "nfsp": {
        "config_class": NFSPBargainConfig,
        "trainer_class": NFSPBargainTrainer,
        "total_timesteps": 2_000_000,
        "num_envs": 4096,
        "eta": 0.1,
    },
    "sampled_cfr": {
        "config_class": SampledCFRConfig,
        "trainer_class": SampledCFRTrainer,
        "total_timesteps": 1000,  # iterations
        "num_envs": 4096,
        "lr": 1e-3,
    },
    "psro": {
        "config_class": PSROConfig,
        "trainer_class": PSROTrainer,
        "total_timesteps": 15,  # PSRO iterations
        "num_envs": 2048,
        "lr": 3e-4,
        "br_training_steps": 100000,
        "max_policies": 20,
        "psro_iterations": 15,
    },
    "mappo": {
        "config_class": MAPPOConfig,
        "trainer_class": MAPPOTrainer,
        "total_timesteps": 5_000_000,
        "num_envs": 4096,
        "lr": 3e-4,
        "rollout_steps": 64,
        "share_actor": False,
    },
    "fcp": {
        "config_class": FCPConfig,
        "trainer_class": FCPTrainer,
        "total_timesteps": 5_000_000,
        "num_envs": 4096,
        "lr": 3e-4,
        "population_size": 10,
        "snapshot_interval": 10000,
    },
}


def evaluate_final_performance(
    trainer,
    algorithm: str,
    num_eval_games: int = 10000,
    cuda_device: int = 0,
) -> Dict[str, float]:
    """
    Evaluate final trained policy performance.

    Returns standardized metrics for comparison.
    """
    device = f"cuda:{cuda_device}"

    # Create fresh eval environment
    eval_env = BargainEnvWrapper(
        num_envs=min(num_eval_games, 4096),
        self_play=True,
        device=cuda_device,
        seed=99999,  # Different seed for eval
    )

    total_rewards = np.zeros(2)
    games_completed = 0

    obs, info = eval_env.reset()
    action_mask = info['action_mask']

    # Get policy based on algorithm type
    if algorithm == "psro":
        # Use Nash mixture
        pop_p0, nash_p0 = trainer.get_nash_policy(0)
        pop_p1, nash_p1 = trainer.get_nash_policy(1)

        while games_completed < num_eval_games:
            current_player = eval_env.get_current_player()
            actions = torch.zeros(eval_env.num_envs, dtype=torch.long, device=device)

            p0_mask = current_player == 0
            p1_mask = current_player == 1

            if p0_mask.any():
                actions[p0_mask] = pop_p0.sample_from_mixture(
                    nash_p0, obs[p0_mask], action_mask[p0_mask]
                )
            if p1_mask.any():
                actions[p1_mask] = pop_p1.sample_from_mixture(
                    nash_p1, obs[p1_mask], action_mask[p1_mask]
                )

            result = eval_env.step(actions.int())
            dones = result.terminated

            if dones.any():
                rewards = result.rewards[dones].cpu().numpy()
                total_rewards += rewards.sum(axis=0)
                games_completed += dones.sum().item()
                eval_env.auto_reset()

            obs = result.observations
            action_mask = result.info['action_mask']

    elif algorithm == "fcp":
        # Use current policies
        policies = [trainer.get_policy(p) for p in range(2)]

        while games_completed < num_eval_games:
            current_player = eval_env.get_current_player()
            actions = torch.zeros(eval_env.num_envs, dtype=torch.long, device=device)

            for p in range(2):
                p_mask = current_player == p
                if p_mask.any():
                    with torch.no_grad():
                        p_actions, _, _ = policies[p].get_action(
                            obs[p_mask], action_mask[p_mask]
                        )
                    actions[p_mask] = p_actions

            result = eval_env.step(actions.int())
            dones = result.terminated

            if dones.any():
                rewards = result.rewards[dones].cpu().numpy()
                total_rewards += rewards.sum(axis=0)
                games_completed += dones.sum().item()
                eval_env.auto_reset()

            obs = result.observations
            action_mask = result.info['action_mask']

    elif algorithm == "mappo":
        actors = [trainer.get_actor(p) for p in range(2)]

        while games_completed < num_eval_games:
            current_player = eval_env.get_current_player()
            actions = torch.zeros(eval_env.num_envs, dtype=torch.long, device=device)

            for p in range(2):
                p_mask = current_player == p
                if p_mask.any():
                    with torch.no_grad():
                        p_actions, _ = actors[p].get_action(
                            obs[p_mask], action_mask[p_mask]
                        )
                    actions[p_mask] = p_actions

            result = eval_env.step(actions.int())
            dones = result.terminated

            if dones.any():
                rewards = result.rewards[dones].cpu().numpy()
                total_rewards += rewards.sum(axis=0)
                games_completed += dones.sum().item()
                eval_env.auto_reset()

            obs = result.observations
            action_mask = result.info['action_mask']

    else:
        # PPO, NFSP, Sampled CFR - single shared policy
        if algorithm == "nfsp":
            policy = trainer.get_average_policy(0)
        elif algorithm == "sampled_cfr":
            policy = trainer.get_strategy_network(0)
        else:
            policy = trainer.get_policy()

        while games_completed < num_eval_games:
            with torch.no_grad():
                if hasattr(policy, 'get_action'):
                    actions, _, _ = policy.get_action(obs, action_mask)
                else:
                    # For networks without get_action method
                    logits = policy(obs, action_mask)
                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)

            result = eval_env.step(actions.int())
            dones = result.terminated

            if dones.any():
                rewards = result.rewards[dones].cpu().numpy()
                total_rewards += rewards.sum(axis=0)
                games_completed += dones.sum().item()
                eval_env.auto_reset()

            obs = result.observations
            action_mask = result.info['action_mask']

    eval_env.close()

    avg_rewards = total_rewards / num_eval_games
    return {
        "final_reward_p0": avg_rewards[0],
        "final_reward_p1": avg_rewards[1],
        "final_total_reward": avg_rewards.sum(),
        "eval_games": num_eval_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified algorithm sweep runner")

    parser.add_argument("--algorithm", type=str, required=True,
                       choices=list(ALGORITHM_DEFAULTS.keys()),
                       help="Algorithm to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device")
    parser.add_argument("--eval-games", type=int, default=10000, help="Games for final eval")

    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="causal-bargain-sweeps")
    parser.add_argument("--wandb-name", type=str, default=None)

    # Results management
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory for organized results")

    # Override defaults
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)

    args = parser.parse_args()

    # Get algorithm defaults
    algo_config = ALGORITHM_DEFAULTS[args.algorithm].copy()
    config_class = algo_config.pop("config_class")
    trainer_class = algo_config.pop("trainer_class")

    # Apply overrides
    if args.lr is not None:
        algo_config["lr"] = args.lr
    if args.num_envs is not None:
        algo_config["num_envs"] = args.num_envs
    if args.total_timesteps is not None:
        algo_config["total_timesteps"] = args.total_timesteps

    algo_config["seed"] = args.seed
    algo_config["cuda_device"] = args.cuda_device
    algo_config["checkpoint_dir"] = f"checkpoints/sweep_{args.algorithm}_{args.seed}"

    print("=" * 60)
    print(f"SWEEP RUN: {args.algorithm.upper()}")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Config: {algo_config}")
    print()

    # Create config
    config = config_class(**algo_config)

    # Initialize results manager
    results_manager = ResultsManager(args.results_dir)
    run_context = results_manager.create_run(
        algorithm=args.algorithm,
        seed=args.seed,
        config=config.to_dict(),
        environment="bargain",
        tags=["sweep", args.algorithm],
    )

    print(f"Results will be saved to: {run_context.run_dir}")
    print()

    # Create logger
    run_name = args.wandb_name or f"{args.algorithm}_seed{args.seed}"
    logger = create_logger(
        console=True,
        wandb_project=args.wandb_project if args.wandb else None,
        wandb_name=run_name,
        wandb_config={
            "algorithm": args.algorithm,
            "run_id": run_context.run_id,
            **config.to_dict(),
        },
        wandb_tags=[args.algorithm, "sweep"],
        log_interval=config.log_interval,
    )

    # Train
    trainer = trainer_class(config, logger=logger)
    start_time = time.time()

    try:
        trainer.train()

        # Save final policy
        print("\nSaving trained policy...")
        checkpoint_data = trainer._get_checkpoint_data()
        run_context.save_final_policy(checkpoint_data)

        # Final evaluation
        print("Running final evaluation...")
        eval_metrics = evaluate_final_performance(
            trainer,
            args.algorithm,
            num_eval_games=args.eval_games,
            cuda_device=args.cuda_device,
        )

        print(f"Final P0 reward: {eval_metrics['final_reward_p0']:.4f}")
        print(f"Final P1 reward: {eval_metrics['final_reward_p1']:.4f}")
        print(f"Final total reward: {eval_metrics['final_total_reward']:.4f}")

        # Save results
        run_context.set_final_metrics(eval_metrics)
        run_context.set_actual_timesteps(config.total_timesteps)
        run_context.complete()

        # Log to W&B
        if args.wandb:
            import wandb
            wandb.log(eval_metrics)

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        run_context.fail("Interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        run_context.fail(str(e))
        raise

    finally:
        logger.close()

    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    print(f"Results saved to: {run_context.run_dir}")
    print("\nSweep run complete!")


if __name__ == "__main__":
    main()
