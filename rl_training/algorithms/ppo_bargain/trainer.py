"""
PPO trainer for CUDA bargaining game self-play.

Adapted from SGRD train_selfplay.py to integrate with rl_training framework.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.ppo_bargain.config import PPOBargainConfig
from rl_training.envs.bargain_wrapper import BargainEnvWrapper
from rl_training.networks.transformer_policy import TransformerPolicyNetwork
from rl_training.networks.bargain_mlp import BargainMLP


@register_algorithm("ppo_bargain")
class PPOBargainTrainer(BaseTrainer[PPOBargainConfig]):
    """
    PPO self-play trainer for CUDA bargaining game.

    Features:
        - Self-play with shared policy for both players
        - CUDA-accelerated vectorized environment
        - GAE advantage estimation
        - Clipped PPO objective
        - Support for Transformer and MLP policy networks
    """

    def __init__(
        self,
        config: PPOBargainConfig,
        env_fn: Optional[Callable] = None,
        policy_fn: Optional[Callable] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            config: PPO configuration
            env_fn: Optional environment factory function
            policy_fn: Optional policy factory function
            logger: Optional logger instance
        """
        super().__init__(config, env_fn, logger)
        self.policy_fn = policy_fn

        # Will be initialized in _setup
        self.env: Optional[BargainEnvWrapper] = None
        self.policy: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Tracking
        self.episode_rewards_p1: deque = deque(maxlen=1000)
        self.episode_rewards_p2: deque = deque(maxlen=1000)

    def _setup(self) -> None:
        """Initialize PPO components."""
        # Create environment
        if self.env_fn is not None:
            self.env = self.env_fn()
        else:
            self.env = BargainEnvWrapper(
                num_envs=self.config.num_envs,
                self_play=True,
                device=self.config.cuda_device,
                seed=self.config.seed,
            )

        # Create policy network
        if self.policy_fn is not None:
            self.policy = self.policy_fn()
        else:
            if self.config.network_type == "transformer":
                self.policy = TransformerPolicyNetwork(
                    obs_dim=BargainEnvWrapper.OBS_DIM,
                    num_actions=BargainEnvWrapper.NUM_ACTIONS,
                    d_model=self.config.d_model,
                    nhead=self.config.nhead,
                    num_layers=self.config.num_layers,
                )
            else:
                self.policy = BargainMLP(
                    obs_dim=BargainEnvWrapper.OBS_DIM,
                    num_actions=BargainEnvWrapper.NUM_ACTIONS,
                    hidden_dims=self.config.hidden_dims,
                )

        # Move to device
        self.policy = self.policy.cuda(self.config.cuda_device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.lr,
        )

        # Set seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _train_step(self) -> Dict[str, Any]:
        """
        Execute one PPO training step.

        This consists of:
        1. Collecting a rollout
        2. Computing advantages with GAE
        3. PPO update with multiple epochs

        Returns:
            Metrics dictionary
        """
        # Collect rollout
        rollout_data = self._collect_rollout()

        # PPO update
        update_metrics = self._ppo_update(rollout_data)

        # Compute metrics
        timesteps = self.config.rollout_steps * self.config.num_envs

        metrics = {
            "timesteps": timesteps,
            **update_metrics,
        }

        # Add reward tracking
        if len(self.episode_rewards_p1) > 0:
            metrics["avg_reward_p1"] = np.mean(list(self.episode_rewards_p1))
            metrics["avg_reward_p2"] = np.mean(list(self.episode_rewards_p2))

        return metrics

    def _collect_rollout(self) -> Dict[str, torch.Tensor]:
        """
        Collect a rollout from the environment.

        Returns:
            Dictionary containing rollout data
        """
        obs_list = []
        action_list = []
        log_prob_list = []
        value_list = []
        reward_list = []
        done_list = []
        mask_list = []
        player_list = []

        # Reset environment
        obs, info = self.env.reset()
        action_mask = info['action_mask']

        for _ in range(self.config.rollout_steps):
            current_player = self.env.get_current_player()

            # Store current state
            obs_list.append(obs)
            mask_list.append(action_mask)
            player_list.append(current_player)

            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs, action_mask)

            action_list.append(action)
            log_prob_list.append(log_prob)
            value_list.append(value)

            # Step environment
            result = self.env.step(action)
            obs = result.observations
            rewards = result.rewards
            dones = result.terminated
            action_mask = result.info['action_mask']

            # Compute rewards for the player who just acted
            player_rewards = torch.zeros(
                self.config.num_envs,
                device=f"cuda:{self.config.cuda_device}"
            )
            for i in range(self.config.num_envs):
                p = player_list[-1][i].item()
                player_rewards[i] = rewards[i, p]

            reward_list.append(player_rewards)
            done_list.append(dones.float())

            # Track episode rewards when games end
            if dones.any():
                done_indices = dones.nonzero().squeeze(-1)
                for idx in done_indices:
                    i = idx.item()
                    self.episode_rewards_p1.append(rewards[i, 0].item())
                    self.episode_rewards_p2.append(rewards[i, 1].item())
                self.env.auto_reset()

        # Get final value for bootstrapping
        with torch.no_grad():
            _, final_value = self.policy(obs, action_mask)

        # Stack tensors
        obs_batch = torch.stack(obs_list)
        action_batch = torch.stack(action_list)
        log_prob_batch = torch.stack(log_prob_list)
        value_batch = torch.stack(value_list)
        reward_batch = torch.stack(reward_list)
        done_batch = torch.stack(done_list)
        mask_batch = torch.stack(mask_list)

        # Compute advantages with GAE
        advantages = torch.zeros_like(reward_batch)
        returns = torch.zeros_like(reward_batch)
        gae = 0

        for t in reversed(range(self.config.rollout_steps)):
            if t == self.config.rollout_steps - 1:
                next_value = final_value
            else:
                next_value = value_batch[t + 1]

            delta = (
                reward_batch[t]
                + self.config.gamma * next_value * (1 - done_batch[t])
                - value_batch[t]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - done_batch[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + value_batch[t]

        # Flatten batch
        batch_size = self.config.rollout_steps * self.config.num_envs
        obs_flat = obs_batch.view(batch_size, -1)
        action_flat = action_batch.view(batch_size)
        log_prob_flat = log_prob_batch.view(batch_size)
        returns_flat = returns.view(batch_size)
        advantages_flat = advantages.view(batch_size)
        mask_flat = mask_batch.view(batch_size, -1)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        return {
            "obs": obs_flat,
            "actions": action_flat,
            "log_probs_old": log_prob_flat,
            "returns": returns_flat,
            "advantages": advantages_flat,
            "masks": mask_flat,
        }

    def _ppo_update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            rollout_data: Dictionary containing rollout tensors

        Returns:
            Update metrics
        """
        obs = rollout_data["obs"]
        actions = rollout_data["actions"]
        log_probs_old = rollout_data["log_probs_old"]
        returns = rollout_data["returns"]
        advantages = rollout_data["advantages"]
        masks = rollout_data["masks"]

        batch_size = obs.shape[0]
        device = obs.device

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            # Random permutation for minibatches
            indices = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_masks = masks[mb_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    mb_obs, mb_actions, mb_masks
                )

                # Policy loss (clipped PPO)
                ratio = torch.exp(log_probs - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_eps,
                    1 + self.config.clip_eps
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data."""
        return {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_rewards_p1": list(self.episode_rewards_p1),
            "episode_rewards_p2": list(self.episode_rewards_p2),
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        """Load checkpoint data."""
        self.policy.load_state_dict(data["policy_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.episode_rewards_p1 = deque(data.get("episode_rewards_p1", []), maxlen=1000)
        self.episode_rewards_p2 = deque(data.get("episode_rewards_p2", []), maxlen=1000)

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            self.env.close()

    def get_policy(self) -> nn.Module:
        """Get the trained policy network."""
        return self.policy
