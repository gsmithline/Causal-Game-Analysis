"""
MAPPO (Multi-Agent PPO) trainer for bargaining game.

Implements centralized training with decentralized execution.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.mappo.config import MAPPOConfig
from rl_training.envs.bargain_wrapper import BargainEnvWrapper


class MAPPOActor(nn.Module):
    """Actor network for MAPPO."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: tuple = (256, 256),
    ):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            ])
            prev_dim = hidden

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_dim, num_actions)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return action logits."""
        features = self.backbone(obs)
        logits = self.policy_head(features)

        # Mask invalid actions
        logits = logits.masked_fill(action_mask == 0, -1e9)
        return logits

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple:
        """Sample action and return log probability."""
        logits = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple:
        """Evaluate actions for PPO update."""
        logits = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """Centralized critic for MAPPO."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (256, 256),
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            ])
            prev_dim = hidden

        self.backbone = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return state value."""
        features = self.backbone(global_state)
        value = self.value_head(features)
        return value.squeeze(-1)


class RunningMeanStd:
    """Running mean and standard deviation for value normalization."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count


@register_algorithm("mappo")
class MAPPOTrainer(BaseTrainer[MAPPOConfig]):
    """
    MAPPO trainer for bargaining game.

    Features:
    - Separate actor for each player
    - Centralized critic with global state observation
    - PPO clipped objective
    - Value normalization
    """

    def __init__(
        self,
        config: MAPPOConfig,
        env_fn: Optional[Callable] = None,
        logger: Optional[Any] = None,
    ):
        super().__init__(config, env_fn, logger)

        self.env: Optional[BargainEnvWrapper] = None
        self.actors: List[MAPPOActor] = []
        self.critic: Optional[MAPPOCritic] = None
        self.actor_optimizers: List[torch.optim.Optimizer] = []
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None

        # Value normalization
        self.value_normalizer: Optional[RunningMeanStd] = None

        # Tracking
        self.episode_rewards: List[deque] = [deque(maxlen=1000) for _ in range(2)]

    def _setup(self) -> None:
        """Initialize MAPPO components."""
        self.device = f"cuda:{self.config.cuda_device}"

        # Create environment
        if self.env_fn:
            self.env = self.env_fn()
        else:
            self.env = BargainEnvWrapper(
                num_envs=self.config.num_envs,
                self_play=True,
                device=self.config.cuda_device,
                seed=self.config.seed,
            )

        # Create actors
        for _ in range(2):
            actor = MAPPOActor(
                obs_dim=BargainEnvWrapper.OBS_DIM,
                num_actions=BargainEnvWrapper.NUM_ACTIONS,
                hidden_dims=self.config.actor_hidden_dims,
            ).to(self.device)
            self.actors.append(actor)
            self.actor_optimizers.append(
                torch.optim.Adam(actor.parameters(), lr=self.config.lr)
            )

        # Optionally share actor weights
        if self.config.share_actor:
            self.actors[1].load_state_dict(self.actors[0].state_dict())

        # Create centralized critic
        if self.config.use_centralized_critic:
            critic_input = self.config.critic_input_dim
        else:
            critic_input = BargainEnvWrapper.OBS_DIM

        self.critic = MAPPOCritic(
            input_dim=critic_input,
            hidden_dims=self.config.critic_hidden_dims,
        ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.lr
        )

        # Value normalizer
        if self.config.use_value_norm:
            self.value_normalizer = RunningMeanStd()

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _train_step(self) -> Dict[str, Any]:
        """Execute one MAPPO training step."""
        # Collect rollout
        rollout = self._collect_rollout()

        # PPO update
        update_metrics = self._ppo_update(rollout)

        # Compute metrics
        metrics = {
            "timesteps": self.config.rollout_steps * self.config.num_envs,
            **update_metrics,
        }

        for p in range(2):
            if len(self.episode_rewards[p]) > 0:
                metrics[f"avg_reward_p{p}"] = np.mean(list(self.episode_rewards[p]))

        return metrics

    def _collect_rollout(self) -> Dict[str, Any]:
        """Collect rollout data."""
        # Storage per player
        obs_storage = [[] for _ in range(2)]
        action_storage = [[] for _ in range(2)]
        log_prob_storage = [[] for _ in range(2)]
        reward_storage = [[] for _ in range(2)]
        done_storage = [[] for _ in range(2)]
        mask_storage = [[] for _ in range(2)]
        global_state_storage = [[] for _ in range(2)]

        obs, info = self.env.reset()
        action_mask = info['action_mask']

        # Track observations for both players (for centralized critic)
        # In bargaining, we can construct global state from shared info
        global_obs = torch.zeros(
            self.config.num_envs,
            self.config.critic_input_dim,
            device=self.device
        )

        for step in range(self.config.rollout_steps):
            current_player = self.env.get_current_player()

            # Update global state (concatenate relevant info)
            # For bargaining: [current_obs, opponent_info]
            global_obs[:, :BargainEnvWrapper.OBS_DIM] = obs

            # Get actions for each player
            actions = torch.zeros(
                self.config.num_envs,
                dtype=torch.long,
                device=self.device
            )

            for p in range(2):
                player_mask = current_player == p
                if not player_mask.any():
                    continue

                player_obs = obs[player_mask]
                player_action_mask = action_mask[player_mask]
                player_global = global_obs[player_mask]

                with torch.no_grad():
                    player_actions, log_probs = self.actors[p].get_action(
                        player_obs, player_action_mask
                    )

                actions[player_mask] = player_actions

                # Store for this player
                obs_storage[p].append(player_obs)
                action_storage[p].append(player_actions)
                log_prob_storage[p].append(log_probs)
                mask_storage[p].append(player_action_mask)
                global_state_storage[p].append(player_global)

            # Step environment
            result = self.env.step(actions.int())
            next_obs = result.observations
            rewards = result.rewards
            dones = result.terminated

            # Store rewards per player
            for p in range(2):
                player_mask = current_player == p
                if player_mask.any():
                    reward_storage[p].append(rewards[player_mask, p])
                    done_storage[p].append(dones[player_mask].float())

            # Track episode rewards
            if dones.any():
                for p in range(2):
                    done_rewards = rewards[dones, p].cpu().numpy()
                    for r in done_rewards:
                        self.episode_rewards[p].append(r)
                self.env.auto_reset()

            obs = next_obs
            action_mask = result.info['action_mask']

        # Process into batches per player
        rollout_data = {}
        for p in range(2):
            if len(obs_storage[p]) == 0:
                continue

            obs_batch = torch.cat(obs_storage[p])
            action_batch = torch.cat(action_storage[p])
            log_prob_batch = torch.cat(log_prob_storage[p])
            reward_batch = torch.cat(reward_storage[p])
            done_batch = torch.cat(done_storage[p])
            mask_batch = torch.cat(mask_storage[p])
            global_batch = torch.cat(global_state_storage[p])

            # Get value estimates
            with torch.no_grad():
                values = self.critic(global_batch)

            # Compute advantages (simplified - per-step)
            if self.config.use_value_norm and self.value_normalizer is not None:
                self.value_normalizer.update(values.cpu().numpy())
                values_normalized = values

            advantages = reward_batch - values
            returns = reward_batch

            # Normalize advantages
            if self.config.use_advantage_norm and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            rollout_data[p] = {
                "obs": obs_batch,
                "actions": action_batch,
                "log_probs_old": log_prob_batch,
                "returns": returns,
                "advantages": advantages,
                "masks": mask_batch,
                "global_state": global_batch,
            }

        return rollout_data

    def _ppo_update(self, rollout: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Perform PPO update for all agents."""
        metrics = {}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            for p in range(2):
                if p not in rollout:
                    continue

                data = rollout[p]
                batch_size = len(data["obs"])

                if batch_size < self.config.minibatch_size:
                    continue

                indices = torch.randperm(batch_size, device=self.device)

                for start in range(0, batch_size, self.config.minibatch_size):
                    end = min(start + self.config.minibatch_size, batch_size)
                    mb_idx = indices[start:end]

                    mb_obs = data["obs"][mb_idx]
                    mb_actions = data["actions"][mb_idx]
                    mb_log_probs_old = data["log_probs_old"][mb_idx]
                    mb_returns = data["returns"][mb_idx]
                    mb_advantages = data["advantages"][mb_idx]
                    mb_masks = data["masks"][mb_idx]
                    mb_global = data["global_state"][mb_idx]

                    # Actor update
                    log_probs, entropy = self.actors[p].evaluate_actions(
                        mb_obs, mb_actions, mb_masks
                    )

                    ratio = torch.exp(log_probs - mb_log_probs_old)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.config.clip_eps,
                        1 + self.config.clip_eps
                    ) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy.mean()

                    actor_loss = policy_loss + self.config.entropy_coef * entropy_loss

                    self.actor_optimizers[p].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[p].parameters(),
                        self.config.max_grad_norm
                    )
                    self.actor_optimizers[p].step()

                    # Critic update
                    values = self.critic(mb_global)
                    value_loss = F.mse_loss(values, mb_returns)

                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.config.max_grad_norm
                    )
                    self.critic_optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_updates += 1

            # Sync actors if sharing
            if self.config.share_actor:
                self.actors[1].load_state_dict(self.actors[0].state_dict())

        if num_updates > 0:
            metrics["policy_loss"] = total_policy_loss / num_updates
            metrics["value_loss"] = total_value_loss / num_updates
            metrics["entropy"] = total_entropy / num_updates

        return metrics

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "actors": [actor.state_dict() for actor in self.actors],
            "critic": self.critic.state_dict(),
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        for actor, state in zip(self.actors, data["actors"]):
            actor.load_state_dict(state)
        self.critic.load_state_dict(data["critic"])
        for opt, state in zip(self.actor_optimizers, data["actor_optimizers"]):
            opt.load_state_dict(state)
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])

    def _cleanup(self) -> None:
        if self.env:
            self.env.close()

    def get_actor(self, player: int = 0) -> nn.Module:
        """Get actor network for a player."""
        return self.actors[player]
