"""
FCP (Fictitious Co-Play) trainer for bargaining game.

Trains agents by playing against historical policy snapshots.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from copy import deepcopy

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.fcp.config import FCPConfig
from rl_training.envs.bargain_wrapper import BargainEnvWrapper
from rl_training.networks.bargain_mlp import BargainMLP
from rl_training.networks.transformer_policy import TransformerPolicyNetwork


class PolicyPool:
    """
    Pool of historical policy snapshots.

    Maintains a fixed-size pool with FIFO replacement.
    """

    def __init__(self, max_size: int, device: str):
        self.max_size = max_size
        self.device = device
        self.policies: List[nn.Module] = []
        self.timestamps: List[int] = []

    def add(self, policy: nn.Module, timestamp: int):
        """Add a policy snapshot to the pool."""
        # Create a frozen copy
        policy_copy = deepcopy(policy)
        policy_copy.eval()
        for param in policy_copy.parameters():
            param.requires_grad = False

        if len(self.policies) >= self.max_size:
            # Remove oldest
            self.policies.pop(0)
            self.timestamps.pop(0)

        self.policies.append(policy_copy)
        self.timestamps.append(timestamp)

    def sample(
        self,
        prioritized: bool = False,
        uniform_prob: float = 0.5,
        recent_window: int = 3,
    ) -> nn.Module:
        """
        Sample a policy from the pool.

        Args:
            prioritized: Whether to prioritize recent policies
            uniform_prob: Probability of uniform sampling
            recent_window: Window for recent policies

        Returns:
            Sampled policy
        """
        if len(self.policies) == 0:
            raise ValueError("Policy pool is empty")

        if len(self.policies) == 1:
            return self.policies[0]

        if prioritized and np.random.random() > uniform_prob:
            # Sample from recent policies
            recent_start = max(0, len(self.policies) - recent_window)
            idx = np.random.randint(recent_start, len(self.policies))
        else:
            # Uniform sampling
            idx = np.random.randint(len(self.policies))

        return self.policies[idx]

    def sample_batch(
        self,
        batch_size: int,
        prioritized: bool = False,
        uniform_prob: float = 0.5,
        recent_window: int = 3,
    ) -> List[int]:
        """Sample policy indices for a batch of environments."""
        if len(self.policies) == 0:
            raise ValueError("Policy pool is empty")

        indices = []
        for _ in range(batch_size):
            if len(self.policies) == 1:
                indices.append(0)
            elif prioritized and np.random.random() > uniform_prob:
                recent_start = max(0, len(self.policies) - recent_window)
                indices.append(np.random.randint(recent_start, len(self.policies)))
            else:
                indices.append(np.random.randint(len(self.policies)))

        return indices

    def get_actions_batched(
        self,
        policy_indices: List[int],
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get actions from multiple policies for batched observations."""
        device = obs.device
        batch_size = obs.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Group by policy index
        for idx in set(policy_indices):
            mask = torch.tensor(
                [i for i, pi in enumerate(policy_indices) if pi == idx],
                device=device
            )
            if len(mask) > 0:
                with torch.no_grad():
                    policy_actions, _, _ = self.policies[idx].get_action(
                        obs[mask], action_mask[mask]
                    )
                actions[mask] = policy_actions

        return actions

    def __len__(self):
        return len(self.policies)


@register_algorithm("fcp")
class FCPTrainer(BaseTrainer[FCPConfig]):
    """
    Fictitious Co-Play trainer for bargaining game.

    Features:
    - Maintains population of historical policy snapshots
    - Trains against sampled partner policies
    - Encourages robustness through diversity
    """

    def __init__(
        self,
        config: FCPConfig,
        env_fn: Optional[Callable] = None,
        logger: Optional[Any] = None,
    ):
        super().__init__(config, env_fn, logger)

        self.env: Optional[BargainEnvWrapper] = None
        self.policies: List[nn.Module] = []  # Current training policies
        self.optimizers: List[torch.optim.Optimizer] = []
        self.policy_pools: List[PolicyPool] = []  # Historical snapshots

        self.episode_rewards: List[deque] = [deque(maxlen=1000) for _ in range(2)]
        self._total_steps = 0

    def _setup(self) -> None:
        """Initialize FCP components."""
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

        # Create policies for each player
        for _ in range(2):
            policy = self._create_policy()
            self.policies.append(policy)
            self.optimizers.append(
                torch.optim.Adam(policy.parameters(), lr=self.config.lr)
            )
            self.policy_pools.append(
                PolicyPool(self.config.population_size, self.device)
            )

        # Add initial policies to pools
        for p in range(2):
            self.policy_pools[p].add(self.policies[p], 0)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _create_policy(self) -> nn.Module:
        """Create a new policy network."""
        if self.config.network_type == "transformer":
            policy = TransformerPolicyNetwork(
                obs_dim=BargainEnvWrapper.OBS_DIM,
                num_actions=BargainEnvWrapper.NUM_ACTIONS,
            )
        else:
            policy = BargainMLP(
                obs_dim=BargainEnvWrapper.OBS_DIM,
                num_actions=BargainEnvWrapper.NUM_ACTIONS,
                hidden_dims=self.config.hidden_dims,
            )
        return policy.to(self.device)

    def _train_step(self) -> Dict[str, Any]:
        """Execute one FCP training step."""
        metrics = {}

        # Train each player
        for player in range(2):
            player_metrics = self._train_player(player)
            for k, v in player_metrics.items():
                metrics[f"p{player}_{k}"] = v

        # Update step counter
        self._total_steps += self.config.rollout_steps * self.config.num_envs

        # Maybe add snapshots to pools
        if self._total_steps % self.config.snapshot_interval == 0:
            for p in range(2):
                self.policy_pools[p].add(self.policies[p], self._total_steps)

        metrics["timesteps"] = self.config.rollout_steps * self.config.num_envs
        metrics["pool_size_p0"] = len(self.policy_pools[0])
        metrics["pool_size_p1"] = len(self.policy_pools[1])

        for p in range(2):
            if len(self.episode_rewards[p]) > 0:
                metrics[f"avg_reward_p{p}"] = np.mean(list(self.episode_rewards[p]))

        return metrics

    def _train_player(self, player: int) -> Dict[str, float]:
        """Train one player against opponent's policy pool."""
        opponent = 1 - player

        # Sample opponent policies for each env
        opponent_policy_indices = self.policy_pools[opponent].sample_batch(
            self.config.num_envs,
            prioritized=self.config.prioritized_sampling,
            uniform_prob=self.config.uniform_prob,
            recent_window=self.config.recent_window,
        )

        # Collect rollout
        rollout = self._collect_rollout(player, opponent, opponent_policy_indices)

        # PPO update
        update_metrics = self._ppo_update(player, rollout)

        return update_metrics

    def _collect_rollout(
        self,
        player: int,
        opponent: int,
        opponent_policy_indices: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Collect rollout for one player."""
        obs_list = []
        action_list = []
        log_prob_list = []
        value_list = []
        reward_list = []
        done_list = []
        mask_list = []

        obs, info = self.env.reset()
        action_mask = info['action_mask']

        for _ in range(self.config.rollout_steps):
            current_player = self.env.get_current_player()

            actions = torch.zeros(
                self.config.num_envs,
                dtype=torch.long,
                device=self.device
            )

            # Player's actions (training)
            player_mask = current_player == player
            if player_mask.any():
                player_obs = obs[player_mask]
                player_action_mask = action_mask[player_mask]

                with torch.no_grad():
                    player_actions, log_probs, values = self.policies[player].get_action(
                        player_obs, player_action_mask
                    )

                actions[player_mask] = player_actions

                obs_list.append(player_obs)
                action_list.append(player_actions)
                log_prob_list.append(log_probs)
                value_list.append(values)
                mask_list.append(player_action_mask)

            # Opponent's actions (from pool)
            opp_mask = current_player == opponent
            if opp_mask.any():
                opp_indices = [
                    opponent_policy_indices[i]
                    for i in range(self.config.num_envs)
                    if opp_mask[i]
                ]
                opp_actions = self.policy_pools[opponent].get_actions_batched(
                    opp_indices,
                    obs[opp_mask],
                    action_mask[opp_mask],
                )
                actions[opp_mask] = opp_actions

            # Step environment
            result = self.env.step(actions.int())
            next_obs = result.observations
            rewards = result.rewards
            dones = result.terminated

            # Store rewards for training player
            if player_mask.any():
                player_rewards = rewards[player_mask, player]
                reward_list.append(player_rewards)
                done_list.append(dones[player_mask].float())

            # Track episode rewards
            if dones.any():
                for p in range(2):
                    done_rewards = rewards[dones, p].cpu().numpy()
                    for r in done_rewards:
                        self.episode_rewards[p].append(r)
                self.env.auto_reset()

            obs = next_obs
            action_mask = result.info['action_mask']

        # Stack tensors
        if len(obs_list) == 0:
            return {
                "obs": torch.zeros(1, BargainEnvWrapper.OBS_DIM, device=self.device),
                "actions": torch.zeros(1, dtype=torch.long, device=self.device),
                "log_probs_old": torch.zeros(1, device=self.device),
                "returns": torch.zeros(1, device=self.device),
                "advantages": torch.zeros(1, device=self.device),
                "masks": torch.zeros(1, BargainEnvWrapper.NUM_ACTIONS, device=self.device),
            }

        obs_batch = torch.cat(obs_list)
        action_batch = torch.cat(action_list)
        log_prob_batch = torch.cat(log_prob_list)
        value_batch = torch.cat(value_list)
        reward_batch = torch.cat(reward_list)
        done_batch = torch.cat(done_list)
        mask_batch = torch.cat(mask_list)

        # Compute advantages
        advantages = reward_batch - value_batch
        returns = reward_batch

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs": obs_batch,
            "actions": action_batch,
            "log_probs_old": log_prob_batch,
            "returns": returns,
            "advantages": advantages,
            "masks": mask_batch,
        }

    def _ppo_update(
        self,
        player: int,
        rollout: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Perform PPO update for one player."""
        obs = rollout["obs"]
        actions = rollout["actions"]
        log_probs_old = rollout["log_probs_old"]
        returns = rollout["returns"]
        advantages = rollout["advantages"]
        masks = rollout["masks"]

        batch_size = len(obs)
        if batch_size < self.config.minibatch_size:
            return {"loss": 0.0}

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, batch_size)
                mb_idx = indices[start:end]

                log_probs, values, entropy = self.policies[player].evaluate_actions(
                    obs[mb_idx], actions[mb_idx], masks[mb_idx]
                )

                # PPO loss
                ratio = torch.exp(log_probs - log_probs_old[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_eps,
                    1 + self.config.clip_eps
                ) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns[mb_idx])
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizers[player].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policies[player].parameters(),
                    self.config.max_grad_norm
                )
                self.optimizers[player].step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        if num_updates == 0:
            return {"loss": 0.0}

        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "policies": [p.state_dict() for p in self.policies],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "policy_pools": [
                {
                    "policies": [p.state_dict() for p in pool.policies],
                    "timestamps": pool.timestamps,
                }
                for pool in self.policy_pools
            ],
            "total_steps": self._total_steps,
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        for policy, state in zip(self.policies, data["policies"]):
            policy.load_state_dict(state)
        for opt, state in zip(self.optimizers, data["optimizers"]):
            opt.load_state_dict(state)

        for pool, pool_data in zip(self.policy_pools, data["policy_pools"]):
            pool.policies = []
            pool.timestamps = pool_data["timestamps"]
            for state in pool_data["policies"]:
                policy = self._create_policy()
                policy.load_state_dict(state)
                policy.eval()
                pool.policies.append(policy)

        self._total_steps = data["total_steps"]

    def _cleanup(self) -> None:
        if self.env:
            self.env.close()

    def get_policy(self, player: int = 0) -> nn.Module:
        """Get current training policy for a player."""
        return self.policies[player]

    def get_pool(self, player: int = 0) -> PolicyPool:
        """Get policy pool for a player."""
        return self.policy_pools[player]
