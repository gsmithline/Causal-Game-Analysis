"""
NFSP trainer adapted for CUDA bargaining game.

Uses vectorized environments with per-environment mode tracking.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.nfsp_bargain.config import NFSPBargainConfig
from rl_training.envs.bargain_wrapper import BargainEnvWrapper


class VectorizedNFSPAgent:
    """NFSP agent adapted for vectorized GPU environments."""

    def __init__(
        self,
        player_id: int,
        obs_dim: int,
        num_actions: int,
        config: NFSPBargainConfig,
        device: int,
    ):
        self.player_id = player_id
        self.config = config
        self.device = f"cuda:{device}"
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Best response network (Q-network)
        self.br_network = self._make_network().to(self.device)
        self.br_target = self._make_network().to(self.device)
        self.br_target.load_state_dict(self.br_network.state_dict())
        self.br_optimizer = torch.optim.SGD(self.br_network.parameters(), lr=config.br_lr)

        # Average policy network
        self.avg_network = self._make_network().to(self.device)
        self.avg_optimizer = torch.optim.SGD(self.avg_network.parameters(), lr=config.avg_lr)

        # GPU buffers (circular)
        self.br_buffer_obs = torch.zeros(config.br_buffer_size, obs_dim, device=self.device)
        self.br_buffer_actions = torch.zeros(config.br_buffer_size, dtype=torch.long, device=self.device)
        self.br_buffer_rewards = torch.zeros(config.br_buffer_size, device=self.device)
        self.br_buffer_next_obs = torch.zeros(config.br_buffer_size, obs_dim, device=self.device)
        self.br_buffer_dones = torch.zeros(config.br_buffer_size, device=self.device)
        self.br_buffer_idx = 0
        self.br_buffer_size = 0

        # Reservoir buffer for average policy
        self.avg_buffer_obs = torch.zeros(config.avg_buffer_size, obs_dim, device=self.device)
        self.avg_buffer_actions = torch.zeros(config.avg_buffer_size, dtype=torch.long, device=self.device)
        self.avg_buffer_count = 0

        self._epsilon = config.epsilon_start
        self._step = 0

    def _make_network(self) -> nn.Module:
        layers = []
        prev_dim = self.obs_dim
        for hidden in self.config.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden), nn.ReLU()])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, self.num_actions))
        return nn.Sequential(*layers)

    def get_action_batched(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        modes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get actions for batch of observations.

        Args:
            obs: [batch, obs_dim]
            action_mask: [batch, num_actions]
            modes: [batch] bool - True for BR, False for average

        Returns:
            actions: [batch] int
        """
        batch_size = obs.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Split by mode
        br_mask = modes
        avg_mask = ~modes

        # Best response actions (epsilon-greedy)
        if br_mask.any():
            br_obs = obs[br_mask]
            br_action_mask = action_mask[br_mask]

            with torch.no_grad():
                q_values = self.br_network(br_obs)

            # Mask invalid actions
            q_values = q_values.masked_fill(br_action_mask == 0, -1e9)

            # Epsilon-greedy
            random_mask = torch.rand(br_obs.shape[0], device=self.device) < self._epsilon
            greedy_actions = q_values.argmax(dim=-1)

            # Sample random valid actions
            random_actions = self._sample_valid_actions(br_action_mask)

            br_actions = torch.where(random_mask, random_actions, greedy_actions)
            actions[br_mask] = br_actions

        # Average policy actions
        if avg_mask.any():
            avg_obs = obs[avg_mask]
            avg_action_mask = action_mask[avg_mask]

            with torch.no_grad():
                logits = self.avg_network(avg_obs)

            # Mask and sample
            logits = logits.masked_fill(avg_action_mask == 0, -1e9)
            probs = F.softmax(logits, dim=-1)
            avg_actions = torch.multinomial(probs, 1).squeeze(-1)
            actions[avg_mask] = avg_actions

        return actions

    def _sample_valid_actions(self, action_mask: torch.Tensor) -> torch.Tensor:
        """Sample uniformly from valid actions."""
        batch_size = action_mask.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            valid_indices = action_mask[i].nonzero().squeeze(-1)
            if len(valid_indices) > 0:
                idx = torch.randint(len(valid_indices), (1,), device=self.device)
                actions[i] = valid_indices[idx]

        return actions

    def store_br_transition(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Store transitions in BR buffer (only where mask is True)."""
        valid_indices = mask.nonzero().squeeze(-1)
        if len(valid_indices) == 0:
            return

        n = len(valid_indices)
        start_idx = self.br_buffer_idx
        end_idx = start_idx + n

        if end_idx <= self.config.br_buffer_size:
            indices = torch.arange(start_idx, end_idx, device=self.device)
        else:
            indices = torch.arange(start_idx, end_idx, device=self.device) % self.config.br_buffer_size

        self.br_buffer_obs[indices] = obs[valid_indices]
        self.br_buffer_actions[indices] = actions[valid_indices]
        self.br_buffer_rewards[indices] = rewards[valid_indices]
        self.br_buffer_next_obs[indices] = next_obs[valid_indices]
        self.br_buffer_dones[indices] = dones[valid_indices].float()

        self.br_buffer_idx = end_idx % self.config.br_buffer_size
        self.br_buffer_size = min(self.br_buffer_size + n, self.config.br_buffer_size)

    def store_avg_data(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Store observation-action pairs in reservoir buffer."""
        valid_indices = mask.nonzero().squeeze(-1)
        if len(valid_indices) == 0:
            return

        valid_obs = obs[valid_indices]
        valid_actions = actions[valid_indices]

        for i in range(len(valid_obs)):
            self.avg_buffer_count += 1
            if self.avg_buffer_count <= self.config.avg_buffer_size:
                idx = self.avg_buffer_count - 1
            else:
                idx = torch.randint(self.avg_buffer_count, (1,), device=self.device).item()
                if idx >= self.config.avg_buffer_size:
                    continue
            self.avg_buffer_obs[idx] = valid_obs[i]
            self.avg_buffer_actions[idx] = valid_actions[i]

    def update_br(self) -> Optional[float]:
        """Update BR network (DQN update)."""
        if self.br_buffer_size < self.config.br_batch_size:
            return None

        indices = torch.randint(self.br_buffer_size, (self.config.br_batch_size,), device=self.device)

        obs = self.br_buffer_obs[indices]
        actions = self.br_buffer_actions[indices]
        rewards = self.br_buffer_rewards[indices]
        next_obs = self.br_buffer_next_obs[indices]
        dones = self.br_buffer_dones[indices]

        q_values = self.br_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.br_target(next_obs).max(dim=1)[0]
            target_q = rewards + self.config.discount * next_q * (1 - dones)

        loss = F.mse_loss(q_values, target_q)

        self.br_optimizer.zero_grad()
        loss.backward()
        self.br_optimizer.step()

        return loss.item()

    def update_avg(self) -> Optional[float]:
        """Update average policy network."""
        buffer_size = min(self.avg_buffer_count, self.config.avg_buffer_size)
        if buffer_size < self.config.avg_batch_size:
            return None

        indices = torch.randint(buffer_size, (self.config.avg_batch_size,), device=self.device)

        obs = self.avg_buffer_obs[indices]
        actions = self.avg_buffer_actions[indices]

        logits = self.avg_network(obs)
        loss = F.cross_entropy(logits, actions)

        self.avg_optimizer.zero_grad()
        loss.backward()
        self.avg_optimizer.step()

        return loss.item()

    def update_target(self):
        self.br_target.load_state_dict(self.br_network.state_dict())

    def decay_epsilon(self):
        self._step += 1
        progress = min(1.0, self._step / self.config.epsilon_decay_steps)
        self._epsilon = self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "br_network": self.br_network.state_dict(),
            "br_target": self.br_target.state_dict(),
            "br_optimizer": self.br_optimizer.state_dict(),
            "avg_network": self.avg_network.state_dict(),
            "avg_optimizer": self.avg_optimizer.state_dict(),
            "epsilon": self._epsilon,
            "step": self._step,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.br_network.load_state_dict(state["br_network"])
        self.br_target.load_state_dict(state["br_target"])
        self.br_optimizer.load_state_dict(state["br_optimizer"])
        self.avg_network.load_state_dict(state["avg_network"])
        self.avg_optimizer.load_state_dict(state["avg_optimizer"])
        self._epsilon = state["epsilon"]
        self._step = state["step"]


@register_algorithm("nfsp_bargain")
class NFSPBargainTrainer(BaseTrainer[NFSPBargainConfig]):
    """
    NFSP trainer adapted for vectorized CUDA bargaining environment.

    Key adaptations:
    - Per-environment mode tracking (BR vs average)
    - GPU-based replay buffers
    - Batched network updates
    """

    def __init__(
        self,
        config: NFSPBargainConfig,
        env_fn: Optional[Callable] = None,
        logger: Optional[Any] = None,
    ):
        super().__init__(config, env_fn, logger)

        self.env: Optional[BargainEnvWrapper] = None
        self.agents: List[VectorizedNFSPAgent] = []
        self._total_steps = 0

    def _setup(self) -> None:
        if self.env_fn:
            self.env = self.env_fn()
        else:
            self.env = BargainEnvWrapper(
                num_envs=self.config.num_envs,
                self_play=True,
                device=self.config.cuda_device,
                seed=self.config.seed,
            )

        # Create agents for each player
        self.agents = [
            VectorizedNFSPAgent(
                player_id=p,
                obs_dim=BargainEnvWrapper.OBS_DIM,
                num_actions=BargainEnvWrapper.NUM_ACTIONS,
                config=self.config,
                device=self.config.cuda_device,
            )
            for p in range(2)
        ]

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _train_step(self) -> Dict[str, Any]:
        """Execute one training step (play games, update networks)."""
        device = f"cuda:{self.config.cuda_device}"

        obs, info = self.env.reset()
        action_mask = info['action_mask']

        step_count = 0
        metrics = {}

        # Play until all games complete at least once
        while step_count < 64:  # Fixed rollout length
            current_player = self.env.get_current_player()

            # Sample modes (BR vs average) per environment
            modes = torch.rand(self.config.num_envs, device=device) < self.config.eta

            # Get actions based on current player
            actions = torch.zeros(self.config.num_envs, dtype=torch.long, device=device)

            for p in range(2):
                player_mask = current_player == p
                if player_mask.any():
                    player_actions = self.agents[p].get_action_batched(
                        obs[player_mask],
                        action_mask[player_mask],
                        modes[player_mask],
                    )
                    actions[player_mask] = player_actions

            # Step environment
            result = self.env.step(actions.int())
            next_obs = result.observations
            rewards = result.rewards
            dones = result.terminated

            # Store data for each player
            for p in range(2):
                player_mask = current_player == p
                br_mode_mask = player_mask & modes

                # Store BR transitions
                player_rewards = rewards[:, p]
                self.agents[p].store_br_transition(
                    obs, actions, player_rewards, next_obs, dones, player_mask
                )

                # Store avg data (only for BR mode)
                self.agents[p].store_avg_data(obs, actions, br_mode_mask)

            # Auto-reset done games
            if dones.any():
                self.env.auto_reset()

            obs = next_obs
            action_mask = result.info['action_mask']
            step_count += 1
            self._total_steps += self.config.num_envs

        # Update networks
        for p, agent in enumerate(self.agents):
            br_loss = agent.update_br()
            if br_loss is not None:
                metrics[f"br_loss_p{p}"] = br_loss

            if self._total_steps % self.config.avg_train_freq == 0:
                avg_loss = agent.update_avg()
                if avg_loss is not None:
                    metrics[f"avg_loss_p{p}"] = avg_loss

            if self._total_steps % self.config.br_target_update_freq == 0:
                agent.update_target()

            agent.decay_epsilon()

        metrics["timesteps"] = step_count * self.config.num_envs
        metrics["epsilon"] = self.agents[0]._epsilon

        return metrics

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "agents": [agent.state_dict() for agent in self.agents],
            "total_steps": self._total_steps,
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        for agent, agent_data in zip(self.agents, data["agents"]):
            agent.load_state_dict(agent_data)
        self._total_steps = data["total_steps"]

    def _cleanup(self) -> None:
        if self.env:
            self.env.close()

    def get_average_policy(self, player: int = 0) -> nn.Module:
        return self.agents[player].avg_network
