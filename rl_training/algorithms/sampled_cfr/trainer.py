"""
Sampled CFR / Deep CFR trainer for bargaining game.

Uses neural network approximation for regrets and strategies
since the game has continuous private values.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.sampled_cfr.config import SampledCFRConfig
from rl_training.envs.bargain_wrapper import BargainEnvWrapper


class AdvantageNetwork(nn.Module):
    """Network to predict advantages for each action."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dims=(256, 256)):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden), nn.ReLU()])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


class StrategyNetwork(nn.Module):
    """Network to predict strategy (action probabilities)."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dims=(256, 256)):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden), nn.ReLU()])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, obs, action_mask=None):
        logits = self.network(obs)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        return F.softmax(logits, dim=-1)


@register_algorithm("sampled_cfr")
class SampledCFRTrainer(BaseTrainer[SampledCFRConfig]):
    """
    Sampled CFR / Deep CFR trainer for bargaining game.

    Key features:
    - Uses CUDA env for fast trajectory sampling
    - Neural networks approximate advantages and strategies
    - Outcome sampling CFR variant
    """

    def __init__(
        self,
        config: SampledCFRConfig,
        env_fn: Optional[Callable] = None,
        logger: Optional[Any] = None,
    ):
        super().__init__(config, env_fn, logger)

        self.env: Optional[BargainEnvWrapper] = None

        # Networks for each player
        self.advantage_nets = []
        self.strategy_nets = []
        self.adv_optimizers = []
        self.strat_optimizers = []

        # Memory buffers
        self.advantage_memory = []
        self.strategy_memory = []

        self._iteration = 0

    def _setup(self) -> None:
        device = f"cuda:{self.config.cuda_device}"

        if self.env_fn:
            self.env = self.env_fn()
        else:
            self.env = BargainEnvWrapper(
                num_envs=self.config.num_envs,
                self_play=True,
                device=self.config.cuda_device,
                seed=self.config.seed,
            )

        obs_dim = BargainEnvWrapper.OBS_DIM
        num_actions = BargainEnvWrapper.NUM_ACTIONS

        # Create networks for each player
        for _ in range(2):
            adv_net = AdvantageNetwork(obs_dim, num_actions, self.config.hidden_dims).to(device)
            strat_net = StrategyNetwork(obs_dim, num_actions, self.config.hidden_dims).to(device)

            self.advantage_nets.append(adv_net)
            self.strategy_nets.append(strat_net)
            self.adv_optimizers.append(torch.optim.Adam(adv_net.parameters(), lr=self.config.lr))
            self.strat_optimizers.append(torch.optim.Adam(strat_net.parameters(), lr=self.config.lr))

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _train_step(self) -> Dict[str, Any]:
        """Execute one CFR iteration."""
        device = f"cuda:{self.config.cuda_device}"
        self._iteration += 1

        # Sample trajectories and compute advantages
        trajectory_data = self._sample_trajectories()

        # Add to advantage memory
        for data in trajectory_data:
            self.advantage_memory.append(data)

        # Limit memory size
        if len(self.advantage_memory) > self.config.advantage_memory_size:
            self.advantage_memory = self.advantage_memory[-self.config.advantage_memory_size:]

        metrics = {"timesteps": 1}  # 1 iteration

        # Train advantage networks
        if len(self.advantage_memory) >= self.config.advantage_batch_size:
            for p in range(2):
                adv_loss = self._train_advantage_network(p)
                metrics[f"adv_loss_p{p}"] = adv_loss

        # Train strategy networks periodically
        if self._iteration % self.config.strategy_train_freq == 0:
            # Add current strategies to memory
            self._add_strategy_samples()

            if len(self.strategy_memory) > self.config.strategy_memory_size:
                self.strategy_memory = self.strategy_memory[-self.config.strategy_memory_size:]

            if len(self.strategy_memory) >= self.config.strategy_batch_size:
                for p in range(2):
                    strat_loss = self._train_strategy_network(p)
                    metrics[f"strat_loss_p{p}"] = strat_loss

        metrics["iteration"] = self._iteration
        metrics["adv_memory_size"] = len(self.advantage_memory)

        return metrics

    def _sample_trajectories(self):
        """Sample game trajectories and compute counterfactual values."""
        device = f"cuda:{self.config.cuda_device}"

        trajectory_data = []

        obs, info = self.env.reset()
        action_mask = info['action_mask']

        # Store trajectory info
        trajectory_obs = [obs.clone()]
        trajectory_masks = [action_mask.clone()]
        trajectory_players = [self.env.get_current_player().clone()]
        trajectory_actions = []

        # Play games using current strategy
        max_steps = 10  # Max steps per game

        for _ in range(max_steps):
            current_player = self.env.get_current_player()

            # Get actions from strategy networks
            actions = torch.zeros(self.config.num_envs, dtype=torch.long, device=device)

            for p in range(2):
                player_mask = current_player == p
                if player_mask.any():
                    with torch.no_grad():
                        probs = self.strategy_nets[p](obs[player_mask], action_mask[player_mask])
                    player_actions = torch.multinomial(probs, 1).squeeze(-1)
                    actions[player_mask] = player_actions

            trajectory_actions.append(actions.clone())

            result = self.env.step(actions.int())
            dones = result.terminated

            if dones.all():
                break

            obs = result.observations
            action_mask = result.info['action_mask']

            trajectory_obs.append(obs.clone())
            trajectory_masks.append(action_mask.clone())
            trajectory_players.append(self.env.get_current_player().clone())

            if dones.any():
                self.env.auto_reset()

        # Compute advantages (simplified - using final rewards as proxy)
        final_rewards = result.rewards

        # Create advantage training data
        for t, (t_obs, t_mask, t_player, t_action) in enumerate(
            zip(trajectory_obs[:-1], trajectory_masks[:-1], trajectory_players[:-1], trajectory_actions)
        ):
            for p in range(2):
                player_mask = t_player == p

                if player_mask.any():
                    # Compute regret targets (simplified)
                    # Full CFR would compute counterfactual values
                    rewards = final_rewards[player_mask, p]
                    advantages = torch.zeros(player_mask.sum(), BargainEnvWrapper.NUM_ACTIONS, device=device)
                    advantages[torch.arange(player_mask.sum(), device=device), t_action[player_mask]] = rewards

                    trajectory_data.append({
                        "player": p,
                        "obs": t_obs[player_mask].cpu(),
                        "mask": t_mask[player_mask].cpu(),
                        "advantages": advantages.cpu(),
                    })

        return trajectory_data

    def _train_advantage_network(self, player: int) -> float:
        """Train advantage network for a player."""
        device = f"cuda:{self.config.cuda_device}"

        # Sample batch
        player_data = [d for d in self.advantage_memory if d["player"] == player]
        if len(player_data) < self.config.advantage_batch_size:
            return 0.0

        indices = np.random.choice(len(player_data), self.config.advantage_batch_size, replace=False)

        obs_list = []
        target_list = []

        for i in indices:
            obs_list.append(player_data[i]["obs"])
            target_list.append(player_data[i]["advantages"])

        obs = torch.cat(obs_list).to(device)
        targets = torch.cat(target_list).to(device)

        # Forward and loss
        predictions = self.advantage_nets[player](obs)
        loss = F.mse_loss(predictions, targets)

        self.adv_optimizers[player].zero_grad()
        loss.backward()
        self.adv_optimizers[player].step()

        return loss.item()

    def _add_strategy_samples(self):
        """Add current strategy samples to memory."""
        device = f"cuda:{self.config.cuda_device}"

        obs, info = self.env.reset()
        action_mask = info['action_mask']
        current_player = self.env.get_current_player()

        for p in range(2):
            player_mask = current_player == p
            if player_mask.any():
                # Get current regret-matched strategy
                with torch.no_grad():
                    advantages = self.advantage_nets[p](obs[player_mask])
                    # Regret matching
                    pos_advantages = F.relu(advantages)
                    strategy = pos_advantages / (pos_advantages.sum(dim=-1, keepdim=True) + 1e-8)

                    # Mask invalid actions
                    strategy = strategy * action_mask[player_mask]
                    strategy = strategy / (strategy.sum(dim=-1, keepdim=True) + 1e-8)

                self.strategy_memory.append({
                    "player": p,
                    "obs": obs[player_mask].cpu(),
                    "strategy": strategy.cpu(),
                })

    def _train_strategy_network(self, player: int) -> float:
        """Train strategy network for a player."""
        device = f"cuda:{self.config.cuda_device}"

        player_data = [d for d in self.strategy_memory if d["player"] == player]
        if len(player_data) < self.config.strategy_batch_size:
            return 0.0

        indices = np.random.choice(len(player_data), self.config.strategy_batch_size, replace=False)

        obs_list = []
        target_list = []

        for i in indices:
            obs_list.append(player_data[i]["obs"])
            target_list.append(player_data[i]["strategy"])

        obs = torch.cat(obs_list).to(device)
        targets = torch.cat(target_list).to(device)

        # Forward and loss (KL divergence)
        predictions = self.strategy_nets[player](obs)
        loss = F.kl_div(predictions.log(), targets, reduction='batchmean')

        self.strat_optimizers[player].zero_grad()
        loss.backward()
        self.strat_optimizers[player].step()

        return loss.item()

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "advantage_nets": [net.state_dict() for net in self.advantage_nets],
            "strategy_nets": [net.state_dict() for net in self.strategy_nets],
            "adv_optimizers": [opt.state_dict() for opt in self.adv_optimizers],
            "strat_optimizers": [opt.state_dict() for opt in self.strat_optimizers],
            "iteration": self._iteration,
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        for net, state in zip(self.advantage_nets, data["advantage_nets"]):
            net.load_state_dict(state)
        for net, state in zip(self.strategy_nets, data["strategy_nets"]):
            net.load_state_dict(state)
        for opt, state in zip(self.adv_optimizers, data["adv_optimizers"]):
            opt.load_state_dict(state)
        for opt, state in zip(self.strat_optimizers, data["strat_optimizers"]):
            opt.load_state_dict(state)
        self._iteration = data["iteration"]

    def _cleanup(self) -> None:
        if self.env:
            self.env.close()

    def get_strategy_network(self, player: int = 0) -> nn.Module:
        return self.strategy_nets[player]
