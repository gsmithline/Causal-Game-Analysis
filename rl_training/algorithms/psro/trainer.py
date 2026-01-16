"""
PSRO (Policy Space Response Oracles) trainer for bargaining game.

Implements the PSRO algorithm which iteratively builds a population
of policies by computing Nash equilibria and training best responses.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.psro.config import PSROConfig
from rl_training.envs.bargain_wrapper import BargainEnvWrapper
from rl_training.networks.bargain_mlp import BargainMLP
from rl_training.networks.transformer_policy import TransformerPolicyNetwork


def solve_nash_replicator(
    payoff_matrix: np.ndarray,
    iterations: int = 10000,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for Nash equilibrium using replicator dynamics.

    Args:
        payoff_matrix: Shape [n_policies_p1, n_policies_p2, 2] payoff tensor
        iterations: Number of replicator iterations
        dt: Step size

    Returns:
        (strategy_p1, strategy_p2) - Nash equilibrium mixed strategies
    """
    n1, n2 = payoff_matrix.shape[:2]

    # Initialize uniform
    x = np.ones(n1) / n1  # Player 1 strategy
    y = np.ones(n2) / n2  # Player 2 strategy

    for _ in range(iterations):
        # Player 1's expected payoff for each pure strategy
        u1 = payoff_matrix[:, :, 0] @ y  # [n1]
        avg_u1 = x @ u1

        # Player 2's expected payoff for each pure strategy
        u2 = payoff_matrix[:, :, 1].T @ x  # [n2]
        avg_u2 = y @ u2

        # Replicator dynamics update
        x = x * (1 + dt * (u1 - avg_u1))
        y = y * (1 + dt * (u2 - avg_u2))

        # Normalize
        x = np.maximum(x, 1e-10)
        y = np.maximum(y, 1e-10)
        x = x / x.sum()
        y = y / y.sum()

    return x, y


def solve_nash_fictitious_play(
    payoff_matrix: np.ndarray,
    iterations: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for Nash equilibrium using fictitious play.
    """
    n1, n2 = payoff_matrix.shape[:2]

    counts1 = np.zeros(n1)
    counts2 = np.zeros(n2)

    for t in range(1, iterations + 1):
        # Player 1 best responds to empirical distribution of player 2
        if t == 1:
            y = np.ones(n2) / n2
        else:
            y = counts2 / counts2.sum()
        u1 = payoff_matrix[:, :, 0] @ y
        br1 = np.argmax(u1)
        counts1[br1] += 1

        # Player 2 best responds to empirical distribution of player 1
        x = counts1 / counts1.sum()
        u2 = payoff_matrix[:, :, 1].T @ x
        br2 = np.argmax(u2)
        counts2[br2] += 1

    return counts1 / counts1.sum(), counts2 / counts2.sum()


class PolicyPopulation:
    """Manages a population of policies for one player."""

    def __init__(self, player_id: int, device: str):
        self.player_id = player_id
        self.device = device
        self.policies: List[nn.Module] = []

    def add_policy(self, policy: nn.Module):
        """Add a policy to the population."""
        policy_copy = deepcopy(policy)
        policy_copy.eval()
        self.policies.append(policy_copy)

    def get_policy(self, idx: int) -> nn.Module:
        """Get policy by index."""
        return self.policies[idx]

    def sample_from_mixture(
        self,
        mixture: np.ndarray,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions from mixture of policies.

        Args:
            mixture: Probability distribution over policies
            obs: Observations [batch, obs_dim]
            action_mask: Action masks [batch, num_actions]

        Returns:
            Actions [batch]
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Sample which policy to use for each env
        policy_indices = np.random.choice(
            len(self.policies),
            size=batch_size,
            p=mixture,
        )

        actions = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Get actions from each policy
        for idx in range(len(self.policies)):
            mask = torch.tensor(policy_indices == idx, device=device)
            if mask.any():
                with torch.no_grad():
                    policy_actions, _, _ = self.policies[idx].get_action(
                        obs[mask], action_mask[mask]
                    )
                actions[mask] = policy_actions

        return actions

    def __len__(self):
        return len(self.policies)


@register_algorithm("psro")
class PSROTrainer(BaseTrainer[PSROConfig]):
    """
    PSRO trainer for bargaining game.

    Algorithm:
    1. Initialize population with random policy
    2. For each iteration:
       a. Compute payoff matrix for current population
       b. Solve for Nash equilibrium
       c. Train best response against Nash mixture
       d. Add BR to population
    """

    def __init__(
        self,
        config: PSROConfig,
        env_fn: Optional[Callable] = None,
        logger: Optional[Any] = None,
    ):
        super().__init__(config, env_fn, logger)

        self.env: Optional[BargainEnvWrapper] = None
        self.populations: List[PolicyPopulation] = []
        self.payoff_matrix: Optional[np.ndarray] = None
        self.nash_strategies: List[np.ndarray] = [None, None]

        self._psro_iteration = 0
        self._br_policy: Optional[nn.Module] = None
        self._br_optimizer: Optional[torch.optim.Optimizer] = None

    def _setup(self) -> None:
        """Initialize PSRO components."""
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

        # Initialize populations
        self.populations = [
            PolicyPopulation(p, self.device) for p in range(2)
        ]

        # Add initial random policies
        for p in range(2):
            for _ in range(self.config.initial_policies):
                policy = self._create_policy()
                self.populations[p].add_policy(policy)

        # Initialize payoff matrix
        n = self.config.initial_policies
        self.payoff_matrix = np.zeros((n, n, 2))

        # Compute initial payoffs
        self._update_payoff_matrix()

        # Initialize Nash strategies (uniform)
        for p in range(2):
            self.nash_strategies[p] = np.ones(len(self.populations[p])) / len(self.populations[p])

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
        """Execute one PSRO iteration."""
        self._psro_iteration += 1
        metrics = {"psro_iteration": self._psro_iteration}

        # 1. Solve Nash equilibrium
        self._solve_nash()
        metrics["nash_support_p0"] = (self.nash_strategies[0] > 0.01).sum()
        metrics["nash_support_p1"] = (self.nash_strategies[1] > 0.01).sum()

        # 2. Train best response for each player
        for player in range(2):
            if len(self.populations[player]) >= self.config.max_policies:
                continue

            br_metrics = self._train_best_response(player)
            for k, v in br_metrics.items():
                metrics[f"br_p{player}_{k}"] = v

            # Add BR to population
            self.populations[player].add_policy(self._br_policy)

        # 3. Update payoff matrix with new policies
        self._update_payoff_matrix()

        metrics["population_size_p0"] = len(self.populations[0])
        metrics["population_size_p1"] = len(self.populations[1])
        metrics["timesteps"] = 1  # 1 PSRO iteration

        return metrics

    def _solve_nash(self):
        """Solve for Nash equilibrium over current population."""
        if self.config.nash_solver == "replicator":
            self.nash_strategies[0], self.nash_strategies[1] = solve_nash_replicator(
                self.payoff_matrix,
                iterations=self.config.replicator_iterations,
                dt=self.config.replicator_dt,
            )
        elif self.config.nash_solver == "fictitious_play":
            self.nash_strategies[0], self.nash_strategies[1] = solve_nash_fictitious_play(
                self.payoff_matrix,
                iterations=self.config.replicator_iterations,
            )
        else:
            # Uniform fallback
            n1 = len(self.populations[0])
            n2 = len(self.populations[1])
            self.nash_strategies[0] = np.ones(n1) / n1
            self.nash_strategies[1] = np.ones(n2) / n2

    def _train_best_response(self, player: int) -> Dict[str, float]:
        """
        Train a best response policy against opponent's Nash mixture.

        Args:
            player: Player ID to train BR for

        Returns:
            Training metrics
        """
        opponent = 1 - player

        # Create fresh policy
        self._br_policy = self._create_policy()
        self._br_optimizer = torch.optim.Adam(
            self._br_policy.parameters(),
            lr=self.config.lr,
        )

        total_reward = 0
        num_episodes = 0

        # Training loop (simplified PPO)
        steps = 0
        while steps < self.config.br_training_steps:
            rollout = self._collect_br_rollout(player, opponent)
            update_metrics = self._ppo_update_br(rollout)

            steps += self.config.br_rollout_steps * self.config.num_envs
            total_reward += rollout["total_reward"]
            num_episodes += rollout["num_episodes"]

        avg_reward = total_reward / max(1, num_episodes)
        return {"avg_reward": avg_reward, **update_metrics}

    def _collect_br_rollout(
        self,
        player: int,
        opponent: int,
    ) -> Dict[str, torch.Tensor]:
        """Collect rollout for BR training."""
        obs_list = []
        action_list = []
        log_prob_list = []
        value_list = []
        reward_list = []
        done_list = []
        mask_list = []

        obs, info = self.env.reset()
        action_mask = info['action_mask']

        total_reward = 0
        num_episodes = 0

        for _ in range(self.config.br_rollout_steps):
            current_player = self.env.get_current_player()

            # Get actions
            actions = torch.zeros(
                self.config.num_envs,
                dtype=torch.long,
                device=self.device
            )

            player_mask = current_player == player
            opp_mask = current_player == opponent

            # BR policy actions
            if player_mask.any():
                obs_list.append(obs[player_mask])
                mask_list.append(action_mask[player_mask])

                with torch.no_grad():
                    br_actions, log_probs, values = self._br_policy.get_action(
                        obs[player_mask], action_mask[player_mask]
                    )
                actions[player_mask] = br_actions
                action_list.append(br_actions)
                log_prob_list.append(log_probs)
                value_list.append(values)

            # Opponent actions from Nash mixture
            if opp_mask.any():
                opp_actions = self.populations[opponent].sample_from_mixture(
                    self.nash_strategies[opponent],
                    obs[opp_mask],
                    action_mask[opp_mask],
                )
                actions[opp_mask] = opp_actions

            # Step
            result = self.env.step(actions.int())
            dones = result.terminated
            rewards = result.rewards

            # Store rewards for BR player
            if player_mask.any():
                player_rewards = rewards[player_mask, player]
                reward_list.append(player_rewards)
                done_list.append(dones[player_mask].float())

            if dones.any():
                total_reward += rewards[dones, player].sum().item()
                num_episodes += dones.sum().item()
                self.env.auto_reset()

            obs = result.observations
            action_mask = result.info['action_mask']

        # Stack if we have data
        if len(obs_list) == 0:
            return {
                "obs": torch.zeros(1, BargainEnvWrapper.OBS_DIM, device=self.device),
                "actions": torch.zeros(1, dtype=torch.long, device=self.device),
                "log_probs_old": torch.zeros(1, device=self.device),
                "returns": torch.zeros(1, device=self.device),
                "advantages": torch.zeros(1, device=self.device),
                "masks": torch.zeros(1, BargainEnvWrapper.NUM_ACTIONS, device=self.device),
                "total_reward": total_reward,
                "num_episodes": num_episodes,
            }

        obs_batch = torch.cat(obs_list)
        action_batch = torch.cat(action_list)
        log_prob_batch = torch.cat(log_prob_list)
        value_batch = torch.cat(value_list)
        reward_batch = torch.cat(reward_list)
        done_batch = torch.cat(done_list)
        mask_batch = torch.cat(mask_list)

        # Compute advantages (simplified - no bootstrapping across boundaries)
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
            "total_reward": total_reward,
            "num_episodes": num_episodes,
        }

    def _ppo_update_br(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """PPO update for best response policy."""
        obs = rollout["obs"]
        actions = rollout["actions"]
        log_probs_old = rollout["log_probs_old"]
        returns = rollout["returns"]
        advantages = rollout["advantages"]
        masks = rollout["masks"]

        if len(obs) < self.config.br_minibatch_size:
            return {"loss": 0.0}

        total_loss = 0.0
        num_updates = 0

        for _ in range(self.config.br_ppo_epochs):
            indices = torch.randperm(len(obs), device=self.device)

            for start in range(0, len(obs), self.config.br_minibatch_size):
                end = min(start + self.config.br_minibatch_size, len(obs))
                mb_idx = indices[start:end]

                log_probs, values, entropy = self._br_policy.evaluate_actions(
                    obs[mb_idx], actions[mb_idx], masks[mb_idx]
                )

                # PPO loss
                ratio = torch.exp(log_probs - log_probs_old[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.br_clip_eps,
                    1 + self.config.br_clip_eps
                ) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns[mb_idx])
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.br_value_coef * value_loss
                    + self.config.br_entropy_coef * entropy_loss
                )

                self._br_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._br_policy.parameters(),
                    self.config.br_max_grad_norm
                )
                self._br_optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return {"loss": total_loss / max(1, num_updates)}

    def _update_payoff_matrix(self):
        """Update payoff matrix with new policies."""
        n1 = len(self.populations[0])
        n2 = len(self.populations[1])

        # Expand matrix if needed
        old_n1, old_n2 = self.payoff_matrix.shape[:2]
        if n1 > old_n1 or n2 > old_n2:
            new_matrix = np.zeros((n1, n2, 2))
            new_matrix[:old_n1, :old_n2, :] = self.payoff_matrix
            self.payoff_matrix = new_matrix

        # Compute payoffs for new matchups
        for i in range(n1):
            for j in range(n2):
                # Skip if already computed
                if i < old_n1 and j < old_n2:
                    continue

                payoffs = self._evaluate_matchup(i, j)
                self.payoff_matrix[i, j, 0] = payoffs[0]
                self.payoff_matrix[i, j, 1] = payoffs[1]

    def _evaluate_matchup(self, policy_idx_p0: int, policy_idx_p1: int) -> Tuple[float, float]:
        """Evaluate a matchup between two policies."""
        policy_p0 = self.populations[0].get_policy(policy_idx_p0)
        policy_p1 = self.populations[1].get_policy(policy_idx_p1)

        total_rewards = np.zeros(2)
        games_played = 0

        obs, info = self.env.reset()
        action_mask = info['action_mask']

        while games_played < self.config.num_eval_games:
            current_player = self.env.get_current_player()

            actions = torch.zeros(
                self.config.num_envs,
                dtype=torch.long,
                device=self.device
            )

            # Player 0 actions
            p0_mask = current_player == 0
            if p0_mask.any():
                with torch.no_grad():
                    p0_actions, _, _ = policy_p0.get_action(
                        obs[p0_mask], action_mask[p0_mask]
                    )
                actions[p0_mask] = p0_actions

            # Player 1 actions
            p1_mask = current_player == 1
            if p1_mask.any():
                with torch.no_grad():
                    p1_actions, _, _ = policy_p1.get_action(
                        obs[p1_mask], action_mask[p1_mask]
                    )
                actions[p1_mask] = p1_actions

            result = self.env.step(actions.int())
            dones = result.terminated

            if dones.any():
                rewards = result.rewards[dones].cpu().numpy()
                total_rewards += rewards.sum(axis=0)
                games_played += dones.sum().item()
                self.env.auto_reset()

            obs = result.observations
            action_mask = result.info['action_mask']

        return total_rewards / games_played

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data."""
        return {
            "populations": [
                [p.state_dict() for p in pop.policies]
                for pop in self.populations
            ],
            "payoff_matrix": self.payoff_matrix,
            "nash_strategies": self.nash_strategies,
            "psro_iteration": self._psro_iteration,
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        """Load checkpoint data."""
        for p, pop_data in enumerate(data["populations"]):
            self.populations[p].policies = []
            for state_dict in pop_data:
                policy = self._create_policy()
                policy.load_state_dict(state_dict)
                policy.eval()
                self.populations[p].policies.append(policy)

        self.payoff_matrix = data["payoff_matrix"]
        self.nash_strategies = data["nash_strategies"]
        self._psro_iteration = data["psro_iteration"]

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()

    def get_nash_policy(self, player: int = 0) -> Tuple[PolicyPopulation, np.ndarray]:
        """Get the Nash mixture policy for a player."""
        return self.populations[player], self.nash_strategies[player]
