"""
Ex²PSRO (Explicit Exploration PSRO) trainer for bargaining game.

Extends PSRO to find high-welfare equilibria by:
1. Creating exploration policies that imitate high-welfare behavior
2. Regularizing best response training toward the exploration policy
3. Biasing equilibrium selection toward prosocial outcomes

Based on "Explicit Exploration for High-Welfare Equilibria in
Game-Theoretic Multiagent Reinforcement Learning" (OpenReview 2025).
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from collections import deque

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.ex2psro.config import Ex2PSROConfig
from rl_training.algorithms.psro.trainer import (
    PolicyPopulation,
    solve_nash_replicator,
    solve_nash_fictitious_play,
)
from rl_training.envs.bargain_wrapper import BargainEnvWrapper
from rl_training.networks.bargain_mlp import BargainMLP
from rl_training.networks.transformer_policy import TransformerPolicyNetwork


def compute_welfare(
    rewards: np.ndarray,
    welfare_fn: str = "utilitarian",
) -> float:
    """
    Compute welfare from reward vector.

    Args:
        rewards: Array of rewards for each player [num_players]
        welfare_fn: "utilitarian", "nash", or "egalitarian"

    Returns:
        Scalar welfare value
    """
    if welfare_fn == "utilitarian":
        return rewards.sum()
    elif welfare_fn == "nash":
        # Nash product (geometric mean for positive values)
        # For bargaining games with non-negative rewards, use geometric mean
        clipped = np.maximum(rewards, 1e-8)
        return np.prod(clipped) ** (1 / len(rewards))
    elif welfare_fn == "egalitarian":
        return rewards.min()
    else:
        return rewards.sum()


class WelfareTracker:
    """Tracks welfare metrics for policies and matchups."""

    def __init__(self, welfare_fn: str = "utilitarian"):
        self.welfare_fn = welfare_fn
        self.policy_welfare: Dict[Tuple[int, int], float] = {}  # (player, idx) -> welfare
        self.matchup_welfare: Dict[Tuple[int, int], float] = {}  # (p0_idx, p1_idx) -> welfare
        self.welfare_history: List[float] = []

    def update_matchup_welfare(self, p0_idx: int, p1_idx: int, rewards: np.ndarray):
        """Update welfare for a matchup."""
        welfare = compute_welfare(rewards, self.welfare_fn)
        self.matchup_welfare[(p0_idx, p1_idx)] = welfare

    def get_top_k_policies(
        self,
        player: int,
        population_size: int,
        k: int,
        payoff_matrix: np.ndarray,
    ) -> List[int]:
        """
        Get indices of top-k highest welfare policies for a player.

        Welfare is computed as average welfare when playing against
        all opponent policies.
        """
        welfare_scores = []

        for idx in range(population_size):
            if player == 0:
                # Average welfare across all opponents
                avg_welfare = np.mean([
                    compute_welfare(payoff_matrix[idx, j, :], self.welfare_fn)
                    for j in range(payoff_matrix.shape[1])
                ])
            else:
                avg_welfare = np.mean([
                    compute_welfare(payoff_matrix[i, idx, :], self.welfare_fn)
                    for i in range(payoff_matrix.shape[0])
                ])
            welfare_scores.append((idx, avg_welfare))

        # Sort by welfare descending
        welfare_scores.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in welfare_scores[:k]]

    def get_welfare_weighted_distribution(
        self,
        player: int,
        population_size: int,
        payoff_matrix: np.ndarray,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get welfare-weighted distribution over policies.

        Higher welfare policies get higher probability.
        """
        welfare_scores = []

        for idx in range(population_size):
            if player == 0:
                avg_welfare = np.mean([
                    compute_welfare(payoff_matrix[idx, j, :], self.welfare_fn)
                    for j in range(payoff_matrix.shape[1])
                ])
            else:
                avg_welfare = np.mean([
                    compute_welfare(payoff_matrix[i, idx, :], self.welfare_fn)
                    for i in range(payoff_matrix.shape[0])
                ])
            welfare_scores.append(avg_welfare)

        welfare_scores = np.array(welfare_scores)

        # Apply temperature and softmax
        if top_k is not None and top_k < population_size:
            # Zero out non-top-k
            top_indices = np.argsort(welfare_scores)[-top_k:]
            mask = np.zeros_like(welfare_scores)
            mask[top_indices] = 1.0
            welfare_scores = welfare_scores * mask

        # Softmax with temperature
        welfare_scores = welfare_scores / temperature
        welfare_scores = welfare_scores - welfare_scores.max()
        probs = np.exp(welfare_scores)
        probs = probs / probs.sum()

        return probs


@register_algorithm("ex2psro")
class Ex2PSROTrainer(BaseTrainer[Ex2PSROConfig]):
    """
    Ex²PSRO trainer for bargaining game.

    Algorithm extends PSRO with welfare-biased exploration:
    1. Initialize population with random policy
    2. For each iteration:
       a. Compute payoff matrix for current population
       b. Solve for Nash equilibrium
       c. Identify high-welfare policies
       d. Create exploration policy that imitates high-welfare behavior
       e. Train best response regularized toward exploration policy
       f. Add BR to population
    """

    def __init__(
        self,
        config: Ex2PSROConfig,
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

        # Ex²PSRO specific
        self._exploration_policy: Optional[nn.Module] = None
        self._welfare_tracker: Optional[WelfareTracker] = None
        self._kl_coef = config.kl_coef
        self._kl_history: deque = deque(maxlen=config.kl_horizon)

    def _setup(self) -> None:
        """Initialize Ex²PSRO components."""
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

        # Initialize welfare tracker
        self._welfare_tracker = WelfareTracker(self.config.welfare_fn)

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
        """Execute one Ex²PSRO iteration."""
        self._psro_iteration += 1
        metrics = {"psro_iteration": self._psro_iteration}

        # 1. Solve Nash equilibrium
        self._solve_nash()
        metrics["nash_support_p0"] = (self.nash_strategies[0] > 0.01).sum()
        metrics["nash_support_p1"] = (self.nash_strategies[1] > 0.01).sum()

        # 2. Compute current equilibrium welfare
        eq_welfare = self._compute_equilibrium_welfare()
        metrics["equilibrium_welfare"] = eq_welfare
        self._welfare_tracker.welfare_history.append(eq_welfare)

        # 3. Train best response for each player with exploration regularization
        for player in range(2):
            if len(self.populations[player]) >= self.config.max_policies:
                continue

            # Create exploration policy for this player
            self._create_exploration_policy(player)

            # Train regularized best response
            br_metrics = self._train_regularized_best_response(player)
            for k, v in br_metrics.items():
                metrics[f"br_p{player}_{k}"] = v

            # Add BR to population
            self.populations[player].add_policy(self._br_policy)

        # 4. Update payoff matrix with new policies
        self._update_payoff_matrix()

        # 5. Track welfare improvement
        new_eq_welfare = self._compute_equilibrium_welfare()
        metrics["welfare_improvement"] = new_eq_welfare - eq_welfare

        metrics["population_size_p0"] = len(self.populations[0])
        metrics["population_size_p1"] = len(self.populations[1])
        metrics["kl_coef"] = self._kl_coef
        metrics["timesteps"] = 1

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
            n1 = len(self.populations[0])
            n2 = len(self.populations[1])
            self.nash_strategies[0] = np.ones(n1) / n1
            self.nash_strategies[1] = np.ones(n2) / n2

    def _compute_equilibrium_welfare(self) -> float:
        """Compute welfare at current Nash equilibrium."""
        expected_payoffs = np.zeros(2)

        for i in range(len(self.populations[0])):
            for j in range(len(self.populations[1])):
                prob = self.nash_strategies[0][i] * self.nash_strategies[1][j]
                expected_payoffs += prob * self.payoff_matrix[i, j, :]

        return compute_welfare(expected_payoffs, self.config.welfare_fn)

    def _create_exploration_policy(self, player: int) -> None:
        """
        Create exploration policy that imitates high-welfare behavior.

        The exploration policy is trained via imitation learning to
        mimic the behavior of top-k highest welfare policies.
        """
        population_size = len(self.populations[player])

        if population_size <= 1:
            # Not enough policies to learn from, use current policy
            self._exploration_policy = deepcopy(self.populations[player].policies[0])
            return

        # Get top-k welfare policies
        top_k = min(self.config.exploration_top_k, population_size)
        top_k_indices = self._welfare_tracker.get_top_k_policies(
            player, population_size, top_k, self.payoff_matrix
        )

        # Get welfare-weighted distribution
        if self.config.use_welfare_weighted_sampling:
            expert_probs = self._welfare_tracker.get_welfare_weighted_distribution(
                player,
                population_size,
                self.payoff_matrix,
                temperature=self.config.exploration_temperature,
                top_k=top_k,
            )
        else:
            # Uniform over top-k
            expert_probs = np.zeros(population_size)
            expert_probs[top_k_indices] = 1.0 / top_k

        # Collect imitation data from expert policies
        imitation_data = self._collect_imitation_data(player, expert_probs)

        # Train exploration policy via behavior cloning
        self._exploration_policy = self._train_exploration_policy(imitation_data)

    def _collect_imitation_data(
        self,
        player: int,
        expert_probs: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Collect (obs, action) pairs from expert policies."""
        obs_list = []
        action_list = []
        mask_list = []

        opponent = 1 - player
        num_samples = 0

        obs, info = self.env.reset()
        action_mask = info['action_mask']

        while num_samples < self.config.imitation_data_size:
            current_player = self.env.get_current_player()

            actions = torch.zeros(
                self.config.num_envs,
                dtype=torch.long,
                device=self.device
            )

            player_mask = current_player == player
            opp_mask = current_player == opponent

            # Expert policy actions (from welfare-weighted mixture)
            if player_mask.any():
                expert_actions = self.populations[player].sample_from_mixture(
                    expert_probs,
                    obs[player_mask],
                    action_mask[player_mask],
                )
                actions[player_mask] = expert_actions

                # Store for imitation
                obs_list.append(obs[player_mask].clone())
                action_list.append(expert_actions.clone())
                mask_list.append(action_mask[player_mask].clone())
                num_samples += player_mask.sum().item()

            # Opponent actions from Nash
            if opp_mask.any():
                opp_actions = self.populations[opponent].sample_from_mixture(
                    self.nash_strategies[opponent],
                    obs[opp_mask],
                    action_mask[opp_mask],
                )
                actions[opp_mask] = opp_actions

            result = self.env.step(actions.int())
            if result.terminated.any():
                self.env.auto_reset()

            obs = result.observations
            action_mask = result.info['action_mask']

        return {
            "obs": torch.cat(obs_list)[:self.config.imitation_data_size],
            "actions": torch.cat(action_list)[:self.config.imitation_data_size],
            "masks": torch.cat(mask_list)[:self.config.imitation_data_size],
        }

    def _train_exploration_policy(
        self,
        imitation_data: Dict[str, torch.Tensor],
    ) -> nn.Module:
        """Train exploration policy via behavior cloning."""
        policy = self._create_policy()
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=self.config.imitation_lr,
        )

        obs = imitation_data["obs"]
        actions = imitation_data["actions"]
        masks = imitation_data["masks"]

        num_samples = len(obs)
        batch_size = self.config.imitation_batch_size

        for epoch in range(self.config.imitation_epochs):
            indices = torch.randperm(num_samples, device=self.device)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                mb_idx = indices[start:end]

                # Get policy logits
                logits = policy.forward_actor(obs[mb_idx])

                # Mask invalid actions
                logits = logits.masked_fill(~masks[mb_idx].bool(), float('-inf'))

                # Cross entropy loss
                loss = F.cross_entropy(logits, actions[mb_idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        policy.eval()
        return policy

    def _train_regularized_best_response(self, player: int) -> Dict[str, float]:
        """
        Train best response policy regularized toward exploration policy.

        The KL divergence between BR and exploration policy is added
        as a regularization term to bias toward high-welfare behavior.
        """
        opponent = 1 - player

        # Create fresh policy
        self._br_policy = self._create_policy()
        self._br_optimizer = torch.optim.Adam(
            self._br_policy.parameters(),
            lr=self.config.lr,
        )

        total_reward = 0
        total_kl = 0
        num_episodes = 0
        num_updates = 0

        # Training loop
        steps = 0
        while steps < self.config.br_training_steps:
            rollout = self._collect_br_rollout(player, opponent)
            update_metrics = self._ppo_update_with_kl_regularization(rollout)

            steps += self.config.br_rollout_steps * self.config.num_envs
            total_reward += rollout["total_reward"]
            total_kl += update_metrics.get("kl", 0)
            num_episodes += rollout["num_episodes"]
            num_updates += 1

            # Adaptive KL coefficient
            if self.config.use_adaptive_kl and num_updates > 0:
                self._adapt_kl_coef(update_metrics.get("kl", 0))

        avg_reward = total_reward / max(1, num_episodes)
        avg_kl = total_kl / max(1, num_updates)

        return {
            "avg_reward": avg_reward,
            "avg_kl": avg_kl,
            "kl_coef": self._kl_coef,
            **{k: v for k, v in update_metrics.items() if k != "kl"},
        }

    def _adapt_kl_coef(self, current_kl: float):
        """Adapt KL coefficient based on target."""
        self._kl_history.append(current_kl)

        if len(self._kl_history) >= self.config.kl_horizon:
            avg_kl = np.mean(self._kl_history)

            if avg_kl > 2.0 * self.config.kl_target:
                self._kl_coef *= 1.5
            elif avg_kl < 0.5 * self.config.kl_target:
                self._kl_coef *= 0.5

            self._kl_coef = np.clip(self._kl_coef, 1e-4, 10.0)

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

            result = self.env.step(actions.int())
            dones = result.terminated
            rewards = result.rewards

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

        advantages = reward_batch - value_batch
        returns = reward_batch

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

    def _ppo_update_with_kl_regularization(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """PPO update with KL regularization toward exploration policy."""
        obs = rollout["obs"]
        actions = rollout["actions"]
        log_probs_old = rollout["log_probs_old"]
        returns = rollout["returns"]
        advantages = rollout["advantages"]
        masks = rollout["masks"]

        if len(obs) < self.config.br_minibatch_size:
            return {"loss": 0.0, "kl": 0.0}

        total_loss = 0.0
        total_kl = 0.0
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

                # KL regularization toward exploration policy
                kl_loss, kl_value = self._compute_kl_regularization(
                    obs[mb_idx], masks[mb_idx]
                )

                loss = (
                    policy_loss
                    + self.config.br_value_coef * value_loss
                    + self.config.br_entropy_coef * entropy_loss
                    + self._kl_coef * kl_loss
                )

                self._br_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._br_policy.parameters(),
                    self.config.br_max_grad_norm
                )
                self._br_optimizer.step()

                total_loss += loss.item()
                total_kl += kl_value
                num_updates += 1

        return {
            "loss": total_loss / max(1, num_updates),
            "kl": total_kl / max(1, num_updates),
        }

    def _compute_kl_regularization(
        self,
        obs: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute KL divergence between BR policy and exploration policy.

        Returns:
            (kl_loss, kl_value): KL loss tensor and scalar KL value
        """
        if self._exploration_policy is None:
            return torch.tensor(0.0, device=self.device), 0.0

        # Get BR policy logits
        br_logits = self._br_policy.forward_actor(obs)
        br_logits = br_logits.masked_fill(~masks.bool(), float('-inf'))
        br_log_probs = F.log_softmax(br_logits, dim=-1)

        # Get exploration policy logits
        with torch.no_grad():
            exp_logits = self._exploration_policy.forward_actor(obs)
            exp_logits = exp_logits.masked_fill(~masks.bool(), float('-inf'))
            exp_probs = F.softmax(exp_logits, dim=-1)

        # KL(exploration || BR) - encourage BR to stay close to exploration
        # Using KL(p||q) = sum(p * log(p/q)) = sum(p * log(p)) - sum(p * log(q))
        kl = (exp_probs * (F.log_softmax(exp_logits, dim=-1) - br_log_probs)).sum(dim=-1)

        # Handle invalid actions (where both have -inf)
        kl = kl.clamp(min=0)

        kl_loss = kl.mean()
        kl_value = kl_loss.item()

        return kl_loss, kl_value

    def _update_payoff_matrix(self):
        """Update payoff matrix with new policies."""
        n1 = len(self.populations[0])
        n2 = len(self.populations[1])

        old_n1, old_n2 = self.payoff_matrix.shape[:2]
        if n1 > old_n1 or n2 > old_n2:
            new_matrix = np.zeros((n1, n2, 2))
            new_matrix[:old_n1, :old_n2, :] = self.payoff_matrix
            self.payoff_matrix = new_matrix

        for i in range(n1):
            for j in range(n2):
                if i < old_n1 and j < old_n2:
                    continue

                payoffs = self._evaluate_matchup(i, j)
                self.payoff_matrix[i, j, 0] = payoffs[0]
                self.payoff_matrix[i, j, 1] = payoffs[1]

                # Update welfare tracker
                self._welfare_tracker.update_matchup_welfare(i, j, payoffs)

    def _evaluate_matchup(
        self,
        policy_idx_p0: int,
        policy_idx_p1: int,
    ) -> Tuple[float, float]:
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

            p0_mask = current_player == 0
            if p0_mask.any():
                with torch.no_grad():
                    p0_actions, _, _ = policy_p0.get_action(
                        obs[p0_mask], action_mask[p0_mask]
                    )
                actions[p0_mask] = p0_actions

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
            "kl_coef": self._kl_coef,
            "welfare_history": self._welfare_tracker.welfare_history,
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
        self._kl_coef = data.get("kl_coef", self.config.kl_coef)

        if "welfare_history" in data:
            self._welfare_tracker.welfare_history = data["welfare_history"]

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()

    def get_nash_policy(self, player: int = 0) -> Tuple[PolicyPopulation, np.ndarray]:
        """Get the Nash mixture policy for a player."""
        return self.populations[player], self.nash_strategies[player]

    def get_welfare_history(self) -> List[float]:
        """Get the welfare history over PSRO iterations."""
        return self._welfare_tracker.welfare_history
