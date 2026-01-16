"""
Information-Set Monte Carlo Tree Search implementation.

Supports both standard UCT and Gumbel AlphaZero variants.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from rl_training.algorithms.is_mcts.config import ISMCTSConfig
from rl_training.envs.bargain_wrapper import BargainEnvWrapper


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""

    # State information
    info_state_key: str = ""  # Key identifying the information set
    player: int = 0
    action_mask: Optional[np.ndarray] = None

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0

    # Tree structure
    parent: Optional["MCTSNode"] = None
    parent_action: int = -1
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)

    # For Gumbel
    completed_q: Optional[np.ndarray] = None

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0


class ISMCTS:
    """
    Information-Set Monte Carlo Tree Search.

    Implements search for imperfect information games where players
    have different information about the game state.
    """

    def __init__(
        self,
        config: ISMCTSConfig,
        policy_network: nn.Module,
        device: str = "cuda:0",
    ):
        self.config = config
        self.policy_network = policy_network
        self.device = device

        # Single env for simulation (will be cloned)
        self.eval_env: Optional[BargainEnvWrapper] = None

    def search(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        player: int,
        env_state: Any = None,
    ) -> np.ndarray:
        """
        Run MCTS search from current state.

        Args:
            obs: Current observation
            action_mask: Valid action mask
            player: Current player
            env_state: Environment state for cloning (if available)

        Returns:
            Action probabilities after search
        """
        if self.config.use_gumbel:
            return self._gumbel_search(obs, action_mask, player)
        else:
            return self._uct_search(obs, action_mask, player)

    def _uct_search(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        player: int,
    ) -> np.ndarray:
        """Standard UCT MCTS search."""
        # Get prior policy and value from network
        with torch.no_grad():
            logits, value = self.policy_network(
                obs.unsqueeze(0), action_mask.unsqueeze(0)
            )
            prior = F.softmax(logits[0], dim=-1).cpu().numpy()

        # Create root node
        root = MCTSNode(
            info_state_key=self._get_info_state_key(obs),
            player=player,
            action_mask=action_mask.cpu().numpy(),
            prior=1.0,
        )

        # Add Dirichlet noise to root
        valid_actions = np.where(action_mask.cpu().numpy() > 0)[0]
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))
        noisy_prior = prior.copy()
        for i, a in enumerate(valid_actions):
            noisy_prior[a] = (
                (1 - self.config.dirichlet_epsilon) * prior[a]
                + self.config.dirichlet_epsilon * noise[i]
            )

        # Expand root
        self._expand_node(root, noisy_prior, action_mask.cpu().numpy())

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree using UCB
            while node.is_expanded() and not self._is_terminal(node):
                action, child = self._select_child(node)
                node = child
                search_path.append(node)

            # Evaluation
            if self._is_terminal(node):
                value = 0.0  # Terminal nodes have 0 future value
            else:
                # Use value network
                value = self._evaluate_node(node, player)

            # Backup
            self._backup(search_path, value, player)

        # Return visit count distribution
        visit_counts = np.zeros(BargainEnvWrapper.NUM_ACTIONS)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        # Apply temperature
        if self.config.temperature == 0:
            # Greedy
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            visit_counts = visit_counts ** (1 / self.config.temperature)
            probs = visit_counts / visit_counts.sum()

        return probs

    def _gumbel_search(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        player: int,
    ) -> np.ndarray:
        """
        Gumbel AlphaZero search.

        Uses Gumbel-Top-k trick for action selection without visit counts.
        """
        # Get prior policy and value
        with torch.no_grad():
            logits, value = self.policy_network(
                obs.unsqueeze(0), action_mask.unsqueeze(0)
            )
            log_prior = F.log_softmax(logits[0], dim=-1).cpu().numpy()

        valid_actions = np.where(action_mask.cpu().numpy() > 0)[0]
        num_actions = len(valid_actions)

        if num_actions == 0:
            return np.zeros(BargainEnvWrapper.NUM_ACTIONS)

        # Sample Gumbel noise
        gumbel = np.random.gumbel(size=num_actions) * self.config.gumbel_scale

        # Compute scores: log_prior + gumbel
        scores = np.full(BargainEnvWrapper.NUM_ACTIONS, -np.inf)
        for i, a in enumerate(valid_actions):
            scores[a] = log_prior[a] + gumbel[i]

        # Select top-k actions to consider
        k = min(self.config.max_num_considered, num_actions)
        top_actions = np.argsort(scores)[-k:][::-1]

        # Evaluate each action with simulations
        q_values = np.zeros(BargainEnvWrapper.NUM_ACTIONS)
        visit_counts = np.zeros(BargainEnvWrapper.NUM_ACTIONS)

        simulations_per_action = self.config.num_simulations // k

        for action in top_actions:
            total_value = 0.0
            for _ in range(simulations_per_action):
                # Simulate action and get value
                sim_value = self._simulate_action(obs, action, player)
                total_value += sim_value
                visit_counts[action] += 1

            if visit_counts[action] > 0:
                q_values[action] = total_value / visit_counts[action]

        # Compute improved policy using Sequential Halving with Gumbel
        # Simplified: use softmax over Q + log_prior
        improved_logits = np.full(BargainEnvWrapper.NUM_ACTIONS, -np.inf)
        for a in valid_actions:
            if visit_counts[a] > 0:
                improved_logits[a] = q_values[a] + log_prior[a]
            else:
                improved_logits[a] = log_prior[a]

        # Normalize
        improved_logits = improved_logits - improved_logits.max()
        probs = np.exp(improved_logits)
        probs = probs / probs.sum()

        return probs

    def _simulate_action(
        self,
        obs: torch.Tensor,
        action: int,
        player: int,
    ) -> float:
        """
        Simulate taking an action and return estimated value.

        For now, uses network value estimate.
        Full implementation would clone env and simulate.
        """
        # Use value network as estimate
        with torch.no_grad():
            _, value = self.policy_network(
                obs.unsqueeze(0),
                torch.ones(1, BargainEnvWrapper.NUM_ACTIONS, device=self.device)
            )
        return value.item() * self.config.discount

    def _expand_node(
        self,
        node: MCTSNode,
        prior: np.ndarray,
        action_mask: np.ndarray,
    ):
        """Expand a node by creating children for valid actions."""
        valid_actions = np.where(action_mask > 0)[0]
        for action in valid_actions:
            child = MCTSNode(
                player=1 - node.player,  # Alternating players
                prior=prior[action],
                parent=node,
                parent_action=action,
            )
            node.children[action] = child

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child using PUCT formula."""
        best_score = -np.inf
        best_action = -1
        best_child = None

        sqrt_parent_visits = np.sqrt(node.visit_count)

        for action, child in node.children.items():
            # PUCT formula
            q_value = child.value if child.visit_count > 0 else 0.0
            u_value = (
                self.config.c_puct
                * child.prior
                * sqrt_parent_visits
                / (1 + child.visit_count)
            )
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _evaluate_node(self, node: MCTSNode, root_player: int) -> float:
        """Evaluate a leaf node using value network."""
        # Without full env simulation, return 0
        # Full implementation would evaluate the position
        return 0.0

    def _backup(
        self,
        search_path: List[MCTSNode],
        value: float,
        root_player: int,
    ):
        """Backup value through the search path."""
        for node in reversed(search_path):
            # Flip value for opponent
            if node.player != root_player:
                value = -value
            node.value_sum += value
            node.visit_count += 1

    def _is_terminal(self, node: MCTSNode) -> bool:
        """Check if node is terminal."""
        return not node.is_expanded() and node.visit_count > 0

    def _get_info_state_key(self, obs: torch.Tensor) -> str:
        """Get a hashable key for the information state."""
        return obs.cpu().numpy().tobytes()


class SearchEnhancedPolicy:
    """
    Wrapper that enhances any policy with MCTS search.

    Can be used at test time to improve decision quality.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        config: ISMCTSConfig,
        device: str = "cuda:0",
    ):
        self.base_policy = base_policy
        self.config = config
        self.device = device
        self.mcts = ISMCTS(config, base_policy, device)

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        player: int = 0,
        use_search: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, optionally using MCTS search.

        Args:
            obs: Observations [batch, obs_dim] or [obs_dim]
            action_mask: Action masks
            player: Current player
            use_search: Whether to use MCTS

        Returns:
            (actions, log_probs, values)
        """
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        batch_size = obs.shape[0]
        device = obs.device

        if use_search and batch_size == 1:
            # Use MCTS for single observations
            probs = self.mcts.search(obs[0], action_mask[0], player)
            probs = torch.tensor(probs, device=device, dtype=torch.float32)

            # Sample action
            action = torch.multinomial(probs.unsqueeze(0), 1).squeeze(-1)
            log_prob = torch.log(probs[action] + 1e-8)

            # Get value from network
            with torch.no_grad():
                _, value = self.base_policy(obs, action_mask)

            if single:
                return action.squeeze(0), log_prob, value.squeeze(0)
            return action, log_prob.unsqueeze(0), value
        else:
            # Fall back to base policy for batched inference
            return self.base_policy.get_action(obs, action_mask)

    def __call__(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """Forward pass using base policy."""
        return self.base_policy(obs, action_mask)
