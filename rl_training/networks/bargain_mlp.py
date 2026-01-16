"""
Simple MLP policy network for the bargaining game.

A faster, simpler alternative to the Transformer policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BargainMLP(nn.Module):
    """
    Simple MLP actor-critic network for bargaining game.

    Architecture:
        obs -> Linear -> ReLU -> Linear -> ReLU -> Linear
                                                    ↓
                                          [policy_head, value_head]
    """

    def __init__(
        self,
        obs_dim: int = 92,
        num_actions: int = 82,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ):
        """
        Initialize the MLP policy network.

        Args:
            obs_dim: Observation dimension (default 92 for bargaining game)
            num_actions: Number of actions (default 82)
            hidden_dims: Hidden layer dimensions
            activation: Activation function ("relu" or "tanh")
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Build trunk (shared layers)
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.policy_head = nn.Linear(hidden_dims[-1], num_actions)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observations [batch_size, obs_dim]
            action_mask: Optional action mask [batch_size, num_actions]
                        where 1 = valid, 0 = invalid

        Returns:
            logits: Action logits [batch_size, num_actions]
            value: Value estimates [batch_size]
        """
        features = self.trunk(obs)

        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Observations [batch_size, obs_dim]
            action_mask: Optional action mask [batch_size, num_actions]
            deterministic: If True, return argmax action

        Returns:
            action: Sampled actions [batch_size]
            log_prob: Log probabilities of actions [batch_size]
            value: Value estimates [batch_size]
        """
        logits, value = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions.

        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Actions to evaluate [batch_size]
            action_mask: Optional action mask [batch_size, num_actions]

        Returns:
            log_prob: Log probabilities of actions [batch_size]
            value: Value estimates [batch_size]
            entropy: Entropy of the policy [batch_size]
        """
        logits, value = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy
