"""Random baseline policy."""

import torch
from typing import Optional


class RandomPolicy:
    """
    Uniformly random policy over valid actions.

    Serves as a lower-bound baseline.
    """

    def __init__(self):
        self.name = "random"

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get random valid actions.

        Args:
            obs: [batch, obs_dim] observations
            action_mask: [batch, num_actions] where 1=valid, 0=invalid
            deterministic: Ignored (always stochastic)

        Returns:
            actions: [batch] selected actions
        """
        device = obs.device
        batch_size = obs.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            valid_indices = action_mask[i].nonzero().squeeze(-1)
            if len(valid_indices) > 0:
                idx = torch.randint(len(valid_indices), (1,), device=device)
                actions[i] = valid_indices[idx]

        return actions

    def get_action_probs(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get uniform distribution over valid actions.

        Args:
            obs: [batch, obs_dim] observations
            action_mask: [batch, num_actions]

        Returns:
            probs: [batch, num_actions] action probabilities
        """
        # Uniform over valid actions
        probs = action_mask / (action_mask.sum(dim=-1, keepdim=True) + 1e-8)
        return probs
