"""
Walk heuristic strategy.

Always walks away at the first opportunity.
Both players get their BATNA (outside option).

Reference: Section 4.2 of LLM Meta-Game paper.
"""

import torch
import torch.nn as nn

ACTION_WALK = 81
NUM_ACTIONS = 82


class WalkPolicy(nn.Module):
    """
    Heuristic policy that always walks away.

    At any turn, returns logits that strongly favor ACTION_WALK (81).
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter so this is a valid nn.Module
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """
        Args:
            obs: [batch, 92] observation tensor
            action_mask: [batch, 82] action mask tensor

        Returns:
            logits: [batch, 82] with high value for WALK action
            value: [batch] dummy value (zeros)
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Create logits that strongly favor WALK (action 81)
        logits = torch.full((batch_size, NUM_ACTIONS), -1e9, device=device)
        logits[:, ACTION_WALK] = 0.0  # WALK gets highest probability

        # Dummy value output
        value = torch.zeros(batch_size, device=device)

        return logits, value
