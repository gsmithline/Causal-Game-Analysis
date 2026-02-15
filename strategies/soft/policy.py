"""
Soft heuristic strategy.

Accepts any offer received. When required to propose (P1 in round 1),
generates an offer uniformly at random.

Reference: Section 4.2 of LLM Meta-Game paper.
"""

import torch
import torch.nn as nn

NUM_OFFERS = 80  # Actions 0-79 are offers
ACTION_ACCEPT = 80
ACTION_WALK = 81
NUM_ACTIONS = 82


class SoftPolicy(nn.Module):
    """
    Heuristic policy that accepts any offer.

    - If there's a valid offer: ACCEPT
    - If no offer (must propose): uniform random over 80 offers
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter so this is a valid nn.Module
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """
        Args:
            obs: [batch, 92] observation tensor
                - obs[:, 7] is offer_valid flag (1.0 if there's an offer to accept)
            action_mask: [batch, 82] action mask tensor

        Returns:
            logits: [batch, 82] action logits
            value: [batch] dummy value (zeros)
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Check if there's a valid offer to accept
        # obs[7] is the offer_valid flag
        offer_valid = obs[:, 7] > 0.5  # [batch] boolean

        # Initialize logits
        logits = torch.full((batch_size, NUM_ACTIONS), -1e9, device=device)

        # For envs with valid offer: ACCEPT
        logits[offer_valid, ACTION_ACCEPT] = 0.0

        # For envs without valid offer: uniform over all 80 offers
        no_offer = ~offer_valid
        if no_offer.any():
            # Set uniform logits over offers (actions 0-79)
            logits[no_offer, :NUM_OFFERS] = 0.0

        # Dummy value output
        value = torch.zeros(batch_size, device=device)

        return logits, value
