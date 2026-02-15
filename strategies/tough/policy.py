"""
Tough heuristic strategy.

Offers opponent a single item of its least-valued type, and insists
rigidly on this offer (always counters with the same offer).

Reference: Section 4.2 of LLM Meta-Game paper.
"""

import torch
import torch.nn as nn

NUM_OFFERS = 80  # Actions 0-79 are offers
ACTION_ACCEPT = 80
ACTION_WALK = 81
NUM_ACTIONS = 82

# Item quantities: [7, 4, 1]
ITEM_QUANTITIES = torch.tensor([7, 4, 1])


class ToughPolicy(nn.Module):
    """
    Heuristic policy that makes an extremely unfavorable offer.

    - Finds the item type the agent values LEAST
    - Offers opponent exactly 1 item of that type (keeping everything else)
    - Always insists on this same offer (never accepts, never walks)

    Important: The offer encoding always represents "items going to P2".
    So P1 and P2 must compute their offers differently:
    - P1: offer[k]=1 gives P2 one item of type k
    - P2: offer = [7,4,1] - [1 item of type k] keeps almost everything for P2
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter so this is a valid nn.Module
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """
        Args:
            obs: [batch, 92] observation tensor
                - obs[:, 0:3] are normalized item values for the 3 types
                - obs[:, 9] is current player (0 or 1)
            action_mask: [batch, 82] action mask tensor

        Returns:
            logits: [batch, 82] action logits
            value: [batch] dummy value (zeros)
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Get item values (obs[0:3])
        item_values = obs[:, 0:3]  # [batch, 3]

        # Get current player (obs[9])
        current_player = obs[:, 9]  # [batch], 0.0 for P1, 1.0 for P2

        # Find least-valued item type for each env
        least_valued_type = torch.argmin(item_values, dim=1)  # [batch]

        # Compute action index based on player position
        # Offer encoding: action = offer[0]*10 + offer[1]*2 + offer[2]
        # offer[i] = number of items of type i going to P2
        #
        # As P1 (give P2 one item of type k):
        #   offer = [0,0,0] with offer[k]=1
        #   type 0: [1,0,0] -> action = 10
        #   type 1: [0,1,0] -> action = 2
        #   type 2: [0,0,1] -> action = 1
        #
        # As P2 (keep almost everything, give P1 one item of type k):
        #   offer = [7,4,1] - [0,0,0 with 1 at position k] = keep all but 1 of type k
        #   type 0: [6,4,1] -> action = 6*10 + 4*2 + 1 = 69
        #   type 1: [7,3,1] -> action = 7*10 + 3*2 + 1 = 77
        #   type 2: [7,4,0] -> action = 7*10 + 4*2 + 0 = 78

        is_p1 = (current_player < 0.5)  # [batch] boolean
        is_p2 = ~is_p1

        action_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

        # P1 actions (give P2 one item)
        action_indices[(is_p1) & (least_valued_type == 0)] = 10  # [1, 0, 0]
        action_indices[(is_p1) & (least_valued_type == 1)] = 2   # [0, 1, 0]
        action_indices[(is_p1) & (least_valued_type == 2)] = 1   # [0, 0, 1]

        # P2 actions (keep almost everything)
        action_indices[(is_p2) & (least_valued_type == 0)] = 69  # [6, 4, 1]
        action_indices[(is_p2) & (least_valued_type == 1)] = 77  # [7, 3, 1]
        action_indices[(is_p2) & (least_valued_type == 2)] = 78  # [7, 4, 0]

        # Create logits that strongly favor the computed action
        logits = torch.full((batch_size, NUM_ACTIONS), -1e9, device=device)

        # Set high logit for each env's tough offer action
        batch_idx = torch.arange(batch_size, device=device)
        logits[batch_idx, action_indices] = 0.0

        # Tough NEVER accepts - if we can't make our rigid offer, we WALK
        # Set WALK as fallback (will be selected when offer is masked out on final turn)
        # -100 is much lower than offer (0.0) but much higher than ACCEPT (-1e9)
        logits[:, ACTION_WALK] = -100.0

        # Dummy value output
        value = torch.zeros(batch_size, device=device)

        return logits, value
