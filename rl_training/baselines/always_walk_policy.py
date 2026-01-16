"""Always-walk baseline policy."""

import torch


class AlwaysWalkPolicy:
    """
    Policy that always walks (takes outside offer).

    Very conservative baseline that represents risk-averse behavior.
    """

    ACTION_WALK = 81

    def __init__(self):
        self.name = "always_walk"

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Always return WALK action.

        Args:
            obs: [batch, obs_dim] observations
            action_mask: [batch, num_actions]
            deterministic: Ignored

        Returns:
            actions: [batch] all WALK actions
        """
        batch_size = obs.shape[0]
        return torch.full(
            (batch_size,),
            self.ACTION_WALK,
            dtype=torch.long,
            device=obs.device
        )

    def get_action_probs(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get deterministic action as one-hot."""
        batch_size = obs.shape[0]
        num_actions = action_mask.shape[1]
        probs = torch.zeros(batch_size, num_actions, device=obs.device)
        probs[:, self.ACTION_WALK] = 1.0
        return probs
