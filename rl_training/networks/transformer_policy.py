"""
Transformer-based policy network for the bargaining game.

Adapted from SGRD train_selfplay.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class TransformerPolicyNetwork(nn.Module):
    """
    Transformer encoder policy network for bargaining game.

    Processes the observation as a sequence of tokens:
        - Token 0: Player values (3 floats)
        - Token 1: Outside offer (1 float)
        - Token 2: Current offer (3 floats)
        - Token 3: Game state (offer_valid, round, current_player = 3 floats)
        - Token 4: Action mask embedding (82 -> d_model)

    Uses mean pooling over tokens for the final representation.
    """

    def __init__(
        self,
        obs_dim: int = 92,
        num_actions: int = 82,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize the Transformer policy network.

        Args:
            obs_dim: Observation dimension (default 92 for bargaining game)
            num_actions: Number of actions (default 82)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.d_model = d_model

        # Number of tokens we split the observation into
        self.num_tokens = 5

        # Embedding layers for each token type
        self.value_embed = nn.Linear(3, d_model)      # Player values
        self.outside_embed = nn.Linear(1, d_model)    # Outside offer
        self.offer_embed = nn.Linear(3, d_model)      # Current offer
        self.state_embed = nn.Linear(3, d_model)      # Game state
        self.mask_embed = nn.Linear(82, d_model)      # Action mask

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
        )

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
        batch_size = obs.shape[0]

        # Parse observation into components
        # obs layout: [0-2] values, [3] outside, [4-6] offer,
        #            [7] valid, [8] round, [9] player, [10-91] mask
        player_values = obs[:, 0:3]
        outside_offer = obs[:, 3:4]
        current_offer = obs[:, 4:7]
        game_state = obs[:, 7:10]    # valid, round, player
        obs_mask = obs[:, 10:92]     # Action mask from observation

        # Embed each component
        tok_values = self.value_embed(player_values)   # [B, d_model]
        tok_outside = self.outside_embed(outside_offer)
        tok_offer = self.offer_embed(current_offer)
        tok_state = self.state_embed(game_state)
        tok_mask = self.mask_embed(obs_mask)

        # Stack into sequence [B, num_tokens, d_model]
        tokens = torch.stack([tok_values, tok_outside, tok_offer, tok_state, tok_mask], dim=1)

        # Add positional encoding
        tokens = tokens + self.pos_embed

        # Transformer encoding
        encoded = self.transformer(tokens)

        # Mean pooling over tokens
        pooled = encoded.mean(dim=1)  # [B, d_model]

        # Output heads
        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)

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


# Alias for compatibility
PolicyNetwork = TransformerPolicyNetwork
