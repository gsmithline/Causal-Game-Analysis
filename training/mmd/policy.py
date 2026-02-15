"""
Policy network architectures for MMD training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryTransformerPolicy(nn.Module):
    """
    History-based transformer policy network.

    Each token represents a full game state at each turn (175 dims):
        - [0:92]   - Full observation at that turn
        - [92:174] - One-hot encoded action taken (82 dims)
        - [174]    - Turn validity flag (1 = real turn, 0 = padding)

    Variable sequence length (1-6 tokens) based on game progression.
    """

    def __init__(
        self,
        token_dim: int = 175,
        num_actions: int = 82,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 6,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.num_actions = num_actions
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embed = nn.Linear(token_dim, d_model)

        # Learned positional encoding for up to 6 positions
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

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
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """
        Forward pass with variable-length history sequences.

        Args:
            history: History tensor of shape (batch, max_seq_len, token_dim)
            seq_lens: Actual sequence lengths of shape (batch,)
            action_mask: Action mask of shape (batch, num_actions)

        Returns:
            logits: Action logits of shape (batch, num_actions)
            value: State value of shape (batch,)
        """
        batch_size = history.size(0)
        max_len = history.size(1)

        # Embed tokens
        tokens = self.token_embed(history)  # (batch, max_len, d_model)

        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :max_len]

        # Create padding mask: True means ignore this position
        # Shape: (batch, max_len)
        positions = torch.arange(max_len, device=history.device).unsqueeze(0)
        padding_mask = positions >= seq_lens.unsqueeze(1)

        # Transformer encoding
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Pool using last valid token for each sequence
        # Get the index of the last valid token (seq_lens - 1)
        last_idx = (seq_lens - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=history.device)
        pooled = encoded[batch_idx, last_idx]  # (batch, d_model)

        # Output heads
        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)

        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits, value

    def get_action(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """Sample an action from the policy."""
        logits, value = self.forward(history, seq_lens, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
