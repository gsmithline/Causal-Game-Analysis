"""
Policy network architecture for the bargaining game.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    MLP policy network for the bargaining game.

    Architecture:
        - Input: 92-dim observation vector
        - Hidden: 2x256 fully connected layers with ReLU
        - Output: 82-dim action logits + 1-dim value estimate

    The observation space (92 dimensions):
        [0:3]   - Player's item values (normalized 0-1)
        [3]     - Outside offer (normalized by max possible value)
        [4:7]   - Current offer on table (normalized, -1 if no offer)
        [7]     - Offer validity flag (0 or 1)
        [8]     - Current round (normalized 0-1)
        [9]     - Current player (0 or 1)
        [10:92] - Action mask (82 actions)

    The action space (82 actions):
        [0:80]  - Counteroffers: action i = offer[i//10, (i//2)%5, i%2]
                  where offer[a,b,c] means give (a,b,c) items to opponent
        [80]    - Accept current offer
        [81]    - Walk away (take outside option)
    """

    def __init__(self, obs_dim: int = 92, num_actions: int = 82, hidden_dim: int = 256):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head (action logits)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

        # Value head (state value estimate)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Forward pass.

        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            action_mask: Optional mask of shape (batch, num_actions)
                         where 1 = valid action, 0 = invalid

        Returns:
            logits: Action logits of shape (batch, num_actions)
            value: State value of shape (batch,)
        """
        features = self.net(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        # Mask invalid actions with large negative value
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits, value

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Sample an action from the policy.

        Args:
            obs: Observation tensor
            action_mask: Optional action validity mask

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value estimate
        """
        logits, value = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def get_action_probs(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """Get action probabilities (for analysis)."""
        logits, _ = self.forward(obs, action_mask)
        return F.softmax(logits, dim=-1)


class TransformerPolicyNetwork(nn.Module):
    """
    Transformer-based policy network (alternative architecture).

    Tokenizes the observation into semantic components:
        - Token 0: Player values (3 floats)
        - Token 1: Outside offer (1 float)
        - Token 2: Current offer (3 floats)
        - Token 3: Game state (offer_valid, round, current_player)
        - Token 4: Action mask embedding (82 floats)
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
        super().__init__()

        self.d_model = d_model
        self.num_tokens = 5

        # Embedding layers for each token type
        self.value_embed = nn.Linear(3, d_model)
        self.outside_embed = nn.Linear(1, d_model)
        self.offer_embed = nn.Linear(3, d_model)
        self.state_embed = nn.Linear(3, d_model)
        self.mask_embed = nn.Linear(82, d_model)

        # Positional encoding
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

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """Forward pass with tokenized observation."""
        # Parse observation into components
        player_values = obs[:, 0:3]
        outside_offer = obs[:, 3:4]
        current_offer = obs[:, 4:7]
        game_state = obs[:, 7:10]
        obs_mask = obs[:, 10:92]

        # Embed each component
        tokens = torch.stack([
            self.value_embed(player_values),
            self.outside_embed(outside_offer),
            self.offer_embed(current_offer),
            self.state_embed(game_state),
            self.mask_embed(obs_mask),
        ], dim=1)

        # Add positional encoding
        tokens = tokens + self.pos_embed

        # Transformer encoding
        encoded = self.transformer(tokens)

        # Mean pooling
        pooled = encoded.mean(dim=1)

        # Output heads
        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits, value

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """Sample an action from the policy."""
        logits, value = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


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
