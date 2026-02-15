"""
Neural network architectures for NFSP.

Two networks per player:
1. Q-Network: Learns best response via DQN
2. Policy Network: Learns average policy via supervised learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-Network for learning the best response.

    Takes observation and outputs Q-values for all actions.
    Used with DQN-style training.
    """

    def __init__(
        self,
        obs_dim: int = 92,
        num_actions: int = 82,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Forward pass.

        Args:
            obs: Observation tensor (batch, obs_dim)
            action_mask: Valid actions mask (batch, num_actions), 1=valid, 0=invalid

        Returns:
            q_values: Q-values for all actions (batch, num_actions)
        """
        q_values = self.net(obs)

        # Mask invalid actions with large negative value
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, -1e9)

        return q_values

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor, epsilon: float = 0.06):
        """
        Get action using epsilon-greedy over Q-values.

        Args:
            obs: Observation tensor
            action_mask: Valid actions mask
            epsilon: Probability of random action

        Returns:
            action: Selected action
        """
        if torch.rand(1).item() < epsilon:
            # Random action from legal actions
            legal_indices = action_mask.nonzero().squeeze(-1)
            return legal_indices[torch.randint(len(legal_indices), (1,))].squeeze()
        else:
            # Greedy action (argmax of Q-values)
            q_values = self.forward(obs.unsqueeze(0), action_mask.unsqueeze(0))
            return q_values.argmax(dim=-1).squeeze()


class AveragePolicyNetwork(nn.Module):
    """
    Policy network for learning the average policy.

    Trained via supervised learning to match actions taken during
    best-response play, weighted uniformly over history.
    """

    def __init__(
        self,
        obs_dim: int = 92,
        num_actions: int = 82,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Forward pass - returns logits.

        Args:
            obs: Observation tensor (batch, obs_dim)
            action_mask: Valid actions mask (batch, num_actions)

        Returns:
            logits: Action logits (batch, num_actions)
        """
        logits = self.net(obs)

        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits

    def get_action_probs(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        """Get action probabilities."""
        logits = self.forward(obs, action_mask)
        return F.softmax(logits, dim=-1)

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """
        Sample action from the policy.

        Args:
            obs: Observation tensor
            action_mask: Valid actions mask

        Returns:
            action: Sampled action
        """
        probs = self.get_action_probs(obs.unsqueeze(0), action_mask.unsqueeze(0))
        dist = torch.distributions.Categorical(probs)
        return dist.sample().squeeze(0)


class NFSPAgent:
    """
    NFSP Agent combining Q-network and average policy network.

    During play:
    - With probability eta: use best response (epsilon-greedy over Q-values)
    - With probability 1-eta: use average policy
    """

    def __init__(
        self,
        obs_dim: int = 92,
        num_actions: int = 82,
        hidden_dim: int = 256,
        eta: float = 0.1,
        epsilon: float = 0.06,
        device: str = 'cuda',
    ):
        self.eta = eta
        self.epsilon = epsilon
        self.device = device

        self.q_net = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.q_net_target = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.policy_net = AveragePolicyNetwork(obs_dim, num_actions, hidden_dim).to(device)

    def select_action(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """
        Select action using NFSP strategy.

        Returns:
            action: Selected action
            is_best_response: Whether best response was used (for buffer storage)
        """
        use_best_response = torch.rand(1).item() < self.eta

        if use_best_response:
            action = self.q_net.get_action(obs, action_mask, self.epsilon)
            return action, True
        else:
            action = self.policy_net.get_action(obs, action_mask)
            return action, False

    def update_target_network(self, tau: float = 1.0):
        """Soft update of target network."""
        for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class QNetworkHistory(nn.Module):
    """
    History-based Q-Network using Transformer architecture.

    Each token represents a full game state at each turn (175 dims):
        - [0:92]   - Full observation at that turn
        - [92:174] - One-hot encoded action taken (82 dims)
        - [174]    - Turn validity flag (1 = real turn, 0 = padding)
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

        # Learned positional encoding
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

        # Q-value output head
        self.q_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_actions),
        )

    def forward(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """
        Forward pass.

        Args:
            history: History tensor of shape (batch, max_seq_len, token_dim)
            seq_lens: Actual sequence lengths of shape (batch,)
            action_mask: Action mask of shape (batch, num_actions)

        Returns:
            q_values: Q-values for all actions (batch, num_actions)
        """
        batch_size = history.size(0)
        max_len = history.size(1)

        # Embed tokens
        tokens = self.token_embed(history)

        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :max_len]

        # Create padding mask
        positions = torch.arange(max_len, device=history.device).unsqueeze(0)
        padding_mask = positions >= seq_lens.unsqueeze(1)

        # Transformer encoding
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Pool using last valid token
        last_idx = (seq_lens - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=history.device)
        pooled = encoded[batch_idx, last_idx]

        # Q-values
        q_values = self.q_head(pooled)

        # Mask invalid actions
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, -1e9)

        return q_values

    def get_action(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor,
        epsilon: float = 0.06,
    ):
        """Get action using epsilon-greedy over Q-values."""
        if torch.rand(1).item() < epsilon:
            # Random action from legal actions
            legal_indices = action_mask.nonzero().squeeze(-1)
            return legal_indices[torch.randint(len(legal_indices), (1,))].squeeze()
        else:
            # Greedy action (argmax of Q-values)
            q_values = self.forward(
                history.unsqueeze(0),
                seq_lens.unsqueeze(0),
                action_mask.unsqueeze(0)
            )
            return q_values.argmax(dim=-1).squeeze()


class AveragePolicyNetworkHistory(nn.Module):
    """
    History-based Average Policy Network using Transformer architecture.

    Same token structure as QNetworkHistory.
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

        # Learned positional encoding
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

        # Policy output head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_actions),
        )

    def forward(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """
        Forward pass - returns logits.

        Args:
            history: History tensor of shape (batch, max_seq_len, token_dim)
            seq_lens: Actual sequence lengths of shape (batch,)
            action_mask: Action mask of shape (batch, num_actions)

        Returns:
            logits: Action logits (batch, num_actions)
        """
        batch_size = history.size(0)
        max_len = history.size(1)

        # Embed tokens
        tokens = self.token_embed(history)

        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :max_len]

        # Create padding mask
        positions = torch.arange(max_len, device=history.device).unsqueeze(0)
        padding_mask = positions >= seq_lens.unsqueeze(1)

        # Transformer encoding
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Pool using last valid token
        last_idx = (seq_lens - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=history.device)
        pooled = encoded[batch_idx, last_idx]

        # Logits
        logits = self.policy_head(pooled)

        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits

    def get_action_probs(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """Get action probabilities."""
        logits = self.forward(history, seq_lens, action_mask)
        return F.softmax(logits, dim=-1)

    def get_action(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """Sample action from the policy."""
        probs = self.get_action_probs(
            history.unsqueeze(0),
            seq_lens.unsqueeze(0),
            action_mask.unsqueeze(0)
        )
        dist = torch.distributions.Categorical(probs)
        return dist.sample().squeeze(0)


class NFSPAgentHistory:
    """
    NFSP Agent using history-based transformer networks.

    During play:
    - With probability eta: use best response (epsilon-greedy over Q-values)
    - With probability 1-eta: use average policy
    """

    def __init__(
        self,
        token_dim: int = 175,
        num_actions: int = 82,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        eta: float = 0.1,
        epsilon: float = 0.06,
        device: str = 'cuda',
    ):
        self.eta = eta
        self.epsilon = epsilon
        self.device = device

        self.q_net = QNetworkHistory(
            token_dim, num_actions, d_model, nhead, num_layers
        ).to(device)
        self.q_net_target = QNetworkHistory(
            token_dim, num_actions, d_model, nhead, num_layers
        ).to(device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.policy_net = AveragePolicyNetworkHistory(
            token_dim, num_actions, d_model, nhead, num_layers
        ).to(device)

    def select_action(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """
        Select action using NFSP strategy.

        Returns:
            action: Selected action
            is_best_response: Whether best response was used
        """
        use_best_response = torch.rand(1).item() < self.eta

        if use_best_response:
            action = self.q_net.get_action(history, seq_lens, action_mask, self.epsilon)
            return action, True
        else:
            action = self.policy_net.get_action(history, seq_lens, action_mask)
            return action, False

    def update_target_network(self, tau: float = 1.0):
        """Soft update of target network."""
        for target_param, param in zip(
            self.q_net_target.parameters(),
            self.q_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)