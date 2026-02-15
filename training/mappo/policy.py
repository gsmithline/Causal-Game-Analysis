"""
MAPPO Policy Network with Centralized Critic.

Multi-Agent PPO uses Centralized Training with Decentralized Execution (CTDE):
- Actors (policies): Decentralized, each agent only sees its own observation
- Critic (value): Centralized, sees joint state from both agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAPPOPolicy(nn.Module):
    """
    MAPPO policy network for multi-agent bargaining.

    Architecture:
        - Actor: Takes single agent's observation (92-dim) -> action logits (82-dim)
        - Critic: Takes joint observation (184-dim = 2 * 92) -> value estimate (1-dim)

    The centralized critic sees both players' observations during training,
    enabling better credit assignment in cooperative/competitive settings.
    """

    def __init__(
        self,
        obs_dim: int = 92,
        num_actions: int = 82,
        hidden_dim: int = 256,
        critic_hidden_dim: int = 512,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Decentralized Actor: only sees own observation
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Centralized Critic: sees joint observation (both players)
        # Input: concatenation of both players' observations
        self.critic = nn.Sequential(
            nn.Linear(obs_dim * 2, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1),
        )

    def get_action_logits(
        self, obs: torch.Tensor, action_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get action logits from the decentralized actor.

        Args:
            obs: Single agent's observation of shape (batch, obs_dim)
            action_mask: Optional mask of shape (batch, num_actions)

        Returns:
            logits: Action logits of shape (batch, num_actions)
        """
        logits = self.actor(obs)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        return logits

    def get_value(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate from the centralized critic.

        Args:
            joint_obs: Joint observation of shape (batch, obs_dim * 2)
                       Concatenation of [own_obs, opponent_obs]

        Returns:
            value: State value of shape (batch,)
        """
        return self.critic(joint_obs).squeeze(-1)

    def forward(
        self,
        obs: torch.Tensor,
        joint_obs: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """
        Forward pass for both actor and critic.

        Args:
            obs: Single agent's observation (batch, obs_dim)
            joint_obs: Joint observation (batch, obs_dim * 2)
            action_mask: Optional action validity mask

        Returns:
            logits: Action logits (batch, num_actions)
            value: State value (batch,)
        """
        logits = self.get_action_logits(obs, action_mask)
        value = self.get_value(joint_obs)
        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        joint_obs: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """
        Sample an action from the policy.

        Args:
            obs: Single agent's observation
            joint_obs: Joint observation for value estimation
            action_mask: Optional action validity mask

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value estimate from centralized critic
        """
        logits, value = self.forward(obs, joint_obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class MAPPOCritic(nn.Module):
    """
    Standalone centralized critic for MAPPO.

    This can be used when you want a shared critic for both agents,
    or when the critic architecture differs significantly from the actor.
    """

    def __init__(
        self,
        obs_dim: int = 92,
        hidden_dim: int = 512,
        num_agents: int = 2,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_agents = num_agents

        # Centralized critic takes concatenated observations from all agents
        joint_obs_dim = obs_dim * num_agents

        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for the joint state.

        Args:
            joint_obs: Concatenated observations from all agents
                       Shape: (batch, obs_dim * num_agents)

        Returns:
            value: State value of shape (batch,)
        """
        return self.net(joint_obs).squeeze(-1)


class MAPPOHistoryPolicy(nn.Module):
    """
    MAPPO policy with History Transformer architecture.

    Actor: History-aware transformer (sees own game history)
    Critic: Centralized, sees joint history from both players
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

        # Actor: Decentralized transformer for own history
        self.actor_token_embed = nn.Linear(token_dim, d_model)
        self.actor_pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        actor_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.actor_transformer = nn.TransformerEncoder(actor_encoder_layer, num_layers=num_layers)

        self.actor_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_actions),
        )

        # Critic: Centralized transformer for joint history
        # Takes concatenated histories from both players
        self.critic_token_embed = nn.Linear(token_dim, d_model)
        # Position embedding for 2x max_seq_len (both players' histories)
        self.critic_pos_embed = nn.Parameter(torch.randn(1, max_seq_len * 2, d_model) * 0.02)
        # Player embedding to distinguish P1 vs P2 tokens
        self.player_embed = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

        critic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward * 2,  # Larger critic
            dropout=dropout,
            batch_first=True,
        )
        self.critic_transformer = nn.TransformerEncoder(critic_encoder_layer, num_layers=num_layers)

        self.critic_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward * 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward * 2, 1),
        )

    def get_action_logits(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Get action logits from decentralized actor.

        Args:
            history: Own history tensor (batch, max_seq_len, token_dim)
            seq_lens: Actual sequence lengths (batch,)
            action_mask: Optional action validity mask

        Returns:
            logits: Action logits (batch, num_actions)
        """
        batch_size = history.size(0)
        max_len = history.size(1)

        # Embed tokens
        tokens = self.actor_token_embed(history)
        tokens = tokens + self.actor_pos_embed[:, :max_len]

        # Padding mask
        positions = torch.arange(max_len, device=history.device).unsqueeze(0)
        padding_mask = positions >= seq_lens.unsqueeze(1)

        # Transformer encoding
        encoded = self.actor_transformer(tokens, src_key_padding_mask=padding_mask)

        # Pool using last valid token
        last_idx = (seq_lens - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=history.device)
        pooled = encoded[batch_idx, last_idx]

        # Get logits
        logits = self.actor_head(pooled)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits

    def get_value(
        self,
        history_p1: torch.Tensor,
        seq_lens_p1: torch.Tensor,
        history_p2: torch.Tensor,
        seq_lens_p2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get value estimate from centralized critic.

        Args:
            history_p1: Player 1's history (batch, max_seq_len, token_dim)
            seq_lens_p1: Player 1's sequence lengths (batch,)
            history_p2: Player 2's history (batch, max_seq_len, token_dim)
            seq_lens_p2: Player 2's sequence lengths (batch,)

        Returns:
            value: State value (batch,)
        """
        batch_size = history_p1.size(0)
        max_len = self.max_seq_len

        # Embed tokens for both players
        tokens_p1 = self.critic_token_embed(history_p1) + self.player_embed[:, 0:1]
        tokens_p2 = self.critic_token_embed(history_p2) + self.player_embed[:, 1:2]

        # Concatenate histories: [P1 history, P2 history]
        joint_tokens = torch.cat([tokens_p1, tokens_p2], dim=1)  # (batch, 2*max_len, d_model)

        # Add positional encoding
        joint_tokens = joint_tokens + self.critic_pos_embed[:, :max_len * 2]

        # Create joint padding mask
        positions = torch.arange(max_len, device=history_p1.device).unsqueeze(0)
        mask_p1 = positions >= seq_lens_p1.unsqueeze(1)
        mask_p2 = positions >= seq_lens_p2.unsqueeze(1)
        joint_mask = torch.cat([mask_p1, mask_p2], dim=1)

        # Transformer encoding
        encoded = self.critic_transformer(joint_tokens, src_key_padding_mask=joint_mask)

        # Mean pooling over valid tokens
        valid_mask = ~joint_mask
        valid_counts = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        masked_encoded = encoded * valid_mask.unsqueeze(-1).float()
        pooled = masked_encoded.sum(dim=1) / valid_counts

        # Get value
        value = self.critic_head(pooled).squeeze(-1)
        return value

    def get_action(
        self,
        history: torch.Tensor,
        seq_lens: torch.Tensor,
        history_p1: torch.Tensor,
        seq_lens_p1: torch.Tensor,
        history_p2: torch.Tensor,
        seq_lens_p2: torch.Tensor,
        action_mask: torch.Tensor = None,
    ):
        """
        Sample an action from the policy.

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value from centralized critic
        """
        logits = self.get_action_logits(history, seq_lens, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.get_value(history_p1, seq_lens_p1, history_p2, seq_lens_p2)

        return action, log_prob, value
