#!/usr/bin/env python3
"""
Self-play RL training for the CUDA bargaining game.

Uses a simple PPO implementation with a shared policy for both players.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from cuda_bargain import BargainEnv, NUM_ACTIONS, OBS_DIM


class TransformerPolicyNetwork(nn.Module):
    """Transformer encoder policy network for bargaining game."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Split observation into tokens:
        # Token 0: Player values (3 floats)
        # Token 1: Outside offer (1 float)
        # Token 2: Current offer (3 floats)
        # Token 3: Game state (offer_valid, round, current_player = 3 floats)
        # Token 4: Action mask summary (we'll project the 82-dim mask)
        self.num_tokens = 5

        # Embedding layers for each token type
        self.value_embed = nn.Linear(3, d_model)  # Player values
        self.outside_embed = nn.Linear(1, d_model)  # Outside offer
        self.offer_embed = nn.Linear(3, d_model)  # Current offer
        self.state_embed = nn.Linear(3, d_model)  # Game state
        self.mask_embed = nn.Linear(82, d_model)  # Action mask

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

    def forward(self, obs, action_mask=None):
        batch_size = obs.shape[0]

        # Parse observation into components
        # obs layout: [0-2] values, [3] outside, [4-6] offer, [7] valid, [8] round, [9] player, [10-91] mask
        player_values = obs[:, 0:3]
        outside_offer = obs[:, 3:4]
        current_offer = obs[:, 4:7]
        game_state = obs[:, 7:10]  # valid, round, player
        obs_mask = obs[:, 10:92]  # Action mask from observation

        # Embed each component
        tok_values = self.value_embed(player_values)  # [B, d_model]
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

        # Use CLS-style aggregation (mean pooling over tokens)
        pooled = encoded.mean(dim=1)  # [B, d_model]

        # Output heads
        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)

        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        return logits, value

    def get_action(self, obs, action_mask=None):
        """Sample action from policy."""
        logits, value = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# Alias for compatibility
PolicyNetwork = TransformerPolicyNetwork


class PPOTrainer:
    """Simple PPO trainer for self-play."""

    def __init__(
        self,
        env: BargainEnv,
        policy: PolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 128,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
    ):
        self.env = env
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        self.device = torch.device('cuda')

    def collect_rollout(self):
        """Collect experience from environment."""
        obs_list = []
        action_list = []
        log_prob_list = []
        value_list = []
        reward_list = []
        done_list = []
        mask_list = []
        player_list = []

        obs, info = self.env.reset()
        action_mask = info['action_mask']

        for _ in range(self.rollout_steps):
            current_player = self.env.get_current_player()

            # Store current state
            obs_list.append(obs)
            mask_list.append(action_mask)
            player_list.append(current_player)

            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs, action_mask)

            action_list.append(action)
            log_prob_list.append(log_prob)
            value_list.append(value)

            # Step environment
            obs, rewards, dones, _, info = self.env.step(action)
            action_mask = info['action_mask']

            # Store rewards (for the player who just acted)
            # Use player-specific rewards
            player_rewards = torch.zeros(self.env.num_envs, device=self.device)
            for i in range(self.env.num_envs):
                p = player_list[-1][i].item()
                player_rewards[i] = rewards[i, p]

            reward_list.append(player_rewards)
            done_list.append(dones.float())

            # Auto-reset done environments
            if dones.any():
                self.env.auto_reset()

        # Get final value for bootstrapping
        with torch.no_grad():
            _, final_value = self.policy.forward(obs, action_mask)

        # Convert to tensors
        obs_batch = torch.stack(obs_list)
        action_batch = torch.stack(action_list)
        log_prob_batch = torch.stack(log_prob_list)
        value_batch = torch.stack(value_list)
        reward_batch = torch.stack(reward_list)
        done_batch = torch.stack(done_list)
        mask_batch = torch.stack(mask_list)

        # Compute advantages with GAE
        advantages = torch.zeros_like(reward_batch)
        returns = torch.zeros_like(reward_batch)
        gae = 0

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_value = final_value
            else:
                next_value = value_batch[t + 1]

            delta = reward_batch[t] + self.gamma * next_value * (1 - done_batch[t]) - value_batch[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - done_batch[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + value_batch[t]

        # Flatten batch
        obs_flat = obs_batch.view(-1, OBS_DIM)
        action_flat = action_batch.view(-1)
        log_prob_old = log_prob_batch.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)
        mask_flat = mask_batch.view(-1, NUM_ACTIONS)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        return obs_flat, action_flat, log_prob_old, returns_flat, advantages_flat, mask_flat

    def update(self, obs, actions, log_probs_old, returns, advantages, masks):
        """PPO update step."""
        batch_size = obs.shape[0]
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.ppo_epochs):
            # Random permutation for minibatches
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_masks = masks[mb_indices]

                # Forward pass
                logits, values = self.policy(mb_obs, mb_masks)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Policy loss (clipped PPO)
                ratio = torch.exp(log_probs - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        num_updates = self.ppo_epochs * (batch_size // self.minibatch_size)
        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }


def train_selfplay(
    num_envs: int = 4096,
    num_iterations: int = 500,
    rollout_steps: int = 64,
    seed: int = 42,
):
    """Train agent with self-play."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # Create policy
    policy = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        rollout_steps=rollout_steps,
        lr=3e-4,
        ppo_epochs=4,
        minibatch_size=512,
    )

    # Tracking
    reward_history_p1 = []
    reward_history_p2 = []
    episode_rewards_p1 = deque(maxlen=1000)
    episode_rewards_p2 = deque(maxlen=1000)

    print("=" * 60)
    print("SELF-PLAY RL TRAINING")
    print("=" * 60)
    print(f"Environments: {num_envs:,}")
    print(f"Iterations: {num_iterations}")
    print(f"Rollout steps: {rollout_steps}")
    print()

    for iteration in range(num_iterations):
        # Collect rollout and track rewards per player
        obs, info = env.reset()
        action_mask = info['action_mask']

        iter_rewards_p1 = []
        iter_rewards_p2 = []

        # Collect experience
        obs_list = []
        action_list = []
        log_prob_list = []
        value_list = []
        reward_list = []
        done_list = []
        mask_list = []

        for _ in range(rollout_steps):
            current_player = env.get_current_player()

            obs_list.append(obs)
            mask_list.append(action_mask)

            with torch.no_grad():
                action, log_prob, value = policy.get_action(obs, action_mask)

            action_list.append(action)
            log_prob_list.append(log_prob)
            value_list.append(value)

            obs, rewards, dones, _, info = env.step(action)
            action_mask = info['action_mask']

            # Track rewards by player when game ends
            if dones.any():
                done_indices = dones.nonzero().squeeze(-1)
                for idx in done_indices:
                    i = idx.item()
                    episode_rewards_p1.append(rewards[i, 0].item())
                    episode_rewards_p2.append(rewards[i, 1].item())
                env.auto_reset()

            # Reward for acting player
            player_rewards = torch.zeros(num_envs, device='cuda')
            for i in range(num_envs):
                p = current_player[i].item()
                player_rewards[i] = rewards[i, p]

            reward_list.append(player_rewards)
            done_list.append(dones.float())

        # Get final value
        with torch.no_grad():
            _, final_value = policy.forward(obs, action_mask)

        # Stack tensors
        obs_batch = torch.stack(obs_list)
        action_batch = torch.stack(action_list)
        log_prob_batch = torch.stack(log_prob_list)
        value_batch = torch.stack(value_list)
        reward_batch = torch.stack(reward_list)
        done_batch = torch.stack(done_list)
        mask_batch = torch.stack(mask_list)

        # Compute GAE
        advantages = torch.zeros_like(reward_batch)
        returns = torch.zeros_like(reward_batch)
        gae = 0

        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_value = final_value
            else:
                next_value = value_batch[t + 1]

            delta = reward_batch[t] + 0.99 * next_value * (1 - done_batch[t]) - value_batch[t]
            gae = delta + 0.99 * 0.95 * (1 - done_batch[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + value_batch[t]

        # Flatten
        obs_flat = obs_batch.view(-1, OBS_DIM)
        action_flat = action_batch.view(-1)
        log_prob_old = log_prob_batch.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)
        mask_flat = mask_batch.view(-1, NUM_ACTIONS)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # PPO update
        stats = trainer.update(
            obs_flat, action_flat, log_prob_old,
            returns_flat, advantages_flat, mask_flat
        )

        # Record average rewards
        if len(episode_rewards_p1) > 0:
            avg_p1 = np.mean(list(episode_rewards_p1))
            avg_p2 = np.mean(list(episode_rewards_p2))
            reward_history_p1.append(avg_p1)
            reward_history_p2.append(avg_p2)

        # Print progress
        if (iteration + 1) % 50 == 0 or iteration == 0:
            avg_p1 = np.mean(list(episode_rewards_p1)) if episode_rewards_p1 else 0
            avg_p2 = np.mean(list(episode_rewards_p2)) if episode_rewards_p2 else 0
            print(f"Iter {iteration+1:4d} | "
                  f"P1 Reward: {avg_p1:.3f} | "
                  f"P2 Reward: {avg_p2:.3f} | "
                  f"Loss: {stats['loss']:.4f} | "
                  f"Entropy: {stats['entropy']:.3f}")

    return reward_history_p1, reward_history_p2


def plot_rewards(reward_history_p1, reward_history_p2, save_path='selfplay_rewards.png'):
    """Plot average rewards over training."""
    plt.figure(figsize=(12, 6))

    # Smooth the curves
    window = 20
    if len(reward_history_p1) > window:
        smooth_p1 = np.convolve(reward_history_p1, np.ones(window)/window, mode='valid')
        smooth_p2 = np.convolve(reward_history_p2, np.ones(window)/window, mode='valid')
        x = np.arange(window-1, len(reward_history_p1))
    else:
        smooth_p1 = reward_history_p1
        smooth_p2 = reward_history_p2
        x = np.arange(len(reward_history_p1))

    plt.plot(x, smooth_p1, label='Player 1', color='blue', linewidth=2)
    plt.plot(x, smooth_p2, label='Player 2', color='red', linewidth=2)

    # Plot raw data with transparency
    plt.plot(reward_history_p1, alpha=0.3, color='blue', linewidth=0.5)
    plt.plot(reward_history_p2, alpha=0.3, color='red', linewidth=0.5)

    plt.xlabel('Training Iteration', fontsize=12)
    plt.ylabel('Average Normalized Reward', fontsize=12)
    plt.title('Self-Play Training: Average Reward Over Time', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


if __name__ == '__main__':
    # Train
    reward_history_p1, reward_history_p2 = train_selfplay(
        num_envs=4096,
        num_iterations=500,
        rollout_steps=64,
        seed=42,
    )

    # Plot
    plot_rewards(reward_history_p1, reward_history_p2)
