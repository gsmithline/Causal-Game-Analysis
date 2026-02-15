#!/usr/bin/env python3
"""
Self-play training for the CUDA bargaining game.

Trains separate policy networks for Player 1 and Player 2 using PPO
with proper episode-level reward attribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cuda_bargain import BargainEnv, NUM_ACTIONS, OBS_DIM

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from .policy import PolicyNetwork, HistoryTransformerPolicy
except ImportError:
    from policy import PolicyNetwork, HistoryTransformerPolicy


def train_selfplay(
    num_envs: int = 4096,
    num_iterations: int = 200,
    min_episodes_per_iter: int = 2000,
    lr: float = 1e-3,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-ppo",
    wandb_run_name: str = None,
    log_interval: int = 10,
):
    """
    Train P1 and P2 policies via self-play.

    Uses episode-level reward attribution: terminal rewards are assigned
    to all actions taken by a player during that episode.

    Advantage is computed as (reward - walk_value) where walk_value is
    the player's outside option at the time of each action.

    Args:
        num_envs: Number of parallel environments
        num_iterations: Training iterations
        min_episodes_per_iter: Minimum completed episodes before update
        lr: Learning rate
        seed: Random seed
        save_dir: Directory to save models and plots
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        log_interval: How often to log metrics to wandb (every N iterations)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "PPO",
                "architecture": "MLP",
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "min_episodes_per_iter": min_episodes_per_iter,
                "lr": lr,
                "seed": seed,
                "ppo_epochs": 4,
                "batch_size": 512,
                "clip_range": 0.2,
                "entropy_coef": 0.01,
                "advantage": "reward - walk_value",
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping logging")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # Separate networks for P1 and P2
    policy_p1 = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()
    policy_p2 = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()
    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    reward_history_p1 = []
    reward_history_p2 = []

    print("=" * 60)
    print("SELF-PLAY TRAINING")
    print("=" * 60)
    print(f"Environments: {num_envs}")
    print(f"Iterations: {num_iterations}")
    print(f"Episodes per iteration: {min_episodes_per_iter}")
    print(f"Advantage: reward - walk_value")
    print()

    for iteration in range(num_iterations):
        # Collect complete episodes for both players
        # Each entry: (obs, action, log_prob, walk_value, reward)
        p1_episodes = []
        p2_episodes = []
        total_games = 0

        while total_games < min_episodes_per_iter:
            obs, info = env.reset()
            action_mask = info['action_mask']

            # Track each player's actions per environment
            # Store: (obs, action, log_prob, walk_value)
            p1_data = [[] for _ in range(num_envs)]
            p2_data = [[] for _ in range(num_envs)]
            active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

            for step in range(12):  # Max steps per episode
                if not active.any():
                    break

                current_player = env.get_current_player()
                p1_turn = (current_player == 0) & active
                p2_turn = (current_player == 1) & active

                actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

                # P1 actions
                if p1_turn.any():
                    p1_obs = obs[p1_turn]
                    p1_am = action_mask[p1_turn]
                    # Walk value is at obs index 3 (normalized outside offer)
                    p1_walk = p1_obs[:, 3]
                    with torch.no_grad():
                        logits, _ = policy_p1(p1_obs, p1_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        p1_acts = dist.sample()
                        p1_lps = dist.log_prob(p1_acts)
                    actions[p1_turn] = p1_acts

                    for i, idx in enumerate(p1_turn.nonzero().squeeze(-1)):
                        p1_data[idx.item()].append((p1_obs[i], p1_acts[i], p1_lps[i], p1_walk[i]))

                # P2 actions
                if p2_turn.any():
                    p2_obs = obs[p2_turn]
                    p2_am = action_mask[p2_turn]
                    # Walk value is at obs index 3 (normalized outside offer)
                    p2_walk = p2_obs[:, 3]
                    with torch.no_grad():
                        logits, _ = policy_p2(p2_obs, p2_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        p2_acts = dist.sample()
                        p2_lps = dist.log_prob(p2_acts)
                    actions[p2_turn] = p2_acts

                    for i, idx in enumerate(p2_turn.nonzero().squeeze(-1)):
                        p2_data[idx.item()].append((p2_obs[i], p2_acts[i], p2_lps[i], p2_walk[i]))

                obs, rewards, dones, _, info = env.step(actions)
                action_mask = info['action_mask']

                # Collect completed episodes with terminal rewards
                if dones.any():
                    for idx in dones.nonzero().squeeze(-1):
                        idx = idx.item()
                        r1 = rewards[idx, 0].item()
                        r2 = rewards[idx, 1].item()

                        # Assign terminal reward to all actions in episode
                        for o, a, lp, walk in p1_data[idx]:
                            p1_episodes.append((o, a, lp, walk, r1))
                        p1_data[idx] = []

                        for o, a, lp, walk in p2_data[idx]:
                            p2_episodes.append((o, a, lp, walk, r2))
                        p2_data[idx] = []

                        total_games += 1

                    active = active & ~dones
                    env.auto_reset()

        # PPO update for P1
        p1_stats = None
        if len(p1_episodes) > 64:
            p1_stats = _ppo_update(policy_p1, optimizer_p1, p1_episodes)
            reward_history_p1.append(p1_stats["reward"])

        # PPO update for P2
        p2_stats = None
        if len(p2_episodes) > 64:
            p2_stats = _ppo_update(policy_p2, optimizer_p2, p2_episodes)
            reward_history_p2.append(p2_stats["reward"])

        # Log metrics
        r1 = reward_history_p1[-1] if reward_history_p1 else 0
        r2 = reward_history_p2[-1] if reward_history_p2 else 0

        if use_wandb and WANDB_AVAILABLE and (iteration + 1) % log_interval == 0:
            log_dict = {
                "iteration": iteration + 1,
                "p1/reward": r1,
                "p2/reward": r2,
                "p1/episodes": len(p1_episodes),
                "p2/episodes": len(p2_episodes),
                "games_this_iter": total_games,
            }
            if p1_stats:
                log_dict["p1/advantage"] = p1_stats["advantage"]
                log_dict["p1/walk_value"] = p1_stats["walk_value"]
            if p2_stats:
                log_dict["p2/advantage"] = p2_stats["advantage"]
                log_dict["p2/walk_value"] = p2_stats["walk_value"]
            wandb.log(log_dict)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | Games: {total_games}")

    # Save models
    torch.save(policy_p1.state_dict(), save_path / "policy_p1.pt")
    torch.save(policy_p2.state_dict(), save_path / "policy_p2.pt")
    print(f"\nModels saved to {save_path}")

    # Save training curves
    _plot_training(reward_history_p1, reward_history_p2, save_path / "training_curves.png")

    # Log final models and plots to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.save(str(save_path / "policy_p1.pt"))
        wandb.save(str(save_path / "policy_p2.pt"))
        wandb.log({"training_curves": wandb.Image(str(save_path / "training_curves.png"))})
        wandb.finish()

    return policy_p1, policy_p2, reward_history_p1, reward_history_p2


def _ppo_update(policy, optimizer, episodes, ppo_epochs=4, batch_size=512):
    """Perform PPO update on collected episodes.

    Advantage is computed as (reward - walk_value) to use the outside
    option as a baseline.

    Returns:
        dict with avg_reward, avg_advantage, avg_walk_value
    """
    all_obs = torch.stack([e[0] for e in episodes])
    all_acts = torch.stack([e[1] for e in episodes])
    all_lps = torch.stack([e[2] for e in episodes])
    all_walks = torch.stack([e[3] for e in episodes])
    all_rews = torch.tensor([e[4] for e in episodes], device='cuda')

    avg_reward = all_rews.mean().item()
    avg_walk = all_walks.mean().item()

    # Compute advantage: reward - walk_value (outside option baseline)
    advantages = all_rews - all_walks
    avg_advantage = advantages.mean().item()

    # Normalize advantages
    #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    bs = len(episodes)
    for _ in range(ppo_epochs):
        idx = torch.randperm(bs, device='cuda')
        for s in range(0, bs, batch_size):
            mb = idx[s:min(s + batch_size, bs)]

            # Get action mask from observation
            masks = all_obs[mb, 10:92]
            logits, _ = policy(all_obs[mb], masks)
            probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
            dist = torch.distributions.Categorical(probs)
            new_lps = dist.log_prob(all_acts[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lps - all_lps[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[mb]
            loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    return {
        "reward": avg_reward,
        "advantage": avg_advantage,
        "walk_value": avg_walk,
    }


def _plot_training(history_p1, history_p2, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 6))
    plt.plot(history_p1, 'b-', label='Player 1', linewidth=1.5)
    plt.plot(history_p2, 'r-', label='Player 2', linewidth=1.5)
    plt.axhline(y=0.50, color='green', linestyle='--', label='Walk baseline', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Self-Play Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {save_path}")


# History-based transformer training
TOKEN_DIM = 175  # 92 (obs) + 82 (one-hot action) + 1 (validity flag)
MAX_SEQ_LEN = 6


def train_selfplay_history(
    num_envs: int = 4096,
    num_iterations: int = 200,
    min_episodes_per_iter: int = 2000,
    lr: float = 1e-3,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-ppo",
    wandb_run_name: str = None,
    log_interval: int = 10,
):
    """
    Train P1 and P2 policies with history-based transformer architecture.

    Each decision uses full history of the game as a sequence of tokens,
    where each token is (obs, action_taken, validity_flag).

    Advantage is computed as (reward - walk_value) where walk_value is
    the player's outside option at the time of each action.

    Args:
        num_envs: Number of parallel environments
        num_iterations: Training iterations
        min_episodes_per_iter: Minimum completed episodes before update
        lr: Learning rate
        seed: Random seed
        save_dir: Directory to save models and plots
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        log_interval: How often to log metrics to wandb (every N iterations)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "PPO",
                "architecture": "HistoryTransformer",
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "min_episodes_per_iter": min_episodes_per_iter,
                "lr": lr,
                "seed": seed,
                "token_dim": TOKEN_DIM,
                "max_seq_len": MAX_SEQ_LEN,
                "ppo_epochs": 4,
                "batch_size": 512,
                "clip_range": 0.2,
                "entropy_coef": 0.01,
                "advantage": "reward - walk_value",
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping logging")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # History-based transformer networks for P1 and P2
    policy_p1 = HistoryTransformerPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    reward_history_p1 = []
    reward_history_p2 = []

    print("=" * 60)
    print("SELF-PLAY TRAINING (History Transformer)")
    print("=" * 60)
    print(f"Environments: {num_envs}")
    print(f"Iterations: {num_iterations}")
    print(f"Episodes per iteration: {min_episodes_per_iter}")
    print(f"Token dim: {TOKEN_DIM}, Max seq len: {MAX_SEQ_LEN}")
    print(f"Advantage: reward - walk_value")
    print()

    for iteration in range(num_iterations):
        # Collect complete episodes for both players
        # Each entry: (history_snapshot, seq_len, action, log_prob, walk_value, reward)
        p1_episodes = []
        p2_episodes = []
        total_games = 0

        while total_games < min_episodes_per_iter:
            obs, info = env.reset()
            action_mask = info['action_mask']

            # History buffer: (num_envs, max_seq_len, token_dim)
            history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
            turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

            # Track each player's actions per environment
            # Store: (history_snapshot, seq_len, action, log_prob, walk_value)
            p1_data = [[] for _ in range(num_envs)]
            p2_data = [[] for _ in range(num_envs)]
            active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

            for step in range(12):  # Max steps per episode
                if not active.any():
                    break

                current_player = env.get_current_player()
                p1_turn = (current_player == 0) & active
                p2_turn = (current_player == 1) & active

                actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

                # Build current token (obs + zeros for action, will fill after)
                # Store obs in current position
                active_envs = active.nonzero().squeeze(-1)
                if active_envs.numel() > 0:
                    tc = turn_count[active_envs].clamp(max=MAX_SEQ_LEN - 1)
                    history[active_envs, tc, :OBS_DIM] = obs[active_envs]
                    history[active_envs, tc, TOKEN_DIM - 1] = 1.0  # validity flag

                # P1 actions
                if p1_turn.any():
                    p1_indices = p1_turn.nonzero().squeeze(-1)
                    p1_hist = history[p1_indices]
                    p1_seq_lens = turn_count[p1_indices] + 1  # +1 because we include current
                    p1_am = action_mask[p1_indices]
                    # Walk value is at obs index 3 (normalized outside offer)
                    p1_walk = obs[p1_indices, 3]

                    with torch.no_grad():
                        logits, _ = policy_p1(p1_hist, p1_seq_lens, p1_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        p1_acts = dist.sample()
                        p1_lps = dist.log_prob(p1_acts)
                    actions[p1_indices] = p1_acts

                    # Store episode data (clone history snapshot)
                    for i, idx in enumerate(p1_indices):
                        idx_item = idx.item()
                        hist_snap = history[idx].clone()
                        seq_len = (turn_count[idx] + 1).clone()
                        p1_data[idx_item].append((hist_snap, seq_len, p1_acts[i].clone(), p1_lps[i].clone(), p1_walk[i].clone()))

                # P2 actions
                if p2_turn.any():
                    p2_indices = p2_turn.nonzero().squeeze(-1)
                    p2_hist = history[p2_indices]
                    p2_seq_lens = turn_count[p2_indices] + 1
                    p2_am = action_mask[p2_indices]
                    # Walk value is at obs index 3 (normalized outside offer)
                    p2_walk = obs[p2_indices, 3]

                    with torch.no_grad():
                        logits, _ = policy_p2(p2_hist, p2_seq_lens, p2_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        p2_acts = dist.sample()
                        p2_lps = dist.log_prob(p2_acts)
                    actions[p2_indices] = p2_acts

                    for i, idx in enumerate(p2_indices):
                        idx_item = idx.item()
                        hist_snap = history[idx].clone()
                        seq_len = (turn_count[idx] + 1).clone()
                        p2_data[idx_item].append((hist_snap, seq_len, p2_acts[i].clone(), p2_lps[i].clone(), p2_walk[i].clone()))

                # Update history with action taken (one-hot encoding)
                if active_envs.numel() > 0:
                    tc = turn_count[active_envs].clamp(max=MAX_SEQ_LEN - 1)
                    # One-hot encode action into positions 92:174
                    action_onehot = F.one_hot(actions[active_envs], num_classes=NUM_ACTIONS).float()
                    history[active_envs, tc, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
                    # Increment turn count
                    turn_count[active_envs] = (turn_count[active_envs] + 1).clamp(max=MAX_SEQ_LEN - 1)

                obs, rewards, dones, _, info = env.step(actions)
                action_mask = info['action_mask']

                # Collect completed episodes with terminal rewards
                if dones.any():
                    done_indices = dones.nonzero().squeeze(-1)
                    for idx in done_indices:
                        idx_item = idx.item()
                        r1 = rewards[idx, 0].item()
                        r2 = rewards[idx, 1].item()

                        # Assign terminal reward to all actions in episode
                        for hist, seq_len, act, lp, walk in p1_data[idx_item]:
                            p1_episodes.append((hist, seq_len, act, lp, walk, r1))
                        p1_data[idx_item] = []

                        for hist, seq_len, act, lp, walk in p2_data[idx_item]:
                            p2_episodes.append((hist, seq_len, act, lp, walk, r2))
                        p2_data[idx_item] = []

                        # Reset history for this env
                        history[idx] = 0
                        turn_count[idx] = 0

                        total_games += 1

                    active = active & ~dones
                    env.auto_reset()

        # PPO update for P1
        p1_stats = None
        if len(p1_episodes) > 64:
            p1_stats = _ppo_update_history(policy_p1, optimizer_p1, p1_episodes)
            reward_history_p1.append(p1_stats["reward"])

        # PPO update for P2
        p2_stats = None
        if len(p2_episodes) > 64:
            p2_stats = _ppo_update_history(policy_p2, optimizer_p2, p2_episodes)
            reward_history_p2.append(p2_stats["reward"])

        # Log metrics
        r1 = reward_history_p1[-1] if reward_history_p1 else 0
        r2 = reward_history_p2[-1] if reward_history_p2 else 0

        if use_wandb and WANDB_AVAILABLE and (iteration + 1) % log_interval == 0:
            log_dict = {
                "iteration": iteration + 1,
                "p1/reward": r1,
                "p2/reward": r2,
                "p1/episodes": len(p1_episodes),
                "p2/episodes": len(p2_episodes),
                "games_this_iter": total_games,
            }
            if p1_stats:
                log_dict["p1/advantage"] = p1_stats["advantage"]
                log_dict["p1/walk_value"] = p1_stats["walk_value"]
            if p2_stats:
                log_dict["p2/advantage"] = p2_stats["advantage"]
                log_dict["p2/walk_value"] = p2_stats["walk_value"]
            wandb.log(log_dict)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | Games: {total_games}")

    # Save models
    torch.save(policy_p1.state_dict(), save_path / "policy_history_p1.pt")
    torch.save(policy_p2.state_dict(), save_path / "policy_history_p2.pt")
    print(f"\nModels saved to {save_path}")

    # Save training curves
    _plot_training(reward_history_p1, reward_history_p2, save_path / "training_curves_history.png")

    # Log final models and plots to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.save(str(save_path / "policy_history_p1.pt"))
        wandb.save(str(save_path / "policy_history_p2.pt"))
        wandb.log({"training_curves": wandb.Image(str(save_path / "training_curves_history.png"))})
        wandb.finish()

    return policy_p1, policy_p2, reward_history_p1, reward_history_p2


def _ppo_update_history(policy, optimizer, episodes, ppo_epochs=4, batch_size=512):
    """Perform PPO update for history-based policy.

    Advantage is computed as (reward - walk_value) to use the outside
    option as a baseline.

    Returns:
        dict with avg_reward, avg_advantage, avg_walk_value
    """
    # Stack episode data
    all_hist = torch.stack([e[0] for e in episodes])
    all_seq_lens = torch.stack([e[1] for e in episodes])
    all_acts = torch.stack([e[2] for e in episodes])
    all_lps = torch.stack([e[3] for e in episodes])
    all_walks = torch.stack([e[4] for e in episodes])
    all_rews = torch.tensor([e[5] for e in episodes], device='cuda')

    avg_reward = all_rews.mean().item()
    avg_walk = all_walks.mean().item()

    # Compute advantage: reward - walk_value (outside option baseline)
    advantages = all_rews - all_walks
    avg_advantage = advantages.mean().item()

    # Normalize advantages
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    bs = len(episodes)
    for _ in range(ppo_epochs):
        idx = torch.randperm(bs, device='cuda')
        for s in range(0, bs, batch_size):
            mb = idx[s:min(s + batch_size, bs)]

            # Get action mask from the last valid token's observation
            mb_hist = all_hist[mb]
            mb_seq_lens = all_seq_lens[mb]

            # Extract action mask from observation in last token
            # Last token index is seq_len - 1
            batch_idx = torch.arange(len(mb), device='cuda')
            last_idx = (mb_seq_lens - 1).clamp(min=0)
            last_obs = mb_hist[batch_idx, last_idx, :OBS_DIM]
            masks = last_obs[:, 10:92]  # Action mask is at positions 10-91

            logits, _ = policy(mb_hist, mb_seq_lens, masks)
            probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
            dist = torch.distributions.Categorical(probs)
            new_lps = dist.log_prob(all_acts[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lps - all_lps[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[mb]
            loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    return {
        "reward": avg_reward,
        "advantage": avg_advantage,
        "walk_value": avg_walk,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO Self-Play Training for Bargaining Game")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--iterations", type=int, default=200, help="Training iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=2000, help="Min episodes per iteration")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save models")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="bargaining-ppo", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default="PPO_test_run", help="W&B run name")
    parser.add_argument("--log-interval", type=int, default=10_000, help="Log to wandb every N iterations")
    parser.add_argument("--history", action="store_true", help="Use history transformer architecture")

    args = parser.parse_args()

    if args.history:
        train_selfplay_history(
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            min_episodes_per_iter=args.episodes_per_iter,
            lr=args.lr,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
        )
    else:
        train_selfplay(
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            min_episodes_per_iter=args.episodes_per_iter,
            lr=args.lr,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
        )
