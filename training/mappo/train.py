#!/usr/bin/env python3
"""
MAPPO (Multi-Agent PPO) training for the CUDA bargaining game.

MAPPO uses Centralized Training with Decentralized Execution (CTDE):
- Each agent has a decentralized actor (only sees own observation)
- Both agents share information through a centralized critic (sees joint state)

Key differences from IDPPO:
1. Centralized critic sees both players' observations
2. Better credit assignment through joint state value estimation
3. Can optionally share parameters between agents
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
    from .policy import MAPPOPolicy, MAPPOHistoryPolicy
except ImportError:
    from policy import MAPPOPolicy, MAPPOHistoryPolicy


def train_mappo(
    num_envs: int = 4096,
    num_iterations: int = 200,
    min_episodes_per_iter: int = 2000,
    lr: float = 1e-3,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-mappo",
    wandb_run_name: str = None,
    log_interval: int = 10,
    share_parameters: bool = False,
):
    """
    Train P1 and P2 policies via MAPPO self-play.

    Uses centralized training with decentralized execution:
    - Each agent's actor only sees its own observation
    - The centralized critic sees both agents' observations
    - Advantage is computed using the centralized value function

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
        share_parameters: Whether to share actor parameters between agents
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "MAPPO",
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
                "share_parameters": share_parameters,
                "centralized_critic": True,
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping logging")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # MAPPO networks with centralized critics
    if share_parameters:
        # Shared actor-critic for both agents
        shared_policy = MAPPOPolicy(OBS_DIM, NUM_ACTIONS).cuda()
        policy_p1 = shared_policy
        policy_p2 = shared_policy
        optimizer = torch.optim.Adam(shared_policy.parameters(), lr=lr)
        optimizer_p1 = optimizer
        optimizer_p2 = optimizer
    else:
        # Separate networks but each has centralized critic
        policy_p1 = MAPPOPolicy(OBS_DIM, NUM_ACTIONS).cuda()
        policy_p2 = MAPPOPolicy(OBS_DIM, NUM_ACTIONS).cuda()
        optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
        optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    reward_history_p1 = []
    reward_history_p2 = []

    print("=" * 60)
    print("MAPPO TRAINING (Centralized Critic)")
    print("=" * 60)
    print(f"Environments: {num_envs}")
    print(f"Iterations: {num_iterations}")
    print(f"Episodes per iteration: {min_episodes_per_iter}")
    print(f"Share parameters: {share_parameters}")
    print(f"Critic: Centralized (joint observation)")
    print()

    for iteration in range(num_iterations):
        # Collect complete episodes for both players
        # Each entry: (obs, joint_obs, action, log_prob, value, reward)
        p1_episodes = []
        p2_episodes = []
        total_games = 0

        while total_games < min_episodes_per_iter:
            obs, info = env.reset()
            action_mask = info['action_mask']

            # Track each player's actions per environment
            # Store: (obs, joint_obs, action, log_prob, value)
            p1_data = [[] for _ in range(num_envs)]
            p2_data = [[] for _ in range(num_envs)]

            # Track opponent's last observation for joint state
            # Initialize with zeros (no opponent obs yet)
            p1_last_obs = torch.zeros((num_envs, OBS_DIM), device='cuda')
            p2_last_obs = torch.zeros((num_envs, OBS_DIM), device='cuda')

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
                    p1_indices = p1_turn.nonzero().squeeze(-1)
                    p1_obs = obs[p1_indices]
                    p1_am = action_mask[p1_indices]
                    # Walk value is at obs index 3 (normalized outside offer)
                    p1_walk = p1_obs[:, 3]

                    # Build joint observation: [own_obs, opponent_obs]
                    p2_obs_for_p1 = p2_last_obs[p1_indices]
                    joint_obs_p1 = torch.cat([p1_obs, p2_obs_for_p1], dim=1)

                    with torch.no_grad():
                        logits = policy_p1.get_action_logits(p1_obs, p1_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        p1_acts = dist.sample()
                        p1_lps = dist.log_prob(p1_acts)
                        p1_vals = policy_p1.get_value(joint_obs_p1)

                    actions[p1_indices] = p1_acts

                    # Update P1's last observation
                    p1_last_obs[p1_indices] = p1_obs

                    for i, idx in enumerate(p1_indices):
                        p1_data[idx.item()].append((
                            p1_obs[i],
                            joint_obs_p1[i],
                            p1_acts[i],
                            p1_lps[i],
                            p1_walk[i],  # Store walk value instead of critic value
                        ))

                # P2 actions
                if p2_turn.any():
                    p2_indices = p2_turn.nonzero().squeeze(-1)
                    p2_obs = obs[p2_indices]
                    p2_am = action_mask[p2_indices]
                    # Walk value is at obs index 3 (normalized outside offer)
                    p2_walk = p2_obs[:, 3]

                    # Build joint observation: [own_obs, opponent_obs]
                    p1_obs_for_p2 = p1_last_obs[p2_indices]
                    joint_obs_p2 = torch.cat([p2_obs, p1_obs_for_p2], dim=1)

                    with torch.no_grad():
                        logits = policy_p2.get_action_logits(p2_obs, p2_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        p2_acts = dist.sample()
                        p2_lps = dist.log_prob(p2_acts)
                        p2_vals = policy_p2.get_value(joint_obs_p2)

                    actions[p2_indices] = p2_acts

                    # Update P2's last observation
                    p2_last_obs[p2_indices] = p2_obs

                    for i, idx in enumerate(p2_indices):
                        p2_data[idx.item()].append((
                            p2_obs[i],
                            joint_obs_p2[i],
                            p2_acts[i],
                            p2_lps[i],
                            p2_walk[i],  # Store walk value instead of critic value
                        ))

                obs, rewards, dones, _, info = env.step(actions)
                action_mask = info['action_mask']

                # Collect completed episodes with terminal rewards
                if dones.any():
                    for idx in dones.nonzero().squeeze(-1):
                        idx_item = idx.item()
                        r1 = rewards[idx, 0].item()
                        r2 = rewards[idx, 1].item()

                        # Assign terminal reward to all actions in episode
                        for o, jo, a, lp, v in p1_data[idx_item]:
                            p1_episodes.append((o, jo, a, lp, v, r1))
                        p1_data[idx_item] = []

                        for o, jo, a, lp, v in p2_data[idx_item]:
                            p2_episodes.append((o, jo, a, lp, v, r2))
                        p2_data[idx_item] = []

                        # Reset opponent observation tracking
                        p1_last_obs[idx] = 0
                        p2_last_obs[idx] = 0

                        total_games += 1

                    active = active & ~dones
                    env.auto_reset()

        # PPO update for P1
        p1_stats = None
        if len(p1_episodes) > 64:
            p1_stats = _mappo_update(policy_p1, optimizer_p1, p1_episodes)
            reward_history_p1.append(p1_stats["reward"])

        # PPO update for P2 (skip if sharing parameters)
        p2_stats = None
        if len(p2_episodes) > 64:
            if share_parameters:
                # Just compute stats, don't update again
                p2_stats = _compute_stats(p2_episodes)
            else:
                p2_stats = _mappo_update(policy_p2, optimizer_p2, p2_episodes)
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
    if share_parameters:
        torch.save(policy_p1.state_dict(), save_path / "mappo_shared.pt")
    else:
        torch.save(policy_p1.state_dict(), save_path / "mappo_p1.pt")
        torch.save(policy_p2.state_dict(), save_path / "mappo_p2.pt")
    print(f"\nModels saved to {save_path}")

    # Save training curves
    _plot_training(reward_history_p1, reward_history_p2, save_path / "mappo_training_curves.png")

    # Log final models and plots to wandb
    if use_wandb and WANDB_AVAILABLE:
        if share_parameters:
            wandb.save(str(save_path / "mappo_shared.pt"))
        else:
            wandb.save(str(save_path / "mappo_p1.pt"))
            wandb.save(str(save_path / "mappo_p2.pt"))
        wandb.log({"training_curves": wandb.Image(str(save_path / "mappo_training_curves.png"))})
        wandb.finish()

    return policy_p1, policy_p2, reward_history_p1, reward_history_p2


def _compute_stats(episodes):
    """Compute statistics without updating."""
    all_rews = torch.tensor([e[5] for e in episodes], device='cuda')
    all_walks = torch.stack([e[4] for e in episodes])
    advantages = all_rews - all_walks
    return {
        "reward": all_rews.mean().item(),
        "advantage": advantages.mean().item(),
        "walk_value": all_walks.mean().item(),
    }


def _mappo_update(policy, optimizer, episodes, ppo_epochs=4, batch_size=512):
    """
    Perform MAPPO PPO update on collected episodes.

    Uses walk value (outside option) as baseline for advantage computation,
    same as IDPPO in the bargaining game.

    Advantage = reward - walk_value

    This makes the policy learn: "did I do better than just walking away?"
    """
    all_obs = torch.stack([e[0] for e in episodes])
    all_joint_obs = torch.stack([e[1] for e in episodes])
    all_acts = torch.stack([e[2] for e in episodes])
    all_lps = torch.stack([e[3] for e in episodes])
    all_walks = torch.stack([e[4] for e in episodes])  # Walk value (outside option)
    all_rews = torch.tensor([e[5] for e in episodes], device='cuda')

    avg_reward = all_rews.mean().item()
    avg_walk = all_walks.mean().item()

    # Compute advantage: reward - walk_value (outside option baseline)
    # This is the same formulation as IDPPO for the bargaining game
    advantages = all_rews - all_walks
    avg_advantage = advantages.mean().item()

    # Normalize advantages (important for stable training)
    # Note: Not normalizing keeps the signal meaningful relative to walk value
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    bs = len(episodes)
    for _ in range(ppo_epochs):
        idx = torch.randperm(bs, device='cuda')
        for s in range(0, bs, batch_size):
            mb = idx[s:min(s + batch_size, bs)]

            # Get action mask from observation
            masks = all_obs[mb, 10:92]

            # Actor loss (decentralized) - PPO clipped surrogate objective
            logits = policy.get_action_logits(all_obs[mb], masks)
            probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
            dist = torch.distributions.Categorical(probs)
            new_lps = dist.log_prob(all_acts[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lps - all_lps[mb].detach())
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[mb]
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (centralized) - train to predict returns
            new_vals = policy.get_value(all_joint_obs[mb])
            value_loss = F.mse_loss(new_vals, all_rews[mb])

            # Combined loss
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy

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
    plt.title('MAPPO Self-Play Training (Centralized Critic)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {save_path}")


# History-based transformer training
TOKEN_DIM = 175  # 92 (obs) + 82 (one-hot action) + 1 (validity flag)
MAX_SEQ_LEN = 6


def train_mappo_history(
    num_envs: int = 4096,
    num_iterations: int = 200,
    min_episodes_per_iter: int = 2000,
    lr: float = 1e-3,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-mappo",
    wandb_run_name: str = None,
    log_interval: int = 10,
):
    """
    Train P1 and P2 policies with MAPPO using history transformer architecture.

    Uses centralized training with decentralized execution:
    - Actor: History-aware transformer sees own game history
    - Critic: Centralized transformer sees both players' histories
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "MAPPO",
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
                "centralized_critic": True,
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping logging")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # History-based MAPPO networks for P1 and P2
    policy_p1 = MAPPOHistoryPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
    policy_p2 = MAPPOHistoryPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    reward_history_p1 = []
    reward_history_p2 = []

    print("=" * 60)
    print("MAPPO TRAINING (History Transformer + Centralized Critic)")
    print("=" * 60)
    print(f"Environments: {num_envs}")
    print(f"Iterations: {num_iterations}")
    print(f"Episodes per iteration: {min_episodes_per_iter}")
    print(f"Token dim: {TOKEN_DIM}, Max seq len: {MAX_SEQ_LEN}")
    print(f"Critic: Centralized (joint history)")
    print()

    for iteration in range(num_iterations):
        # Collect complete episodes for both players
        # Entry: (history, seq_len, history_p1, seq_len_p1, history_p2, seq_len_p2, action, log_prob, value, reward)
        p1_episodes = []
        p2_episodes = []
        total_games = 0

        while total_games < min_episodes_per_iter:
            obs, info = env.reset()
            action_mask = info['action_mask']

            # History buffers for both players
            history_p1 = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
            history_p2 = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
            turn_count_p1 = torch.zeros(num_envs, dtype=torch.long, device='cuda')
            turn_count_p2 = torch.zeros(num_envs, dtype=torch.long, device='cuda')

            # Track each player's data per environment
            p1_data = [[] for _ in range(num_envs)]
            p2_data = [[] for _ in range(num_envs)]
            active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

            for step in range(12):
                if not active.any():
                    break

                current_player = env.get_current_player()
                p1_turn = (current_player == 0) & active
                p2_turn = (current_player == 1) & active

                actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

                # P1 actions
                if p1_turn.any():
                    p1_indices = p1_turn.nonzero().squeeze(-1)

                    # Store obs in current position
                    tc = turn_count_p1[p1_indices].clamp(max=MAX_SEQ_LEN - 1)
                    history_p1[p1_indices, tc, :OBS_DIM] = obs[p1_indices]
                    history_p1[p1_indices, tc, TOKEN_DIM - 1] = 1.0

                    p1_hist = history_p1[p1_indices]
                    p1_seq_lens = turn_count_p1[p1_indices] + 1
                    p1_am = action_mask[p1_indices]
                    # Walk value is at obs index 3 (normalized outside offer)
                    p1_walk = obs[p1_indices, 3]

                    # Get opponent's history for centralized critic
                    p2_hist_for_p1 = history_p2[p1_indices]
                    p2_seq_lens_for_p1 = turn_count_p2[p1_indices].clamp(min=1)

                    with torch.no_grad():
                        logits = policy_p1.get_action_logits(p1_hist, p1_seq_lens, p1_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        p1_acts = dist.sample()
                        p1_lps = dist.log_prob(p1_acts)

                    actions[p1_indices] = p1_acts

                    # Store episode data (with walk value for advantage computation)
                    for i, idx in enumerate(p1_indices):
                        idx_item = idx.item()
                        p1_data[idx_item].append((
                            history_p1[idx].clone(),
                            (turn_count_p1[idx] + 1).clone(),
                            history_p1[idx].clone(),
                            (turn_count_p1[idx] + 1).clone(),
                            history_p2[idx].clone(),
                            turn_count_p2[idx].clamp(min=1).clone(),
                            p1_acts[i].clone(),
                            p1_lps[i].clone(),
                            p1_walk[i].clone(),  # Walk value instead of critic value
                        ))

                    # Update history with action
                    action_onehot = F.one_hot(p1_acts, num_classes=NUM_ACTIONS).float()
                    history_p1[p1_indices, tc, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
                    turn_count_p1[p1_indices] = (turn_count_p1[p1_indices] + 1).clamp(max=MAX_SEQ_LEN - 1)

                # P2 actions
                if p2_turn.any():
                    p2_indices = p2_turn.nonzero().squeeze(-1)

                    # Store obs in current position
                    tc = turn_count_p2[p2_indices].clamp(max=MAX_SEQ_LEN - 1)
                    history_p2[p2_indices, tc, :OBS_DIM] = obs[p2_indices]
                    history_p2[p2_indices, tc, TOKEN_DIM - 1] = 1.0

                    p2_hist = history_p2[p2_indices]
                    p2_seq_lens = turn_count_p2[p2_indices] + 1
                    p2_am = action_mask[p2_indices]
                    # Walk value is at obs index 3 (normalized outside offer)
                    p2_walk = obs[p2_indices, 3]

                    # Get opponent's history for centralized critic
                    p1_hist_for_p2 = history_p1[p2_indices]
                    p1_seq_lens_for_p2 = turn_count_p1[p2_indices].clamp(min=1)

                    with torch.no_grad():
                        logits = policy_p2.get_action_logits(p2_hist, p2_seq_lens, p2_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        p2_acts = dist.sample()
                        p2_lps = dist.log_prob(p2_acts)

                    actions[p2_indices] = p2_acts

                    # Store episode data (with walk value for advantage computation)
                    for i, idx in enumerate(p2_indices):
                        idx_item = idx.item()
                        p2_data[idx_item].append((
                            history_p2[idx].clone(),
                            (turn_count_p2[idx] + 1).clone(),
                            history_p1[idx].clone(),
                            turn_count_p1[idx].clamp(min=1).clone(),
                            history_p2[idx].clone(),
                            (turn_count_p2[idx] + 1).clone(),
                            p2_acts[i].clone(),
                            p2_lps[i].clone(),
                            p2_walk[i].clone(),  # Walk value instead of critic value
                        ))

                    # Update history with action
                    action_onehot = F.one_hot(p2_acts, num_classes=NUM_ACTIONS).float()
                    history_p2[p2_indices, tc, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
                    turn_count_p2[p2_indices] = (turn_count_p2[p2_indices] + 1).clamp(max=MAX_SEQ_LEN - 1)

                obs, rewards, dones, _, info = env.step(actions)
                action_mask = info['action_mask']

                # Collect completed episodes
                if dones.any():
                    done_indices = dones.nonzero().squeeze(-1)
                    for idx in done_indices:
                        idx_item = idx.item()
                        r1 = rewards[idx, 0].item()
                        r2 = rewards[idx, 1].item()

                        for data in p1_data[idx_item]:
                            p1_episodes.append(data + (r1,))
                        p1_data[idx_item] = []

                        for data in p2_data[idx_item]:
                            p2_episodes.append(data + (r2,))
                        p2_data[idx_item] = []

                        # Reset histories
                        history_p1[idx] = 0
                        history_p2[idx] = 0
                        turn_count_p1[idx] = 0
                        turn_count_p2[idx] = 0

                        total_games += 1

                    active = active & ~dones
                    env.auto_reset()

        # PPO updates
        p1_stats = None
        if len(p1_episodes) > 64:
            p1_stats = _mappo_update_history(policy_p1, optimizer_p1, p1_episodes)
            reward_history_p1.append(p1_stats["reward"])

        p2_stats = None
        if len(p2_episodes) > 64:
            p2_stats = _mappo_update_history(policy_p2, optimizer_p2, p2_episodes)
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
    torch.save(policy_p1.state_dict(), save_path / "mappo_history_p1.pt")
    torch.save(policy_p2.state_dict(), save_path / "mappo_history_p2.pt")
    print(f"\nModels saved to {save_path}")

    # Save training curves
    _plot_training(reward_history_p1, reward_history_p2, save_path / "mappo_history_training_curves.png")

    # Log final models and plots to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.save(str(save_path / "mappo_history_p1.pt"))
        wandb.save(str(save_path / "mappo_history_p2.pt"))
        wandb.log({"training_curves": wandb.Image(str(save_path / "mappo_history_training_curves.png"))})
        wandb.finish()

    return policy_p1, policy_p2, reward_history_p1, reward_history_p2


def _mappo_update_history(policy, optimizer, episodes, ppo_epochs=4, batch_size=256):
    """
    Perform MAPPO update for history-based policy.

    Uses walk value (outside option) as baseline for advantage computation,
    same as IDPPO in the bargaining game.

    Advantage = reward - walk_value

    This makes the policy learn: "did I do better than just walking away?"
    """
    # Unpack episode data
    all_hist = torch.stack([e[0] for e in episodes])
    all_seq_lens = torch.stack([e[1] for e in episodes])
    all_hist_p1 = torch.stack([e[2] for e in episodes])
    all_seq_lens_p1 = torch.stack([e[3] for e in episodes])
    all_hist_p2 = torch.stack([e[4] for e in episodes])
    all_seq_lens_p2 = torch.stack([e[5] for e in episodes])
    all_acts = torch.stack([e[6] for e in episodes])
    all_lps = torch.stack([e[7] for e in episodes])
    all_walks = torch.stack([e[8] for e in episodes])  # Walk value (outside option)
    all_rews = torch.tensor([e[9] for e in episodes], device='cuda')

    avg_reward = all_rews.mean().item()
    avg_walk = all_walks.mean().item()

    # Compute advantage: reward - walk_value (outside option baseline)
    # Same formulation as IDPPO for the bargaining game
    advantages = all_rews - all_walks
    avg_advantage = advantages.mean().item()

    # Note: Not normalizing keeps the signal meaningful relative to walk value
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    bs = len(episodes)
    for _ in range(ppo_epochs):
        idx = torch.randperm(bs, device='cuda')
        for s in range(0, bs, batch_size):
            mb = idx[s:min(s + batch_size, bs)]

            # Get action mask from last token's observation
            mb_hist = all_hist[mb]
            mb_seq_lens = all_seq_lens[mb]
            batch_idx = torch.arange(len(mb), device='cuda')
            last_idx = (mb_seq_lens - 1).clamp(min=0)
            last_obs = mb_hist[batch_idx, last_idx, :OBS_DIM]
            masks = last_obs[:, 10:92]

            # Actor loss - PPO clipped surrogate objective
            logits = policy.get_action_logits(mb_hist, mb_seq_lens, masks)
            probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
            dist = torch.distributions.Categorical(probs)
            new_lps = dist.log_prob(all_acts[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lps - all_lps[mb].detach())
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[mb]
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss - train to predict returns
            new_vals = policy.get_value(
                all_hist_p1[mb], all_seq_lens_p1[mb],
                all_hist_p2[mb], all_seq_lens_p2[mb]
            )
            value_loss = F.mse_loss(new_vals, all_rews[mb])

            # Combined loss
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy

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

    parser = argparse.ArgumentParser(description="MAPPO Training for Bargaining Game")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--iterations", type=int, default=200, help="Training iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=2000, help="Min episodes per iteration")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save models")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="bargaining-mappo", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default="MAPPO_test_run", help="W&B run name")
    parser.add_argument("--log-interval", type=int, default=10, help="Log to wandb every N iterations")
    parser.add_argument("--history", action="store_true", help="Use history transformer architecture")
    parser.add_argument("--share-params", action="store_true", help="Share parameters between agents")

    args = parser.parse_args()

    if args.history:
        train_mappo_history(
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
        train_mappo(
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
            share_parameters=args.share_params,
        )