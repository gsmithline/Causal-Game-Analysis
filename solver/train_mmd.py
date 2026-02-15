#!/usr/bin/env python3
"""
Magnetic Mirror Descent (MMD) for the bargaining game.

MMD = PPO + KL penalty to a magnet (reference) distribution.

Implements two magnet distributions:
1. Uniform: uniform over all 82 actions
2. Hierarchical: uniform over {walk, accept, offer}, then uniform within offers

Reference: https://arxiv.org/abs/2206.05825
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from cuda_bargain import BargainEnv, NUM_ACTIONS, OBS_DIM

from policy import HistoryTransformerPolicy

# Action indices
NUM_OFFERS = 80  # actions 0-79 are offers
ACTION_ACCEPT = 80
ACTION_WALK = 81
MAX_SEQ_LEN = 6


def create_uniform_magnet(num_actions: int = NUM_ACTIONS) -> torch.Tensor:
    """Uniform distribution over all actions."""
    return torch.ones(num_actions) / num_actions


def create_hierarchical_magnet(num_actions: int = NUM_ACTIONS) -> torch.Tensor:
    """
    Hierarchical uniform distribution:
    - 1/3 probability for walk
    - 1/3 probability for accept
    - 1/3 probability for offer (split uniformly among 80 offers)
    """
    magnet = torch.zeros(num_actions)
    magnet[:NUM_OFFERS] = (1/3) / NUM_OFFERS  # offers: 1/3 total, split among 80
    magnet[ACTION_ACCEPT] = 1/3  # accept
    magnet[ACTION_WALK] = 1/3    # walk
    return magnet


def create_end_magnet(num_actions: int = NUM_ACTIONS) -> torch.Tensor:
    """
    End-focused magnet:
    - 50% probability for "end" (split between walk and accept)
    - 50% probability for offers (split uniformly among 80 offers)
    """
    magnet = torch.zeros(num_actions)
    magnet[:NUM_OFFERS] = 0.5 / NUM_OFFERS  # offers: 50% total, split among 80
    magnet[ACTION_ACCEPT] = 0.25  # accept: 25% (half of end)
    magnet[ACTION_WALK] = 0.25    # walk: 25% (half of end)
    return magnet


def apply_rational_end(actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    If model picks walk or accept, choose the one with higher value.

    - If action is WALK or ACCEPT, pick the better one
    - If action is an offer, keep it as is

    Args:
        actions: (batch,) action indices
        obs: (batch, obs_dim) observations

    Returns:
        Modified actions with rational end choices
    """
    item_quantities = torch.tensor([7.0, 4.0, 1.0], device=obs.device)

    # Get values from obs
    my_values = obs[:, 0:3]  # normalized per-item values
    walk_value = obs[:, 3]   # normalized walk value
    offer_items = obs[:, 4:7]  # offer as fractions
    offer_valid = obs[:, 7]   # whether there's a valid offer

    # Compute offer value (normalized)
    offer_counts = offer_items.clamp(min=0) * item_quantities.unsqueeze(0)
    max_value = (item_quantities.unsqueeze(0) * my_values).sum(dim=1)
    max_value = max_value.clamp(min=0.01)
    offer_value = (offer_counts * my_values).sum(dim=1) / max_value

    # Mask for end actions (walk or accept)
    end_mask = (actions == ACTION_WALK) | (actions == ACTION_ACCEPT)

    # For end actions, pick the better choice
    # If offer_value > walk_value AND offer is valid -> accept
    # Otherwise -> walk
    should_accept = (offer_value > walk_value) & (offer_valid > 0.5)

    new_actions = actions.clone()
    # Where we have an end action, pick rationally
    new_actions[end_mask & should_accept] = ACTION_ACCEPT
    new_actions[end_mask & ~should_accept] = ACTION_WALK

    return new_actions


def train_mmd(
    magnet_type: str = "uniform",
    num_envs: int = 4096,
    num_iterations: int = 200,
    min_episodes_per_iter: int = 2000,
    lr: float = 3e-4,
    xi: float = 0.01,  # Magnet KL penalty coefficient
    clip_ratio: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 512,
    entropy_coef: float = 0.01,
    seed: int = 42,
):
    """
    Train using Magnetic Mirror Descent (PPO + KL penalty to magnet).

    Loss = PPO_loss + xi * KL(π, magnet)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # Create magnet distribution
    if magnet_type == "uniform":
        magnet = create_uniform_magnet().cuda()
    elif magnet_type == "hierarchical":
        magnet = create_hierarchical_magnet().cuda()
    elif magnet_type == "end":
        magnet = create_end_magnet().cuda()
    else:
        raise ValueError(f"Unknown magnet type: {magnet_type}")

    # Use History Transformer policy (same as the 30M training)
    token_dim = 175  # 92 obs + 82 action + 1 validity
    policy_p1 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()

    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"MAGNETIC MIRROR DESCENT (PPO + KL penalty)")
    print(f"Magnet: {magnet_type}")
    print(f"ξ (KL penalty): {xi}")
    print(f"Parameters: {param_count:,}")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    total_games = 0
    start_time = time.time()

    # Checkpoint tracking
    save_dir = Path(__file__).parent
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
        # Collect episodes with history tracking
        p1_data, p2_data, games = _collect_episodes(
            env, policy_p1, policy_p2, num_envs, min_episodes_per_iter
        )
        total_games += games

        # MMD updates (PPO + KL penalty to magnet)
        if p1_data[0].size(0) > 64:
            avg_r1 = _mmd_update(policy_p1, optimizer_p1, p1_data, magnet, xi,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p1.append(avg_r1)

        if p2_data[0].size(0) > 64:
            avg_r2 = _mmd_update(policy_p2, optimizer_p2, p2_data, magnet, xi,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p2.append(avg_r2)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            # Save best checkpoint
            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_dir / f"mmd_{magnet_type}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_dir / f"mmd_{magnet_type}_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    # Save final
    torch.save(policy_p1.state_dict(), save_dir / f"mmd_{magnet_type}_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_dir / f"mmd_{magnet_type}_final_p2.pt")
    print(f"Saved: mmd_{magnet_type}_final_p1.pt, mmd_{magnet_type}_final_p2.pt")

    return {
        'magnet_type': magnet_type,
        'xi': xi,
        'total_games': total_games,
        'time': elapsed,
        'final_p1': final_r1,
        'final_p2': final_r2,
        'welfare': final_r1 + final_r2,
        'best_welfare': best_welfare,
        'history_p1': reward_history_p1,
        'history_p2': reward_history_p2,
    }


def _collect_episodes(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Collect episodes with history tracking for History Transformer."""

    # History buffers: (num_envs, max_seq_len, token_dim)
    token_dim = 175
    history = torch.zeros((num_envs, MAX_SEQ_LEN, token_dim), device='cuda')
    turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

    # Episode data storage
    p1_hist_list, p1_seq_list, p1_acts_list, p1_lps_list, p1_rews_list = [], [], [], [], []
    p2_hist_list, p2_seq_list, p2_acts_list, p2_lps_list, p2_rews_list = [], [], [], [], []

    # Per-env episode buffers
    p1_ep_hist = [[] for _ in range(num_envs)]
    p1_ep_seq = [[] for _ in range(num_envs)]
    p1_ep_acts = [[] for _ in range(num_envs)]
    p1_ep_lps = [[] for _ in range(num_envs)]
    p2_ep_hist = [[] for _ in range(num_envs)]
    p2_ep_seq = [[] for _ in range(num_envs)]
    p2_ep_acts = [[] for _ in range(num_envs)]
    p2_ep_lps = [[] for _ in range(num_envs)]

    obs, info = env.reset()
    action_mask = info['action_mask']
    games_collected = 0

    # Initialize first token
    history[:, 0, :OBS_DIM] = obs
    history[:, 0, 174] = 1.0  # validity flag
    turn_count[:] = 1

    while games_collected < min_episodes:
        current_player = obs[:, 9]
        p1_mask = current_player == 0
        p2_mask = current_player == 1

        actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')
        log_probs = torch.zeros(num_envs, device='cuda')

        # Get actions from policies
        with torch.no_grad():
            if p1_mask.any():
                p1_idx = p1_mask.nonzero().squeeze(-1)
                p1_hist = history[p1_idx]
                p1_seq = turn_count[p1_idx]
                p1_am = action_mask[p1_idx]

                p1_acts, p1_lps, _ = policy_p1.get_action(p1_hist, p1_seq, p1_am)
                actions[p1_idx] = p1_acts
                log_probs[p1_idx] = p1_lps

                # Store for training
                for i, idx in enumerate(p1_idx.tolist()):
                    p1_ep_hist[idx].append(p1_hist[i].clone())
                    p1_ep_seq[idx].append(p1_seq[i].clone())
                    p1_ep_acts[idx].append(p1_acts[i])
                    p1_ep_lps[idx].append(p1_lps[i])

            if p2_mask.any():
                p2_idx = p2_mask.nonzero().squeeze(-1)
                p2_hist = history[p2_idx]
                p2_seq = turn_count[p2_idx]
                p2_am = action_mask[p2_idx]

                p2_acts, p2_lps, _ = policy_p2.get_action(p2_hist, p2_seq, p2_am)
                actions[p2_idx] = p2_acts
                log_probs[p2_idx] = p2_lps

                for i, idx in enumerate(p2_idx.tolist()):
                    p2_ep_hist[idx].append(p2_hist[i].clone())
                    p2_ep_seq[idx].append(p2_seq[i].clone())
                    p2_ep_acts[idx].append(p2_acts[i])
                    p2_ep_lps[idx].append(p2_lps[i])

        # Update history with action taken
        active_idx = (obs[:, 9] >= 0).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t > 0 and t < MAX_SEQ_LEN:
                    # Store action in previous token
                    prev_t = t - 1
                    action_onehot = F.one_hot(actions[idx], NUM_ACTIONS).float()
                    history[idx, prev_t, OBS_DIM:OBS_DIM+NUM_ACTIONS] = action_onehot

            turn_count[active_idx] = (turn_count[active_idx] + 1).clamp(max=MAX_SEQ_LEN - 1)

        obs, rewards, dones, _, info = env.step(actions)
        action_mask = info['action_mask']

        # Update history with new observation
        active_idx = (~dones).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t < MAX_SEQ_LEN:
                    history[idx, t, :OBS_DIM] = obs[idx]
                    history[idx, t, 174] = 1.0

        # Process completed episodes
        if dones.any():
            done_idx = dones.nonzero().squeeze(-1)
            for idx in done_idx.tolist():
                r1 = rewards[idx, 0]
                r2 = rewards[idx, 1]

                # Assign rewards to all steps
                for h, s, a, lp in zip(p1_ep_hist[idx], p1_ep_seq[idx], p1_ep_acts[idx], p1_ep_lps[idx]):
                    p1_hist_list.append(h)
                    p1_seq_list.append(s)
                    p1_acts_list.append(a)
                    p1_lps_list.append(lp)
                    p1_rews_list.append(r1)

                for h, s, a, lp in zip(p2_ep_hist[idx], p2_ep_seq[idx], p2_ep_acts[idx], p2_ep_lps[idx]):
                    p2_hist_list.append(h)
                    p2_seq_list.append(s)
                    p2_acts_list.append(a)
                    p2_lps_list.append(lp)
                    p2_rews_list.append(r2)

                # Reset
                p1_ep_hist[idx] = []
                p1_ep_seq[idx] = []
                p1_ep_acts[idx] = []
                p1_ep_lps[idx] = []
                p2_ep_hist[idx] = []
                p2_ep_seq[idx] = []
                p2_ep_acts[idx] = []
                p2_ep_lps[idx] = []

                # Reset history for this env
                history[idx] = 0
                turn_count[idx] = 1
                history[idx, 0, :OBS_DIM] = obs[idx]
                history[idx, 0, 174] = 1.0

                games_collected += 1

    # Stack into tensors
    def stack_data(hist_list, seq_list, acts_list, lps_list, rews_list):
        if not hist_list:
            return (torch.zeros(0, MAX_SEQ_LEN, 175, device='cuda'),
                    torch.zeros(0, dtype=torch.long, device='cuda'),
                    torch.zeros(0, dtype=torch.long, device='cuda'),
                    torch.zeros(0, device='cuda'),
                    torch.zeros(0, device='cuda'))
        return (
            torch.stack(hist_list),
            torch.stack(seq_list),
            torch.stack(acts_list),
            torch.stack(lps_list),
            torch.stack(rews_list),
        )

    p1_data = stack_data(p1_hist_list, p1_seq_list, p1_acts_list, p1_lps_list, p1_rews_list)
    p2_data = stack_data(p2_hist_list, p2_seq_list, p2_acts_list, p2_lps_list, p2_rews_list)

    return p1_data, p2_data, games_collected


def _mmd_update(policy, optimizer, data, magnet, xi, clip_ratio, ppo_epochs, batch_size, entropy_coef):
    """
    MMD update: PPO loss + KL penalty to magnet distribution.

    Loss = -PPO_objective + xi * KL(π, magnet)
    """
    all_hist, all_seq, all_acts, all_old_lps, all_rews = data

    if all_hist.size(0) == 0:
        return 0.0

    avg_reward = all_rews.mean().item()

    # Normalize rewards
    all_rews = (all_rews - all_rews.mean()) / (all_rews.std() + 1e-8)

    for _ in range(ppo_epochs):
        perm = torch.randperm(all_hist.size(0))

        for start in range(0, all_hist.size(0), batch_size):
            end = min(start + batch_size, all_hist.size(0))
            idx = perm[start:end]

            hist = all_hist[idx]
            seq = all_seq[idx]
            acts = all_acts[idx]
            old_lps = all_old_lps[idx]
            rews = all_rews[idx]

            # Get current policy outputs
            logits, values = policy(hist, seq, action_mask=None)

            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)
            new_lps = log_probs.gather(1, acts.unsqueeze(1)).squeeze(1)

            # PPO ratio and clipped objective
            ratio = torch.exp(new_lps - old_lps)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            ppo_loss = -torch.min(ratio * rews, clipped_ratio * rews).mean()

            # Entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # KL penalty to magnet: KL(π || magnet) = Σ π(a) * log(π(a) / magnet(a))
            magnet_expanded = magnet.unsqueeze(0).expand_as(probs)
            kl_to_magnet = (probs * (log_probs - torch.log(magnet_expanded + 1e-10))).sum(dim=-1).mean()

            # Total loss: PPO loss - entropy bonus + KL penalty
            loss = ppo_loss - entropy_coef * entropy + xi * kl_to_magnet

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    return avg_reward


def train_mmd_scheduled(
    magnet_type: str = "uniform",
    num_envs: int = 4096,
    num_iterations: int = 2441,  # ~10M games
    min_episodes_per_iter: int = 2000,
    lr: float = 3e-4,
    xi_base: float = 0.05,  # Base coefficient for schedule
    xi_scale: float = 10_000_000,  # Scale factor (10 million)
    clip_ratio: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 512,
    entropy_coef: float = 0.01,
    seed: int = 42,
):
    """
    Train using Magnetic Mirror Descent with annealing schedule.

    ξ_t = xi_base * sqrt(xi_scale / t)

    Reference: https://arxiv.org/abs/2206.05825
    "For our deep multi-agent reinforcement learning experiments, we implemented MMD
    as a modification to PPO... setting η_t, α_t according to the following schedule:
    η_t = 0.05 * sqrt(10 million / t)"
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # Create magnet distribution
    if magnet_type == "uniform":
        magnet = create_uniform_magnet().cuda()
    elif magnet_type == "hierarchical":
        magnet = create_hierarchical_magnet().cuda()
    elif magnet_type == "end":
        magnet = create_end_magnet().cuda()
    else:
        raise ValueError(f"Unknown magnet type: {magnet_type}")

    # Use History Transformer policy
    token_dim = 175
    policy_p1 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()

    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"MAGNETIC MIRROR DESCENT (Scheduled)")
    print(f"Magnet: {magnet_type}")
    print(f"ξ schedule: {xi_base} * sqrt({xi_scale:,} / t)")
    print(f"Parameters: {param_count:,}")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    total_games = 0
    start_time = time.time()

    # Checkpoint tracking
    save_dir = Path(__file__).parent
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
        # Collect episodes
        p1_data, p2_data, games = _collect_episodes(
            env, policy_p1, policy_p2, num_envs, min_episodes_per_iter
        )
        total_games += games

        # Compute scheduled xi based on total games
        # Use max(1, total_games) to avoid division by zero at start
        xi_current = xi_base * np.sqrt(xi_scale / max(1, total_games))

        # MMD updates with scheduled xi
        if p1_data[0].size(0) > 64:
            avg_r1 = _mmd_update(policy_p1, optimizer_p1, p1_data, magnet, xi_current,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p1.append(avg_r1)

        if p2_data[0].size(0) > 64:
            avg_r2 = _mmd_update(policy_p2, optimizer_p2, p2_data, magnet, xi_current,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p2.append(avg_r2)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | ξ: {xi_current:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            # Save best checkpoint
            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_dir / f"mmd_scheduled_{magnet_type}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_dir / f"mmd_scheduled_{magnet_type}_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    # Save final
    torch.save(policy_p1.state_dict(), save_dir / f"mmd_scheduled_{magnet_type}_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_dir / f"mmd_scheduled_{magnet_type}_final_p2.pt")
    print(f"Saved: mmd_scheduled_{magnet_type}_final_p1.pt, mmd_scheduled_{magnet_type}_final_p2.pt")

    return {
        'magnet_type': magnet_type,
        'xi_base': xi_base,
        'xi_scale': xi_scale,
        'total_games': total_games,
        'time': elapsed,
        'final_p1': final_r1,
        'final_p2': final_r2,
        'welfare': final_r1 + final_r2,
        'best_welfare': best_welfare,
        'history_p1': reward_history_p1,
        'history_p2': reward_history_p2,
    }


def _collect_episodes_rational_end(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Collect episodes with rational end constraint.

    When model picks walk or accept, we choose the one with higher value:
    - If offer_value > walk_value -> accept
    - Otherwise -> walk
    """
    token_dim = 175
    history = torch.zeros((num_envs, MAX_SEQ_LEN, token_dim), device='cuda')
    turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

    p1_hist_list, p1_seq_list, p1_acts_list, p1_lps_list, p1_rews_list = [], [], [], [], []
    p2_hist_list, p2_seq_list, p2_acts_list, p2_lps_list, p2_rews_list = [], [], [], [], []

    p1_ep_hist = [[] for _ in range(num_envs)]
    p1_ep_seq = [[] for _ in range(num_envs)]
    p1_ep_acts = [[] for _ in range(num_envs)]
    p1_ep_lps = [[] for _ in range(num_envs)]
    p2_ep_hist = [[] for _ in range(num_envs)]
    p2_ep_seq = [[] for _ in range(num_envs)]
    p2_ep_acts = [[] for _ in range(num_envs)]
    p2_ep_lps = [[] for _ in range(num_envs)]

    obs, info = env.reset()
    action_mask = info['action_mask']
    games_collected = 0

    history[:, 0, :OBS_DIM] = obs
    history[:, 0, 174] = 1.0
    turn_count[:] = 1

    while games_collected < min_episodes:
        current_player = obs[:, 9]
        p1_mask = current_player == 0
        p2_mask = current_player == 1

        actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')
        log_probs = torch.zeros(num_envs, device='cuda')

        with torch.no_grad():
            if p1_mask.any():
                p1_idx = p1_mask.nonzero().squeeze(-1)
                p1_hist = history[p1_idx]
                p1_seq = turn_count[p1_idx]
                p1_am = action_mask[p1_idx]

                p1_acts, p1_lps, _ = policy_p1.get_action(p1_hist, p1_seq, p1_am)
                actions[p1_idx] = p1_acts
                log_probs[p1_idx] = p1_lps

                for i, idx in enumerate(p1_idx.tolist()):
                    p1_ep_hist[idx].append(p1_hist[i].clone())
                    p1_ep_seq[idx].append(p1_seq[i].clone())
                    p1_ep_acts[idx].append(p1_acts[i])
                    p1_ep_lps[idx].append(p1_lps[i])

            if p2_mask.any():
                p2_idx = p2_mask.nonzero().squeeze(-1)
                p2_hist = history[p2_idx]
                p2_seq = turn_count[p2_idx]
                p2_am = action_mask[p2_idx]

                p2_acts, p2_lps, _ = policy_p2.get_action(p2_hist, p2_seq, p2_am)
                actions[p2_idx] = p2_acts
                log_probs[p2_idx] = p2_lps

                for i, idx in enumerate(p2_idx.tolist()):
                    p2_ep_hist[idx].append(p2_hist[i].clone())
                    p2_ep_seq[idx].append(p2_seq[i].clone())
                    p2_ep_acts[idx].append(p2_acts[i])
                    p2_ep_lps[idx].append(p2_lps[i])

        # Apply rational end constraint
        actions = apply_rational_end(actions, obs)

        # Update history with action taken
        active_idx = (obs[:, 9] >= 0).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t > 0 and t < MAX_SEQ_LEN:
                    prev_t = t - 1
                    action_onehot = F.one_hot(actions[idx], NUM_ACTIONS).float()
                    history[idx, prev_t, OBS_DIM:OBS_DIM+NUM_ACTIONS] = action_onehot

            turn_count[active_idx] = (turn_count[active_idx] + 1).clamp(max=MAX_SEQ_LEN - 1)

        obs, rewards, dones, _, info = env.step(actions)
        action_mask = info['action_mask']

        active_idx = (~dones).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t < MAX_SEQ_LEN:
                    history[idx, t, :OBS_DIM] = obs[idx]
                    history[idx, t, 174] = 1.0

        if dones.any():
            done_idx = dones.nonzero().squeeze(-1)
            for idx in done_idx.tolist():
                r1 = rewards[idx, 0]
                r2 = rewards[idx, 1]

                for h, s, a, lp in zip(p1_ep_hist[idx], p1_ep_seq[idx], p1_ep_acts[idx], p1_ep_lps[idx]):
                    p1_hist_list.append(h)
                    p1_seq_list.append(s)
                    p1_acts_list.append(a)
                    p1_lps_list.append(lp)
                    p1_rews_list.append(r1)

                for h, s, a, lp in zip(p2_ep_hist[idx], p2_ep_seq[idx], p2_ep_acts[idx], p2_ep_lps[idx]):
                    p2_hist_list.append(h)
                    p2_seq_list.append(s)
                    p2_acts_list.append(a)
                    p2_lps_list.append(lp)
                    p2_rews_list.append(r2)

                p1_ep_hist[idx] = []
                p1_ep_seq[idx] = []
                p1_ep_acts[idx] = []
                p1_ep_lps[idx] = []
                p2_ep_hist[idx] = []
                p2_ep_seq[idx] = []
                p2_ep_acts[idx] = []
                p2_ep_lps[idx] = []

                history[idx] = 0
                history[idx, 0, :OBS_DIM] = obs[idx]
                history[idx, 0, 174] = 1.0
                turn_count[idx] = 1

            games_collected += dones.sum().item()

    if not p1_hist_list:
        p1_data = (torch.zeros(0, MAX_SEQ_LEN, 175, device='cuda'),
                   torch.zeros(0, dtype=torch.long, device='cuda'),
                   torch.zeros(0, dtype=torch.long, device='cuda'),
                   torch.zeros(0, device='cuda'),
                   torch.zeros(0, device='cuda'))
    else:
        p1_data = (torch.stack(p1_hist_list),
                   torch.stack(p1_seq_list),
                   torch.stack(p1_acts_list),
                   torch.stack(p1_lps_list),
                   torch.tensor(p1_rews_list, device='cuda'))

    if not p2_hist_list:
        p2_data = (torch.zeros(0, MAX_SEQ_LEN, 175, device='cuda'),
                   torch.zeros(0, dtype=torch.long, device='cuda'),
                   torch.zeros(0, dtype=torch.long, device='cuda'),
                   torch.zeros(0, device='cuda'),
                   torch.zeros(0, device='cuda'))
    else:
        p2_data = (torch.stack(p2_hist_list),
                   torch.stack(p2_seq_list),
                   torch.stack(p2_acts_list),
                   torch.stack(p2_lps_list),
                   torch.tensor(p2_rews_list, device='cuda'))

    return p1_data, p2_data, games_collected


def train_mmd_rational_end(
    num_envs: int = 4096,
    num_iterations: int = 2441,
    min_episodes_per_iter: int = 2000,
    lr: float = 3e-4,
    xi_base: float = 0.01,
    xi_scale: float = 10_000_000,
    clip_ratio: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 512,
    entropy_coef: float = 0.01,
    seed: int = 42,
):
    """
    Train with end magnet and rational end constraint.

    Magnet: 50% end (walk+accept), 50% offers
    Constraint: When model picks walk or accept, choose the one with higher value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # Use end magnet: 50% end, 50% offers
    magnet = create_end_magnet().cuda()

    token_dim = 175
    policy_p1 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()

    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print("MMD WITH RATIONAL END CONSTRAINT")
    print("Magnet: end (50% end, 50% offers)")
    print(f"ξ schedule: {xi_base} * sqrt({xi_scale:,} / t)")
    print("Constraint: walk/accept -> pick higher value")
    print(f"Parameters: {param_count:,}")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    total_games = 0
    start_time = time.time()

    save_dir = Path(__file__).parent
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
        p1_data, p2_data, games = _collect_episodes_rational_end(
            env, policy_p1, policy_p2, num_envs, min_episodes_per_iter
        )
        total_games += games

        xi_current = xi_base * np.sqrt(xi_scale / max(1, total_games))

        if p1_data[0].size(0) > 64:
            avg_r1 = _mmd_update(policy_p1, optimizer_p1, p1_data, magnet, xi_current,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p1.append(avg_r1)

        if p2_data[0].size(0) > 64:
            avg_r2 = _mmd_update(policy_p2, optimizer_p2, p2_data, magnet, xi_current,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p2.append(avg_r2)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | ξ: {xi_current:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_dir / "mmd_rational_end_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_dir / "mmd_rational_end_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    torch.save(policy_p1.state_dict(), save_dir / "mmd_rational_end_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_dir / "mmd_rational_end_final_p2.pt")

    return {
        'magnet_type': 'end',
        'xi_base': xi_base,
        'xi_scale': xi_scale,
        'total_games': total_games,
        'time': elapsed,
        'final_p1': final_r1,
        'final_p2': final_r2,
        'welfare': final_r1 + final_r2,
        'best_welfare': best_welfare,
    }


def _collect_episodes_advantage(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Collect episodes with normalized advantage rewards.

    Advantage = (reward - walk_value) / (max_value - walk_value)
    This makes rewards relative to the outside option.
    """
    token_dim = 175
    history = torch.zeros((num_envs, MAX_SEQ_LEN, token_dim), device='cuda')
    turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

    p1_hist_list, p1_seq_list, p1_acts_list, p1_lps_list, p1_rews_list = [], [], [], [], []
    p2_hist_list, p2_seq_list, p2_acts_list, p2_lps_list, p2_rews_list = [], [], [], [], []

    p1_ep_hist = [[] for _ in range(num_envs)]
    p1_ep_seq = [[] for _ in range(num_envs)]
    p1_ep_acts = [[] for _ in range(num_envs)]
    p1_ep_lps = [[] for _ in range(num_envs)]
    p2_ep_hist = [[] for _ in range(num_envs)]
    p2_ep_seq = [[] for _ in range(num_envs)]
    p2_ep_acts = [[] for _ in range(num_envs)]
    p2_ep_lps = [[] for _ in range(num_envs)]

    # Track outside options for each player in each env
    p1_outside = torch.zeros(num_envs, device='cuda')
    p2_outside = torch.zeros(num_envs, device='cuda')

    obs, info = env.reset()
    action_mask = info['action_mask']
    games_collected = 0

    history[:, 0, :OBS_DIM] = obs
    history[:, 0, 174] = 1.0
    turn_count[:] = 1

    # Capture initial outside options (obs[3] is normalized outside option)
    # P1 starts, so obs[3] is P1's outside option
    p1_outside[:] = obs[:, 3]

    while games_collected < min_episodes:
        current_player = obs[:, 9]
        p1_mask = current_player == 0
        p2_mask = current_player == 1

        # Capture P2's outside option when it's their turn
        if p2_mask.any():
            p2_idx = p2_mask.nonzero().squeeze(-1)
            p2_outside[p2_idx] = obs[p2_idx, 3]

        actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')
        log_probs = torch.zeros(num_envs, device='cuda')

        with torch.no_grad():
            if p1_mask.any():
                p1_idx = p1_mask.nonzero().squeeze(-1)
                p1_hist = history[p1_idx]
                p1_seq = turn_count[p1_idx]
                p1_am = action_mask[p1_idx]

                p1_acts, p1_lps, _ = policy_p1.get_action(p1_hist, p1_seq, p1_am)
                actions[p1_idx] = p1_acts
                log_probs[p1_idx] = p1_lps

                for i, idx in enumerate(p1_idx.tolist()):
                    p1_ep_hist[idx].append(p1_hist[i].clone())
                    p1_ep_seq[idx].append(p1_seq[i].clone())
                    p1_ep_acts[idx].append(p1_acts[i])
                    p1_ep_lps[idx].append(p1_lps[i])

            if p2_mask.any():
                p2_idx = p2_mask.nonzero().squeeze(-1)
                p2_hist = history[p2_idx]
                p2_seq = turn_count[p2_idx]
                p2_am = action_mask[p2_idx]

                p2_acts, p2_lps, _ = policy_p2.get_action(p2_hist, p2_seq, p2_am)
                actions[p2_idx] = p2_acts
                log_probs[p2_idx] = p2_lps

                for i, idx in enumerate(p2_idx.tolist()):
                    p2_ep_hist[idx].append(p2_hist[i].clone())
                    p2_ep_seq[idx].append(p2_seq[i].clone())
                    p2_ep_acts[idx].append(p2_acts[i])
                    p2_ep_lps[idx].append(p2_lps[i])

        active_idx = (obs[:, 9] >= 0).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t > 0 and t < MAX_SEQ_LEN:
                    prev_t = t - 1
                    action_onehot = F.one_hot(actions[idx], NUM_ACTIONS).float()
                    history[idx, prev_t, OBS_DIM:OBS_DIM+NUM_ACTIONS] = action_onehot

            turn_count[active_idx] = (turn_count[active_idx] + 1).clamp(max=MAX_SEQ_LEN - 1)

        obs, rewards, dones, _, info = env.step(actions)
        action_mask = info['action_mask']

        active_idx = (~dones).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t < MAX_SEQ_LEN:
                    history[idx, t, :OBS_DIM] = obs[idx]
                    history[idx, t, 174] = 1.0

        if dones.any():
            done_idx = dones.nonzero().squeeze(-1)
            for idx in done_idx.tolist():
                r1 = rewards[idx, 0]
                r2 = rewards[idx, 1]

                # Compute normalized advantage:
                # advantage = (reward - walk_value) / (1 - walk_value)
                # This is 0 when reward = walk_value, positive when better, negative when worse
                walk1 = p1_outside[idx]
                walk2 = p2_outside[idx]

                # Avoid division by zero when walk_value is close to 1
                denom1 = (1 - walk1).clamp(min=0.01)
                denom2 = (1 - walk2).clamp(min=0.01)

                adv1 = (r1 - walk1) / denom1
                adv2 = (r2 - walk2) / denom2

                for h, s, a, lp in zip(p1_ep_hist[idx], p1_ep_seq[idx], p1_ep_acts[idx], p1_ep_lps[idx]):
                    p1_hist_list.append(h)
                    p1_seq_list.append(s)
                    p1_acts_list.append(a)
                    p1_lps_list.append(lp)
                    p1_rews_list.append(adv1)

                for h, s, a, lp in zip(p2_ep_hist[idx], p2_ep_seq[idx], p2_ep_acts[idx], p2_ep_lps[idx]):
                    p2_hist_list.append(h)
                    p2_seq_list.append(s)
                    p2_acts_list.append(a)
                    p2_lps_list.append(lp)
                    p2_rews_list.append(adv2)

                p1_ep_hist[idx] = []
                p1_ep_seq[idx] = []
                p1_ep_acts[idx] = []
                p1_ep_lps[idx] = []
                p2_ep_hist[idx] = []
                p2_ep_seq[idx] = []
                p2_ep_acts[idx] = []
                p2_ep_lps[idx] = []

                history[idx] = 0
                turn_count[idx] = 1
                history[idx, 0, :OBS_DIM] = obs[idx]
                history[idx, 0, 174] = 1.0

                # Reset outside options for new episode
                p1_outside[idx] = obs[idx, 3]
                p2_outside[idx] = 0  # Will be set when P2's turn comes

                games_collected += 1

    def stack_data(hist_list, seq_list, acts_list, lps_list, rews_list):
        if not hist_list:
            return (torch.zeros(0, MAX_SEQ_LEN, 175, device='cuda'),
                    torch.zeros(0, dtype=torch.long, device='cuda'),
                    torch.zeros(0, dtype=torch.long, device='cuda'),
                    torch.zeros(0, device='cuda'),
                    torch.zeros(0, device='cuda'))
        return (
            torch.stack(hist_list),
            torch.stack(seq_list),
            torch.stack(acts_list),
            torch.stack(lps_list),
            torch.stack(rews_list),
        )

    p1_data = stack_data(p1_hist_list, p1_seq_list, p1_acts_list, p1_lps_list, p1_rews_list)
    p2_data = stack_data(p2_hist_list, p2_seq_list, p2_acts_list, p2_lps_list, p2_rews_list)

    return p1_data, p2_data, games_collected


def train_mmd_advantage(
    magnet_type: str = "hierarchical",
    num_envs: int = 4096,
    num_iterations: int = 2441,
    min_episodes_per_iter: int = 2000,
    lr: float = 3e-4,
    xi_base: float = 0.01,
    xi_scale: float = 10_000_000,
    clip_ratio: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 512,
    entropy_coef: float = 0.01,
    seed: int = 42,
):
    """
    Train using MMD with normalized advantage rewards.

    Advantage = (reward - walk_value) / (max_value - walk_value)
    This makes the reward signal relative to the outside option,
    teaching the model to only accept offers better than walking.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    if magnet_type == "uniform":
        magnet = create_uniform_magnet().cuda()
    elif magnet_type == "hierarchical":
        magnet = create_hierarchical_magnet().cuda()
    else:
        raise ValueError(f"Unknown magnet type: {magnet_type}")

    token_dim = 175
    policy_p1 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()

    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"MMD WITH NORMALIZED ADVANTAGE REWARDS")
    print(f"Magnet: {magnet_type}")
    print(f"ξ schedule: {xi_base} * sqrt({xi_scale:,} / t)")
    print(f"Reward: (value - walk) / (max - walk)")
    print(f"Parameters: {param_count:,}")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    total_games = 0
    start_time = time.time()

    save_dir = Path(__file__).parent
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
        # Use advantage-based episode collection
        p1_data, p2_data, games = _collect_episodes_advantage(
            env, policy_p1, policy_p2, num_envs, min_episodes_per_iter
        )
        total_games += games

        xi_current = xi_base * np.sqrt(xi_scale / max(1, total_games))

        if p1_data[0].size(0) > 64:
            avg_r1 = _mmd_update(policy_p1, optimizer_p1, p1_data, magnet, xi_current,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p1.append(avg_r1)

        if p2_data[0].size(0) > 64:
            avg_r2 = _mmd_update(policy_p2, optimizer_p2, p2_data, magnet, xi_current,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p2.append(avg_r2)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1 adv: {r1:.4f} | P2 adv: {r2:.4f} | ξ: {xi_current:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_dir / f"mmd_advantage_{magnet_type}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_dir / f"mmd_advantage_{magnet_type}_best_p2.pt")
                print(f"  -> New best advantage: {welfare:.4f} (saved)")

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1 adv={final_r1:.4f}, P2 adv={final_r2:.4f}")

    torch.save(policy_p1.state_dict(), save_dir / f"mmd_advantage_{magnet_type}_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_dir / f"mmd_advantage_{magnet_type}_final_p2.pt")
    print(f"Saved: mmd_advantage_{magnet_type}_final_p1.pt, mmd_advantage_{magnet_type}_final_p2.pt")

    return {
        'magnet_type': magnet_type,
        'xi_base': xi_base,
        'xi_scale': xi_scale,
        'total_games': total_games,
        'time': elapsed,
        'final_p1': final_r1,
        'final_p2': final_r2,
        'welfare': final_r1 + final_r2,
        'best_welfare': best_welfare,
        'history_p1': reward_history_p1,
        'history_p2': reward_history_p2,
    }


def compute_rational_mask(obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute a constrained action mask based on rationality rules.

    Only allow actions that are not dominated:
    1. Offers: kept_value > walk_value, and if there's a current offer, kept_value > offer_value
    2. Accept: only if offer_value > walk_value
    3. Walk: always allowed

    Args:
        obs: [batch, 92] observation tensor
        action_mask: [batch, 82] original action mask from environment

    Returns:
        rational_mask: [batch, 82] constrained action mask
    """
    batch_size = obs.shape[0]
    device = obs.device

    # Extract observation components
    # obs[0:3] = normalized item values (value per item type, 0-1 scale)
    # obs[3] = normalized outside option (walk value, 0-1 scale)
    # obs[4:7] = current offer items (-1 if none)
    # obs[7] = offer valid flag

    my_values = obs[:, 0:3]  # [batch, 3]
    walk_value = obs[:, 3:4]  # [batch, 1]
    offer_items = obs[:, 4:7]  # [batch, 3]
    offer_valid = obs[:, 7:8]  # [batch, 1]

    # Item quantities: [7, 4, 1]
    item_quantities = torch.tensor([7.0, 4.0, 1.0], device=device)

    # Compute max possible value (keeping all items) for normalization
    # This puts kept_value and offer_value on same 0-1 scale as walk_value
    max_value = (item_quantities.unsqueeze(0) * my_values).sum(dim=1, keepdim=True)  # [batch, 1]
    max_value = max_value.clamp(min=0.01)  # Avoid division by zero

    # Compute normalized value of current offer (if valid)
    # IMPORTANT: offer_items are FRACTIONS (item_count / max_quantity), not raw counts!
    # Need to multiply by item_quantities to get actual item counts
    offer_counts = offer_items.clamp(min=0) * item_quantities.unsqueeze(0)  # [batch, 3] actual counts
    current_offer_value = (offer_counts * my_values).sum(dim=1, keepdim=True)  # [batch, 1]
    current_offer_value_norm = current_offer_value / max_value  # Normalize to 0-1

    # Initialize rational mask as copy of original
    rational_mask = action_mask.clone()

    # === ACCEPT (action 80) ===
    # Only allow accept if offer_valid AND offer_value > walk_value
    accept_allowed = (offer_valid > 0.5) & (current_offer_value_norm > walk_value)
    rational_mask[:, ACTION_ACCEPT] = rational_mask[:, ACTION_ACCEPT] * accept_allowed.squeeze()

    # === WALK (action 81) ===
    # Always allowed (no change needed)

    # === OFFERS (actions 0-79) ===
    # For each offer action, compute kept value and apply constraints
    for action_idx in range(NUM_OFFERS):
        # Decode offer: action_idx = offer0*10 + offer1*2 + offer2
        offer2 = action_idx % 2
        temp = action_idx // 2
        offer1 = temp % 5
        offer0 = temp // 5

        # Items we give to opponent
        offer_to_opponent = torch.tensor([offer0, offer1, offer2], dtype=torch.float32, device=device)
        # Items we keep
        items_kept = item_quantities - offer_to_opponent  # [3]

        # Value of items we keep (normalized to 0-1 scale)
        kept_value = (items_kept.unsqueeze(0) * my_values).sum(dim=1, keepdim=True)  # [batch, 1]
        kept_value_norm = kept_value / max_value  # Normalize to 0-1

        # Constraint 1: kept_value > walk_value (both on 0-1 scale now)
        constraint1 = kept_value_norm > walk_value

        # Constraint 3: if offer_valid, kept_value > current_offer_value
        # (only counter-offer if we'd keep more than what they're offering us)
        constraint3 = (offer_valid < 0.5) | (kept_value_norm > current_offer_value_norm)

        # Combine constraints
        offer_allowed = constraint1 & constraint3

        rational_mask[:, action_idx] = rational_mask[:, action_idx] * offer_allowed.squeeze()

    # Ensure at least walk is always available (failsafe)
    rational_mask[:, ACTION_WALK] = action_mask[:, ACTION_WALK].clamp(min=0.0)

    return rational_mask


def _collect_episodes_constrained(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Collect episodes with constrained (rational) action masks.

    Only allows dominated strategies:
    - Offers where kept_value > walk_value AND kept_value > current_offer_value
    - Accept only if offer_value > walk_value
    - Walk always available
    """
    token_dim = 175
    history = torch.zeros((num_envs, MAX_SEQ_LEN, token_dim), device='cuda')
    turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

    p1_hist_list, p1_seq_list, p1_acts_list, p1_lps_list, p1_rews_list = [], [], [], [], []
    p2_hist_list, p2_seq_list, p2_acts_list, p2_lps_list, p2_rews_list = [], [], [], [], []

    p1_ep_hist = [[] for _ in range(num_envs)]
    p1_ep_seq = [[] for _ in range(num_envs)]
    p1_ep_acts = [[] for _ in range(num_envs)]
    p1_ep_lps = [[] for _ in range(num_envs)]
    p2_ep_hist = [[] for _ in range(num_envs)]
    p2_ep_seq = [[] for _ in range(num_envs)]
    p2_ep_acts = [[] for _ in range(num_envs)]
    p2_ep_lps = [[] for _ in range(num_envs)]

    obs, info = env.reset()
    action_mask = info['action_mask']
    games_collected = 0

    history[:, 0, :OBS_DIM] = obs
    history[:, 0, 174] = 1.0
    turn_count[:] = 1

    while games_collected < min_episodes:
        current_player = obs[:, 9]
        p1_mask = current_player == 0
        p2_mask = current_player == 1

        actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')
        log_probs = torch.zeros(num_envs, device='cuda')

        with torch.no_grad():
            if p1_mask.any():
                p1_idx = p1_mask.nonzero().squeeze(-1)
                p1_hist = history[p1_idx]
                p1_seq = turn_count[p1_idx]
                p1_obs = obs[p1_idx]
                p1_am = action_mask[p1_idx]

                # Apply rational constraints
                p1_rational_am = compute_rational_mask(p1_obs, p1_am)

                p1_acts, p1_lps, _ = policy_p1.get_action(p1_hist, p1_seq, p1_rational_am)
                actions[p1_idx] = p1_acts
                log_probs[p1_idx] = p1_lps

                for i, idx in enumerate(p1_idx.tolist()):
                    p1_ep_hist[idx].append(p1_hist[i].clone())
                    p1_ep_seq[idx].append(p1_seq[i].clone())
                    p1_ep_acts[idx].append(p1_acts[i])
                    p1_ep_lps[idx].append(p1_lps[i])

            if p2_mask.any():
                p2_idx = p2_mask.nonzero().squeeze(-1)
                p2_hist = history[p2_idx]
                p2_seq = turn_count[p2_idx]
                p2_obs = obs[p2_idx]
                p2_am = action_mask[p2_idx]

                # Apply rational constraints
                p2_rational_am = compute_rational_mask(p2_obs, p2_am)

                p2_acts, p2_lps, _ = policy_p2.get_action(p2_hist, p2_seq, p2_rational_am)
                actions[p2_idx] = p2_acts
                log_probs[p2_idx] = p2_lps

                for i, idx in enumerate(p2_idx.tolist()):
                    p2_ep_hist[idx].append(p2_hist[i].clone())
                    p2_ep_seq[idx].append(p2_seq[i].clone())
                    p2_ep_acts[idx].append(p2_acts[i])
                    p2_ep_lps[idx].append(p2_lps[i])

        active_idx = (obs[:, 9] >= 0).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t > 0 and t < MAX_SEQ_LEN:
                    prev_t = t - 1
                    action_onehot = F.one_hot(actions[idx], NUM_ACTIONS).float()
                    history[idx, prev_t, OBS_DIM:OBS_DIM+NUM_ACTIONS] = action_onehot

            turn_count[active_idx] = (turn_count[active_idx] + 1).clamp(max=MAX_SEQ_LEN - 1)

        obs, rewards, dones, _, info = env.step(actions)
        action_mask = info['action_mask']

        active_idx = (~dones).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t < MAX_SEQ_LEN:
                    history[idx, t, :OBS_DIM] = obs[idx]
                    history[idx, t, 174] = 1.0

        if dones.any():
            done_idx = dones.nonzero().squeeze(-1)
            for idx in done_idx.tolist():
                r1 = rewards[idx, 0]
                r2 = rewards[idx, 1]

                for h, s, a, lp in zip(p1_ep_hist[idx], p1_ep_seq[idx], p1_ep_acts[idx], p1_ep_lps[idx]):
                    p1_hist_list.append(h)
                    p1_seq_list.append(s)
                    p1_acts_list.append(a)
                    p1_lps_list.append(lp)
                    p1_rews_list.append(r1)

                for h, s, a, lp in zip(p2_ep_hist[idx], p2_ep_seq[idx], p2_ep_acts[idx], p2_ep_lps[idx]):
                    p2_hist_list.append(h)
                    p2_seq_list.append(s)
                    p2_acts_list.append(a)
                    p2_lps_list.append(lp)
                    p2_rews_list.append(r2)

                p1_ep_hist[idx] = []
                p1_ep_seq[idx] = []
                p1_ep_acts[idx] = []
                p1_ep_lps[idx] = []
                p2_ep_hist[idx] = []
                p2_ep_seq[idx] = []
                p2_ep_acts[idx] = []
                p2_ep_lps[idx] = []

                history[idx] = 0
                turn_count[idx] = 1
                history[idx, 0, :OBS_DIM] = obs[idx]
                history[idx, 0, 174] = 1.0

                games_collected += 1

    def stack_data(hist_list, seq_list, acts_list, lps_list, rews_list):
        if not hist_list:
            return (torch.zeros(0, MAX_SEQ_LEN, 175, device='cuda'),
                    torch.zeros(0, dtype=torch.long, device='cuda'),
                    torch.zeros(0, dtype=torch.long, device='cuda'),
                    torch.zeros(0, device='cuda'),
                    torch.zeros(0, device='cuda'))
        return (
            torch.stack(hist_list),
            torch.stack(seq_list),
            torch.stack(acts_list),
            torch.stack(lps_list),
            torch.stack(rews_list),
        )

    p1_data = stack_data(p1_hist_list, p1_seq_list, p1_acts_list, p1_lps_list, p1_rews_list)
    p2_data = stack_data(p2_hist_list, p2_seq_list, p2_acts_list, p2_lps_list, p2_rews_list)

    return p1_data, p2_data, games_collected


def train_mmd_constrained(
    magnet_type: str = "uniform",
    num_envs: int = 4096,
    num_iterations: int = 2441,
    min_episodes_per_iter: int = 2000,
    lr: float = 3e-4,
    xi_base: float = 0.01,
    xi_scale: float = 10_000_000,
    clip_ratio: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 512,
    entropy_coef: float = 0.01,
    seed: int = 42,
):
    """
    Train using MMD with constrained (rational) action masking.

    Only allows non-dominated actions:
    1. Offers where kept_value > walk_value
    2. Accept only if offer_value > walk_value
    3. Counter-offers only if kept_value > current_offer_value
    4. Walk always allowed

    This prevents the model from exploring dominated strategies.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    if magnet_type == "uniform":
        magnet = create_uniform_magnet().cuda()
    elif magnet_type == "hierarchical":
        magnet = create_hierarchical_magnet().cuda()
    else:
        raise ValueError(f"Unknown magnet type: {magnet_type}")

    token_dim = 175
    policy_p1 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=token_dim, num_actions=NUM_ACTIONS).cuda()

    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"MMD WITH CONSTRAINED (RATIONAL) ACTIONS")
    print(f"Magnet: {magnet_type}")
    print(f"ξ schedule: {xi_base} * sqrt({xi_scale:,} / t)")
    print(f"Constraints: offer > walk, accept > walk, counter > current")
    print(f"Parameters: {param_count:,}")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    total_games = 0
    start_time = time.time()

    save_dir = Path(__file__).parent
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
        # Use constrained episode collection
        p1_data, p2_data, games = _collect_episodes_constrained(
            env, policy_p1, policy_p2, num_envs, min_episodes_per_iter
        )
        total_games += games

        xi_current = xi_base * np.sqrt(xi_scale / max(1, total_games))

        if p1_data[0].size(0) > 64:
            avg_r1 = _mmd_update(policy_p1, optimizer_p1, p1_data, magnet, xi_current,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p1.append(avg_r1)

        if p2_data[0].size(0) > 64:
            avg_r2 = _mmd_update(policy_p2, optimizer_p2, p2_data, magnet, xi_current,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p2.append(avg_r2)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | ξ: {xi_current:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_dir / f"mmd_constrained_{magnet_type}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_dir / f"mmd_constrained_{magnet_type}_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    torch.save(policy_p1.state_dict(), save_dir / f"mmd_constrained_{magnet_type}_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_dir / f"mmd_constrained_{magnet_type}_final_p2.pt")
    print(f"Saved: mmd_constrained_{magnet_type}_final_p1.pt, mmd_constrained_{magnet_type}_final_p2.pt")

    return {
        'magnet_type': magnet_type,
        'xi_base': xi_base,
        'xi_scale': xi_scale,
        'total_games': total_games,
        'time': elapsed,
        'final_p1': final_r1,
        'final_p2': final_r2,
        'welfare': final_r1 + final_r2,
        'best_welfare': best_welfare,
        'history_p1': reward_history_p1,
        'history_p2': reward_history_p2,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("MMD COMPARISON: Uniform vs Hierarchical Magnet")
    print("=" * 70)

    print("\n>>> Training with UNIFORM magnet...")
    result_uniform = train_mmd(magnet_type="uniform", num_iterations=200, xi=0.01)

    print("\n" + "=" * 70 + "\n")

    print(">>> Training with HIERARCHICAL magnet...")
    result_hier = train_mmd(magnet_type="hierarchical", num_iterations=200, xi=0.01)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Config':<20} {'ξ':>6} {'P1':>8} {'P2':>8} {'Welfare':>10} {'Best':>10}")
    print("-" * 70)
    print(f"{'Uniform':<20} {result_uniform['xi']:>6.3f} {result_uniform['final_p1']:>8.4f} {result_uniform['final_p2']:>8.4f} {result_uniform['welfare']:>10.4f} {result_uniform['best_welfare']:>10.4f}")
    print(f"{'Hierarchical':<20} {result_hier['xi']:>6.3f} {result_hier['final_p1']:>8.4f} {result_hier['final_p2']:>8.4f} {result_hier['welfare']:>10.4f} {result_hier['best_welfare']:>10.4f}")
    print("-" * 70)
    print(f"{'PPO baseline':<20} {'':>6} {0.656:>8.4f} {0.591:>8.4f} {1.247:>10.4f}")
