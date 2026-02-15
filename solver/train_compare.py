#!/usr/bin/env python3
"""
Vectorized training comparison for all three architectures:
1. MLP (PolicyNetwork)
2. 5-token Transformer (TransformerPolicyNetwork)
3. History Transformer (HistoryTransformerPolicy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from cuda_bargain import BargainEnv, NUM_ACTIONS, OBS_DIM

from policy import PolicyNetwork, TransformerPolicyNetwork, HistoryTransformerPolicy


def train_vectorized(
    policy_class,
    policy_name: str,
    num_envs: int = 4096,
    num_iterations: int = 200,
    min_episodes_per_iter: int = 2000,
    lr: float = 1e-3,
    seed: int = 42,
    **policy_kwargs,
):
    """
    Vectorized self-play training for any policy architecture.

    Uses tensor operations instead of Python loops for maximum speed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # Determine if this is a history-based policy
    is_history_policy = policy_class == HistoryTransformerPolicy

    # Create policies
    if is_history_policy:
        token_dim = 175  # 92 obs + 82 action + 1 validity
        policy_p1 = policy_class(token_dim=token_dim, num_actions=NUM_ACTIONS, **policy_kwargs).cuda()
        policy_p2 = policy_class(token_dim=token_dim, num_actions=NUM_ACTIONS, **policy_kwargs).cuda()
    else:
        policy_p1 = policy_class(OBS_DIM, NUM_ACTIONS, **policy_kwargs).cuda()
        policy_p2 = policy_class(OBS_DIM, NUM_ACTIONS, **policy_kwargs).cuda()

    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"TRAINING: {policy_name}")
    print("=" * 70)
    print(f"Parameters: {param_count:,}")
    print(f"Environments: {num_envs}")
    print(f"Iterations: {num_iterations}")
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    total_games = 0
    start_time = time.time()

    # Checkpoint tracking
    save_dir = Path(__file__).parent
    best_welfare = -float('inf')
    checkpoint_interval = 500  # Save latest every N iterations

    # Pre-allocate buffers for vectorized collection
    max_steps = 12
    max_buffer = num_envs * max_steps * 2  # Upper bound on data points per iteration

    for iteration in range(num_iterations):
        # Collect episodes
        if is_history_policy:
            p1_data, p2_data, games = _collect_episodes_history(
                env, policy_p1, policy_p2, num_envs, min_episodes_per_iter
            )
        else:
            p1_data, p2_data, games = _collect_episodes_standard(
                env, policy_p1, policy_p2, num_envs, min_episodes_per_iter
            )

        total_games += games

        # PPO updates
        if p1_data[0].size(0) > 64:
            if is_history_policy:
                avg_r1 = _ppo_update_history(policy_p1, optimizer_p1, p1_data)
            else:
                avg_r1 = _ppo_update_standard(policy_p1, optimizer_p1, p1_data)
            reward_history_p1.append(avg_r1)

        if p2_data[0].size(0) > 64:
            if is_history_policy:
                avg_r2 = _ppo_update_history(policy_p2, optimizer_p2, p2_data)
            else:
                avg_r2 = _ppo_update_standard(policy_p2, optimizer_p2, p2_data)
            reward_history_p2.append(avg_r2)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            # Save best checkpoint if welfare improved
            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_dir / f"{policy_name}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_dir / f"{policy_name}_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

        # Save latest checkpoint periodically
        if (iteration + 1) % checkpoint_interval == 0:
            torch.save(policy_p1.state_dict(), save_dir / f"{policy_name}_latest_p1.pt")
            torch.save(policy_p2.state_dict(), save_dir / f"{policy_name}_latest_p2.pt")
            print(f"  -> Saved latest checkpoint at iter {iteration+1}")

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    # Save final model weights
    p1_path = save_dir / f"{policy_name}_final_p1.pt"
    p2_path = save_dir / f"{policy_name}_final_p2.pt"
    torch.save(policy_p1.state_dict(), p1_path)
    torch.save(policy_p2.state_dict(), p2_path)
    print(f"Saved final: {p1_path}, {p2_path}")
    print(f"Best welfare checkpoint: {save_dir / f'{policy_name}_best_p1.pt'}")

    return {
        'name': policy_name,
        'params': param_count,
        'total_games': total_games,
        'time': elapsed,
        'speed': total_games / elapsed,
        'final_p1': final_r1,
        'final_p2': final_r2,
        'welfare': final_r1 + final_r2,
        'history_p1': reward_history_p1,
        'history_p2': reward_history_p2,
    }


def _collect_episodes_standard(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Vectorized episode collection for standard (non-history) policies."""

    # Storage buffers - will collect variable amounts
    p1_obs_list = []
    p1_acts_list = []
    p1_lps_list = []
    p1_rews_list = []

    p2_obs_list = []
    p2_acts_list = []
    p2_lps_list = []
    p2_rews_list = []

    total_games = 0

    while total_games < min_episodes:
        obs, info = env.reset()
        action_mask = info['action_mask']

        # Per-env episode buffers (track indices, not data)
        # We'll store obs/acts as we go and assign rewards at episode end
        p1_ep_obs = [[] for _ in range(num_envs)]
        p1_ep_acts = [[] for _ in range(num_envs)]
        p1_ep_lps = [[] for _ in range(num_envs)]

        p2_ep_obs = [[] for _ in range(num_envs)]
        p2_ep_acts = [[] for _ in range(num_envs)]
        p2_ep_lps = [[] for _ in range(num_envs)]

        active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

        for step in range(12):
            if not active.any():
                break

            current_player = env.get_current_player()
            p1_mask = (current_player == 0) & active
            p2_mask = (current_player == 1) & active

            actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

            # P1 actions - vectorized
            if p1_mask.any():
                p1_idx = p1_mask.nonzero().squeeze(-1)
                p1_obs = obs[p1_idx]
                p1_am = action_mask[p1_idx]

                with torch.no_grad():
                    logits, _ = policy_p1(p1_obs, p1_am)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    p1_acts = dist.sample()
                    p1_lps = dist.log_prob(p1_acts)

                actions[p1_idx] = p1_acts

                # Store per-env (still need loop here but it's fast)
                for i, idx in enumerate(p1_idx.tolist()):
                    p1_ep_obs[idx].append(p1_obs[i])
                    p1_ep_acts[idx].append(p1_acts[i])
                    p1_ep_lps[idx].append(p1_lps[i])

            # P2 actions - vectorized
            if p2_mask.any():
                p2_idx = p2_mask.nonzero().squeeze(-1)
                p2_obs = obs[p2_idx]
                p2_am = action_mask[p2_idx]

                with torch.no_grad():
                    logits, _ = policy_p2(p2_obs, p2_am)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    p2_acts = dist.sample()
                    p2_lps = dist.log_prob(p2_acts)

                actions[p2_idx] = p2_acts

                for i, idx in enumerate(p2_idx.tolist()):
                    p2_ep_obs[idx].append(p2_obs[i])
                    p2_ep_acts[idx].append(p2_acts[i])
                    p2_ep_lps[idx].append(p2_lps[i])

            obs, rewards, dones, _, info = env.step(actions)
            action_mask = info['action_mask']

            # Handle completed episodes
            if dones.any():
                done_idx = dones.nonzero().squeeze(-1)

                for idx in done_idx.tolist():
                    r1 = rewards[idx, 0]
                    r2 = rewards[idx, 1]

                    # Assign terminal reward to all steps
                    for o, a, lp in zip(p1_ep_obs[idx], p1_ep_acts[idx], p1_ep_lps[idx]):
                        p1_obs_list.append(o)
                        p1_acts_list.append(a)
                        p1_lps_list.append(lp)
                        p1_rews_list.append(r1)
                    p1_ep_obs[idx] = []
                    p1_ep_acts[idx] = []
                    p1_ep_lps[idx] = []

                    for o, a, lp in zip(p2_ep_obs[idx], p2_ep_acts[idx], p2_ep_lps[idx]):
                        p2_obs_list.append(o)
                        p2_acts_list.append(a)
                        p2_lps_list.append(lp)
                        p2_rews_list.append(r2)
                    p2_ep_obs[idx] = []
                    p2_ep_acts[idx] = []
                    p2_ep_lps[idx] = []

                    total_games += 1

                active = active & ~dones
                env.auto_reset()

    # Stack into tensors
    p1_data = (
        torch.stack(p1_obs_list) if p1_obs_list else torch.empty(0, OBS_DIM, device='cuda'),
        torch.stack(p1_acts_list) if p1_acts_list else torch.empty(0, dtype=torch.long, device='cuda'),
        torch.stack(p1_lps_list) if p1_lps_list else torch.empty(0, device='cuda'),
        torch.stack(p1_rews_list) if p1_rews_list else torch.empty(0, device='cuda'),
    )

    p2_data = (
        torch.stack(p2_obs_list) if p2_obs_list else torch.empty(0, OBS_DIM, device='cuda'),
        torch.stack(p2_acts_list) if p2_acts_list else torch.empty(0, dtype=torch.long, device='cuda'),
        torch.stack(p2_lps_list) if p2_lps_list else torch.empty(0, device='cuda'),
        torch.stack(p2_rews_list) if p2_rews_list else torch.empty(0, device='cuda'),
    )

    return p1_data, p2_data, total_games


def _collect_episodes_history(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Vectorized episode collection for history-based policy."""

    TOKEN_DIM = 175
    MAX_SEQ_LEN = 6

    p1_hist_list = []
    p1_seq_list = []
    p1_acts_list = []
    p1_lps_list = []
    p1_rews_list = []

    p2_hist_list = []
    p2_seq_list = []
    p2_acts_list = []
    p2_lps_list = []
    p2_rews_list = []

    total_games = 0

    while total_games < min_episodes:
        obs, info = env.reset()
        action_mask = info['action_mask']

        # History buffer
        history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
        turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

        # Per-env episode buffers
        p1_ep_hist = [[] for _ in range(num_envs)]
        p1_ep_seq = [[] for _ in range(num_envs)]
        p1_ep_acts = [[] for _ in range(num_envs)]
        p1_ep_lps = [[] for _ in range(num_envs)]

        p2_ep_hist = [[] for _ in range(num_envs)]
        p2_ep_seq = [[] for _ in range(num_envs)]
        p2_ep_acts = [[] for _ in range(num_envs)]
        p2_ep_lps = [[] for _ in range(num_envs)]

        active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

        for step in range(12):
            if not active.any():
                break

            current_player = env.get_current_player()
            p1_mask = (current_player == 0) & active
            p2_mask = (current_player == 1) & active

            actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

            # Update history with current obs
            active_idx = active.nonzero().squeeze(-1)
            if active_idx.numel() > 0:
                tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
                history[active_idx, tc, :OBS_DIM] = obs[active_idx]
                history[active_idx, tc, TOKEN_DIM - 1] = 1.0

            # P1 actions
            if p1_mask.any():
                p1_idx = p1_mask.nonzero().squeeze(-1)
                p1_hist = history[p1_idx]
                p1_seq_lens = turn_count[p1_idx] + 1
                p1_am = action_mask[p1_idx]

                with torch.no_grad():
                    logits, _ = policy_p1(p1_hist, p1_seq_lens, p1_am)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    p1_acts = dist.sample()
                    p1_lps = dist.log_prob(p1_acts)

                actions[p1_idx] = p1_acts

                for i, idx in enumerate(p1_idx.tolist()):
                    p1_ep_hist[idx].append(history[idx].clone())
                    p1_ep_seq[idx].append(turn_count[idx] + 1)
                    p1_ep_acts[idx].append(p1_acts[i])
                    p1_ep_lps[idx].append(p1_lps[i])

            # P2 actions
            if p2_mask.any():
                p2_idx = p2_mask.nonzero().squeeze(-1)
                p2_hist = history[p2_idx]
                p2_seq_lens = turn_count[p2_idx] + 1
                p2_am = action_mask[p2_idx]

                with torch.no_grad():
                    logits, _ = policy_p2(p2_hist, p2_seq_lens, p2_am)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    p2_acts = dist.sample()
                    p2_lps = dist.log_prob(p2_acts)

                actions[p2_idx] = p2_acts

                for i, idx in enumerate(p2_idx.tolist()):
                    p2_ep_hist[idx].append(history[idx].clone())
                    p2_ep_seq[idx].append(turn_count[idx] + 1)
                    p2_ep_acts[idx].append(p2_acts[i])
                    p2_ep_lps[idx].append(p2_lps[i])

            # Update history with action taken
            if active_idx.numel() > 0:
                tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
                action_onehot = F.one_hot(actions[active_idx], num_classes=NUM_ACTIONS).float()
                history[active_idx, tc, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
                turn_count[active_idx] = (turn_count[active_idx] + 1).clamp(max=MAX_SEQ_LEN - 1)

            obs, rewards, dones, _, info = env.step(actions)
            action_mask = info['action_mask']

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
                    p1_ep_hist[idx] = []
                    p1_ep_seq[idx] = []
                    p1_ep_acts[idx] = []
                    p1_ep_lps[idx] = []

                    for h, s, a, lp in zip(p2_ep_hist[idx], p2_ep_seq[idx], p2_ep_acts[idx], p2_ep_lps[idx]):
                        p2_hist_list.append(h)
                        p2_seq_list.append(s)
                        p2_acts_list.append(a)
                        p2_lps_list.append(lp)
                        p2_rews_list.append(r2)
                    p2_ep_hist[idx] = []
                    p2_ep_seq[idx] = []
                    p2_ep_acts[idx] = []
                    p2_ep_lps[idx] = []

                    history[idx] = 0
                    turn_count[idx] = 0
                    total_games += 1

                active = active & ~dones
                env.auto_reset()

    p1_data = (
        torch.stack(p1_hist_list) if p1_hist_list else torch.empty(0, MAX_SEQ_LEN, TOKEN_DIM, device='cuda'),
        torch.stack(p1_seq_list) if p1_seq_list else torch.empty(0, dtype=torch.long, device='cuda'),
        torch.stack(p1_acts_list) if p1_acts_list else torch.empty(0, dtype=torch.long, device='cuda'),
        torch.stack(p1_lps_list) if p1_lps_list else torch.empty(0, device='cuda'),
        torch.stack(p1_rews_list) if p1_rews_list else torch.empty(0, device='cuda'),
    )

    p2_data = (
        torch.stack(p2_hist_list) if p2_hist_list else torch.empty(0, MAX_SEQ_LEN, TOKEN_DIM, device='cuda'),
        torch.stack(p2_seq_list) if p2_seq_list else torch.empty(0, dtype=torch.long, device='cuda'),
        torch.stack(p2_acts_list) if p2_acts_list else torch.empty(0, dtype=torch.long, device='cuda'),
        torch.stack(p2_lps_list) if p2_lps_list else torch.empty(0, device='cuda'),
        torch.stack(p2_rews_list) if p2_rews_list else torch.empty(0, device='cuda'),
    )

    return p1_data, p2_data, total_games


def _ppo_update_standard(policy, optimizer, data, ppo_epochs=4, batch_size=512):
    """PPO update for standard policies."""
    all_obs, all_acts, all_lps, all_rews = data

    if all_obs.size(0) == 0:
        return 0.0

    avg_reward = all_rews.mean().item()
    all_rews = (all_rews - all_rews.mean()) / (all_rews.std() + 1e-8)

    bs = all_obs.size(0)
    for _ in range(ppo_epochs):
        idx = torch.randperm(bs, device='cuda')
        for s in range(0, bs, batch_size):
            mb = idx[s:min(s + batch_size, bs)]

            masks = all_obs[mb, 10:92]
            logits, _ = policy(all_obs[mb], masks)
            probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
            dist = torch.distributions.Categorical(probs)
            new_lps = dist.log_prob(all_acts[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lps - all_lps[mb])
            surr1 = ratio * all_rews[mb]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * all_rews[mb]
            loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    return avg_reward


def _ppo_update_history(policy, optimizer, data, ppo_epochs=4, batch_size=512):
    """PPO update for history-based policy."""
    all_hist, all_seq, all_acts, all_lps, all_rews = data

    if all_hist.size(0) == 0:
        return 0.0

    avg_reward = all_rews.mean().item()
    all_rews = (all_rews - all_rews.mean()) / (all_rews.std() + 1e-8)

    bs = all_hist.size(0)
    for _ in range(ppo_epochs):
        idx = torch.randperm(bs, device='cuda')
        for s in range(0, bs, batch_size):
            mb = idx[s:min(s + batch_size, bs)]

            mb_hist = all_hist[mb]
            mb_seq = all_seq[mb]

            # Extract action mask from last token's observation
            batch_idx = torch.arange(len(mb), device='cuda')
            last_idx = (mb_seq - 1).clamp(min=0)
            last_obs = mb_hist[batch_idx, last_idx, :OBS_DIM]
            masks = last_obs[:, 10:92]

            logits, _ = policy(mb_hist, mb_seq, masks)
            probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
            dist = torch.distributions.Categorical(probs)
            new_lps = dist.log_prob(all_acts[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lps - all_lps[mb])
            surr1 = ratio * all_rews[mb]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * all_rews[mb]
            loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    return avg_reward


def compare_all(num_iterations=200, num_envs=4096, min_episodes=2000):
    """Train and compare all three architectures."""

    results = []

    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(f"Iterations: {num_iterations}")
    print(f"Environments: {num_envs}")
    print(f"Min episodes per iter: {min_episodes}")
    print()

    # 1. MLP
    print("\n[1/3] Training MLP...")
    r1 = train_vectorized(
        PolicyNetwork, "MLP",
        num_iterations=num_iterations, num_envs=num_envs,
        min_episodes_per_iter=min_episodes
    )
    results.append(r1)

    # 2. 5-token Transformer
    print("\n[2/3] Training 5-token Transformer...")
    r2 = train_vectorized(
        TransformerPolicyNetwork, "5-Token Transformer",
        num_iterations=num_iterations, num_envs=num_envs,
        min_episodes_per_iter=min_episodes
    )
    results.append(r2)

    # 3. History Transformer
    print("\n[3/3] Training History Transformer...")
    r3 = train_vectorized(
        HistoryTransformerPolicy, "History Transformer",
        num_iterations=num_iterations, num_envs=num_envs,
        min_episodes_per_iter=min_episodes
    )
    results.append(r3)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Architecture':<25} {'Params':>10} {'Speed':>10} {'P1':>8} {'P2':>8} {'Welfare':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['params']:>10,} {r['speed']:>9.0f}/s {r['final_p1']:>8.4f} {r['final_p2']:>8.4f} {r['welfare']:>8.4f}")
    print("-" * 70)
    print(f"{'Walk Baseline':<25} {'-':>10} {'-':>10} {'0.5000':>8} {'0.5000':>8} {'1.0000':>8}")

    # Plot comparison
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for r in results:
        plt.plot(r['history_p1'], label=f"{r['name']} P1", linewidth=1.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Walk baseline')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Player 1 Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for r in results:
        plt.plot(r['history_p2'], label=f"{r['name']} P2", linewidth=1.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Walk baseline')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Player 2 Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to architecture_comparison.png")

    return results


if __name__ == "__main__":
    compare_all(num_iterations=200, num_envs=4096, min_episodes=2000)
