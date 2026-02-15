#!/usr/bin/env python3
"""
Magnetic Mirror Descent (MMD) training for the bargaining game.

MMD = PPO + KL penalty to a magnet (reference) distribution.

Implements magnet distributions:
1. Uniform: uniform over all 82 actions
2. Hierarchical: uniform over {walk, accept, offer}, then uniform within offers
3. End: 50% end (walk+accept), 50% offers

Reference: https://arxiv.org/abs/2206.05825
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from cuda_bargain import BargainEnv, NUM_ACTIONS, OBS_DIM
from .policy import HistoryTransformerPolicy

# Action indices
NUM_OFFERS = 80  # actions 0-79 are offers
ACTION_ACCEPT = 80
ACTION_WALK = 81
MAX_SEQ_LEN = 6
TOKEN_DIM = 175  # 92 obs + 82 action + 1 validity


def measure_exploitability(
    policy_p1: nn.Module,
    policy_p2: nn.Module,
    num_envs: int = 4096,
    br_episodes: int = 50000,
    eval_episodes: int = 10000,
    br_iterations: int = 100,
    lr: float = 3e-4,
    seed: int = None,
) -> dict:
    """
    Measure exploitability (NashConv) of current policy pair.

    Exploitability = sum of how much each player could gain by deviating
    to their best response.
    """
    device = 'cuda'

    # Step 1: Evaluate current policy pair
    current_r1, current_r2 = _evaluate_policy_pair(
        policy_p1, policy_p2, num_envs, eval_episodes, seed
    )

    # Step 2: Train P1 best response against fixed P2
    br_p1 = HistoryTransformerPolicy(
        token_dim=TOKEN_DIM,
        num_actions=NUM_ACTIONS,
    ).to(device)

    br_reward_p1 = _train_best_response(
        br_policy=br_p1,
        opponent_policy=policy_p2,
        player_id=0,
        num_envs=num_envs,
        num_episodes=br_episodes,
        num_iterations=br_iterations,
        lr=lr,
        seed=seed,
    )

    # Step 3: Train P2 best response against fixed P1
    br_p2 = HistoryTransformerPolicy(
        token_dim=TOKEN_DIM,
        num_actions=NUM_ACTIONS,
    ).to(device)

    br_reward_p2 = _train_best_response(
        br_policy=br_p2,
        opponent_policy=policy_p1,
        player_id=1,
        num_envs=num_envs,
        num_episodes=br_episodes,
        num_iterations=br_iterations,
        lr=lr,
        seed=seed,
    )

    # Step 4: Compute exploitability
    exploit_p1 = max(0, br_reward_p1 - current_r1)
    exploit_p2 = max(0, br_reward_p2 - current_r2)
    nashconv = exploit_p1 + exploit_p2

    return {
        'nashconv': nashconv,
        'exploit_p1': exploit_p1,
        'exploit_p2': exploit_p2,
        'current_r1': current_r1,
        'current_r2': current_r2,
        'br_reward_p1': br_reward_p1,
        'br_reward_p2': br_reward_p2,
    }


def _evaluate_policy_pair(
    policy_p1: nn.Module,
    policy_p2: nn.Module,
    num_envs: int,
    num_episodes: int,
    seed: int = None,
) -> tuple:
    """Evaluate current policy pair by playing games."""
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed or 12345)

    total_r1 = 0.0
    total_r2 = 0.0
    games_played = 0

    history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
    turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

    obs, info = env.reset()
    action_mask = info['action_mask']

    history[:, 0, :OBS_DIM] = obs
    history[:, 0, 174] = 1.0
    turn_count[:] = 1

    while games_played < num_episodes:
        current_player = obs[:, 9]
        p1_mask = current_player == 0
        p2_mask = current_player == 1

        actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

        with torch.no_grad():
            if p1_mask.any():
                p1_idx = p1_mask.nonzero().squeeze(-1)
                p1_hist = history[p1_idx]
                p1_seq = turn_count[p1_idx]
                p1_am = action_mask[p1_idx]
                logits, _ = policy_p1(p1_hist, p1_seq, p1_am)
                probs = F.softmax(logits, dim=-1)
                p1_acts = torch.distributions.Categorical(probs).sample()
                actions[p1_idx] = p1_acts

            if p2_mask.any():
                p2_idx = p2_mask.nonzero().squeeze(-1)
                p2_hist = history[p2_idx]
                p2_seq = turn_count[p2_idx]
                p2_am = action_mask[p2_idx]
                logits, _ = policy_p2(p2_hist, p2_seq, p2_am)
                probs = F.softmax(logits, dim=-1)
                p2_acts = torch.distributions.Categorical(probs).sample()
                actions[p2_idx] = p2_acts

        # Update history with action
        tc = turn_count.clamp(max=MAX_SEQ_LEN - 1)
        for i in range(num_envs):
            t = tc[i].item()
            if t > 0 and t < MAX_SEQ_LEN:
                prev_t = t - 1
                action_onehot = F.one_hot(actions[i], NUM_ACTIONS).float()
                history[i, prev_t, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
        turn_count = (turn_count + 1).clamp(max=MAX_SEQ_LEN - 1)

        obs, rewards, dones, _, info = env.step(actions)
        action_mask = info['action_mask']

        # Update history with new observation
        for i in range(num_envs):
            if not dones[i]:
                t = turn_count[i].item()
                if t < MAX_SEQ_LEN:
                    history[i, t, :OBS_DIM] = obs[i]
                    history[i, t, 174] = 1.0

        if dones.any():
            done_idx = dones.nonzero().squeeze(-1)
            total_r1 += rewards[done_idx, 0].sum().item()
            total_r2 += rewards[done_idx, 1].sum().item()
            games_played += done_idx.numel()

            for idx in done_idx:
                history[idx] = 0
                turn_count[idx] = 1
                history[idx, 0, :OBS_DIM] = obs[idx]
                history[idx, 0, 174] = 1.0

            env.auto_reset()

    return total_r1 / games_played, total_r2 / games_played


def _train_best_response(
    br_policy: nn.Module,
    opponent_policy: nn.Module,
    player_id: int,
    num_envs: int,
    num_episodes: int,
    num_iterations: int,
    lr: float,
    seed: int = None,
) -> float:
    """Train a best response policy against fixed opponent."""
    optimizer = torch.optim.Adam(br_policy.parameters(), lr=lr)
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed or 54321)

    episodes_per_iter = num_episodes // num_iterations
    best_reward = -float('inf')

    for iteration in range(num_iterations):
        data, avg_reward = _collect_br_episodes(
            br_policy, opponent_policy, player_id, env, num_envs, episodes_per_iter
        )

        if avg_reward > best_reward:
            best_reward = avg_reward

        if data[0].size(0) > 64:
            _ppo_update_br(br_policy, optimizer, data)

    return best_reward


def _collect_br_episodes(
    br_policy: nn.Module,
    opponent_policy: nn.Module,
    player_id: int,
    env: BargainEnv,
    num_envs: int,
    min_episodes: int,
) -> tuple:
    """Collect episodes for BR training."""
    history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
    turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

    hist_list, seq_list, acts_list, lps_list, rews_list = [], [], [], [], []
    ep_hist = [[] for _ in range(num_envs)]
    ep_seq = [[] for _ in range(num_envs)]
    ep_acts = [[] for _ in range(num_envs)]
    ep_lps = [[] for _ in range(num_envs)]

    obs, info = env.reset()
    action_mask = info['action_mask']
    games_collected = 0
    total_reward = 0.0

    history[:, 0, :OBS_DIM] = obs
    history[:, 0, 174] = 1.0
    turn_count[:] = 1

    while games_collected < min_episodes:
        current_player = obs[:, 9]
        our_mask = current_player == player_id
        opp_mask = current_player == (1 - player_id)

        actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

        with torch.no_grad():
            if our_mask.any():
                our_idx = our_mask.nonzero().squeeze(-1)
                our_hist = history[our_idx]
                our_seq = turn_count[our_idx]
                our_am = action_mask[our_idx]

                logits, _ = br_policy(our_hist, our_seq, our_am)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                our_acts = dist.sample()
                our_lps = dist.log_prob(our_acts)
                actions[our_idx] = our_acts

                for i, idx in enumerate(our_idx.tolist()):
                    ep_hist[idx].append(our_hist[i].clone())
                    ep_seq[idx].append(our_seq[i].clone())
                    ep_acts[idx].append(our_acts[i])
                    ep_lps[idx].append(our_lps[i])

            if opp_mask.any():
                opp_idx = opp_mask.nonzero().squeeze(-1)
                opp_hist = history[opp_idx]
                opp_seq = turn_count[opp_idx]
                opp_am = action_mask[opp_idx]

                logits, _ = opponent_policy(opp_hist, opp_seq, opp_am)
                probs = F.softmax(logits, dim=-1)
                opp_acts = torch.distributions.Categorical(probs).sample()
                actions[opp_idx] = opp_acts

        # Update history with action
        active_idx = (obs[:, 9] >= 0).nonzero().squeeze(-1)
        if active_idx.numel() > 0:
            tc = turn_count[active_idx].clamp(max=MAX_SEQ_LEN - 1)
            for i, idx in enumerate(active_idx.tolist()):
                t = tc[i].item()
                if t > 0 and t < MAX_SEQ_LEN:
                    prev_t = t - 1
                    action_onehot = F.one_hot(actions[idx], NUM_ACTIONS).float()
                    history[idx, prev_t, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
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

        if dones.any():
            done_idx = dones.nonzero().squeeze(-1)
            for idx in done_idx.tolist():
                r = rewards[idx, player_id]
                total_reward += r.item()

                for h, s, a, lp in zip(ep_hist[idx], ep_seq[idx], ep_acts[idx], ep_lps[idx]):
                    hist_list.append(h)
                    seq_list.append(s)
                    acts_list.append(a)
                    lps_list.append(lp)
                    rews_list.append(r)

                ep_hist[idx] = []
                ep_seq[idx] = []
                ep_acts[idx] = []
                ep_lps[idx] = []

                history[idx] = 0
                turn_count[idx] = 1
                history[idx, 0, :OBS_DIM] = obs[idx]
                history[idx, 0, 174] = 1.0

                games_collected += 1

    if not hist_list:
        return (
            torch.zeros(0, MAX_SEQ_LEN, TOKEN_DIM, device='cuda'),
            torch.zeros(0, dtype=torch.long, device='cuda'),
            torch.zeros(0, dtype=torch.long, device='cuda'),
            torch.zeros(0, device='cuda'),
            torch.zeros(0, device='cuda'),
        ), 0.0

    data = (
        torch.stack(hist_list),
        torch.stack(seq_list),
        torch.stack(acts_list),
        torch.stack(lps_list),
        torch.stack(rews_list),
    )
    return data, total_reward / games_collected


def _ppo_update_br(policy, optimizer, data, epochs=4, batch_size=512, clip_ratio=0.2):
    """PPO update for best response training."""
    all_hist, all_seq, all_acts, all_old_lps, all_rews = data

    if all_hist.size(0) == 0:
        return

    all_rews = (all_rews - all_rews.mean()) / (all_rews.std() + 1e-8)

    for _ in range(epochs):
        perm = torch.randperm(all_hist.size(0))

        for start in range(0, all_hist.size(0), batch_size):
            end = min(start + batch_size, all_hist.size(0))
            idx = perm[start:end]

            hist = all_hist[idx]
            seq = all_seq[idx]
            acts = all_acts[idx]
            old_lps = all_old_lps[idx]
            rews = all_rews[idx]

            logits, _ = policy(hist, seq, action_mask=None)
            log_probs = F.log_softmax(logits, dim=-1)
            new_lps = log_probs.gather(1, acts.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(new_lps - old_lps)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            loss = -torch.min(ratio * rews, clipped_ratio * rews).mean()

            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            loss = loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


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
    magnet[:NUM_OFFERS] = (1/3) / NUM_OFFERS
    magnet[ACTION_ACCEPT] = 1/3
    magnet[ACTION_WALK] = 1/3
    return magnet


def create_end_magnet(num_actions: int = NUM_ACTIONS) -> torch.Tensor:
    """
    End-focused magnet:
    - 50% probability for "end" (split between walk and accept)
    - 50% probability for offers (split uniformly among 80 offers)
    """
    magnet = torch.zeros(num_actions)
    magnet[:NUM_OFFERS] = 0.5 / NUM_OFFERS
    magnet[ACTION_ACCEPT] = 0.25
    magnet[ACTION_WALK] = 0.25
    return magnet


def apply_rational_end(actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    If model picks walk or accept, choose the one with higher value.
    """
    item_quantities = torch.tensor([7.0, 4.0, 1.0], device=obs.device)

    my_values = obs[:, 0:3]
    walk_value = obs[:, 3]
    offer_items = obs[:, 4:7]
    offer_valid = obs[:, 7]

    offer_counts = offer_items.clamp(min=0) * item_quantities.unsqueeze(0)
    max_value = (item_quantities.unsqueeze(0) * my_values).sum(dim=1)
    max_value = max_value.clamp(min=0.01)
    offer_value = (offer_counts * my_values).sum(dim=1) / max_value

    end_mask = (actions == ACTION_WALK) | (actions == ACTION_ACCEPT)
    should_accept = (offer_value > walk_value) & (offer_valid > 0.5)

    new_actions = actions.clone()
    new_actions[end_mask & should_accept] = ACTION_ACCEPT
    new_actions[end_mask & ~should_accept] = ACTION_WALK

    return new_actions


def compute_rational_mask(obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute a constrained action mask based on rationality rules.

    Only allow actions that are not dominated:
    1. Offers: kept_value > walk_value, and if there's a current offer, kept_value > offer_value
    2. Accept: only if offer_value > walk_value
    3. Walk: always allowed
    """
    batch_size = obs.shape[0]
    device = obs.device

    my_values = obs[:, 0:3]
    walk_value = obs[:, 3:4]
    offer_items = obs[:, 4:7]
    offer_valid = obs[:, 7:8]

    item_quantities = torch.tensor([7.0, 4.0, 1.0], device=device)

    max_value = (item_quantities.unsqueeze(0) * my_values).sum(dim=1, keepdim=True)
    max_value = max_value.clamp(min=0.01)

    offer_counts = offer_items.clamp(min=0) * item_quantities.unsqueeze(0)
    current_offer_value = (offer_counts * my_values).sum(dim=1, keepdim=True)
    current_offer_value_norm = current_offer_value / max_value

    rational_mask = action_mask.clone()

    # ACCEPT: only if offer_valid AND offer_value > walk_value
    accept_allowed = (offer_valid > 0.5) & (current_offer_value_norm > walk_value)
    rational_mask[:, ACTION_ACCEPT] = rational_mask[:, ACTION_ACCEPT] * accept_allowed.squeeze()

    # OFFERS: check each one
    for action_idx in range(NUM_OFFERS):
        offer2 = action_idx % 2
        temp = action_idx // 2
        offer1 = temp % 5
        offer0 = temp // 5

        offer_to_opponent = torch.tensor([offer0, offer1, offer2], dtype=torch.float32, device=device)
        items_kept = item_quantities - offer_to_opponent

        kept_value = (items_kept.unsqueeze(0) * my_values).sum(dim=1, keepdim=True)
        kept_value_norm = kept_value / max_value

        constraint1 = kept_value_norm > walk_value
        constraint3 = (offer_valid < 0.5) | (kept_value_norm > current_offer_value_norm)

        offer_allowed = constraint1 & constraint3
        rational_mask[:, action_idx] = rational_mask[:, action_idx] * offer_allowed.squeeze()

    rational_mask[:, ACTION_WALK] = action_mask[:, ACTION_WALK].clamp(min=0.0)

    return rational_mask


def train_mmd(
    magnet_type: str = "uniform",
    num_envs: int = 4096,
    num_iterations: int = 200,
    min_episodes_per_iter: int = 2000,
    lr: float = 3e-4,
    xi: float = 0.01,
    clip_ratio: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 512,
    entropy_coef: float = 0.01,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-mmd",
    wandb_run_name: str = None,
    log_interval: int = 25,
    exploitability_interval: int = 0,
    exploitability_br_episodes: int = 50000,
    exploitability_br_iterations: int = 100,
    warmstart_p1: str = None,
    warmstart_p2: str = None,
    
):
    """
    Train using Magnetic Mirror Descent (PPO + KL penalty to magnet).

    Loss = PPO_loss + xi * KL(policy, magnet)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "MMD",
                "magnet_type": magnet_type,
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "min_episodes_per_iter": min_episodes_per_iter,
                "lr": lr,
                "xi": xi,
                "seed": seed,
            },
        )

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    if magnet_type == "uniform":
        magnet = create_uniform_magnet().cuda()
    elif magnet_type == "hierarchical":
        magnet = create_hierarchical_magnet().cuda()
    elif magnet_type == "end":
        magnet = create_end_magnet().cuda()
    else:
        raise ValueError(f"Unknown magnet type: {magnet_type}")

    policy_p1 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()

    if warmstart_p1:
        print(f"Loading P1 weights from {warmstart_p1}")
        policy_p1.load_state_dict(torch.load(warmstart_p1, map_location='cuda'))

    if warmstart_p2:
        print(f"Loading P2 weights from {warmstart_p2}")
        policy_p2.load_state_dict(torch.load(warmstart_p2, map_location='cuda'))
    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"MAGNETIC MIRROR DESCENT (PPO + KL penalty)")
    print(f"Magnet: {magnet_type}")
    if warmstart_p1:
        print(f"Starting warm-start P1: {warmstart_p1}")
    if warmstart_p2:
        print(f"Starting warm-start P2: {warmstart_p2}")
    
    print(f"Xi (KL penalty): {xi}")
    print(f"Parameters: {param_count:,}")
    if exploitability_interval > 0:
        print(f"Exploitability check every {exploitability_interval} iterations")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    exploitability_history = []
    total_games = 0
    start_time = time.time()
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
        p1_data, p2_data, games = _collect_episodes(
            env, policy_p1, policy_p2, num_envs, min_episodes_per_iter
        )
        total_games += games

        if p1_data[0].size(0) > 64:
            avg_r1 = _mmd_update(policy_p1, optimizer_p1, p1_data, magnet, xi,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p1.append(avg_r1)

        if p2_data[0].size(0) > 64:
            avg_r2 = _mmd_update(policy_p2, optimizer_p2, p2_data, magnet, xi,
                                  clip_ratio, ppo_epochs, batch_size, entropy_coef)
            reward_history_p2.append(avg_r2)

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "iteration": iteration + 1,
                    "p1_reward": r1,
                    "p2_reward": r2,
                    "welfare": r1 + r2,
                    "total_games": total_games,
                })

            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_path / f"mmd_{magnet_type}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_path / f"mmd_{magnet_type}_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

        if exploitability_interval > 0 and (iteration + 1) % exploitability_interval == 0:
            print(f"\n  Measuring exploitability at iteration {iteration + 1}...")
            exploit_results = measure_exploitability(
                policy_p1, policy_p2,
                num_envs=num_envs,
                br_episodes=exploitability_br_episodes,
                br_iterations=exploitability_br_iterations,
                lr=lr,
                seed=seed + iteration,
            )
            exploitability_history.append({
                'iteration': iteration + 1,
                'nashconv': exploit_results['nashconv'],
                'exploit_p1': exploit_results['exploit_p1'],
                'exploit_p2': exploit_results['exploit_p2'],
            })
            print(f"  NashConv: {exploit_results['nashconv']:.4f} "
                  f"(P1: {exploit_results['exploit_p1']:.4f}, P2: {exploit_results['exploit_p2']:.4f})")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "nashconv": exploit_results['nashconv'],
                    "exploit_p1": exploit_results['exploit_p1'],
                    "exploit_p2": exploit_results['exploit_p2'],
                })

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    if exploitability_history:
        print(f"\nExploitability History:")
        for entry in exploitability_history:
            print(f"  Iter {entry['iteration']}: NashConv={entry['nashconv']:.4f}")

    torch.save(policy_p1.state_dict(), save_path / f"mmd_{magnet_type}_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_path / f"mmd_{magnet_type}_final_p2.pt")
    print(f"Saved: mmd_{magnet_type}_final_p1.pt, mmd_{magnet_type}_final_p2.pt")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

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
        'exploitability_history': exploitability_history,
    }


def train_mmd_scheduled(
    magnet_type: str = "uniform",
    num_envs: int = 4096,
    num_iterations: int = 2441,
    min_episodes_per_iter: int = 2000,
    lr: float = 3e-4,
    xi_base: float = 0.05,
    xi_scale: float = 10_000_000,
    clip_ratio: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 512,
    entropy_coef: float = 0.01,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-mmd",
    wandb_run_name: str = None,
    log_interval: int = 25,
    exploitability_interval: int = 0,
    exploitability_br_episodes: int = 50000,
    exploitability_br_iterations: int = 100,
    warmstart_p1: str = None,
    warmstart_p2: str = None,
):
    """
    Train using MMD with annealing schedule.

    Xi_t = xi_base * sqrt(xi_scale / t)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "MMD-Scheduled",
                "magnet_type": magnet_type,
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "xi_base": xi_base,
                "xi_scale": xi_scale,
                "seed": seed,
            },
        )

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    if magnet_type == "uniform":
        magnet = create_uniform_magnet().cuda()
    elif magnet_type == "hierarchical":
        magnet = create_hierarchical_magnet().cuda()
    elif magnet_type == "end":
        magnet = create_end_magnet().cuda()
    else:
        raise ValueError(f"Unknown magnet type: {magnet_type}")

    policy_p1 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    if warmstart_p1:
        print(f"Loading P1 weights from {warmstart_p1}")
        policy_p1.load_state_dict(torch.load(warmstart_p1, map_location='cuda'))

    if warmstart_p2:
        print(f"Loading P2 weights from {warmstart_p2}")
        policy_p2.load_state_dict(torch.load(warmstart_p2, map_location='cuda'))

    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"MAGNETIC MIRROR DESCENT (Scheduled)")
    print(f"Magnet: {magnet_type}")
    if warmstart_p1:
        print(f"Starting warm-start P1: {warmstart_p1}")
    if warmstart_p2:
        print(f"Starting warm-start P2: {warmstart_p2}")
    print(f"Xi schedule: {xi_base} * sqrt({xi_scale:,} / t)")
    print(f"Parameters: {param_count:,}")
    if exploitability_interval > 0:
        print(f"Exploitability check every {exploitability_interval} iterations")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    exploitability_history = []
    total_games = 0
    start_time = time.time()
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
        p1_data, p2_data, games = _collect_episodes(
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

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | xi: {xi_current:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "iteration": iteration + 1,
                    "p1_reward": r1,
                    "p2_reward": r2,
                    "welfare": r1 + r2,
                    "xi": xi_current,
                    "total_games": total_games,
                })

            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_path / f"mmd_scheduled_{magnet_type}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_path / f"mmd_scheduled_{magnet_type}_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

        if exploitability_interval > 0 and (iteration + 1) % exploitability_interval == 0:
            print(f"\n  Measuring exploitability at iteration {iteration + 1}...")
            exploit_results = measure_exploitability(
                policy_p1, policy_p2,
                num_envs=num_envs,
                br_episodes=exploitability_br_episodes,
                br_iterations=exploitability_br_iterations,
                lr=lr,
                seed=seed + iteration,
            )
            exploitability_history.append({
                'iteration': iteration + 1,
                'nashconv': exploit_results['nashconv'],
                'exploit_p1': exploit_results['exploit_p1'],
                'exploit_p2': exploit_results['exploit_p2'],
            })
            print(f"  NashConv: {exploit_results['nashconv']:.4f} "
                  f"(P1: {exploit_results['exploit_p1']:.4f}, P2: {exploit_results['exploit_p2']:.4f})")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "nashconv": exploit_results['nashconv'],
                    "exploit_p1": exploit_results['exploit_p1'],
                    "exploit_p2": exploit_results['exploit_p2'],
                })

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    if exploitability_history:
        print(f"\nExploitability History:")
        for entry in exploitability_history:
            print(f"  Iter {entry['iteration']}: NashConv={entry['nashconv']:.4f}")

    torch.save(policy_p1.state_dict(), save_path / f"mmd_scheduled_{magnet_type}_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_path / f"mmd_scheduled_{magnet_type}_final_p2.pt")
    print(f"Saved: mmd_scheduled_{magnet_type}_final_p1.pt, mmd_scheduled_{magnet_type}_final_p2.pt")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

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
        'exploitability_history': exploitability_history,
    }


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
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-mmd",
    wandb_run_name: str = None,
    log_interval: int = 25,
    exploitability_interval: int = 0,
    exploitability_br_episodes: int = 50000,
    exploitability_br_iterations: int = 100,
    warmstart_p1: str = None,
    warmstart_p2: str = None,
):
    """
    Train with end magnet and rational end constraint.

    Magnet: 50% end (walk+accept), 50% offers
    Constraint: When model picks walk or accept, choose the one with higher value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "MMD-RationalEnd",
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "xi_base": xi_base,
                "xi_scale": xi_scale,
                "seed": seed,
            },
        )

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    magnet = create_end_magnet().cuda()

    policy_p1 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    if warmstart_p1:
        print(f"Loading P1 weights from {warmstart_p1}")
        policy_p1.load_state_dict(torch.load(warmstart_p1, map_location='cuda'))

    if warmstart_p2:
        print(f"Loading P2 weights from {warmstart_p2}")
        policy_p2.load_state_dict(torch.load(warmstart_p2, map_location='cuda'))
    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print("MMD WITH RATIONAL END CONSTRAINT")
    print("Magnet: end (50% end, 50% offers)")
    if warmstart_p1:
        print(f"Starting warm-start P1: {warmstart_p1}")
    if warmstart_p2:
        print(f"Starting warm-start P2: {warmstart_p2}")
    print(f"Xi schedule: {xi_base} * sqrt({xi_scale:,} / t)")
    print("Constraint: walk/accept -> pick higher value")
    print(f"Parameters: {param_count:,}")
    if exploitability_interval > 0:
        print(f"Exploitability check every {exploitability_interval} iterations")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    exploitability_history = []
    total_games = 0
    start_time = time.time()
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

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | xi: {xi_current:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "iteration": iteration + 1,
                    "p1_reward": r1,
                    "p2_reward": r2,
                    "welfare": r1 + r2,
                    "xi": xi_current,
                    "total_games": total_games,
                })

            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_path / "mmd_rational_end_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_path / "mmd_rational_end_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

        if exploitability_interval > 0 and (iteration + 1) % exploitability_interval == 0:
            print(f"\n  Measuring exploitability at iteration {iteration + 1}...")
            exploit_results = measure_exploitability(
                policy_p1, policy_p2,
                num_envs=num_envs,
                br_episodes=exploitability_br_episodes,
                br_iterations=exploitability_br_iterations,
                lr=lr,
                seed=seed + iteration,
            )
            exploitability_history.append({
                'iteration': iteration + 1,
                'nashconv': exploit_results['nashconv'],
                'exploit_p1': exploit_results['exploit_p1'],
                'exploit_p2': exploit_results['exploit_p2'],
            })
            print(f"  NashConv: {exploit_results['nashconv']:.4f}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"nashconv": exploit_results['nashconv']})

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    torch.save(policy_p1.state_dict(), save_path / "mmd_rational_end_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_path / "mmd_rational_end_final_p2.pt")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

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
        'exploitability_history': exploitability_history,
    }


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
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-mmd",
    wandb_run_name: str = None,
    log_interval: int = 25,
    exploitability_interval: int = 0,
    exploitability_br_episodes: int = 50000,
    exploitability_br_iterations: int = 100,
    warmstart_p1: str = None,
    warmstart_p2: str = None,
):
    """
    Train using MMD with normalized advantage rewards.

    Advantage = (reward - walk_value) / (max_value - walk_value)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "MMD-Advantage",
                "magnet_type": magnet_type,
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "xi_base": xi_base,
                "xi_scale": xi_scale,
                "seed": seed,
            },
        )

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    if magnet_type == "uniform":
        magnet = create_uniform_magnet().cuda()
    elif magnet_type == "hierarchical":
        magnet = create_hierarchical_magnet().cuda()
    else:
        raise ValueError(f"Unknown magnet type: {magnet_type}")

    policy_p1 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    if warmstart_p1:
        print(f"Loading P1 weights from {warmstart_p1}")
        policy_p1.load_state_dict(torch.load(warmstart_p1, map_location='cuda'))

    if warmstart_p2:
        print(f"Loading P2 weights from {warmstart_p2}")
        policy_p2.load_state_dict(torch.load(warmstart_p2, map_location='cuda'))
    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"MMD WITH NORMALIZED ADVANTAGE REWARDS")
    print(f"Magnet: {magnet_type}")
    if warmstart_p1:
        print(f"Starting warm-start P1: {warmstart_p1}")
    if warmstart_p2:
        print(f"Starting warm-start P2: {warmstart_p2}")
    print(f"Xi schedule: {xi_base} * sqrt({xi_scale:,} / t)")
    print(f"Reward: (value - walk) / (max - walk)")
    print(f"Parameters: {param_count:,}")
    if exploitability_interval > 0:
        print(f"Exploitability check every {exploitability_interval} iterations")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    exploitability_history = []
    total_games = 0
    start_time = time.time()
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
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

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1 adv: {r1:.4f} | P2 adv: {r2:.4f} | xi: {xi_current:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "iteration": iteration + 1,
                    "p1_advantage": r1,
                    "p2_advantage": r2,
                    "xi": xi_current,
                    "total_games": total_games,
                })

            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_path / f"mmd_advantage_{magnet_type}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_path / f"mmd_advantage_{magnet_type}_best_p2.pt")
                print(f"  -> New best advantage: {welfare:.4f} (saved)")

        if exploitability_interval > 0 and (iteration + 1) % exploitability_interval == 0:
            print(f"\n  Measuring exploitability at iteration {iteration + 1}...")
            exploit_results = measure_exploitability(
                policy_p1, policy_p2,
                num_envs=num_envs,
                br_episodes=exploitability_br_episodes,
                br_iterations=exploitability_br_iterations,
                lr=lr,
                seed=seed + iteration,
            )
            exploitability_history.append({
                'iteration': iteration + 1,
                'nashconv': exploit_results['nashconv'],
            })
            print(f"  NashConv: {exploit_results['nashconv']:.4f}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"nashconv": exploit_results['nashconv']})

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1 adv={final_r1:.4f}, P2 adv={final_r2:.4f}")

    torch.save(policy_p1.state_dict(), save_path / f"mmd_advantage_{magnet_type}_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_path / f"mmd_advantage_{magnet_type}_final_p2.pt")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

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
        'exploitability_history': exploitability_history,
    }


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
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-mmd",
    wandb_run_name: str = None,
    log_interval: int = 25,
    exploitability_interval: int = 0,
    exploitability_br_episodes: int = 50000,
    exploitability_br_iterations: int = 100,
    warmstart_p1: str = None,                                                                                     
    warmstart_p2: str = None,    
):
    """
    Train using MMD with constrained (rational) action masking.

    Only allows non-dominated actions:
    1. Offers where kept_value > walk_value
    2. Accept only if offer_value > walk_value
    3. Counter-offers only if kept_value > current_offer_value
    4. Walk always allowed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "MMD-Constrained",
                "magnet_type": magnet_type,
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "xi_base": xi_base,
                "xi_scale": xi_scale,
                "seed": seed,
            },
        )

    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    if magnet_type == "uniform":
        magnet = create_uniform_magnet().cuda()
    elif magnet_type == "hierarchical":
        magnet = create_hierarchical_magnet().cuda()
    else:
        raise ValueError(f"Unknown magnet type: {magnet_type}")

    policy_p1 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    policy_p2 = HistoryTransformerPolicy(token_dim=TOKEN_DIM, num_actions=NUM_ACTIONS).cuda()
    if warmstart_p1:
        print(f"Loading P1 weights from {warmstart_p1}")
        policy_p1.load_state_dict(torch.load(warmstart_p1, map_location='cuda'))

    if warmstart_p2:
        print(f"Loading P2 weights from {warmstart_p2}")
        policy_p2.load_state_dict(torch.load(warmstart_p2, map_location='cuda'))

    optimizer_p1 = torch.optim.Adam(policy_p1.parameters(), lr=lr)
    optimizer_p2 = torch.optim.Adam(policy_p2.parameters(), lr=lr)

    param_count = sum(p.numel() for p in policy_p1.parameters())

    print("=" * 70)
    print(f"MMD WITH CONSTRAINED (RATIONAL) ACTIONS")
    print(f"Magnet: {magnet_type}")
    if warmstart_p1:
        print(f"Starting warm-start P1: {warmstart_p1}")
    if warmstart_p2:
        print(f"Starting warm-start P2: {warmstart_p2}")
    print(f"Xi schedule: {xi_base} * sqrt({xi_scale:,} / t)")
    print(f"Constraints: offer > walk, accept > walk, counter > current")
    print(f"Parameters: {param_count:,}")
    if exploitability_interval > 0:
        print(f"Exploitability check every {exploitability_interval} iterations")
    print("=" * 70)
    print()

    reward_history_p1 = []
    reward_history_p2 = []
    exploitability_history = []
    total_games = 0
    start_time = time.time()
    best_welfare = -float('inf')

    for iteration in range(num_iterations):
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

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            r1 = reward_history_p1[-1] if reward_history_p1 else 0
            r2 = reward_history_p2[-1] if reward_history_p2 else 0
            elapsed = time.time() - start_time
            speed = total_games / elapsed
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | xi: {xi_current:.4f} | {speed:.0f} g/s | Total: {total_games:,}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "iteration": iteration + 1,
                    "p1_reward": r1,
                    "p2_reward": r2,
                    "welfare": r1 + r2,
                    "xi": xi_current,
                    "total_games": total_games,
                })

            welfare = r1 + r2
            if welfare > best_welfare:
                best_welfare = welfare
                torch.save(policy_p1.state_dict(), save_path / f"mmd_constrained_{magnet_type}_best_p1.pt")
                torch.save(policy_p2.state_dict(), save_path / f"mmd_constrained_{magnet_type}_best_p2.pt")
                print(f"  -> New best welfare: {welfare:.4f} (saved)")

        if exploitability_interval > 0 and (iteration + 1) % exploitability_interval == 0:
            print(f"\n  Measuring exploitability at iteration {iteration + 1}...")
            exploit_results = measure_exploitability(
                policy_p1, policy_p2,
                num_envs=num_envs,
                br_episodes=exploitability_br_episodes,
                br_iterations=exploitability_br_iterations,
                lr=lr,
                seed=seed + iteration,
            )
            exploitability_history.append({
                'iteration': iteration + 1,
                'nashconv': exploit_results['nashconv'],
            })
            print(f"  NashConv: {exploit_results['nashconv']:.4f}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.log({"nashconv": exploit_results['nashconv']})

    elapsed = time.time() - start_time
    final_r1 = reward_history_p1[-1] if reward_history_p1 else 0
    final_r2 = reward_history_p2[-1] if reward_history_p2 else 0

    print()
    print(f"COMPLETE: {total_games:,} games in {elapsed:.1f}s ({total_games/elapsed:.0f} g/s)")
    print(f"Final: P1={final_r1:.4f}, P2={final_r2:.4f}, Welfare={final_r1+final_r2:.4f}")

    torch.save(policy_p1.state_dict(), save_path / f"mmd_constrained_{magnet_type}_final_p1.pt")
    torch.save(policy_p2.state_dict(), save_path / f"mmd_constrained_{magnet_type}_final_p2.pt")

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

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
        'exploitability_history': exploitability_history,
    }


def _collect_episodes(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Collect episodes with history tracking for History Transformer."""
    history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
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
            return (torch.zeros(0, MAX_SEQ_LEN, TOKEN_DIM, device='cuda'),
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


def _collect_episodes_rational_end(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Collect episodes with rational end constraint."""
    history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
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

    def stack_data(hist_list, seq_list, acts_list, lps_list, rews_list):
        if not hist_list:
            return (torch.zeros(0, MAX_SEQ_LEN, TOKEN_DIM, device='cuda'),
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


def _collect_episodes_advantage(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Collect episodes with normalized advantage rewards."""
    history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
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

    p1_outside = torch.zeros(num_envs, device='cuda')
    p2_outside = torch.zeros(num_envs, device='cuda')

    obs, info = env.reset()
    action_mask = info['action_mask']
    games_collected = 0

    history[:, 0, :OBS_DIM] = obs
    history[:, 0, 174] = 1.0
    turn_count[:] = 1

    p1_outside[:] = obs[:, 3]

    while games_collected < min_episodes:
        current_player = obs[:, 9]
        p1_mask = current_player == 0
        p2_mask = current_player == 1

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

                walk1 = p1_outside[idx]
                walk2 = p2_outside[idx]

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

                p1_outside[idx] = obs[idx, 3]
                p2_outside[idx] = 0

                games_collected += 1

    def stack_data(hist_list, seq_list, acts_list, lps_list, rews_list):
        if not hist_list:
            return (torch.zeros(0, MAX_SEQ_LEN, TOKEN_DIM, device='cuda'),
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


def _collect_episodes_constrained(env, policy_p1, policy_p2, num_envs, min_episodes):
    """Collect episodes with constrained (rational) action masks."""
    history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
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
            return (torch.zeros(0, MAX_SEQ_LEN, TOKEN_DIM, device='cuda'),
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

    Loss = -PPO_objective + xi * KL(policy, magnet)
    """
    all_hist, all_seq, all_acts, all_old_lps, all_rews = data

    if all_hist.size(0) == 0:
        return 0.0

    avg_reward = all_rews.mean().item()

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

            logits, values = policy(hist, seq, action_mask=None)

            log_probs = F.log_softmax(logits, dim=-1)
            new_lps = log_probs.gather(1, acts.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(new_lps - old_lps)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            ppo_loss = -torch.min(ratio * rews, clipped_ratio * rews).mean()

            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            magnet_expanded = magnet.unsqueeze(0).expand_as(probs)
            kl_to_magnet = (probs * (log_probs - torch.log(magnet_expanded + 1e-10))).sum(dim=-1).mean()

            loss = ppo_loss - entropy_coef * entropy + xi * kl_to_magnet

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    return avg_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMD Training for Bargaining Game")
    parser.add_argument("--magnet-type", type=str, default="uniform",
                        choices=["uniform", "hierarchical", "end"],
                        help="Type of magnet distribution")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--iterations", type=int, default=200, help="Training iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=2000, help="Episodes per iteration")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--xi", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--scheduled", action="store_true", help="Use scheduled xi annealing")
    parser.add_argument("--xi-base", type=float, default=0.05, help="Base xi for scheduled")
    parser.add_argument("--xi-scale", type=float, default=10_000_000, help="Scale for scheduled xi")
    parser.add_argument("--rational-end", action="store_true", help="Use rational end constraint")
    parser.add_argument("--advantage", action="store_true", help="Use advantage-based rewards")
    parser.add_argument("--constrained", action="store_true", help="Use constrained action masking")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save models")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="bargaining-mmd", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--log-interval", type=int, default=25, help="Logging interval")
    parser.add_argument("--exploitability-interval", type=int, default=0,
                        help="Measure exploitability every N iterations (0 = disabled)")
    parser.add_argument("--exploitability-br-episodes", type=int, default=50000,
                        help="Episodes for BR training when measuring exploitability")
    parser.add_argument("--exploitability-br-iterations", type=int, default=100,
                        help="Iterations for BR training when measuring exploitability")
    parser.add_argument("--warmstart-p1", type=str, default=None,                                                              
                      help="Path to pretrained weights for player 1")                                                        
    parser.add_argument("--warmstart-p2", type=str, default=None,                                                              
                      help="Path to pretrained weights for player 2")   
    args = parser.parse_args()

    if args.rational_end:
        train_mmd_rational_end(
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            min_episodes_per_iter=args.episodes_per_iter,
            lr=args.lr,
            xi_base=args.xi_base,
            xi_scale=args.xi_scale,
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
            exploitability_interval=args.exploitability_interval,
            exploitability_br_episodes=args.exploitability_br_episodes,
            exploitability_br_iterations=args.exploitability_br_iterations,
            warmstart_p1=args.warmstart_p1,
            warmstart_p2=args.warmstart_p2,
        )
    elif args.advantage:
        train_mmd_advantage(
            magnet_type=args.magnet_type,
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            min_episodes_per_iter=args.episodes_per_iter,
            lr=args.lr,
            xi_base=args.xi_base,
            xi_scale=args.xi_scale,
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
            exploitability_interval=args.exploitability_interval,
            exploitability_br_episodes=args.exploitability_br_episodes,
            exploitability_br_iterations=args.exploitability_br_iterations,
            warmstart_p1=args.warmstart_p1,
            warmstart_p2=args.warmstart_p2,
        )
    elif args.constrained:
        train_mmd_constrained(
            magnet_type=args.magnet_type,
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            min_episodes_per_iter=args.episodes_per_iter,
            lr=args.lr,
            xi_base=args.xi_base,
            xi_scale=args.xi_scale,
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
            exploitability_interval=args.exploitability_interval,
            exploitability_br_episodes=args.exploitability_br_episodes,
            exploitability_br_iterations=args.exploitability_br_iterations,
            warmstart_p1=args.warmstart_p1,
            warmstart_p2=args.warmstart_p2,
        )
    elif args.scheduled:
        train_mmd_scheduled(
            magnet_type=args.magnet_type,
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            min_episodes_per_iter=args.episodes_per_iter,
            lr=args.lr,
            xi_base=args.xi_base,
            xi_scale=args.xi_scale,
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
            exploitability_interval=args.exploitability_interval,
            exploitability_br_episodes=args.exploitability_br_episodes,
            exploitability_br_iterations=args.exploitability_br_iterations,
            warmstart_p1=args.warmstart_p1,
            warmstart_p2=args.warmstart_p2,
        )
    else:
        train_mmd(
            magnet_type=args.magnet_type,
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            min_episodes_per_iter=args.episodes_per_iter,
            lr=args.lr,
            xi=args.xi,
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
            exploitability_interval=args.exploitability_interval,
            exploitability_br_episodes=args.exploitability_br_episodes,
            exploitability_br_iterations=args.exploitability_br_iterations,
            warmstart_p1=args.warmstart_p1,
            warmstart_p2=args.warmstart_p2,
        )
