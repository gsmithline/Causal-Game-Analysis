#!/usr/bin/env python3
"""
PSRO (Policy Space Response Oracles) training for the bargaining game.

Based on: Lanctot et al. "A Unified Game-Theoretic Approach to Multiagent
Reinforcement Learning" (NeurIPS 2017).

Algorithm:
1. Initialize policy populations for P1 and P2
2. Compute empirical payoff matrix by simulating all policy pairs
3. Solve meta-game for Nash equilibrium to get mixing distributions
4. Train best response for each player against opponent's meta-strategy
5. Add best responses to populations
6. Repeat until convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from simulator.python import BargainEnv, NUM_ACTIONS, OBS_DIM
import copy

try:
    from ..ppo.policy import PolicyNetwork, HistoryTransformerPolicy
    from ..exploitability import measure_exploitability, measure_psro_exploitability, measure_psro_exploitability_inline, PSROMixturePolicy
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "ppo"))
    from policy import PolicyNetwork, HistoryTransformerPolicy
    sys.path.append(str(Path(__file__).parent.parent))
    from exploitability import measure_exploitability, measure_psro_exploitability, measure_psro_exploitability_inline, PSROMixturePolicy

# Constants for history transformer
TOKEN_DIM = 175  # 92 (obs) + 82 (one-hot action) + 1 (validity flag)
MAX_SEQ_LEN = 6

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def compute_payoff_matrix(
    p1_policies: List[nn.Module],
    p2_policies: List[nn.Module],
    num_envs: int = 2048,
    num_episodes: int = 1000,
    device: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical payoff matrices by simulating all policy pairs.

    Args:
        p1_policies: List of P1 policies
        p2_policies: List of P2 policies
        num_envs: Number of parallel environments
        num_episodes: Episodes to sample per policy pair
        device: CUDA device

    Returns:
        payoff_p1: Payoff matrix for P1 of shape (len(p1), len(p2))
        payoff_p2: Payoff matrix for P2 of shape (len(p1), len(p2))
    """
    n_p1 = len(p1_policies)
    n_p2 = len(p2_policies)
    payoff_p1 = np.zeros((n_p1, n_p2))
    payoff_p2 = np.zeros((n_p1, n_p2))

    env = BargainEnv(num_envs=num_envs, self_play=True, device=device)

    for i, policy_p1 in enumerate(p1_policies):
        for j, policy_p2 in enumerate(p2_policies):
            total_r1 = 0.0
            total_r2 = 0.0
            games_played = 0

            while games_played < num_episodes:
                obs, info = env.reset()
                action_mask = info['action_mask']
                active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

                for _ in range(12):
                    if not active.any():
                        break

                    current_player = env.get_current_player()
                    p1_turn = (current_player == 0) & active
                    p2_turn = (current_player == 1) & active

                    actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

                    if p1_turn.any():
                        p1_obs = obs[p1_turn]
                        p1_am = action_mask[p1_turn]
                        with torch.no_grad():
                            logits, _ = policy_p1(p1_obs, p1_am)
                            probs = F.softmax(logits, dim=-1)
                            p1_acts = torch.distributions.Categorical(probs).sample()
                        actions[p1_turn] = p1_acts

                    if p2_turn.any():
                        p2_obs = obs[p2_turn]
                        p2_am = action_mask[p2_turn]
                        with torch.no_grad():
                            logits, _ = policy_p2(p2_obs, p2_am)
                            probs = F.softmax(logits, dim=-1)
                            p2_acts = torch.distributions.Categorical(probs).sample()
                        actions[p2_turn] = p2_acts

                    obs, rewards, dones, _, info = env.step(actions)
                    action_mask = info['action_mask']

                    if dones.any():
                        done_mask = dones & active
                        total_r1 += rewards[done_mask, 0].sum().item()
                        total_r2 += rewards[done_mask, 1].sum().item()
                        games_played += done_mask.sum().item()
                        active = active & ~dones
                        env.auto_reset()

            payoff_p1[i, j] = total_r1 / games_played
            payoff_p2[i, j] = total_r2 / games_played

    return payoff_p1, payoff_p2


def compute_entropy(sigma: np.ndarray) -> float:
    """Compute entropy of a probability distribution."""
    sigma = np.clip(sigma, 1e-10, 1.0)
    return -np.sum(sigma * np.log(sigma))


def solve_nash_equilibrium(
    payoff_p1: np.ndarray,
    payoff_p2: np.ndarray,
    use_max_entropy: bool = True,
    entropy_weight: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for Nash equilibrium of the meta-game using entropy-regularized dynamics.

    For ASYMMETRIC games (like bargaining where P1 and P2 have different roles),
    this finds Nash equilibria where P1 and P2 can have DIFFERENT strategies.

    Uses entropy-regularized replicator dynamics which:
    1. Converges to approximate Nash equilibrium
    2. Naturally biases toward higher entropy (more mixed) strategies
    3. Is fast and handles asymmetric games correctly

    Args:
        payoff_p1: Payoff matrix for P1, shape (n_p1, n_p2)
        payoff_p2: Payoff matrix for P2, shape (n_p1, n_p2)
        use_max_entropy: If True, add entropy regularization to favor mixed strategies
        entropy_weight: Weight for entropy regularization (higher = more mixing)

    Returns:
        sigma_p1: Meta-strategy (distribution over policies) for P1
        sigma_p2: Meta-strategy for P2
    """
    n_p1, n_p2 = payoff_p1.shape

    sigma_p1 = np.ones(n_p1) / n_p1
    sigma_p2 = np.ones(n_p2) / n_p2

    #Entropy-regularized multiplicative weights / replicator dynamics
    #This is equivalent to mirror descent with KL divergence
    lr = .3
    tau = entropy_weight if use_max_entropy else 0.0

    for iteration in range(3000):
        # P1 expected payoffs against sigma_p2 (plus entropy regularization)
        u1 = payoff_p1 @ sigma_p2
        if tau > 0:
            # Add entropy gradient: d/d(sigma) of -tau * sum(sigma * log(sigma)) = -tau * (1 + log(sigma))
            # But in multiplicative weights, we just scale down the payoff advantage
            u1 = u1 + tau * (1 - np.log(sigma_p1 + 1e-10))

        # P2 expected payoffs against sigma_p1 (plus entropy regularization)
        u2 = payoff_p2.T @ sigma_p1
        if tau > 0:
            u2 = u2 + tau * (1 - np.log(sigma_p2 + 1e-10))

        # Multiplicative weights update (softmax-like)
        sigma_p1_new = sigma_p1 * np.exp(lr * (u1 - u1.max()))
        sigma_p1_new = sigma_p1_new / sigma_p1_new.sum()

        sigma_p2_new = sigma_p2 * np.exp(lr * (u2 - u2.max()))
        sigma_p2_new = sigma_p2_new / sigma_p2_new.sum()

        # Check for convergence
        if np.allclose(sigma_p1, sigma_p1_new, atol=1e-8) and np.allclose(sigma_p2, sigma_p2_new, atol=1e-8):
            break

        sigma_p1 = sigma_p1_new
        sigma_p2 = sigma_p2_new

        # Decrease learning rate for stability
        if iteration > 1000:
            lr = 0.01
        if iteration > 2000:
            lr = 0.005

    return sigma_p1, sigma_p2


def compute_exploitability(
    payoff_p1: np.ndarray,
    payoff_p2: np.ndarray,
    sigma_p1: np.ndarray,
    sigma_p2: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute exploitability (NashConv) of the current meta-strategy.

    Args:
        payoff_p1, payoff_p2: Payoff matrices
        sigma_p1, sigma_p2: Current meta-strategies

    Returns:
        exploit_p1: P1's regret (how much P1 could gain by deviating)
        exploit_p2: P2's regret
        nashconv: Total exploitability (sum of regrets)
    """
    v1_current = sigma_p1 @ payoff_p1 @ sigma_p2

    br_payoffs_p1 = payoff_p1 @ sigma_p2
    v1_br = br_payoffs_p1.max()

    v2_current = sigma_p1 @ payoff_p2 @ sigma_p2

    br_payoffs_p2 = payoff_p2.T @ sigma_p1
    v2_br = br_payoffs_p2.max()

    exploit_p1 = max(0, v1_br - v1_current)
    exploit_p2 = max(0, v2_br - v2_current)
    nashconv = exploit_p1 + exploit_p2

    return exploit_p1, exploit_p2, nashconv


def evaluate_br_vs_meta(
    policy: nn.Module,
    opponent_policies: List[nn.Module],
    opponent_meta_strategy: np.ndarray,
    player_id: int,
    num_envs: int = 2048,
    num_episodes: int = 2000,
    device: int = 0,
) -> float:
    """
    Evaluate a trained BR policy against opponent's meta-strategy.
    
    This gives the TRUE best response value (not training average).
    """
    env = BargainEnv(num_envs=num_envs, self_play=True, device=device)
    
    def sample_opponent_idx():
        return np.random.choice(len(opponent_policies), p=opponent_meta_strategy)
    
    total_reward = 0.0
    games_played = 0
    
    while games_played < num_episodes:
        obs, info = env.reset()
        action_mask = info['action_mask']
        
        # Sample opponent for each env
        opp_indices = [sample_opponent_idx() for _ in range(num_envs)]
        active = torch.ones(num_envs, dtype=torch.bool, device='cuda')
        
        for _ in range(12):
            if not active.any():
                break
            
            current_player = env.get_current_player()
            our_turn = (current_player == player_id) & active
            opp_turn = (current_player == (1 - player_id)) & active
            
            actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')
            
            # Our policy's turn
            if our_turn.any():
                our_obs = obs[our_turn]
                our_am = action_mask[our_turn]
                with torch.no_grad():
                    logits, _ = policy(our_obs, our_am)
                    probs = F.softmax(logits, dim=-1)
                    our_acts = torch.distributions.Categorical(probs).sample()
                actions[our_turn] = our_acts
            
            #sample from metastrategy
            if opp_turn.any():
                opp_indices_tensor = torch.tensor(opp_indices, device='cuda')
                for opp_idx in range(len(opponent_policies)):
                    mask = (opp_indices_tensor == opp_idx) & opp_turn
                    if mask.any():
                        opp_obs = obs[mask]
                        opp_am = action_mask[mask]
                        with torch.no_grad():
                            logits, _ = opponent_policies[opp_idx](opp_obs, opp_am)
                            probs = F.softmax(logits, dim=-1)
                            opp_acts = torch.distributions.Categorical(probs).sample()
                        actions[mask] = opp_acts
            
            obs, rewards, dones, _, info = env.step(actions)
            action_mask = info['action_mask']
            
            if dones.any():
                for idx in dones.nonzero().squeeze(-1):
                    if not active[idx]:
                        continue
                    reward = rewards[idx, player_id].item()
                    total_reward += reward
                    games_played += 1
                active = active & ~dones
                env.auto_reset()
    
    return total_reward / games_played if games_played > 0 else 0.0


def evaluate_br_vs_meta_history(
    policy: nn.Module,
    opponent_policies: List[nn.Module],
    opponent_meta_strategy: np.ndarray,
    player_id: int,
    num_envs: int = 2048,
    num_episodes: int = 2000,
    device: int = 0,
) -> float:
    """
    Evaluate a trained BR policy (history transformer) against opponent's meta-strategy.
    """
    env = BargainEnv(num_envs=num_envs, self_play=True, device=device)
    
    def sample_opponent_idx():
        return np.random.choice(len(opponent_policies), p=opponent_meta_strategy)
    
    total_reward = 0.0
    games_played = 0
    
    while games_played < num_episodes:
        obs, info = env.reset()
        action_mask = info['action_mask']
        
        opp_indices = [sample_opponent_idx() for _ in range(num_envs)]
        
        # History buffers
        history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
        turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')
        
        active = torch.ones(num_envs, dtype=torch.bool, device='cuda')
        
        for step in range(12):
            if not active.any():
                break
            
            current_player = env.get_current_player()
            our_turn = (current_player == player_id) & active
            opp_turn = (current_player == (1 - player_id)) & active
            
            actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')
            
            # Update history with current observation
            active_envs = active.nonzero().squeeze(-1)
            if active_envs.numel() > 0:
                tc = turn_count[active_envs].clamp(max=MAX_SEQ_LEN - 1)
                history[active_envs, tc, :OBS_DIM] = obs[active_envs]
                history[active_envs, tc, TOKEN_DIM - 1] = 1.0
            
            # Our policy's turn
            if our_turn.any():
                our_indices = our_turn.nonzero().squeeze(-1)
                our_hist = history[our_indices]
                our_seq_lens = turn_count[our_indices] + 1
                our_am = action_mask[our_indices]
                with torch.no_grad():
                    logits, _ = policy(our_hist, our_seq_lens, our_am)
                    probs = F.softmax(logits, dim=-1)
                    our_acts = torch.distributions.Categorical(probs).sample()
                actions[our_indices] = our_acts
            
            # Opponent's turn
            if opp_turn.any():
                opp_indices_tensor = torch.tensor(opp_indices, device='cuda')
                for opp_idx in range(len(opponent_policies)):
                    mask = (opp_indices_tensor == opp_idx) & opp_turn
                    if mask.any():
                        opp_env_indices = mask.nonzero().squeeze(-1)
                        opp_hist = history[opp_env_indices]
                        opp_seq_lens = turn_count[opp_env_indices] + 1
                        opp_am = action_mask[opp_env_indices]
                        with torch.no_grad():
                            logits, _ = opponent_policies[opp_idx](opp_hist, opp_seq_lens, opp_am)
                            probs = F.softmax(logits, dim=-1)
                            opp_acts = torch.distributions.Categorical(probs).sample()
                        actions[opp_env_indices] = opp_acts
            
            # Update history with actions taken
            if active_envs.numel() > 0:
                tc = turn_count[active_envs].clamp(max=MAX_SEQ_LEN - 1)
                action_onehot = F.one_hot(actions[active_envs], num_classes=NUM_ACTIONS).float()
                history[active_envs, tc, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
                turn_count[active_envs] = (turn_count[active_envs] + 1).clamp(max=MAX_SEQ_LEN - 1)
            
            obs, rewards, dones, _, info = env.step(actions)
            action_mask = info['action_mask']
            
            if dones.any():
                for idx in dones.nonzero().squeeze(-1):
                    if not active[idx]:
                        continue
                    reward = rewards[idx, player_id].item()
                    total_reward += reward
                    games_played += 1
                active = active & ~dones
                env.auto_reset()
    
    return total_reward / games_played if games_played > 0 else 0.0


def train_best_response(
    policy: nn.Module,
    opponent_policies: List[nn.Module],
    opponent_meta_strategy: np.ndarray,
    player_id: int,
    num_envs: int = 4096,
    num_episodes: int = 5000,
    ppo_epochs: int = 4,
    lr: float = 3e-4,
    device: int = 0,
    episodes_per_iter: int = 3000,  # Episodes per PPO update iteration
) -> float:
    """
    Train a best response policy against the opponent's meta-strategy.

    Uses iterative PPO training: collect episodes, update, repeat.
    This allows the policy to improve incrementally rather than doing
    one update on data collected with initial weights.

    Args:
        policy: Policy to train (will be modified in place)
        opponent_policies: List of opponent policies
        opponent_meta_strategy: Distribution over opponent policies
        player_id: 0 for P1, 1 for P2
        num_envs: Number of parallel environments
        num_episodes: Total training episodes
        ppo_epochs: PPO update epochs per iteration
        lr: Learning rate
        device: CUDA device
        episodes_per_iter: Episodes to collect before each PPO update

    Returns:
        Average reward achieved (from last iteration)
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    env = BargainEnv(num_envs=num_envs, self_play=True, device=device)

    # Sample opponent policy indices according to meta-strategy
    def sample_opponent_idx():
        return np.random.choice(len(opponent_policies), p=opponent_meta_strategy)

    # Calculate number of iterations
    num_iterations = max(1, num_episodes // episodes_per_iter)
    #num_iterations = 100
    
    total_reward = 0.0
    total_games = 0
    last_avg_reward = 0.0

    for iteration in range(num_iterations):
        # Collect episodes for this iteration
        episodes = []
        iter_reward = 0.0
        iter_games = 0

        while iter_games < episodes_per_iter:
            obs, info = env.reset()
            action_mask = info['action_mask']

            # Assign opponent policies to each environment
            opp_indices = [sample_opponent_idx() for _ in range(num_envs)]

            # Track data for our player
            player_data = [[] for _ in range(num_envs)]
            active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

            for _ in range(12):
                if not active.any():
                    break

                current_player = env.get_current_player()
                our_turn = (current_player == player_id) & active
                opp_turn = (current_player == (1 - player_id)) & active

                actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

                # Our policy's turn
                if our_turn.any():
                    our_obs = obs[our_turn]
                    our_am = action_mask[our_turn]
                    our_walk = our_obs[:, 3]

                    with torch.no_grad():
                        logits, _ = policy(our_obs, our_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        our_acts = dist.sample()
                        our_lps = dist.log_prob(our_acts)
                    actions[our_turn] = our_acts

                    for i, idx in enumerate(our_turn.nonzero().squeeze(-1)):
                        player_data[idx.item()].append((
                            our_obs[i], our_acts[i], our_lps[i], our_walk[i]
                        ))

                # Opponent's turn - use sampled opponent policies
                if opp_turn.any():
                    opp_indices_tensor = torch.tensor(opp_indices, device='cuda')
                    for opp_idx in range(len(opponent_policies)):
                        mask = (opp_indices_tensor == opp_idx) & opp_turn
                        if mask.any():
                            opp_obs = obs[mask]
                            opp_am = action_mask[mask]
                            with torch.no_grad():
                                logits, _ = opponent_policies[opp_idx](opp_obs, opp_am)
                                probs = F.softmax(logits, dim=-1)
                                opp_acts = torch.distributions.Categorical(probs).sample()
                            actions[mask] = opp_acts

                obs, rewards, dones, _, info = env.step(actions)
                action_mask = info['action_mask']

                if dones.any():
                    for idx in dones.nonzero().squeeze(-1):
                        idx = idx.item()
                        if not active[idx]:
                            continue
                        reward = rewards[idx, player_id].item()
                        iter_reward += reward
                        iter_games += 1
                        total_reward += reward
                        total_games += 1

                        for o, a, lp, walk in player_data[idx]:
                            episodes.append((o, a, lp, walk, reward))
                        player_data[idx] = []

                    active = active & ~dones
                    env.auto_reset()

        # PPO update after collecting episodes for this iteration
        if len(episodes) > 64:
            all_obs = torch.stack([e[0] for e in episodes])
            all_acts = torch.stack([e[1] for e in episodes])
            all_lps = torch.stack([e[2] for e in episodes])
            all_walks = torch.stack([e[3] for e in episodes])
            all_rews = torch.tensor([e[4] for e in episodes], device='cuda')

            # Advantage: reward - walk_value, then normalize
            advantages = all_rews - all_walks
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            bs = len(episodes)
            batch_size = 512
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
                    surr1 = ratio * advantages[mb]
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[mb]
                    loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

        last_avg_reward = iter_reward / iter_games if iter_games > 0 else 0.0

    return last_avg_reward


def compute_payoff_matrix_history(
    p1_policies: List[nn.Module],
    p2_policies: List[nn.Module],
    num_envs: int = 2048,
    num_episodes: int = 1000,
    device: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical payoff matrices for history-aware transformer policies.
    """
    n_p1 = len(p1_policies)
    n_p2 = len(p2_policies)
    payoff_p1 = np.zeros((n_p1, n_p2))
    payoff_p2 = np.zeros((n_p1, n_p2))

    env = BargainEnv(num_envs=num_envs, self_play=True, device=device)

    for i, policy_p1 in enumerate(p1_policies):
        for j, policy_p2 in enumerate(p2_policies):
            total_r1 = 0.0
            total_r2 = 0.0
            games_played = 0

            while games_played < num_episodes:
                obs, info = env.reset()
                action_mask = info['action_mask']
                active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

                # History buffers
                history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
                turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

                for step in range(12):
                    if not active.any():
                        break

                    current_player = env.get_current_player()
                    p1_turn = (current_player == 0) & active
                    p2_turn = (current_player == 1) & active

                    actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

                    # Update history with current observation
                    active_envs = active.nonzero().squeeze(-1)
                    if active_envs.numel() > 0:
                        tc = turn_count[active_envs].clamp(max=MAX_SEQ_LEN - 1)
                        history[active_envs, tc, :OBS_DIM] = obs[active_envs]
                        history[active_envs, tc, TOKEN_DIM - 1] = 1.0

                    if p1_turn.any():
                        p1_indices = p1_turn.nonzero().squeeze(-1)
                        p1_hist = history[p1_indices]
                        p1_seq_lens = turn_count[p1_indices] + 1
                        p1_am = action_mask[p1_indices]
                        with torch.no_grad():
                            logits, _ = policy_p1(p1_hist, p1_seq_lens, p1_am)
                            probs = F.softmax(logits, dim=-1)
                            p1_acts = torch.distributions.Categorical(probs).sample()
                        actions[p1_indices] = p1_acts

                    if p2_turn.any():
                        p2_indices = p2_turn.nonzero().squeeze(-1)
                        p2_hist = history[p2_indices]
                        p2_seq_lens = turn_count[p2_indices] + 1
                        p2_am = action_mask[p2_indices]
                        with torch.no_grad():
                            logits, _ = policy_p2(p2_hist, p2_seq_lens, p2_am)
                            probs = F.softmax(logits, dim=-1)
                            p2_acts = torch.distributions.Categorical(probs).sample()
                        actions[p2_indices] = p2_acts

                    # Update history with actions taken
                    if active_envs.numel() > 0:
                        tc = turn_count[active_envs].clamp(max=MAX_SEQ_LEN - 1)
                        action_onehot = F.one_hot(actions[active_envs], num_classes=NUM_ACTIONS).float()
                        history[active_envs, tc, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
                        turn_count[active_envs] = (turn_count[active_envs] + 1).clamp(max=MAX_SEQ_LEN - 1)

                    obs, rewards, dones, _, info = env.step(actions)
                    action_mask = info['action_mask']

                    if dones.any():
                        done_mask = dones & active
                        total_r1 += rewards[done_mask, 0].sum().item()
                        total_r2 += rewards[done_mask, 1].sum().item()
                        games_played += done_mask.sum().item()

                        # Reset history for done envs
                        for idx in dones.nonzero().squeeze(-1):
                            history[idx] = 0
                            turn_count[idx] = 0

                        active = active & ~dones
                        env.auto_reset()

            payoff_p1[i, j] = total_r1 / games_played
            payoff_p2[i, j] = total_r2 / games_played

    return payoff_p1, payoff_p2


def train_best_response_history(
    policy: nn.Module,
    opponent_policies: List[nn.Module],
    opponent_meta_strategy: np.ndarray,
    player_id: int,
    num_envs: int = 4096,
    num_episodes: int = 5000,
    ppo_epochs: int = 4,
    lr: float = 1e-3,
    device: int = 0,
    episodes_per_iter: int = 3000,  # Episodes per PPO update iteration
) -> float:
    """
    Train a best response policy (history transformer) against opponent's meta-strategy.
    
    Uses iterative PPO training for proper RL learning.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    env = BargainEnv(num_envs=num_envs, self_play=True, device=device)

    def sample_opponent_idx():
        return np.random.choice(len(opponent_policies), p=opponent_meta_strategy)

    # Calculate number of iterations
    num_iterations = max(1, num_episodes // episodes_per_iter)
    #num_iterations = 100
    
    total_reward = 0.0
    total_games = 0
    last_avg_reward = 0.0

    for iteration in range(num_iterations):
        # Collect episodes for this iteration
        episodes = []
        iter_reward = 0.0
        iter_games = 0

        while iter_games < episodes_per_iter:
            obs, info = env.reset()
            action_mask = info['action_mask']

            opp_indices = [sample_opponent_idx() for _ in range(num_envs)]

            # History buffers
            history = torch.zeros((num_envs, MAX_SEQ_LEN, TOKEN_DIM), device='cuda')
            turn_count = torch.zeros(num_envs, dtype=torch.long, device='cuda')

            player_data = [[] for _ in range(num_envs)]
            active = torch.ones(num_envs, dtype=torch.bool, device='cuda')

            for step in range(12):
                if not active.any():
                    break

                current_player = env.get_current_player()
                our_turn = (current_player == player_id) & active
                opp_turn = (current_player == (1 - player_id)) & active

                actions = torch.zeros(num_envs, dtype=torch.long, device='cuda')

                # Update history with current observation
                active_envs = active.nonzero().squeeze(-1)
                if active_envs.numel() > 0:
                    tc = turn_count[active_envs].clamp(max=MAX_SEQ_LEN - 1)
                    history[active_envs, tc, :OBS_DIM] = obs[active_envs]
                    history[active_envs, tc, TOKEN_DIM - 1] = 1.0

                # Our policy's turn
                if our_turn.any():
                    our_indices = our_turn.nonzero().squeeze(-1)
                    our_hist = history[our_indices]
                    our_seq_lens = turn_count[our_indices] + 1
                    our_am = action_mask[our_indices]
                    our_walk = obs[our_indices, 3]

                    with torch.no_grad():
                        logits, _ = policy(our_hist, our_seq_lens, our_am)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        our_acts = dist.sample()
                        our_lps = dist.log_prob(our_acts)
                    actions[our_indices] = our_acts

                    for i, idx in enumerate(our_indices):
                        idx_item = idx.item()
                        hist_snap = history[idx].clone()
                        seq_len = (turn_count[idx] + 1).clone()
                        player_data[idx_item].append((
                            hist_snap, seq_len, our_acts[i].clone(), our_lps[i].clone(), our_walk[i].clone()
                        ))

                # Opponent's turn
                if opp_turn.any():
                    opp_indices_tensor = torch.tensor(opp_indices, device='cuda')
                    for opp_idx in range(len(opponent_policies)):
                        mask = (opp_indices_tensor == opp_idx) & opp_turn
                        if mask.any():
                            opp_env_indices = mask.nonzero().squeeze(-1)
                            opp_hist = history[opp_env_indices]
                            opp_seq_lens = turn_count[opp_env_indices] + 1
                            opp_am = action_mask[opp_env_indices]
                            with torch.no_grad():
                                logits, _ = opponent_policies[opp_idx](opp_hist, opp_seq_lens, opp_am)
                                probs = F.softmax(logits, dim=-1)
                                opp_acts = torch.distributions.Categorical(probs).sample()
                            actions[opp_env_indices] = opp_acts

                # Update history with actions taken
                if active_envs.numel() > 0:
                    tc = turn_count[active_envs].clamp(max=MAX_SEQ_LEN - 1)
                    action_onehot = F.one_hot(actions[active_envs], num_classes=NUM_ACTIONS).float()
                    history[active_envs, tc, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
                    turn_count[active_envs] = (turn_count[active_envs] + 1).clamp(max=MAX_SEQ_LEN - 1)

                obs, rewards, dones, _, info = env.step(actions)
                action_mask = info['action_mask']

                if dones.any():
                    for idx in dones.nonzero().squeeze(-1):
                        idx_item = idx.item()
                        if not active[idx]:
                            continue
                        reward = rewards[idx, player_id].item()
                        iter_reward += reward
                        iter_games += 1
                        total_reward += reward
                        total_games += 1

                        for hist, seq_len, act, lp, walk in player_data[idx_item]:
                            episodes.append((hist, seq_len, act, lp, walk, reward))
                        player_data[idx_item] = []

                        # Reset history for done env
                        history[idx] = 0
                        turn_count[idx] = 0

                    active = active & ~dones
                    env.auto_reset()

        # PPO update after collecting episodes for this iteration
        if len(episodes) > 64:
            all_hist = torch.stack([e[0] for e in episodes])
            all_seq_lens = torch.stack([e[1] for e in episodes])
            all_acts = torch.stack([e[2] for e in episodes])
            all_lps = torch.stack([e[3] for e in episodes])
            all_walks = torch.stack([e[4] for e in episodes])
            all_rews = torch.tensor([e[5] for e in episodes], device='cuda')

            advantages = all_rews - all_walks
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            bs = len(episodes)
            batch_size = 512
            for _ in range(ppo_epochs):
                idx = torch.randperm(bs, device='cuda')
                for s in range(0, bs, batch_size):
                    mb = idx[s:min(s + batch_size, bs)]

                    mb_hist = all_hist[mb]
                    mb_seq_lens = all_seq_lens[mb]

                    # Get action mask from last token's observation
                    batch_idx = torch.arange(len(mb), device='cuda')
                    last_idx = (mb_seq_lens - 1).clamp(min=0)
                    last_obs = mb_hist[batch_idx, last_idx, :OBS_DIM]
                    masks = last_obs[:, 10:92]

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

        last_avg_reward = iter_reward / iter_games if iter_games > 0 else 0.0

    return last_avg_reward


def train_psro(
    num_iterations: int = 20,
    num_envs: int = 4096,
    episodes_per_br: int = 10000,
    episodes_per_eval: int = 2000,
    lr: float = 1e-3,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-psro",
    wandb_run_name: str = None,
    log_interval: int = 1,
    architecture: str = "mlp",  # "mlp" or "transformer"
    exploitability_interval: int = 0,  # 0 = only at end (PSRO already tracks NashConv)
    warmstart_p1: str = None,  # Path to pre-trained P1 policy for warm-start
    warmstart_p2: str = None,  # Path to pre-trained P2 policy for warm-start
    resume_checkpoint: str = None,  # Path to PSRO checkpoint to resume from
):
    """
    Train policies using PSRO.

    Args:
        num_iterations: Number of PSRO iterations
        num_envs: Parallel environments for simulation
        episodes_per_br: Episodes for training each best response
        episodes_per_eval: Episodes for payoff matrix evaluation
        lr: Learning rate for best response training
        seed: Random seed
        save_dir: Directory to save results
        use_wandb: Enable wandb logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        log_interval: How often to log metrics
        architecture: "mlp" or "transformer" (history-aware)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    use_history = architecture == "transformer"
    start_iteration = 0  # Will be updated if resuming from checkpoint

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "PSRO",
                "architecture": architecture,
                "num_iterations": num_iterations,
                "num_envs": num_envs,
                "episodes_per_br": episodes_per_br,
                "episodes_per_eval": episodes_per_eval,
                "lr": lr,
                "seed": seed,
                "resume_checkpoint": resume_checkpoint,
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping logging")

    # Track metrics
    nashconv_history = []
    p1_population_size = []
    p2_population_size = []

    # Resume from checkpoint if provided
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location='cuda', weights_only=False)

        # Get architecture from checkpoint (override if different)
        ckpt_arch = checkpoint.get('architecture', 'mlp')
        if ckpt_arch != architecture:
            print(f"Warning: Checkpoint architecture '{ckpt_arch}' differs from requested '{architecture}'")
            print(f"Using checkpoint architecture: {ckpt_arch}")
            architecture = ckpt_arch
            use_history = architecture == "transformer"

        # Recreate policies from checkpoint
        p1_state_dicts = checkpoint['p1_policies']
        p2_state_dicts = checkpoint['p2_policies']

        p1_policies: List[nn.Module] = []
        p2_policies: List[nn.Module] = []

        print(f"Loading {len(p1_state_dicts)} P1 policies and {len(p2_state_dicts)} P2 policies...")

        for i, state_dict in enumerate(p1_state_dicts):
            if use_history:
                policy = HistoryTransformerPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
            else:
                policy = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()
            policy.load_state_dict(state_dict)
            policy.eval()
            p1_policies.append(policy)

        for i, state_dict in enumerate(p2_state_dicts):
            if use_history:
                policy = HistoryTransformerPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
            else:
                policy = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()
            policy.load_state_dict(state_dict)
            policy.eval()
            p2_policies.append(policy)

        # Load training history
        nashconv_history = checkpoint.get('nashconv_history', [])
        start_iteration = checkpoint.get('iteration', len(nashconv_history))

        # Load payoff matrices (avoids recomputing all pairs on resume)
        loaded_payoff_p1 = checkpoint.get('payoff_p1', None)
        loaded_payoff_p2 = checkpoint.get('payoff_p2', None)

        print(f"Resumed with {len(p1_policies)} P1 policies, {len(p2_policies)} P2 policies")
        print(f"Resuming from iteration {start_iteration}")
        if loaded_payoff_p1 is not None:
            print(f"Loaded payoff matrix: {loaded_payoff_p1.shape}")
        if nashconv_history:
            print(f"Last NashConv values: {[f'{nc:.4f}' for nc in nashconv_history[-5:]]}")

    # Initialize populations from scratch (with optional warm-start from pre-trained policies)
    else:
        loaded_payoff_p1 = None
        loaded_payoff_p2 = None

        if use_history:
            p1_init = HistoryTransformerPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
            p2_init = HistoryTransformerPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
            if warmstart_p1:
                print(f"Warm-starting P1 from: {warmstart_p1}")
                p1_init.load_state_dict(torch.load(warmstart_p1, map_location='cuda'))
            if warmstart_p2:
                print(f"Warm-starting P2 from: {warmstart_p2}")
                p2_init.load_state_dict(torch.load(warmstart_p2, map_location='cuda'))
            p1_policies: List[nn.Module] = [p1_init]
            p2_policies: List[nn.Module] = [p2_init]
        else:
            p1_init = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()
            p2_init = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()
            if warmstart_p1:
                print(f"Warm-starting P1 from: {warmstart_p1}")
                p1_init.load_state_dict(torch.load(warmstart_p1, map_location='cuda'))
            if warmstart_p2:
                print(f"Warm-starting P2 from: {warmstart_p2}")
                p2_init.load_state_dict(torch.load(warmstart_p2, map_location='cuda'))
            p1_policies: List[nn.Module] = [p1_init]
            p2_policies: List[nn.Module] = [p2_init]

    print("=" * 60)
    print("PSRO TRAINING")
    print("=" * 60)
    print(f"Architecture: {architecture}")
    print(f"Iterations: {num_iterations}")
    print(f"Starting from iteration: {start_iteration}")
    print(f"Episodes per BR: {episodes_per_br}")
    print(f"Episodes per eval: {episodes_per_eval}")
    if resume_checkpoint:
        print(f"Resumed from: {resume_checkpoint}")
    print()

    for iteration in range(start_iteration, num_iterations):
        print(f"\n--- PSRO Iteration {iteration + 1}/{num_iterations} ---")

        # Step 1: Compute empirical payoff matrix
        # On first iteration after resume, use loaded payoff matrix if available
        if iteration == start_iteration and loaded_payoff_p1 is not None and loaded_payoff_p2 is not None:
            print("Using loaded payoff matrix from checkpoint...")
            payoff_p1 = loaded_payoff_p1
            payoff_p2 = loaded_payoff_p2
            # Clear loaded payoffs so subsequent iterations recompute
            loaded_payoff_p1 = None
            loaded_payoff_p2 = None
        else:
            print("Computing payoff matrix...")
            if use_history:
                payoff_p1, payoff_p2 = compute_payoff_matrix_history(
                    p1_policies, p2_policies,
                    num_envs=num_envs,
                    num_episodes=episodes_per_eval,
                )
            else:
                payoff_p1, payoff_p2 = compute_payoff_matrix(
                    p1_policies, p2_policies,
                    num_envs=num_envs,
                    num_episodes=episodes_per_eval,
                )

        # Step 2: Solve for Nash equilibrium
        print("Solving meta-game Nash equilibrium...")
        sigma_p1, sigma_p2 = solve_nash_equilibrium(payoff_p1, payoff_p2, use_max_entropy=True, entropy_weight = .001)

        # Compute current meta-strategy values (for NashConv calculation later)
        v1_current = sigma_p1 @ payoff_p1 @ sigma_p2
        v2_current = sigma_p1 @ payoff_p2 @ sigma_p2

        print(f"Population sizes: P1={len(p1_policies)}, P2={len(p2_policies)}")
        print(f"Meta-strategies: P1={sigma_p1.round(3)}, P2={sigma_p2.round(3)}")
        print(f"Meta-values: P1={v1_current:.4f}, P2={v2_current:.4f}")

        # Step 4: Train best responses
        print("Training P1 best response...")
        if use_history:
            new_p1 = HistoryTransformerPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
            br_reward_p1 = train_best_response_history(
                new_p1, p2_policies, sigma_p2,
                player_id=0,
                num_envs=num_envs,
                num_episodes=episodes_per_br,
                lr=lr,
            )
        else:
            new_p1 = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()
            br_reward_p1 = train_best_response(
                new_p1, p2_policies, sigma_p2,
                player_id=0,
                num_envs=num_envs,
                num_episodes=episodes_per_br,
                lr=lr,
            )

        print("Training P2 best response...")
        if use_history:
            new_p2 = HistoryTransformerPolicy(TOKEN_DIM, NUM_ACTIONS).cuda()
            br_reward_p2 = train_best_response_history(
                new_p2, p1_policies, sigma_p1,
                player_id=1,
                num_envs=num_envs,
                num_episodes=episodes_per_br,
                lr=lr,
            )
        else:
            new_p2 = PolicyNetwork(OBS_DIM, NUM_ACTIONS).cuda()
            br_reward_p2 = train_best_response(
                new_p2, p1_policies, sigma_p1,
                player_id=1,
                num_envs=num_envs,
                num_episodes=episodes_per_br,
                lr=lr,
            )

        print(f"BR training rewards: P1={br_reward_p1:.4f}, P2={br_reward_p2:.4f}")

        # Step 5: EVALUATE trained BRs against opponent meta-strategy
        # This gives the TRUE BR value (not noisy training average)
        print("Evaluating trained best responses...")
        if use_history:
            br_eval_p1 = evaluate_br_vs_meta_history(
                new_p1, p2_policies, sigma_p2, player_id=0,
                num_envs=num_envs, num_episodes=episodes_per_eval,
            )
            br_eval_p2 = evaluate_br_vs_meta_history(
                new_p2, p1_policies, sigma_p1, player_id=1,
                num_envs=num_envs, num_episodes=episodes_per_eval,
            )
        else:
            br_eval_p1 = evaluate_br_vs_meta(
                new_p1, p2_policies, sigma_p2, player_id=0,
                num_envs=num_envs, num_episodes=episodes_per_eval,
            )
            br_eval_p2 = evaluate_br_vs_meta(
                new_p2, p1_policies, sigma_p1, player_id=1,
                num_envs=num_envs, num_episodes=episodes_per_eval,
            )

        # Step 6: Compute TRUE NashConv using EVALUATED BR values
        # NashConv = how much each player can gain by deviating from meta-strategy
        exploit_p1 = max(0, br_eval_p1 - v1_current)  # How much P1's BR beats meta-value
        exploit_p2 = max(0, br_eval_p2 - v2_current)  # How much P2's BR beats meta-value
        nashconv = exploit_p1 + exploit_p2
        nashconv_history.append(nashconv)
        p1_population_size.append(len(p1_policies))
        p2_population_size.append(len(p2_policies))

        print(f"BR eval values: P1={br_eval_p1:.4f}, P2={br_eval_p2:.4f}")
        print(f"NashConv: {nashconv:.4f} (P1: {exploit_p1:.4f}, P2: {exploit_p2:.4f})")

        # Step 7: Add best responses to populations
        p1_policies.append(new_p1)
        p2_policies.append(new_p2)

        # Periodic checkpoint saving (every 5 iterations)
        if (iteration + 1) % 5 == 0:
            checkpoint_path = save_path / f"checkpoint_iter_{iteration+1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                "p1_policies": [p.state_dict() for p in p1_policies],
                "p2_policies": [p.state_dict() for p in p2_policies],
                "payoff_p1": payoff_p1,
                "payoff_p2": payoff_p2,
                "sigma_p1": sigma_p1,
                "sigma_p2": sigma_p2,
                "nashconv_history": nashconv_history,
                "iteration": iteration + 1,
                "architecture": architecture,
            }, checkpoint_path / "psro_checkpoint.pt")
            print(f"Checkpoint saved to {checkpoint_path}")

        # Measure PSRO mixture exploitability (every iteration)
        psro_exploit = None
        if not use_history:  # Only for MLP for now (history requires more work)
            print(f"Measuring PSRO mixture exploitability...")
            psro_exploit = measure_psro_exploitability_inline(
                p1_policies, p2_policies, sigma_p1, sigma_p2,
                num_envs=num_envs, br_iterations=50, br_episodes_per_iter=5000,
                eval_episodes=10000, seed=seed + iteration
            )
            print(f"PSRO Mixture NashConv: {psro_exploit['psro_exploitability/total']:.4f}")

        # Log to wandb
        if use_wandb and WANDB_AVAILABLE and (iteration + 1) % log_interval == 0:
            log_dict = {
                "iteration": iteration + 1,
                "nashconv": nashconv,
                "p1/exploitability": exploit_p1,
                "p2/exploitability": exploit_p2,
                "p1/br_value": br_eval_p1,
                "p2/br_value": br_eval_p2,
                "p1/meta_value": v1_current,
                "p2/meta_value": v2_current,
                "p1/population_size": len(p1_policies),
                "p2/population_size": len(p2_policies),
            }
            if psro_exploit is not None:
                log_dict["psro_mixture_nashconv"] = psro_exploit['psro_exploitability/total']
            wandb.log(log_dict)

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    if use_history:
        payoff_p1, payoff_p2 = compute_payoff_matrix_history(
            p1_policies, p2_policies,
            num_envs=num_envs,
            num_episodes=episodes_per_eval * 2,
        )
    else:
        payoff_p1, payoff_p2 = compute_payoff_matrix(
            p1_policies, p2_policies,
            num_envs=num_envs,
            num_episodes=episodes_per_eval * 2,
        )
    sigma_p1, sigma_p2 = solve_nash_equilibrium(payoff_p1, payoff_p2, use_max_entropy=True)
    
    # Final meta-strategy values
    v1_final = sigma_p1 @ payoff_p1 @ sigma_p2
    v2_final = sigma_p1 @ payoff_p2 @ sigma_p2

    print(f"Final NashConv (last iteration): {nashconv_history[-1]:.4f}")
    print(f"Final meta-strategy values: P1={v1_final:.4f}, P2={v2_final:.4f}")
    print(f"Final meta-strategies:")
    print(f"  P1: {sigma_p1.round(3)}")
    print(f"  P2: {sigma_p2.round(3)}")

    # Final PSRO mixture exploitability measurement
    print("\nMeasuring final PSRO mixture exploitability...")
    if not use_history:  # MLP architecture
        psro_exploit_final = measure_psro_exploitability_inline(
            p1_policies, p2_policies, sigma_p1, sigma_p2,
            num_envs=num_envs, br_iterations=100, br_episodes_per_iter=2000,
            eval_episodes=10000, seed=seed
        )
        print(f"Final PSRO Mixture NashConv: {psro_exploit_final['psro_exploitability/total']:.4f}")
        print(f"  Self-play: P1={psro_exploit_final['psro_exploitability/selfplay_r1']:.4f}, P2={psro_exploit_final['psro_exploitability/selfplay_r2']:.4f}")
        print(f"  Exploitability: P1={psro_exploit_final['psro_exploitability/p1']:.4f}, P2={psro_exploit_final['psro_exploitability/p2']:.4f}")
    else:
        # Fallback for history transformer (measure last BRs only)
        print("(History transformer: measuring last BR policies only)")
        exploit_metrics = measure_exploitability(
            p1_policies[-1], p2_policies[-1], "history_transformer", seed=seed
        )
        psro_exploit_final = {"psro_exploitability/total": exploit_metrics['exploitability/total']}
        print(f"Final exploitability (last BRs): {exploit_metrics['exploitability/total']:.4f}")

    # Save results
    torch.save({
        "p1_policies": [p.state_dict() for p in p1_policies],
        "p2_policies": [p.state_dict() for p in p2_policies],
        "payoff_p1": payoff_p1,
        "payoff_p2": payoff_p2,
        "sigma_p1": sigma_p1,
        "sigma_p2": sigma_p2,
        "nashconv_history": nashconv_history,
        "architecture": architecture,
    }, save_path / "psro_result.pt")
    print(f"\nResults saved to {save_path / 'psro_result.pt'}")

    # Plot NashConv
    plt.figure(figsize=(10, 6))
    plt.plot(nashconv_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('PSRO Iteration')
    plt.ylabel('NashConv (Exploitability)')
    plt.title('PSRO Training: NashConv over Iterations')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path / "psro_nashconv.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path / 'psro_nashconv.png'}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.save(str(save_path / "psro_result.pt"))
        wandb.log({"nashconv_plot": wandb.Image(str(save_path / "psro_nashconv.png"))})
        wandb.log({f"final/{k.split('/')[-1]}": v for k, v in psro_exploit_final.items()})
        wandb.finish()

    return p1_policies, p2_policies, sigma_p1, sigma_p2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PSRO Training for Bargaining Game")
    parser.add_argument("--iterations", type=int, default=20, help="PSRO iterations")
    parser.add_argument("--num-envs", type=int, default=4096, help="Parallel environments")
    parser.add_argument("--episodes-per-br", type=int, default=3_000_000, help="Episodes for BR training")
    parser.add_argument("--episodes-per-eval", type=int, default=10_000, help="Episodes for payoff eval")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default=".", help="Save directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--wandb-project", type=str, default="bargaining-psro", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--log-interval", type=int, default=1, help="Log every N iterations")
    parser.add_argument("--architecture", type=str, default="mlp", choices=["mlp", "transformer"],
                        help="Policy architecture: mlp or transformer (history-aware)")
    parser.add_argument("--exploitability-interval", type=int, default=0, help="Measure exploitability every N iterations (0 = only at end)")
    parser.add_argument("--warmstart-p1", type=str, default=None, help="Path to pre-trained P1 policy for warm-start")
    parser.add_argument("--warmstart-p2", type=str, default=None, help="Path to pre-trained P2 policy for warm-start")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to PSRO checkpoint to resume from (loads full population)")

    args = parser.parse_args()

    train_psro(
        num_iterations=args.iterations,
        num_envs=args.num_envs,
        episodes_per_br=args.episodes_per_br,
        episodes_per_eval=args.episodes_per_eval,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_interval=args.log_interval,
        architecture=args.architecture,
        exploitability_interval=args.exploitability_interval,
        warmstart_p1=args.warmstart_p1,
        warmstart_p2=args.warmstart_p2,
        resume_checkpoint=args.resume_checkpoint,
    )