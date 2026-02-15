#!/usr/bin/env python3
"""
NFSP (Neural Fictitious Self-Play) training for the CUDA bargaining game.

NFSP combines:
1. Best Response learning via DQN (Q-network)
2. Average Policy learning via supervised learning (Policy network)

Key idea:
- With probability eta: play best response (sample from softmax over Q-values)
- With probability 1-eta: play average policy

The average policy converges to an approximate Nash equilibrium.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from cuda_bargain import BargainEnv, NUM_ACTIONS, OBS_DIM

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from .policy import QNetwork, AveragePolicyNetwork, QNetworkHistory, AveragePolicyNetworkHistory
    from .buffers import (
        NFSPBuffers, TransitionTuple, PolicyTuple,
        NFSPBuffersHistory, HistoryTransitionTuple, HistoryPolicyTuple,
    )
    from ..exploitability import measure_exploitability
except ImportError:
    from policy import QNetwork, AveragePolicyNetwork, QNetworkHistory, AveragePolicyNetworkHistory
    from buffers import (
        NFSPBuffers, TransitionTuple, PolicyTuple,
        NFSPBuffersHistory, HistoryTransitionTuple, HistoryPolicyTuple,
    )
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from exploitability import measure_exploitability


class _NFSPPolicyWrapper(nn.Module):
    """Wrapper to make AveragePolicyNetwork compatible with exploitability measurement."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, obs, action_mask):
        return self.policy(obs, action_mask), None


class _NFSPHistoryPolicyWrapper(nn.Module):
    """Wrapper to make AveragePolicyNetworkHistory compatible with exploitability measurement."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, history, seq_lens, action_mask):
        return self.policy(history, seq_lens, action_mask), None


def train_nfsp(
    num_envs: int = 1024,
    num_iterations: int = 1000,
    episodes_per_iter: int = 500,
    eta: float = 0.1,
    epsilon: float = 0.06,
    gamma: float = 0.99,
    q_lr: float = 1e-4,
    policy_lr: float = 1e-3,
    batch_size: int = 256,
    q_buffer_size: int = 500_000,
    policy_buffer_size: int = 2_000_000,
    target_update_freq: int = 300,
    updates_per_iter: int = 2,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-nfsp",
    wandb_run_name: str = None,
    log_interval: int = 10,
    save_interval: int = 0,  # Periodic checkpoint saving (0 = disabled)
    exploitability_interval: int = 0,  # 0 = only at end
):
    """
    Train agents using NFSP.

    NFSP alternates between:
    1. Generating episodes with mixed strategy (eta best-response, 1-eta average)
    2. Training Q-network with DQN on transitions from best-response play
    3. Training policy network to match actions from best-response play

    Args:
        num_envs: Number of parallel environments
        num_iterations: Training iterations
        episodes_per_iter: Episodes to collect per iteration
        eta: Anticipatory parameter (probability of using best response)
        epsilon: Epsilon for epsilon-greedy exploration in best response
        gamma: Discount factor for Q-learning (standard DQN uses 0.99)
        q_lr: Learning rate for Q-network
        policy_lr: Learning rate for policy network
        batch_size: Batch size for training
        q_buffer_size: Size of Q-learning replay buffer
        policy_buffer_size: Size of average policy reservoir buffer
        target_update_freq: How often to update target Q-network
        updates_per_iter: Number of gradient updates per iteration
        seed: Random seed
        save_dir: Directory to save models
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        log_interval: How often to log metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "NFSP",
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "episodes_per_iter": episodes_per_iter,
                "eta": eta,
                "epsilon": epsilon,
                "gamma": gamma,
                "q_lr": q_lr,
                "policy_lr": policy_lr,
                "batch_size": batch_size,
                "q_buffer_size": q_buffer_size,
                "policy_buffer_size": policy_buffer_size,
                "seed": seed,
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping logging")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create environment
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # Create networks for both players
    q_nets = [QNetwork(OBS_DIM, NUM_ACTIONS).to(device) for _ in range(2)]
    q_nets_target = [deepcopy(q_net) for q_net in q_nets]
    policy_nets = [AveragePolicyNetwork(OBS_DIM, NUM_ACTIONS).to(device) for _ in range(2)]

    # Optimizers
    q_optimizers = [torch.optim.Adam(q_net.parameters(), lr=q_lr) for q_net in q_nets]
    policy_optimizers = [torch.optim.Adam(policy_net.parameters(), lr=policy_lr) for policy_net in policy_nets]

    # Replay buffers
    buffers = NFSPBuffers(
        num_players=2,
        q_buffer_size=q_buffer_size,
        policy_buffer_size=policy_buffer_size,
    )

    reward_history = [[], []]
    q_loss_history = []
    policy_loss_history = []

    print("=" * 60)
    print("NFSP TRAINING")
    print("=" * 60)
    print(f"Environments: {num_envs}")
    print(f"Iterations: {num_iterations}")
    print(f"Episodes per iteration: {episodes_per_iter}")
    print(f"Eta (best response prob): {eta}")
    print(f"Epsilon (exploration): {epsilon}")
    print(f"Gamma (discount factor): {gamma}")
    print()

    total_updates = 0

    for iteration in range(num_iterations):
        # =====================
        # 1. Generate episodes
        # =====================
        total_episodes = 0
        episode_rewards = [[], []]

        while total_episodes < episodes_per_iter:
            obs, info = env.reset()
            action_mask = info['action_mask']

            # Decide strategy for each env and player at episode start
            # use_br[env_idx][player] = True if using best response
            use_br = [[torch.rand(1).item() < eta for _ in range(2)] for _ in range(num_envs)]

            # Track last state/action for each player in each env
            last_obs = [{} for _ in range(num_envs)]
            last_action = [{} for _ in range(num_envs)]
            last_mask = [{} for _ in range(num_envs)]

            active = torch.ones(num_envs, dtype=torch.bool, device=device)

            for step in range(12):  # Max steps
                if not active.any():
                    break

                current_player = env.get_current_player()
                actions = torch.zeros(num_envs, dtype=torch.long, device=device)

                for p in range(2):
                    player_mask = (current_player == p) & active
                    if not player_mask.any():
                        continue

                    indices = player_mask.nonzero().squeeze(-1)
                    p_obs = obs[indices]
                    p_am = action_mask[indices]

                    # Select actions based on strategy
                    with torch.no_grad():
                        for i, idx in enumerate(indices):
                            idx_item = idx.item()
                            single_obs = p_obs[i]
                            single_mask = p_am[i]

                            if use_br[idx_item][p]:
                                # Best response: epsilon-greedy over Q-values
                                if torch.rand(1).item() < epsilon:
                                    # Random action from legal actions
                                    legal_indices = single_mask.nonzero().squeeze(-1)
                                    action = legal_indices[torch.randint(len(legal_indices), (1,))].item()
                                else:
                                    # Greedy action (argmax of Q-values)
                                    q_vals = q_nets[p](single_obs.unsqueeze(0), single_mask.unsqueeze(0))
                                    action = q_vals.argmax(dim=-1).item()

                                # Store in policy buffer (action from BR play)
                                buffers.add_policy_sample(p, PolicyTuple(
                                    obs=single_obs.cpu(),
                                    action_mask=single_mask.cpu(),
                                    action=action,
                                ))
                            else:
                                # Average policy
                                logits = policy_nets[p](single_obs.unsqueeze(0), single_mask.unsqueeze(0))
                                probs = F.softmax(logits, dim=-1)
                                action = torch.multinomial(probs, 1).item()

                            actions[idx] = action

                            # Store for transition
                            if p in last_obs[idx_item]:
                                # Add transition from last state
                                buffers.add_transition(p, TransitionTuple(
                                    obs=last_obs[idx_item][p],
                                    action_mask=last_mask[idx_item][p],
                                    action=last_action[idx_item][p],
                                    reward=0.0,  # Intermediate reward
                                    next_obs=single_obs.cpu(),
                                    next_action_mask=single_mask.cpu(),
                                    done=False,
                                ))

                            last_obs[idx_item][p] = single_obs.cpu()
                            last_action[idx_item][p] = action
                            last_mask[idx_item][p] = single_mask.cpu()

                obs, rewards, dones, _, info = env.step(actions)
                action_mask = info['action_mask']

                # Handle terminal states
                if dones.any():
                    for idx in dones.nonzero().squeeze(-1):
                        idx_item = idx.item()

                        for p in range(2):
                            if p in last_obs[idx_item]:
                                # Final transition with terminal reward
                                final_obs = obs[idx].cpu()
                                final_mask = action_mask[idx].cpu()
                                reward = rewards[idx, p].item()

                                buffers.add_transition(p, TransitionTuple(
                                    obs=last_obs[idx_item][p],
                                    action_mask=last_mask[idx_item][p],
                                    action=last_action[idx_item][p],
                                    reward=reward,
                                    next_obs=final_obs,
                                    next_action_mask=final_mask,
                                    done=True,
                                ))

                                episode_rewards[p].append(reward)

                        # Reset tracking for this env
                        last_obs[idx_item] = {}
                        last_action[idx_item] = {}
                        last_mask[idx_item] = {}
                        use_br[idx_item] = [torch.rand(1).item() < eta for _ in range(2)]

                        total_episodes += 1

                    active = active & ~dones
                    env.auto_reset()

        # =====================
        # 2. Train networks
        # =====================
        q_losses = []
        policy_losses = []

        for _ in range(updates_per_iter):
            for p in range(2):
                # Train Q-network
                if buffers.q_buffer_size(p) >= batch_size:
                    batch = buffers.sample_transitions(p, batch_size)

                    obs_batch = torch.stack([t.obs for t in batch]).to(device)
                    mask_batch = torch.stack([t.action_mask for t in batch]).to(device)
                    action_batch = torch.tensor([t.action for t in batch], device=device)
                    reward_batch = torch.tensor([t.reward for t in batch], device=device, dtype=torch.float32)
                    next_obs_batch = torch.stack([t.next_obs for t in batch]).to(device)
                    next_mask_batch = torch.stack([t.next_action_mask for t in batch]).to(device)
                    done_batch = torch.tensor([t.done for t in batch], device=device, dtype=torch.float32)

                    # Current Q-values
                    q_values = q_nets[p](obs_batch, mask_batch)
                    q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                    # Standard DQN target: r + gamma * max_a' Q_target(s', a') * (1 - done)
                    with torch.no_grad():
                        next_q_values = q_nets_target[p](next_obs_batch, next_mask_batch)
                        next_q_values = next_q_values.max(dim=1)[0]
                        target = reward_batch + gamma * next_q_values * (1 - done_batch)

                    q_loss = F.mse_loss(q_values, target)

                    q_optimizers[p].zero_grad()
                    q_loss.backward()
                    nn.utils.clip_grad_norm_(q_nets[p].parameters(), 10.0)
                    q_optimizers[p].step()

                    q_losses.append(q_loss.item())

                # Train policy network
                if buffers.policy_buffer_size(p) >= batch_size:
                    batch = buffers.sample_policy(p, batch_size)

                    obs_batch = torch.stack([t.obs for t in batch]).to(device)
                    mask_batch = torch.stack([t.action_mask for t in batch]).to(device)
                    action_batch = torch.tensor([t.action for t in batch], device=device)

                    # Supervised learning: cross-entropy loss
                    logits = policy_nets[p](obs_batch, mask_batch)
                    policy_loss = F.cross_entropy(logits, action_batch)

                    policy_optimizers[p].zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(policy_nets[p].parameters(), 10.0)
                    policy_optimizers[p].step()

                    policy_losses.append(policy_loss.item())

        total_updates += 1

        # Update target networks
        if total_updates % target_update_freq == 0:
            for p in range(2):
                q_nets_target[p].load_state_dict(q_nets[p].state_dict())

        # Record statistics
        for p in range(2):
            if episode_rewards[p]:
                reward_history[p].append(np.mean(episode_rewards[p]))

        if q_losses:
            q_loss_history.append(np.mean(q_losses))
        if policy_losses:
            policy_loss_history.append(np.mean(policy_losses))

        # Logging
        r1 = reward_history[0][-1] if reward_history[0] else 0
        r2 = reward_history[1][-1] if reward_history[1] else 0

        if use_wandb and WANDB_AVAILABLE and (iteration + 1) % log_interval == 0:
            log_dict = {
                "iteration": iteration + 1,
                "p1/reward": r1,
                "p2/reward": r2,
                "q_loss": q_loss_history[-1] if q_loss_history else 0,
                "policy_loss": policy_loss_history[-1] if policy_losses else 0,
                "q_buffer_size": buffers.q_buffer_size(0),
                "policy_buffer_size": buffers.policy_buffer_size(0),
                "episodes": total_episodes,
            }
            wandb.log(log_dict)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | "
                  f"Q-buf: {buffers.q_buffer_size(0)} | P-buf: {buffers.policy_buffer_size(0)}")

        # Periodic exploitability measurement (using average policy networks)
        if exploitability_interval > 0 and (iteration + 1) % exploitability_interval == 0:
            print(f"Measuring exploitability at iteration {iteration+1}...")
            exploit_metrics = measure_exploitability(
                _NFSPPolicyWrapper(policy_nets[0]), _NFSPPolicyWrapper(policy_nets[1]), "mlp", seed=seed
            )
            print(f"  Total exploitability: {exploit_metrics['exploitability/total']:.4f}")
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({**exploit_metrics, "iteration": iteration + 1})

        # Periodic checkpoint saving
        if save_interval > 0 and (iteration + 1) % save_interval == 0:
            checkpoint_path = save_path / f"checkpoint_iter_{iteration+1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            for p in range(2):
                torch.save(q_nets[p].state_dict(), checkpoint_path / f"nfsp_q_net_p{p+1}.pt")
                torch.save(policy_nets[p].state_dict(), checkpoint_path / f"nfsp_policy_net_p{p+1}.pt")
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save models
    for p in range(2):
        torch.save(q_nets[p].state_dict(), save_path / f"nfsp_q_net_p{p+1}.pt")
        torch.save(policy_nets[p].state_dict(), save_path / f"nfsp_policy_net_p{p+1}.pt")
    print(f"\nModels saved to {save_path}")

    # Plot training curves
    _plot_training(reward_history, q_loss_history, policy_loss_history, save_path / "nfsp_training_curves.png")

    # Final exploitability measurement (using average policy networks)
    print("Measuring final exploitability...")
    exploit_metrics = measure_exploitability(
        _NFSPPolicyWrapper(policy_nets[0]), _NFSPPolicyWrapper(policy_nets[1]), "mlp", seed=seed
    )
    print(f"Final exploitability: {exploit_metrics['exploitability/total']:.4f}")

    if use_wandb and WANDB_AVAILABLE:
        for p in range(2):
            wandb.save(str(save_path / f"nfsp_q_net_p{p+1}.pt"))
            wandb.save(str(save_path / f"nfsp_policy_net_p{p+1}.pt"))
        wandb.log({"training_curves": wandb.Image(str(save_path / "nfsp_training_curves.png"))})
        wandb.log({f"final/{k.split('/')[-1]}": v for k, v in exploit_metrics.items()})
        wandb.finish()

    return q_nets, policy_nets, reward_history


def train_nfsp_history(
    num_envs: int = 1024,
    num_iterations: int = 1000,
    episodes_per_iter: int = 500,
    eta: float = 0.1,
    epsilon: float = 0.06,
    gamma: float = 0.99,
    q_lr: float = 1e-4,
    policy_lr: float = 1e-3,
    batch_size: int = 256,
    q_buffer_size: int = 20_000_000,
    policy_buffer_size: int = 100_000_000,
    target_update_freq: int = 300,
    updates_per_iter: int = 2,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    seed: int = 42,
    save_dir: str = ".",
    use_wandb: bool = True,
    wandb_project: str = "bargaining-nfsp",
    wandb_run_name: str = None,
    log_interval: int = 10,
    save_interval: int = 0,  # Periodic checkpoint saving (0 = disabled)
    exploitability_interval: int = 0,  # 0 = only at end
):
    """
    Train agents using NFSP with History Transformer architecture.

    Uses transformer networks that process the full game history as a sequence,
    enabling agents to reason about past moves.

    Args:
        num_envs: Number of parallel environments
        num_iterations: Training iterations
        episodes_per_iter: Episodes to collect per iteration
        eta: Anticipatory parameter (probability of using best response)
        epsilon: Epsilon for epsilon-greedy exploration in best response
        gamma: Discount factor for Q-learning (standard DQN uses 0.99)
        q_lr: Learning rate for Q-network
        policy_lr: Learning rate for policy network
        batch_size: Batch size for training
        q_buffer_size: Size of Q-learning replay buffer
        policy_buffer_size: Size of average policy reservoir buffer
        target_update_freq: How often to update target Q-network
        updates_per_iter: Number of gradient updates per iteration
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        seed: Random seed
        save_dir: Directory to save models
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        log_interval: How often to log metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Constants for history tokens
    TOKEN_DIM = 175  # 92 obs + 82 action one-hot + 1 validity
    MAX_SEQ_LEN = 6

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algorithm": "NFSP-History",
                "num_envs": num_envs,
                "num_iterations": num_iterations,
                "episodes_per_iter": episodes_per_iter,
                "eta": eta,
                "epsilon": epsilon,
                "gamma": gamma,
                "q_lr": q_lr,
                "policy_lr": policy_lr,
                "batch_size": batch_size,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "seed": seed,
            },
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping logging")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create environment
    env = BargainEnv(num_envs=num_envs, self_play=True, device=0, seed=seed)

    # Create history-based networks for both players
    q_nets = [QNetworkHistory(TOKEN_DIM, NUM_ACTIONS, d_model, nhead, num_layers).to(device) for _ in range(2)]
    q_nets_target = [deepcopy(q_net) for q_net in q_nets]
    policy_nets = [AveragePolicyNetworkHistory(TOKEN_DIM, NUM_ACTIONS, d_model, nhead, num_layers).to(device) for _ in range(2)]

    # Optimizers
    q_optimizers = [torch.optim.Adam(q_net.parameters(), lr=q_lr) for q_net in q_nets]
    policy_optimizers = [torch.optim.Adam(policy_net.parameters(), lr=policy_lr) for policy_net in policy_nets]

    # Replay buffers for history
    buffers = NFSPBuffersHistory(
        num_players=2,
        q_buffer_size=q_buffer_size,
        policy_buffer_size=policy_buffer_size,
    )

    reward_history = [[], []]
    q_loss_history = []
    policy_loss_history = []

    print("=" * 60)
    print("NFSP TRAINING (History Transformer)")
    print("=" * 60)
    print(f"Environments: {num_envs}")
    print(f"Iterations: {num_iterations}")
    print(f"Episodes per iteration: {episodes_per_iter}")
    print(f"Eta (best response prob): {eta}")
    print(f"Epsilon (exploration): {epsilon}")
    print(f"Gamma (discount factor): {gamma}")
    print(f"Transformer: d_model={d_model}, nhead={nhead}, layers={num_layers}")
    print()

    total_updates = 0

    for iteration in range(num_iterations):
        # =====================
        # 1. Generate episodes
        # =====================
        total_episodes = 0
        episode_rewards = [[], []]

        while total_episodes < episodes_per_iter:
            obs, info = env.reset()
            action_mask = info['action_mask']

            # Decide strategy for each env and player at episode start
            use_br = [[torch.rand(1).item() < eta for _ in range(2)] for _ in range(num_envs)]

            # Track history for each player in each env
            # history[env_idx][player] = list of (obs, action) tuples
            history_data = [{} for _ in range(num_envs)]
            last_history = [{} for _ in range(num_envs)]
            last_seq_len = [{} for _ in range(num_envs)]
            last_action = [{} for _ in range(num_envs)]
            last_mask = [{} for _ in range(num_envs)]

            active = torch.ones(num_envs, dtype=torch.bool, device=device)

            for step in range(12):  # Max steps
                if not active.any():
                    break

                current_player = env.get_current_player()
                actions = torch.zeros(num_envs, dtype=torch.long, device=device)

                for p in range(2):
                    player_mask = (current_player == p) & active
                    if not player_mask.any():
                        continue

                    indices = player_mask.nonzero().squeeze(-1)
                    p_obs = obs[indices]
                    p_am = action_mask[indices]

                    # Select actions based on strategy
                    with torch.no_grad():
                        for i, idx in enumerate(indices):
                            idx_item = idx.item()
                            single_obs = p_obs[i]
                            single_mask = p_am[i]

                            # Build history tensor for this player
                            if p not in history_data[idx_item]:
                                history_data[idx_item][p] = []

                            hist_list = history_data[idx_item][p]
                            seq_len = len(hist_list) + 1

                            # Create current history tensor
                            history = torch.zeros(MAX_SEQ_LEN, TOKEN_DIM, device=device)
                            for t, (h_obs, h_action) in enumerate(hist_list):
                                history[t, :OBS_DIM] = h_obs
                                history[t, OBS_DIM + h_action] = 1.0  # One-hot action
                                history[t, -1] = 1.0  # Validity flag

                            # Current observation (action not yet taken)
                            history[seq_len - 1, :OBS_DIM] = single_obs
                            history[seq_len - 1, -1] = 1.0

                            seq_len_t = torch.tensor(seq_len, device=device)

                            if use_br[idx_item][p]:
                                # Best response: epsilon-greedy over Q-values
                                if torch.rand(1).item() < epsilon:
                                    # Random action from legal actions
                                    legal_indices = single_mask.nonzero().squeeze(-1)
                                    action = legal_indices[torch.randint(len(legal_indices), (1,))].item()
                                else:
                                    # Greedy action (argmax of Q-values)
                                    q_vals = q_nets[p](
                                        history.unsqueeze(0),
                                        seq_len_t.unsqueeze(0),
                                        single_mask.unsqueeze(0)
                                    )
                                    action = q_vals.argmax(dim=-1).item()

                                # Store in policy buffer (action from BR play)
                                buffers.add_policy_sample(p, HistoryPolicyTuple(
                                    history=history.cpu().clone(),
                                    seq_len=seq_len,
                                    action_mask=single_mask.cpu(),
                                    action=action,
                                ))
                            else:
                                # Average policy
                                logits = policy_nets[p](
                                    history.unsqueeze(0),
                                    seq_len_t.unsqueeze(0),
                                    single_mask.unsqueeze(0)
                                )
                                probs = F.softmax(logits, dim=-1)
                                action = torch.multinomial(probs, 1).item()

                            actions[idx] = action

                            # Store for transition
                            if p in last_history[idx_item]:
                                # Add transition from last state
                                buffers.add_transition(p, HistoryTransitionTuple(
                                    history=last_history[idx_item][p],
                                    seq_len=last_seq_len[idx_item][p],
                                    action_mask=last_mask[idx_item][p],
                                    action=last_action[idx_item][p],
                                    reward=0.0,  # Intermediate reward
                                    next_history=history.cpu().clone(),
                                    next_seq_len=seq_len,
                                    next_action_mask=single_mask.cpu(),
                                    done=False,
                                ))

                            last_history[idx_item][p] = history.cpu().clone()
                            last_seq_len[idx_item][p] = seq_len
                            last_action[idx_item][p] = action
                            last_mask[idx_item][p] = single_mask.cpu()

                            # Update history data with the action taken
                            history_data[idx_item][p].append((single_obs.cpu(), action))

                obs, rewards, dones, _, info = env.step(actions)
                action_mask = info['action_mask']

                # Handle terminal states
                if dones.any():
                    for idx in dones.nonzero().squeeze(-1):
                        idx_item = idx.item()

                        for p in range(2):
                            if p in last_history[idx_item]:
                                # Final transition with terminal reward
                                final_obs = obs[idx]
                                final_mask = action_mask[idx]
                                reward = rewards[idx, p].item()

                                # Create final history tensor
                                final_history = torch.zeros(MAX_SEQ_LEN, TOKEN_DIM, device=device)
                                for t, (h_obs, h_action) in enumerate(history_data[idx_item].get(p, [])):
                                    final_history[t, :OBS_DIM] = h_obs.to(device)
                                    final_history[t, OBS_DIM + h_action] = 1.0
                                    final_history[t, -1] = 1.0
                                final_seq_len = len(history_data[idx_item].get(p, []))
                                if final_seq_len == 0:
                                    final_seq_len = 1
                                    final_history[0, :OBS_DIM] = final_obs
                                    final_history[0, -1] = 1.0

                                buffers.add_transition(p, HistoryTransitionTuple(
                                    history=last_history[idx_item][p],
                                    seq_len=last_seq_len[idx_item][p],
                                    action_mask=last_mask[idx_item][p],
                                    action=last_action[idx_item][p],
                                    reward=reward,
                                    next_history=final_history.cpu(),
                                    next_seq_len=final_seq_len,
                                    next_action_mask=final_mask.cpu(),
                                    done=True,
                                ))

                                episode_rewards[p].append(reward)

                        # Reset tracking for this env
                        history_data[idx_item] = {}
                        last_history[idx_item] = {}
                        last_seq_len[idx_item] = {}
                        last_action[idx_item] = {}
                        last_mask[idx_item] = {}
                        use_br[idx_item] = [torch.rand(1).item() < eta for _ in range(2)]

                        total_episodes += 1

                    active = active & ~dones
                    env.auto_reset()

        # =====================
        # 2. Train networks
        # =====================
        q_losses = []
        policy_losses = []

        for _ in range(updates_per_iter):
            for p in range(2):
                # Train Q-network
                if buffers.q_buffer_size(p) >= batch_size:
                    batch = buffers.sample_transitions(p, batch_size)

                    # Stack history tensors
                    history_batch = torch.stack([t.history for t in batch]).to(device)
                    seq_len_batch = torch.tensor([t.seq_len for t in batch], device=device)
                    mask_batch = torch.stack([t.action_mask for t in batch]).to(device)
                    action_batch = torch.tensor([t.action for t in batch], device=device)
                    reward_batch = torch.tensor([t.reward for t in batch], device=device, dtype=torch.float32)
                    next_history_batch = torch.stack([t.next_history for t in batch]).to(device)
                    next_seq_len_batch = torch.tensor([t.next_seq_len for t in batch], device=device)
                    next_mask_batch = torch.stack([t.next_action_mask for t in batch]).to(device)
                    done_batch = torch.tensor([t.done for t in batch], device=device, dtype=torch.float32)

                    # Current Q-values
                    q_values = q_nets[p](history_batch, seq_len_batch, mask_batch)
                    q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                    # Standard DQN target: r + gamma * max_a' Q_target(s', a') * (1 - done)
                    with torch.no_grad():
                        next_q_values = q_nets_target[p](next_history_batch, next_seq_len_batch, next_mask_batch)
                        next_q_values = next_q_values.max(dim=1)[0]
                        target = reward_batch + gamma * next_q_values * (1 - done_batch)

                    q_loss = F.mse_loss(q_values, target)

                    q_optimizers[p].zero_grad()
                    q_loss.backward()
                    nn.utils.clip_grad_norm_(q_nets[p].parameters(), 10.0)
                    q_optimizers[p].step()

                    q_losses.append(q_loss.item())

                # Train policy network
                if buffers.policy_buffer_size(p) >= batch_size:
                    batch = buffers.sample_policy(p, batch_size)

                    history_batch = torch.stack([t.history for t in batch]).to(device)
                    seq_len_batch = torch.tensor([t.seq_len for t in batch], device=device)
                    mask_batch = torch.stack([t.action_mask for t in batch]).to(device)
                    action_batch = torch.tensor([t.action for t in batch], device=device)

                    # Supervised learning: cross-entropy loss
                    logits = policy_nets[p](history_batch, seq_len_batch, mask_batch)
                    policy_loss = F.cross_entropy(logits, action_batch)

                    policy_optimizers[p].zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(policy_nets[p].parameters(), 10.0)
                    policy_optimizers[p].step()

                    policy_losses.append(policy_loss.item())

        total_updates += 1

        # Update target networks
        if total_updates % target_update_freq == 0:
            for p in range(2):
                q_nets_target[p].load_state_dict(q_nets[p].state_dict())

        # Record statistics
        for p in range(2):
            if episode_rewards[p]:
                reward_history[p].append(np.mean(episode_rewards[p]))

        if q_losses:
            q_loss_history.append(np.mean(q_losses))
        if policy_losses:
            policy_loss_history.append(np.mean(policy_losses))

        # Logging
        r1 = reward_history[0][-1] if reward_history[0] else 0
        r2 = reward_history[1][-1] if reward_history[1] else 0

        if use_wandb and WANDB_AVAILABLE and (iteration + 1) % log_interval == 0:
            log_dict = {
                "iteration": iteration + 1,
                "p1/reward": r1,
                "p2/reward": r2,
                "q_loss": q_loss_history[-1] if q_loss_history else 0,
                "policy_loss": policy_loss_history[-1] if policy_losses else 0,
                "q_buffer_size": buffers.q_buffer_size(0),
                "policy_buffer_size": buffers.policy_buffer_size(0),
                "episodes": total_episodes,
            }
            wandb.log(log_dict)

        if (iteration + 1) % 25 == 0 or iteration == 0:
            print(f"Iter {iteration+1:3d} | P1: {r1:.4f} | P2: {r2:.4f} | "
                  f"Q-buf: {buffers.q_buffer_size(0)} | P-buf: {buffers.policy_buffer_size(0)}")

        # Periodic exploitability measurement
        if exploitability_interval > 0 and (iteration + 1) % exploitability_interval == 0:
            print(f"Measuring exploitability at iteration {iteration+1}...")
            exploit_metrics = measure_exploitability(
                _NFSPHistoryPolicyWrapper(policy_nets[0]), _NFSPHistoryPolicyWrapper(policy_nets[1]),
                "history_transformer", seed=seed
            )
            print(f"  Total exploitability: {exploit_metrics['exploitability/total']:.4f}")
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({**exploit_metrics, "iteration": iteration + 1})

        # Periodic checkpoint saving
        if save_interval > 0 and (iteration + 1) % save_interval == 0:
            checkpoint_path = save_path / f"checkpoint_iter_{iteration+1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            for p in range(2):
                torch.save(q_nets[p].state_dict(), checkpoint_path / f"nfsp_history_q_net_p{p+1}.pt")
                torch.save(policy_nets[p].state_dict(), checkpoint_path / f"nfsp_history_policy_net_p{p+1}.pt")
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save models
    for p in range(2):
        torch.save(q_nets[p].state_dict(), save_path / f"nfsp_history_q_net_p{p+1}.pt")
        torch.save(policy_nets[p].state_dict(), save_path / f"nfsp_history_policy_net_p{p+1}.pt")
    print(f"\nModels saved to {save_path}")

    # Plot training curves
    _plot_training(reward_history, q_loss_history, policy_loss_history, save_path / "nfsp_history_training_curves.png")

    # Final exploitability measurement
    print("Measuring final exploitability...")
    exploit_metrics = measure_exploitability(
        _NFSPHistoryPolicyWrapper(policy_nets[0]), _NFSPHistoryPolicyWrapper(policy_nets[1]),
        "history_transformer", seed=seed
    )
    print(f"Final exploitability: {exploit_metrics['exploitability/total']:.4f}")

    if use_wandb and WANDB_AVAILABLE:
        for p in range(2):
            wandb.save(str(save_path / f"nfsp_history_q_net_p{p+1}.pt"))
            wandb.save(str(save_path / f"nfsp_history_policy_net_p{p+1}.pt"))
        wandb.log({"training_curves": wandb.Image(str(save_path / "nfsp_history_training_curves.png"))})
        wandb.log({f"final/{k.split('/')[-1]}": v for k, v in exploit_metrics.items()})
        wandb.finish()

    return q_nets, policy_nets, reward_history


def _plot_training(reward_history, q_loss_history, policy_loss_history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Rewards
    axes[0].plot(reward_history[0], 'b-', label='Player 1', alpha=0.7)
    axes[0].plot(reward_history[1], 'r-', label='Player 2', alpha=0.7)
    axes[0].axhline(y=0.5, color='green', linestyle='--', label='Walk baseline', alpha=0.5)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-loss
    if q_loss_history:
        axes[1].plot(q_loss_history, 'g-', alpha=0.7)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Q-Network Loss')
        axes[1].grid(True, alpha=0.3)

    # Policy loss
    if policy_loss_history:
        axes[2].plot(policy_loss_history, 'm-', alpha=0.7)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Policy Network Loss')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NFSP Training for Bargaining Game")
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=500, help="Episodes per iteration")
    parser.add_argument("--eta", type=float, default=0.1, help="Anticipatory parameter")
    parser.add_argument("--epsilon", type=float, default=0.06, help="Epsilon for epsilon-greedy exploration")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for Q-learning")
    parser.add_argument("--q-lr", type=float, default=1e-4, help="Q-network learning rate")
    parser.add_argument("--policy-lr", type=float, default=1e-3, help="Policy network learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save models")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="bargaining-nfsp", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")
    # History Transformer options
    parser.add_argument("--history", action="store_true", help="Use History Transformer architecture")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--exploitability-interval", type=int, default=0, help="Measure exploitability every N iterations (0 = only at end)")
    parser.add_argument("--save-interval", type=int, default=0, help="Save checkpoint every N iterations (0 = disabled)")

    args = parser.parse_args()

    if args.history:
        train_nfsp_history(
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            episodes_per_iter=args.episodes_per_iter,
            eta=args.eta,
            epsilon=args.epsilon,
            gamma=args.gamma,
            q_lr=args.q_lr,
            policy_lr=args.policy_lr,
            batch_size=args.batch_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            exploitability_interval=args.exploitability_interval,
        )
    else:
        train_nfsp(
            num_envs=args.num_envs,
            num_iterations=args.iterations,
            episodes_per_iter=args.episodes_per_iter,
            eta=args.eta,
            epsilon=args.epsilon,
            gamma=args.gamma,
            q_lr=args.q_lr,
            policy_lr=args.policy_lr,
            batch_size=args.batch_size,
            seed=args.seed,
            save_dir=args.save_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            exploitability_interval=args.exploitability_interval,
        )
