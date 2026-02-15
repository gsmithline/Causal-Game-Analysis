#!/usr/bin/env python3
"""
Benchmark comparing CUDA bargaining simulator to OpenSpiel bargaining game.

Measures throughput in games/second for both implementations.
"""

import time
import random
import numpy as np
import pyspiel
import torch
import sys
import os

# Add project root to path for solver imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'solver'))

from cuda_bargain import BargainEnv


def benchmark_openspiel(num_games: int = 10000) -> dict:
    """
    Benchmark OpenSpiel bargaining game (sequential CPU).

    Returns:
        dict with 'games', 'time', 'games_per_second'
    """
    game = pyspiel.load_game('bargaining')

    start = time.perf_counter()

    for _ in range(num_games):
        state = game.new_initial_state()

        # Process chance nodes (deal private info)
        while state.current_player() == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            action, _ = random.choice(outcomes)
            state.apply_action(action)

        # Play game with random actions
        while not state.is_terminal():
            legal = state.legal_actions()
            action = random.choice(legal)
            state.apply_action(action)

    elapsed = time.perf_counter() - start

    return {
        'games': num_games,
        'time': elapsed,
        'games_per_second': num_games / elapsed
    }


def benchmark_cuda_simulator(num_games: int = 100000, batch_size: int = 4096) -> dict:
    """
    Benchmark CUDA bargaining simulator (parallel GPU).

    Args:
        num_games: Total games to simulate
        batch_size: Number of parallel environments

    Returns:
        dict with 'games', 'time', 'games_per_second'
    """
    env = BargainEnv(num_envs=batch_size, self_play=True, device=0)

    # Warmup
    obs, info = env.reset()
    for _ in range(10):
        actions = env.sample_valid_actions()
        obs, rewards, dones, truncs, info = env.step(actions)
        env.auto_reset()
    torch.cuda.synchronize()

    games_completed = 0
    start = time.perf_counter()

    obs, info = env.reset()
    while games_completed < num_games:
        actions = env.sample_valid_actions()
        obs, rewards, dones, truncs, info = env.step(actions)
        games_completed += dones.sum().item()
        env.auto_reset()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        'games': games_completed,
        'time': elapsed,
        'games_per_second': games_completed / elapsed
    }


def benchmark_cuda_simulator_with_policy(num_games: int = 100000, batch_size: int = 4096) -> dict:
    """
    Benchmark CUDA simulator with neural network policy inference.

    This more closely matches actual training throughput.
    """
    from policy import PolicyNetwork

    env = BargainEnv(num_envs=batch_size, self_play=True, device=0)

    # Create a simple policy network
    policy = PolicyNetwork(env.OBS_DIM, env.NUM_ACTIONS).cuda()
    policy.eval()

    # Warmup
    obs, info = env.reset()
    for _ in range(10):
        with torch.no_grad():
            logits, _ = policy(obs)
            logits = logits.masked_fill(info['action_mask'] == 0, -1e9)
            actions = torch.argmax(logits, dim=-1).int()
        obs, rewards, dones, truncs, info = env.step(actions)
        env.auto_reset()
    torch.cuda.synchronize()

    games_completed = 0
    start = time.perf_counter()

    obs, info = env.reset()
    while games_completed < num_games:
        with torch.no_grad():
            logits, _ = policy(obs)
            logits = logits.masked_fill(info['action_mask'] == 0, -1e9)
            actions = torch.argmax(logits, dim=-1).int()
        obs, rewards, dones, truncs, info = env.step(actions)
        games_completed += dones.sum().item()
        env.auto_reset()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        'games': games_completed,
        'time': elapsed,
        'games_per_second': games_completed / elapsed
    }


def main():
    print("=" * 70)
    print("BARGAINING GAME THROUGHPUT BENCHMARK")
    print("=" * 70)

    # OpenSpiel benchmark
    print("\n[1/3] OpenSpiel Bargaining (CPU, sequential)...")
    openspiel_results = benchmark_openspiel(num_games=10000)
    print(f"  Games: {openspiel_results['games']:,}")
    print(f"  Time: {openspiel_results['time']:.2f}s")
    print(f"  Throughput: {openspiel_results['games_per_second']:,.0f} games/sec")

    # CUDA benchmark (random actions)
    print("\n[2/3] CUDA Simulator (GPU, random actions)...")
    cuda_random_results = benchmark_cuda_simulator(num_games=1_000_000, batch_size=4096)
    print(f"  Games: {cuda_random_results['games']:,}")
    print(f"  Time: {cuda_random_results['time']:.2f}s")
    print(f"  Throughput: {cuda_random_results['games_per_second']:,.0f} games/sec")

    # CUDA benchmark (with policy)
    print("\n[3/3] CUDA Simulator (GPU, with MLP policy)...")
    cuda_policy_results = benchmark_cuda_simulator_with_policy(num_games=1_000_000, batch_size=4096)
    print(f"  Games: {cuda_policy_results['games']:,}")
    print(f"  Time: {cuda_policy_results['time']:.2f}s")
    print(f"  Throughput: {cuda_policy_results['games_per_second']:,.0f} games/sec")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Implementation':<35} {'Throughput':>15} {'Speedup':>12}")
    print("-" * 62)
    print(f"{'OpenSpiel (CPU)':<35} {openspiel_results['games_per_second']:>12,.0f}/s {'1x':>12}")

    speedup_random = cuda_random_results['games_per_second'] / openspiel_results['games_per_second']
    print(f"{'CUDA Simulator (random)':<35} {cuda_random_results['games_per_second']:>12,.0f}/s {speedup_random:>11.0f}x")

    speedup_policy = cuda_policy_results['games_per_second'] / openspiel_results['games_per_second']
    print(f"{'CUDA Simulator (with policy)':<35} {cuda_policy_results['games_per_second']:>12,.0f}/s {speedup_policy:>11.0f}x")

    return {
        'openspiel': openspiel_results,
        'cuda_random': cuda_random_results,
        'cuda_policy': cuda_policy_results
    }


if __name__ == "__main__":
    main()
