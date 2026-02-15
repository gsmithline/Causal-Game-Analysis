#!/usr/bin/env python3
"""
Comparison benchmark: OpenSpiel bargaining game vs CUDA implementation.

Measures throughput for both implementations and presents results.
"""

import time
import random
import pyspiel
import torch


def benchmark_openspiel(num_games: int, warmup_games: int = 100) -> dict:
    """
    Benchmark OpenSpiel's bargaining game.

    OpenSpiel runs sequentially on CPU, one game at a time.
    """
    game = pyspiel.load_game('bargaining')

    # Warmup
    for _ in range(warmup_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action = random.choices(
                    [o[0] for o in outcomes],
                    weights=[o[1] for o in outcomes]
                )[0]
            else:
                legal = state.legal_actions()
                action = random.choice(legal)
            state.apply_action(action)

    # Timed run
    total_steps = 0
    games_completed = 0

    start = time.perf_counter()

    for _ in range(num_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action = random.choices(
                    [o[0] for o in outcomes],
                    weights=[o[1] for o in outcomes]
                )[0]
            else:
                legal = state.legal_actions()
                action = random.choice(legal)
            state.apply_action(action)
            total_steps += 1
        games_completed += 1

    elapsed = time.perf_counter() - start

    return {
        'implementation': 'OpenSpiel',
        'num_games': num_games,
        'total_steps': total_steps,
        'elapsed': elapsed,
        'games_per_second': games_completed / elapsed,
        'steps_per_second': total_steps / elapsed,
    }


def benchmark_cuda(num_envs: int, num_steps: int, warmup_steps: int = 100) -> dict:
    """
    Benchmark CUDA bargaining game implementation.

    Runs in parallel on GPU.
    """
    from cuda_bargain import BargainEnv

    env = BargainEnv(num_envs=num_envs, device=0)
    obs, info = env.reset(seed=42)

    # Warmup
    for _ in range(warmup_steps):
        actions = env.sample_valid_actions()
        obs, rewards, dones, truncateds, info = env.step(actions)
        if dones.any():
            env.auto_reset()

    torch.cuda.synchronize()

    # Timed run
    games_completed = 0
    start = time.perf_counter()

    for _ in range(num_steps):
        actions = env.sample_valid_actions()
        obs, rewards, dones, truncateds, info = env.step(actions)
        games_completed += dones.sum().item()
        if dones.any():
            env.auto_reset()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_steps = num_steps * num_envs

    return {
        'implementation': f'CUDA ({num_envs:,} envs)',
        'num_games': int(games_completed),
        'total_steps': total_steps,
        'elapsed': elapsed,
        'games_per_second': games_completed / elapsed,
        'steps_per_second': total_steps / elapsed,
    }


def main():
    print("=" * 80)
    print("BARGAINING GAME BENCHMARK: OpenSpiel vs CUDA")
    print("=" * 80)

    # Get GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    results = []

    # OpenSpiel benchmark (CPU, sequential)
    print("\n" + "-" * 40)
    print("Benchmarking OpenSpiel (CPU, sequential)...")
    print("-" * 40)

    openspiel_games = 10000
    result = benchmark_openspiel(num_games=openspiel_games)
    results.append(result)
    print(f"  Games: {result['num_games']:,}")
    print(f"  Steps/sec: {result['steps_per_second']:,.0f}")
    print(f"  Games/sec: {result['games_per_second']:,.0f}")

    # CUDA benchmarks (GPU, parallel) at different batch sizes
    cuda_configs = [
        (1024, 1000),
        (8192, 500),
        (32768, 500),
        (65536, 500),
    ]

    print("\n" + "-" * 40)
    print("Benchmarking CUDA (GPU, parallel)...")
    print("-" * 40)

    for num_envs, num_steps in cuda_configs:
        print(f"\n  Testing {num_envs:,} parallel environments...")
        result = benchmark_cuda(num_envs=num_envs, num_steps=num_steps)
        results.append(result)
        print(f"    Steps/sec: {result['steps_per_second']:,.0f}")
        print(f"    Games/sec: {result['games_per_second']:,.0f}")

    # Print comparison table
    print("\n")
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Table header
    print(f"\n{'Implementation':<25} {'Steps/sec':>15} {'Games/sec':>15} {'Speedup':>12}")
    print("-" * 70)

    baseline_steps = results[0]['steps_per_second']
    baseline_games = results[0]['games_per_second']

    for r in results:
        speedup_steps = r['steps_per_second'] / baseline_steps
        print(f"{r['implementation']:<25} "
              f"{r['steps_per_second']:>15,.0f} "
              f"{r['games_per_second']:>15,.0f} "
              f"{speedup_steps:>11.0f}x")

    # Summary
    best_cuda = max(results[1:], key=lambda x: x['steps_per_second'])
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nOpenSpiel (CPU):     {results[0]['steps_per_second']:>12,.0f} steps/sec")
    print(f"Best CUDA (GPU):     {best_cuda['steps_per_second']:>12,.0f} steps/sec")
    print(f"Speedup:             {best_cuda['steps_per_second']/results[0]['steps_per_second']:>12,.0f}x")

    print("\nNote: OpenSpiel's bargaining game differs slightly (no walk-away value,")
    print("      10 rounds max vs 3). Comparison shows order-of-magnitude difference.")


if __name__ == '__main__':
    main()
