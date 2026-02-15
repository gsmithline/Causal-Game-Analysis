#!/usr/bin/env python3
"""
Benchmark script for CUDA Bargaining Game Environment.

Measures throughput in terms of:
- Steps per second (total env.step() calls * num_envs)
- Games completed per second
- GPU memory usage

Usage:
    python benchmark.py [--num-envs 32768] [--num-steps 1000] [--device 0]
"""

import argparse
import time
import torch
import gc


def get_gpu_info(device: int = 0) -> dict:
    """Get GPU information."""
    props = torch.cuda.get_device_properties(device)
    return {
        'name': props.name,
        'total_memory_gb': props.total_memory / 1e9,
        'compute_capability': f"{props.major}.{props.minor}",
        'sm_count': props.multi_processor_count,
    }


def get_memory_usage(device: int = 0) -> dict:
    """Get current GPU memory usage."""
    torch.cuda.synchronize(device)
    return {
        'allocated_mb': torch.cuda.memory_allocated(device) / 1e6,
        'reserved_mb': torch.cuda.memory_reserved(device) / 1e6,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1e6,
    }


def benchmark_throughput(
    num_envs: int,
    num_steps: int,
    warmup_steps: int = 100,
    device: int = 0,
    self_play: bool = True,
) -> dict:
    """
    Benchmark environment throughput.

    Args:
        num_envs: Number of parallel environments
        num_steps: Number of steps to benchmark
        warmup_steps: Warmup steps (not timed)
        device: CUDA device ID
        self_play: Whether to use self-play mode

    Returns:
        Dictionary with benchmark results
    """
    from cuda_bargain import BargainEnv

    # Clear memory and reset stats
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Create environment
    env = BargainEnv(num_envs=num_envs, self_play=self_play, device=device)
    obs, info = env.reset(seed=42)

    # Warmup phase
    print(f"Warming up ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        actions = env.sample_valid_actions()
        obs, rewards, dones, truncateds, info = env.step(actions)
        if dones.any():
            env.auto_reset()

    torch.cuda.synchronize(device)

    # Timed benchmark
    print(f"Benchmarking ({num_steps} steps)...")
    total_games_completed = 0
    total_step_calls = 0

    start_time = time.perf_counter()

    for step in range(num_steps):
        actions = env.sample_valid_actions()
        obs, rewards, dones, truncateds, info = env.step(actions)

        games_done = dones.sum().item()
        total_games_completed += games_done
        total_step_calls += 1

        if dones.any():
            env.auto_reset()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    # Calculate metrics
    total_env_steps = num_steps * num_envs
    steps_per_second = total_env_steps / elapsed
    games_per_second = total_games_completed / elapsed
    us_per_step = (elapsed * 1e6) / num_steps  # microseconds per step call

    # Memory usage
    memory = get_memory_usage(device)

    return {
        'num_envs': num_envs,
        'num_steps': num_steps,
        'elapsed_seconds': elapsed,
        'total_env_steps': total_env_steps,
        'steps_per_second': steps_per_second,
        'games_completed': total_games_completed,
        'games_per_second': games_per_second,
        'us_per_step_call': us_per_step,
        'memory_allocated_mb': memory['allocated_mb'],
        'memory_peak_mb': memory['max_allocated_mb'],
    }


def benchmark_scaling(
    env_sizes: list,
    num_steps: int = 500,
    device: int = 0,
) -> list:
    """
    Benchmark how throughput scales with batch size.

    Args:
        env_sizes: List of environment counts to test
        num_steps: Steps per benchmark
        device: CUDA device

    Returns:
        List of benchmark results
    """
    results = []

    for num_envs in env_sizes:
        print(f"\n{'='*60}")
        print(f"Testing {num_envs:,} environments...")
        print('='*60)

        try:
            result = benchmark_throughput(
                num_envs=num_envs,
                num_steps=num_steps,
                warmup_steps=50,
                device=device,
            )
            results.append(result)

            print(f"  Steps/second: {result['steps_per_second']:,.0f}")
            print(f"  Games/second: {result['games_per_second']:,.0f}")
            print(f"  Memory: {result['memory_allocated_mb']:.1f} MB")

        except RuntimeError as e:
            print(f"  FAILED: {e}")
            results.append({
                'num_envs': num_envs,
                'error': str(e),
            })

        # Clear between runs
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_results_table(results: list):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    print(f"\n{'Envs':>12} {'Steps/sec':>15} {'Games/sec':>15} {'us/step':>12} {'Memory MB':>12}")
    print("-"*70)

    for r in results:
        if 'error' in r:
            print(f"{r['num_envs']:>12,} {'ERROR':>15}")
        else:
            print(f"{r['num_envs']:>12,} "
                  f"{r['steps_per_second']:>15,.0f} "
                  f"{r['games_per_second']:>15,.0f} "
                  f"{r['us_per_step_call']:>12.1f} "
                  f"{r['memory_allocated_mb']:>12.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark CUDA Bargaining Game Environment'
    )
    parser.add_argument(
        '--num-envs', type=int, default=32768,
        help='Number of parallel environments (default: 32768)'
    )
    parser.add_argument(
        '--num-steps', type=int, default=1000,
        help='Number of steps to benchmark (default: 1000)'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='CUDA device ID (default: 0)'
    )
    parser.add_argument(
        '--scaling-test', action='store_true',
        help='Run scaling test across multiple batch sizes'
    )
    parser.add_argument(
        '--warmup', type=int, default=100,
        help='Warmup steps (default: 100)'
    )
    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    # Print GPU info
    gpu_info = get_gpu_info(args.device)
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)
    print(f"  Device: {gpu_info['name']}")
    print(f"  Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"  Compute: SM {gpu_info['compute_capability']}")
    print(f"  SMs: {gpu_info['sm_count']}")

    if args.scaling_test:
        # Test multiple batch sizes
        env_sizes = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
        results = benchmark_scaling(
            env_sizes=env_sizes,
            num_steps=args.num_steps,
            device=args.device,
        )
        print_results_table(results)

    else:
        # Single benchmark
        print(f"\nBenchmarking with {args.num_envs:,} environments, {args.num_steps:,} steps...")

        result = benchmark_throughput(
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            warmup_steps=args.warmup,
            device=args.device,
        )

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"  Environments: {result['num_envs']:,}")
        print(f"  Steps: {result['num_steps']:,}")
        print(f"  Elapsed: {result['elapsed_seconds']:.2f} seconds")
        print(f"")
        print(f"  Total env steps: {result['total_env_steps']:,}")
        print(f"  Steps/second: {result['steps_per_second']:,.0f}")
        print(f"  Games completed: {result['games_completed']:,}")
        print(f"  Games/second: {result['games_per_second']:,.0f}")
        print(f"  Microseconds/step call: {result['us_per_step_call']:.1f}")
        print(f"")
        print(f"  Memory allocated: {result['memory_allocated_mb']:.1f} MB")
        print(f"  Memory peak: {result['memory_peak_mb']:.1f} MB")


if __name__ == '__main__':
    main()
