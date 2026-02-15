# CUDA Bargaining Game Simulator

High-performance GPU-accelerated simulator for the bargaining game, designed for deep reinforcement learning research.

## Overview

This simulator implements a two-player bargaining game where players negotiate over the division of items. The entire game logic runs on GPU using CUDA kernels, enabling massive parallelism for RL training.

## Performance

| Implementation | Throughput | Speedup |
|----------------|------------|---------|
| OpenSpiel (CPU) | ~25,000 games/s | 1x |
| CUDA Simulator (random) | ~1,500,000 games/s | **60x** |
| CUDA Simulator (with MLP policy) | ~1,900,000 games/s | **75x** |

*Benchmarked on NVIDIA GPU with 4,096 parallel environments.*

The CUDA simulator achieves **60-75x speedup** over OpenSpiel's CPU implementation, enabling training runs that would take days to complete in hours.

## Game Rules

### Setup
- **3 item types** with quantities `[7, 4, 1]`
- Each player has **private valuations** (1-100) for each item type
- Each player has a **private outside option** (walk-away value)

### Gameplay
1. **Round 1**: P1 makes an offer (or walks)
2. **Round 2**: P2 can accept, counter-offer, or walk
3. **Round 3**: P1 can accept, counter-offer, or walk
4. ... continues for up to 3 rounds (6 turns)
5. **Final turn**: P2 can only accept or walk (no counter-offer)

### Outcomes
- **Accept**: Items split according to accepted offer
- **Walk**: Both players receive their outside option

## API

### Environment Class

```python
from cuda_bargain import BargainEnv

# Create environment with 4096 parallel games
env = BargainEnv(num_envs=4096, self_play=True, device=0)

# Reset all environments
obs, info = env.reset()
# obs: [4096, 92] float32 tensor
# info['action_mask']: [4096, 82] float32 tensor
# info['current_player']: [4096] uint8 tensor

# Step with actions
actions = torch.randint(0, 82, (4096,), dtype=torch.int32, device='cuda')
obs, rewards, dones, truncs, info = env.step(actions)
# rewards: [4096, 2] normalized rewards for both players

# Auto-reset finished games
env.auto_reset()
```

### Observation Space (92 dimensions)

| Index | Description |
|-------|-------------|
| 0-2 | Player's item values (normalized 0-1) |
| 3 | Outside option (normalized) |
| 4-6 | Current offer on table (-1 if none) |
| 7 | Offer validity flag (0 or 1) |
| 8 | Current round (0, 0.5, or 1.0) |
| 9 | Current player (0 or 1) |
| 10-91 | Action mask (82 booleans) |

### Action Space (82 actions)

| Action | Description |
|--------|-------------|
| 0-79 | Counter-offers: `action = item0*10 + item1*2 + item2` |
| 80 | Accept current offer |
| 81 | Walk away (take outside option) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Python API Layer                         │
│                    (bargain_env.py)                          │
├─────────────────────────────────────────────────────────────┤
│                   PyBind11 Bindings                          │
│                 (python_bindings.cpp)                        │
├─────────────────────────────────────────────────────────────┤
│                    CUDA Kernels                              │
│                  (bargain_game.cu)                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │
│  │ reset_games │ │ step_games  │ │ auto_reset_kernel   │    │
│  │   kernel    │ │   kernel    │ │                     │    │
│  └─────────────┘ └─────────────┘ └─────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                   GPU Memory                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ States   │ │ Obs      │ │ Actions  │ │ RNG States   │   │
│  │ [N]      │ │ [N,92]   │ │ [N,82]   │ │ [N]          │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **All state on GPU**: Game states, observations, and RNG states remain on GPU throughout training, minimizing CPU-GPU transfers.

2. **Philox RNG**: Uses cuRAND's Philox generator for high-quality, parallel random number generation.

3. **Batch operations**: All operations (reset, step, auto-reset) process the entire batch in parallel.

4. **Zero-copy tensors**: PyTorch tensors share memory with CUDA arrays for seamless integration.

## Building

### Prerequisites
- CUDA Toolkit 11.0+
- CMake 3.18+
- Python 3.10+
- PyTorch with CUDA support

### Build Steps

```bash
cd simulator
mkdir build && cd build
cmake ..
make -j

# Install Python package
pip install -e .
```

## Files

| File | Description |
|------|-------------|
| `include/bargain_game.h` | Game constants and data structures |
| `include/bargain_kernels.cuh` | CUDA kernel declarations |
| `src/bargain_game.cu` | CUDA kernel implementations |
| `src/python_bindings.cpp` | PyBind11 Python bindings |
| `python/bargain_env.py` | High-level Python API |
| `benchmark.py` | Performance comparison with OpenSpiel |

## Comparison with OpenSpiel

| Feature | CUDA Simulator | OpenSpiel |
|---------|---------------|-----------|
| Language | CUDA C++ | C++ |
| Execution | GPU parallel | CPU sequential |
| Max batch size | Limited by VRAM | 1 |
| Throughput | ~1.5M games/s | ~25K games/s |
| RL integration | Native PyTorch | NumPy arrays |
| Game variations | Fixed rules | Configurable |

### When to use which

**Use CUDA Simulator when:**
- Training RL agents with millions of games
- Need maximum throughput
- Using PyTorch-based training

**Use OpenSpiel when:**
- Need game rule variations
- Running on CPU-only systems
- Prototyping algorithms
- Need other bargaining game variants

## License

MIT License
