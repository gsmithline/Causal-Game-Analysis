#ifndef BARGAIN_KERNELS_CUH
#define BARGAIN_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "bargain_game.h"

// Use Philox RNG for best parallel performance
typedef curandStatePhilox4_32_10_t RNGState;

// Thread block configuration
#define THREADS_PER_BLOCK 256
#define BLOCKS_FOR_N(n) (((n) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

// ============================================================================
// Kernel Declarations
// ============================================================================

// Initialize RNG states for all games
__global__ void init_rng_kernel(
    RNGState* rng_states,
    uint64_t seed,
    uint32_t num_games
);

// Reset all games to initial state
__global__ void reset_games_kernel(
    GameState* states,
    RNGState* rng_states,
    float* observations,
    float* action_masks,
    uint8_t* current_players,
    uint32_t num_games
);

// Step all games with given actions
__global__ void step_games_kernel(
    GameState* states,
    const int32_t* actions,
    float* observations,
    float* action_masks,
    float* rewards,
    uint8_t* dones,
    uint8_t* truncateds,
    uint8_t* current_players,
    uint32_t num_games
);

// Generate random valid actions for specified player (vs-random mode)
__global__ void random_opponent_kernel(
    GameState* states,
    RNGState* rng_states,
    const float* action_masks,
    int32_t* output_actions,
    uint8_t target_player,
    uint32_t num_games
);

// Auto-reset done games
__global__ void auto_reset_kernel(
    GameState* states,
    RNGState* rng_states,
    float* observations,
    float* action_masks,
    uint8_t* current_players,
    uint8_t* dones,
    uint32_t num_games
);

// ============================================================================
// Host-side wrapper functions
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Allocate GPU memory for game batch
GameBatch* create_game_batch(uint32_t num_games, int device_id);

// Free GPU memory
void destroy_game_batch(GameBatch* batch);

// Initialize RNG (call once after creation)
void init_rng(GameBatch* batch, uint64_t seed);

// Reset all games
void reset_games(GameBatch* batch);

// Step all games with actions
void step_games(GameBatch* batch, const int32_t* d_actions);

// Generate random actions for target player
void generate_random_actions(GameBatch* batch, int32_t* d_output_actions, uint8_t target_player);

// Auto-reset completed games
void auto_reset_done_games(GameBatch* batch);

#ifdef __cplusplus
}
#endif

#endif // BARGAIN_KERNELS_CUH
