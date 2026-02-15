#include "bargain_kernels.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

// ============================================================================
// Constant memory for item quantities (fast broadcast read)
// ============================================================================
__constant__ int c_item_quantities[NUM_ITEM_TYPES] = {ITEM_QTY_0, ITEM_QTY_1, ITEM_QTY_2};

// ============================================================================
// Device helper functions
// ============================================================================

__device__ void decode_offer_device(int action_idx, int8_t* offer) {
    offer[2] = action_idx % 2;
    action_idx /= 2;
    offer[1] = action_idx % 5;
    offer[0] = action_idx / 5;
}

__device__ void build_action_mask(const GameState* state, float* mask) {
    // Clear mask
    for (int i = 0; i < NUM_TOTAL_ACTIONS; i++) {
        mask[i] = 0.0f;
    }

    bool is_round1_p1 = (state->current_round == 0 && state->current_player == 0);
    bool is_final_round_p2 = (state->current_round == (MAX_ROUNDS - 1) && state->current_player == 1);
    bool has_offer = state->offer_valid;

    // ACCEPT (action 80): Valid only if there's an offer to accept
    // Round 1 P1 has no offer to accept
    if (has_offer && !is_round1_p1) {
        mask[ACTION_ACCEPT] = 1.0f;
    }

    // WALK (action 81): Always valid
    mask[ACTION_WALK] = 1.0f;

    // Counteroffers (actions 0-79): Valid except in final round for P2
    if (!is_final_round_p2) {
        for (int i = 0; i < NUM_COUNTEROFFER_ACTIONS; i++) {
            mask[i] = 1.0f;
        }
    }
}

__device__ void build_observation(
    const GameState* state,
    uint32_t game_idx,
    float* observations,
    float* action_masks,
    uint32_t num_games
) {
    uint8_t player = state->current_player;
    float* obs = &observations[game_idx * OBS_TOTAL_DIM];
    float* mask = &action_masks[game_idx * NUM_TOTAL_ACTIONS];

    // Player's own normalized values (0-1 scale)
    for (int i = 0; i < NUM_ITEM_TYPES; i++) {
        obs[i] = state->player_values[player][i] / (float)MAX_VALUE;
    }

    // Normalized outside offer (relative to max possible)
    obs[3] = state->outside_offers[player] / state->max_possible_values[player];

    // Current offer on table
    // The offer represents items being offered TO player 2
    // So player 1 sees what they're giving, player 2 sees what they're receiving
    if (state->offer_valid) {
        for (int i = 0; i < NUM_ITEM_TYPES; i++) {
            if (player == 1) {
                // P2 sees items offered to them
                obs[4 + i] = (float)state->current_offer[i] / (float)c_item_quantities[i];
            } else {
                // P1 sees items they would keep
                obs[4 + i] = (float)(c_item_quantities[i] - state->current_offer[i]) / (float)c_item_quantities[i];
            }
        }
    } else {
        // No offer - use -1 as indicator
        for (int i = 0; i < NUM_ITEM_TYPES; i++) {
            obs[4 + i] = -1.0f;
        }
    }

    // Offer validity flag
    obs[7] = (float)state->offer_valid;

    // Normalized round (0, 0.5, 1.0 for rounds 0, 1, 2)
    obs[8] = (float)state->current_round / (float)(MAX_ROUNDS - 1);

    // Current player (0 or 1)
    obs[9] = (float)state->current_player;

    // Build action validity mask (indices 10-91)
    build_action_mask(state, mask);

    // Copy mask to observation
    for (int i = 0; i < NUM_TOTAL_ACTIONS; i++) {
        obs[10 + i] = mask[i];
    }
}

__device__ void calculate_accept_rewards(GameState* state) {
    // Current offer represents items given to P2
    // P1 keeps the remainder
    float p1_utility = 0.0f;
    float p2_utility = 0.0f;

    for (int i = 0; i < NUM_ITEM_TYPES; i++) {
        int p2_items = state->current_offer[i];
        int p1_items = c_item_quantities[i] - p2_items;

        p1_utility += (float)p1_items * state->player_values[0][i];
        p2_utility += (float)p2_items * state->player_values[1][i];
    }

    state->rewards[0] = p1_utility;
    state->rewards[1] = p2_utility;
}

// ============================================================================
// Kernel Implementations
// ============================================================================

__global__ void init_rng_kernel(
    RNGState* rng_states,
    uint64_t seed,
    uint32_t num_games
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_games) return;

    // Each game gets unique subsequence for independence
    curand_init(seed, idx, 0, &rng_states[idx]);
}

__global__ void reset_games_kernel(
    GameState* states,
    RNGState* rng_states,
    float* observations,
    float* action_masks,
    uint8_t* current_players,
    uint32_t num_games
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_games) return;

    GameState* state = &states[idx];
    RNGState* rng = &rng_states[idx];

    // Generate player values and outside offers for both players
    for (int p = 0; p < NUM_PLAYERS; p++) {
        float total_value = 0.0f;
        for (int i = 0; i < NUM_ITEM_TYPES; i++) {
            // Random value 1-100
            int raw_value = (curand(rng) % MAX_VALUE) + MIN_VALUE;
            state->player_values[p][i] = (float)raw_value;
            total_value += (float)raw_value * (float)c_item_quantities[i];
        }
        state->max_possible_values[p] = total_value;

        // Generate outside offer (1 to max_possible_value)
        int max_val = (int)total_value;
        int outside = (curand(rng) % max_val) + 1;
        state->outside_offers[p] = (float)outside;
    }

    // Initialize game state
    state->current_offer[0] = -1;
    state->current_offer[1] = -1;
    state->current_offer[2] = -1;
    state->offer_valid = 0;
    state->current_round = 0;
    state->current_player = 0;
    state->action_count = 0;
    state->done = 0;
    state->outcome = OUTCOME_ONGOING;
    state->accepting_player = -1;
    state->rewards[0] = 0.0f;
    state->rewards[1] = 0.0f;

    // Update current player output
    current_players[idx] = 0;

    // Build initial observation for player 0
    build_observation(state, idx, observations, action_masks, num_games);
}

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
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_games) return;

    GameState* state = &states[idx];

    // Skip if game already done
    if (state->done) {
        dones[idx] = 1;
        truncateds[idx] = 0;
        rewards[idx * 2 + 0] = state->rewards[0] / state->max_possible_values[0];
        rewards[idx * 2 + 1] = state->rewards[1] / state->max_possible_values[1];
        return;
    }

    int32_t action = actions[idx];

    // Validate action - build temp mask
    float temp_mask[NUM_TOTAL_ACTIONS];
    build_action_mask(state, temp_mask);

    // Invalid action treated as WALK
    if (action < 0 || action >= NUM_TOTAL_ACTIONS || temp_mask[action] < 0.5f) {
        action = ACTION_WALK;
    }

    // Process action
    if (action == ACTION_ACCEPT) {
        // Calculate rewards based on accepted offer
        calculate_accept_rewards(state);
        state->done = 1;
        state->outcome = OUTCOME_ACCEPT;
        state->accepting_player = state->current_player;
    }
    else if (action == ACTION_WALK) {
        // Each player gets their outside offer
        state->rewards[0] = state->outside_offers[0];
        state->rewards[1] = state->outside_offers[1];
        state->done = 1;
        state->outcome = OUTCOME_WALK;
    }
    else {
        // Counteroffer: decode action to offer
        decode_offer_device(action, state->current_offer);
        state->offer_valid = 1;

        // Advance turn
        state->action_count++;
        if (state->current_player == 1) {
            state->current_round++;
        }
        state->current_player = 1 - state->current_player;
    }

    // Write outputs
    rewards[idx * 2 + 0] = state->rewards[0] / state->max_possible_values[0];
    rewards[idx * 2 + 1] = state->rewards[1] / state->max_possible_values[1];
    dones[idx] = state->done;
    truncateds[idx] = 0;
    current_players[idx] = state->current_player;

    // Build next observation if game continues
    if (!state->done) {
        build_observation(state, idx, observations, action_masks, num_games);
    }
}

__global__ void random_opponent_kernel(
    GameState* states,
    RNGState* rng_states,
    const float* action_masks,
    int32_t* output_actions,
    uint8_t target_player,
    uint32_t num_games
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_games) return;

    GameState* state = &states[idx];

    // Only generate for target player's turn, non-done games
    if (state->done || state->current_player != target_player) {
        output_actions[idx] = -1;  // No action needed
        return;
    }

    RNGState* rng = &rng_states[idx];
    const float* mask = &action_masks[idx * NUM_TOTAL_ACTIONS];

    // Count valid actions
    int valid_count = 0;
    for (int i = 0; i < NUM_TOTAL_ACTIONS; i++) {
        if (mask[i] > 0.5f) valid_count++;
    }

    if (valid_count == 0) {
        output_actions[idx] = ACTION_WALK;
        return;
    }

    // Randomly select valid action
    int selection = curand(rng) % valid_count;
    int count = 0;
    for (int i = 0; i < NUM_TOTAL_ACTIONS; i++) {
        if (mask[i] > 0.5f) {
            if (count == selection) {
                output_actions[idx] = i;
                return;
            }
            count++;
        }
    }

    // Fallback (shouldn't reach here)
    output_actions[idx] = ACTION_WALK;
}

__global__ void auto_reset_kernel(
    GameState* states,
    RNGState* rng_states,
    float* observations,
    float* action_masks,
    uint8_t* current_players,
    uint8_t* dones,
    uint32_t num_games
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_games) return;

    // Only reset done games
    if (!dones[idx]) return;

    GameState* state = &states[idx];
    RNGState* rng = &rng_states[idx];

    // Regenerate game state (same logic as reset)
    for (int p = 0; p < NUM_PLAYERS; p++) {
        float total_value = 0.0f;
        for (int i = 0; i < NUM_ITEM_TYPES; i++) {
            int raw_value = (curand(rng) % MAX_VALUE) + MIN_VALUE;
            state->player_values[p][i] = (float)raw_value;
            total_value += (float)raw_value * (float)c_item_quantities[i];
        }
        state->max_possible_values[p] = total_value;
        int max_val = (int)total_value;
        int outside = (curand(rng) % max_val) + 1;
        state->outside_offers[p] = (float)outside;
    }

    state->current_offer[0] = -1;
    state->current_offer[1] = -1;
    state->current_offer[2] = -1;
    state->offer_valid = 0;
    state->current_round = 0;
    state->current_player = 0;
    state->action_count = 0;
    state->done = 0;
    state->outcome = OUTCOME_ONGOING;
    state->accepting_player = -1;
    state->rewards[0] = 0.0f;
    state->rewards[1] = 0.0f;

    // Clear done flag in the output array too
    dones[idx] = 0;

    current_players[idx] = 0;
    build_observation(state, idx, observations, action_masks, num_games);
}

// ============================================================================
// Host-side wrapper functions
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

extern "C" {

GameBatch* create_game_batch(uint32_t num_games, int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));

    GameBatch* batch = (GameBatch*)malloc(sizeof(GameBatch));
    batch->num_games = num_games;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&batch->d_states, num_games * sizeof(GameState)));
    CUDA_CHECK(cudaMalloc(&batch->d_rng_states, num_games * sizeof(RNGState)));
    CUDA_CHECK(cudaMalloc(&batch->d_observations, num_games * OBS_TOTAL_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch->d_action_masks, num_games * NUM_TOTAL_ACTIONS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch->d_rewards, num_games * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&batch->d_dones, num_games * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&batch->d_truncateds, num_games * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&batch->d_current_players, num_games * sizeof(uint8_t)));

    // Zero initialize
    CUDA_CHECK(cudaMemset(batch->d_dones, 0, num_games * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(batch->d_truncateds, 0, num_games * sizeof(uint8_t)));

    return batch;
}

void destroy_game_batch(GameBatch* batch) {
    if (!batch) return;

    cudaFree(batch->d_states);
    cudaFree(batch->d_rng_states);
    cudaFree(batch->d_observations);
    cudaFree(batch->d_action_masks);
    cudaFree(batch->d_rewards);
    cudaFree(batch->d_dones);
    cudaFree(batch->d_truncateds);
    cudaFree(batch->d_current_players);

    free(batch);
}

void init_rng(GameBatch* batch, uint64_t seed) {
    int blocks = BLOCKS_FOR_N(batch->num_games);
    init_rng_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        (RNGState*)batch->d_rng_states,
        seed,
        batch->num_games
    );
    CUDA_CHECK(cudaGetLastError());
}

void reset_games(GameBatch* batch) {
    int blocks = BLOCKS_FOR_N(batch->num_games);
    reset_games_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        batch->d_states,
        (RNGState*)batch->d_rng_states,
        batch->d_observations,
        batch->d_action_masks,
        batch->d_current_players,
        batch->num_games
    );
    CUDA_CHECK(cudaGetLastError());
}

void step_games(GameBatch* batch, const int32_t* d_actions) {
    int blocks = BLOCKS_FOR_N(batch->num_games);
    step_games_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        batch->d_states,
        d_actions,
        batch->d_observations,
        batch->d_action_masks,
        batch->d_rewards,
        batch->d_dones,
        batch->d_truncateds,
        batch->d_current_players,
        batch->num_games
    );
    CUDA_CHECK(cudaGetLastError());
}

void generate_random_actions(GameBatch* batch, int32_t* d_output_actions, uint8_t target_player) {
    int blocks = BLOCKS_FOR_N(batch->num_games);
    random_opponent_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        batch->d_states,
        (RNGState*)batch->d_rng_states,
        batch->d_action_masks,
        d_output_actions,
        target_player,
        batch->num_games
    );
    CUDA_CHECK(cudaGetLastError());
}

void auto_reset_done_games(GameBatch* batch) {
    int blocks = BLOCKS_FOR_N(batch->num_games);
    auto_reset_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        batch->d_states,
        (RNGState*)batch->d_rng_states,
        batch->d_observations,
        batch->d_action_masks,
        batch->d_current_players,
        batch->d_dones,
        batch->num_games
    );
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
