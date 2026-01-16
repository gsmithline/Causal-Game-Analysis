#ifndef BARGAIN_GAME_H
#define BARGAIN_GAME_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Game Constants
// ============================================================================

#define NUM_ITEM_TYPES 3
#define MAX_ROUNDS 3
#define NUM_PLAYERS 2

// Item quantities: [7, 4, 1]
#define ITEM_QTY_0 7
#define ITEM_QTY_1 4
#define ITEM_QTY_2 1

// Action space: 80 counteroffers + ACCEPT + WALK = 82 total
// Counteroffers: (0-7) x (0-4) x (0-1) = 8 * 5 * 2 = 80
#define NUM_COUNTEROFFER_ACTIONS 80
#define ACTION_ACCEPT 80
#define ACTION_WALK 81
#define NUM_TOTAL_ACTIONS 82

// Value ranges
#define MIN_VALUE 1
#define MAX_VALUE 100

// Observation dimensions
#define OBS_PLAYER_VALUES 3      // Normalized values for each item type
#define OBS_OUTSIDE_OFFER 1      // Normalized outside offer
#define OBS_CURRENT_OFFER 3      // Current offer on table
#define OBS_OFFER_VALID 1        // 1 if offer exists, 0 otherwise
#define OBS_CURRENT_ROUND 1      // Normalized round (0-1)
#define OBS_CURRENT_PLAYER 1     // Which player's turn (0 or 1)
#define OBS_ACTION_MASK 82       // Boolean mask for valid actions
#define OBS_TOTAL_DIM 92         // Total observation dimensions

// Game outcomes
#define OUTCOME_ONGOING 0
#define OUTCOME_ACCEPT 1
#define OUTCOME_WALK 2

// ============================================================================
// Data Structures
// ============================================================================

// Game state for a single game instance
typedef struct {
    // Player private information
    float player_values[NUM_PLAYERS][NUM_ITEM_TYPES];  // Each player's values per item
    float outside_offers[NUM_PLAYERS];                  // Each player's outside offer
    float max_possible_values[NUM_PLAYERS];             // Max value for normalization

    // Current game state
    int8_t current_offer[NUM_ITEM_TYPES];    // Items offered to P2 (-1 if no offer)
    uint8_t offer_valid;                      // Whether an offer is on the table
    uint8_t current_round;                    // 0-indexed round (0, 1, 2)
    uint8_t current_player;                   // 0 or 1
    uint8_t action_count;                     // Total actions taken (0-5)

    // Termination state
    uint8_t done;                             // Game has ended
    uint8_t outcome;                          // OUTCOME_ONGOING/ACCEPT/WALK
    int8_t accepting_player;                  // Which player accepted (-1 if N/A)

    // Results (set when game ends)
    float rewards[NUM_PLAYERS];               // Final rewards for each player
} GameState;

// Batch of games for parallel GPU processing
typedef struct {
    uint32_t num_games;

    // Device pointers
    GameState* d_states;                      // [num_games] game states
    void* d_rng_states;                       // curandState array

    // Observation/output tensors (contiguous for GPU)
    float* d_observations;                    // [num_games, OBS_TOTAL_DIM]
    float* d_action_masks;                    // [num_games, NUM_TOTAL_ACTIONS]
    float* d_rewards;                         // [num_games, 2]
    uint8_t* d_dones;                         // [num_games]
    uint8_t* d_truncateds;                    // [num_games]
    uint8_t* d_current_players;               // [num_games]
} GameBatch;

// ============================================================================
// Action Encoding/Decoding
// ============================================================================

// Decode action index (0-79) to offer quantities
// offer[0]: 0-7 (for item with qty 7)
// offer[1]: 0-4 (for item with qty 4)
// offer[2]: 0-1 (for item with qty 1)
static inline void decode_action_to_offer(int action_idx, int8_t* offer) {
    offer[2] = action_idx % 2;
    action_idx /= 2;
    offer[1] = action_idx % 5;
    offer[0] = action_idx / 5;
}

// Encode offer to action index
static inline int encode_offer_to_action(const int8_t* offer) {
    return offer[0] * 10 + offer[1] * 2 + offer[2];
}

#ifdef __cplusplus
}
#endif

#endif // BARGAIN_GAME_H
