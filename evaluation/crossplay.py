"""
Cross-play evaluation engine.

Runs games between strategy pairs and collects trajectories + outcomes.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import sys


from evaluation.policy_loader import Strategy
from evaluation.policy_loader import discover_strategies
from simulator.python.bargain_env_cpu import BargainEnvCPU, ITEM_QUANTITIES, ACTION_ACCEPT, ACTION_WALK


# Constants for HistoryTransformerPolicy
OBS_DIM = 92
NUM_ACTIONS = 82
MAX_SEQ_LEN = 6
TOKEN_DIM = 175

# For rational end computation
_ITEM_QTY = np.array([7.0, 4.0, 1.0])


def apply_rational_end_cpu(action: int, obs: np.ndarray) -> int:
    """
    If action is walk or accept, pick whichever gives higher value.

    Matches solver/train_mmd.py:apply_rational_end logic.
    This ensures the agent never accepts an offer worse than its BATNA
    and never walks away from an offer better than its BATNA.
    """
    if action != ACTION_ACCEPT and action != ACTION_WALK:
        return action

    offer_valid = obs[7]
    if offer_valid < 0.5:
        return ACTION_WALK  # No offer to accept

    walk_value = obs[3]
    offer_counts = np.maximum(obs[4:7], 0) * _ITEM_QTY
    my_values = obs[0:3]
    max_value = max(np.sum(_ITEM_QTY * my_values), 0.01)
    offer_value = np.sum(offer_counts * my_values) / max_value

    return ACTION_ACCEPT if offer_value > walk_value else ACTION_WALK

WALK_NUM_SAMPLES = 300_000  # Paper uses 3×10^5 BATNA samples for walk profiles
WALK_SEED = 0  # Fixed seed so ALL walk matchups use identical BATNA samples


def generate_walk_matchup(
    strategy_p1_name: str,
    strategy_p2_name: str,
    seed: int = None,  # Ignored - always uses WALK_SEED for consistency
    verbose: bool = True,
) -> "MatchupResult":
    """
    Generate synthetic game data for matchups involving 'walk'.

    When either strategy is walk, the outcome is deterministic:
    both players receive their BATNA. We sample game settings
    and record the BATNAs as payoffs, without running actual games.

    This follows the paper's methodology (Section 5):
    "We further short-circuited computation involving the walk strategy
    by noticing that the result is BATNA for both players regardless of
    the other agent strategy."

    Uses 300,000 samples as per the paper (3×10^5 single-round instances).

    IMPORTANT: Uses a fixed seed (WALK_SEED) so ALL walk-involved matchups
    use identical BATNA samples. This ensures fair comparison since walk
    outcomes should be the same regardless of opponent.

    Args:
        strategy_p1_name: Name of P1 strategy
        strategy_p2_name: Name of P2 strategy
        seed: Ignored - always uses WALK_SEED for consistency
        verbose: Print progress

    Returns:
        MatchupResult with game records where payoffs = BATNAs
    """
    num_games = WALK_NUM_SAMPLES

    if verbose:
        print(f"  Sampling {num_games:,} game settings for walk (fixed seed={WALK_SEED})...")

    # Use FIXED seed so all walk matchups have identical BATNA samples
    env = BargainEnvCPU(num_envs=num_games, self_play=True, seed=WALK_SEED)
    env.reset()

    # Vectorized: extract all BATNAs at once
    batnas_p1 = env._outside_options[:, 0] / env._max_possible_values[:, 0]
    batnas_p2 = env._outside_options[:, 1] / env._max_possible_values[:, 1]

    # Create all game records (list comprehension is faster than loop with append)
    games = [
        GameRecord(
            game_id=i,
            actions=[],
            outcome=GameOutcome(
                payoff_p1=float(batnas_p1[i]),
                payoff_p2=float(batnas_p2[i]),
                batna_p1=float(batnas_p1[i]),
                batna_p2=float(batnas_p2[i]),
                result="walk",
                utilities_p1=env._player_values[i, 0].tolist(),
                utilities_p2=env._player_values[i, 1].tolist(),
                allocation_p1=None,
                allocation_p2=None,
            )
        )
        for i in range(num_games)
    ]

    if verbose:
        print(f"  Generated {num_games:,} walk outcomes (no games played)")

    return MatchupResult(
        strategy_p1=strategy_p1_name,
        strategy_p2=strategy_p2_name,
        num_games=num_games,
        games=games
    )


@dataclass
class GameOutcome:
    """Outcome of a single game."""
    payoff_p1: float
    payoff_p2: float
    batna_p1: float
    batna_p2: float
    result: str  # "accept" or "walk"
    utilities_p1: List[float] = None
    utilities_p2: List[float] = None
    allocation_p1: List[int] = None
    allocation_p2: List[int] = None



@dataclass
class GameRecord:
    """Record of a single game with trajectory and outcome."""
    game_id: int
    actions: List[int]  # Sequence of actions taken
    outcome: GameOutcome
    reasoning_traces: List[Dict[str, Any]] = None  # LLM traces per move (None for non-LLM strategies)


@dataclass
class MatchupResult:
    """Result of running multiple games between two strategies."""
    strategy_p1: str
    strategy_p2: str
    num_games: int
    games: List[GameRecord]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy_p1": self.strategy_p1,
            "strategy_p2": self.strategy_p2,
            "num_games": self.num_games,
            "games": [
                {
                    "game_id": g.game_id,
                    "actions": g.actions,
                    "outcome": asdict(g.outcome),
                    **({"reasoning_traces": g.reasoning_traces} if g.reasoning_traces else {})
                }
                for g in self.games
            ]
        }

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for this matchup."""
        payoffs_p1 = [g.outcome.payoff_p1 for g in self.games]
        payoffs_p2 = [g.outcome.payoff_p2 for g in self.games]
        results = [g.outcome.result for g in self.games]

        accept_count = sum(1 for r in results if r == "accept")

        return {
            "strategy_p1": self.strategy_p1,
            "strategy_p2": self.strategy_p2,
            "num_games": self.num_games,
            "avg_payoff_p1": float(np.mean(payoffs_p1)),
            "avg_payoff_p2": float(np.mean(payoffs_p2)),
            "std_payoff_p1": float(np.std(payoffs_p1)),
            "std_payoff_p2": float(np.std(payoffs_p2)),
            "accept_rate": accept_count / self.num_games,
            "walk_rate": 1.0 - (accept_count / self.num_games),
        }


def get_policy_action(policy, obs: np.ndarray, action_mask: np.ndarray) -> int:
    """
    Get action from a policy given observation and action mask.

    Handles different policy architectures (PolicyNetwork, MAPPOPolicy, AveragePolicyNetwork).
    Always respects the action mask - invalid actions are masked out before sampling.
    """
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(action_mask).float().unsqueeze(0)

    with torch.no_grad():
        # All our policy types have similar forward interfaces
        # PolicyNetwork: forward(obs, action_mask) -> (logits, value)
        # MAPPOPolicy: get_action_logits(obs, action_mask) -> logits
        # AveragePolicyNetwork: forward(obs, action_mask) -> logits

        if hasattr(policy, 'get_action_logits'):
            # MAPPOPolicy
            logits = policy.get_action_logits(obs_tensor, mask_tensor)
        else:
            # PolicyNetwork or AveragePolicyNetwork
            output = policy(obs_tensor, mask_tensor)
            if isinstance(output, tuple):
                logits = output[0]  # (logits, value)
            else:
                logits = output

        # Apply action mask: set invalid actions to -inf before softmax
        # This ensures we never sample invalid actions (e.g., counteroffer on final turn)
        logits = logits.masked_fill(mask_tensor == 0, float('-inf'))

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        action = torch.distributions.Categorical(probs).sample()

    return action.item()


def get_history_policy_action(
    policy, history: torch.Tensor, seq_lens: torch.Tensor,
    action_mask: np.ndarray,
) -> int:
    """
    Get action from a HistoryTransformerPolicy given game history.

    Args:
        policy: HistoryTransformerPolicy instance
        history: History tensor of shape (1, max_seq_len, 175)
        seq_lens: Sequence length tensor of shape (1,)
        action_mask: Action mask array of shape (num_actions,)
    """
    mask_tensor = torch.from_numpy(action_mask).float().unsqueeze(0)

    with torch.no_grad():
        logits, _ = policy(history, seq_lens, mask_tensor)
        logits = logits.masked_fill(mask_tensor == 0, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        action = torch.distributions.Categorical(probs).sample()

    return action.item()


def run_single_game(
    env: BargainEnvCPU,
    strategy_p1: Strategy,
    strategy_p2: Strategy,
    env_idx: int = 0,
) -> Tuple[List[int], GameOutcome]:
    """
    Run a single game between two strategies.

    Args:
        env: The bargaining environment (already reset)
        strategy_p1: Strategy playing as P1
        strategy_p2: Strategy playing as P2
        env_idx: Which environment index to use (for batched envs)

    Returns:
        Tuple of (action_sequence, outcome)
    """
    actions_taken = []
    traces = []

    # Get initial observation
    obs = env._observations[env_idx]
    action_mask = env._action_masks[env_idx]

    # Store BATNAs at game start (normalized)
    batna_p1 = env._outside_options[env_idx, 0] / env._max_possible_values[env_idx, 0]
    batna_p2 = env._outside_options[env_idx, 1] / env._max_possible_values[env_idx, 1]

    # Initialize history tracking for HistoryTransformerPolicy strategies
    needs_hist = strategy_p1.needs_history or strategy_p2.needs_history
    if needs_hist:
        history = torch.zeros(1, MAX_SEQ_LEN, TOKEN_DIM)
        history[0, 0, :OBS_DIM] = torch.from_numpy(obs).float()
        history[0, 0, TOKEN_DIM - 1] = 1.0  # validity flag
        turn_count = 1

    max_steps = 12  # Safety limit
    for _ in range(max_steps):
        if env._done[env_idx]:
            break

        current_player = env._current_player[env_idx]

        # Get action from appropriate policy
        if current_player == 0:
            policy = strategy_p1.get_policy(0)
            use_history = strategy_p1.needs_history
        else:
            policy = strategy_p2.get_policy(1)
            use_history = strategy_p2.needs_history

        if use_history:
            seq_lens = torch.tensor([turn_count])
            action = get_history_policy_action(policy, history, seq_lens, action_mask)
        else:
            action = get_policy_action(policy, obs, action_mask)

        # Rational end override for MMD: pick max(offer_value, walk_value)
        if (current_player == 0 and strategy_p1.algorithm == "mmd") or \
           (current_player == 1 and strategy_p2.algorithm == "mmd"):
            action = apply_rational_end_cpu(action, obs)

        actions_taken.append(action)

        # Update history with action taken (matches training code in solver/train_mmd.py)
        if needs_hist:
            t = min(turn_count, MAX_SEQ_LEN - 1)
            if t > 0 and t < MAX_SEQ_LEN:
                prev_t = t - 1
                action_onehot = F.one_hot(torch.tensor(action, dtype=torch.long), NUM_ACTIONS).float()
                history[0, prev_t, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
            turn_count = min(turn_count + 1, MAX_SEQ_LEN - 1)

        # Collect LLM reasoning trace if available
        if hasattr(policy, '_last_trace') and policy._last_trace is not None:
            traces.append(policy._last_trace)
            policy._last_trace = None

        # Create action array for env.step (only this env matters)
        actions_array = np.zeros(env.num_envs, dtype=np.int32)
        actions_array[env_idx] = action

        # Step environment
        obs_all, rewards, dones, _, info = env.step(actions_array)
        obs = obs_all[env_idx]
        action_mask = info['action_mask'][env_idx]

        # Update history with new observation
        if needs_hist and not env._done[env_idx]:
            t = min(turn_count, MAX_SEQ_LEN - 1)
            if t < MAX_SEQ_LEN:
                history[0, t, :OBS_DIM] = torch.from_numpy(obs).float()
                history[0, t, TOKEN_DIM - 1] = 1.0

    # Determine outcome

    outcome_code = env._outcome[env_idx]
    if outcome_code == 1:  # OUTCOME_ACCEPT
        result = "accept"
        offer = env._current_offer[env_idx]
        allocation_p1 = (ITEM_QUANTITIES - offer).tolist()
        allocation_p2 = offer.tolist()

    else:  # OUTCOME_WALK
        result = "walk"
        allocation_p1 = None
        allocation_p2 = None


    payoff_p1 = float(env._rewards[env_idx, 0])
    payoff_p2 = float(env._rewards[env_idx, 1])

    utilities_p1 = env._player_values[env_idx, 0].tolist()
    utilities_p2 = env._player_values[env_idx, 1].tolist()


    outcome = GameOutcome(
        payoff_p1=payoff_p1,
        payoff_p2=payoff_p2,
        batna_p1=float(batna_p1),
        batna_p2=float(batna_p2),
        utilities_p1=utilities_p1,
        utilities_p2=utilities_p2,
        allocation_p1=allocation_p1,
        allocation_p2=allocation_p2,
        result=result
    )

    return actions_taken, outcome, traces if traces else None


def run_crossplay(
    strategy_p1: Strategy,
    strategy_p2: Strategy,
    num_games: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> MatchupResult:
    """
    Run cross-play evaluation between two strategies.

    Args:
        strategy_p1: Strategy to play as P1
        strategy_p2: Strategy to play as P2
        num_games: Number of games to play
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        MatchupResult with all game records
    """
    # Short-circuit for walk: no actual games needed (uses 300k samples)
    # Also short-circuit tough vs tough since it always ends in walk (BATNA for both)
    if (strategy_p1.name == "walk" or strategy_p2.name == "walk" or
        (strategy_p1.name == "tough" and strategy_p2.name == "tough")):
        return generate_walk_matchup(
            strategy_p1.name,
            strategy_p2.name,
            seed=seed,
            verbose=verbose,
        )

    # Use single environment for simplicity (sequential games)
    env = BargainEnvCPU(num_envs=1, self_play=True, seed=seed)

    games = []

    skipped = 0
    for game_id in range(num_games):
        # Reset environment for new game
        env.reset()

        # Reset mixture sampling for PSRO strategies (sample new policy for this game)
        strategy_p1.reset_mixture_sampling()
        strategy_p2.reset_mixture_sampling()

        # Run single game — skip on API errors (e.g. OpenAI content filter)
        try:
            actions, outcome, traces = run_single_game(env, strategy_p1, strategy_p2, env_idx=0)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"    Game {game_id} skipped: {e}")
            elif skipped == 4:
                print(f"    (suppressing further skip messages)")
            continue

        games.append(GameRecord(
            game_id=game_id,
            actions=actions,
            outcome=outcome,
            reasoning_traces=traces,
        ))

        if verbose and (game_id + 1) % 20 == 0:
            print(f"  Completed {game_id + 1}/{num_games} games ({skipped} skipped)")

    if skipped > 0:
        print(f"  Total skipped: {skipped}/{num_games}")

    return MatchupResult(
        strategy_p1=strategy_p1.name,
        strategy_p2=strategy_p2.name,
        num_games=len(games),
        games=games
    )


def run_crossplay_batched(
    strategy_p1: Strategy,
    strategy_p2: Strategy,
    num_games: int = 100,
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = True,
) -> MatchupResult:
    """
    Run cross-play evaluation with batched environments for efficiency.

    Args:
        strategy_p1: Strategy to play as P1
        strategy_p2: Strategy to play as P2
        num_games: Total number of games to play
        batch_size: Number of parallel environments
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        MatchupResult with all game records
    """
    # Short-circuit for walk: no actual games needed (uses 300k samples)
    # Also short-circuit tough vs tough since it always ends in walk (BATNA for both)
    if (strategy_p1.name == "walk" or strategy_p2.name == "walk" or
        (strategy_p1.name == "tough" and strategy_p2.name == "tough")):
        return generate_walk_matchup(
            strategy_p1.name,
            strategy_p2.name,
            seed=seed,
            verbose=verbose,
        )

    env = BargainEnvCPU(num_envs=batch_size, self_play=True, seed=seed)

    games = []
    games_completed = 0

    # Track per-environment state
    env_actions = [[] for _ in range(batch_size)]
    env_traces = [[] for _ in range(batch_size)]
    env_batnas = [(0.0, 0.0) for _ in range(batch_size)]

    # For PSRO mixtures: track which policy each env is using
    # Sample once per game, not per timestep
    p1_is_mixture = strategy_p1.is_mixture
    p2_is_mixture = strategy_p2.is_mixture
    p1_policy_indices = None
    p2_policy_indices = None

    if p1_is_mixture:
        p1_policy_indices = np.random.choice(
            len(strategy_p1.p1_policy.policies),
            size=batch_size,
            p=strategy_p1.p1_policy.sigma
        )
    if p2_is_mixture:
        p2_policy_indices = np.random.choice(
            len(strategy_p2.p2_policy.policies),
            size=batch_size,
            p=strategy_p2.p2_policy.sigma
        )

    # History tracking for HistoryTransformerPolicy strategies
    needs_hist = strategy_p1.needs_history or strategy_p2.needs_history

    env.reset()

    for i in range(batch_size):
        batna_p1 = env._outside_options[i, 0] / env._max_possible_values[i, 0]
        batna_p2 = env._outside_options[i, 1] / env._max_possible_values[i, 1]
        env_batnas[i] = (float(batna_p1), float(batna_p2))

    # Initialize history buffers (matching solver/train_mmd.py)
    if needs_hist:
        histories = torch.zeros(batch_size, MAX_SEQ_LEN, TOKEN_DIM)
        turn_counts = np.ones(batch_size, dtype=np.int32)
        for i in range(batch_size):
            histories[i, 0, :OBS_DIM] = torch.from_numpy(env._observations[i]).float()
            histories[i, 0, TOKEN_DIM - 1] = 1.0  # validity flag

    while games_completed < num_games:
        actions = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            if env._done[i]:
                continue

            current_player = env._current_player[i]
            obs = env._observations[i]
            mask = env._action_masks[i]

            if current_player == 0:
                if p1_is_mixture:
                    policy = strategy_p1.p1_policy.policies[p1_policy_indices[i]]
                    use_history = False
                else:
                    policy = strategy_p1.get_policy(0)
                    use_history = strategy_p1.needs_history
            else:
                if p2_is_mixture:
                    policy = strategy_p2.p2_policy.policies[p2_policy_indices[i]]
                    use_history = False
                else:
                    policy = strategy_p2.get_policy(1)
                    use_history = strategy_p2.needs_history

            if use_history:
                seq_lens = torch.tensor([turn_counts[i]])
                hist = histories[i:i+1]  # (1, max_seq_len, token_dim)
                action = get_history_policy_action(policy, hist, seq_lens, mask)
            else:
                action = get_policy_action(policy, obs, mask)

            # Rational end override for MMD: pick max(offer_value, walk_value)
            if (current_player == 0 and strategy_p1.algorithm == "mmd") or \
               (current_player == 1 and strategy_p2.algorithm == "mmd"):
                action = apply_rational_end_cpu(action, obs)

            actions[i] = action
            env_actions[i].append(action)

            # Collect LLM reasoning trace if available
            if hasattr(policy, '_last_trace') and policy._last_trace is not None:
                env_traces[i].append(policy._last_trace)
                policy._last_trace = None

        # Update history with actions taken (before env.step, matching training)
        if needs_hist:
            for i in range(batch_size):
                if env._done[i]:
                    continue
                t = min(turn_counts[i], MAX_SEQ_LEN - 1)
                if t > 0 and t < MAX_SEQ_LEN:
                    prev_t = t - 1
                    action_onehot = F.one_hot(torch.tensor(int(actions[i]), dtype=torch.long), NUM_ACTIONS).float()
                    histories[i, prev_t, OBS_DIM:OBS_DIM + NUM_ACTIONS] = action_onehot
                turn_counts[i] = min(turn_counts[i] + 1, MAX_SEQ_LEN - 1)

        _, rewards, dones, _, _ = env.step(actions)

        # Update history with new observations (after env.step, matching training)
        if needs_hist:
            for i in range(batch_size):
                if not dones[i]:
                    t = min(turn_counts[i], MAX_SEQ_LEN - 1)
                    if t < MAX_SEQ_LEN:
                        histories[i, t, :OBS_DIM] = torch.from_numpy(
                            env._observations[i]).float()
                        histories[i, t, TOKEN_DIM - 1] = 1.0

        for i in range(batch_size):
            if dones[i] and games_completed < num_games:
                outcome_code = env._outcome[i]

                if outcome_code == 1:
                    result = "accept"
                    offer = env._current_offer[i]
                    allocation_p1 = (ITEM_QUANTITIES - offer).tolist()
                    allocation_p2 = offer.tolist()
                else:
                    result = "walk"
                    allocation_p1 = None
                    allocation_p2 = None

                utilities_p1 = env._player_values[i, 0].tolist()
                utilities_p2 = env._player_values[i, 1].tolist()
                outcome = GameOutcome(
                    payoff_p1=float(rewards[i, 0]),
                    payoff_p2=float(rewards[i, 1]),
                    batna_p1=env_batnas[i][0],
                    batna_p2=env_batnas[i][1],
                    result=result,
                    utilities_p1=utilities_p1,
                    utilities_p2=utilities_p2,
                    allocation_p1=allocation_p1,
                    allocation_p2=allocation_p2,
                )

                games.append(GameRecord(
                    game_id=games_completed,
                    actions=env_actions[i].copy(),
                    outcome=outcome,
                    reasoning_traces=env_traces[i].copy() if env_traces[i] else None,
                ))

                games_completed += 1

                if verbose and games_completed % 20 == 0:
                    print(f"  Completed {games_completed}/{num_games} games")

        # Auto-reset done environments and update tracking
        if dones.any():
            env.auto_reset()

            for i in range(batch_size):
                if dones[i]:
                    env_actions[i] = []
                    env_traces[i] = []
                    batna_p1 = env._outside_options[i, 0] / env._max_possible_values[i, 0]
                    batna_p2 = env._outside_options[i, 1] / env._max_possible_values[i, 1]
                    env_batnas[i] = (float(batna_p1), float(batna_p2))

                    # Reset history for this env
                    if needs_hist:
                        histories[i] = 0
                        turn_counts[i] = 1
                        histories[i, 0, :OBS_DIM] = torch.from_numpy(
                            env._observations[i]).float()
                        histories[i, 0, TOKEN_DIM - 1] = 1.0

                    # Resample policies for new game if using mixtures
                    if p1_is_mixture:
                        p1_policy_indices[i] = np.random.choice(
                            len(strategy_p1.p1_policy.policies),
                            p=strategy_p1.p1_policy.sigma
                        )
                    if p2_is_mixture:
                        p2_policy_indices[i] = np.random.choice(
                            len(strategy_p2.p2_policy.policies),
                            p=strategy_p2.p2_policy.sigma
                        )

    return MatchupResult(
        strategy_p1=strategy_p1.name,
        strategy_p2=strategy_p2.name,
        num_games=num_games,
        games=games
    )


if __name__ == "__main__":

    strategies_dir = Path(__file__).parent.parent / "strategies"
    strategies = discover_strategies(strategies_dir)

    if len(strategies) >= 2:
        strat_names = list(strategies.keys())
        s1, s2 = strategies[strat_names[0]], strategies[strat_names[1]]

        print(f"\nRunning cross-play: {s1.name} (P1) vs {s2.name} (P2)")
        print("-" * 50)

        result = run_crossplay(s1, s2, num_games=10, verbose=True)
        summary = result.compute_summary()

        print("-" * 50)
        print(f"Results:")
        print(f"  Avg payoff P1 ({s1.name}): {summary['avg_payoff_p1']:.4f}")
        print(f"  Avg payoff P2 ({s2.name}): {summary['avg_payoff_p2']:.4f}")
        print(f"  Accept rate: {summary['accept_rate']:.2%}")
