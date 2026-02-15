"""
Utility functions for evaluation module.

Includes action decoding and game trajectory formatting.
"""

from typing import List, Tuple, Dict, Any, Optional
import json
from pathlib import Path


# Game constants (matching the environment)
ITEM_QUANTITIES = [7, 4, 1]  # Max items of each type
ACTION_ACCEPT = 80
ACTION_WALK = 81
NUM_ACTIONS = 82


def decode_action(action: int) -> Tuple[str, Optional[Tuple[int, int, int]]]:
    """
    Decode action index to human-readable format.

    Args:
        action: Action index (0-81)

    Returns:
        (action_type, offer):
            - action_type: "OFFER", "ACCEPT", or "WALK"
            - offer: Tuple of (item0, item1, item2) for offers, None otherwise
    """
    if action == ACTION_ACCEPT:
        return ("ACCEPT", None)
    elif action == ACTION_WALK:
        return ("WALK", None)
    elif 0 <= action < 80:
        # Decode: action = offer[0]*10 + offer[1]*2 + offer[2]
        item2 = action % 2
        action //= 2
        item1 = action % 5
        item0 = action // 5
        return ("OFFER", (item0, item1, item2))
    else:
        raise ValueError(f"Invalid action: {action}")


def encode_offer(offer: Tuple[int, int, int]) -> int:
    """Encode an offer tuple to action index."""
    n0, n1, n2 = offer
    return n0 * 10 + n1 * 2 + n2


def format_offer(offer: Tuple[int, int, int]) -> str:
    """Format an offer as a readable string."""
    return f"[{offer[0]}, {offer[1]}, {offer[2]}]"


def format_action(action: int, player: int) -> str:
    """
    Format a single action as a readable string.

    Args:
        action: Action index
        player: Player who took the action (0 or 1)

    Returns:
        Human-readable action string
    """
    action_type, offer = decode_action(action)
    player_str = f"P{player + 1}"

    if action_type == "OFFER":
        # The offer represents items given to the OTHER player
        return f"{player_str}: Offer {format_offer(offer)} to opponent"
    elif action_type == "ACCEPT":
        return f"{player_str}: ACCEPT"
    else:  # WALK
        return f"{player_str}: WALK (take outside option)"


def format_game_trajectory(game: Dict[str, Any], verbose: bool = True) -> str:
    """
    Format a complete game trajectory as readable text.

    Args:
        game: Game record dict with 'actions' and 'outcome'
        verbose: Include detailed outcome info

    Returns:
        Formatted string representation of the game
    """
    lines = []
    lines.append(f"Game {game.get('game_id', '?')}:")

    actions = game['actions']
    outcome = game['outcome']

    # Format each turn
    for turn, action in enumerate(actions):
        player = turn % 2  # P1 starts (player 0), then alternates
        action_str = format_action(action, player)
        lines.append(f"  Turn {turn + 1}: {action_str}")

    # Format outcome
    lines.append(f"  Result: {outcome['result'].upper()}")
    lines.append(f"  Payoffs: P1={outcome['payoff_p1']:.3f}, P2={outcome['payoff_p2']:.3f}")

    if verbose:
        lines.append(f"  BATNAs:  P1={outcome['batna_p1']:.3f}, P2={outcome['batna_p2']:.3f}")

        # Calculate surplus if accepted
        if outcome['result'] == 'accept':
            surplus_p1 = outcome['payoff_p1'] - outcome['batna_p1']
            surplus_p2 = outcome['payoff_p2'] - outcome['batna_p2']
            lines.append(f"  Surplus: P1={surplus_p1:+.3f}, P2={surplus_p2:+.3f}")

    return "\n".join(lines)


def print_game(game: Dict[str, Any], verbose: bool = True) -> None:
    """Print a formatted game trajectory."""
    print(format_game_trajectory(game, verbose))


def print_games(games: List[Dict[str, Any]], max_games: int = 10, verbose: bool = True) -> None:
    """Print multiple game trajectories."""
    for i, game in enumerate(games[:max_games]):
        if i > 0:
            print()
        print_game(game, verbose)

    if len(games) > max_games:
        print(f"\n... and {len(games) - max_games} more games")


def load_and_print_matchup(
    matchup_dir: Path,
    max_games: int = 10,
    verbose: bool = True
) -> None:
    """
    Load and print games from a matchup directory.

    Args:
        matchup_dir: Path to matchup directory (e.g., data/crossplay/ppo_p1_vs_nfsp_p2)
        max_games: Maximum number of games to print
        verbose: Include detailed outcome info
    """
    games_path = Path(matchup_dir) / "games.json"

    if not games_path.exists():
        print(f"Error: {games_path} not found")
        return

    with open(games_path) as f:
        data = json.load(f)

    print(f"Matchup: {data['strategy_p1']} (P1) vs {data['strategy_p2']} (P2)")
    print(f"Total games: {data['num_games']}")
    print("=" * 60)

    print_games(data['games'], max_games, verbose)


def summarize_actions(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize action statistics across games.

    Returns:
        Dict with action statistics
    """
    total_actions = 0
    action_counts = {i: 0 for i in range(NUM_ACTIONS)}
    game_lengths = []

    for game in games:
        actions = game['actions']
        game_lengths.append(len(actions))
        for action in actions:
            action_counts[action] += 1
            total_actions += 1

    # Most common actions
    sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
    top_actions = []
    for action, count in sorted_actions[:10]:
        if count > 0:
            action_type, offer = decode_action(action)
            if action_type == "OFFER":
                desc = f"Offer {format_offer(offer)}"
            else:
                desc = action_type
            top_actions.append({
                "action": action,
                "description": desc,
                "count": count,
                "frequency": count / total_actions if total_actions > 0 else 0
            })

    return {
        "total_actions": total_actions,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "min_game_length": min(game_lengths) if game_lengths else 0,
        "max_game_length": max(game_lengths) if game_lengths else 0,
        "top_actions": top_actions,
    }


if __name__ == "__main__":
    # Demo: load and print some games
    from pathlib import Path

    crossplay_dir = Path(__file__).parent.parent / "data" / "crossplay"

    # Find first available matchup
    matchup_dirs = [d for d in crossplay_dir.iterdir() if d.is_dir()]

    if matchup_dirs:
        print("Sample game trajectories:")
        print()
        load_and_print_matchup(matchup_dirs[0], max_games=5)
    else:
        print("No crossplay data found. Run evaluation first.")
