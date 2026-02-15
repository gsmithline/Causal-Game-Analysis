"""
Evaluation module for cross-play between trained strategies.
"""

from .policy_loader import (
    Strategy,
    load_strategy,
    discover_strategies,
    load_ppo_strategy,
    load_mmd_strategy,
    load_mappo_strategy,
    load_nfsp_strategy,
    load_psro_strategy,
)

from .crossplay import (
    GameOutcome,
    GameRecord,
    MatchupResult,
    run_crossplay,
    run_crossplay_batched,
)

from .data_saver import (
    save_matchup_result,
    load_matchup_result,
    load_matchup_summary,
    build_metagame_matrix,
    save_metagame_matrix,
)

from .utils import (
    decode_action,
    encode_offer,
    format_action,
    format_game_trajectory,
    print_game,
    print_games,
    load_and_print_matchup,
    summarize_actions,
)
