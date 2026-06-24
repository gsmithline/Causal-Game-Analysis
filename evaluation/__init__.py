"""
Evaluation module for cross-play between trained strategies.

Heavy imports (policy_loader, crossplay, simulator) are deferred
to avoid pulling in torch/CUDA when only analysis code is needed.
"""

def __getattr__(name):
    if name in ("Strategy", "load_strategy", "discover_strategies",
                "load_ppo_strategy", "load_mmd_strategy", "load_mappo_strategy",
                "load_nfsp_strategy", "load_psro_strategy"):
        from . import policy_loader
        return getattr(policy_loader, name)
    if name in ("GameOutcome", "GameRecord", "MatchupResult",
                "run_crossplay", "run_crossplay_batched"):
        from . import crossplay
        return getattr(crossplay, name)
    if name in ("save_matchup_result", "load_matchup_result",
                "load_matchup_summary", "build_metagame_matrix",
                "save_metagame_matrix"):
        from . import data_saver
        return getattr(data_saver, name)
    if name in ("decode_action", "encode_offer", "format_action",
                "format_game_trajectory", "print_game", "print_games",
                "load_and_print_matchup", "summarize_actions"):
        from . import utils
        return getattr(utils, name)
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
