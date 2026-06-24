"""Smoke test for AspirationAgent.

Runs aspiration self-play and aspiration vs PPO for a handful of games and
prints per-game action sequences, final allocations, payoffs, and BATNAs.
Intended to be eyeballed, not asserted against.
"""

import random as pyrandom
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.crossplay import get_policy_action  # noqa: E402
from evaluation.policy_loader import (  # noqa: E402
    Strategy,
    load_aspiration_strategy,
    load_ppo_strategy,
)
from simulator.python.bargain_env_cpu import (  # noqa: E402
    ACTION_ACCEPT,
    ACTION_WALK,
    ITEM_QUANTITIES,
    BargainEnvCPU,
)
from strategies.aspiration.aspiration_agent import AspirationAgent  # noqa: E402

SEED = 1234
NUM_GAMES = 10


def action_str(action: int) -> str:
    if action == ACTION_ACCEPT:
        return "ACCEPT"
    if action == ACTION_WALK:
        return "WALK"
    n0 = action // 10
    n1 = (action % 10) // 2
    n2 = action % 2
    return f"offer[to_P2={n0},{n1},{n2}]"


def run_one_game(env: BargainEnvCPU, strat_p1: Strategy, strat_p2: Strategy, env_idx: int = 0):
    obs = env._observations[env_idx]
    mask = env._action_masks[env_idx]
    actions = []
    max_val_p1 = float(env._max_possible_values[env_idx, 0])
    max_val_p2 = float(env._max_possible_values[env_idx, 1])
    batna_p1 = float(env._outside_options[env_idx, 0])
    batna_p2 = float(env._outside_options[env_idx, 1])
    vals_p1 = env._player_values[env_idx, 0].copy()
    vals_p2 = env._player_values[env_idx, 1].copy()

    for _ in range(12):
        if env._done[env_idx]:
            break
        player = int(env._current_player[env_idx])
        policy = strat_p1.get_policy(0) if player == 0 else strat_p2.get_policy(1)
        action = get_policy_action(policy, obs, mask)
        actions.append((player, int(action)))
        step_actions = np.zeros(env.num_envs, dtype=np.int32)
        step_actions[env_idx] = action
        obs_all, _, _, _, info = env.step(step_actions)
        obs = obs_all[env_idx]
        mask = info["action_mask"][env_idx]

    outcome = int(env._outcome[env_idx])
    outcome_str = {0: "ongoing", 1: "accept", 2: "walk"}[outcome]
    payoff_p1_norm = float(env._rewards[env_idx, 0])
    payoff_p2_norm = float(env._rewards[env_idx, 1])
    payoff_p1_raw = payoff_p1_norm * max_val_p1
    payoff_p2_raw = payoff_p2_norm * max_val_p2

    return {
        "actions": actions,
        "outcome": outcome_str,
        "payoff_p1": payoff_p1_raw,
        "payoff_p2": payoff_p2_raw,
        "payoff_p1_norm": payoff_p1_norm,
        "payoff_p2_norm": payoff_p2_norm,
        "batna_p1": batna_p1,
        "batna_p2": batna_p2,
        "max_val_p1": max_val_p1,
        "max_val_p2": max_val_p2,
        "vals_p1": vals_p1,
        "vals_p2": vals_p2,
    }


def print_schedule_for_game(env: BargainEnvCPU, env_idx: int, label: str):
    values = env._player_values[env_idx, 0].astype(float)
    batna = float(env._outside_options[env_idx, 0])
    agent = AspirationAgent(max_rounds=3)
    thresholds = agent._threshold_schedule(values, batna, agent.default_discount)
    print(f"    {label} (P1 perspective):")
    print(f"      vals={values.tolist()}, BATNA={batna:.1f}, max={float(np.dot(values, ITEM_QUANTITIES)):.1f}")
    print(f"      thresholds={['%.2f' % t for t in thresholds]}")


def run_matchup(label: str, strat_p1: Strategy, strat_p2: Strategy, num_games: int, seed: int):
    # Show schedule for a sample game
    print(f"\n=== {label} ===")
    sample_env = BargainEnvCPU(num_envs=1, self_play=True, seed=seed)
    sample_env.reset(seed=seed)
    print_schedule_for_game(sample_env, 0, "sample schedule")

    outcomes = {"accept": 0, "walk": 0, "ongoing": 0}
    p1_norms = []
    p2_norms = []

    for g in range(num_games):
        # Fresh env per game to avoid cross-contamination between parallel envs.
        env = BargainEnvCPU(num_envs=1, self_play=True, seed=seed + g)
        env.reset(seed=seed + g)
        res = run_one_game(env, strat_p1, strat_p2, env_idx=0)
        outcomes[res["outcome"]] += 1
        p1_norms.append(res["payoff_p1_norm"])
        p2_norms.append(res["payoff_p2_norm"])
        if g < 3:
            print(f"\n  Game {g}: {res['outcome']}")
            print(f"    P1 vals={res['vals_p1'].tolist()} BATNA={res['batna_p1']:.1f} max={res['max_val_p1']:.1f}")
            print(f"    P2 vals={res['vals_p2'].tolist()} BATNA={res['batna_p2']:.1f} max={res['max_val_p2']:.1f}")
            for i, (player, act) in enumerate(res["actions"]):
                print(f"    step {i}: P{player + 1} -> {action_str(act)}")
            print(
                f"    payoffs: P1={res['payoff_p1']:.1f} (norm {res['payoff_p1_norm']:.3f}), "
                f"P2={res['payoff_p2']:.1f} (norm {res['payoff_p2_norm']:.3f})"
            )

    print(f"\n  Outcomes over {num_games} games: {outcomes}")
    print(f"  Mean P1 payoff (norm): {np.mean(p1_norms):.3f} +/- {np.std(p1_norms):.3f}")
    print(f"  Mean P2 payoff (norm): {np.mean(p2_norms):.3f} +/- {np.std(p2_norms):.3f}")


def main():
    pyrandom.seed(SEED)
    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)

    asp = load_aspiration_strategy("aspiration", ROOT / "strategies" / "aspiration")
    print(f"Loaded: {asp.name}")

    ppo = None
    try:
        ppo = load_ppo_strategy("ppo", ROOT / "strategies" / "ppo")
        print(f"Loaded: {ppo.name}")
    except Exception as exc:
        print(f"Could not load PPO: {exc}")

    run_matchup("Aspiration (self-play)", asp, asp, NUM_GAMES, seed=SEED)
    if ppo is not None:
        run_matchup("Aspiration P1 vs PPO P2", asp, ppo, NUM_GAMES, seed=SEED + 1)
        run_matchup("PPO P1 vs Aspiration P2", ppo, asp, NUM_GAMES, seed=SEED + 2)


if __name__ == "__main__":
    main()
