"""
Policy loader for loading trained strategies from different algorithms.

Each strategy is a pair of (p1_policy, p2_policy) that can be used
in cross-play evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass, field
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.policy import PolicyNetwork, HistoryTransformerPolicy
from training.mappo.policy import MAPPOPolicy
from training.nfsp.policy import AveragePolicyNetwork

from strategies.walk.policy import WalkPolicy
from strategies.soft.policy import SoftPolicy
from strategies.tough.policy import ToughPolicy
from strategies.llm_strategies.openai import OpenAIPolicy

class MixturePolicy(nn.Module):
    """
    A policy that samples from a mixture of policies according to sigma.

    At the start of each episode, sample which policy to use according to sigma.
    For batch inference, we sample a policy for each environment.
    """
    def __init__(self, policies: List[nn.Module], sigma: np.ndarray):
        super().__init__()
        self.policies = nn.ModuleList(policies)
        self.sigma = sigma
        self._current_policy_indices = None

    def sample_policies(self, batch_size: int):
        """Sample which policy to use for each environment in the batch."""
        self._current_policy_indices = np.random.choice(
            len(self.policies), size=batch_size, p=self.sigma
        )

    def forward(self, obs, action_mask):
        """
        Forward pass that routes each observation to its sampled policy.

        For simplicity in crossplay evaluation, we compute all policies
        and then select based on the sampled indices.
        """
        batch_size = obs.shape[0]
        device = obs.device

        # If no policies sampled yet, sample now
        if self._current_policy_indices is None or len(self._current_policy_indices) != batch_size:
            self.sample_policies(batch_size)

        # Compute outputs for all policies and gather
        all_logits = []
        all_values = []

        for policy in self.policies:
            logits, value = policy(obs, action_mask)
            all_logits.append(logits)
            all_values.append(value)

        # Stack: (num_policies, batch_size, num_actions)
        all_logits = torch.stack(all_logits, dim=0)
        all_values = torch.stack(all_values, dim=0)

        # Select based on sampled indices
        batch_idx = torch.arange(batch_size, device=device)
        policy_idx = torch.tensor(self._current_policy_indices, device=device)

        logits = all_logits[policy_idx, batch_idx]
        values = all_values[policy_idx, batch_idx]

        return logits, values

    def reset_sampling(self):
        """Reset the sampled policy indices (call at start of new episodes)."""
        self._current_policy_indices = None


@dataclass
class Strategy:
    """A strategy is a pair of policies for P1 and P2."""
    name: str
    p1_policy: nn.Module
    p2_policy: nn.Module
    algorithm: str  # ppo, mappo, nfsp, psro
    is_mixture: bool = False  # True if this is a PSRO mixture
    needs_history: bool = False  # True if policy uses HistoryTransformerPolicy

    def get_policy(self, player_id: int) -> nn.Module:
        """Get the policy for a specific player (0 or 1)."""
        return self.p1_policy if player_id == 0 else self.p2_policy

    def reset_mixture_sampling(self):
        """Reset mixture sampling for new episodes (only relevant for PSRO mixtures)."""
        if self.is_mixture:
            if hasattr(self.p1_policy, 'reset_sampling'):
                self.p1_policy.reset_sampling()
            if hasattr(self.p2_policy, 'reset_sampling'):
                self.p2_policy.reset_sampling()


def load_ppo_strategy(name: str, path: Path) -> Strategy:
    """
    Load a PPO strategy from checkpoint files.

    Expects:
        path/policy_p1.pt
        path/policy_p2.pt
    """
    p1_path = path / "policy_p1.pt"
    p2_path = path / "policy_p2.pt"

    if not p1_path.exists() or not p2_path.exists():
        raise FileNotFoundError(f"PPO checkpoints not found in {path}")

    p1_policy = PolicyNetwork()
    p2_policy = PolicyNetwork()

    p1_policy.load_state_dict(torch.load(p1_path, map_location='cpu'))
    p2_policy.load_state_dict(torch.load(p2_path, map_location='cpu'))

    p1_policy.eval()
    p2_policy.eval()

    return Strategy(name=name, p1_policy=p1_policy, p2_policy=p2_policy, algorithm="ppo")


def load_mmd_strategy(name: str, path: Path) -> Strategy:
    """
    Load an MMD strategy from checkpoint files.

    Auto-detects architecture (PolicyNetwork vs HistoryTransformerPolicy)
    from checkpoint state dict keys. Searches for *_p1.pt / *_p2.pt files.
    """
    # Find checkpoint files: try specific names, then glob
    p1_path = path / "mmd_rational_end_best_p1.pt"
    p2_path = path / "mmd_rational_end_best_p2.pt"

    if not p1_path.exists() or not p2_path.exists():
        p1_candidates = sorted(path.glob("*_p1.pt"))
        p2_candidates = sorted(path.glob("*_p2.pt"))
        if not p1_candidates or not p2_candidates:
            raise FileNotFoundError(f"MMD checkpoints not found in {path}")
        p1_path = p1_candidates[0]
        p2_path = p2_candidates[0]

    p1_state = torch.load(p1_path, map_location='cpu', weights_only=True)

    # Detect architecture from state dict keys
    if 'token_embed.weight' in p1_state:
        # HistoryTransformerPolicy (transformer with game history)
        p1_policy = HistoryTransformerPolicy(token_dim=175, num_actions=82)
        p2_policy = HistoryTransformerPolicy(token_dim=175, num_actions=82)
        needs_history = True
        print(f"  Detected HistoryTransformerPolicy architecture")
    else:
        # Legacy PolicyNetwork (MLP)
        p1_policy = PolicyNetwork()
        p2_policy = PolicyNetwork()
        needs_history = False

    p1_policy.load_state_dict(p1_state)
    p2_state = torch.load(p2_path, map_location='cpu', weights_only=True)
    p2_policy.load_state_dict(p2_state)

    p1_policy.eval()
    p2_policy.eval()

    print(f"  Loaded: {p1_path.name}, {p2_path.name}")

    return Strategy(name=name, p1_policy=p1_policy, p2_policy=p2_policy,
                    algorithm="mmd", needs_history=needs_history)


def load_mappo_strategy(name: str, path: Path) -> Strategy:
    """
    Load a MAPPO strategy from checkpoint files.

    Expects:
        path/mappo_p1.pt
        path/mappo_p2.pt
    """
    p1_path = path / "mappo_p1.pt"
    p2_path = path / "mappo_p2.pt"

    if not p1_path.exists() or not p2_path.exists():
        raise FileNotFoundError(f"MAPPO checkpoints not found in {path}")

    p1_policy = MAPPOPolicy()
    p2_policy = MAPPOPolicy()

    p1_policy.load_state_dict(torch.load(p1_path, map_location='cpu'))
    p2_policy.load_state_dict(torch.load(p2_path, map_location='cpu'))

    p1_policy.eval()
    p2_policy.eval()

    return Strategy(name=name, p1_policy=p1_policy, p2_policy=p2_policy, algorithm="mappo")


def load_nfsp_strategy(name: str, path: Path) -> Strategy:
    """
    Load an NFSP strategy from checkpoint files.

    Uses the average policy network (not the Q-network) for evaluation.

    Expects:
        path/nfsp_policy_net_p1.pt
        path/nfsp_policy_net_p2.pt
    """
    p1_path = path / "nfsp_policy_net_p1.pt"
    p2_path = path / "nfsp_policy_net_p2.pt"

    if not p1_path.exists() or not p2_path.exists():
        raise FileNotFoundError(f"NFSP checkpoints not found in {path}")

    p1_policy = AveragePolicyNetwork()
    p2_policy = AveragePolicyNetwork()

    p1_policy.load_state_dict(torch.load(p1_path, map_location='cpu'))
    p2_policy.load_state_dict(torch.load(p2_path, map_location='cpu'))

    p1_policy.eval()
    p2_policy.eval()

    return Strategy(name=name, p1_policy=p1_policy, p2_policy=p2_policy, algorithm="nfsp")


def load_psro_strategy(name: str, path: Path, use_mixture: bool = True) -> Strategy:
    """
    Load a PSRO strategy from checkpoint file.

    PSRO produces a Nash equilibrium as a MIXTURE over a population of policies.
    By default, loads the full mixture (sampling policies according to sigma).
    Set use_mixture=False to load only the last policy (not recommended).

    Expects:
        path/psro_result.pt

    Args:
        name: Strategy name
        path: Path to PSRO checkpoint directory
        use_mixture: If True, create a mixture policy that samples according to sigma.
                     If False, just use the last policy (not the true PSRO solution).
    """
    result_path = path / "psro_result.pt"

    if not result_path.exists():
        raise FileNotFoundError(f"PSRO checkpoint not found: {result_path}")

    # PSRO checkpoints contain numpy arrays, so need weights_only=False
    checkpoint = torch.load(result_path, map_location='cpu', weights_only=False)

    p1_state_dicts = checkpoint["p1_policies"]
    p2_state_dicts = checkpoint["p2_policies"]
    sigma_p1 = checkpoint.get("sigma_p1", None)
    sigma_p2 = checkpoint.get("sigma_p2", None)
    architecture = checkpoint.get("architecture", "mlp")

    if architecture != "mlp":
        raise NotImplementedError(f"PSRO architecture '{architecture}' not yet supported")

    if use_mixture and sigma_p1 is not None and sigma_p2 is not None:
        n_sigma_p1 = len(sigma_p1)
        n_sigma_p2 = len(sigma_p2)

        if len(p1_state_dicts) > n_sigma_p1:
            print(f"  Note: Truncating P1 policies from {len(p1_state_dicts)} to {n_sigma_p1} (sigma length)")
            p1_state_dicts = p1_state_dicts[:n_sigma_p1]
        if len(p2_state_dicts) > n_sigma_p2:
            print(f"  Note: Truncating P2 policies from {len(p2_state_dicts)} to {n_sigma_p2} (sigma length)")
            p2_state_dicts = p2_state_dicts[:n_sigma_p2]

        p1_policies = []
        for state_dict in p1_state_dicts:
            policy = PolicyNetwork()
            policy.load_state_dict(state_dict)
            policy.eval()
            p1_policies.append(policy)

        p2_policies = []
        for state_dict in p2_state_dicts:
            policy = PolicyNetwork()
            policy.load_state_dict(state_dict)
            policy.eval()
            p2_policies.append(policy)

        # Create mixture policies
        p1_mixture = MixturePolicy(p1_policies, sigma_p1)
        p2_mixture = MixturePolicy(p2_policies, sigma_p2)

        p1_mixture.eval()
        p2_mixture.eval()

        print(f"  Loaded PSRO mixture with {len(p1_policies)} P1 policies and {len(p2_policies)} P2 policies")

        return Strategy(name=name, p1_policy=p1_mixture, p2_policy=p2_mixture,
                       algorithm="psro", is_mixture=True)
    else:
        # Fallback: just use the last policy
        if not use_mixture:
            print(f"  Warning: Loading only last PSRO policy (not the mixture)")
        else:
            print(f"  Warning: No sigma found in checkpoint, using last policy")

        p1_policy = PolicyNetwork()
        p2_policy = PolicyNetwork()

        p1_policy.load_state_dict(p1_state_dicts[-1])
        p2_policy.load_state_dict(p2_state_dicts[-1])

        p1_policy.eval()
        p2_policy.eval()

        return Strategy(name=name, p1_policy=p1_policy, p2_policy=p2_policy,
                       algorithm="psro", is_mixture=False)


# Heuristic strategy loaders
def load_walk_strategy(name: str, path: Path) -> Strategy:
    """
    Load the walk heuristic strategy.

    Walk always walks away immediately. No checkpoint files needed.
    """
    p1_policy = WalkPolicy()
    p2_policy = WalkPolicy()

    p1_policy.eval()
    p2_policy.eval()

    return Strategy(name=name, p1_policy=p1_policy, p2_policy=p2_policy, algorithm="walk")


def load_soft_strategy(name: str, path: Path) -> Strategy:
    """
    Load the soft heuristic strategy.

    Soft accepts any offer. When must propose, makes random offer.
    No checkpoint files needed.
    """
    p1_policy = SoftPolicy()
    p2_policy = SoftPolicy()

    p1_policy.eval()
    p2_policy.eval()

    return Strategy(name=name, p1_policy=p1_policy, p2_policy=p2_policy, algorithm="soft")


def load_tough_strategy(name: str, path: Path) -> Strategy:
    """
    Load the tough heuristic strategy.

    Tough offers opponent 1 item of its least-valued type, insists rigidly.
    No checkpoint files needed.
    """
    p1_policy = ToughPolicy()
    p2_policy = ToughPolicy()

    p1_policy.eval()
    p2_policy.eval()

    return Strategy(name=name, p1_policy=p1_policy, p2_policy=p2_policy, algorithm="tough")


def load_openai_strategy(name: str, path: Path, model: str = "5.2",
                         reasoning_effort: str = "none") -> Strategy:
    """
    Load the OpenAI LLM bargaining strategy.

    Uses the same policy instance for both P1 and P2 (the policy reads
    the current player from the observation).
    No checkpoint files needed — requires OPENAI_API_KEY.txt or env var.

    Args:
        name: Base strategy name
        path: Path (unused for LLM strategies)
        model: OpenAI model ID (e.g. "gpt-4o", "5.2", "5.2-pro")
        reasoning_effort: Reasoning effort level ("none", "low", "medium", "high").
                          Only sent to models that support it.
    """
    # Only pass reasoning_effort for models that support it
    REASONING_MODELS = ("5.1", "5.2", "5.2-pro")
    effort = reasoning_effort if model in REASONING_MODELS else None

    # API model ID needs "gpt-" prefix
    GPT_PREFIX_MODELS = ("5", "5.1", "5.2", "5.2-pro")
    api_model = f"gpt-{model}" if model in GPT_PREFIX_MODELS else model

    p1_policy = OpenAIPolicy(model=api_model, reasoning_effort=effort)
    p2_policy = OpenAIPolicy(model=api_model, reasoning_effort=effort)

    p1_policy.eval()
    p2_policy.eval()

    strategy_name = f"openai_{model}_{reasoning_effort}" if effort else f"openai_{model}"
    return Strategy(name=strategy_name, p1_policy=p1_policy, p2_policy=p2_policy, algorithm="openai")


# Registry of loaders by algorithm name
STRATEGY_LOADERS: Dict[str, Callable] = {
    "ppo": load_ppo_strategy,
    "mmd": load_mmd_strategy,
    "mappo": load_mappo_strategy,
    "nfsp": load_nfsp_strategy,
    "psro": load_psro_strategy,
    "walk": load_walk_strategy,
    "soft": load_soft_strategy,
    "tough": load_tough_strategy,
    "llm_strategies": load_openai_strategy,
}


def load_strategy(name: str, path: Path, algorithm: Optional[str] = None) -> Strategy:
    """
    Load a strategy from a checkpoint directory.

    Args:
        name: Name to assign to the strategy
        path: Path to the strategy directory
        algorithm: Algorithm type (ppo, mappo, nfsp, psro).
                   If None, infers from directory name.

    Returns:
        Strategy object with p1 and p2 policies loaded
    """
    path = Path(path)

    if algorithm is None:
        algorithm = path.name.lower()

    if algorithm not in STRATEGY_LOADERS:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Supported: {list(STRATEGY_LOADERS.keys())}")

    return STRATEGY_LOADERS[algorithm](name, path)


def discover_strategies(strategies_dir: Path) -> Dict[str, Strategy]:
    """
    Discover and load all strategies in a directory.

    Expects structure:
        strategies_dir/
            ppo/
            mappo/
            nfsp/
            psro/
            ...

    Returns:
        Dict mapping strategy name to Strategy object
    """
    strategies_dir = Path(strategies_dir)
    strategies = {}

    for subdir in strategies_dir.iterdir():
        if not subdir.is_dir():
            continue

        algorithm = subdir.name.lower()
        if algorithm not in STRATEGY_LOADERS:
            print(f"Skipping unknown algorithm directory: {subdir.name}")
            continue

        try:
            strategy = load_strategy(name=algorithm, path=subdir, algorithm=algorithm)
            strategies[strategy.name] = strategy
            print(f"Loaded strategy: {strategy.name}")
        except FileNotFoundError as e:
            print(f"Could not load {algorithm}: {e}")
        except Exception as e:
            print(f"Error loading {algorithm}: {e}")

    return strategies


if __name__ == "__main__":
    # Test loading strategies
    strategies_dir = Path(__file__).parent.parent / "strategies"

    print(f"Discovering strategies in: {strategies_dir}")
    print("-" * 50)

    strategies = discover_strategies(strategies_dir)

    print("-" * 50)
    print(f"Loaded {len(strategies)} strategies: {list(strategies.keys())}")
