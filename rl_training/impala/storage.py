from dataclasses import dataclass, field
from typing import Dict, Iterable, List
import torch as th 

@dataclass
class ActorRollout:
    '''
    ActorRollout Class
    '''
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    behavior_log_probs: th.Tensor
    actor_id: int
    env_steps: int
    extras: Dict[str, th.Tensor] = field(default_factory=dict)

    def to(self, device: th.device) -> "ActorRollout":
        return ActorRollout(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            behavior_log_probs=self.behavior_log_probs.to(device),
            actor_id=self.actor_id,
            env_steps=self.env_steps,
            extras={k: v.to(device) for k, v in self.extras.items()},
        )

@dataclass
class LearnerBatch:
    '''
    LearnerBatch

    Class for storing batched output from actors for Learner
    '''
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    behavior_log_probs: th.Tensor
    bootstrap_observations: th.Tensor
    actor_ids: th.Tensor
    env_steps: int
    extras: Dict[str, th.Tensor] = field(default_factory=dict)

    def to(self, device: th.device) -> "LearnerBatch":
        return LearnerBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            behavior_log_probs=self.behavior_log_probs.to(device),
            bootstrap_observations=self.bootstrap_observations.to(device),
            actor_ids=self.actor_ids.to(device),
            env_steps=self.env_steps,
            extras={k: v.to(device) for k, v in self.extras.items()},
        )
    
def collate_rollout(rollouts: Iterable[ActorRollout]) -> LearnerBatch:
    '''
    Collates rollout batch for learner 
    '''
    rollouts = list(rollouts)
    if not rollouts:
        raise ValueError("Something happened there are no rollouts")
    
    obs = th.stack([r.observations[:-1] for r in rollouts], dim=1)
    bootstrap_obs = th.stack([r.observations[-1] for r in rollouts], dim=0)

    actions = th.stack([r.actions for r in rollouts ], dim=1)
    rewards = th.stack([r.rewards for r in rollouts ], dim=1)
    dones = th.stack([r.dones for r in rollouts ], dim=1)
    behavior_log_probs = th.stack([r.behavior_log_probs for r in rollouts ], dim=1)
    actor_ids = th.tensor([r.actor_id for r in rollouts], dtype=th.long)
    env_steps = sum(r.env_steps for r in rollouts)
    extras: Dict[str, th.Tensor] = {}
    shared_keys = set(rollouts[0].extras.keys())
    for r in rollouts[1:]:
        shared_keys &= set(r.extras.keys())
    for key in shared_keys:
        extras[key] = th.stack([r.extras[key] for r in rollouts], dim=1)
    return LearnerBatch(
        observations=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        behavior_log_probs=behavior_log_probs,
        bootstrap_observations=bootstrap_obs,
        actor_ids=actor_ids,
        env_steps=env_steps,
        extras=extras,
    )


    

