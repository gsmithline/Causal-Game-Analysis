from dataclasses import dataclass
from typing import Dict, Tuple

import torch as th
from torch import nn
from torch.distributions import Categorical

'''
note this is for testing the impala implementation
'''
@dataclass
class PolicyOutput:
    dist: Categorical
    value: th.Tensor
    extra: Dict[str, th.Tensor]


class SimpleCategoricalMLP(nn.Module):
    """
    Minimal policy/value network for discrete action spaces.

    Implements the API expected by `ActorWorker` and `Learner`:
      - __call__(obs) -> (action_dist, value, extra)
      - forward_batch(observations) -> (logits, values)
      - log_prob(actions, logits) -> [T, B]
      - value(observations) -> [B]
      - entropy(logits) -> [T, B]
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)

        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, self.num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _encode(self, obs: th.Tensor) -> th.Tensor:
        '''
        accept obs as [..., obs_dim].
        '''
        if obs.dtype != th.float32:
            obs = obs.float()
        return self.trunk(obs)

    def forward(self, obs: th.Tensor) -> Tuple[Categorical, th.Tensor, Dict[str, th.Tensor]]:
        h = self._encode(obs)
        logits = self.policy_head(h)
        dist = Categorical(logits=logits)
        value = self.value_head(h).squeeze(-1)
        return dist, value, {"logits": logits}

    def forward_batch(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        '''
        observations: [T, B, obs_dim]
        '''
        h = self._encode(observations)
        logits = self.policy_head(h) 
        values = self.value_head(h).squeeze(-1) 
        return logits, values

    def log_prob(self, actions: th.Tensor, logits: th.Tensor) -> th.Tensor:
        '''
        actions: [T, B] (or [T, B, 1])
        '''
        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions) 

    def value(self, observations: th.Tensor) -> th.Tensor:
        '''
        observations: [B, obs_dim]
        '''
        h = self._encode(observations)
        return self.value_head(h).squeeze(-1) 

    def entropy(self, logits: th.Tensor) -> th.Tensor:
        dist = Categorical(logits=logits)
        return dist.entropy() 


