import threading
from queue import Queue, Full
import torch as th
from rl_training.algorithms.impala.storage import ActorRollout
from rl_training.algorithms.impala.config import ImpalaConfig

class ActorWorker(threading.Thread):
    def __init__(self, actor_id, env_fn, policy_fn, param_server, 
                 queue: Queue, config: ImpalaConfig):
        super().__init__(daemon=True)
        self.actor_id = actor_id
        self.env = env_fn()
        self.policy = policy_fn().to(config.actor_device)
        self.param_server = param_server
        self.queue = queue
        self.config = config
        self.shutdown = threading.Event()
        self.steps_since_refresh = 0
        self.env_steps = 0
    
    def _reset_env(self):
        '''
        this needs to be updated for the new openspiel
        '''
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return obs
        return out

    def _step_env(self, action_np): 
        '''
        this needs to be updated for the new openspiel
        '''
        out = self.env.step(action_np)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, reward, done, info
        return out

    def run(self):
        obs = self._reset_env()
        '''
        store observations unbatched (envs=1). 
        we add a batch dim only when calling the policy to keep 
        learner tensors time-major [T, B, ...].
        '''
        obs = th.as_tensor(obs, device=self.config.actor_device)

        while not self.shutdown.is_set():
            if self.steps_since_refresh == 0 or self.steps_since_refresh >= self.config.refresh_interval:
                _version, weights = self.param_server.get()
                self.policy.load_state_dict(weights, strict=False)
                self.steps_since_refresh = 0
            rollout = self.collect_rollout(obs)
            try:
                self.queue.put(rollout, timeout=1.0)
            except Full:
                continue
            obs = rollout.observations[-1]
            self.steps_since_refresh += rollout.env_steps
            self.env_steps += rollout.env_steps
    
    def collect_rollout(self, obs0):
        obs_list = [obs0]
        actions, rewards, dones, beh_logps = [], [], [], []
        extras = {}

        for _ in range(self.config.unroll_length):
            with th.no_grad():
                action_dist, value, extra = self.policy(obs_list[-1].unsqueeze(0))

            action = action_dist.sample()
            beh_logp = action_dist.log_prob(action)
            if beh_logp.ndim > 1:
                beh_logp = beh_logp.sum(dim=-1)
            beh_logp = beh_logp.squeeze(0)
            action = action.squeeze(0)
            
            next_obs, reward, done, info = self._step_env(action.cpu().numpy())
            next_obs = th.as_tensor(next_obs, device=self.config.actor_device)
            obs_list.append(next_obs)
            actions.append(action)
            rewards.append(th.tensor(reward, device=self.config.actor_device))
            dones.append(th.tensor(done, device=self.config.actor_device))
            beh_logps.append(beh_logp)
            for k, v in extra.items():
                # the policy extras are typically batched but we store unbatched.
                extras.setdefault(k, []).append(v.squeeze(0) if isinstance(v, th.Tensor) else v)
            if done:
                next_obs = th.as_tensor(self._reset_env(), device=self.config.actor_device)
                obs_list[-1] = next_obs
        to_tensor = lambda xs: th.stack(xs, dim=0)
        extras = {k: th.stack(v, dim=0) for k, v in extras.items()}
        return ActorRollout(
            observations=th.stack(obs_list, dim=0),
            actions=to_tensor(actions),
            rewards=to_tensor(rewards),
            dones=to_tensor(dones),
            behavior_log_probs=to_tensor(beh_logps),
            actor_id=self.actor_id,
            env_steps=self.config.unroll_length,
            extras=extras,
        )
    


            














                
        










