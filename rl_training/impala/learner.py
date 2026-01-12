from queue import Queue
import torch as th
from torch import nn, optim
from rl_training.impala.storage import collate_rollout
from rl_training.impala.vtrace import compute_vtrace
from rl_training.impala.config import ImpalaConfig

class Learner:
    def __init__(self, policy: nn.Module, config: ImpalaConfig, param_server, queue: Queue, logger):
        self.policy = policy.to(config.learner_device)
        self.config = config
        self.param_server = param_server
        self.queue = queue
        self.logger = logger
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.learner_steps = 0
    
    def step(self):
        rollouts = [self.queue.get() 
                    for _ in range(self.config.batch_size)]
        batch = collate_rollout(rollouts).to(th.device(self.config.learner_device))

        logits, values = self.policy.forward_batch(batch.observations)
        target_log_probs = self.policy.log_prob(batch.actions, logits)
        bootstrap_value = self.policy.value(batch.bootstrap_observations)
        discounts = self.config.discount * (1.0 - batch.dones.float())
        vtrace = compute_vtrace(
            behavior_log_probs=batch.behavior_log_probs,
            target_log_probs=target_log_probs,
            rewards=batch.rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discounts=discounts,
            clip_rho_threshold=self.config.clip_rho_threshold,
            clip_c_threshold=self.config.clip_c_threshold,
        )

        pg_loss = -(vtrace.pg_advantages.detach() * target_log_probs).mean()

        #treat v-trace outputs as regression targets (stop-gradient).
        #TODO:  
            #explore updating for classification
        baseline_loss = 0.5 * (values - vtrace.vs.detach()).pow(2).mean()

        entropy = self.policy.entropy(logits).mean()
       

        loss = pg_loss + self.config.value_coef * baseline_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.learner_steps += 1
        self.param_server.update(self.policy.state_dict())
        self.logger.log(dict(
            learner_steps=self.learner_steps,
            policy_loss=pg_loss.item(),
            value_loss=baseline_loss.item(),
            entropy=entropy.item()
        ))



