"""
IMPALA Trainer - Importance Weighted Actor-Learner Architecture

Distributed training with multiple actors collecting experience
and a central learner updating the policy using V-trace.
"""
from queue import Queue
from typing import Any, Callable, Dict, Optional

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.impala.config import ImpalaConfig
from rl_training.algorithms.impala.actor import ActorWorker
from rl_training.algorithms.impala.learner import Learner
from rl_training.algorithms.impala.parameter_server import ParameterServer


@register_algorithm("impala")
class ImpalaTrainer(BaseTrainer[ImpalaConfig]):
    """
    IMPALA trainer with actor-learner architecture.

    Multiple actor threads collect experience asynchronously,
    while a central learner updates the policy using V-trace
    off-policy correction.

    Args:
        config: IMPALA configuration.
        env_fn: Factory function that creates environment instances.
        policy_fn: Factory function that creates policy instances.
        logger: Optional logger instance.
    """

    def __init__(
        self,
        config: ImpalaConfig,
        env_fn: Callable,
        policy_fn: Callable,
        logger: Optional[Any] = None,
    ):
        super().__init__(config, env_fn, logger)
        self.policy_fn = policy_fn

        # Will be initialized in _setup
        self.policy = None
        self.param_server = None
        self.learner = None
        self.actors = []
        self.queue = None

    def _setup(self) -> None:
        """Initialize IMPALA components."""
        # Create learner policy
        self.policy = self.policy_fn().to(self.config.learner_device)

        # Parameter server with initial weights
        self.param_server = ParameterServer(
            initial_state_dict=self.policy.state_dict()
        )

        # Shared queue for actor -> learner communication
        self.queue = Queue(maxsize=self.config.actor_queue_size)

        # Learner (uses our logger)
        self.learner = Learner(
            policy=self.policy,
            config=self.config,
            param_server=self.param_server,
            queue=self.queue,
            logger=self.logger,
        )

        # Create actors
        self.actors = []
        for i in range(self.config.num_actors):
            actor = ActorWorker(
                actor_id=i,
                env_fn=self.env_fn,
                policy_fn=self.policy_fn,
                param_server=self.param_server,
                queue=self.queue,
                config=self.config,
            )
            self.actors.append(actor)

    def _on_training_start(self) -> None:
        """Start actor threads."""
        self.logger.info(f"Starting {self.config.num_actors} actors...")
        for actor in self.actors:
            actor.start()

    def _train_step(self) -> Dict[str, Any]:
        """Execute one learner update step."""
        metrics = self.learner.step()

        # Add timesteps for BaseTrainer tracking
        timesteps = self.config.batch_size * self.config.unroll_length
        metrics["timesteps"] = timesteps

        return metrics

    def _cleanup(self) -> None:
        """Shutdown actors gracefully."""
        self.logger.info("Shutting down actors...")
        for actor in self.actors:
            actor.shutdown.set()

        # Wait for actors to finish
        for actor in self.actors:
            actor.join(timeout=2.0)

        # Drain remaining items from queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break

        self.logger.info("Actors shutdown complete.")

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """Return data needed to resume training."""
        return {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.learner.optimizer.state_dict(),
            "learner_steps": self.learner.learner_steps,
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        """Restore trainer state from checkpoint."""
        self.policy.load_state_dict(data["policy_state_dict"])
        self.learner.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.learner.learner_steps = data["learner_steps"]
        # Update parameter server with loaded weights
        self.param_server.update(self.policy.state_dict())

    def get_policy(self):
        """Return the trained policy."""
        return self.policy
