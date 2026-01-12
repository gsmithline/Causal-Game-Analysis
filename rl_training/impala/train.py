"""
IMPALA Training Orchestration

Usage:
    python -m rl_training.impala.train

This script ties together actors, learner, and parameter server
for distributed training with V-trace off-policy correction.
"""
import signal
import sys
from queue import Queue
from typing import Callable, Optional
import torch as th

from rl_training.impala.config import ImpalaConfig
from rl_training.impala.actor import ActorWorker
from rl_training.impala.learner import Learner
from rl_training.impala.parameter_server import ParameterServer


class SimpleLogger:
    """Minimal logger that prints to stdout."""
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0

    def log(self, metrics: dict):
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            metrics_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                      for k, v in metrics.items())
            print(f"[Step {self.step_count}] {metrics_str}")


class ImpalaTrainer:
    """
    Orchestrates IMPALA training with multiple actors and a single learner.

    Args:
        env_fn: Factory function that creates a new environment instance.
        policy_fn: Factory function that creates a new policy instance.
        config: IMPALA configuration.
        logger: Optional logger instance. If None, uses SimpleLogger.
    """

    def __init__(
        self,
        env_fn: Callable,
        policy_fn: Callable,
        config: Optional[ImpalaConfig] = None,
        logger=None,
    ):
        self.config = config or ImpalaConfig()
        self.logger = logger or SimpleLogger()
        self.env_fn = env_fn
        self.policy_fn = policy_fn

        # Create learner policy and initialize parameter server with its weights
        self.policy = policy_fn().to(self.config.learner_device)
        self.param_server = ParameterServer(initial_state_dict=self.policy.state_dict())

        # Create shared queue for actor -> learner communication
        self.queue: Queue = Queue(maxsize=self.config.actor_queue_size)

        # Create learner
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
                env_fn=env_fn,
                policy_fn=policy_fn,
                param_server=self.param_server,
                queue=self.queue,
                config=self.config,
            )
            self.actors.append(actor)

        self._shutdown_requested = False

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        def handler(signum, frame):
            print("\nShutdown requested, stopping actors...")
            self._shutdown_requested = True
            for actor in self.actors:
                actor.shutdown.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def train(self, num_learner_steps: int, checkpoint_path: Optional[str] = None):
        """
        Run training loop.

        Args:
            num_learner_steps: Total number of learner update steps to run.
            checkpoint_path: Optional path to save checkpoints.
        """
        self._setup_signal_handlers()

        # Start all actors
        print(f"Starting {self.config.num_actors} actors...")
        for actor in self.actors:
            actor.start()

        print(f"Training for {num_learner_steps} learner steps...")
        try:
            for step in range(num_learner_steps):
                if self._shutdown_requested:
                    print("Shutdown detected, exiting training loop.")
                    break

                self.learner.step()

                # Checkpoint
                if checkpoint_path and (step + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_path, step + 1)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self._shutdown()

        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, self.learner.learner_steps)

        return self.policy

    def _save_checkpoint(self, path: str, step: int):
        """Save a training checkpoint."""
        checkpoint = {
            "learner_steps": step,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.learner.optimizer.state_dict(),
            "config": self.config,
        }
        save_path = f"{path}_step{step}.pt"
        th.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def _shutdown(self):
        """Gracefully shutdown all actors."""
        print("Shutting down actors...")
        for actor in self.actors:
            actor.shutdown.set()

        # Wait for actors to finish (with timeout)
        for actor in self.actors:
            actor.join(timeout=2.0)

        # Drain remaining items from queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break

        print("Shutdown complete.")


def main():
    """Example training with CartPole."""
    try:
        import gymnasium as gym
    except ImportError:
        print("Please install gymnasium: pip install gymnasium")
        sys.exit(1)

    from rl_training.impala.simple_policy import SimpleCategoricalMLP

    # Environment factory
    def make_env():
        return gym.make("CartPole-v1")

    # Get observation/action dimensions from a temporary env
    temp_env = make_env()
    obs_dim = temp_env.observation_space.shape[0]
    num_actions = temp_env.action_space.n
    temp_env.close()

    # Policy factory
    def make_policy():
        return SimpleCategoricalMLP(obs_dim=obs_dim, num_actions=num_actions)

    # Config
    config = ImpalaConfig(
        num_actors=4,
        batch_size=4,
        unroll_length=32,
        lr=3e-4,
        entropy_coef=0.01,
    )

    # Train
    trainer = ImpalaTrainer(
        env_fn=make_env,
        policy_fn=make_policy,
        config=config,
    )

    trained_policy = trainer.train(num_learner_steps=1000)
    print("Training complete!")


if __name__ == "__main__":
    main()
