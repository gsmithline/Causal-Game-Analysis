"""
NFSP Trainer - Neural Fictitious Self-Play

Combines best-response learning (DQN) with average policy learning (supervised)
to approximate Nash equilibrium in extensive-form games.
"""
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch as th
from torch import nn, optim

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.nfsp.config import NFSPConfig
from rl_training.utils.replay_buffer import ReplayBuffer, ReservoirBuffer
from rl_training.envs.openspiel_wrapper import OpenSpielWrapper


class NFSPAgent:
    """Single NFSP agent with BR and average policy networks."""

    def __init__(
        self,
        player_id: int,
        obs_dim: int,
        num_actions: int,
        config: NFSPConfig,
        device: str,
    ):
        self.player_id = player_id
        self.config = config
        self.device = device
        self.num_actions = num_actions

        # Best response network (DQN)
        self.br_network = self._make_network(obs_dim, num_actions).to(device)
        self.br_target = self._make_network(obs_dim, num_actions).to(device)
        self.br_target.load_state_dict(self.br_network.state_dict())
        self.br_optimizer = optim.SGD(self.br_network.parameters(), lr=config.br_lr)
        self.br_buffer = ReplayBuffer(config.br_buffer_size)

        # Average policy network (supervised)
        self.avg_network = self._make_network(obs_dim, num_actions).to(device)
        self.avg_optimizer = optim.SGD(self.avg_network.parameters(), lr=config.avg_lr)
        self.avg_buffer = ReservoirBuffer(config.avg_buffer_size)

        self._epsilon = config.epsilon_start
        self._step = 0

    def _make_network(self, obs_dim: int, num_actions: int) -> nn.Module:
        """Create MLP network."""
        layers = []
        prev_dim = obs_dim
        for hidden in self.config.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden), nn.ReLU()])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, num_actions))
        return nn.Sequential(*layers)

    def get_action(
        self,
        obs: np.ndarray,
        legal_actions: List[int],
        mode: str = "average",
    ) -> int:
        """Select action using specified policy."""
        obs_t = th.as_tensor(obs, dtype=th.float32, device=self.device).unsqueeze(0)

        if mode == "best_response":
            # Epsilon-greedy best response
            if np.random.random() < self._epsilon:
                return int(np.random.choice(legal_actions))

            with th.no_grad():
                q_values = self.br_network(obs_t).squeeze(0).cpu().numpy()

            # Mask illegal actions
            masked_q = np.full(len(q_values), -np.inf)
            for a in legal_actions:
                masked_q[a] = q_values[a]
            return int(np.argmax(masked_q))

        else:  # average policy
            with th.no_grad():
                logits = self.avg_network(obs_t).squeeze(0)

            # Mask and softmax over legal actions
            mask = th.full_like(logits, -1e9)
            mask[legal_actions] = 0
            logits = logits + mask
            probs = th.softmax(logits, dim=-1).cpu().numpy()

            return int(np.random.choice(len(probs), p=probs))

    def update_br(self, batch_size: int) -> Optional[float]:
        """Update best response network (DQN update)."""
        if len(self.br_buffer) < batch_size:
            return None

        batch = self.br_buffer.sample(batch_size)
        obs = th.as_tensor(batch["obs"], device=self.device, dtype=th.float32)
        actions = th.as_tensor(batch["action"], device=self.device, dtype=th.long)
        rewards = th.as_tensor(batch["reward"], device=self.device, dtype=th.float32)
        next_obs = th.as_tensor(batch["next_obs"], device=self.device, dtype=th.float32)
        dones = th.as_tensor(batch["done"], device=self.device, dtype=th.float32)

        # Current Q values
        q_values = self.br_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with th.no_grad():
            next_q = self.br_target(next_obs).max(dim=1)[0]
            target_q = rewards + self.config.discount * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)

        self.br_optimizer.zero_grad()
        loss.backward()
        self.br_optimizer.step()

        return loss.item()

    def update_avg(self, batch_size: int) -> Optional[float]:
        """Update average policy network (supervised learning)."""
        if len(self.avg_buffer) < batch_size:
            return None

        batch = self.avg_buffer.sample(batch_size)
        obs = th.as_tensor(batch["obs"], device=self.device, dtype=th.float32)
        actions = th.as_tensor(batch["action"], device=self.device, dtype=th.long)

        # Cross-entropy loss
        logits = self.avg_network(obs)
        loss = nn.functional.cross_entropy(logits, actions)

        self.avg_optimizer.zero_grad()
        loss.backward()
        self.avg_optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        """Copy BR network to target network."""
        self.br_target.load_state_dict(self.br_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration."""
        self._step += 1
        progress = min(1.0, self._step / self.config.epsilon_decay_steps)
        self._epsilon = (
            self.config.epsilon_start
            + progress * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "br_network": self.br_network.state_dict(),
            "br_target": self.br_target.state_dict(),
            "br_optimizer": self.br_optimizer.state_dict(),
            "avg_network": self.avg_network.state_dict(),
            "avg_optimizer": self.avg_optimizer.state_dict(),
            "epsilon": self._epsilon,
            "step": self._step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.br_network.load_state_dict(state_dict["br_network"])
        self.br_target.load_state_dict(state_dict["br_target"])
        self.br_optimizer.load_state_dict(state_dict["br_optimizer"])
        self.avg_network.load_state_dict(state_dict["avg_network"])
        self.avg_optimizer.load_state_dict(state_dict["avg_optimizer"])
        self._epsilon = state_dict["epsilon"]
        self._step = state_dict["step"]


@register_algorithm("nfsp")
class NFSPTrainer(BaseTrainer[NFSPConfig]):
    """
    Neural Fictitious Self-Play trainer.

    Combines best-response learning (off-policy DQN) with
    average policy learning (supervised from reservoir buffer).

    Args:
        config: NFSP configuration.
        env_fn: Optional environment factory (uses config.env_id if not provided).
        logger: Optional logger instance.
    """

    def __init__(
        self,
        config: NFSPConfig,
        env_fn: Optional[Callable] = None,
        logger: Optional[Any] = None,
    ):
        super().__init__(config, env_fn, logger)

        self.env: Optional[OpenSpielWrapper] = None
        self.agents: List[NFSPAgent] = []
        self._total_episodes = 0

    def _setup(self) -> None:
        """Initialize NFSP components."""
        # Create environment
        if self.env_fn is not None:
            self.env = self.env_fn()
        else:
            self.env = OpenSpielWrapper(self.config.env_id)

        obs_dim = self.env.observation_shape[0]
        num_actions = self.env.num_actions

        # Create agents for each player
        self.agents = []
        for player in range(self.env.num_players):
            agent = NFSPAgent(
                player_id=player,
                obs_dim=obs_dim,
                num_actions=num_actions,
                config=self.config,
                device=self.config.device,
            )
            self.agents.append(agent)

    def _train_step(self) -> Dict[str, Any]:
        """
        One NFSP training step.

        1. Play episode with mixed strategies
        2. Store transitions for BR training
        3. Store actions for average policy training
        4. Update networks
        """
        # Play one episode
        episode_data = self._play_episode()
        self._total_episodes += 1

        # Store transitions and train
        metrics = {"timesteps": episode_data["length"]}

        for player, agent in enumerate(self.agents):
            # Store BR transitions
            for transition in episode_data["transitions"][player]:
                agent.br_buffer.add(transition)

            # Store average policy data
            for action_data in episode_data["avg_data"][player]:
                agent.avg_buffer.add(action_data)

            # Update BR network
            br_loss = agent.update_br(self.config.br_batch_size)
            if br_loss is not None:
                metrics[f"br_loss_p{player}"] = br_loss

            # Update average policy periodically
            if self._total_episodes % self.config.avg_train_freq == 0:
                avg_loss = agent.update_avg(self.config.avg_batch_size)
                if avg_loss is not None:
                    metrics[f"avg_loss_p{player}"] = avg_loss

            # Update target network
            if self._total_episodes % self.config.br_target_update_freq == 0:
                agent.update_target()

            agent.decay_epsilon()

        metrics["episodes"] = self._total_episodes
        metrics["epsilon"] = self.agents[0]._epsilon

        return metrics

    def _play_episode(self) -> Dict[str, Any]:
        """Play one episode, collecting data for both networks."""
        obs, info = self.env.reset()

        transitions = {p: [] for p in range(len(self.agents))}
        avg_data = {p: [] for p in range(len(self.agents))}
        length = 0

        while True:
            player = self.env.current_player
            legal_actions = info["legal_actions"]

            # Choose mode (best response or average) based on eta
            if np.random.random() < self.config.eta:
                mode = "best_response"
            else:
                mode = "average"

            agent = self.agents[player]
            action = agent.get_action(obs, legal_actions, mode=mode)

            # Store for average policy (only when using BR)
            if mode == "best_response":
                avg_data[player].append({"obs": obs.copy(), "action": action})

            # Take step
            result = self.env.step(action)
            next_obs = result.observation
            reward = result.reward
            done = result.done

            # Store BR transition
            transitions[player].append({
                "obs": obs.copy(),
                "action": action,
                "reward": reward,
                "next_obs": next_obs.copy(),
                "done": float(done),
            })

            length += 1

            if done:
                # Add terminal rewards for other players
                if hasattr(self.env, '_state') and self.env._state is not None:
                    returns = self.env._state.returns()
                    for p in range(len(self.agents)):
                        if p != player and transitions[p]:
                            transitions[p][-1]["reward"] = returns[p]
                break

            obs = next_obs
            info = result.info

        return {
            "transitions": transitions,
            "avg_data": avg_data,
            "length": length,
        }

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "agents": [agent.state_dict() for agent in self.agents],
            "total_episodes": self._total_episodes,
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        for agent, agent_data in zip(self.agents, data["agents"]):
            agent.load_state_dict(agent_data)
        self._total_episodes = data["total_episodes"]

    def get_average_policy(self, player: int = 0) -> nn.Module:
        """Return average policy network for a player."""
        return self.agents[player].avg_network

    def _cleanup(self) -> None:
        """Clean up environment."""
        if self.env is not None:
            self.env.close()
