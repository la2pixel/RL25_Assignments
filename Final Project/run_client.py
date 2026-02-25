from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comprl.client import Agent, launch_client

from hockey_sac import SAC
from hockey_td3 import TD3Agent


def _linear_decay(progress: float, start: float, end: float) -> float:
    return start + (end - start) * progress


class ActorOld(nn.Module):
    """Old TD3 actor: configurable hidden_sizes list (e.g. [1024, 1024] or [512, 512])."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int], dropout: float = 0.0):
        super().__init__()
        sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, state):
        x = state
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            if self.dropout is not None:
                x = self.dropout(x)
        x = torch.tanh(self.layers[-1](x))
        return x


class CriticOld(nn.Module):
    """Old TD3 critic: configurable hidden_sizes list."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int], dropout: float = 0.0):
        super().__init__()
        input_dim = state_dim + action_dim
        sizes = [input_dim] + list(hidden_sizes) + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, state, action):
        x = torch.hstack([state, action])
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            if self.dropout is not None:
                x = self.dropout(x)
        x = self.layers[-1](x)
        return x


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

class TD3AgentOld:
    # When improvement=True we decay to these end values over the schedule
    IMPROVEMENT_POLICY_NOISE_END = 0.05
    IMPROVEMENT_DROPOUT_END = 0.0
    IMPROVEMENT_WEIGHT_DECAY_END = 0.0

    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        polyak=0.995,
        policy_lr=3e-4,
        critic_lr=3e-4,
        act_noise_std=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        hidden_sizes=[1024, 1024],
        improvement=False,
        dropout=0.1,
        weight_decay=1e-5,
        schedule_total_updates=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if improvement:
            drop = dropout
            wd = weight_decay
            self.schedule_total_updates = schedule_total_updates or 1
            self._policy_noise_start = policy_noise
            self._dropout_start = dropout
            self._weight_decay_start = weight_decay
        else:
            drop = 0.0
            wd = 0.0
            self.schedule_total_updates = None
            self._policy_noise_start = self._dropout_start = self._weight_decay_start = None

        self.actor = ActorOld(state_dim, action_dim, hidden_sizes=hidden_sizes, dropout=drop).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr, weight_decay=wd)

        self.actor_target = ActorOld(state_dim, action_dim, hidden_sizes=hidden_sizes, dropout=drop).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = CriticOld(state_dim, action_dim, hidden_sizes=hidden_sizes, dropout=drop).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=wd)

        self.critic1_target = CriticOld(state_dim, action_dim, hidden_sizes=hidden_sizes, dropout=drop).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = CriticOld(state_dim, action_dim, hidden_sizes=hidden_sizes, dropout=drop).to(self.device)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=wd)

        self.critic2_target = CriticOld(state_dim, action_dim, hidden_sizes=hidden_sizes, dropout=drop).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.polyak = polyak
        self.act_noise_std = act_noise_std
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_updates = 0

        print(f"using device: {self.device}")

    # soft update of target network: blend target with main network using polyak
    def polyak_update(self, main_network, target_network):
        for main_param, target_param in zip(main_network.parameters(), target_network.parameters()):
            target_param.data.copy_(
                self.polyak * target_param.data + (1 - self.polyak) * main_param.data
            )

    def select_action(self, state, add_noise=True, deterministic=None):
        if deterministic is not None:
            add_noise = not deterministic

        # ensure we have a batch dimension: (state_dim,) -> (1, state_dim), (batch, state_dim) stays (batch, state_dim)
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.ndim == 1:
            state_batch = state_np.reshape(1, -1)
            single_input = True
        else:
            state_batch = state_np
            single_input = False

        state_tensor = torch.FloatTensor(state_batch).to(self.device)
        action = self.actor(state_tensor).cpu().detach().numpy()

        # add exploration noise per action dimension (same for each row in the batch)
        if add_noise:
            noise = np.random.normal(0, self.act_noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, -1, 1)

        # return same shape as expected: 1D for single obs, 2D for batch
        if single_input:
            return action.flatten()
        return action

    def _schedule(self):
        """ AI USAGE: When improvement=True: linear decay of policy_noise, dropout, weight_decay to fixed end values."""
        if self.schedule_total_updates is None or self.schedule_total_updates <= 0:
            return
        progress = min(1.0, self.total_updates / self.schedule_total_updates)
        self.policy_noise = _linear_decay(progress, self._policy_noise_start, self.IMPROVEMENT_POLICY_NOISE_END)
        drop_p = _linear_decay(progress, self._dropout_start, self.IMPROVEMENT_DROPOUT_END)
        wd = _linear_decay(progress, self._weight_decay_start, self.IMPROVEMENT_WEIGHT_DECAY_END)
        if self.actor.dropout is not None:
            self.actor.dropout.p = drop_p
            self.actor_target.dropout.p = drop_p
        if self.critic1.dropout is not None:
            self.critic1.dropout.p = drop_p
            self.critic1_target.dropout.p = drop_p
            self.critic2.dropout.p = drop_p
            self.critic2_target.dropout.p = drop_p
        for opt in (self.actor_optimizer, self.critic1_optimizer, self.critic2_optimizer):
            for g in opt.param_groups:
                g["weight_decay"] = wd

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        observations, actions, rewards, next_observations, dones = replay_buffer.sample(batch_size)
        self.total_updates += 1
        self._schedule()

        # computing target Q-Values
        with torch.no_grad():
            # target policy smoothing to reduce exploitation in Q-Value errors
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_observations) + noise).clamp(-1, 1)

            target_q1 = self.critic1_target(next_observations, next_actions)
            target_q2 = self.critic2_target(next_observations, next_actions)
            target_q = torch.min(target_q1, target_q2)  # double Q-Learning to reduce overestimation of Q-Values
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # computing current Q-Values
        current_q1 = self.critic1(observations, actions)
        current_q2 = self.critic2(observations, actions)

        # update step for critic 1
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # update step for critic 2
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # delayed policy updates to reduce volatility
        if self.total_updates % self.policy_delay == 0:
            actor_actions = self.actor(observations)
            # update step for the actor
            actor_loss = -self.critic1(observations, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target network for actor, critic1 and critic2
            self.polyak_update(self.actor, self.actor_target)
            self.polyak_update(self.critic1, self.critic1_target)
            self.polyak_update(self.critic2, self.critic2_target)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, filename)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


class TD3ClientAgent(Agent):
    """Hockey agent that uses a trained TD3 model."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        env = h_env.HockeyEnv()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] // 2
        self.td3_agent = TD3Agent(state_dim=obs_dim, action_dim=action_dim, improvement=False)
        self.td3_agent.load(model_path)

    def get_step(self, observation: list[float]) -> list[float]:
        obs = np.asarray(observation, dtype=np.float32)
        action = self.td3_agent.select_action(obs, deterministic=True)
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class SACClientAgent(Agent):
    """Hockey agent that uses a trained SAC model."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        env = h_env.HockeyEnv()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] // 2
        self.sac_agent = SAC(obs_dim, action_dim, device="cpu", hidden_sizes=(512, 512))
        self.sac_agent.load(model_path)

    def get_step(self, observation: list[float]) -> list[float]:
        obs = np.asarray(observation, dtype=np.float32)
        action = self.sac_agent.select_action(obs, deterministic=True)
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class TD3ClientAgentOld(Agent):
    """Hockey agent that uses a trained TD3 model with the old architecture (configurable hidden_sizes)."""

    def __init__(self, model_path: str, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        env = h_env.HockeyEnv()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] // 2
        hidden_sizes = hidden_sizes or [1024, 1024]
        self.td3_agent = TD3AgentOld(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            improvement=False,
        )
        self.td3_agent.load(model_path)

    def get_step(self, observation: list[float]) -> list[float]:
        obs = np.asarray(observation, dtype=np.float32)
        action = self.td3_agent.select_action(obs, deterministic=True)
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "td3", "td3_old", "sac"],
        default="weak",
        help="Which agent to use.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to .pth checkpoint (required when --agent td3, td3_old, or sac).",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes for td3_old (e.g. 512 512). Default: 1024 1024.",
    )
    # comprl may pass --args through; strip it so our parser does not see it
    agent_args = [a for a in agent_args if a != "--args"]
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "td3":
        if not args.model_path:
            parser.error("--model_path is required when --agent td3")
        agent = TD3ClientAgent(args.model_path)
    elif args.agent == "td3_old":
        if not args.model_path:
            parser.error("--model_path is required when --agent td3_old")
        hidden_sizes = args.hidden_sizes if args.hidden_sizes else [1024, 1024]
        agent = TD3ClientAgentOld(args.model_path, hidden_sizes=hidden_sizes)
    elif args.agent == "sac":
        if not args.model_path:
            parser.error("--model_path is required when --agent sac")
        agent = SACClientAgent(args.model_path)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
