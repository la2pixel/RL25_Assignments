"""
Comprl client: run from project root with
  python wandb_version/run_client.py --server-url ... --token ... --args --agent td3 --model_path path/to.pth
  python wandb_version/run_client.py ... --args --agent td3_old --model_path path/to.pth [--hidden_sizes 1024 1024]
Uses hockey_td3.TD3Agent (same as training); td3_old = TD3 with optional hidden_sizes for old checkpoints.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid

# Project root = parent of wandb_version, so hockey, hockey_td3, hockey_sac resolve
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import hockey.hockey_env as h_env
import numpy as np

from comprl.client import Agent, launch_client

from hockey_sac import SAC
from hockey_td3 import TD3Agent


class RandomAgent(Agent):
    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"game ended: {text_result} with my score: {stats[0]} against the opponent with score: {stats[1]}")


class HockeyAgent(Agent):
    def __init__(self, weak: bool) -> None:
        super().__init__()
        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} with my score: {stats[0]} against the opponent with score: {stats[1]}")


class TD3ClientAgent(Agent):
    """Uses hockey_td3.TD3Agent; pass hidden_sizes for td3 vs td3_old (e.g. [512,512] vs [1024,1024])."""

    def __init__(self, model_path: str, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        env = h_env.HockeyEnv()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] // 2
        hidden_sizes = hidden_sizes or [512, 512]
        self.td3_agent = TD3Agent(
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
        print(f"Game ended: {text_result} with my score: {stats[0]} against the opponent with score: {stats[1]}")


class SACClientAgent(Agent):
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
        print(f"Game ended: {text_result} with my score: {stats[0]} against the opponent with score: {stats[1]}")


def initialize_agent(agent_args: list[str]) -> Agent:
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
        help="Path to .pth (required for td3, td3_old, sac).",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=None,
        help="For td3/td3_old: hidden layer sizes (e.g. 1024 1024). Default td3: 512 512, td3_old: 1024 1024.",
    )
    agent_args = [a for a in agent_args if a != "--args"]
    args = parser.parse_args(agent_args)

    if args.agent == "weak":
        return HockeyAgent(weak=True)
    if args.agent == "strong":
        return HockeyAgent(weak=False)
    if args.agent == "random":
        return RandomAgent()
    if args.agent == "td3":
        if not args.model_path:
            parser.error("--model_path is required when --agent td3")
        hidden_sizes = args.hidden_sizes if args.hidden_sizes else [512, 512]
        return TD3ClientAgent(args.model_path, hidden_sizes=hidden_sizes)
    if args.agent == "td3_old":
        if not args.model_path:
            parser.error("--model_path is required when --agent td3_old")
        hidden_sizes = args.hidden_sizes if args.hidden_sizes else [1024, 1024]
        return TD3ClientAgent(args.model_path, hidden_sizes=hidden_sizes)
    if args.agent == "sac":
        if not args.model_path:
            parser.error("--model_path is required when --agent sac")
        return SACClientAgent(args.model_path)
    raise ValueError(f"Unknown agent: {args.agent}")


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
