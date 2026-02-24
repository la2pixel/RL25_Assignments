#!/usr/bin/env python3
"""
Evaluate a trained SAC or TD3 agent on the hockey environment.

Usage:
    python evaluate_hockey.py --model_path checkpoints/sac_hockey_best.pth --opponent weak
    python evaluate_hockey.py --model_path checkpoints/td3_hockey_best.pth --opponent strong
    python evaluate_hockey.py --model_path checkpoints/agent_v1.pth sac --opponent_model checkpoints/agent_v2.pth
    python
"""

import argparse
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hockey_sac import SAC
from hockey_td3 import TD3Agent


def get_state(state):
    if isinstance(state, np.ndarray):
        state = np.ascontiguousarray(state, dtype=np.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        return state
    return np.array(state, dtype=np.float32)


def evaluate(env, agent1, num_episodes=10, render=False, max_steps=500,
             seeds=None, opponent_agent=None, opponent_model=None):
    episode_rewards = []
    wins, losses, draws = 0, 0, 0
    last_info = {}

    if seeds is None:
        seeds = [int(s) for s in np.random.randint(0, 10000, size=num_episodes)]

    if opponent_model is not None:
        opp_name = "Trained Agent (Player 2)"
    elif opponent_agent is not None:
        opp_name = "Built-in AI"
    else:
        opp_name = "Environment Default"

    print(f"\nEvaluating: Agent (P1) vs {opp_name} for {num_episodes} episodes...")
    print("-" * 80)
    print(f"{'Ep':>3} | {'Reward':>8} | {'Steps':>5} | {'Outcome':<8}")
    print("-" * 80)

    for i, seed in enumerate(seeds[:num_episodes]):
        try:
            obs, info = env.reset(seed=seed)
        except TypeError:
            obs, info = env.reset()

        obs = get_state(obs)
        episode_reward = 0.0
        step = 0

        while step < max_steps:
            a1 = agent1.select_action(obs, deterministic=True)
            obs_p2 = env.obs_agent_two()

            if opponent_model is not None:
                a2 = opponent_model.select_action(get_state(obs_p2), deterministic=True)
            elif opponent_agent is not None:
                a2 = opponent_agent.act(obs_p2)
            else:
                a2 = np.zeros(4)

            obs, reward, done, trunc, info = env.step(np.hstack([a1, a2]))
            last_info = info
            obs = get_state(obs)
            episode_reward += reward
            step += 1

            if render:
                env.render()
            if done or trunc:
                break

        episode_rewards.append(episode_reward)
        winner = last_info.get('winner', 0)
        if winner == 1:
            outcome = "WIN"; wins += 1
        elif winner == -1:
            outcome = "LOSS"; losses += 1
        else:
            outcome = "DRAW"; draws += 1

        print(f"{i+1:3d} | {episode_reward:8.2f} | {step:5d} | {outcome}")

    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'win_rate': wins / num_episodes * 100,
        'wins': wins, 'losses': losses, 'draws': draws
    }

    print("-" * 80)
    print(f"Results vs {opp_name}:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"  Win Rate:    {stats['win_rate']:.1f}%")
    print(f"  Record:      {wins}W - {losses}L - {draws}D")
    print("-" * 80)
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC Agent')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--opponent_model', type=str, default=None)
    parser.add_argument('--opponent', type=str, default='weak', choices=['weak', 'strong'])
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # infer agent and opponent type from file name
    agent_type = 'td3' if 'td3' in args.model_path.lower() else 'sac'
    opponent_type = 'td3' if (args.opponent_model and 'td3' in args.opponent_model.lower()) else ('sac' if args.opponent_model else None)

    try:
        import hockey.hockey_env as h_env
    except ImportError:
        print("Could not import hockey package.")
        return

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = h_env.HockeyEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2

    # Load Player 1 (SAC: pink_noise flag doesn't matter for eval — always deterministic)
    print(f"\nLOADING PLAYER 1: {args.model_path}")
    if agent_type == 'sac':
        agent1 = SAC(obs_dim, action_dim, device=device, hidden_sizes=args.hidden_sizes)
    elif agent_type == 'td3':
        agent1 = TD3Agent(obs_dim, action_dim)
    try:
        agent1.load(args.model_path)
    except Exception as e:
        print(f"Failed to load Player 1: {e}")
        return

    opponent_model = None
    opponent_agent = None

    if args.opponent_model:
        print(f"LOADING PLAYER 2: {args.opponent_model}")
        if opponent_type == 'sac':
            opponent_model = SAC(obs_dim, action_dim, device=device, hidden_sizes=args.hidden_sizes)
        elif opponent_type == 'td3':
            opponent_model = TD3Agent(obs_dim, action_dim)
        try:
            opponent_model.load(args.opponent_model)
        except Exception as e:
            print(f"Failed to load Player 2: {e}")
            return
    else:
        weak = (args.opponent == 'weak')
        print(f"LOADING PLAYER 2: Built-in {'Weak' if weak else 'Strong'} AI")
        opponent_agent = h_env.BasicOpponent(weak=weak)

    evaluate(env, agent1, num_episodes=args.num_episodes, render=args.render,
             max_steps=args.max_steps, opponent_agent=opponent_agent,
             opponent_model=opponent_model)
    env.close()


if __name__ == '__main__':
    main()