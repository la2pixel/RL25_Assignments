#!/usr/bin/env python3
"""
Evaluate a trained SAC agent on the hockey environment.
Can evaluate against built-in opponents (Weak/Strong) OR another trained agent.

Usage:
    # 1. Evaluate against internal Weak opponent
    python evaluate_hockey_sac.py --model_path checkpoints/sac_hockey_best.pth --opponent weak

    # 2. Evaluate against internal Strong opponent
    python evaluate_hockey_sac.py --model_path checkpoints/sac_hockey_best.pth --opponent strong

    # 3. Compete two trained agents against each other
    python evaluate_hockey_sac.py --model_path checkpoints/agent_v1.pth --opponent_model checkpoints/agent_v2.pth
"""

import argparse
import platform
import numpy as np
import torch
import gymnasium as gym
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hockey_sac import SAC

def get_state(state):
    """Transform state to proper format"""
    if isinstance(state, np.ndarray):
        state = np.ascontiguousarray(state, dtype=np.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        return state
    return np.array(state, dtype=np.float32)

def evaluate(env, agent1, num_episodes=10, render=False, max_steps=500, 
             seeds=None, opponent_agent=None, opponent_model=None):
    """
    Evaluate a trained SAC agent.
    
    Parameters
    ----------
    env : gym.Env
        Hockey environment
    agent1 : SAC
        Primary agent (Player 1, left side)
    opponent_agent : object (BasicOpponent)
        Built-in opponent (used if opponent_model is None)
    opponent_model : SAC
        Another trained SAC agent (Player 2, right side)
    """
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    draws = 0
    last_info = {}

    if seeds is None:
        # generate random python integer seeds for each episode
        seeds = np.random.randint(0, 10000, size=num_episodes)
        # convert to list of int
        seeds = [int(s) for s in seeds]
    
    # Determine Opponent Name for printing
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
        done = False
        trunc = False
        
        while step < max_steps:
            # 1. Get Player 1 Action (Agent)
            # Deterministic=True is crucial for fair evaluation
            a1 = agent1.select_action(obs, deterministic=True)
            
            # 2. Get Player 2 Action (Opponent)
            # We need the observation from Player 2's perspective
            obs_p2 = env.obs_agent_two()
            
            if opponent_model is not None:
                # Case A: Opponent is another Neural Network
                # Note: We must flip the observation logic if the model expects P1 perspective
                # But obs_agent_two() already gives the rotated view, so we just feed it.
                obs_p2_state = get_state(obs_p2)
                a2 = opponent_model.select_action(obs_p2_state, deterministic=True)
            
            elif opponent_agent is not None:
                # Case B: Opponent is Built-in Heuristic
                a2 = opponent_agent.act(obs_p2)
                
            else:
                # Case C: Should not happen in this setup, but fallback
                a2 = np.zeros(4) 

            # 3. Step Environment
            # Action passed to step() is [left_player_action, right_player_action]
            full_action = np.hstack([a1, a2])
            obs, reward, done, trunc, info = env.step(full_action)

            last_info = info
            obs = get_state(obs)
            episode_reward += reward
            step += 1
            
            if render:
                env.render()
            
            if done or trunc:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        # Determine outcome
        # info['winner']: 1 = Left(P1), -1 = Right(P2), 0 = Draw
        winner = last_info.get('winner', 0)
        
        if winner == 1:
            outcome = "WIN"
            wins += 1
        elif winner == -1:
            outcome = "LOSS"
            losses += 1
        else:
            outcome = "DRAW"
            draws += 1

        print(f"{i+1:3d} | {episode_reward:8.2f} | {step:5d} | {outcome}")
    
    # Compute statistics
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'win_rate': wins / num_episodes * 100,
        'wins': wins, 'losses': losses, 'draws': draws
    }
    
    print("-" * 80)
    print(f"Results vs {opp_name}:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Win Rate:    {stats['win_rate']:.1f}%")
    print(f"  Record:      {wins}W - {losses}L - {draws}D")
    print("-" * 80)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC Agent vs AI or another SAC Agent')
    
    # Player 1 (Our Hero)
    parser.add_argument('--model_path', type=str, required=True, help='Path to Player 1 model (Left side)')
    
    # Player 2 (The Villain)
    parser.add_argument('--opponent_model', type=str, default=None, 
                        help='Path to Player 2 model (Right side). If set, ignores --opponent arg.')
    parser.add_argument('--opponent', type=str, default='weak', choices=['weak', 'strong'],
                        help='Built-in opponent difficulty (ignored if --opponent_model is set)')
    
    # Config
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 512],
                        help='Hidden sizes used for training (must match saved model)')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # 1. Import Environment
    try:
        import hockey.hockey_env as h_env
    except ImportError:
        print("Could not import hockey package.")
        return

    # 2. Setup Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 3. Create Environment
    # We use the raw HockeyEnv to manually control both sides
    env = h_env.HockeyEnv()
    
    # 4. Load Player 1
    print(f"\nLOADING PLAYER 1 (Left): {args.model_path}")
    obs_dim = env.observation_space.shape[0]
    # In HockeyEnv, action_space is 8 (4 per player). We need an agent that outputs 4.
    action_dim = env.action_space.shape[0] // 2 
    
    agent1 = SAC(obs_dim, action_dim, device=device, hidden_sizes=args.hidden_sizes)
    try:
        agent1.load(args.model_path)
    except Exception as e:
        print(f"Failed to load Player 1: {e}")
        return

    # 5. Load Player 2 (Either AI or Model)
    opponent_model = None
    opponent_agent = None
    
    if args.opponent_model:
        # Load second neural network
        print(f"LOADING PLAYER 2 (Right): {args.opponent_model}")
        opponent_model = SAC(obs_dim, action_dim, device=device, hidden_sizes=args.hidden_sizes)
        try:
            opponent_model.load(args.opponent_model)
        except Exception as e:
            print(f"Failed to load Player 2: {e}")
            return
    else:
        # Load built-in AI
        weak = (args.opponent == 'weak')
        print(f"LOADING PLAYER 2 (Right): Built-in {'Weak' if weak else 'Strong'} AI")
        opponent_agent = h_env.BasicOpponent(weak=weak)

    # 6. Run Evaluation
    evaluate(
        env, 
        agent1, 
        num_episodes=args.num_episodes,
        render=args.render,
        max_steps=args.max_steps,
        opponent_agent=opponent_agent,
        opponent_model=opponent_model
    )
    
    env.close()

if __name__ == '__main__':
    main()