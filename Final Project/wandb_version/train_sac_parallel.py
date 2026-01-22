import gymnasium as gym
import numpy as np
import torch
import time
import os
import argparse
import sys
from collections import deque, defaultdict
from gymnasium.vector import AsyncVectorEnv
import hockey.hockey_env as h_env

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hockey_sac import SAC
from hockey_replay_buffer_parallel import ReplayBuffer

try:
    import wandb
except ImportError:
    wandb = None

# Reward Wrapper
class DenseRewardHockeyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.touched = 0
        self.first_time_touch = 1
        self.step_count = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.touched = 0
        self.first_time_touch = 1
        self.step_count = 0
        return obs, info

    def step(self, action):
        self.step_count += 1
        obs, reward, done, trunc, info = self.env.step(action)
        self.touched = max(self.touched, info.get('reward_touch_puck', 0))
        dense_reward = (
            reward
            + 5 * info.get('reward_closeness_to_puck', 0)
            - (1 - self.touched) * 0.1
            + self.touched * self.first_time_touch * 0.1 * self.step_count
        )
        if self.touched: self.first_time_touch = 0
        return obs, dense_reward, done, trunc, info

# Env Factory 
def make_env(opponent_type, rank=0, opponent_model_path=None):
    def _thunk():
        raw_env = h_env.HockeyEnv()
        if opponent_type == 'weak':
            opponent = h_env.BasicOpponent(weak=True)
        elif opponent_type == 'strong':
            opponent = h_env.BasicOpponent(weak=False)
        elif opponent_type == 'model' and opponent_model_path:
            obs_dim = raw_env.observation_space.shape[0]
            action_dim = raw_env.action_space.shape[0] // 2
            op_agent = SAC(obs_dim, action_dim, device="cpu", hidden_sizes=(512, 512))
            try:
                op_agent.load(opponent_model_path)
            except Exception:
                opponent = h_env.BasicOpponent(weak=True)
            else:
                class AgentOpponent:
                    def __init__(self, agent): self.agent = agent
                    def act(self, obs): return self.agent.select_action(obs, deterministic=True)
                opponent = AgentOpponent(op_agent)
        else:
            opponent = h_env.BasicOpponent(weak=True)

        original_step_method = raw_env.step
        def step_with_opponent(action):
            obs_op = raw_env.obs_agent_two() 
            op_action = opponent.act(obs_op)
            return original_step_method(np.hstack([action, op_action]))
        raw_env.step = step_with_opponent
        return DenseRewardHockeyWrapper(raw_env)
    return _thunk

# Evaluation Function 
def evaluate(agent, env, num_episodes=5):
    total_reward = 0
    wins = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        while not (done or trunc):
            with torch.no_grad():
                action = agent.select_action(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            if "winner" in info and info["winner"] == 1:
                wins += 1
    return total_reward / num_episodes, wins / num_episodes

def train_parallel(args):
    start_time = time.time()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Training on {device}")

    opponent_types = []
    if args.opponent_model:
        pattern = ['weak', 'strong', 'model']
        for i in range(args.num_envs): opponent_types.append(pattern[i % 3])
    elif args.opponent == 'mix':
        num_weak = int(args.num_envs * 0.4)
        for i in range(args.num_envs): opponent_types.append('weak' if i < num_weak else 'strong')
    else:
        opponent_types = [args.opponent] * args.num_envs

    if args.wandb and wandb:
        wandb.init(project="hockey-sac-parallel", config=vars(args))

    envs = AsyncVectorEnv([make_env(op, i, args.opponent_model) for i, op in enumerate(opponent_types)])
    
    eval_env_weak = make_env('weak', 900)()
    eval_env_strong = make_env('strong', 901)()
    eval_env_model = make_env('model', 902, args.opponent_model)() if args.opponent_model else None

    dummy_obs = envs.single_observation_space.sample()
    obs_dim = dummy_obs.shape[0]
    action_dim = envs.single_action_space.shape[0] // 2 
    
    agent = SAC(
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        device=device,
        hidden_sizes=(args.hidden_size, args.hidden_size),
        actor_lr=args.actor_learning_rate,
        critic_lr=args.critic_learning_rate,
        alpha_lr=args.alpha_learning_rate,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha
    )

    if args.load_model:
        try:
            agent.load(args.load_model)
            print("Agent loaded")
        except:
            sys.exit(1)

    replay_buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size, device)
    
    obs, _ = envs.reset()
    total_steps = 0
    last_log_step = 0
    best_eval_metric = -float('inf') # Tracking best mean reward
    current_ep_rewards = np.zeros(args.num_envs)
    recent_rewards = deque(maxlen=100)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"{'Step':>8} | {'SPS':>5} | {'Avg Reward':>10} | {'Best Metric':>10}")

    while total_steps < args.total_timesteps:
        if total_steps < args.learning_starts and not args.load_model:
            actions = np.random.uniform(-1, 1, size=(args.num_envs, action_dim))
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                act_t, _ = agent.actor.sample(obs_t)
                actions = act_t.cpu().numpy()
        
        next_obs, rewards, dones, truncs, infos = envs.step(actions)
        current_ep_rewards += rewards
        finished = np.logical_or(dones, truncs)
        
        if finished.any():
            for i, f in enumerate(finished):
                if f:
                    recent_rewards.append(current_ep_rewards[i])
                    if args.wandb and wandb: wandb.log({"train/episode_reward": current_ep_rewards[i], "global_step": total_steps})
                    current_ep_rewards[i] = 0.0

        real_next = next_obs.copy()
        for i, t in enumerate(truncs):
            if t and "final_observation" in infos: real_next[i] = infos["final_observation"][i]
        for i, d in enumerate(dones):
            if d and "final_observation" in infos: real_next[i] = infos["final_observation"][i]
            
        replay_buffer.add_batch(obs, actions, rewards, real_next, finished)
        obs = next_obs
        total_steps += args.num_envs
        
        if total_steps >= args.learning_starts:
            updates = int(args.num_envs * args.update_ratio)
            for _ in range(updates):
                agent.update(replay_buffer, args.batch_size)

        if total_steps - last_log_step >= args.log_freq:
            last_log_step = total_steps
            sps = int(total_steps / (time.time() - start_time))
            avg_rew = np.mean(recent_rewards) if recent_rewards else 0.0
            print(f"{total_steps:8d} | {sps:5d} | {avg_rew:10.2f} | {best_eval_metric:10.2f}")

        if total_steps % args.eval_freq < args.num_envs and total_steps >= args.learning_starts:
            if args.opponent_model:
                r_w, w_w = evaluate(agent, eval_env_weak, 33)
                r_s, w_s = evaluate(agent, eval_env_strong, 33)
                r_m, w_m = evaluate(agent, eval_env_model, 34)
                
                avg_winrate = (w_w + w_s + w_m) / 3.0
                avg_reward = (r_w + r_s + r_m) / 3.0
                
                if args.wandb and wandb:
                    wandb.log({
                        "eval/win_rate_weak": w_w, "eval/win_rate_strong": w_s, "eval/win_rate_model": w_m,
                        "eval/reward_weak": r_w, "eval/reward_strong": r_s, "eval/reward_model": r_m,
                    }, commit=False)
            else:
                r_w, w_w = evaluate(agent, eval_env_weak, 40)
                r_s, w_s = evaluate(agent, eval_env_strong, 60)
                
                avg_winrate = (w_w * 0.40) + (w_s * 0.60)
                avg_reward = (r_w * 0.40) + (r_s * 0.60)

                if args.wandb and wandb:
                    wandb.log({
                        "eval/win_rate_weak": w_w, "eval/win_rate_strong": w_s,
                        "eval/reward_weak": r_w, "eval/reward_strong": r_s,
                    }, commit=False)

            if args.wandb and wandb:
                wandb.log({
                    "eval/overall_win_rate": avg_winrate, 
                    "eval/mean_reward": avg_reward,  
                    "global_step": total_steps
                })
            
            if avg_reward > best_eval_metric:
                best_eval_metric = avg_reward
                agent.save(os.path.join(args.save_dir, "sac_hockey_best.pth"))
                print(f"NEW BEST MODEL! Reward: {avg_reward:.2f}")

            agent.save(os.path.join(args.save_dir, "sac_hockey_latest.pth"))

    envs.close()
    eval_env_weak.close()
    eval_env_strong.close()
    if eval_env_model: eval_env_model.close()
    agent.save(os.path.join(args.save_dir, "sac_hockey_final.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=2000000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--learning_starts', type=int, default=10000)
    parser.add_argument('--update_ratio', type=float, default=0.25)
    parser.add_argument('--log_freq', type=int, default=5000)
    parser.add_argument('--eval_freq', type=int, default=20000)
    parser.add_argument('--device', type=str, default=None)
    
    # Sweep Params
    parser.add_argument('--alpha_learning_rate', type=float, default=0.0003)
    parser.add_argument('--actor_learning_rate', type=float, default=0.0003)
    parser.add_argument('--critic_learning_rate', type=float, default=0.0003)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=512)

    parser.add_argument('--opponent', type=str, default='mix')
    parser.add_argument('--opponent_model', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)
    
    args = parser.parse_args()
    train_parallel(args)