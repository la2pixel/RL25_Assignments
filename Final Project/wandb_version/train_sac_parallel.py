import gymnasium as gym
import numpy as np
import torch
import time
import os
import argparse
import sys
from collections import deque
from gymnasium.vector import AsyncVectorEnv
import hockey.hockey_env as h_env

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hockey_sac import SAC, ColoredNoiseProcess
from hockey_replay_buffer_parallel import ReplayBuffer

try:
    import wandb
except ImportError:
    wandb = None


# Reward Wrapper

class AttackRewardWrapper(gym.Wrapper):
    """
    Offensive reward shaping.
    - Amplified win/loss signal
    - Closeness bonus (fades after first touch)
    - Small touch bonus
    """
    def __init__(self, env):
        super().__init__(env)
        self.touched = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.touched = False
        return obs, info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)

        closeness = info.get('reward_closeness_to_puck', 0)
        touch = info.get('reward_touch_puck', 0)
        winner = info.get('winner', 0)

        if touch > 0:
            self.touched = True

        win_bonus = 0
        if done or trunc:
            if winner == 1:
                win_bonus = 10
            elif winner == -1:
                win_bonus = -10

        closeness_bonus = 3 * closeness if not self.touched else 0.5 * closeness
        touch_bonus = 0.1 * touch

        shaped = reward + closeness_bonus + touch_bonus + win_bonus
        return obs, shaped, done, trunc, info


class DefenseRewardWrapper(gym.Wrapper):
    """
    Defensive reward shaping.
    - Asymmetric: losing penalized more than winning rewarded
    - Zone bonus for keeping puck in opponent half
    - Interception bonus for touching puck in own half
    - Small closeness bonus to guide early learning
    
    Obs layout: obs[12]=puck_x, obs[14]=puck_vx
    """
    def __init__(self, env):
        super().__init__(env)
        self.touched = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.touched = False
        return obs, info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)

        closeness = info.get('reward_closeness_to_puck', 0)
        touch = info.get('reward_touch_puck', 0)
        winner = info.get('winner', 0)

        if touch > 0:
            self.touched = True

        # Asymmetric win/loss — losing hurts more than winning helps
        win_bonus = 0
        if done or trunc:
            if winner == 1:
                win_bonus = 5
            elif winner == -1:
                win_bonus = -15

        # Small closeness bonus
        closeness_bonus = 2 * closeness if not self.touched else 0.3 * closeness

        # Zone bonus: reward when puck is in opponent half (positive x)
        puck_x = obs[12] if len(obs) > 12 else 0
        zone_bonus = 0.2 * max(0, puck_x)

        # Interception: touching puck when it's in our half
        intercept_bonus = 0.5 if (touch > 0 and puck_x < 0) else 0

        shaped = reward + closeness_bonus + win_bonus + zone_bonus + intercept_bonus
        return obs, shaped, done, trunc, info



def make_env(opponent_type, rank=0, opponent_model_path=None, reward_mode='default'):
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

        if reward_mode == 'attack':
            return AttackRewardWrapper(raw_env)
        elif reward_mode == 'defense':
            return DefenseRewardWrapper(raw_env)
        return raw_env
    return _thunk

# Evaluation
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


# Training Loop

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

    mode_str = "Pink Noise" if args.pink_noise else "Vanilla"
    reward_str = args.reward_mode.capitalize()
    opponent_models = args.opponent_models if args.opponent_models else []
    opp_str = f"{len(opponent_models)} models + weak/strong" if opponent_models else args.opponent
    print(f"Training on {device} | Mode: {mode_str} | Reward: {reward_str} | Opponents: {opp_str}")

    # Build per-env opponent assignments
    num_models = len(opponent_models)
    opponent_types = []
    opponent_paths = []

    if num_models > 0:
        pool = ['weak', 'strong'] + [f'model_{i}' for i in range(num_models)]
        for i in range(args.num_envs):
            assignment = pool[i % len(pool)]
            if assignment == 'weak':
                opponent_types.append('weak')
                opponent_paths.append(None)
            elif assignment == 'strong':
                opponent_types.append('strong')
                opponent_paths.append(None)
            else:
                model_idx = int(assignment.split('_')[1])
                opponent_types.append('model')
                opponent_paths.append(opponent_models[model_idx])
    elif args.opponent == 'mix':
        num_weak = int(args.num_envs * 0.4)
        for i in range(args.num_envs):
            opponent_types.append('weak' if i < num_weak else 'strong')
            opponent_paths.append(None)
    else:
        for i in range(args.num_envs):
            opponent_types.append(args.opponent)
            opponent_paths.append(None)

    if args.wandb and wandb:
        wandb.init(project="hockey-sac-parallel", config=vars(args))

    envs = AsyncVectorEnv([
        make_env(op, i, path, reward_mode=args.reward_mode)
        for i, (op, path) in enumerate(zip(opponent_types, opponent_paths))
    ])

    # Eval envs: always raw reward
    eval_env_weak = make_env('weak', 900, reward_mode='default')()
    eval_env_strong = make_env('strong', 901, reward_mode='default')()
    eval_env_models = []
    for j, mp in enumerate(opponent_models):
        eval_env_models.append(
            make_env('model', 902 + j, mp, reward_mode='default')()
        )

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
        alpha=args.alpha,
        pink_noise=args.pink_noise,
        noise_beta=args.noise_beta,
    )

    if args.load_model:
        try:
            agent.load(args.load_model)
            print("Agent loaded")
        except:
            sys.exit(1)

    replay_buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size, device)

    env_noise_processes = None
    if args.pink_noise:
        env_noise_processes = [
            ColoredNoiseProcess(args.noise_beta, action_dim, seq_len=500)
            for _ in range(args.num_envs)
        ]

    obs, _ = envs.reset()
    total_steps = 0
    last_log_step = 0
    best_eval_metric = -float('inf')
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

                if args.pink_noise and env_noise_processes is not None:
                    # Collect one pink noise sample per env
                    noise_batch = np.stack(
                        [proc.sample() for proc in env_noise_processes], axis=0
                    )
                    noise_t = torch.as_tensor(noise_batch, dtype=torch.float32, device=device)
                    act_t, _ = agent.actor.sample(obs_t, external_noise=noise_t)
                else:
                    act_t, _ = agent.actor.sample(obs_t)

                actions = act_t.cpu().numpy()

        next_obs, rewards, dones, truncs, infos = envs.step(actions)
        current_ep_rewards += rewards
        finished = np.logical_or(dones, truncs)

        if finished.any():
            for i, f in enumerate(finished):
                if f:
                    recent_rewards.append(current_ep_rewards[i])
                    if args.wandb and wandb:
                        wandb.log({"train/episode_reward": current_ep_rewards[i], "global_step": total_steps})
                    current_ep_rewards[i] = 0.0
                    # Reset noise process for this env on episode boundary
                    if env_noise_processes is not None:
                        env_noise_processes[i].reset()

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
            num_model_envs = len(eval_env_models)

            if num_model_envs > 0:
                total_eval = 100
                per_builtin = total_eval // (2 + num_model_envs)
                leftover = total_eval - per_builtin * (2 + num_model_envs)

                r_w, w_w = evaluate(agent, eval_env_weak, per_builtin)
                r_s, w_s = evaluate(agent, eval_env_strong, per_builtin)

                model_rewards = []
                model_winrates = []
                for j, ev in enumerate(eval_env_models):
                    eps = per_builtin + (1 if j < leftover else 0)
                    r_m, w_m = evaluate(agent, ev, eps)
                    model_rewards.append(r_m)
                    model_winrates.append(w_m)
                    if args.wandb and wandb:
                        wandb.log({
                            f"eval/win_rate_model_{j}": w_m,
                            f"eval/reward_model_{j}": r_m,
                        }, commit=False)

                all_rewards = [r_w, r_s] + model_rewards
                all_winrates = [w_w, w_s] + model_winrates
                avg_winrate = np.mean(all_winrates)
                avg_reward = np.mean(all_rewards)

                if args.wandb and wandb:
                    wandb.log({
                        "eval/win_rate_weak": w_w, "eval/win_rate_strong": w_s,
                        "eval/reward_weak": r_w, "eval/reward_strong": r_s,
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

            print(f"  [EVAL @ {total_steps}] Reward: {avg_reward:.2f} | WinRate: {avg_winrate:.1%} | W:{w_w:.1%} S:{w_s:.1%}" +
                  (f" M:{' '.join(f'{w:.1%}' for w in model_winrates)}" if num_model_envs > 0 else ""))

            if avg_reward > best_eval_metric:
                best_eval_metric = avg_reward
                agent.save(os.path.join(args.save_dir, "sac_hockey_best.pth"))
                print(f"  >>> NEW BEST MODEL! <<<")

            agent.save(os.path.join(args.save_dir, "sac_hockey_latest.pth"))

    envs.close()
    eval_env_weak.close()
    eval_env_strong.close()
    for ev in eval_env_models:
        ev.close()
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

    # Hyperparameters
    parser.add_argument('--alpha_learning_rate', type=float, default=0.0003)
    parser.add_argument('--actor_learning_rate', type=float, default=0.0003)
    parser.add_argument('--critic_learning_rate', type=float, default=0.0003)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=512)

    # Opponents
    parser.add_argument('--opponent', type=str, default='mix')
    parser.add_argument('--opponent_models', type=str, nargs='+', default=None,
                        help='Paths to opponent model checkpoints (can pass multiple)')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)

    # Experiment flags
    parser.add_argument('--pink_noise', action='store_true',
                        help='Use pink noise exploration (Eberhard et al. 2023)')
    parser.add_argument('--noise_beta', type=float, default=1.0,
                        help='Noise color exponent (1.0=pink, 0=white, 2=red)')
    parser.add_argument('--reward_mode', type=str, default='default',
                        choices=['default', 'attack', 'defense'],
                        help='Reward shaping: default (raw), attack (offensive), defense (defensive)')

    args = parser.parse_args()
    train_parallel(args)