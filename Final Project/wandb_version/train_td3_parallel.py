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
from hockey_td3 import TD3Agent
from hockey_replay_buffer_parallel import ReplayBuffer
from hockey_sac import SAC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import wandb_pool as pool
except ImportError:
    pool = None

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# Reward Wrapper
# ============================================================

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


class ProvenRewardWrapper(gym.Wrapper):
    """
    Proven reward shaping from reference implementation.
    - Strong closeness signal (5x) drives agent toward puck
    - Idle penalty when not touching (-0.1 per step)
    - One-time touch bonus proportional to step (rewards fast engagement)
    """
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
        obs, reward, done, trunc, info = self.env.step(action)
        self.step_count += 1

        closeness = info.get('reward_closeness_to_puck', 0)
        touch = info.get('reward_touch_puck', 0)

        self.touched = max(self.touched, touch)

        shaped = (
            reward
            + 5 * closeness
            - (1 - self.touched) * 0.1
            + self.touched * self.first_time_touch * 0.1 * self.step_count
        )

        self.first_time_touch = 1 - self.touched

        return obs, shaped, done, trunc, info


# ============================================================
# Env Factory
# ============================================================

def make_env(opponent_type, rank=0, opponent_model_path=None, reward_mode='default', opponent_algo=None):
    def _thunk():
        raw_env = h_env.HockeyEnv()
        if opponent_type == 'weak':
            opponent = h_env.BasicOpponent(weak=True)
        elif opponent_type == 'strong':
            opponent = h_env.BasicOpponent(weak=False)
        elif opponent_type == 'model' and opponent_model_path:
            obs_dim = raw_env.observation_space.shape[0]
            action_dim = raw_env.action_space.shape[0] // 2
            if opponent_algo == 'td3':
                op_agent = TD3Agent(state_dim=obs_dim, action_dim=action_dim, hidden_sizes=(512, 512))
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
        elif reward_mode == 'proven':
            return ProvenRewardWrapper(raw_env)
        return raw_env
    return _thunk


# ============================================================
# Evaluation
# ============================================================

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


# ============================================================
# Training Loop
# ============================================================

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

    # Round-based mode: download pool from wandb and set opponent_models + algos
    opponent_algos = []
    if getattr(args, 'slot', None):
        if not getattr(args, 'round', None) or not getattr(args, 'entity', None) or not getattr(args, 'specs', None):
            raise ValueError("When --slot is set, --round, --entity and --specs are required.")
        if pool is None:
            raise ValueError("wandb_pool is required for round-based training.")
        args.wandb = True
        specs = [s.strip() for s in args.specs.split(",") if s.strip()]
        pool_entries = pool.download_pool(args.entity, getattr(args, 'wandb_project', 'hockey-rounds'), specs, cache_dir=None)
        # pool_entries: list of (slot, path, algo) for specs that have an artifact
        pool_slots = [s for s, _, _ in pool_entries]
        opponent_models = [p for _, p, _ in pool_entries]
        opponent_algos = [a for _, _, a in pool_entries]
        args.opponent_models = opponent_models
        slot = args.slot
        round_n = getattr(args, 'round', 0)
        _b = getattr(args, 'builtin_opponents', 'weak,strong')
        if isinstance(_b, str):
            _b = [s.strip() for s in _b.split(",") if s.strip()]
        args.builtin_opponents = _b if _b else ['weak', 'strong']
        print(f"[Spec] Slot: {slot} | Round: {round_n} | Reward: {pool.get_reward_mode_for_slot(slot)}")
        print(f"[Pool] Checking for previous best for this spec... Pool has: {pool_slots if pool_slots else 'none'}")
        same_spec_in_pool = slot in pool_slots
        builtin = args.builtin_opponents
        if not same_spec_in_pool:
            print(f"[Pool] No previous best for spec '{slot}'. Training from scratch vs builtin ({builtin})" + (f" + other specs ({[s for s in pool_slots]})" if pool_slots else "."))
        else:
            others = [s for s in pool_slots if s != slot]
            print(f"[Pool] Previous best exists for this spec. Training vs builtin ({builtin}) + same-spec (prev best)" + (f" + other specs ({others})" if others else "."))
    else:
        pool_slots = []
        opponent_models = args.opponent_models if args.opponent_models else []
        opponent_algos = ['sac'] * len(opponent_models) if opponent_models else []
        _b = getattr(args, 'builtin_opponents', 'weak,strong')
        if isinstance(_b, str):
            _b = [s.strip() for s in _b.split(",") if s.strip()]
        args.builtin_opponents = _b if _b else ['weak', 'strong']

    builtin_opponents = getattr(args, 'builtin_opponents', ['weak', 'strong'])
    if isinstance(builtin_opponents, str):
        builtin_opponents = [s.strip() for s in builtin_opponents.split(",") if s.strip()]
    if not builtin_opponents:
        builtin_opponents = ['weak', 'strong']
    reward_str = args.reward_mode.capitalize()
    num_builtin = len(builtin_opponents)
    opp_str = f"{len(opponent_models)} pool + {num_builtin} builtin ({','.join(builtin_opponents)})" if getattr(args, 'slot', None) else (f"{len(opponent_models)} models + weak/strong" if opponent_models else getattr(args, 'opponent', 'mix'))
    print(f"Training on {device} | Mode: TD3 | Reward: {reward_str} | Opponents: {opp_str}" + (f" (={len(opponent_models) + num_builtin} total)" if getattr(args, 'slot', None) else ""))

    # Build per-env opponent assignments
    num_models = len(opponent_models)
    opponent_types = []
    opponent_paths = []
    opponent_algos_per_env = []

    if num_models > 0 or builtin_opponents:
        pool_list = list(builtin_opponents) + [f'model_{i}' for i in range(num_models)]
        for i in range(args.num_envs):
            assignment = pool_list[i % len(pool_list)]
            if assignment in builtin_opponents:
                opponent_types.append(assignment)
                opponent_paths.append(None)
                opponent_algos_per_env.append(None)
            else:
                model_idx = int(assignment.split('_')[1])
                opponent_types.append('model')
                opponent_paths.append(opponent_models[model_idx])
                opponent_algos_per_env.append(opponent_algos[model_idx] if model_idx < len(opponent_algos) else 'sac')
    elif getattr(args, 'opponent', 'mix') == 'mix':
        num_weak = int(args.num_envs * 0.4)
        for i in range(args.num_envs):
            opponent_types.append('weak' if i < num_weak else 'strong')
            opponent_paths.append(None)
            opponent_algos_per_env.append(None)
    else:
        for i in range(args.num_envs):
            opponent_types.append(args.opponent)
            opponent_paths.append(None)
            opponent_algos_per_env.append(None)

    project = getattr(args, 'wandb_project', None) or 'hockey-td3-parallel'
    if args.wandb and wandb:
        tags = None
        run_name = None
        if getattr(args, 'slot', None) and getattr(args, 'round', None) is not None:
            tags = [f"round-{args.round}", args.slot]
            run_name = f"{args.slot}-r{args.round}"
        wandb.init(project=project, entity=getattr(args, 'entity', None), name=run_name, config=vars(args), tags=tags)

    envs = AsyncVectorEnv([
        make_env(op, i, path, reward_mode=args.reward_mode, opponent_algo=algo)
        for i, (op, path, algo) in enumerate(zip(opponent_types, opponent_paths, opponent_algos_per_env))
    ])

    # Eval envs: always raw reward
    eval_env_weak = make_env('weak', 900, reward_mode='default')()
    eval_env_strong = make_env('strong', 901, reward_mode='default')()
    eval_env_models = []
    for j, mp in enumerate(opponent_models):
        algo = opponent_algos[j] if j < len(opponent_algos) else 'sac'
        eval_env_models.append(
            make_env('model', 902 + j, mp, reward_mode='default', opponent_algo=algo)()
        )

    dummy_obs = envs.single_observation_space.sample()
    obs_dim = dummy_obs.shape[0]
    action_dim = envs.single_action_space.shape[0] // 2

    hidden = getattr(args, 'hidden_size', 512)
    agent = TD3Agent(
        state_dim=obs_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        polyak=args.polyak,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        act_noise_std=args.act_noise_std,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        hidden_sizes=(hidden, hidden),
    )
    device = agent.device

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
    best_eval_metric = -float('inf')
    current_ep_rewards = np.zeros(args.num_envs)
    recent_rewards = deque(maxlen=100)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"{'Step':>8} | {'SPS':>5} | {'Avg Reward':>10} | {'Best Metric':>10}")

    while total_steps < args.total_timesteps:
        if total_steps < args.learning_starts and not args.load_model:
            actions = np.random.uniform(-1, 1, size=(args.num_envs, action_dim))
        else:
            actions = agent.select_action(obs, add_noise=True)

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
                agent.save(os.path.join(args.save_dir, "td3_hockey_best.pth"))
                print(f"  >>> NEW BEST MODEL! <<<")

            agent.save(os.path.join(args.save_dir, "td3_hockey_latest.pth"))

    envs.close()
    eval_env_weak.close()
    eval_env_strong.close()
    for ev in eval_env_models:
        ev.close()
    best_path = os.path.join(args.save_dir, "td3_hockey_best.pth")
    agent.save(os.path.join(args.save_dir, "td3_hockey_final.pth"))

    # Round-based: head-to-head vs previous best, then upload or mark finished
    if getattr(args, 'slot', None) and pool is not None and args.wandb and wandb:
        slot = args.slot
        entity = getattr(args, 'entity', None)
        project = getattr(args, 'wandb_project', 'hockey-rounds')
        if not os.path.isfile(best_path):
            best_path = os.path.join(args.save_dir, "td3_hockey_final.pth")
        # Check if previous best exists for this slot
        try:
            art = wandb.Api().artifact(f"{entity}/{project}/{slot}-best:best")
            prev_root = art.download()
            prev_path = None
            for f in os.listdir(prev_root):
                if f.endswith(".pth"):
                    prev_path = os.path.join(prev_root, f)
                    break
        except Exception:
            prev_path = None
        if prev_path is None:
            pool.upload_best(entity, project, slot, best_path)
            print(f"[Round] No previous best for {slot}; uploaded.")
        else:
            # Head-to-head: candidate (our best) vs previous best, 30 games
            print(f"[1v1] Running 1v1 vs previous best (same spec '{slot}'): 30 games...")
            eval_env = make_env('model', 0, prev_path, reward_mode='default', opponent_algo='td3' if slot.startswith('td3-') else 'sac')()
            _, win_rate = evaluate(agent, eval_env, num_episodes=30)
            eval_env.close()
            print(f"[1v1] Result vs previous best: win rate {win_rate:.1%} (30 games).")
            if win_rate > 0.5:
                pool.upload_best(entity, project, slot, best_path)
                print(f"[Round] {slot} beat previous best; saved new best and uploaded.")
            else:
                pool.mark_slot_finished(entity, project, slot)
                print(f"[Round] {slot} did not beat previous best; slot marked finished (no upload).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=2000000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--learning_starts', type=int, default=10000)
    parser.add_argument('--update_ratio', type=float, default=0.25)
    parser.add_argument('--log_freq', type=int, default=5000)
    parser.add_argument('--eval_freq', type=int, default=20000)
    parser.add_argument('--device', type=str, default=None)

    # Hyperparameters (TD3)
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--act_noise_std', type=float, default=0.1)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--policy_delay', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=512, help='Actor/critic hidden layer size (default 512, same as SAC)')

    # Opponents
    parser.add_argument('--opponent', type=str, default='mix')
    parser.add_argument('--opponent_models', type=str, nargs='+', default=None,
                        help='Paths to opponent model checkpoints (can pass multiple)')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)
    # Round-based training (used by worker)
    parser.add_argument('--round', type=int, default=None, help='Current round number (required when --slot is set)')
    parser.add_argument('--slot', type=str, default=None, help='Slot name, e.g. td3-default (required for round-based)')
    parser.add_argument('--specs', type=str, default=None, help='comma-separated specs for this run (e.g. td3-default,td3-attack). Opponents = N + M builtin.')
    parser.add_argument('--builtin_opponents', type=str, default='weak,strong', help='comma-separated built-in bots (e.g. weak,strong). Passed by coordinator/worker.')
    parser.add_argument('--wandb_project', type=str, default='hockey-rounds', help='wandb project for round-based runs')
    parser.add_argument('--entity', type=str, default=None, help='wandb entity for round-based runs')

    # Experiment flags
    parser.add_argument('--reward_mode', type=str, default='default',
                        choices=['default', 'attack', 'defense', 'proven'],
                        help='Reward shaping: default (raw), attack (offensive), defense (defensive), proven (reference)')

    args = parser.parse_args()
    train_parallel(args)
