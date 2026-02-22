"""
Unified training script: one file for TD3 and SAC.
Round-based: use --algo, --reward_mode, --round, --entity. Pool keys come from the coordinator (round trigger).
Pool key (identifier for this run in the pool) = algo-reward_mode, e.g. td3-default.
Opponents = builtin_opponents + pool (other improving agents).
"""

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
from hockey_sac import SAC, ColoredNoiseProcess

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
# Reward Wrappers (shared)
# ============================================================

class AttackRewardWrapper(gym.Wrapper):
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
        win_bonus = 10 if (done or trunc) and winner == 1 else (-10 if (done or trunc) and winner == -1 else 0)
        closeness_bonus = 3 * closeness if not self.touched else 0.5 * closeness
        shaped = reward + closeness_bonus + 0.1 * touch + win_bonus
        return obs, shaped, done, trunc, info


class DefenseRewardWrapper(gym.Wrapper):
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
        puck_x = info.get('puck_x', obs[12] if len(obs) > 12 else 0)
        if touch > 0:
            self.touched = True
        win_bonus = 10 if (done or trunc) and winner == 1 else (-15 if (done or trunc) and winner == -1 else 0)
        zone_bonus = 0.3 if puck_x > 0 else 0
        intercept_bonus = 0.5 if (touch > 0 and puck_x < 0) else 0
        shaped = reward + 2 * closeness + win_bonus + zone_bonus + intercept_bonus
        return obs, shaped, done, trunc, info


class ProvenRewardWrapper(gym.Wrapper):
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
        shaped = reward + 5 * closeness - (1 - self.touched) * 0.1 + self.touched * self.first_time_touch * 0.1 * self.step_count
        self.first_time_touch = 1 - self.touched
        return obs, shaped, done, trunc, info


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
                op_agent = TD3Agent(
                state_dim=obs_dim, action_dim=action_dim,
                hidden_sizes=[getattr(args, 'hidden_size', 512), getattr(args, 'hidden_size', 512)],
                dropout=getattr(args, 'dropout', 0.0),
                weight_decay=getattr(args, 'weight_decay', 0.0),
            )
            else:
                op_agent = SAC(obs_dim=obs_dim, action_dim=action_dim, device="cpu", hidden_sizes=(512, 512))
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

        original_step = raw_env.step
        def step_with_opponent(action):
            obs_op = raw_env.obs_agent_two()
            op_action = opponent.act(obs_op)
            return original_step(np.hstack([action, op_action]))
        raw_env.step = step_with_opponent

        if reward_mode == 'attack':
            return AttackRewardWrapper(raw_env)
        if reward_mode == 'defense':
            return DefenseRewardWrapper(raw_env)
        if reward_mode == 'proven':
            return ProvenRewardWrapper(raw_env)
        return raw_env
    return _thunk


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
            if info.get("winner") == 1:
                wins += 1
    return total_reward / num_episodes, wins / num_episodes


def train_parallel(args):
    start_time = time.time()
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Round-based: round + entity => pool_keys from coordinator trigger; pool_key = algo-reward_mode
    round_based = bool(getattr(args, 'round', None) and getattr(args, 'entity', None))
    if round_based:
        if pool is None:
            raise ValueError("wandb_pool required for round-based training.")
        args.wandb = True
        pool_key = f"{args.algo}-{args.reward_mode}"
        trigger = pool.read_round_trigger(args.entity, getattr(args, 'wandb_project', 'hockey-rounds'))
        pool_keys = trigger.get("pool_keys", [])
        if pool_key not in pool_keys:
            print(f"[ERROR] (algo, reward_mode)=({args.algo}, {args.reward_mode}) -> pool_key '{pool_key}' is not in coordinator pool_keys {pool_keys}. Conflict.", file=sys.stderr)
            sys.exit(1)
        round_one = getattr(args, 'round', 0) == 1
        if round_one:
            # Round 1: no pool opponents so all agents get the same fair start (builtin only), regardless of run order.
            pool_entries = []
            pool_key_list = []
            opponent_models = []
            opponent_algos = []
            print(f"[Pool key] {pool_key} | Round: {args.round} | Reward: {args.reward_mode}")
            print(f"[Pool] Round 1: no pool opponents (fair start for all agents). Training vs builtin only.")
        else:
            pool_entries = pool.download_pool(args.entity, getattr(args, 'wandb_project', 'hockey-rounds'), pool_keys, cache_dir=None)
            pool_key_list = [k for k, _, _ in pool_entries]
            opponent_models = [p for _, p, _ in pool_entries]
            opponent_algos = [a for _, _, a in pool_entries]
            print(f"[Pool key] {pool_key} | Round: {args.round} | Reward: {args.reward_mode}")
            print(f"[Pool] Previous best for this key: {'yes' if pool_key in pool_key_list else 'no'}. Pool has: {pool_key_list or 'none'}")
            if pool_key not in pool_key_list:
                print(f"[Pool] Training from scratch vs builtin ({args.builtin_opponents})" + (f" + pool ({pool_key_list})" if pool_key_list else "."))
            else:
                others = [k for k in pool_key_list if k != pool_key]
                print(f"[Pool] Training vs builtin + same key (prev best)" + (f" + pool ({others})" if others else "."))
        _b = getattr(args, 'builtin_opponents', 'weak,strong')
        if isinstance(_b, str):
            _b = [s.strip() for s in _b.split(",") if s.strip()]
        args.builtin_opponents = _b or ['weak', 'strong']
    else:
        pool_key = None
        pool_key_list = []
        opponent_models = args.opponent_models if args.opponent_models else []
        opponent_algos = ['sac'] * len(opponent_models)
        _b = getattr(args, 'builtin_opponents', 'weak,strong')
        if isinstance(_b, str):
            _b = [s.strip() for s in _b.split(",") if s.strip()]
        args.builtin_opponents = _b or ['weak', 'strong']

    builtin_opponents = getattr(args, 'builtin_opponents', ['weak', 'strong'])
    if isinstance(builtin_opponents, str):
        builtin_opponents = [s.strip() for s in builtin_opponents.split(",") if s.strip()]
    if not builtin_opponents:
        builtin_opponents = ['weak', 'strong']

    num_models = len(opponent_models)
    num_builtin = len(builtin_opponents)
    opp_str = f"{num_models} pool + {num_builtin} builtin ({','.join(builtin_opponents)})" if round_based else (f"{num_models} models + weak/strong" if opponent_models else "mix (weak/strong)")
    print(f"Training on {device} | Algo: {args.algo.upper()} | Reward: {args.reward_mode} | Opponents: {opp_str}" + (f" (={num_models + num_builtin} total)" if round_based else ""))

    # Opponent assignment
    opponent_types = []
    opponent_paths = []
    opponent_algos_per_env = []
    if num_models > 0 or builtin_opponents:
        pool_list = list(builtin_opponents) + [f'model_{i}' for i in range(num_models)]
        for i in range(args.num_envs):
            a = pool_list[i % len(pool_list)]
            if a in builtin_opponents:
                opponent_types.append(a)
                opponent_paths.append(None)
                opponent_algos_per_env.append(None)
            else:
                idx = int(a.split('_')[1])
                opponent_types.append('model')
                opponent_paths.append(opponent_models[idx])
                opponent_algos_per_env.append(opponent_algos[idx] if idx < len(opponent_algos) else 'sac')
    else:
        nw = int(args.num_envs * 0.4)
        for i in range(args.num_envs):
            opponent_types.append('weak' if i < nw else 'strong')
            opponent_paths.append(None)
            opponent_algos_per_env.append(None)

    project = getattr(args, 'wandb_project', None) or 'hockey-rounds'
    if args.wandb and wandb:
        run_name = f"{pool_key}-r{args.round}" if round_based and pool_key else None
        tags = [f"round-{args.round}", pool_key] if round_based and pool_key else None
        config = vars(args).copy()
        if round_based and pool_key:
            config["pool_key"] = pool_key
        wandb.init(project=project, entity=getattr(args, 'entity', None), name=run_name, config=config, tags=tags)
        # Register this run so coordinator can check run state (running/crashed/finished) for crash recovery
        if round_based and pool_key and pool is not None and getattr(args, 'worker_id', None) and getattr(args, 'round', None):
            run_path = getattr(wandb.run, 'path', None) or f"{getattr(args, 'entity', '')}/{project}/{wandb.run.id}"
            pool.register_worker_run(getattr(args, 'entity', None), project, getattr(args, 'round', 0), args.worker_id, run_path, pool_key=pool_key)

    envs = AsyncVectorEnv([
        make_env(op, i, path, reward_mode=args.reward_mode, opponent_algo=algo)
        for i, (op, path, algo) in enumerate(zip(opponent_types, opponent_paths, opponent_algos_per_env))
    ])
    eval_env_weak = make_env('weak', 900, reward_mode='default')()
    eval_env_strong = make_env('strong', 901, reward_mode='default')()
    eval_env_models = [
        make_env('model', 902 + j, mp, reward_mode='default', opponent_algo=opponent_algos[j] if j < len(opponent_algos) else 'sac')()
        for j, mp in enumerate(opponent_models)
    ]

    dummy_obs = envs.single_observation_space.sample()
    obs_dim = dummy_obs.shape[0]
    action_dim = envs.single_action_space.shape[0] // 2

    if args.algo == 'td3':
        hidden = getattr(args, 'hidden_size', 512)
        dropout = getattr(args, 'dropout', 0.0)
        weight_decay = getattr(args, 'weight_decay', 0.0)
        improvement = getattr(args, 'improvement', False)
        schedule_total_updates = None
        if improvement or getattr(args, 'policy_noise_end', None) is not None or getattr(args, 'dropout_end', None) is not None or getattr(args, 'weight_decay_end', None) is not None:
            schedule_total_updates = int((args.total_timesteps - args.learning_starts) * args.update_ratio)
        agent = TD3Agent(
            state_dim=obs_dim, action_dim=action_dim,
            gamma=args.gamma, polyak=args.polyak,
            policy_lr=args.policy_lr, critic_lr=args.critic_lr,
            act_noise_std=args.act_noise_std, policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_delay=args.policy_delay,
            hidden_sizes=[hidden, hidden], dropout=dropout, weight_decay=weight_decay,
            schedule_total_updates=schedule_total_updates,
            policy_noise_end=getattr(args, 'policy_noise_end', None),
            dropout_end=getattr(args, 'dropout_end', None),
            weight_decay_end=getattr(args, 'weight_decay_end', None),
            improvement=improvement,
        )
        best_name = "td3_hockey_best.pth"
        latest_name = "td3_hockey_latest.pth"
        final_name = "td3_hockey_final.pth"
    else:
        agent = SAC(
            obs_dim=obs_dim, action_dim=action_dim, device=device,
            hidden_sizes=[getattr(args, 'hidden_size', 512), getattr(args, 'hidden_size', 512)],
            actor_lr=getattr(args, 'actor_learning_rate', 3e-4), critic_lr=getattr(args, 'critic_learning_rate', 3e-4),
            alpha_lr=getattr(args, 'alpha_learning_rate', 3e-4), tau=getattr(args, 'tau', 0.005),
            gamma=args.gamma, alpha=getattr(args, 'alpha', 0.2),
            pink_noise=getattr(args, 'pink_noise', False), noise_beta=getattr(args, 'noise_beta', 1.0),
        )
        best_name = "sac_hockey_best.pth"
        latest_name = "sac_hockey_latest.pth"
        final_name = "sac_hockey_final.pth"

    # Round-based: save as [algo]-[reward_mode]-r[round].pth so downloaded artifacts have descriptive names
    if round_based and pool_key and getattr(args, 'round', None) is not None:
        r = args.round
        best_name = f"{pool_key}-r{r}.pth"
        latest_name = f"{pool_key}-r{r}-latest.pth"
        final_name = f"{pool_key}-r{r}-final.pth"

    device = agent.device
    if args.load_model:
        try:
            agent.load(args.load_model)
            print("Agent loaded")
        except Exception:
            sys.exit(1)

    replay_buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size, device)
    env_noise_processes = None
    if args.algo == 'sac' and getattr(args, 'pink_noise', False):
        env_noise_processes = [ColoredNoiseProcess(getattr(args, 'noise_beta', 1.0), action_dim, seq_len=500) for _ in range(args.num_envs)]

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
        elif args.algo == 'td3':
            actions = agent.select_action(obs, add_noise=True)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                if env_noise_processes:
                    noise_batch = np.stack([p.sample() for p in env_noise_processes], axis=0)
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
                    if env_noise_processes:
                        env_noise_processes[i].reset()

        real_next = next_obs.copy()
        for i, t in enumerate(truncs):
            if t and "final_observation" in infos:
                real_next[i] = infos["final_observation"][i]
        for i, d in enumerate(dones):
            if d and "final_observation" in infos:
                real_next[i] = infos["final_observation"][i]
        replay_buffer.add_batch(obs, actions, rewards, real_next, finished)
        obs = next_obs
        total_steps += args.num_envs

        if total_steps >= args.learning_starts:
            for _ in range(int(args.num_envs * args.update_ratio)):
                agent.update(replay_buffer, args.batch_size)

        if total_steps - last_log_step >= args.log_freq:
            last_log_step = total_steps
            sps = int(total_steps / (time.time() - start_time))
            avg_rew = np.mean(recent_rewards) if recent_rewards else 0.0
            print(f"{total_steps:8d} | {sps:5d} | {avg_rew:10.2f} | {best_eval_metric:10.2f}")

        if total_steps % args.eval_freq < args.num_envs and total_steps >= args.learning_starts:
            if eval_env_models:
                per_b = 100 // (2 + len(eval_env_models))
                r_w, w_w = evaluate(agent, eval_env_weak, per_b)
                r_s, w_s = evaluate(agent, eval_env_strong, per_b)
                model_rewards = []
                model_winrates = []
                for j, ev in enumerate(eval_env_models):
                    eps = per_b + (1 if j < 100 - per_b * (2 + len(eval_env_models)) else 0)
                    r_m, w_m = evaluate(agent, ev, eps)
                    model_rewards.append(r_m)
                    model_winrates.append(w_m)
                avg_winrate = np.mean([w_w, w_s] + model_winrates)
                avg_reward = np.mean([r_w, r_s] + model_rewards)
            else:
                r_w, w_w = evaluate(agent, eval_env_weak, 40)
                r_s, w_s = evaluate(agent, eval_env_strong, 60)
                avg_winrate = w_w * 0.4 + w_s * 0.6
                avg_reward = r_w * 0.4 + r_s * 0.6
                model_winrates = []

            if args.wandb and wandb:
                wandb.log({"eval/overall_win_rate": avg_winrate, "eval/mean_reward": avg_reward, "global_step": total_steps})
            print(f"  [EVAL @ {total_steps}] Reward: {avg_reward:.2f} | WinRate: {avg_winrate:.1%} | W:{w_w:.1%} S:{w_s:.1%}" + (f" M:{' '.join(f'{w:.1%}' for w in model_winrates)}" if model_winrates else ""))

            if avg_reward > best_eval_metric:
                best_eval_metric = avg_reward
                agent.save(os.path.join(args.save_dir, best_name))
                print("  >>> NEW BEST MODEL! <<<")
            agent.save(os.path.join(args.save_dir, latest_name))

            # Early stopping: 100% eval win rate
            if avg_winrate >= 1.0:
                if args.wandb and wandb:
                    wandb.log({"train/early_stop": 1, "train/early_stop_reason": "eval_win_rate_100", "global_step": total_steps})
                print(f"  [Early stop] Eval win rate reached 100% at step {total_steps}. Stopping training.")
                break

    envs.close()
    eval_env_weak.close()
    eval_env_strong.close()
    for ev in eval_env_models:
        ev.close()

    best_path = os.path.join(args.save_dir, best_name)
    agent.save(os.path.join(args.save_dir, final_name))
    if not os.path.isfile(best_path):
        best_path = os.path.join(args.save_dir, final_name)

    # Round-based: round 1 = upload first best only (no 1v1). Round 2+ = 1v1 vs previous best, then upload or mark finished
    if round_based and pool_key and pool is not None and args.wandb and wandb:
        entity = getattr(args, 'entity', None)
        project = getattr(args, 'wandb_project', 'hockey-rounds')
        round_one = getattr(args, 'round', 0) == 1
        if round_one:
            pool.upload_best(entity, project, pool_key, best_path)
            pool.mark_pool_key_finished(entity, project, pool_key)
            print(f"[Round] Round 1: no previous best for {pool_key}; uploaded and marked finished (no 1v1).")
        else:
            try:
                art = wandb.Api().artifact(f"{entity}/{project}/{pool_key}-best:best")
                prev_root = art.download()
                prev_path = None
                for f in os.listdir(prev_root):
                    if f.endswith(".pth"):
                        prev_path = os.path.join(prev_root, f)
                        break
            except Exception:
                prev_path = None
            if prev_path is None:
                pool.upload_best(entity, project, pool_key, best_path)
                pool.mark_pool_key_finished(entity, project, pool_key)
                print(f"[Round] No previous best for {pool_key}; uploaded and marked finished.")
            else:
                n_1v1 = getattr(args, 'eval_1v1_games', 100)
                print(f"[1v1] Running 1v1 vs previous best (same pool key '{pool_key}'): {n_1v1} games...")
                eval_env = make_env('model', 0, prev_path, reward_mode='default', opponent_algo=args.algo)()
                _, win_rate = evaluate(agent, eval_env, num_episodes=n_1v1)
                eval_env.close()
                # Log 1v1 result so it's visible in wandb (for both improved and not improved)
                if args.wandb and wandb:
                    wandb.log({"eval/1v1_vs_previous_best_win_rate": win_rate, "eval/1v1_model_improved": win_rate > 0.5})
                print(f"[1v1] Result vs previous best: win rate {win_rate:.1%} ({n_1v1} games). Model improved: {win_rate > 0.5}.")
                if win_rate > 0.5:
                    pool.upload_best(entity, project, pool_key, best_path)
                    pool.mark_pool_key_finished(entity, project, pool_key)
                    print(f"[Round] {pool_key} beat previous best; new best uploaded and marked finished.")
                else:
                    pool.mark_pool_key_finished(entity, project, pool_key)
                    print(f"[Round] {pool_key} did not beat previous best; marked finished (no upload).")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. by another library)
    parser = argparse.ArgumentParser(description="Unified training: --algo td3|sac, --reward_mode; round-based uses --round, --entity (pool keys from coordinator).")
    parser.add_argument('--algo', type=str, default='td3', choices=['td3', 'sac'], help='Algorithm: td3 or sac')
    parser.add_argument('--reward_mode', type=str, default='default', choices=['default', 'attack', 'defense', 'proven'])
    parser.add_argument('--total_timesteps', type=int, default=2000000)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--learning_starts', type=int, default=10000)
    parser.add_argument('--update_ratio', type=float, default=0.25)
    parser.add_argument('--log_freq', type=int, default=5000)
    parser.add_argument('--eval_freq', type=int, default=20000)
    parser.add_argument('--device', type=str, default=None)
    # TD3
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--act_noise_std', type=float, default=0.1)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--policy_delay', type=int, default=2)
    # SAC
    parser.add_argument('--actor_learning_rate', type=float, default=3e-4)
    parser.add_argument('--critic_learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha_learning_rate', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0, help='TD3: dropout after hidden layers (0 = off)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='TD3: L2 weight decay for Adam (0 = off)')
    parser.add_argument('--improvement', action='store_true', help='TD3: improvement bundle (dropout, weight_decay, linear decay)')
    parser.add_argument('--policy_noise_end', type=float, default=None)
    parser.add_argument('--dropout_end', type=float, default=None)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    parser.add_argument('--pink_noise', action='store_true')
    parser.add_argument('--noise_beta', type=float, default=1.0)
    # Opponents (standalone mode only: when not using --round/--entity)
    parser.add_argument('--opponent_models', type=str, nargs='+', default=None, help='Paths to opponent checkpoints (standalone mode). With round-based training, opponents come from pool + builtin_opponents.')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)
    # Round-based (pool keys come from coordinator via round trigger; this run's key = algo-reward_mode)
    parser.add_argument('--round', type=int, default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--builtin_opponents', type=str, default='weak,strong')
    parser.add_argument('--wandb_project', type=str, default='hockey-rounds')
    parser.add_argument('--eval_1v1_games', type=int, default=100, help='Number of games for 1v1 vs previous best (round 2+). Default: 100')
    # Config-based: load hyperparams and algo/reward_mode from YAML (used by worker)
    parser.add_argument('--config', type=str, default=None, help='Path to training config YAML (with --pool_key loads full args for that spec)')
    parser.add_argument('--pool_key', type=str, default=None, help='Pool key (e.g. td3-default); with --config loads spec from config')
    parser.add_argument('--worker_id', type=str, default=None, help='Worker id (set by worker.py) for coordinator to check wandb run status')

    args = parser.parse_args()
    if getattr(args, 'config', None) and getattr(args, 'pool_key', None):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config_loader as cl
        args = cl.apply_config_to_args(args, args.config, args.pool_key)
    train_parallel(args)
