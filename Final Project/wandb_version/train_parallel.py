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


def _resolve_opponents(names, model_dir):
    """
    Resolve opponent list: weak/strong stay as-is; .pth names become (path, algo) under model_dir.
    Returns list of (opponent_type, path, algo): 'weak'|'strong'|'model', path or None, algo or None.
    """
    if not names:
        return []
    result = []
    for name in names:
        s = (name or "").strip()
        if not s:
            continue
        if s.lower() == "weak":
            result.append(("weak", None, None))
        elif s.lower() == "strong":
            result.append(("strong", None, None))
        elif s.endswith(".pth"):
            if os.path.sep in s or '/' in s or '\\' in s:
                path = os.path.abspath(s)
            elif model_dir:
                path = os.path.join(model_dir, s)
            else:
                path = os.path.abspath(s)
            algo = "td3" if "td3" in s.lower() else "sac"
            result.append(("model", path, algo))
        else:
            raise ValueError(
                f"Unknown opponent name {name!r}. Valid: 'weak', 'strong', or a .pth filename (e.g. td3_default_r1.pth)."
            )
    return result


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
                op_agent = TD3Agent(state_dim=obs_dim, action_dim=action_dim)
            else:
                op_agent = SAC(obs_dim=obs_dim, action_dim=action_dim, device="cpu", hidden_sizes=(512, 512))
            try:
                op_agent.load(opponent_model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load opponent from {opponent_model_path!r}. Fix path or checkpoint; no silent fallback to weak."
                ) from e
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

    model_dir = getattr(args, 'model_dir', None) or args.save_dir
    training_opponent_triples = _resolve_opponents(builtin_opponents, model_dir)
    for p, a in zip(opponent_models, opponent_algos):
        training_opponent_triples.append(("model", p, a if a else "sac"))
    num_training_opponents = len(training_opponent_triples)

    # --- Debug: print resolved training opponent triples ---
    print(f"\n{'='*60}")
    print(f"TRAINING OPPONENT POOL ({num_training_opponents} opponents)")
    print(f"{'='*60}")
    for idx, (t, p, a) in enumerate(training_opponent_triples):
        if t in ('weak', 'strong'):
            print(f"  [{idx}] {t} (builtin)")
        else:
            exists = os.path.isfile(p) if p else False
            status = 'OK' if exists else 'MISSING!'
            print(f"  [{idx}] model ({a}) -> {p} [{status}]")
    print(f"{'='*60}\n")

    opp_str = f"{num_training_opponents} training opponents (builtin + pool)" if round_based else (f"{num_training_opponents} opponents" if training_opponent_triples else "mix (weak/strong)")
    print(f"Training on {device} | Algo: {args.algo.upper()} | Reward: {args.reward_mode} | Opponents: {opp_str}")

    # Opponent assignment: cycle over (type, path, algo)
    opponent_types = []
    opponent_paths = []
    opponent_algos_per_env = []
    if training_opponent_triples:
        for i in range(args.num_envs):
            t, p, a = training_opponent_triples[i % num_training_opponents]
            opponent_types.append(t if t in ("weak", "strong") else "model")
            opponent_paths.append(p)
            opponent_algos_per_env.append(a)
    else:
        nw = int(args.num_envs * 0.4)
        for i in range(args.num_envs):
            opponent_types.append('weak' if i < nw else 'strong')
            opponent_paths.append(None)
            opponent_algos_per_env.append(None)

    # --- Debug: print per-env opponent assignment ---
    print(f"\n{'='*60}")
    print(f"ENV OPPONENT ASSIGNMENTS ({args.num_envs} parallel envs)")
    print(f"{'='*60}")
    for i in range(args.num_envs):
        t = opponent_types[i]
        p = opponent_paths[i]
        a = opponent_algos_per_env[i]
        if t in ('weak', 'strong'):
            print(f"  Env {i}: {t} (builtin)")
        else:
            exists = os.path.isfile(p) if p else False
            status = 'OK' if exists else 'MISSING!'
            print(f"  Env {i}: model ({a}) -> {os.path.basename(p) if p else 'None'} [{status}]")
    print(f"{'='*60}\n")

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
    # Evaluation opponents (unseen): from config, resolved with model_dir; not used for training
    eval_opponent_names = getattr(args, 'evaluation_opponents', None)
    if eval_opponent_names is None:
        eval_opponent_names = []
    if isinstance(eval_opponent_names, str):
        eval_opponent_names = [s.strip() for s in eval_opponent_names.split(",") if s.strip()]
    if not eval_opponent_names:
        raise ValueError(
            "evaluation_opponents must be set in config (non-empty list: weak, strong, and/or .pth filenames under model_dir)."
        )
    eval_triples = _resolve_opponents(eval_opponent_names, model_dir)
    eval_envs = []
    for i, (t, path, algo) in enumerate(eval_triples):
        if t == "weak":
            eval_envs.append((f"W{i}", make_env('weak', 900 + i, reward_mode='default')()))
        elif t == "strong":
            eval_envs.append((f"S{i}", make_env('strong', 900 + i, reward_mode='default')()))
        else:
            eval_envs.append((os.path.basename(path) if path else f"M{i}", make_env('model', 900 + i, path, reward_mode='default', opponent_algo=algo or 'sac')()))

    # --- Debug: print evaluation opponent assignments ---
    print(f"\n{'='*60}")
    print(f"EVALUATION OPPONENTS ({len(eval_envs)} envs, NOT used for training)")
    print(f"{'='*60}")
    for idx, (label, _) in enumerate(eval_envs):
        t, path, algo = eval_triples[idx]
        if t in ('weak', 'strong'):
            print(f"  Eval {idx}: {label} -> {t} (builtin)")
        else:
            exists = os.path.isfile(path) if path else False
            status = 'OK' if exists else 'MISSING!'
            print(f"  Eval {idx}: {label} -> model ({algo}) {path} [{status}]")
    print(f"{'='*60}\n")

    dummy_obs = envs.single_observation_space.sample()
    obs_dim = dummy_obs.shape[0]
    action_dim = envs.single_action_space.shape[0] // 2

    if args.algo == 'td3':
        hidden = getattr(args, 'hidden_size', 512)
        improvement = getattr(args, 'improvement', False)
        dropout = getattr(args, 'dropout', 0.1) if improvement else 0.0
        weight_decay = getattr(args, 'weight_decay', 1e-5) if improvement else 0.0
        schedule_total_updates = int((args.total_timesteps - args.learning_starts) * args.update_ratio) if improvement else None
        agent = TD3Agent(
            state_dim=obs_dim, action_dim=action_dim,
            gamma=args.gamma, polyak=args.polyak,
            policy_lr=args.policy_lr, critic_lr=args.critic_lr,
            act_noise_std=args.act_noise_std, policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_delay=args.policy_delay,
            hidden_sizes=[hidden, hidden], dropout=dropout, weight_decay=weight_decay,
            schedule_total_updates=schedule_total_updates,
            improvement=improvement,
        )
        best_name = "td3_hockey_best.pth"
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
        final_name = "sac_hockey_final.pth"

    # Round-based: save best as [algo]-[reward_mode]-r[round].pth (one best per round, uploaded to wandb)
    if round_based and pool_key and getattr(args, 'round', None) is not None:
        r = args.round
        best_name = f"{pool_key}-r{r}.pth"
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
    best_eval_metric = -float('inf')  # best = highest eval reward
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
            n_eval = len(eval_envs)
            if n_eval == 0:
                avg_loss_rate, avg_reward = 1.0, 0.0
                eval_loss_rates = []
            else:
                rewards, winrates = [], []
                for j, (label, ev) in enumerate(eval_envs):
                    r_m, w_m = evaluate(agent, ev, 50)
                    rewards.append(r_m)
                    winrates.append(w_m)
                avg_reward = np.mean(rewards)
                avg_loss_rate = 1.0 - np.mean(winrates)
                eval_loss_rates = [1.0 - w for w in winrates]

            if args.wandb and wandb:
                wandb.log({"eval/loss_rate": avg_loss_rate, "eval/mean_reward": avg_reward, "global_step": total_steps})
            labels = [lab for lab, _ in eval_envs]
            lr_str = " ".join(f"{lab}:{lr:.1%}" for lab, lr in zip(labels, eval_loss_rates)) if eval_loss_rates else ""
            print(f"  [EVAL @ {total_steps}] Reward: {avg_reward:.2f} | LossRate: {avg_loss_rate:.1%} | {lr_str}")
            if n_eval > 0 and avg_reward > best_eval_metric:
                best_eval_metric = avg_reward
                agent.save(os.path.join(args.save_dir, best_name))
                print("  >>> NEW BEST MODEL! <<<")

            # Early stopping: 0% loss rate (never lose) — unlikely in practice
            if n_eval > 0 and avg_loss_rate <= 0.0:
                if args.wandb and wandb:
                    wandb.log({"train/early_stop": 1, "train/early_stop_reason": "eval_loss_rate_0", "global_step": total_steps})
                print(f"  [Early stop] Eval loss rate reached 0% at step {total_steps}. Stopping training.")
                break

    envs.close()
    for _, ev in eval_envs:
        ev.close()

    best_path = os.path.join(args.save_dir, best_name)
    agent.save(os.path.join(args.save_dir, final_name))
    if not os.path.isfile(best_path):
        best_path = os.path.join(args.save_dir, final_name)

    # Round-based: upload best model for this round (one per spec per round) and mark finished
    if round_based and pool_key and pool is not None and args.wandb and wandb:
        entity = getattr(args, 'entity', None)
        project = getattr(args, 'wandb_project', 'hockey-rounds')
        pool.upload_best(entity, project, pool_key, best_path)
        pool.mark_pool_key_finished(entity, project, pool_key)
        print(f"[Round] Round {getattr(args, 'round', 0)}: {pool_key} best uploaded and marked finished.")


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
    parser.add_argument('--builtin_opponents', type=str, default='weak,strong', help='Training opponents: weak, strong, and/or .pth filenames (under model_dir)')
    parser.add_argument('--model_dir', type=str, default=None, help='Root dir for .pth names in builtin_opponents / evaluation_opponents (default: save_dir)')
    parser.add_argument('--evaluation_opponents', type=str, nargs='*', default=None, help='Eval-only opponents: weak, strong, and/or .pth filenames (under model_dir). Unseen during training.')
    parser.add_argument('--wandb_project', type=str, default='hockey-rounds')
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