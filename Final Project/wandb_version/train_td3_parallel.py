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
# Opponent resolution (model_dir + weak/strong/.pth names)
# ============================================================

def _resolve_opponents(names, model_dir):
    """Resolve opponent list: weak/strong as-is; .pth names -> (path, algo) under model_dir. Returns list of (type, path, algo)."""
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
        elif s.endswith(".pth") and model_dir:
            path = os.path.join(model_dir, s)
            algo = "td3" if "td3" in s.lower() else "sac"
            result.append(("model", path, algo))
        elif s.endswith(".pth"):
            result.append(("model", os.path.abspath(s), "td3" if "td3" in s.lower() else "sac"))
        else:
            raise ValueError(
                f"Unknown opponent name {name!r}. Valid: 'weak', 'strong', or a .pth filename (e.g. td3_default_r1.pth)."
            )
    return result


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
                op_agent = TD3Agent(state_dim=obs_dim, action_dim=action_dim)
                try:
                    op_agent.load(opponent_model_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load opponent from {opponent_model_path!r}. Fix path or checkpoint; no silent fallback to weak."
                    ) from e
                class AgentOpponent:
                    def __init__(self, agent): self.agent = agent
                    def act(self, obs): return self.agent.select_action(obs, deterministic=True)
                opponent = AgentOpponent(op_agent)
            else:
                op_agent = SAC(obs_dim, action_dim, device="cpu", hidden_sizes=(512, 512))
                try:
                    op_agent.load(opponent_model_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load opponent from {opponent_model_path!r}. Fix path or checkpoint; no silent fallback to weak."
                    ) from e
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

    model_dir = getattr(args, 'model_dir', None) or args.save_dir
    training_opponent_triples = _resolve_opponents(builtin_opponents, model_dir)
    for p, a in zip(opponent_models, opponent_algos):
        training_opponent_triples.append(("model", p, a if a else "sac"))
    num_training_opponents = len(training_opponent_triples)
    reward_str = args.reward_mode.capitalize()
    print(f"Training on {device} | Mode: TD3 | Reward: {reward_str} | Opponents: {num_training_opponents} (builtin + pool)")

    # Build per-env opponent assignments
    opponent_types = []
    opponent_paths = []
    opponent_algos_per_env = []

    if training_opponent_triples:
        for i in range(args.num_envs):
            t, p, a = training_opponent_triples[i % num_training_opponents]
            opponent_types.append(t if t in ("weak", "strong") else "model")
            opponent_paths.append(p)
            opponent_algos_per_env.append(a)
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

    # Eval envs: from evaluation_opponents only (unseen), resolved with model_dir
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

    dummy_obs = envs.single_observation_space.sample()
    obs_dim = dummy_obs.shape[0]
    action_dim = envs.single_action_space.shape[0] // 2

    hidden = getattr(args, 'hidden_size', 512)
    improvement = getattr(args, 'improvement', False)
    dropout = getattr(args, 'dropout', 0.1) if improvement else 0.0
    weight_decay = getattr(args, 'weight_decay', 1e-5) if improvement else 0.0
    schedule_total_updates = int((args.total_timesteps - args.learning_starts) * args.update_ratio) if improvement else None
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
        hidden_sizes=[hidden, hidden],
        dropout=dropout,
        weight_decay=weight_decay,
        schedule_total_updates=schedule_total_updates,
        improvement=improvement,
    )
    device = agent.device

    # Round-based: save as [algo]-[reward_mode]-r[round].pth so downloaded artifacts have descriptive names
    slot = getattr(args, 'slot', None)
    round_num = getattr(args, 'round', None)
    if slot and round_num is not None:
        best_name = f"{slot}-r{round_num}.pth"
        final_name = f"{slot}-r{round_num}-final.pth"
    else:
        best_name = "td3_hockey_best.pth"
        final_name = "td3_hockey_final.pth"

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
    best_eval_metric = -float('inf')  # best = highest eval reward
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
            n_eval = len(eval_envs)
            if n_eval == 0:
                avg_loss_rate, avg_reward = 1.0, 0.0
                eval_loss_rates = []
            else:
                total_eps = 100
                per_env = max(1, total_eps // n_eval)
                remainder = total_eps - per_env * n_eval
                rewards, winrates = [], []
                for j, (label, ev) in enumerate(eval_envs):
                    eps = per_env + (1 if j < remainder else 0)
                    r_m, w_m = evaluate(agent, ev, eps)
                    rewards.append(r_m)
                    winrates.append(w_m)
                avg_reward = np.mean(rewards)
                avg_loss_rate = 1.0 - np.mean(winrates)
                eval_loss_rates = [1.0 - w for w in winrates]

            if args.wandb and wandb:
                wandb.log({
                    "eval/loss_rate": avg_loss_rate,
                    "eval/mean_reward": avg_reward,
                    "global_step": total_steps
                })
            labels = [lab for lab, _ in eval_envs]
            lr_str = " ".join(f"{lab}:{lr:.1%}" for lab, lr in zip(labels, eval_loss_rates)) if eval_loss_rates else ""
            print(f"  [EVAL @ {total_steps}] Reward: {avg_reward:.2f} | LossRate: {avg_loss_rate:.1%} | {lr_str}")
            if n_eval > 0 and avg_reward > best_eval_metric:
                best_eval_metric = avg_reward
                agent.save(os.path.join(args.save_dir, best_name))
                print(f"  >>> NEW BEST MODEL! <<<")

    envs.close()
    for _, ev in eval_envs:
        ev.close()
    best_path = os.path.join(args.save_dir, best_name)
    agent.save(os.path.join(args.save_dir, final_name))
    if not os.path.isfile(best_path):
        best_path = os.path.join(args.save_dir, final_name)

    # Round-based: upload best model for this round (one per spec per round) and mark finished
    if getattr(args, 'slot', None) and pool is not None and args.wandb and wandb:
        slot = args.slot
        entity = getattr(args, 'entity', None)
        project = getattr(args, 'wandb_project', 'hockey-rounds')
        pool.upload_best(entity, project, slot, best_path)
        pool.mark_slot_finished(entity, project, slot)
        print(f"[Round] {slot} best uploaded and marked finished.")


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
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability after hidden layers (0 = off)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 weight decay for Adam (0 = off)')
    parser.add_argument('--improvement', action='store_true', help='TD3 improvement bundle: dropout=0.1, weight_decay=1e-5, linear decay policy_noise->0.05, dropout->0, weight_decay->0')
    parser.add_argument('--policy_noise_end', type=float, default=None, help='End value for linear decay of policy_noise (optional)')
    parser.add_argument('--dropout_end', type=float, default=None, help='End value for linear decay of dropout (optional)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help='End value for linear decay of weight_decay (optional)')

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
    parser.add_argument('--builtin_opponents', type=str, default='weak,strong', help='Training opponents: weak, strong, and/or .pth filenames (under model_dir).')
    parser.add_argument('--model_dir', type=str, default=None, help='Root dir for .pth names (default: save_dir).')
    parser.add_argument('--evaluation_opponents', type=str, nargs='*', default=None, help='Eval-only opponents: weak, strong, and/or .pth filenames (under model_dir).')
    parser.add_argument('--wandb_project', type=str, default='hockey-rounds', help='wandb project for round-based runs')
    parser.add_argument('--entity', type=str, default=None, help='wandb entity for round-based runs')

    # Experiment flags
    parser.add_argument('--reward_mode', type=str, default='default',
                        choices=['default', 'attack', 'defense', 'proven'],
                        help='Reward shaping: default (raw), attack (offensive), defense (defensive), proven (reference)')

    args = parser.parse_args()
    train_parallel(args)
