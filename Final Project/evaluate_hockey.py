#!/usr/bin/env python3
"""
Evaluate trained SAC or TD3 agents on the hockey environment.

Usage (single 1v1, no wandb):
    python evaluate_hockey.py --model_path checkpoints/sac_hockey_best.pth --opponent weak
    python evaluate_hockey.py --model_path checkpoints/sac_hockey_best.pth --opponent strong
    python evaluate_hockey.py --model_path checkpoints/agent_v1.pth --opponent_model checkpoints/agent_v2.pth

Usage (multiple candidates/opponents, optional wandb):
    python evaluate_hockey.py --candidates_dir checkpoints/candidates --opponents_dir checkpoints/opponents --num_episodes 1000 --wandb
    python evaluate_hockey.py --candidates p1.pth p2.pth --opponents weak strong p3.pth --num_episodes 500

"""

import argparse
import io
import os
import random
import re
import sys
from collections import defaultdict
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hockey_sac import SAC
from hockey_td3 import TD3Agent


def algo_from_path(path):
    """Infer algo from path; must contain 'td3' or 'sac'."""
    p = path.lower()
    if "td3" in p:
        return "td3"
    if "sac" in p:
        return "sac"
    return None


def weight_from_name(name, default=1):
    """Parse weight from name (e.g. filename). Uses the number after attack/defensive/proven/defense; if none, default."""
    # Match number after keyword (e.g. attack12 -> 12, proven4 -> 4)
    m = re.search(r"(?:attack|defensive|proven|defense)(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return default


def models_from_dir(dirpath):
    """Return sorted list of full paths to .pth files in dirpath (one level only)."""
    if not os.path.isdir(dirpath):
        return []
    paths = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(".pth")]
    return sorted(paths)


def get_state(state):
    if isinstance(state, np.ndarray):
        state = np.ascontiguousarray(state, dtype=np.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        return state
    return np.array(state, dtype=np.float32)


def evaluate(env, agent1, num_episodes=10, render=False, max_steps=500,
             seeds=None, opponent_agent=None, opponent_model=None, progress_interval=0):
    """
    progress_interval: print progress every N episodes (0 = every episode; if 0 and num_episodes>100, auto 500).
    """
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

    if progress_interval == 0 and num_episodes > 100:
        progress_interval = min(500, max(1, num_episodes // 20))
    show_every = progress_interval if progress_interval > 0 else 1

    print(f"\nEvaluating: Agent (P1) vs {opp_name} for {num_episodes} episodes...")
    print("-" * 80)
    if show_every == 1:
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

        if (i + 1) % show_every == 0 or (i + 1) == num_episodes:
            if show_every == 1:
                print(f"{i+1:3d} | {episode_reward:8.2f} | {step:5d} | {outcome}")
            else:
                pct = (i + 1) / num_episodes * 100
                print(f"  Episodes {i+1}/{num_episodes} ({pct:.1f}%) | last reward={episode_reward:.2f} | wins={wins} draws={draws} losses={losses}")

    n = num_episodes
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'win_rate': wins / n * 100,
        'draw_rate': draws / n * 100,
        'loss_rate': losses / n * 100,
        'wins': wins, 'losses': losses, 'draws': draws
    }

    print("-" * 80)
    print(f"Results vs {opp_name}:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"  Win Rate:    {stats['win_rate']:.1f}%  Draw Rate: {stats['draw_rate']:.1f}%  Loss Rate: {stats['loss_rate']:.1f}%")
    print(f"  Record:      {wins}W - {losses}L - {draws}D")
    print("-" * 80)
    return stats


def run_single_episode(env, agent1, opponent_agent=None, opponent_model=None, max_steps=500, seed=None):
    """Run one episode; return (episode_reward, winner). winner: 1=win, -1=loss, 0=draw."""
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        obs, info = env.reset()
    obs = get_state(obs)
    episode_reward = 0.0
    step = 0
    last_info = {}
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
        if done or trunc:
            break
    winner = last_info.get("winner", 0)
    return episode_reward, winner


def load_agent(path, obs_dim, action_dim, device, hidden_size=512):
    """Load SAC or TD3 agent from path. Path must contain 'td3' or 'sac' in the name."""
    algo = algo_from_path(path)
    if algo is None:
        print(f"Error: cannot infer algorithm from path (filename must contain 'td3' or 'sac'): {path}")
        sys.exit(1)
    if algo == "sac":
        agent = SAC(obs_dim, action_dim, device=device, hidden_sizes=[hidden_size, hidden_size])
    else:
        agent = TD3Agent(state_dim=obs_dim, action_dim=action_dim)
        if str(agent.device) != str(device):
            agent.device = device
            agent.actor = agent.actor.to(device)
            agent.actor_target = agent.actor_target.to(device)
            agent.critic1 = agent.critic1.to(device)
            agent.critic1_target = agent.critic1_target.to(device)
            agent.critic2 = agent.critic2.to(device)
            agent.critic2_target = agent.critic2_target.to(device)
    agent.load(path)
    return agent


def _resolve_candidates(args):
    """Return list of candidate model paths. Errors if nothing specified or dir is missing/empty."""
    candidates_dir = getattr(args, "candidates_dir", None) or ""
    candidates_list = list(getattr(args, "candidates", None) or [])
    model_path = getattr(args, "model_path", None)

    if candidates_dir:
        if not os.path.isdir(candidates_dir):
            print(f"Error: --candidates_dir is not a directory: {os.path.abspath(candidates_dir)}")
            sys.exit(1)
        from_dir = models_from_dir(candidates_dir)
        if not from_dir:
            print(f"Error: no .pth files in --candidates_dir: {os.path.abspath(candidates_dir)}")
            sys.exit(1)
    else:
        from_dir = []

    if from_dir or candidates_list:
        return from_dir + candidates_list
    if model_path:
        return [model_path]

    print("Error: no candidates specified. Use --model_path, --candidates, or --candidates_dir.")
    sys.exit(1)


def _resolve_opponents(args):
    """Return list of opponent specs: 'weak', 'strong', or paths. Errors if nothing specified or dir is missing/empty."""
    opponents_dir = getattr(args, "opponents_dir", None) or ""
    opponents_list = list(getattr(args, "opponents", None) or [])
    opponent_model = getattr(args, "opponent_model", None)

    if opponents_dir:
        if not os.path.isdir(opponents_dir):
            print(f"Error: --opponents_dir is not a directory: {os.path.abspath(opponents_dir)}")
            sys.exit(1)
        from_dir = models_from_dir(opponents_dir)
        if not from_dir:
            print(f"Error: no .pth files in --opponents_dir: {os.path.abspath(opponents_dir)}")
            sys.exit(1)
    else:
        from_dir = []

    if from_dir or opponents_list:
        return from_dir + opponents_list
    if opponent_model:
        return [opponent_model]

    print("Error: no opponents specified. Use --opponent_model, --opponents, or --opponents_dir.")
    sys.exit(1)


def _print_eval_start_log(candidates, opponents, num_episodes, max_steps, weighted_mode, weights=None, probs=None):
    """Log at start: which models are used and, in weighted mode, episode ratio per opponent."""
    print("\n" + "=" * 60)
    print("EVALUATION CONFIG")
    print("=" * 60)
    print(f"  Total episodes: {num_episodes}  |  Max steps per episode: {max_steps}")
    print(f"  Candidates ({len(candidates)}):")
    for path in candidates:
        algo = algo_from_path(path) or "?"
        print(f"    - {os.path.basename(path)}  [algo: {algo.upper()}]")
    print(f"  Opponents ({len(opponents)}):")
    for i, spec in enumerate(opponents):
        name = spec if spec in ("weak", "strong") else os.path.basename(spec)
        algo = "built-in" if spec in ("weak", "strong") else (algo_from_path(spec) or "?")
        line = f"    - {name}  [algo: {algo}]"
        if weighted_mode and weights is not None and probs is not None:
            w = weights[i]
            expected = int(round(probs[i] * num_episodes))
            pct = probs[i] * 100
            line += f"  |  weight={w}  ->  {expected}/{num_episodes} envs ({pct:.1f}%)"
        print(line)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC or TD3 agents on hockey")
    parser.add_argument("--model_path", type=str, default=None, help="Single candidate (used if --candidates/--candidates_dir not set)")
    parser.add_argument("--opponent_model", type=str, default=None, help="Single opponent path (used if --opponents/--opponents_dir not set)")
    parser.add_argument("--opponent", type=str, default="weak", choices=["weak", "strong"], help="Single built-in opponent when no opponent_model")
    parser.add_argument("--candidates", type=str, nargs="*", default=None, help="Candidate model paths")
    parser.add_argument("--opponents", type=str, nargs="*", default=None, help="Opponent: weak, strong, or model paths")
    parser.add_argument("--candidates_dir", type=str, default=None, help="Folder of .pth files as candidates")
    parser.add_argument("--opponents_dir", type=str, default=None, help="Folder of .pth files as opponents")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size for all models (not stored in checkpoint)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Log results to wandb (aggregate + per-opponent tables)")
    parser.add_argument("--wandb_project", type=str, default="hockey-eval", help="wandb project for this run (use same as training to keep all in one project)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (default: current user)")
    args = parser.parse_args()

    try:
        import hockey.hockey_env as h_env
    except ImportError:
        print("Error: could not import hockey package. Is the hockey env installed and on PYTHONPATH?")
        sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = h_env.HockeyEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    hidden_size = args.hidden_size

    candidates = _resolve_candidates(args)
    opponents = _resolve_opponents(args)
    if not candidates:
        print("Error: no candidates. Use --model_path, --candidates, or --candidates_dir.")
        sys.exit(1)
    if not opponents:
        print("Error: no opponents. Use --opponent_model, --opponents, or --opponents_dir.")
        sys.exit(1)

    # Single candidate, single opponent, no wandb: original 1v1 flow
    if len(candidates) == 1 and len(opponents) == 1 and not args.wandb:
        _print_eval_start_log(candidates, opponents, args.num_episodes, args.max_steps, weighted_mode=False)
        cand_path = candidates[0]
        opp_spec = opponents[0]
        print(f"\nLOADING PLAYER 1: {cand_path}")
        try:
            agent1 = load_agent(cand_path, obs_dim, action_dim, device, hidden_size)
        except Exception as e:
            print(f"Error: failed to load candidate: {e}")
            sys.exit(1)
        opponent_model = None
        opponent_agent = None
        if opp_spec in ("weak", "strong"):
            opponent_agent = h_env.BasicOpponent(weak=(opp_spec == "weak"))
            print(f"LOADING PLAYER 2: Built-in {opp_spec} AI")
        else:
            print(f"LOADING PLAYER 2: {opp_spec}")
            try:
                opponent_model = load_agent(opp_spec, obs_dim, action_dim, device, hidden_size)
            except Exception as e:
                print(f"Error: failed to load opponent: {e}")
                sys.exit(1)
        evaluate(env, agent1, num_episodes=args.num_episodes, render=args.render,
                 max_steps=args.max_steps, opponent_agent=opponent_agent, opponent_model=opponent_model)
        env.close()
        return

    # Multi candidate and/or multi opponent: run eval and optionally log to wandb
    if args.wandb:
        try:
            import wandb
        except ImportError:
            print("Error: --wandb was passed but wandb is not installed. Install with: pip install wandb")
            sys.exit(1)
        if args.wandb:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                job_type="eval",
                config={
                    "candidates": candidates, "opponents": opponents,
                    "num_episodes": args.num_episodes, "max_steps": args.max_steps, "hidden_size": hidden_size,
                },
            )

    # Opponents that are paths get weights from name; built-ins get weight 1. All-path opponents => weighted mode
    opponent_paths = [o for o in opponents if o not in ("weak", "strong")]
    all_opponents_are_paths = len(opponent_paths) == len(opponents) and len(opponents) > 1
    weighted_mode = all_opponents_are_paths and len(opponents) >= 1

    weights = [weight_from_name(o) for o in opponents] if weighted_mode else None
    total_w = sum(weights) if weights else 0
    probs = [w / total_w for w in weights] if total_w else None

    _print_eval_start_log(candidates, opponents, args.num_episodes, args.max_steps,
                          weighted_mode, weights=weights, probs=probs)

    def _opponent_display_name(spec):
        return spec if spec in ("weak", "strong") else os.path.basename(spec)

    def _stacked_bar_image(candidates, win_pcts, draw_pcts, loss_pcts, title, sort_by_loss=True):
        """Build stacked bar chart (win/draw/loss) sorted by loss_rate ascending; return wandb.Image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        win_pcts = np.asarray(win_pcts, dtype=float)
        draw_pcts = np.asarray(draw_pcts, dtype=float)
        loss_pcts = np.asarray(loss_pcts, dtype=float)
        if sort_by_loss:
            # Best (lowest loss, then lowest draw) on the LEFT; worst on the RIGHT
            order = np.lexsort((draw_pcts, loss_pcts))
            candidates = [candidates[i] for i in order]
            win_pcts = win_pcts[order]
            draw_pcts = draw_pcts[order]
            loss_pcts = loss_pcts[order]
        x = np.arange(len(candidates))
        width = 0.6
        fig, ax = plt.subplots(figsize=(max(6, len(candidates) * 0.8), 4))
        ax.bar(x, win_pcts, width, label="Win %", color="#2ecc71")
        ax.bar(x, draw_pcts, width, bottom=win_pcts, label="Draw %", color="#f1c40f")
        ax.bar(x, loss_pcts, width, bottom=win_pcts + draw_pcts, label="Loss %", color="#e74c3c")
        # Percentage labels in each segment (skip if segment too small to fit text)
        for i in range(len(x)):
            if win_pcts[i] >= 4:
                ax.text(x[i], win_pcts[i] / 2, f"{win_pcts[i]:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if draw_pcts[i] >= 4:
                ax.text(x[i], win_pcts[i] + draw_pcts[i] / 2, f"{draw_pcts[i]:.0f}%", ha="center", va="center", fontsize=8, color="black", fontweight="bold")
            if loss_pcts[i] >= 4:
                ax.text(x[i], win_pcts[i] + draw_pcts[i] + loss_pcts[i] / 2, f"{loss_pcts[i]:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(candidates, rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        buf.seek(0)
        import wandb as _wandb
        from PIL import Image
        img = Image.open(buf).copy()
        buf.close()
        return _wandb.Image(img, caption=title)

    if weighted_mode:
        # Weighted: one run per candidate over num_episodes; each episode sample opponent by weight
        # Pre-load all opponent agents (all are paths)
        opp_agents = []
        for opp_path in opponents:
            try:
                opp_agents.append(load_agent(opp_path, obs_dim, action_dim, device, hidden_size))
            except Exception as e:
                print(f"Error: failed to load opponent {opp_path}: {e}")
                env.close()
                sys.exit(1)
        aggregate_rows = []
        per_opponent_rows = []
        n_candidates = len(candidates)
        progress_step = max(1, min(500, args.num_episodes // 20))  # progress every ~5% or 500 eps
        for ci, cand_path in enumerate(candidates):
            try:
                agent1 = load_agent(cand_path, obs_dim, action_dim, device, hidden_size)
            except Exception as e:
                print(f"Error: failed to load candidate {cand_path}: {e}")
                env.close()
                sys.exit(1)
            cand_name = os.path.basename(cand_path)
            print(f"\nCandidate {ci+1}/{n_candidates}: {cand_name} ({args.num_episodes} episodes)...")
            rewards = []
            wins = draws = losses = 0
            per_opp = [{"reward_sum": 0.0, "n": 0, "wins": 0, "draws": 0, "losses": 0} for _ in opponents]
            rng = random.Random(42)
            for ep in range(args.num_episodes):
                opp_idx = rng.choices(range(len(opponents)), weights=probs, k=1)[0]
                opp_agent = opp_agents[opp_idx]
                reward, winner = run_single_episode(env, agent1, opponent_model=opp_agent, max_steps=args.max_steps, seed=ep + 12345)
                rewards.append(reward)
                if winner == 1:
                    wins += 1
                elif winner == -1:
                    losses += 1
                else:
                    draws += 1
                per_opp[opp_idx]["n"] += 1
                per_opp[opp_idx]["reward_sum"] += reward
                if winner == 1:
                    per_opp[opp_idx]["wins"] += 1
                elif winner == -1:
                    per_opp[opp_idx]["losses"] += 1
                else:
                    per_opp[opp_idx]["draws"] += 1
                if (ep + 1) % progress_step == 0 or (ep + 1) == args.num_episodes:
                    pct = (ep + 1) / args.num_episodes * 100
                    print(f"  Episodes {ep+1}/{args.num_episodes} ({pct:.1f}%) | mean_reward={np.mean(rewards):.2f} | wins={wins} draws={draws} losses={losses}")
            n = len(rewards)
            agg_win = wins / n * 100 if n else 0
            agg_draw = draws / n * 100 if n else 0
            agg_loss = losses / n * 100 if n else 0
            aggregate_rows.append((cand_name, np.mean(rewards), agg_win, agg_draw, agg_loss))
            for opp_idx, po in enumerate(per_opp):
                nn = po["n"]
                if nn == 0:
                    continue
                ow = po["wins"] / nn * 100
                od = po["draws"] / nn * 100
                ol = po["losses"] / nn * 100
                per_opponent_rows.append((cand_name, _opponent_display_name(opponents[opp_idx]), nn,
                                         po["reward_sum"] / nn, ow, od, ol))
            print(f"Candidate {cand_name}: mean_reward={np.mean(rewards):.2f} win_rate={agg_win:.1f}% draw_rate={agg_draw:.1f}% loss_rate={agg_loss:.1f}%")
        if args.wandb and aggregate_rows:
            import wandb
            tab_agg = wandb.Table(
                columns=["candidate", "mean_reward", "win_rate", "draw_rate", "loss_rate"],
                data=[[r[0], round(r[1], 2), round(r[2], 1), round(r[3], 1), round(r[4], 1)] for r in aggregate_rows],
            )
            wandb.log({"eval/aggregate_table": tab_agg})
            # Stacked bar: best (lowest loss, then lowest draw) on the left
            sorted_agg = sorted(aggregate_rows, key=lambda r: (r[4], r[3]))
            cands = [r[0] for r in sorted_agg]
            wins = [r[2] for r in sorted_agg]
            draws = [r[3] for r in sorted_agg]
            losses = [r[4] for r in sorted_agg]
            wandb.log({"eval/stacked_bar_aggregate": _stacked_bar_image(cands, wins, draws, losses, "Candidates vs all opponents (sorted by loss rate)", sort_by_loss=False)})
            tab_per = wandb.Table(
                columns=["candidate", "opponent", "n_episodes", "mean_reward", "win_rate", "draw_rate", "loss_rate"],
                data=[[r[0], r[1], r[2], round(r[3], 2), round(r[4], 1), round(r[5], 1), round(r[6], 1)] for r in per_opponent_rows],
            )
            wandb.log({"eval/per_opponent_table": tab_per})
            # One stacked bar plot per opponent: candidates sorted by loss rate vs that opponent
            for opp_name in sorted(set(r[1] for r in per_opponent_rows)):
                opp_rows = [r for r in per_opponent_rows if r[1] == opp_name]
                opp_rows_sorted = sorted(opp_rows, key=lambda r: (r[6], r[5]))
                cands_o = [r[0] for r in opp_rows_sorted]
                wins_o = [r[4] for r in opp_rows_sorted]
                draws_o = [r[5] for r in opp_rows_sorted]
                losses_o = [r[6] for r in opp_rows_sorted]
                slug = re.sub(r"[^\w\-.]", "_", opp_name)[:50]
                wandb.log({f"eval/stacked_bar_vs_{slug}": _stacked_bar_image(cands_o, wins_o, draws_o, losses_o, f"Candidates vs {opp_name} (sorted by loss rate)", sort_by_loss=False)})
    else:
        # Non-weighted: each (candidate, opponent) run num_episodes
        rows = []
        for cand_path in candidates:
            try:
                agent1 = load_agent(cand_path, obs_dim, action_dim, device, hidden_size)
            except Exception as e:
                print(f"Error: failed to load candidate {cand_path}: {e}")
                env.close()
                sys.exit(1)
            cand_name = os.path.basename(cand_path)
            for opp_spec in opponents:
                if opp_spec in ("weak", "strong"):
                    opp_agent = h_env.BasicOpponent(weak=(opp_spec == "weak"))
                    opp_model = None
                else:
                    try:
                        opp_model = load_agent(opp_spec, obs_dim, action_dim, device, hidden_size)
                    except Exception as e:
                        print(f"Error: failed to load opponent {opp_spec}: {e}")
                        env.close()
                        sys.exit(1)
                    opp_agent = None
                stats = evaluate(env, agent1, num_episodes=args.num_episodes, render=args.render,
                                max_steps=args.max_steps, opponent_agent=opp_agent, opponent_model=opp_model,
                                progress_interval=max(1, min(500, args.num_episodes // 20)))
                rows.append((cand_name, _opponent_display_name(opp_spec), stats["mean_reward"],
                             stats["win_rate"], stats["draw_rate"], stats["loss_rate"]))
        if args.wandb and rows:
            import wandb
            tab = wandb.Table(
                columns=["candidate", "opponent", "mean_reward", "win_rate", "draw_rate", "loss_rate"],
                data=[[r[0], r[1], round(r[2], 2), round(r[3], 1), round(r[4], 1), round(r[5], 1)] for r in rows],
            )
            wandb.log({"eval/results_table": tab})
            # Aggregate per candidate (avg over opponents), sorted by loss; one stacked bar
            by_cand = defaultdict(lambda: {"win_rate": [], "draw_rate": [], "loss_rate": []})
            for (cand, _opp, _reward, w, d, l) in rows:
                by_cand[cand]["win_rate"].append(w)
                by_cand[cand]["draw_rate"].append(d)
                by_cand[cand]["loss_rate"].append(l)
            agg_cands = []
            agg_w, agg_d, agg_l = [], [], []
            for cand, v in by_cand.items():
                n = len(v["win_rate"])
                agg_cands.append(cand)
                agg_w.append(sum(v["win_rate"]) / n)
                agg_d.append(sum(v["draw_rate"]) / n)
                agg_l.append(sum(v["loss_rate"]) / n)
            order = np.lexsort((np.array(agg_d), np.array(agg_l)))
            agg_cands = [agg_cands[i] for i in order]
            agg_w = [agg_w[i] for i in order]
            agg_d = [agg_d[i] for i in order]
            agg_l = [agg_l[i] for i in order]
            wandb.log({"eval/stacked_bar_aggregate": _stacked_bar_image(agg_cands, agg_w, agg_d, agg_l, "Candidates vs all opponents (sorted by loss rate)", sort_by_loss=False)})
            for opp_name in sorted(set(r[1] for r in rows)):
                opp_rows = [r for r in rows if r[1] == opp_name]
                opp_rows_sorted = sorted(opp_rows, key=lambda r: (r[5], r[4]))
                cands_o = [r[0] for r in opp_rows_sorted]
                opp_slug = re.sub(r"[^\w\-.]", "_", opp_name)[:50]
                wandb.log({f"eval/stacked_bar_vs_{opp_slug}": _stacked_bar_image(cands_o, [r[3] for r in opp_rows_sorted], [r[4] for r in opp_rows_sorted], [r[5] for r in opp_rows_sorted], f"Candidates vs {opp_name} (sorted by loss rate)", sort_by_loss=False)})

    env.close()


if __name__ == "__main__":
    main()