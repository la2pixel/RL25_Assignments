#!/usr/bin/env python3
"""
Evaluate a trained SAC or TD3 agent on the hockey environment.

Single-model usage:
    python evaluate_hockey.py --model_path checkpoints/sac_hockey_best.pth --opponent weak
    python evaluate_hockey.py --model_path checkpoints/td3_hockey_best.pth --opponent strong
    python evaluate_hockey.py --model_path checkpoints/agent_v1.pth --opponent_model checkpoints/agent_v2.pth

Group evaluation (requires --config):
    python evaluate_hockey.py --config eval_config.json --num_episodes 10 --output_dir eval_results
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hockey_sac import SAC
from hockey_td3 import TD3Agent


def model_name_to_spec(model_name):
    """Derive algorithm-reward spec from model name, e.g. sac-attack-r1 -> sac-attack, td3-defense-r5 -> td3-defense."""
    if model_name in ("weak", "strong"):
        return model_name
    # AI Usage: asked AI for regex to strip the round suffix (e.g. -r1, -r5) from model names
    m = re.match(r"^(.+)-r\d+$", model_name, re.IGNORECASE)
    return m.group(1) if m else model_name


class _BuiltinAgentWrapper:
    """Wraps BasicOpponent so it can be used as agent1 in evaluate() (select_action API)."""

    def __init__(self, basic_opponent):
        self._opp = basic_opponent

    def select_action(self, obs, deterministic=True):
        return self._opp.act(obs)


def load_agent(path, obs_dim, action_dim, device, hidden_sizes, cache):
    """Load a trained agent from a file path. Returns None for builtin opponents ('weak'/'strong').
    cache: dict path -> agent; avoids reloading the same model.
    """
    if path in ("weak", "strong"):
        return None
    if path in cache:
        return cache[path]
    agent_type = "td3" if "td3" in path.lower() else "sac"
    if agent_type == "sac":
        agent = SAC(obs_dim, action_dim, device=device, hidden_sizes=hidden_sizes)
    else:
        agent = TD3Agent(obs_dim, action_dim)
    agent.load(path)
    cache[path] = agent
    return agent


def get_state(state):
    if isinstance(state, np.ndarray):
        state = np.ascontiguousarray(state, dtype=np.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        return state
    return np.array(state, dtype=np.float32)


def evaluate(env, agent1, num_episodes=10, render=False, max_steps=500,
             seeds=None, opponent_agent=None, opponent_model=None, quiet=False):
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

    if not quiet:
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

        if not quiet:
            print(f"{i+1:3d} | {episode_reward:8.2f} | {step:5d} | {outcome}")

    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'win_rate': wins / num_episodes * 100,
        'wins': wins, 'losses': losses, 'draws': draws
    }

    if not quiet:
        print("-" * 80)
        print(f"Results vs {opp_name}:")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
        print(f"  Win Rate:    {stats['win_rate']:.1f}%")
        print(f"  Record:      {wins}W - {losses}L - {draws}D")
        print("-" * 80)
    return stats


def run_group_evaluation(groups, evaluation_group, env, h_env, n_episodes, max_steps, device,
                         hidden_sizes):
    """Run each model in each group vs each model in the evaluation group.
    Model entries are dicts with 'path' and 'name' keys.
    Returns raw_results: list of dicts with group_name, model_a_name, eval_opponent_name, stats.
    """
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    cache = {}
    raw_results = []
    eval_models = evaluation_group["models"]
    total = sum(len(g["models"]) * len(eval_models) for g in groups)
    n_done = 0

    for group in groups:
        group_name = group["name"]
        for model_entry in group["models"]:
            model_a_path = model_entry["path"]
            model_a_name = model_entry["name"]
            agent1 = load_agent(model_a_path, obs_dim, action_dim, device, hidden_sizes, cache)
            if agent1 is None:
                agent1 = _BuiltinAgentWrapper(h_env.BasicOpponent(weak=(model_a_path.lower() == "weak")))

            for eval_entry in eval_models:
                eval_path = eval_entry["path"]
                eval_opponent_name = eval_entry["name"]
                # AI Usage: asked AI how to generate reproducible seeds from model name pairs
                base_seed = hash((model_a_name, eval_opponent_name)) % (2 ** 32)
                seeds = [int(s) for s in np.random.RandomState(base_seed).randint(0, 10000, size=n_episodes)]

                if eval_path.lower() in ("weak", "strong"):
                    opponent_agent = h_env.BasicOpponent(weak=(eval_path.lower() == "weak"))
                    opponent_model = None
                else:
                    opponent_agent = None
                    opponent_model = load_agent(eval_path, obs_dim, action_dim, device, hidden_sizes, cache)

                stats = evaluate(
                    env, agent1, num_episodes=n_episodes, render=False, max_steps=max_steps,
                    seeds=seeds, opponent_agent=opponent_agent, opponent_model=opponent_model, quiet=True
                )
                raw_results.append({
                    "group_name": group_name,
                    "model_a_name": model_a_name,
                    "eval_opponent_name": eval_opponent_name,
                    "mean_reward": stats["mean_reward"],
                    "std_reward": stats["std_reward"],
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "draws": stats["draws"],
                    "n_episodes": n_episodes,
                })
                n_done += 1
                pct = 100.0 * n_done / total if total else 0
                sys.stdout.write(f"\rMatchups: {n_done}/{total} ({pct:.1f}%)")
                sys.stdout.flush()
    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return raw_results


def aggregate_results(raw_results):
    """Aggregate raw results into per-model, per-group, and best-per-group.
    Returns (per_model, per_group, best_per_group).
    """
    model_w = defaultdict(lambda: {"rewards": [], "stds": [], "wins": 0, "losses": 0, "draws": 0, "n": 0})
    for r in raw_results:
        key = (r["group_name"], r["model_a_name"])
        model_w[key]["rewards"].append(r["mean_reward"])
        model_w[key]["stds"].append(r["std_reward"])
        model_w[key]["wins"] += r["wins"]
        model_w[key]["losses"] += r["losses"]
        model_w[key]["draws"] += r["draws"]
        model_w[key]["n"] += r["n_episodes"]

    per_model = {}
    for (group_name, model_a_name), w in model_w.items():
        n = w["n"]
        per_model[(group_name, model_a_name)] = {
            "mean_reward": np.mean(w["rewards"]),
            "std_reward": np.mean(w["stds"]),
            "wins": w["wins"], "losses": w["losses"], "draws": w["draws"], "n_games": n,
            "win_rate": 100.0 * w["wins"] / n if n else 0,
            "draw_rate": 100.0 * w["draws"] / n if n else 0,
            "loss_rate": 100.0 * w["losses"] / n if n else 0,
        }

    group_w = defaultdict(lambda: {"means": [], "wins": 0, "losses": 0, "draws": 0, "n": 0})
    for (group_name, model_a_name), stats in per_model.items():
        group_w[group_name]["means"].append(stats["mean_reward"])
        group_w[group_name]["wins"] += stats["wins"]
        group_w[group_name]["losses"] += stats["losses"]
        group_w[group_name]["draws"] += stats["draws"]
        group_w[group_name]["n"] += stats["n_games"]

    per_group = {}
    for group_name, w in group_w.items():
        n = w["n"]
        per_group[group_name] = {
            "mean_reward": np.mean(w["means"]),
            "std_reward": np.std(w["means"]) if len(w["means"]) > 1 else 0.0,
            "wins": w["wins"], "losses": w["losses"], "draws": w["draws"], "n_games": n,
            "win_rate": 100.0 * w["wins"] / n if n else 0,
            "draw_rate": 100.0 * w["draws"] / n if n else 0,
            "loss_rate": 100.0 * w["losses"] / n if n else 0,
        }

    best_per_group = {}
    for group_name in per_group:
        models_in_group = [m for (g, m) in per_model if g == group_name]
        if not models_in_group:
            best_per_group[group_name] = None
            continue
        best_model = max(models_in_group, key=lambda m: per_model[(group_name, m)]["mean_reward"])
        best_per_group[group_name] = best_model

    return per_model, per_group, best_per_group


def aggregate_by_spec(per_model):
    """Aggregate per_model across groups (rounds) by algorithm-reward spec."""
    spec_w = defaultdict(lambda: {"means": [], "stds": [], "wins": 0, "losses": 0, "draws": 0, "n": 0})
    for (group_name, model_a_name), stats in per_model.items():
        spec = model_name_to_spec(model_a_name)
        spec_w[spec]["means"].append(stats["mean_reward"])
        spec_w[spec]["stds"].append(stats["std_reward"])
        spec_w[spec]["wins"] += stats["wins"]
        spec_w[spec]["losses"] += stats["losses"]
        spec_w[spec]["draws"] += stats["draws"]
        spec_w[spec]["n"] += stats["n_games"]
    per_spec = {}
    for spec, w in spec_w.items():
        n = w["n"]
        per_spec[spec] = {
            "mean_reward": np.mean(w["means"]),
            "std_reward": np.std(w["means"]) if len(w["means"]) > 1 else (np.mean(w["stds"]) if w["stds"] else 0.0),
            "wins": w["wins"], "losses": w["losses"], "draws": w["draws"], "n_games": n,
            "win_rate": 100.0 * w["wins"] / n if n else 0,
            "draw_rate": 100.0 * w["draws"] / n if n else 0,
            "loss_rate": 100.0 * w["losses"] / n if n else 0,
        }
    return per_spec


def plot_within_group_reward(per_model, best_per_group, output_dir):
    """One figure per group: horizontal bar chart of mean reward +/- std; highlight best model."""
    # AI Usage: asked AI how to create horizontal bar charts with error bars and highlight the best model
    os.makedirs(output_dir, exist_ok=True)
    groups = sorted({g for (g, _) in per_model})
    for group_name in groups:
        models_in_group = sorted([m for (g, m) in per_model if g == group_name])
        if not models_in_group:
            continue
        means = [per_model[(group_name, m)]["mean_reward"] for m in models_in_group]
        stds = [per_model[(group_name, m)]["std_reward"] for m in models_in_group]
        best = best_per_group.get(group_name)
        colors = ["#2ecc71" if m == best else "#3498db" for m in models_in_group]
        y_pos = np.arange(len(models_in_group))
        fig, ax = plt.subplots(figsize=(8, max(4, len(models_in_group) * 0.4)))
        bars = ax.barh(y_pos, means, xerr=stds, color=colors, capsize=3, error_kw={"linewidth": 1.5})
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        if best is not None:
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.0)
            idx = models_in_group.index(best)
            bars[idx].set_linewidth(2.5)
            bars[idx].set_edgecolor("black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models_in_group, fontsize=9)
        ax.set_xlabel("Mean reward")
        ax.set_ylabel("Model")
        ax.set_title(f"Mean reward vs evaluation group — {group_name}\n(best (highest mean), highlighted)")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        fig.tight_layout()
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in group_name)
        fig.savefig(os.path.join(output_dir, f"reward_within_group_{safe_name}.png"), dpi=150)
        plt.close(fig)


def plot_group_level_reward(per_group, output_dir):
    """One bar chart: group names vs aggregate mean reward with std"""
    # AI Usage: asked AI how to create bar charts with error bars and best-group highlighting
    os.makedirs(output_dir, exist_ok=True)
    group_names = sorted(per_group.keys())
    means = [per_group[g]["mean_reward"] for g in group_names]
    stds = [per_group[g]["std_reward"] for g in group_names]
    best_idx = int(np.argmax(means)) if group_names else None
    colors = ["#2ecc71" if i == best_idx else "#3498db" for i in range(len(group_names))]
    x = np.arange(len(group_names))
    fig, ax = plt.subplots(figsize=(max(6, len(group_names) * 1.2), 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, error_kw={"linewidth": 1.5})
    if best_idx is not None:
        for b in bars:
            b.set_edgecolor("black")
            b.set_linewidth(1.0)
        bars[best_idx].set_linewidth(2.5)
        bars[best_idx].set_edgecolor("black")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_ylabel("Mean reward (aggregate)")
    ax.set_xlabel("Group")
    ax.set_title("Group-level mean reward vs evaluation group\n(best (highest mean), highlighted)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_by_group.png"), dpi=150)
    plt.close(fig)


def plot_within_group_wdl(per_model, best_per_group, output_dir):
    """One figure per group: stacked bar (win% / draw% / loss%) per model."""
    # AI Usage: asked AI how to create stacked bar charts with percentage labels inside bars
    os.makedirs(output_dir, exist_ok=True)
    groups = sorted({g for (g, _) in per_model})
    for group_name in groups:
        models_in_group = sorted([m for (g, m) in per_model if g == group_name])
        if not models_in_group:
            continue
        win_rates = [per_model[(group_name, m)]["win_rate"] for m in models_in_group]
        draw_rates = [per_model[(group_name, m)]["draw_rate"] for m in models_in_group]
        loss_rates = [per_model[(group_name, m)]["loss_rate"] for m in models_in_group]
        x = np.arange(len(models_in_group))
        width = 0.6
        fig, ax = plt.subplots(figsize=(max(6, len(models_in_group) * 1.0), 5))
        ax.bar(x, win_rates, width, label="Win %", color="#2ecc71")
        ax.bar(x, draw_rates, width, bottom=win_rates, label="Draw %", color="#f1c40f")
        ax.bar(x, loss_rates, width, bottom=np.array(win_rates) + np.array(draw_rates), label="Loss %", color="#e74c3c")
        for i in range(len(models_in_group)):
            bottom = 0
            for rate in [win_rates[i], draw_rates[i], loss_rates[i]]:
                if rate >= 3:
                    ax.text(i, bottom + rate / 2, f"{rate:.0f}%", ha="center", va="center", fontsize=7)
                bottom += rate
        ax.set_xticks(x)
        ax.set_xticklabels(models_in_group, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Rate (%)")
        ax.set_xlabel("Model")
        ax.set_title(f"Win / Draw / Loss vs evaluation group — {group_name}")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 100)
        fig.tight_layout()
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in group_name)
        fig.savefig(os.path.join(output_dir, f"wdl_within_group_{safe_name}.png"), dpi=150)
        plt.close(fig)


def plot_group_level_wdl(per_group, output_dir):
    """One stacked bar chart: win% / draw% / loss% per group."""
    # AI Usage: asked AI how to create stacked bar charts with percentage labels inside bars
    os.makedirs(output_dir, exist_ok=True)
    group_names = sorted(per_group.keys())
    win_rates = [per_group[g]["win_rate"] for g in group_names]
    draw_rates = [per_group[g]["draw_rate"] for g in group_names]
    loss_rates = [per_group[g]["loss_rate"] for g in group_names]
    x = np.arange(len(group_names))
    width = 0.6
    fig, ax = plt.subplots(figsize=(max(6, len(group_names) * 1.2), 5))
    ax.bar(x, win_rates, width, label="Win %", color="#2ecc71")
    ax.bar(x, draw_rates, width, bottom=win_rates, label="Draw %", color="#f1c40f")
    ax.bar(x, loss_rates, width, bottom=np.array(win_rates) + np.array(draw_rates), label="Loss %", color="#e74c3c")
    for i in range(len(group_names)):
        bottom = 0
        for rate in [win_rates[i], draw_rates[i], loss_rates[i]]:
            if rate >= 3:
                ax.text(i, bottom + rate / 2, f"{rate:.0f}%", ha="center", va="center", fontsize=8)
            bottom += rate
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Group")
    ax.set_title("Group-level Win / Draw / Loss vs evaluation group")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "wdl_by_group.png"), dpi=150)
    plt.close(fig)


def plot_spec_reward(per_spec, output_dir):
    """Bar chart: mean reward +/- std per algorithm-reward spec (average across groups/rounds)."""
    # AI Usage: asked AI how to create bar charts with error bars
    os.makedirs(output_dir, exist_ok=True)
    spec_names = sorted(per_spec.keys())
    if not spec_names:
        return
    means = [per_spec[s]["mean_reward"] for s in spec_names]
    stds = [per_spec[s]["std_reward"] for s in spec_names]
    x = np.arange(len(spec_names))
    fig, ax = plt.subplots(figsize=(max(6, len(spec_names) * 1.0), 5))
    ax.bar(x, means, yerr=stds, capsize=5, color="#3498db", error_kw={"linewidth": 1.5})
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(spec_names, rotation=45, ha="right")
    ax.set_ylabel("Mean reward (aggregate across rounds)")
    ax.set_xlabel("Algorithm-reward shape")
    ax.set_title("Mean reward by algorithm-reward shape")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_by_spec.png"), dpi=150)
    plt.close(fig)


def plot_spec_wdl(per_spec, output_dir):
    """Stacked bar: win% / draw% / loss% per algorithm-reward spec (average across groups/rounds)."""
    # AI Usage: asked AI how to create stacked bar charts with percentage labels inside bars
    os.makedirs(output_dir, exist_ok=True)
    spec_names = sorted(per_spec.keys())
    if not spec_names:
        return
    win_rates = [per_spec[s]["win_rate"] for s in spec_names]
    draw_rates = [per_spec[s]["draw_rate"] for s in spec_names]
    loss_rates = [per_spec[s]["loss_rate"] for s in spec_names]
    x = np.arange(len(spec_names))
    width = 0.6
    fig, ax = plt.subplots(figsize=(max(6, len(spec_names) * 1.0), 5))
    ax.bar(x, win_rates, width, label="Win %", color="#2ecc71")
    ax.bar(x, draw_rates, width, bottom=win_rates, label="Draw %", color="#f1c40f")
    ax.bar(x, loss_rates, width, bottom=np.array(win_rates) + np.array(draw_rates), label="Loss %", color="#e74c3c")
    for i in range(len(spec_names)):
        bottom = 0
        for rate in [win_rates[i], draw_rates[i], loss_rates[i]]:
            if rate >= 3:
                ax.text(i, bottom + rate / 2, f"{rate:.0f}%", ha="center", va="center", fontsize=8)
            bottom += rate
    ax.set_xticks(x)
    ax.set_xticklabels(spec_names, rotation=45, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Algorithm-reward shape")
    ax.set_title("Win / Draw / Loss by algorithm-reward shape")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "wdl_by_spec.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC or TD3 Agent')
    parser.add_argument('--model_path', type=str, default=None, help='Single model path (optional if using group eval)')
    parser.add_argument('--opponent_model', type=str, default=None)
    parser.add_argument('--opponent', type=str, default='weak', choices=['weak', 'strong'])
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='JSON config for group eval (required when --model_path is not set)')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Where to save plots and results (group eval)')
    parser.add_argument('--quiet', action='store_true', help='Less per-episode output during group eval')

    args = parser.parse_args()

    try:
        import hockey.hockey_env as h_env
    except ImportError:
        print("Could not import hockey package.")
        return

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_group_eval = args.model_path is None
    if use_group_eval:
        if not args.config:
            print("Error: --config is required for group evaluation.")
            return

        with open(args.config, "r") as f:
            cfg = json.load(f)

        groups = cfg["groups"]
        evaluation_group = cfg["evaluation_group"]

        env = h_env.HockeyEnv()
        print("Running group vs evaluation-group evaluation...")
        raw_results = run_group_evaluation(
            groups, evaluation_group, env, h_env,
            n_episodes=args.num_episodes, max_steps=args.max_steps, device=device,
            hidden_sizes=args.hidden_sizes
        )

        os.makedirs(args.output_dir, exist_ok=True)
        per_model, per_group, best_per_group = aggregate_results(raw_results)
        plot_within_group_reward(per_model, best_per_group, args.output_dir)
        plot_group_level_reward(per_group, args.output_dir)
        plot_within_group_wdl(per_model, best_per_group, args.output_dir)
        plot_group_level_wdl(per_group, args.output_dir)
        per_spec = aggregate_by_spec(per_model)
        plot_spec_reward(per_spec, args.output_dir)
        plot_spec_wdl(per_spec, args.output_dir)
        out = {
            "raw_results": raw_results,
            "per_model": {f"{g}|{m}": v for (g, m), v in per_model.items()},
            "per_group": per_group,
            "best_per_group": best_per_group,
            "per_spec": per_spec,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.output_dir, f"results_{ts}.json")
        with open(results_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output_dir} (results: {results_path})")
        env.close()
        return

    # Single-model evaluation
    agent_type = 'td3' if 'td3' in args.model_path.lower() else 'sac'
    opponent_type = 'td3' if (args.opponent_model and 'td3' in args.opponent_model.lower()) else ('sac' if args.opponent_model else None)
    env = h_env.HockeyEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2

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
