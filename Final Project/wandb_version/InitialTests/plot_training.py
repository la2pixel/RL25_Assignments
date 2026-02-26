#!/usr/bin/env python3
"""
Plot training curves from SAC training logs.
"""

import argparse
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
HAS_MPL = True


def load_train_csv(path):
    data = {'step': [], 'sps': [], 'avg_train_reward': [], 'best_eval_metric': [], 'wall_time': [], 'alpha': []}
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['step'].append(int(row['step']))
            data['sps'].append(int(row['sps']))
            data['avg_train_reward'].append(float(row['avg_train_reward']))
            data['best_eval_metric'].append(float(row['best_eval_metric']))
            data['wall_time'].append(float(row['wall_time']))

            if 'alpha' in row and row['alpha']:
                data['alpha'].append(float(row['alpha']))
    return data if data['step'] else None


def load_eval_csv(path):
    if not os.path.exists(path):
        return None
    data = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for h in headers:
            data[h] = []
        for row in reader:
            for h in headers:
                try:
                    data[h].append(float(row[h]))
                except (ValueError, KeyError):
                    data[h].append(0.0)
    if 'step' in data:
        data['step'] = [int(s) for s in data['step']]
    return data if data.get('step') else None


def load_config(path):
    """Load config.json if it exists."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def get_run_label(run_dir, config):
    dirname = os.path.basename(os.path.normpath(run_dir))
    label_map = {
        'alpha_auto':   'Auto Alpha',
        'alpha_fixed':  'Alpha = 0.2',
        'alpha_fixed2': 'Alpha = 0.1',
    }
    if dirname in label_map:
        return label_map[dirname]

    parts = []
    mode = config.get('reward_mode', 'default')
    if mode != 'default':
        parts.append(mode)
    if config.get('pink_noise', False):
        parts.append('pink')
    else:
        parts.append('vanilla')
    return ' '.join(parts) if parts else dirname


def smooth(values, weight=0.8):
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_runs(run_dirs, save_path=None, smooth_weight=0.6, labels=None):
    """Generate comparison plots for one or more training runs."""
    if not HAS_MPL:
        print("Cannot plot without matplotlib.")
        return

    runs = []
    for i, d in enumerate(run_dirs):
        train_data = load_train_csv(os.path.join(d, 'train_log.csv'))
        eval_data = load_eval_csv(os.path.join(d, 'eval_log.csv'))
        config = load_config(os.path.join(d, 'config.json'))
        label = labels[i] if labels and i < len(labels) else get_run_label(d, config)

        if train_data is None and eval_data is None:
            print(f"Warning: No logs found in {d}, skipping.")
            continue

        runs.append({
            'dir': d,
            'label': label,
            'train': train_data,
            'eval': eval_data,
            'config': config,
        })

    if not runs:
        print("No valid runs to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SAC Hockey Training', fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))

    ax = axes[0, 0]
    for i, run in enumerate(runs):
        if run['train']:
            steps = np.array(run['train']['step']) / 1000
            rewards = run['train']['avg_train_reward']
            ax.plot(steps, smooth(rewards, smooth_weight), color=colors[i],
                    label=run['label'], linewidth=1.5)
            ax.plot(steps, rewards, color=colors[i], alpha=0.15, linewidth=0.5)
    ax.set_xlabel('Steps (k)')
    ax.set_ylabel('Avg Train Reward')
    ax.set_title('Training Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i, run in enumerate(runs):
        if run['eval']:
            steps = np.array(run['eval']['step']) / 1000
            ax.plot(steps, smooth(run['eval']['avg_reward'], smooth_weight), color=colors[i],
                    label=run['label'], marker='o', markersize=3, linewidth=1.5)
    ax.set_xlabel('Steps (k)')
    ax.set_ylabel('Eval Reward (raw)')
    ax.set_title('Eval Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax = axes[1, 0]
    for i, run in enumerate(runs):
        if run['eval']:
            steps = np.array(run['eval']['step']) / 1000
            ax.plot(steps, [w * 100 for w in smooth(run['eval']['avg_winrate'], smooth_weight)], color=colors[i],
                    label=f"{run['label']} (overall)", linewidth=1.5, marker='o', markersize=3)
            ax.plot(steps, [w * 100 for w in smooth(run['eval']['win_rate_weak'], smooth_weight)], color=colors[i],
                    linestyle='--', alpha=0.5, linewidth=1, label=f"{run['label']} (weak)")
            ax.plot(steps, [w * 100 for w in smooth(run['eval']['win_rate_strong'], smooth_weight)], color=colors[i],
                    linestyle=':', alpha=0.5, linewidth=1, label=f"{run['label']} (strong)")
    ax.set_xlabel('Steps (k)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rates')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    ax = axes[1, 1]
    has_alpha = False
    for i, run in enumerate(runs):
        if run['train'] and run['train']['alpha']:
            has_alpha = True
            steps = np.array(run['train']['step'][:len(run['train']['alpha'])]) / 1000
            ax.plot(steps, run['train']['alpha'], color=colors[i],
                    label=run['label'], linewidth=1.5)
    ax.set_xlabel('Steps (k)')
    ax.set_ylabel('Alpha (entropy coeff)')
    ax.set_title('Alpha (Learned)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if not has_alpha:
        ax.text(0.5, 0.5, 'No alpha data\n(older logs)', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        # Save to save_dir of the first run
        default_path = os.path.join(runs[0]['dir'], 'training_curves.png')
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {default_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot SAC training curves')
    parser.add_argument('--dirs', type=str, nargs='+', required=True,
                        help='Checkpoint directories to plot (can pass multiple for comparison)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Custom labels for each run dir (must match number of --dirs)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to this path (default: <first_dir>/training_curves.png)')
    parser.add_argument('--smooth', type=float, default=0.6,
                        help='Smoothing weight for training reward (0=none, 0.9=heavy)')
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.dirs):
        parser.error(f'--labels count ({len(args.labels)}) must match --dirs count ({len(args.dirs)})')

    plot_runs(args.dirs, save_path=args.save, smooth_weight=args.smooth, labels=args.labels)


if __name__ == '__main__':
    main()