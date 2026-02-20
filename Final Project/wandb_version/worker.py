"""
Worker for round-based training: polls coordinator (via wandb round trigger), requests one pool key
(via round-requests / round-assignments), runs training, marks key finished.
One key per round per worker; exits when coordinator writes round=0 and no pool_keys (max rounds or stop).
"""

import argparse
import os
import random
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_env
load_env.load_dotenv()

import wandb_pool as pool
import config_loader as cl


def main():
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "training.yaml")
    parser = argparse.ArgumentParser(description="Worker: all settings from config (training.yaml). .env only for WANDB_API_KEY.")
    parser.add_argument("--config", type=str, default=default_config, help="Path to training config YAML (same as coordinator)")
    args = parser.parse_args()

    cfg = cl.load_config(args.config)
    entity = cfg.get("entity")
    if not entity:
        print("Entity is required in config (training.yaml): set 'entity: your_wandb_username'.", file=sys.stderr)
        sys.exit(1)
    project = cfg.get("project") or "hockey-rounds"
    worker_cfg = cfg.get("worker") or {}
    poll_interval = worker_cfg.get("poll_interval", 60)
    args.entity = entity
    args.project = project
    args.poll_interval = poll_interval
    print(f"Using project: {project!r} | entity: {entity!r}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    train_script = os.path.join(script_dir, "train_parallel.py")

    worker_id = uuid.uuid4().hex
    last_round_done = None
    has_seen_round = False

    while True:
        try:
            trigger = pool.read_round_trigger(args.entity, args.project)
        except Exception as e:
            print(f"[Poll] Error reading round trigger: {e}")
            time.sleep(args.poll_interval)
            continue

        current_round = trigger["round"]
        pool_keys = trigger.get("pool_keys") or []
        builtin_opponents = trigger.get("builtin_opponents", ["weak", "strong"])
        print(f"[Poll] Round: {current_round} | Pool keys (from coordinator): {pool_keys}")

        if current_round > 0:
            has_seen_round = True
        if current_round <= 0 or not pool_keys:
            if has_seen_round and (current_round <= 0 and not pool_keys):
                print("Coordinator finished (max rounds or stop condition). Exiting.")
                return
            print(f"[Poll] Round trigger has no pool_keys. Skipping.")
            time.sleep(args.poll_interval)
            continue

        if current_round == last_round_done:
            print(f"[Round {current_round}] Already done by this worker. Waiting for next round (every {args.poll_interval}s).")
            time.sleep(args.poll_interval)
            continue

        finished = pool.read_finished_pool_keys(args.entity, args.project, current_round)
        if len(finished) >= len(pool_keys):
            time.sleep(args.poll_interval)
            continue

        # Stagger requests so workers don't all request at once (avoids race where coordinator
        # wasn't ready yet and both workers got the same key).
        delay = random.uniform(2.0, 15.0)
        print(f"[Round {current_round}] Staggering key request by {delay:.1f}s to reduce collision.")
        time.sleep(delay)
        # Re-check finished count after delay in case round completed
        finished = pool.read_finished_pool_keys(args.entity, args.project, current_round)
        if len(finished) >= len(pool_keys):
            time.sleep(args.poll_interval)
            continue

        # Request a key; coordinator will assign one (sole writer of round-assignments)
        pool.append_key_request(args.entity, args.project, current_round, worker_id)
        pool_key = None
        wait_log_interval = 30  # log at most every 30s while waiting for assignment
        last_wait_log = 0
        while pool_key is None:
            pool_key = pool.get_assigned_pool_key(args.entity, args.project, current_round, worker_id)
            if pool_key is None:
                now = time.time()
                if now - last_wait_log >= wait_log_interval:
                    print(f"[Round {current_round}] Waiting for coordinator to assign a pool key (poll every {args.poll_interval}s)...")
                    last_wait_log = now
                time.sleep(args.poll_interval)

        print(f"[Round {current_round}] Assigned pool key: {pool_key}. Opponents = {len(builtin_opponents)} builtin + {len(pool_keys)} pool.")

        cmd = [
            sys.executable,
            train_script,
            "--config", os.path.abspath(args.config),
            "--pool_key", pool_key,
            "--round", str(current_round),
            "--entity", args.entity,
            "--wandb_project", args.project,
            "--builtin_opponents", ",".join(builtin_opponents),
        ]
        ret = subprocess.run(cmd, cwd=parent_dir)
        if ret.returncode != 0:
            print(f"[Round {current_round}] Pool key {pool_key} exited with code {ret.returncode}")

        pool.mark_pool_key_finished(args.entity, args.project, pool_key)
        last_round_done = current_round
        print(f"[Round {current_round}] Run finished. Polling for next round (every {args.poll_interval}s).")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
