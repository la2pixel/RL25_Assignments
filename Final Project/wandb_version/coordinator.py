"""
Coordinator for round-based training: starts each round, waits for all pool keys to finish, then next round.
Uses a config file for pool_keys (specs), project, entity, and builtin_opponents so the command is minimal.
"""

import argparse
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_env
load_env.load_dotenv()

try:
    import wandb
except ImportError:
    wandb = None

import wandb_pool as pool
import config_loader as cl


def _resolve_entity(config_entity, cli_entity):
    # CLI > env > config
    return cli_entity or os.environ.get("ENTITY") or config_entity or None


def _resolve_project(config_project, cli_project):
    # CLI > env > config > default
    return cli_project or os.environ.get("WANDB_PROJECT") or config_project or "hockey-rounds"


def main():
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "training.yaml")
    parser = argparse.ArgumentParser(description="Coordinator: drive rounds using config (pool_keys, project, entity, coordinator section).")
    parser.add_argument("--config", type=str, default=default_config, help=f"Path to training config YAML (default: {default_config})")
    parser.add_argument("--entity", type=str, default=None, help="wandb entity (overrides config and ENTITY env)")
    parser.add_argument("--project", type=str, default=None, help="wandb project (overrides config and WANDB_PROJECT env)")
    parser.add_argument("--max_rounds", type=int, default=None, help="stop after this many rounds (default from config coordinator.max_rounds)")
    parser.add_argument("--poll_interval", type=int, default=None, help="seconds between checks (default from config coordinator.poll_interval)")
    parser.add_argument("--timeout_hours", type=float, default=None, help="abort round after this many hours (default from config coordinator.timeout_hours)")
    args = parser.parse_args()

    if wandb is None:
        print("wandb is required. pip install wandb")
        sys.exit(1)

    cfg = cl.load_config(args.config)
    coord = cfg.get("coordinator") or {}
    if args.max_rounds is None:
        args.max_rounds = coord.get("max_rounds", 50)
    if args.poll_interval is None:
        args.poll_interval = coord.get("poll_interval", 120)
    if args.timeout_hours is None:
        args.timeout_hours = coord.get("timeout_hours", 24)
    pool_keys = cl.get_pool_keys_from_config(args.config)
    project = _resolve_project(cfg.get("project"), args.project)
    entity = _resolve_entity(cfg.get("entity"), args.entity)
    if not entity:
        print("Entity is required: set in config (entity: your_username), pass --entity, or set env ENTITY.", file=sys.stderr)
        sys.exit(1)
    builtin_cfg = cfg.get("builtin_opponents")
    if isinstance(builtin_cfg, list):
        builtin_opponents = [str(x).strip() for x in builtin_cfg if x]
    else:
        builtin_opponents = [s.strip() for s in (builtin_cfg or "weak,strong").split(",") if s.strip()]
    if not builtin_opponents:
        builtin_opponents = ["weak", "strong"]
    print(f"Using project: {project!r} | entity: {entity!r} | pool_keys: {pool_keys}")

    # One minimal wandb run for the whole session (only for uploading artifacts; no metrics). Silent so only agent runs are visible in console.
    try:
        settings = wandb.Settings(console="off", silent=True)
    except Exception:
        settings = None
    wandb.init(project=project, entity=entity, name="coordinator", job_type="coordinator", config={"pool_keys": pool_keys}, tags=["system", "coordinator"], settings=settings)

    # Infer which round to start from: read wandb trigger + finished-pool (no start_round parameter)
    try:
        trigger = pool.read_round_trigger(entity, project)
    except Exception:
        trigger = {}
    trigger_round = trigger.get("round", 0)
    finished = pool.read_finished_pool_keys_merged(entity, project, trigger_round) if trigger_round > 0 else []
    if trigger_round == 0 or len(finished) >= len(pool_keys):
        start_round = trigger_round + 1 if trigger_round > 0 else 1
        if trigger_round > 0:
            print(f"Resuming: last trigger was round {trigger_round}, all keys finished → starting at round {start_round}.")
    else:
        start_round = trigger_round + 1
        print(f"Resuming: round {trigger_round} in progress ({len(finished)}/{len(pool_keys)} finished). Waiting for round {trigger_round} to complete...")
        deadline = time.time() + args.timeout_hours * 3600
        while time.time() < deadline:
            try:
                finished = pool.read_finished_pool_keys_merged(entity, project, trigger_round)
            except Exception as e:
                print(f"Error reading finished pool keys: {e}")
                time.sleep(args.poll_interval)
                continue
            if len(finished) >= len(pool_keys):
                print(f"Round {trigger_round} complete. Starting from round {start_round}.")
                break
            time.sleep(args.poll_interval)
        else:
            print(f"Round {trigger_round} timed out. Exiting.")
            sys.exit(1)

    for round_n in range(start_round, args.max_rounds + 1):
        pool.clear_finished_pool_keys(entity, project, round_n)
        finished = pool.read_finished_pool_keys_merged(entity, project, round_n)
        if len(finished) >= len(pool_keys):
            print("All pool keys are finished. Exiting.")
            break

        active = [k for k in pool_keys if k not in finished]
        print(f"Round {round_n}: pool_keys {pool_keys} | active {active}")

        pool.write_round_trigger(entity, project, round_n, pool_keys, builtin_opponents, [])

        # Wait until all active pool keys are finished; meanwhile process key requests and assign one key per worker
        deadline = time.time() + args.timeout_hours * 3600
        last_log = 0
        while time.time() < deadline:
            try:
                # Process key requests: merge all artifact versions so we see every worker (no lost concurrent appends)
                requests = pool.read_key_requests_merged(entity, project, round_n)
                assignments = pool.read_key_assignments(entity, project, round_n)
                n_before = len(assignments)
                finished = pool.read_finished_pool_keys_merged(entity, project, round_n)
                assigned_keys = set(assignments.values())
                available = [k for k in pool_keys if k not in finished and k not in assigned_keys]
                for wid in set(requests):
                    if wid in assignments:
                        continue
                    if not available:
                        break
                    assignments[wid] = available.pop(0)
                if len(assignments) > n_before:
                    pool.write_key_assignments(entity, project, round_n, assignments)

                done_count = sum(1 for k in active if k in finished)
                if done_count >= len(active):
                    print(f"Round {round_n} complete: all {len(active)} pool keys finished.")
                    break
                now = time.time()
                if done_count > 0 and now - last_log >= 60:
                    print(f"Round {round_n}: {done_count}/{len(active)} pool keys finished, waiting...")
                    last_log = now
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(args.poll_interval)
        else:
            print(f"Round {round_n} timed out after {args.timeout_hours}h. Exiting.")
            break
    # Signal workers to stop (round=0, no pool_keys)
    try:
        pool.write_round_trigger(entity, project, 0, [], builtin_opponents, [])
    except Exception as e:
        print(f"Warning: could not write stop trigger: {e}")
    wandb.finish()
    print("Coordinator done.")


if __name__ == "__main__":
    main()
