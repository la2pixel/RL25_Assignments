"""
One-off script to set the curated "finished" list for a round when the merged artifact has stale entries
(e.g. td3-default from a deleted run). Creates artifact finished-pool-round-{N}-curated. Coordinator and
workers will use this list instead of the merged finished-pool-round-{N}.

Example (round 2, 6 keys actually finished; td3-default and sac-default still to run):
  python set_finished_curated.py --entity meloneneis --project hockey-training --round 2 \\
    td3-attack td3-proven td3-defense sac-attack sac-proven sac-defense
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_env
load_env.load_dotenv()
import wandb_pool as pool


def main():
    parser = argparse.ArgumentParser(description="Write curated finished list for a round.")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("keys", nargs="+", help="Pool keys to mark as finished for this round (e.g. td3-attack sac-attack ...)")
    args = parser.parse_args()
    pool.write_finished_pool_keys_curated(args.entity, args.project, args.round, args.keys)
    print(f"Created finished-pool-round-{args.round}-curated with keys: {args.keys}")


if __name__ == "__main__":
    main()
