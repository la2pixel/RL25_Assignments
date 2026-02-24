#!/usr/bin/env python3
"""
Remap TD3 checkpoint state_dict keys from layers.0 / layers.1 / out to layer1 / layer2 / layer3
so they load into the current hockey_td3.py Actor and Critic.

Only processes .pth files that have "td3" in the filename (case-insensitive).
Usage:
    python remap_td3_checkpoints.py <folder>
    python remap_td3_checkpoints.py checkpoints --dry-run
"""

import argparse
from pathlib import Path

import torch

# Map old key prefix -> new key prefix for each state_dict (actor, critic1, critic2)
KEY_MAP = [
    ("layers.0.", "layer1."),
    ("layers.1.", "layer2."),
    ("out.", "layer3."),
]


def remap_state_dict(state_dict):
    """Remap keys from layers.0/layers.1/out to layer1/layer2/layer3. Returns new dict."""
    new_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for old_prefix, new_prefix in KEY_MAP:
            if k.startswith(old_prefix):
                new_key = new_prefix + k[len(old_prefix) :]
                break
        new_dict[new_key] = v
    return new_dict


def needs_remap(state_dict):
    """True if this state_dict uses the old naming (layers.0, out)."""
    keys = set(state_dict.keys())
    return any(k.startswith("layers.0.") or k.startswith("out.") for k in keys)


def process_checkpoint(path, dry_run=False):
    """Load checkpoint, remap actor/critic1/critic2 if needed, save back (unless dry_run)."""
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        return False, "not a dict checkpoint"

    changed = False
    for key in ("actor", "critic1", "critic2"):
        if key not in ckpt or not isinstance(ckpt[key], dict):
            continue
        if needs_remap(ckpt[key]):
            ckpt[key] = remap_state_dict(ckpt[key])
            changed = True

    if not changed:
        return False, "no remap needed"

    if dry_run:
        return True, "would remap (dry-run)"
    torch.save(ckpt, path)
    return True, "remapped and saved"


def main():
    parser = argparse.ArgumentParser(
        description="Remap TD3 .pth files (layers.0/1, out -> layer1/2/3) in a folder."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder to search for td3 .pth files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be done, do not write files",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        return

    # All .pth files under folder whose name contains "td3"
    pth_files = [
        f
        for f in folder.rglob("*.pth")
        if "td3" in f.name.lower()
    ]
    if not pth_files:
        print(f"No .pth files with 'td3' in the name under {folder}")
        return

    print(f"Found {len(pth_files)} td3 .pth file(s).")
    for f in sorted(pth_files):
        ok, msg = process_checkpoint(f, dry_run=args.dry_run)
        status = msg if ok else f"skip: {msg}"
        print(f"  {f.name}: {status}")


if __name__ == "__main__":
    main()
