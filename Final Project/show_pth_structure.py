#!/usr/bin/env python3
"""
Show the structure of PyTorch .pth checkpoint files.
Displays top-level keys and, for state_dicts, parameter names with tensor shapes/dtypes.
"""

import argparse
from pathlib import Path

import torch


def describe_tensor(t):
    """Return a short description of a tensor (shape, dtype)."""
    if not isinstance(t, torch.Tensor):
        return str(type(t).__name__)
    return f"Tensor{tuple(t.shape)} {t.dtype}"


def describe_value(v, max_nested=2, indent=0):
    """Describe a value: dict → keys and recurse; tensor → shape; else → type."""
    prefix = "  " * indent
    if isinstance(v, dict):
        lines = [f"{prefix}{{  # dict, {len(v)} keys"]
        for k in sorted(v.keys()):
            sub = v[k]
            if isinstance(sub, torch.Tensor):
                lines.append(f"{prefix}  {k!r}: {describe_tensor(sub)}")
            elif isinstance(sub, dict) and indent < max_nested:
                lines.append(f"{prefix}  {k!r}:")
                lines.append(describe_value(sub, max_nested=max_nested, indent=indent + 2))
            else:
                lines.append(f"{prefix}  {k!r}: {type(sub).__name__}")
        lines.append(f"{prefix}}}")
        return "\n".join(lines)
    if isinstance(v, torch.Tensor):
        return f"{prefix}{describe_tensor(v)}"
    return f"{prefix}{type(v).__name__} = {repr(v)[:80]}"


def show_structure(path, device="cpu", weights_only=False, verbose=False):
    """Load a .pth file and print its structure."""
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        return
    if path.suffix.lower() != ".pth":
        print(f"Not a .pth file: {path}")
        return

    try:
        data = torch.load(path, map_location=device, weights_only=weights_only)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return

    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"{'='*60}")

    if not isinstance(data, dict):
        print(f"Top-level type: {type(data).__name__}")
        print(describe_value(data, max_nested=3))
        return

    print(f"Top-level keys ({len(data)}): {sorted(data.keys())}\n")
    for key in sorted(data.keys()):
        v = data[key]
        print(f"  [{key}]")
        if isinstance(v, dict):
            if verbose:
                print(describe_value(v, max_nested=2, indent=2))
            else:
                # Compact: list param names and shapes
                for k, t in sorted(v.items()):
                    if isinstance(t, torch.Tensor):
                        print(f"    {k}: {describe_tensor(t)}")
                    else:
                        print(f"    {k}: {type(t).__name__}")
        elif isinstance(v, torch.Tensor):
            print(f"    {describe_tensor(v)}")
        else:
            print(f"    {type(v).__name__} = {repr(v)[:100]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Show structure of PyTorch .pth checkpoint files")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to .pth files or directories to search for .pth files",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load tensors onto (default: cpu)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Use weights_only=True when loading (safer for untrusted files; may fail on older checkpoints)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show nested dict structure in tree form",
    )
    args = parser.parse_args()

    files = []
    for p in args.paths:
        path = Path(p)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.pth"))
        else:
            print(f"Not found: {path}")

    if not files:
        print("No .pth files found.")
        return

    for f in sorted(files):
        show_structure(f, device=args.device, weights_only=args.weights_only, verbose=args.verbose)


if __name__ == "__main__":
    main()
