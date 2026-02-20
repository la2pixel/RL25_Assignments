"""
Load training config YAML: global training params + per-spec (pool key) overrides.
Used by coordinator (pool_keys, project, entity, builtin_opponents) and workers/train_parallel (full args for a spec).
"""

import os


def load_config(path):
    """Load YAML config. Returns dict with keys: project, entity, training, specs, builtin_opponents."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required for config. pip install PyYAML")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError("Config file is empty")
    return data


def get_pool_keys_from_config(config_path):
    """Return list of pool keys (spec names) from config specs section."""
    data = load_config(config_path)
    specs = data.get("specs") or {}
    if not specs:
        raise ValueError("Config must have a 'specs' section with at least one spec (pool key -> algo, reward_mode)")
    return list(specs.keys())


def get_merged_training_args(config_path, pool_key):
    """
    Return a flat dict of training arguments for the given pool key.
    Merges: training (global) + specs[pool_key] (overrides). Keys match train_parallel argparse names.
    """
    data = load_config(config_path)
    specs = data.get("specs") or {}
    if pool_key not in specs:
        raise ValueError(f"Pool key '{pool_key}' not found in config specs. Available: {list(specs.keys())}")
    spec = dict(specs[pool_key])
    if "algo" not in spec or "reward_mode" not in spec:
        raise ValueError(f"Spec '{pool_key}' must have 'algo' and 'reward_mode'")
    training = dict(data.get("training") or {})
    # Merge: training first, then spec overrides (spec can override hyperparams, not just algo/reward_mode)
    merged = {**training, **spec}
    return merged


def apply_config_to_args(args, config_path, pool_key):
    """
    Apply config (training + spec[pool_key]) to an argparse Namespace.
    Only sets attributes that exist on args and are present in the merged config.
    """
    merged = get_merged_training_args(config_path, pool_key)
    for key, value in merged.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args
