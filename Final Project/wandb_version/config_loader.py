"""
Load training config YAML: global training params + specs (list of pool keys).
Used by coordinator (pool_keys, project, entity, builtin_opponents) and workers/train_parallel (full args for a spec).

Config shape:
  specs: list of pool keys in format {algo}-{reward_mode} (e.g. td3-default, sac-attack). Algo and reward_mode are derived from the string.
  dedicated_specs: boolean. When true, all workers must run with --algo and --reward_mode.
"""

import os


def load_config(path):
    """Load YAML config. Returns dict with keys: project, entity, training, specs (list), dedicated_specs (bool)."""
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


VALID_ALGOS = ("td3", "sac")
VALID_REWARD_MODES = ("default", "attack", "defense", "proven")


def parse_pool_key(pool_key):
    """Derive (algo, reward_mode) from a pool key string (e.g. sac-attack -> ('sac', 'attack')). Raises on invalid format or unknown algo/reward_mode."""
    s = (pool_key or "").strip()
    if not s or "-" not in s:
        raise ValueError(f"Pool key must be in format algo-reward_mode, got: {pool_key!r}")
    algo, reward_mode = s.split("-", 1)
    algo, reward_mode = algo.strip().lower(), reward_mode.strip().lower()
    if algo not in VALID_ALGOS:
        raise ValueError(f"Unknown algo {algo!r} in pool key {pool_key!r}. Valid: {VALID_ALGOS}")
    if reward_mode not in VALID_REWARD_MODES:
        raise ValueError(f"Unknown reward_mode {reward_mode!r} in pool key {pool_key!r}. Valid: {VALID_REWARD_MODES}")
    return algo, reward_mode


def get_pool_keys_from_config(config_path):
    """Return list of pool keys from config. specs must be a list of pool key names (algo-reward_mode)."""
    data = load_config(config_path)
    specs = data.get("specs")
    if specs is None:
        raise ValueError("Config must have a 'specs' section (list of pool keys, e.g. [td3-default, sac-attack, ...])")
    if isinstance(specs, list):
        return [str(k).strip() for k in specs if k]
    if isinstance(specs, dict):
        return list(specs.keys())
    raise ValueError("Config 'specs' must be a list of pool keys or a dict (legacy)")


def get_merged_training_args(config_path, pool_key):
    """
    Return a flat dict of training arguments for the given pool key.
    Algo and reward_mode are derived from the pool key (format: algo-reward_mode). Merged with training section.
    """
    data = load_config(config_path)
    pool_keys = get_pool_keys_from_config(config_path)
    if pool_key not in pool_keys:
        raise ValueError(f"Pool key '{pool_key}' not in config specs. Available: {pool_keys}")
    algo, reward_mode = parse_pool_key(pool_key)
    training = dict(data.get("training") or {})
    return {**training, "algo": algo, "reward_mode": reward_mode}


def apply_config_to_args(args, config_path, pool_key):
    """
    Apply config (training + algo/reward_mode from pool_key) to an argparse Namespace.
    Only sets attributes that exist on args and are present in the merged config.
    """
    merged = get_merged_training_args(config_path, pool_key)
    for key, value in merged.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args
