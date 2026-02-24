"""
Shared wandb helpers for round-based training: pool (improving opponents), round trigger, finished pool keys, claimed keys.
Pool key = identifier for one agent in the pool (e.g. td3-default = algo + reward_mode).
Only the coordinator sets pool_keys; workers claim one key per round from the coordinator's list.
"""

import os
import json
import tempfile

# Reward shape name -> reward_mode for env (must match pool key suffix)
REWARD_MAP = {
    "default": "default",
    "attack": "attack",
    "defense": "defense",
    "proven": "proven",
}


def _api(entity, project):
    try:
        import wandb
        return wandb.Api(timeout=60)
    except ImportError:
        raise RuntimeError("wandb is required for round-based training. pip install wandb")


def _artifact_name(pool_key):
    return f"{pool_key}-best"


def _algo_from_pool_key(pool_key):
    return "td3" if pool_key.startswith("td3-") else "sac"


def parse_pool_key(pool_key):
    """Return (algo, reward_mode) for a pool key (e.g. td3-default -> ('td3', 'default')). Raises if format or reward mode is invalid."""
    if not pool_key or "-" not in pool_key:
        raise ValueError(f"Pool key must be in format algo-reward_mode (e.g. td3-default, sac-attack), got: {pool_key!r}")
    algo, reward = pool_key.strip().split("-", 1)
    algo, reward = algo.strip().lower(), reward.strip().lower()
    if reward not in REWARD_MAP:
        raise ValueError(
            f"Unknown reward_mode {reward!r} in pool key {pool_key!r}. Valid: {list(REWARD_MAP.keys())}."
        )
    return algo, REWARD_MAP[reward]


def download_pool(entity, project, pool_keys, cache_dir=None):
    """
    Download best artifacts for the given pool keys (improving opponents).
    Returns list of (pool_key, local_path, algo) in pool_keys order.
    Raises on download failure for any key (no silent skip).
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "pool_cache")
    os.makedirs(cache_dir, exist_ok=True)
    api = _api(entity, project)
    result = []
    for key in pool_keys:
        name = f"{entity}/{project}/{_artifact_name(key)}:best"
        try:
            art = api.artifact(name)
            root = art.download(root=os.path.join(cache_dir, key.replace("-", "_")))
            for f in os.listdir(root):
                if f.endswith(".pth"):
                    result.append((key, os.path.join(root, f), _algo_from_pool_key(key)))
                    break
            else:
                result.append((key, root, _algo_from_pool_key(key)))
        except Exception as e:
            err_msg = str(e).lower()
            # Round 1 or new spec: artifact may not exist yet; skip only for "not found" style errors
            if "not found" in err_msg or "does not exist" in err_msg or "no such" in err_msg:
                continue
            raise RuntimeError(
                f"Failed to download pool artifact for key {key!r} ({name}). Fix artifact or network."
            ) from e
    return result


def upload_best(entity, project, pool_key, checkpoint_path):
    """Upload checkpoint as the best for this pool key (overwrites alias 'best')."""
    import wandb
    name = _artifact_name(pool_key)
    art = wandb.Artifact(name=name, type="model")
    art.add_file(checkpoint_path, name=os.path.basename(checkpoint_path))
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="upload")
    run.log_artifact(art, aliases=["best"])


def read_round_trigger(entity, project):
    """Read round trigger. Returns dict with round, pool_keys, builtin_opponents, finished_pool_keys."""
    api = _api(entity, project)
    try:
        art = api.artifact(f"{entity}/{project}/round-trigger:latest")
        d = art.metadata if hasattr(art, "metadata") and art.metadata else {}
        pool_keys = d.get("pool_keys", [])
        builtin = d.get("builtin_opponents", ["weak", "strong"])
        finished = d.get("finished_pool_keys", [])
        return {
            "round": int(d.get("round", 0)),
            "pool_keys": list(pool_keys) if isinstance(pool_keys, list) else [],
            "builtin_opponents": list(builtin) if isinstance(builtin, list) else ["weak", "strong"],
            "finished_pool_keys": list(finished) if isinstance(finished, list) else [],
        }
    except Exception:
        return {"round": 0, "pool_keys": [], "builtin_opponents": ["weak", "strong"], "finished_pool_keys": []}


def write_round_trigger(entity, project, round_n, pool_keys, builtin_opponents=None, finished_pool_keys=None):
    """Write round trigger so workers see round, pool_keys, builtin opponents, and finished_pool_keys."""
    import wandb
    if finished_pool_keys is None:
        finished_pool_keys = []
    if builtin_opponents is None:
        builtin_opponents = ["weak", "strong"]
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="coordinator")
    art = wandb.Artifact("round-trigger", type="config")
    art.metadata["round"] = round_n
    art.metadata["pool_keys"] = list(pool_keys)
    art.metadata["builtin_opponents"] = list(builtin_opponents)
    art.metadata["finished_pool_keys"] = list(finished_pool_keys)
    run.log_artifact(art, aliases=["latest"])


def _finished_pool_artifact_name(round_n):
    """Per-round finished list so concurrent worker writes don't overwrite each other."""
    return f"finished-pool-round-{round_n}"


def _resolve_round(entity, project, round_n):
    """If round_n is None, read current round from round-trigger; otherwise return round_n."""
    if round_n is not None:
        return round_n
    trigger = read_round_trigger(entity, project)
    return int(trigger.get("round", 0))


def clear_finished_pool_keys(entity, project, round_n):
    """Clear the finished list for this round (coordinator calls at the start of each round)."""
    import wandb
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="coordinator")
    name = _finished_pool_artifact_name(round_n)
    art = wandb.Artifact(name, type="config")
    art.metadata["pool_keys"] = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([], f)
        f.flush()
        art.add_file(f.name, "finished_pool_keys.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])


def _pool_keys_from_finished_artifact(art):
    """Extract pool_keys list from a finished-pool-round artifact."""
    if hasattr(art, "metadata") and art.metadata and "pool_keys" in art.metadata:
        return list(art.metadata["pool_keys"])
    try:
        root = art.download()
        path = os.path.join(root, "finished_pool_keys.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def read_finished_pool_keys(entity, project, round_n=None):
    """Read the list of finished pool keys for this round (single version). If round_n is omitted, uses current round from round-trigger."""
    round_n = _resolve_round(entity, project, round_n)
    api = _api(entity, project)
    try:
        art = api.artifact(f"{entity}/{project}/{_finished_pool_artifact_name(round_n)}:latest")
        return _pool_keys_from_finished_artifact(art)
    except Exception:
        pass
    return []


def read_finished_pool_keys_merged(entity, project, round_n=None):
    """Read merged list of all finished pool keys for this round (all artifact versions).
    If round_n is omitted, uses current round from round-trigger. Use in coordinator so concurrent worker mark_pool_key_finished writes are all seen."""
    round_n = _resolve_round(entity, project, round_n)
    api = _api(entity, project)
    name = f"{entity}/{project}/{_finished_pool_artifact_name(round_n)}"
    seen = set()
    result = []
    try:
        for art in api.artifacts(type_name="config", name=name, per_page=20):
            for k in _pool_keys_from_finished_artifact(art):
                if k not in seen:
                    seen.add(k)
                    result.append(k)
    except Exception:
        pass
    return result


def _finished_pool_curated_artifact_name(round_n):
    return f"finished-pool-round-{round_n}-curated"


def read_finished_pool_keys_curated(entity, project, round_n=None):
    """If someone created a curated artifact (e.g. finished-pool-round-2-curated) with the correct list of
    finished keys for this round, return that list. Otherwise return None. Use this to fix rounds where
    the merge has stale entries (e.g. from a deleted run)."""
    round_n = _resolve_round(entity, project, round_n)
    api = _api(entity, project)
    name = _finished_pool_curated_artifact_name(round_n)
    try:
        art = api.artifact(f"{entity}/{project}/{name}:latest")
        return _pool_keys_from_finished_artifact(art)
    except Exception:
        pass
    return None


def write_finished_pool_keys_curated(entity, project, round_n, pool_keys):
    """Write a curated finished list for this round (artifact finished-pool-round-{n}-curated).
    Use this to fix a round where the merge has stale keys (e.g. td3-default from a deleted run).
    Coordinator and workers will use this list instead of the merged finished-pool-round-{n} when present."""
    import wandb
    round_n = _resolve_round(entity, project, round_n)
    name = _finished_pool_curated_artifact_name(round_n)
    keys = list(pool_keys)
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="curated", name="curated-finished")
    art = wandb.Artifact(name, type="config")
    art.metadata["pool_keys"] = keys
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(keys, f)
        f.flush()
        art.add_file(f.name, "finished_pool_keys.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])


def read_finished_pool_keys_merged_filtered(entity, project, round_n=None):
    """Same as read_finished_pool_keys_merged, but:
    1) If a curated artifact exists (finished-pool-round-{n}-curated), use that list.
    2) Else only count a key as finished if it has a registered run in round-worker-runs-{n} (pool_keys_by_worker).
       Avoids counting stale entries from deleted runs. If no registry data exists, returns the raw merge."""
    curated = read_finished_pool_keys_curated(entity, project, round_n)
    if curated is not None:
        return curated
    finished_raw = read_finished_pool_keys_merged(entity, project, round_n)
    keys_with_run = read_pool_keys_with_registered_run(entity, project, round_n)
    if keys_with_run:
        return [k for k in finished_raw if k in keys_with_run]
    return finished_raw


# Key assignment: workers request a key via round-requests-{N}; coordinator is sole writer of round-assignments-{N}
def _round_requests_artifact_name(round_n):
    return f"round-requests-{round_n}"


def _round_assignments_artifact_name(round_n):
    return f"round-assignments-{round_n}"


def append_key_request(entity, project, round_n, worker_id):
    """Worker calls: append my worker_id to the request list for this round (so coordinator can assign me a key)."""
    import wandb
    api = _api(entity, project)
    current = []
    try:
        art = api.artifact(f"{entity}/{project}/{_round_requests_artifact_name(round_n)}:latest")
        if hasattr(art, "metadata") and art.metadata and "worker_ids" in art.metadata:
            current = list(art.metadata["worker_ids"])
        else:
            root = art.download()
            path = os.path.join(root, "worker_ids.json")
            if os.path.isfile(path):
                with open(path) as f:
                    current = json.load(f)
    except Exception:
        pass
    if worker_id in current:
        return
    current = list(current) + [worker_id]
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="worker", name="worker")
    art = wandb.Artifact(_round_requests_artifact_name(round_n), type="config")
    art.metadata["worker_ids"] = current
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(current, f)
        f.flush()
        art.add_file(f.name, "worker_ids.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])


def _worker_ids_from_artifact(art):
    """Extract worker_ids list from a round-requests artifact (metadata or file)."""
    if hasattr(art, "metadata") and art.metadata and "worker_ids" in art.metadata:
        return list(art.metadata["worker_ids"])
    try:
        root = art.download()
        path = os.path.join(root, "worker_ids.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def read_key_requests(entity, project, round_n):
    """Return list of worker_ids that requested a key this round (single version, coordinator dedupes)."""
    api = _api(entity, project)
    try:
        art = api.artifact(f"{entity}/{project}/{_round_requests_artifact_name(round_n)}:latest")
        return _worker_ids_from_artifact(art)
    except Exception:
        pass
    return []


def read_key_requests_merged(entity, project, round_n):
    """Return merged list of all worker_ids from every version of round-requests-{n}.
    Use this in the coordinator so concurrent worker appends (different artifact versions) are all seen."""
    api = _api(entity, project)
    name = f"{entity}/{project}/{_round_requests_artifact_name(round_n)}"
    seen = set()
    result = []
    try:
        for art in api.artifacts(type_name="config", name=name, per_page=20):
            for wid in _worker_ids_from_artifact(art):
                if wid not in seen:
                    seen.add(wid)
                    result.append(wid)
    except Exception:
        pass
    return result


def read_key_assignments(entity, project, round_n):
    """Return dict {worker_id: pool_key} for this round. Only the coordinator writes this artifact."""
    api = _api(entity, project)
    try:
        art = api.artifact(f"{entity}/{project}/{_round_assignments_artifact_name(round_n)}:latest")
        if hasattr(art, "metadata") and art.metadata and "assignments" in art.metadata:
            d = art.metadata["assignments"]
            return dict(d) if isinstance(d, dict) else {}
        root = art.download()
        path = os.path.join(root, "assignments.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def write_key_assignments(entity, project, round_n, assignments):
    """Coordinator only: write the assignments dict {worker_id: pool_key} for this round."""
    import wandb
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="coordinator")
    art = wandb.Artifact(_round_assignments_artifact_name(round_n), type="config")
    art.metadata["assignments"] = dict(assignments)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(assignments, f)
        f.flush()
        art.add_file(f.name, "assignments.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])


def _round_assignment_timestamps_artifact_name(round_n):
    return f"round-assignment-timestamps-{round_n}"


def read_assignment_timestamps(entity, project, round_n):
    """Return dict {worker_id: unix_timestamp} for when each worker was assigned. Coordinator uses this to free keys after timeout (crashed workers)."""
    api = _api(entity, project)
    try:
        art = api.artifact(f"{entity}/{project}/{_round_assignment_timestamps_artifact_name(round_n)}:latest")
        if hasattr(art, "metadata") and art.metadata and "timestamps" in art.metadata:
            d = art.metadata["timestamps"]
            return dict(d) if isinstance(d, dict) else {}
        root = art.download()
        path = os.path.join(root, "timestamps.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def write_assignment_timestamps(entity, project, round_n, timestamps):
    """Coordinator only: write assignment timestamps {worker_id: unix_float} for crash recovery."""
    import wandb
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="coordinator")
    name = _round_assignment_timestamps_artifact_name(round_n)
    art = wandb.Artifact(name, type="config")
    art.metadata["timestamps"] = dict(timestamps)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(timestamps, f)
        f.flush()
        art.add_file(f.name, "timestamps.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])


def _worker_run_registry_artifact_name(round_n):
    return f"round-worker-runs-{round_n}"


def _registry_from_artifact(art):
    """Extract {worker_id: run_path} from a round-worker-runs artifact (supports legacy string or dict value)."""
    d = {}
    if hasattr(art, "metadata") and art.metadata and "registry" in art.metadata:
        d = dict(art.metadata["registry"])
    else:
        try:
            root = art.download()
            path = os.path.join(root, "registry.json")
            if os.path.isfile(path):
                with open(path) as f:
                    d = json.load(f)
        except Exception:
            pass
    result = {}
    for k, v in d.items():
        if isinstance(v, dict) and "run_path" in v:
            result[k] = v["run_path"]
        else:
            result[k] = v
    return result


def _pool_keys_by_worker_from_artifact(art):
    """Extract {worker_id: pool_key} from artifact metadata if present."""
    if hasattr(art, "metadata") and art.metadata and "pool_keys_by_worker" in art.metadata:
        return dict(art.metadata["pool_keys_by_worker"])
    return {}


def read_worker_run_registry(entity, project, round_n):
    """Return dict {worker_id: run_path} by merging all versions of round-worker-runs-{n}.
    Coordinator uses this to check wandb run state (running/crashed/finished) for crash recovery."""
    round_n = _resolve_round(entity, project, round_n)
    api = _api(entity, project)
    art_name = _worker_run_registry_artifact_name(round_n)
    merged = {}
    try:
        for art in api.artifacts(type_name="config", name=f"{entity}/{project}/{art_name}", per_page=50):
            d = _registry_from_artifact(art)
            for k, v in d.items():
                merged[k] = v
    except Exception:
        pass
    return merged


def read_pool_keys_with_registered_run(entity, project, round_n=None):
    """Return set of pool keys that have at least one worker run registered for this round (from round-worker-runs-{n}).
    Used to filter finished list: only count a key as finished if it has a registered run (avoids stale finished entries)."""
    round_n = _resolve_round(entity, project, round_n)
    api = _api(entity, project)
    art_name = _worker_run_registry_artifact_name(round_n)
    keys = set()
    try:
        for art in api.artifacts(type_name="config", name=f"{entity}/{project}/{art_name}", per_page=50):
            for wid, pk in _pool_keys_by_worker_from_artifact(art).items():
                if pk:
                    keys.add(pk)
    except Exception:
        pass
    return keys


def read_pool_keys_by_worker_merged(entity, project, round_n=None):
    """Return {worker_id: pool_key} by merging all versions of round-worker-runs-{n}.
    Used by coordinator to map each registered run to its pool key for the three-list state."""
    round_n = _resolve_round(entity, project, round_n)
    api = _api(entity, project)
    art_name = _worker_run_registry_artifact_name(round_n)
    merged = {}
    try:
        for art in api.artifacts(type_name="config", name=f"{entity}/{project}/{art_name}", per_page=50):
            for wid, pk in _pool_keys_by_worker_from_artifact(art).items():
                if pk and wid not in merged:
                    merged[wid] = pk
    except Exception:
        pass
    return merged


def _round_pool_state_artifact_name(round_n):
    return f"round-{round_n}-pool-state"


def read_round_pool_state(entity, project, round_n=None):
    """Read the three-list state for this round (not_started, running, finished). Coordinator is sole writer.
    Returns (not_started_list, running_list, finished_list) or (None, None, None) if artifact missing."""
    round_n = _resolve_round(entity, project, round_n)
    api = _api(entity, project)
    name = _round_pool_state_artifact_name(round_n)
    try:
        art = api.artifact(f"{entity}/{project}/{name}:latest")
        if hasattr(art, "metadata") and art.metadata:
            m = art.metadata
            return (
                list(m.get("not_started", [])),
                list(m.get("running", [])),
                list(m.get("finished", [])),
            )
        root = art.download()
        path = os.path.join(root, "pool_state.json")
        if os.path.isfile(path):
            with open(path) as f:
                d = json.load(f)
            return (
                list(d.get("not_started", [])),
                list(d.get("running", [])),
                list(d.get("finished", [])),
            )
    except Exception:
        pass
    return (None, None, None)


def write_round_pool_state(entity, project, round_n, not_started, running, finished):
    """Coordinator only: write the three-list state for this round. Each spec appears in exactly one list."""
    import wandb
    round_n = _resolve_round(entity, project, round_n)
    name = _round_pool_state_artifact_name(round_n)
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="coordinator")
    art = wandb.Artifact(name, type="config")
    art.metadata["not_started"] = list(not_started)
    art.metadata["running"] = list(running)
    art.metadata["finished"] = list(finished)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"not_started": list(not_started), "running": list(running), "finished": list(finished)}, f)
        f.flush()
        art.add_file(f.name, "pool_state.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])


# Run states that mean the spec is still "running" (active). Crashed/killed/failed go back to not_started.
_ACTIVE_RUN_STATES = frozenset({"running", "pending"})


def compute_round_pool_state(entity, project, round_n, pool_keys):
    """Compute (not_started, running, finished) from run registry and run state. Each key in exactly one list.
    Crashed/killed/failed runs are treated as not_started so another worker can pick up the spec.
    For old rounds (no pool_key in registry), falls back to the old finished-pool-round-{n} artifact so progress is preserved."""
    round_n = _resolve_round(entity, project, round_n)
    registry = read_worker_run_registry(entity, project, round_n)
    pool_keys_by_worker = read_pool_keys_by_worker_merged(entity, project, round_n)
    # Read old finished list first so we don't treat already-finished keys as "running" when a later run registers
    old_finished_raw = read_finished_pool_keys_curated(entity, project, round_n) or read_finished_pool_keys_merged_filtered(entity, project, round_n)
    old_finished_set = set(old_finished_raw or [])
    finished_from_registry = set()
    running_set = set()
    for wid, pk in pool_keys_by_worker.items():
        run_path = registry.get(wid)
        state = get_run_state(run_path) if run_path else None
        if state == "finished":
            finished_from_registry.add(pk)
        elif state in _ACTIVE_RUN_STATES:
            if pk not in old_finished_set:
                running_set.add(pk)
    if not pool_keys_by_worker:
        # No registry data for this round → treat as old round: finished from old artifact, rest not_started
        finished = list(old_finished_raw) if old_finished_raw else []
        not_started = [k for k in pool_keys if k not in finished]
        return (not_started, [], finished)
    # Merge: finished = from registry + any from old artifact not in running (each key in exactly one list)
    finished_set = set(finished_from_registry)
    for k in (old_finished_raw or []):
        if k not in running_set:
            finished_set.add(k)
    running_set -= finished_set  # key can't be both finished and running
    not_started = [k for k in pool_keys if k not in finished_set and k not in running_set]
    return (not_started, sorted(running_set), sorted(finished_set))


def register_worker_run(entity, project, round_n, worker_id, run_path, pool_key=None):
    """Training run calls this after wandb.init(): register worker_id -> run_path so coordinator can check run state.
    If pool_key is provided, it is stored so coordinator can filter finished list to only keys with a registered run."""
    import wandb
    round_n = _resolve_round(entity, project, round_n)
    merged = read_worker_run_registry(entity, project, round_n)
    merged[worker_id] = run_path
    merged_pool_keys = {}
    try:
        for art in _api(entity, project).artifacts(type_name="config", name=f"{entity}/{project}/{_worker_run_registry_artifact_name(round_n)}", per_page=50):
            for wid, pk in _pool_keys_by_worker_from_artifact(art).items():
                if pk and wid not in merged_pool_keys:
                    merged_pool_keys[wid] = pk
    except Exception:
        pass
    if pool_key:
        merged_pool_keys[worker_id] = pool_key
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="worker")
    art = wandb.Artifact(_worker_run_registry_artifact_name(round_n), type="config")
    art.metadata["registry"] = dict(merged)
    art.metadata["pool_keys_by_worker"] = dict(merged_pool_keys)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(merged, f)
        f.flush()
        art.add_file(f.name, "registry.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])


def get_run_state(run_path):
    """Return run state string ('running', 'finished', 'crashed', 'killed', etc.) or None if unavailable."""
    if not run_path:
        return None
    try:
        api = _api(None, None)
        # run_path is entity/project/run_id
        run = api.run(run_path)
        return getattr(run, "state", None)
    except Exception:
        return None


def get_assigned_pool_key(entity, project, round_n, worker_id):
    """Return the pool_key assigned to this worker_id for this round, or None if not yet assigned."""
    return read_key_assignments(entity, project, round_n).get(worker_id)


def _round_claims_artifact_name(round_n):
    return f"round-claims-{round_n}"


def clear_claimed_pool_keys(entity, project, round_n):
    """Clear the claimed list for this round (coordinator calls at start of round so workers see no stale claims)."""
    import wandb
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="coordinator")
    art = wandb.Artifact(_round_claims_artifact_name(round_n), type="config")
    art.metadata["claimed"] = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([], f)
        f.flush()
        art.add_file(f.name, "claimed.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])


def read_claimed_pool_keys(entity, project, round_n):
    """Read the list of pool keys currently claimed (worker running) for this round."""
    api = _api(entity, project)
    try:
        art = api.artifact(f"{entity}/{project}/{_round_claims_artifact_name(round_n)}:latest")
        if hasattr(art, "metadata") and art.metadata and "claimed" in art.metadata:
            return list(art.metadata["claimed"])
        root = art.download()
        path = os.path.join(root, "claimed.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def claim_pool_key(entity, project, round_n, pool_key):
    """
    Mark a pool key as claimed for this round (best-effort; concurrent claims may race).
    Returns True if we recorded the claim.
    """
    import wandb
    api = _api(entity, project)
    current = []
    try:
        art = api.artifact(f"{entity}/{project}/{_round_claims_artifact_name(round_n)}:latest")
        if hasattr(art, "metadata") and art.metadata and "claimed" in art.metadata:
            current = list(art.metadata["claimed"])
        else:
            root = art.download()
            path = os.path.join(root, "claimed.json")
            if os.path.isfile(path):
                with open(path) as f:
                    current = json.load(f)
    except Exception:
        pass
    if pool_key in current:
        return True
    current = list(current) + [pool_key]
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="worker")
    art = wandb.Artifact(_round_claims_artifact_name(round_n), type="config")
    art.metadata["claimed"] = current
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(current, f)
        f.flush()
        art.add_file(f.name, "claimed.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])
    return True


def mark_pool_key_finished(entity, project, pool_key, round_n=None):
    """Append pool_key to the finished list for this round (read current, add if missing, write new version). If round_n is omitted, uses current round from round-trigger."""
    round_n = _resolve_round(entity, project, round_n)
    import wandb
    api = _api(entity, project)
    name = _finished_pool_artifact_name(round_n)
    current = []
    try:
        art = api.artifact(f"{entity}/{project}/{name}:latest")
        current = _pool_keys_from_finished_artifact(art)
    except Exception:
        pass
    if pool_key in current:
        return
    current = list(current) + [pool_key]
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="mark_finished", name="mark-finished")
    art = wandb.Artifact(name, type="config")
    art.metadata["pool_keys"] = current
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(current, f)
        f.flush()
        art.add_file(f.name, "finished_pool_keys.json")
    try:
        os.unlink(f.name)
    except Exception:
        pass
    run.log_artifact(art, aliases=["latest"])
