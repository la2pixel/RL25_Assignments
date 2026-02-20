"""
Shared wandb helpers for round-based training: pool (improving opponents), round trigger, finished pool keys, claimed keys.
Pool key = identifier for one agent in the pool (e.g. td3-default = algo + reward_mode).
Only the coordinator sets pool_keys; workers claim one key per round from the coordinator's list.
"""

import os
import json
import tempfile

# Reward shape name -> reward_mode for env
REWARD_MAP = {
    "default": "default",
    "attack": "attack",
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
    """Return (algo, reward_mode) for a pool key (e.g. td3-default -> ('td3', 'default'))."""
    if "-" in pool_key:
        algo, reward = pool_key.split("-", 1)
        return algo.strip().lower(), REWARD_MAP.get(reward.strip().lower(), "default")
    return "td3", "default"


def download_pool(entity, project, pool_keys, cache_dir=None):
    """
    Download best artifacts for the given pool keys (improving opponents).
    Returns list of (pool_key, local_path, algo) in pool_keys order.
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
        except Exception:
            pass
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


def read_worker_run_registry(entity, project, round_n):
    """Return dict {worker_id: run_path} by merging all versions of round-worker-runs-{n}.
    Coordinator uses this to check wandb run state (running/crashed/finished) for crash recovery."""
    api = _api(entity, project)
    art_name = _worker_run_registry_artifact_name(round_n)
    merged = {}
    try:
        for art in api.artifacts(type_name="config", name=f"{entity}/{project}/{art_name}", per_page=50):
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
            for k, v in d.items():
                merged[k] = v
    except Exception:
        pass
    return merged


def register_worker_run(entity, project, round_n, worker_id, run_path):
    """Training run calls this after wandb.init(): register worker_id -> run_path so coordinator can check run state."""
    import wandb
    merged = read_worker_run_registry(entity, project, round_n)
    merged[worker_id] = run_path
    run = wandb.run
    if run is None:
        run = wandb.init(project=project, entity=entity, job_type="worker")
    art = wandb.Artifact(_worker_run_registry_artifact_name(round_n), type="config")
    art.metadata["registry"] = dict(merged)
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
