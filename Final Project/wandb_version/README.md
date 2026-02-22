# Round-based hockey training (wandb)

Multi-agent training where each "spec" is one algorithm + reward shape (e.g. `td3-default`, `td3-attack`). **Hyperparameters and run settings are defined in a single config file**; the coordinator and workers use it so the command line stays minimal (e.g. project/entity or just `ENTITY` in config/env).

## Config file

All run parameters and per-spec hyperparameters live in **`wandb_version/config/training.yaml`** (or another file passed as `--config`).

- **Top-level**: `project`, `entity` (optional; can use env `ENTITY` or CLI `--entity`), `builtin_opponents`.
- **`training`**: Global training args (e.g. `total_timesteps`, `num_envs`, `batch_size`, `policy_lr`, `gamma`, …). Same names as `train_parallel.py` CLI.
- **`specs`**: One entry per pool key. Each must have `algo` and `reward_mode`; any other keys override the global `training` section for that spec.

Example:

```yaml
project: hockey-rounds
entity: null
builtin_opponents: [weak, strong]
training:
  total_timesteps: 2000000
  num_envs: 8
  batch_size: 256
  policy_lr: 3.0e-4
  # ...
specs:
  td3-default:
    algo: td3
    reward_mode: default
  td3-attack:
    algo: td3
    reward_mode: attack
```

The **coordinator** reads `pool_keys` from the keys of `specs`, and uses `project` / `entity` / `builtin_opponents` from the config. **Workers** use the same config; when they run the training script they pass `--config` and `--pool_key`, and the trainer loads all hyperparameters from the config for that spec.

## Concepts

- **Pool key**: One agent in the pool, e.g. `td3-default` or `sac-attack`. Format: `{algo}-{reward_mode}`. Defined in config `specs`.
- **Pool keys**: Taken from config by the coordinator (keys of `specs`); workers learn them from the round trigger and claim one at a time.
- **Claiming**: Each worker polls the trigger, claims one available pool key, runs training (with hyperparams from config for that spec), then the key is marked finished. If no key is available and the round is not complete, the worker exits with an error (more workers than pool keys).
- **Round 1**: No pool opponents (fair start); round 2+ each run trains vs builtin + pool, then uploads the new best and marks the pool key finished (no 1v1 evaluation).

## Prerequisites

- Conda env `hockey` (or equivalent) with deps from the parent project.
- `wandb` and `PyYAML` installed (or `WANDB_API_KEY` set for wandb).
- Run all commands from the **Final Project** directory (parent of `wandb_version/`) so imports like `hockey_td3` resolve.

## Small test run (2 specs, 2 rounds)

Use two specs (e.g. `td3-default`, `td3-attack`), 2 rounds, and short timesteps so the test finishes quickly.

### 1. Set environment

**Option A – .env file (recommended)**

From the project root (e.g. `Final Project`):

```bash
cp wandb_version/.env.example .env
# Edit .env and set:
#   WANDB_API_KEY=your_key_here
#   ENTITY=your_wandb_username
#   WANDB_PROJECT=hockey-rounds   # or a new project name for a fresh run
```

Then:

```bash
cd "/path/to/Final Project"
conda activate hockey
```

The coordinator and worker load `.env` from the current directory (or the `wandb_version` directory) automatically. Existing shell env vars are not overridden.

**Option B – shell exports**

```bash
export WANDB_API_KEY=your_key_here
export ENTITY=your_wandb_username
export WANDB_PROJECT=hockey-rounds   # optional; default from config is hockey-rounds
cd "/path/to/Final Project"
conda activate hockey
```

### 2. Fresh wandb project

To use a **new project** (new wandb project id) instead of `hockey-rounds`:

- Set **`WANDB_PROJECT`** in `.env` (e.g. `WANDB_PROJECT=my-hockey-exp1`), or
- Set it in `wandb_version/config/training.yaml` under `project: my-hockey-exp1`, or
- Pass `--project my-hockey-exp1` to the coordinator and every worker.

Precedence: CLI `--project` > config `project` > env `WANDB_PROJECT` > default `hockey-rounds`. Coordinator and workers must use the same project so they see the same round trigger and pool.

### 3. Edit config (optional)

Edit `wandb_version/config/training.yaml`: set `entity` and/or `project` if you prefer (or use .env / CLI). Optionally set `total_timesteps` under `training` for a shorter test.

### 4. Terminal 1 – Coordinator

Uses the config file for pool_keys (from `specs`), project, entity, and builtin_opponents. Minimal run:

```bash
python wandb_version/coordinator.py --entity $ENTITY --max_rounds 2 --poll_interval 15 --timeout_hours 0.25
```

If you use a `.env` with `ENTITY` (and optionally `WANDB_PROJECT`), you can run:

```bash
python wandb_version/coordinator.py --max_rounds 2 --poll_interval 15 --timeout_hours 0.25
```

Override config path: `--config /path/to/training.yaml`.

### 5. Terminal 2 – Single worker (claims one key per round)

Workers use the same config; hyperparameters are read from the config when the trainer runs with `--config` and `--pool_key`. Minimal:

```bash
python wandb_version/worker.py --entity $ENTITY --poll_interval 10
```

Optional: `--total_timesteps 5000` overrides the config for a quick test. With one worker and two pool keys, that worker runs both keys sequentially each round.

### 6. Run each agent in its own terminal (parallel)

Start the coordinator in terminal 1, then one worker per pool key. Each worker needs only entity (and optionally project); the rest comes from the config.

**Terminal 2 – Worker 1:**
```bash
cd "/path/to/Final Project"
conda activate hockey
# If using .env, no exports needed:
python wandb_version/worker.py --poll_interval 10
```

**Terminal 3 – Worker 2:** same as terminal 2 (or use `export ENTITY=...` / `WANDB_PROJECT=...` if not using .env).

Start the coordinator first, then both workers. Each worker claims one pool key per round; the trainer loads that spec’s hyperparameters from the config. If you start **more workers than pool keys**, the extra worker exits with: `No available pool key to claim. You have more workers than pool keys; this worker is not needed.`

## Features

- **Early stopping**: If the evaluation win rate reaches 100%, training stops early (logged as `train/early_stop` in wandb).

## What you’ll see

- **Round 1**: Each run logs `Round 1: no previous best for {pool_key}; uploaded and marked finished.` Training vs builtin only; uploads the first best for that pool key.
- **Round 2+**: Each run downloads the pool, trains vs weak + strong + pool agents. After training: uploads the new best and marks the pool key finished (no 1v1 evaluation).

## Files

- **`config/training.yaml`** – Single source of run and hyperparameters: `project`, `entity`, `builtin_opponents`, global `training` section, and `specs` (one per pool key: `algo`, `reward_mode`, optional overrides).
- **`config_loader.py`** – Loads the YAML config and merges global + per-spec args for the trainer.
- **`train_parallel.py`** – Training script. With `--config` and `--pool_key`, loads all hyperparameters from the config for that spec; round-based pool keys come from the coordinator trigger.
- **`coordinator.py`** – Reads config for pool_keys (`specs` keys), project, entity, builtin_opponents. Writes round trigger, clears finished list each round, waits for all pool keys to finish, then next round.
- **`worker.py`** – Reads config for entity/project. Polls trigger, claims one pool key, runs `train_parallel.py --config ... --pool_key ...`. Errors if more workers than pool keys.
- **`wandb_pool.py`** – Pool download/upload, round trigger, finished and claimed pool keys.
- **`load_env.py`** – Loads `.env` from cwd or script dir into `os.environ` (no python-dotenv required).
- **`.env.example`** – Template for `.env`; copy to `.env` and set `WANDB_API_KEY`, `ENTITY`, `WANDB_PROJECT`. Do not commit `.env`.
