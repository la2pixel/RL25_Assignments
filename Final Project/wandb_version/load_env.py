"""
Load .env file into os.environ (simple parser, no python-dotenv required).
Looks for .env in script_dir and cwd; first found wins. Does not override existing env vars.
"""

import os


def load_dotenv(extra_dirs=None):
    """Load .env from script dir, then cwd, then extra_dirs. Existing env vars are kept."""
    dirs = list(extra_dirs or [])
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(script_dir)
        dirs.insert(0, script_dir)
        dirs.insert(0, parent)
    except Exception:
        pass
    dirs.insert(0, os.getcwd())
    for d in dirs:
        path = os.path.join(d, ".env")
        if os.path.isfile(path):
            _load_file(path)
            return path
    return None


def _load_file(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
