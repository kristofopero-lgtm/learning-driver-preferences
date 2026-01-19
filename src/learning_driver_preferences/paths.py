from pathlib import Path
import os

def _resolve_base_dir() -> Path:
    # 1) explicit override (useful in notebooks)
    env = os.getenv("LDP_BASE_DIR")
    if env:
        return Path(env).resolve()

    # 2) infer from this file's location
    here = Path(__file__).resolve()  # .../src/learning_driver_preferences/paths.py
    candidates = [
        here.parents[2],  # .../<project root>
        here.parents[3],  # in case your structure is deeper for some reason
    ]
    for p in candidates:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p

    # 3) last resort: two levels up
    return here.parents[2]

BASE_DIR      = _resolve_base_dir()
DATA_DIR      = BASE_DIR / "data"
REQUESTS_DIR  = DATA_DIR / "requests"
RESPONSES_DIR = DATA_DIR / "responses"
OUTPUT        = DATA_DIR / "output"

OUTPUT.mkdir(parents=True, exist_ok=True)
