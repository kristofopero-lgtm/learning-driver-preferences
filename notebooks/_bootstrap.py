
# notebooks/_bootstrap.py
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent  # parent of notebooks/

# Put the repo root and src on sys.path. With __init__.py in scripts/,
# importing "scripts.define_cutoffs" will now work.
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)
