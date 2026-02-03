"""
Bootstrap import paths for notebooks.

This ensures that:
- learning_driver_preferences (under src/) can be imported
- scripts/ modules at project root can also be imported

Usage in notebook:
    %run -i _bootstrap.py
"""

import sys
from pathlib import Path

# Adjust this depending on where your notebooks folder is
# → It should point to the PROJECT ROOT (the folder that contains src/ and scripts/)
NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent    # If notebooks/ is directly under project root
# If notebooks are deeper, adjust: PROJECT_ROOT = NOTEBOOK_DIR.parents[1]

SRC_DIR = PROJECT_ROOT / "src"

# Add project root (so "scripts" can be imported)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add src (so "learning_driver_preferences" can be imported)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("BOOTSTRAP: added to sys.path:")
print(" -", PROJECT_ROOT)
print(" -", SRC_DIR)
