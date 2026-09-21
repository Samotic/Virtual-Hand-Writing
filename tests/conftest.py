import sys
from pathlib import Path

# The project is a flat set of modules, so put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
