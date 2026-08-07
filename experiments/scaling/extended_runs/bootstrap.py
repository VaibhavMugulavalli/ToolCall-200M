from __future__ import annotations

import sys
from pathlib import Path


EXTENDED_RUNS_ROOT = Path(__file__).resolve().parent
SCALING_PARENT = EXTENDED_RUNS_ROOT.parent
SCALING_RUNS_ROOT = SCALING_PARENT / "scaling_runs"


def ensure_scaling_runs_importable() -> Path:
    package = SCALING_RUNS_ROOT / "scaling" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(
            "The sibling scaling_runs package is missing. Expected "
            f"{package}. Put extended_runs next to scaling_runs, or use the "
            "complete Colab bundle."
        )
    value = str(SCALING_RUNS_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    return SCALING_RUNS_ROOT

