#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scaling.dashboard_data import export_static_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static dashboard JSON")
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "dashboard" / "data"))
    args = parser.parse_args()
    count = export_static_dashboard(
        Path(args.runs_dir).resolve(), Path(args.output_dir).resolve()
    )
    print(f"Exported {count} run(s).")


if __name__ == "__main__":
    main()

