#!/usr/bin/env python3
"""Standalone data-sync check, meant to be scheduled independently of the
running web server (cron on Linux/macOS, Task Scheduler on Windows - see
README.md for exact setup). It does NOT run inside run.py / the FastAPI
process; you run it yourself, on whatever cadence you choose.

    python sync_data.py                 # check for new ATP data; retrain (fast) if found
    python sync_data.py --check-only    # just check and report, never retrain
    python sync_data.py --mode tuned    # use tuned-mode retraining when new data is found

Uses the same venv as run.py (creating/updating it if needed), so it never
touches the system Python.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.launcher_common import bootstrap_venv

bootstrap_venv(Path(__file__).resolve())


def main() -> None:
    import backend.ml.sync as data_sync

    parser = argparse.ArgumentParser(description="Check tennis-data.co.uk for new data and retrain if found.")
    parser.add_argument("--mode", choices=["fast", "tuned"], default="fast",
                         help="Retraining mode to use if new data is found (default: fast).")
    parser.add_argument("--check-only", action="store_true",
                         help="Only check for new data; never retrain.")
    args = parser.parse_args()

    if args.check_only:
        changed, years = data_sync.fetch_and_check_updates(progress=print)
        if changed:
            print(f"New data available for season(s): {years}")
        else:
            print("No new data available.")
        return

    meta = data_sync.sync_and_retrain(mode=args.mode, progress=print)
    if meta is None:
        print("No new data - nothing to retrain.")
    else:
        print(f"Retrained model {meta.version_id} (mode={meta.mode}, test_acc={meta.test_accuracy:.3f})")


if __name__ == "__main__":
    main()
