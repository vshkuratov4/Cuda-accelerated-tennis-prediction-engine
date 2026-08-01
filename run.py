#!/usr/bin/env python3
"""Single-command cross-platform launcher for the Tennis Match Predictor.

    python run.py                  # normal launch (prompts only if training is needed)
    python run.py --train fast     # force a fast (re)train, then serve
    python run.py --train tuned    # force a tuned (re)train, then serve
    python run.py --force-retrain  # retrain even if the dataset hasn't changed
    python run.py --no-train       # never train automatically; error if no model exists

On first run this creates an isolated .venv, installs dependencies into it, and
re-executes itself inside that venv - the system/global Python is never touched.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from backend.launcher_common import bootstrap_venv, get_lan_ip

ROOT = Path(__file__).resolve().parent


def _print_banner(port: int) -> None:
    lan_ip = get_lan_ip()
    lines = [
        "",
        "Tennis Match Predictor is running",
        "",
        f"  Local:   http://127.0.0.1:{port}",
        f"  Network: http://{lan_ip}:{port}",
        "",
        "Press Ctrl+C to stop.",
        "",
    ]
    width = max(len(line) for line in lines) + 4
    print("+" + "-" * width + "+")
    for line in lines:
        print("|" + line.ljust(width) + "|")
    print("+" + "-" * width + "+")


def _ensure_data() -> None:
    from backend.config import CLEANED_CSV
    from backend.ml import bootstrap as ml_bootstrap

    if CLEANED_CSV.exists():
        print(f"Using existing processed dataset: {CLEANED_CSV}")
        return
    print("No processed dataset found - running the data pipeline (download / merge / clean)...")
    ml_bootstrap.ensure_data(progress=print)


def _prompt_choice(prompt: str, options: dict, default: str) -> str:
    if not sys.stdin.isatty():
        return default
    try:
        choice = input(prompt).strip().lower()
    except EOFError:
        return default
    return options.get(choice, default)


def _ensure_model(args: argparse.Namespace) -> None:
    import pandas as pd

    from backend.config import CLEANED_CSV
    from backend.hardware import detect
    from backend.ml import bootstrap as ml_bootstrap
    from backend.ml import registry
    from backend.ml.estimate import estimate_training_time

    active = registry.get_active_meta()

    if args.no_train:
        if active is None:
            raise SystemExit("No trained model exists and --no-train was passed. Aborting.")
        return

    if args.train:
        print(f"--train {args.train} requested; training now...")
        ml_bootstrap.train_and_save(args.train, progress=print)
        return

    needs_training = active is None
    new_data = registry.has_new_data() if active is not None else False

    if not needs_training and not new_data and not args.force_retrain:
        print(
            f"Using existing model version {active['version_id']} "
            f"(mode={active['mode']}, test accuracy={active['test_accuracy']:.3f})"
        )
        return

    if new_data and not needs_training:
        print("New data detected since the active model was last trained.")

    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    hw = detect()
    estimate = estimate_training_time(df, hw)
    print(f"\nHardware detected: {estimate['hardware']}")
    print(f"  fast  training - estimated {estimate['fast_label']}")
    print(f"  tuned training - estimated {estimate['tuned_label']} (RFECV + hyperparameter search)")

    if needs_training:
        prompt = "\nNo trained model yet. Train now - [f]ast (default) or [t]uned? "
        default = "fast"
    else:
        prompt = "\nKeep existing model, or retrain? [k]eep (default) / [f]ast / [t]uned? "
        default = "keep"

    choice = _prompt_choice(
        prompt,
        {"f": "fast", "fast": "fast", "t": "tuned", "tuned": "tuned", "k": "keep", "keep": "keep"},
        default=default,
    )

    if choice == "keep":
        print("Keeping existing model.")
        return

    ml_bootstrap.train_and_save(choice, progress=print)


def _ensure_frontend() -> None:
    from backend.config import FRONTEND_DIR, FRONTEND_DIST_DIR

    npm = shutil.which("npm")
    dist_index = FRONTEND_DIST_DIR / "index.html"
    src_dir = FRONTEND_DIR / "src"

    if not npm:
        if dist_index.exists():
            print("npm not found on PATH; serving the pre-built frontend/dist as-is.")
        else:
            print(
                "WARNING: npm not found on PATH and frontend/dist is missing.\n"
                "         Install Node.js (https://nodejs.org) so run.py can build the UI, or run\n"
                "         'npm install && npm run build' inside frontend/ yourself."
            )
        return

    needs_build = not dist_index.exists()
    if not needs_build and src_dir.exists():
        newest_src = max((p.stat().st_mtime for p in src_dir.rglob("*") if p.is_file()), default=0)
        needs_build = newest_src > dist_index.stat().st_mtime

    if not needs_build:
        print("Frontend is up to date.")
        return

    print("Building frontend (npm install && npm run build)...")
    if not (FRONTEND_DIR / "node_modules").exists():
        subprocess.run([npm, "install"], cwd=FRONTEND_DIR, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Tennis Match Predictor")
    parser.add_argument("--train", choices=["fast", "tuned"], default=None,
                         help="Force a (re)train in this mode before serving.")
    parser.add_argument("--force-retrain", action="store_true",
                         help="Retrain even if the active model's data hash matches current data.")
    parser.add_argument("--no-train", action="store_true",
                         help="Never train automatically; error out if no model exists.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    bootstrap_venv(Path(__file__).resolve())

    from backend.hardware import detect

    print(f"Hardware detected: {detect().label}")

    _ensure_data()
    _ensure_model(args)
    _ensure_frontend()

    import uvicorn

    from backend.app import app

    _print_banner(args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
