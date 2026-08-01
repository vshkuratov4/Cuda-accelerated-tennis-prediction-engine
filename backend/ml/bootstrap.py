"""Glue between the data pipeline, the trainer and the model registry.
This is what both run.py (interactive first-launch flow) and the
/api/model/retrain endpoint (background job) call into."""

from __future__ import annotations

import shutil
from typing import Callable

import joblib

from backend.config import CLEANED_CSV
from backend.ml import pipeline, registry, train

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


def train_and_save(mode: str, progress: Progress = _noop) -> registry.ModelMeta:
    import pandas as pd

    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])

    result = train.train(mode=mode, df=df, progress=progress)

    version_id = registry.new_version_id()
    version_dir = registry.version_dir(version_id)
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result["calibrator"], version_dir / "model.pkl")
    joblib.dump(result["scaler"], version_dir / "scaler.pkl")
    if result["selector"] is not None:
        joblib.dump(result["selector"], version_dir / "selector.pkl")

    state_src = CLEANED_CSV.with_name("feature_state.pkl")
    shutil.copy(state_src, version_dir / "feature_state.pkl")
    state = joblib.load(state_src)

    meta = registry.ModelMeta(
        version_id=version_id,
        mode=mode,
        device=result["device"],
        device_label=result["device_label"],
        trained_at=__import__("datetime").datetime.now().isoformat(),
        data_hash=pipeline.data_hash(),
        train_rows=result["train_rows"],
        test_rows=result["test_rows"],
        train_accuracy=result["train_accuracy"],
        test_accuracy=result["test_accuracy"],
        date_range=result["date_range"],
        num_players=len(state.players),
        feature_importances=result["feature_importances"],
        selected_features=result["selected_features"],
    )
    registry.register_version(meta)
    progress(f"Saved model version {version_id} (mode={mode}, test_acc={result['test_accuracy']:.3f})")
    return meta


def ensure_data(progress: Progress = _noop):
    return pipeline.run_full_pipeline(progress=progress)
