"""Quick local benchmark used to give the user a real, hardware-aware time
estimate before they choose between "fast" and "tuned" training, instead of a
guess that ignores whether they're on a laptop CPU or a GPU box."""

from __future__ import annotations

import time

import pandas as pd
from xgboost import XGBClassifier

from backend.hardware import HardwareInfo
from backend.ml.features import FEATURE_COLUMNS
from backend.ml.train import (
    FAST_PARAMS,
    RFECV_CV,
    RFECV_STEP,
    TUNED_CV,
    TUNED_N_ITER,
    build_training_matrix,
)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"~{max(1, round(seconds))}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"~{round(minutes)} min"
    return f"~{minutes / 60:.1f} hr"


def estimate_training_time(df: pd.DataFrame, hw: HardwareInfo) -> dict:
    sample_n = min(3000, len(df))
    sample = df.sample(n=sample_n, random_state=42) if len(df) > sample_n else df
    sample = build_training_matrix(sample)
    X = sample[FEATURE_COLUMNS].fillna(0)
    y = sample["y"]

    # XGBoost/OpenMP thread-pool spin-up is a fixed cost that swamps a small
    # timed probe if left in - warm it up untimed first so the timed fit below
    # measures actual per-tree throughput, not one-time startup overhead.
    warmup = XGBClassifier(
        n_estimators=10, max_depth=3, tree_method="hist", device=hw.device,
        eval_metric="logloss", verbosity=0,
    )
    warmup.fit(X.iloc[: min(200, len(X))], y.iloc[: min(200, len(y))])

    probe_estimators = 100
    clf = XGBClassifier(
        n_estimators=probe_estimators, max_depth=5, learning_rate=0.1,
        tree_method="hist", device=hw.device, eval_metric="logloss", verbosity=0,
    )
    start = time.time()
    clf.fit(X, y)
    probe_seconds = max(time.time() - start, 0.05)

    scale_factor = max(len(df) / sample_n, 1.0)
    single_full_fit_seconds = probe_seconds * scale_factor * (FAST_PARAMS["n_estimators"] / probe_estimators)

    # fast mode ~= one full fit + 3 calibration-fold fits
    fast_seconds = single_full_fit_seconds * 4

    # tuned mode ~= RFECV fits + randomized-search fits + final fit + calibration folds
    rfecv_fits = (len(FEATURE_COLUMNS) // RFECV_STEP + 1) * RFECV_CV
    search_fits = TUNED_N_ITER * TUNED_CV
    tuned_fit_equivalents = rfecv_fits + search_fits + 1 + 3
    tuned_seconds = single_full_fit_seconds * tuned_fit_equivalents

    return {
        "fast_seconds": round(fast_seconds, 1),
        "tuned_seconds": round(tuned_seconds, 1),
        "fast_label": format_duration(fast_seconds),
        "tuned_label": format_duration(tuned_seconds),
        "hardware": hw.label,
    }
