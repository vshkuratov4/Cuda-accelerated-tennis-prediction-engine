"""Unified trainer, replacing the duplicated baseline_model.py / custom_model.py.

Two modes:
  - "fast":  fixed, sane hyperparameters, no feature selection. Seconds to a
             couple of minutes. This is the default / recommended path.
  - "tuned": XGBoost-native CV to pick a good round count, RFECV feature
             selection, then a *bounded* RandomizedSearchCV (not the original
             exhaustive grid of thousands of combinations, which could take
             hours) - a deliberate, bounded stand-in for custom_model.py's
             tuning so it finishes in a predictable amount of time.

Both modes calibrate probabilities with isotonic CalibratedClassifierCV so the
win probabilities served to the UI (and used for edge/Kelly calculations) are
well-calibrated, not raw XGBoost scores.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb_mod
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.config import CLEANED_CSV
from backend.hardware import HardwareInfo, detect
from backend.ml.features import FEATURE_COLUMNS

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


FAST_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0,
)

TUNED_PARAM_DIST = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 1, 3, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 2, 5],
}
TUNED_N_ITER = 25
TUNED_CV = 3
RFECV_STEP = 3
RFECV_CV = 3


def build_training_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Map Winner/Loser-oriented engineered columns (from FeatureState.fit)
    into the uniform P1/P2 orientation used for both training and inference.
    P1 = the better-ranked (lower WRank/LRank) player, purely for label
    construction - at inference time player1/player2 are used as-is."""
    df = df.copy()
    better = df["WRank"] <= df["LRank"]
    df["y"] = better.astype(int)

    pairs = [
        ("P1_rank", "P2_rank", "WRank", "LRank"),
        ("P1_pts", "P2_pts", "WPts", "LPts"),
        ("P1_b365", "P2_b365", "B365W", "B365L"),
        ("P1_Elo_pre", "P2_Elo_pre", "Elo_W_pre", "Elo_L_pre"),
        ("P1_Form5", "P2_Form5", "Form5_w", "Form5_l"),
        ("P1_SurfForm20", "P2_SurfForm20", "SurfForm20_w", "SurfForm20_l"),
        ("P1_streak", "P2_streak", "Streak_w", "Streak_l"),
        ("P1_tourny_ct", "P2_tourny_ct", "Tourny_match_w", "Tourny_match_l"),
        ("P1_load30", "P2_load30", "Load30_w", "Load30_l"),
        ("P1_load30_missing", "P2_load30_missing", "Load30_w_missing", "Load30_l_missing"),
        ("P1_SetWinPct", "P2_SetWinPct", "SetWinPct_w", "SetWinPct_l"),
        ("P1_SurfSetPct", "P2_SurfSetPct", "SurfSetWinPct_w", "SurfSetWinPct_l"),
        ("P1_RankMom", "P2_RankMom", "RankMom_w", "RankMom_l"),
        ("P1_rest", "P2_rest", "Rest_days_w", "Rest_days_l"),
        ("P1_rest_missing", "P2_rest_missing", "Rest_w_missing", "Rest_l_missing"),
    ]
    for p1_col, p2_col, w_col, l_col in pairs:
        df[p1_col] = df[w_col].where(better, df[l_col])
        df[p2_col] = df[l_col].where(better, df[w_col])

    df["rank_diff"] = df["P1_rank"] - df["P2_rank"]
    df["odds_diff"] = df["P1_b365"] - df["P2_b365"]
    df["Elo_diff"] = df["P1_Elo_pre"] - df["P2_Elo_pre"]
    df["P1_H2H_pct"] = df["H2H_win_pct"].where(better, 1 - df["H2H_win_pct"])
    df["P2_H2H_pct"] = 1 - df["P1_H2H_pct"]
    df["P1_H2H_surf"] = df["H2H_surf_pct"].where(better, 1 - df["H2H_surf_pct"])
    df["P2_H2H_surf"] = 1 - df["P1_H2H_surf"]

    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held-out test = most recent year on record; falls back to the last 10%
    of rows by date if that leaves too little data on either side (e.g. a
    partially-downloaded dataset)."""
    max_year = int(df["YEAR"].max())
    train_df = df[df["YEAR"] < max_year]
    test_df = df[df["YEAR"] == max_year]

    if len(test_df) < 50 or len(train_df) < 200:
        df_sorted = df.sort_values("Date")
        split_idx = int(len(df_sorted) * 0.9)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]

    return train_df, test_df


def fit_calibrated(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    clf_params: dict,
    hw: HardwareInfo,
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Fits an isotonic-calibrated XGBoost classifier. Shared by train() (to
    produce the model actually served) and backtest.py's per-season
    walk-forward fits, so both use the exact same calibration procedure."""
    calibrator = CalibratedClassifierCV(
        estimator=XGBClassifier(
            **clf_params, tree_method="hist", device=hw.device,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
        method="isotonic",
        cv=cv,
    )
    calibrator.fit(X_train, y_train)
    return calibrator


def _find_best_nrounds(X_train: pd.DataFrame, y_train: pd.Series, hw: HardwareInfo, progress: Progress) -> int:
    dtrain = xgb_mod.DMatrix(X_train, label=y_train)
    cv_params = {
        "tree_method": "hist",
        "device": hw.device,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "seed": 42,
    }
    cv_results = xgb_mod.cv(
        params=cv_params,
        dtrain=dtrain,
        num_boost_round=500,
        nfold=3,
        early_stopping_rounds=20,
        as_pandas=True,
        seed=42,
    )
    best_nrounds = max(len(cv_results), 10)
    progress(f"Optimal n_estimators (via XGBoost CV): {best_nrounds}")
    return best_nrounds


def train(
    mode: str,
    df: Optional[pd.DataFrame] = None,
    progress: Progress = _noop,
) -> dict:
    """Trains a model end-to-end and returns everything needed to persist it:
    {calibrator, scaler, selector, selected_features, train_acc, test_acc,
     train_rows, test_rows, date_range, feature_importances, device}."""
    assert mode in ("fast", "tuned")
    hw = detect()
    progress(f"Training in '{mode}' mode on {hw.label}")

    if df is None:
        df = pd.read_csv(CLEANED_CSV, low_memory=False)
        df["Date"] = pd.to_datetime(df["Date"])

    df = build_training_matrix(df)
    train_df, test_df = time_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["y"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["y"]
    progress(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index)

    selector = None
    selected_features = list(FEATURE_COLUMNS)

    n_jobs = 1 if hw.device == "cuda" else -1

    if mode == "tuned":
        progress("Running RFECV feature selection (this can take a few minutes)...")
        selector = RFECV(
            estimator=XGBClassifier(
                tree_method="hist", device=hw.device, eval_metric="logloss",
                random_state=42, verbosity=0,
            ),
            step=RFECV_STEP,
            cv=RFECV_CV,
            scoring="accuracy",
            n_jobs=n_jobs,
        )
        selector.fit(X_train_scaled, y_train)
        selected_features = [c for c, keep in zip(FEATURE_COLUMNS, selector.support_) if keep]
        progress(f"RFECV kept {len(selected_features)}/{len(FEATURE_COLUMNS)} features")
        X_train_sel = X_train_scaled[selected_features]
        X_test_sel = X_test_scaled[selected_features]

        best_nrounds = _find_best_nrounds(X_train_sel, y_train, hw, progress)

        progress(f"Randomized hyperparameter search ({TUNED_N_ITER} candidates x {TUNED_CV}-fold CV)...")
        search = RandomizedSearchCV(
            estimator=XGBClassifier(
                n_estimators=best_nrounds, tree_method="hist", device=hw.device,
                eval_metric="logloss", random_state=42, verbosity=0,
            ),
            param_distributions=TUNED_PARAM_DIST,
            n_iter=TUNED_N_ITER,
            cv=TUNED_CV,
            scoring="accuracy",
            random_state=42,
            n_jobs=n_jobs,
        )
        search.fit(X_train_sel, y_train)
        clf_params = {**search.best_params_, "n_estimators": best_nrounds}
        progress(f"Best params: {clf_params}")
    else:
        X_train_sel = X_train_scaled
        X_test_sel = X_test_scaled
        clf_params = dict(FAST_PARAMS)

    clf = XGBClassifier(
        **clf_params, tree_method="hist", device=hw.device,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    clf.fit(X_train_sel, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_sel))
    test_acc = accuracy_score(y_test, clf.predict(X_test_sel))
    progress(f"Train accuracy: {train_acc:.3f}  |  Test accuracy: {test_acc:.3f}")

    feature_importances = {
        f: float(v) for f, v in sorted(
            zip(selected_features, clf.feature_importances_), key=lambda kv: -kv[1]
        )
    }

    progress("Calibrating probabilities (isotonic)...")
    calibrator = fit_calibrated(X_train_sel, y_train, clf_params, hw, cv=3)

    return {
        "calibrator": calibrator,
        "scaler": scaler,
        "selector": selector,
        "selected_features": selected_features,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "date_range": [int(df["YEAR"].min()), int(df["YEAR"].max())],
        "feature_importances": feature_importances,
        "device": hw.device,
        "device_label": hw.label,
    }
