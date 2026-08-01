"""Walk-forward betting backtest: trains on every season before year Y,
evaluates on year Y, rolls forward, and repeats - so every bet is placed
against a model that never saw that season's matches or outcomes.

Deliberately simpler than the main trainer: no RFECV, no hyperparameter
search - each step uses the fixed "fast" hyperparameters so a walk over
several seasons finishes in a few minutes instead of tens of minutes. This
is a real leakage-free evaluation of the feature/model approach, not a
reproduction of the tuned production model's exact accuracy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.config import CLEANED_CSV
from backend.hardware import detect
from backend.ml.features import FEATURE_COLUMNS
from backend.ml.train import FAST_PARAMS, build_training_matrix, fit_calibrated

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


MIN_TRAIN_YEARS = 5

# Raw (even fractional) Kelly sizing assumes the edge estimate is exact. It
# isn't - it's a machine-learning probability with real estimation error, and
# on short-odds favorites the Kelly formula alone can recommend staking 25%+
# of the bankroll on a single bet. Across hundreds of sequential compounding
# bets in a season, that turns ordinary model error into a near-total wipeout
# instead of a bounded drawdown. Capping the per-bet stake is standard
# practical risk management for exactly this reason.
MAX_STAKE_FRACTION = 0.10


def _eligible_test_years(df: pd.DataFrame, max_seasons: int) -> list[int]:
    years = sorted(int(y) for y in df["YEAR"].unique())
    if len(years) <= MIN_TRAIN_YEARS:
        return []
    eligible = years[MIN_TRAIN_YEARS:]
    return eligible[-max_seasons:]


def _simulate_season(
    test_df: pd.DataFrame,
    proba1: np.ndarray,
    bankroll: float,
    edge_threshold: float,
    kelly_fraction: float,
    equity_curve: list,
) -> dict:
    bets = 0
    wins = 0
    start_bankroll = bankroll

    ordered = test_df.sort_values("Date")
    proba_by_index = dict(zip(test_df.index, proba1))

    for idx, row in ordered.iterrows():
        p1 = float(proba_by_index[idx])
        p2 = 1 - p1
        odds1, odds2 = row["P1_b365"], row["P2_b365"]
        if not (odds1 > 1 and odds2 > 1):
            continue

        implied1, implied2 = 1 / odds1, 1 / odds2
        total = implied1 + implied2
        norm1, norm2 = implied1 / total, implied2 / total
        edge1, edge2 = p1 - norm1, p2 - norm2

        side = None
        if edge1 >= edge_threshold and edge1 >= edge2:
            side, edge, odds, win = 1, edge1, odds1, row["y"] == 1
        elif edge2 >= edge_threshold:
            side, edge, odds, win = 2, edge2, odds2, row["y"] == 0

        if side is None:
            continue

        kelly = float(np.clip((edge / (odds - 1)) * kelly_fraction, 0, MAX_STAKE_FRACTION))
        stake = bankroll * kelly
        bankroll += (odds - 1) * stake if win else -stake
        bets += 1
        wins += 1 if win else 0
        equity_curve.append({"date": str(row["Date"].date()), "bankroll": round(bankroll, 2)})

    return {
        "bets": bets,
        "wins": wins,
        "win_rate": (wins / bets) if bets else 0.0,
        "roi_pct": ((bankroll - start_bankroll) / start_bankroll * 100) if start_bankroll else 0.0,
        "final_bankroll": bankroll,
    }


def walk_forward_backtest(
    edge_threshold: float = 0.05,
    kelly_fraction: float = 0.5,
    max_seasons: int = 8,
    starting_bankroll: float = 10_000.0,
    progress: Progress = _noop,
) -> dict:
    hw = detect()
    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = build_training_matrix(df)

    test_years = _eligible_test_years(df, max_seasons)
    if not test_years:
        raise ValueError(
            f"Not enough seasons of data to backtest (need > {MIN_TRAIN_YEARS} years on record)."
        )
    progress(f"Walking forward over seasons: {test_years} (device: {hw.label})")

    bankroll = starting_bankroll
    equity_curve: list = []
    season_breakdown: list = []
    total_bets = 0
    total_wins = 0

    for year in test_years:
        train_df = df[df["YEAR"] < year]
        test_df = df[df["YEAR"] == year]
        if len(train_df) < 200 or len(test_df) < 10:
            progress(f"Skipping {year}: not enough rows (train={len(train_df)}, test={len(test_df)})")
            continue

        progress(f"Season {year}: training on {len(train_df)} rows, betting on {len(test_df)} matches...")

        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(train_df[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS, index=train_df.index
        )
        X_test = pd.DataFrame(
            scaler.transform(test_df[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS, index=test_df.index
        )
        calibrator = fit_calibrated(X_train, train_df["y"], dict(FAST_PARAMS), hw, cv=3)
        proba1 = calibrator.predict_proba(X_test)[:, 1]

        season_result = _simulate_season(
            test_df, proba1, bankroll, edge_threshold, kelly_fraction, equity_curve
        )
        bankroll = season_result["final_bankroll"]
        total_bets += season_result["bets"]
        total_wins += season_result["wins"]
        season_breakdown.append({
            "year": year,
            "bets": season_result["bets"],
            "win_rate": season_result["win_rate"],
            "roi_pct": season_result["roi_pct"],
        })
        progress(
            f"Season {year} done: {season_result['bets']} bets, "
            f"win rate {season_result['win_rate']:.1%}, bankroll now ${bankroll:,.2f}"
        )

    return {
        "seasons_tested": test_years,
        "starting_bankroll": starting_bankroll,
        "final_bankroll": round(bankroll, 2),
        "roi_pct": round((bankroll - starting_bankroll) / starting_bankroll * 100, 2),
        "total_bets": total_bets,
        "win_rate": (total_wins / total_bets) if total_bets else 0.0,
        "equity_curve": equity_curve,
        "season_breakdown": season_breakdown,
        "params": {
            "edge_threshold": edge_threshold,
            "kelly_fraction": kelly_fraction,
            "max_seasons": max_seasons,
        },
    }
