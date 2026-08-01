"""Loads the active model version's artifacts and serves predictions.
Reload() is called after a retrain finishes so the API picks up the new
version without needing a server restart."""

from __future__ import annotations

import threading
from typing import Optional

import joblib
import numpy as np

from backend.ml import registry
from backend.ml.backtest import MAX_STAKE_FRACTION
from backend.ml.features import FEATURE_COLUMNS, FeatureState

# Confidence-label thresholds. std is the spread of "player1 wins" probability
# across the 3 sub-models inside the isotonic CalibratedClassifierCV ensemble -
# a real measure of how much the calibration folds disagree on this exact
# matchup, not a fabricated number. It's gated by how much real history either
# player has on record, since the model can look confident on a fresh face
# with almost no data simply because there's little signal to disagree about.
MIN_MATCHES_FOR_ANY_CONFIDENCE = 10
MIN_MATCHES_FOR_HIGH_CONFIDENCE = 30
LOW_CONFIDENCE_STD = 0.08
HIGH_CONFIDENCE_STD = 0.03


def _confidence_label(std: float, matches1: int, matches2: int) -> str:
    if min(matches1, matches2) < MIN_MATCHES_FOR_ANY_CONFIDENCE or std > LOW_CONFIDENCE_STD:
        return "Low"
    if min(matches1, matches2) >= MIN_MATCHES_FOR_HIGH_CONFIDENCE and std <= HIGH_CONFIDENCE_STD:
        return "High"
    return "Medium"


class InferenceService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calibrator = None
        self.scaler = None
        self.selector = None
        self.selected_features: list[str] = list(FEATURE_COLUMNS)
        self.feature_state: Optional[FeatureState] = None
        self.version_id: Optional[str] = None
        self.reload()

    def reload(self) -> bool:
        with self._lock:
            version_dir = registry.get_active_dir()
            if version_dir is None:
                self.calibrator = None
                self.feature_state = None
                self.version_id = None
                return False

            self.calibrator = joblib.load(version_dir / "model.pkl")
            self.scaler = joblib.load(version_dir / "scaler.pkl")
            selector_path = version_dir / "selector.pkl"
            self.selector = joblib.load(selector_path) if selector_path.exists() else None
            self.feature_state = joblib.load(version_dir / "feature_state.pkl")

            meta = registry.get_active_meta()
            self.selected_features = meta["selected_features"] if meta else list(FEATURE_COLUMNS)
            self.version_id = meta["version_id"] if meta else None
            return True

    @property
    def is_ready(self) -> bool:
        return self.calibrator is not None and self.feature_state is not None

    def players(self) -> list:
        return self.feature_state.players if self.feature_state else []

    def surfaces(self) -> list:
        return self.feature_state.surfaces() if self.feature_state else []

    def predict(
        self,
        player1: str,
        player2: str,
        surface: str,
        round_name: Optional[str] = None,
        odds1: Optional[float] = None,
        odds2: Optional[float] = None,
    ) -> dict:
        if not self.is_ready:
            raise RuntimeError("No trained model is available yet.")

        row = self.feature_state.transform_live(
            player1, player2, surface, round_name,
            odds1=odds1, odds2=odds2,
        )
        row_scaled = self.scaler.transform(row[FEATURE_COLUMNS])
        row_scaled = np.asarray(row_scaled)
        col_index = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
        row_final = row_scaled[:, [col_index[c] for c in self.selected_features]]

        proba = self.calibrator.predict_proba(row_final)[0]
        prob1, prob2 = float(proba[1]), float(proba[0])
        winner = player1 if prob1 >= prob2 else player2

        sub_probs1 = np.array([
            sub.predict_proba(row_final)[0][1] for sub in self.calibrator.calibrated_classifiers_
        ])
        prob1_std = float(sub_probs1.std())
        ci_half_width = min(1.96 * prob1_std, 0.5)
        ci_low1 = max(0.0, prob1 - ci_half_width)
        ci_high1 = min(1.0, prob1 + ci_half_width)

        matches1 = self.feature_state.match_count(player1)
        matches2 = self.feature_state.match_count(player2)
        confidence = _confidence_label(prob1_std, matches1, matches2)

        result = {
            "player1": player1,
            "player2": player2,
            "winner": winner,
            "prob1": prob1,
            "prob2": prob2,
            "confidence": confidence,
            "prob1_std": prob1_std,
            "ci_low1": ci_low1,
            "ci_high1": ci_high1,
            "ci_low2": 1 - ci_high1,
            "ci_high2": 1 - ci_low1,
            "player1_matches": matches1,
            "player2_matches": matches2,
        }

        if odds1 and odds2 and odds1 > 1 and odds2 > 1:
            implied1, implied2 = 1 / odds1, 1 / odds2
            total = implied1 + implied2
            norm1, norm2 = implied1 / total, implied2 / total
            edge1, edge2 = prob1 - norm1, prob2 - norm2
            # Capped at MAX_STAKE_FRACTION: raw Kelly sizing on short odds can
            # recommend staking 25%+ of bankroll on a single bet, which
            # assumes the model's edge estimate is exact. It isn't - see
            # backend/ml/backtest.py for the full rationale.
            kelly1 = float(np.clip((edge1 / (odds1 - 1)) * 0.5, 0, MAX_STAKE_FRACTION)) if edge1 > 0 else 0.0
            kelly2 = float(np.clip((edge2 / (odds2 - 1)) * 0.5, 0, MAX_STAKE_FRACTION)) if edge2 > 0 else 0.0
            result.update({
                "implied_prob1": norm1,
                "implied_prob2": norm2,
                "edge1": edge1,
                "edge2": edge2,
                "kelly_stake1": kelly1,
                "kelly_stake2": kelly2,
            })

        return result


inference_service = InferenceService()
