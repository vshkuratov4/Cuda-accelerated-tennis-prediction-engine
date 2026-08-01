from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    player1: str
    player2: str
    surface: str
    round: Optional[str] = None
    odds1: Optional[float] = Field(default=None, gt=1)
    odds2: Optional[float] = Field(default=None, gt=1)


class PredictResponse(BaseModel):
    player1: str
    player2: str
    winner: str
    prob1: float
    prob2: float
    confidence: str
    prob1_std: float
    ci_low1: float
    ci_high1: float
    ci_low2: float
    ci_high2: float
    player1_matches: int
    player2_matches: int
    implied_prob1: Optional[float] = None
    implied_prob2: Optional[float] = None
    edge1: Optional[float] = None
    edge2: Optional[float] = None
    kelly_stake1: Optional[float] = None
    kelly_stake2: Optional[float] = None


class RetrainRequest(BaseModel):
    mode: str = Field(pattern="^(fast|tuned)$")


class BacktestRequest(BaseModel):
    # 5% looked reasonable in isolation but empirically qualifies 50-70% of all
    # matches as "+edge" against a per-season fast-mode model, which is far
    # more than any genuine market edge - that's calibration noise, not
    # signal, and naive Kelly sizing on noisy edges erodes the bankroll even
    # at a >50% win rate. 10% is a more realistic starting point.
    edge_threshold: float = Field(default=0.10, ge=0, le=1)
    kelly_fraction: float = Field(default=0.5, gt=0, le=1)
    max_seasons: int = Field(default=8, ge=1, le=20)


class MetaResponse(BaseModel):
    surfaces: list[str]
    rounds: list[str]


class HealthResponse(BaseModel):
    status: str
    device: str
    device_label: str
    model_ready: bool
