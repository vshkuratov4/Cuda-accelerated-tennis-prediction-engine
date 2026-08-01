from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    BacktestRequest,
    HealthResponse,
    MetaResponse,
    PredictRequest,
    PredictResponse,
    RetrainRequest,
)
from backend.hardware import detect
from backend.ml import registry
from backend.ml.pipeline import data_hash
from backend.services.backtest_job import backtest_job
from backend.services.inference import inference_service
from backend.services.train_job import train_job

router = APIRouter(prefix="/api")

ROUND_CODES = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]


@router.get("/health", response_model=HealthResponse)
def health():
    hw = detect()
    return HealthResponse(
        status="ok",
        device=hw.device,
        device_label=hw.label,
        model_ready=inference_service.is_ready,
    )


@router.get("/players")
def players():
    return {"players": inference_service.players()}


@router.get("/meta", response_model=MetaResponse)
def meta():
    surfaces = inference_service.surfaces() or ["Hard", "Clay", "Grass", "Carpet"]
    return MetaResponse(surfaces=surfaces, rounds=ROUND_CODES)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not inference_service.is_ready:
        raise HTTPException(status_code=503, detail="No trained model is available yet.")
    if req.player1 == req.player2:
        raise HTTPException(status_code=400, detail="Please choose two different players.")
    try:
        result = inference_service.predict(
            player1=req.player1,
            player2=req.player2,
            surface=req.surface,
            round_name=req.round,
            odds1=req.odds1,
            odds2=req.odds2,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.get("/model/info")
def model_info():
    active = registry.get_active_meta()
    return {
        "active": active,
        "versions": registry.list_versions(),
        "has_new_data": registry.has_new_data(),
        "current_data_hash": data_hash(),
    }


@router.post("/model/retrain")
def retrain(req: RetrainRequest):
    started = train_job.start(req.mode)
    if not started:
        raise HTTPException(status_code=409, detail="A training job is already running.")
    return {"status": "started", "mode": req.mode}


@router.get("/model/train-status")
def train_status():
    return train_job.to_dict()


@router.get("/model/estimate")
def model_estimate():
    import pandas as pd

    from backend.config import CLEANED_CSV
    from backend.hardware import detect
    from backend.ml.estimate import estimate_training_time

    if not CLEANED_CSV.exists():
        raise HTTPException(status_code=503, detail="No processed dataset available yet.")

    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return estimate_training_time(df, detect())


@router.post("/backtest")
def start_backtest(req: BacktestRequest):
    from backend.config import CLEANED_CSV

    if not CLEANED_CSV.exists():
        raise HTTPException(status_code=503, detail="No processed dataset available yet.")
    started = backtest_job.start(req.edge_threshold, req.kelly_fraction, req.max_seasons)
    if not started:
        raise HTTPException(status_code=409, detail="A backtest is already running.")
    return {"status": "started"}


@router.get("/backtest/status")
def backtest_status():
    return backtest_job.to_dict()


@router.get("/data/sync-info")
def data_sync_info():
    from backend.ml.sync import read_sync_state

    return read_sync_state()
