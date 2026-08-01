"""Versioned model artifacts on disk.

Every trained model lives in models/<version_id>/ with its own model.json
(XGBoost native format), scaler.pkl, selector.pkl (tuned mode only),
feature_state.pkl (the live-inference snapshot it was trained with) and
meta.json (human-facing stats). models/registry.json tracks every version and
which one is "active" - i.e. the one the API actually serves.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.config import CLEANED_CSV, MODELS_DIR, REGISTRY_FILE


@dataclass
class ModelMeta:
    version_id: str
    mode: str                  # "fast" | "tuned"
    device: str                # "cuda" | "cpu"
    device_label: str
    trained_at: str
    data_hash: Optional[str]
    train_rows: int
    test_rows: int
    train_accuracy: float
    test_accuracy: float
    date_range: list           # [min_year, max_year]
    num_players: int
    feature_importances: dict = field(default_factory=dict)
    selected_features: list = field(default_factory=list)


def _read_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {"active": None, "versions": []}
    return json.loads(REGISTRY_FILE.read_text())


def _write_registry(registry: dict) -> None:
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def new_version_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def version_dir(version_id: str) -> Path:
    return MODELS_DIR / version_id


def register_version(meta: ModelMeta) -> None:
    registry = _read_registry()
    registry["versions"] = [v for v in registry["versions"] if v["version_id"] != meta.version_id]
    registry["versions"].append(asdict(meta))
    registry["active"] = meta.version_id
    _write_registry(registry)


def get_active_meta() -> Optional[dict]:
    registry = _read_registry()
    active_id = registry.get("active")
    if not active_id:
        return None
    for v in registry["versions"]:
        if v["version_id"] == active_id:
            return v
    return None


def get_active_dir() -> Optional[Path]:
    meta = get_active_meta()
    if not meta:
        return None
    d = version_dir(meta["version_id"])
    return d if d.exists() else None


def list_versions() -> list:
    return _read_registry()["versions"]


def has_new_data() -> bool:
    """True if the currently-cleaned dataset differs from what the active
    model was trained on (or there is no active model at all)."""
    from backend.ml.pipeline import data_hash

    current_hash = data_hash()
    if current_hash is None:
        return False
    active = get_active_meta()
    if active is None:
        return True
    return active.get("data_hash") != current_hash


def delete_version(version_id: str) -> None:
    registry = _read_registry()
    registry["versions"] = [v for v in registry["versions"] if v["version_id"] != version_id]
    if registry.get("active") == version_id:
        registry["active"] = registry["versions"][-1]["version_id"] if registry["versions"] else None
    _write_registry(registry)
    shutil.rmtree(version_dir(version_id), ignore_errors=True)
