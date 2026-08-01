"""Runs (re)training in a background thread so the API can return immediately
and the frontend can poll for progress instead of holding a long HTTP request
open for a tuned-mode training run."""

from __future__ import annotations

import threading
from typing import Optional


class TrainJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "idle"          # idle | running | done | error
        self.mode: Optional[str] = None
        self.logs: list[str] = []
        self.error: Optional[str] = None

    def _log(self, msg: str) -> None:
        with self._lock:
            self.logs.append(msg)
        print(f"[train] {msg}")

    def start(self, mode: str) -> bool:
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.mode = mode
            self.logs = []
            self.error = None

        thread = threading.Thread(target=self._run, args=(mode,), daemon=True)
        thread.start()
        return True

    def _run(self, mode: str) -> None:
        from backend.ml import bootstrap
        from backend.services.inference import inference_service

        try:
            self._log(f"Starting '{mode}' training...")
            bootstrap.train_and_save(mode, progress=self._log)
            inference_service.reload()
            self._log("Training complete, model is now active.")
            with self._lock:
                self.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self._log(f"Training failed: {exc}")
            with self._lock:
                self.status = "error"
                self.error = str(exc)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "mode": self.mode,
                "logs": list(self.logs[-50:]),
                "error": self.error,
            }


train_job = TrainJob()
