"""Runs the walk-forward backtest in a background thread (it retrains several
fast-mode models back to back, so it can take a couple of minutes) and lets
the frontend poll for progress/result, the same pattern as train_job.py."""

from __future__ import annotations

import threading
from typing import Optional


class BacktestJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "idle"          # idle | running | done | error
        self.logs: list[str] = []
        self.error: Optional[str] = None
        self.result: Optional[dict] = None

    def _log(self, msg: str) -> None:
        with self._lock:
            self.logs.append(msg)
        print(f"[backtest] {msg}")

    def start(self, edge_threshold: float, kelly_fraction: float, max_seasons: int) -> bool:
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.logs = []
            self.error = None
            self.result = None

        thread = threading.Thread(
            target=self._run, args=(edge_threshold, kelly_fraction, max_seasons), daemon=True
        )
        thread.start()
        return True

    def _run(self, edge_threshold: float, kelly_fraction: float, max_seasons: int) -> None:
        from backend.ml.backtest import walk_forward_backtest

        try:
            self._log("Starting walk-forward backtest...")
            result = walk_forward_backtest(
                edge_threshold=edge_threshold,
                kelly_fraction=kelly_fraction,
                max_seasons=max_seasons,
                progress=self._log,
            )
            with self._lock:
                self.result = result
                self.status = "done"
            self._log("Backtest complete.")
        except Exception as exc:  # noqa: BLE001
            self._log(f"Backtest failed: {exc}")
            with self._lock:
                self.status = "error"
                self.error = str(exc)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "logs": list(self.logs[-50:]),
                "error": self.error,
                "result": self.result,
            }


backtest_job = BacktestJob()
