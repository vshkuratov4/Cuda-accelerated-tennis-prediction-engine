"""Checks tennis-data.co.uk for an updated current-season file and retrains if
anything changed. Meant to be run standalone (see sync_data.py at the repo
root) on a schedule the user sets up themselves (cron / Task Scheduler) -
the running web server never triggers this itself.

Only the current (and prior, in case of late corrections) season's file is
ever re-downloaded here: earlier seasons are finished and don't change, so
re-fetching the whole 26-year archive on every check would be wasteful.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Callable, Optional

import requests
from bs4 import BeautifulSoup

from backend.config import CLEANED_CSV, MERGED_CSV, PROCESSED_DATA_DIR, RAW_DATA_DIR
from backend.ml.pipeline import ATP_BASE_URL, ATP_INDEX_URL

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


SYNC_STATE_FILE = PROCESSED_DATA_DIR / "sync_state.json"
YEAR_PATTERN = re.compile(r"^(\d{4})/\d{4}\.xlsx?$", re.IGNORECASE)


def _candidate_years() -> set[int]:
    current = datetime.now().year
    return {current - 1, current}


def fetch_and_check_updates(progress: Progress = _noop) -> tuple[bool, list[int]]:
    """Re-downloads the current (and prior) season's file and overwrites the
    local copy if it differs. Returns (changed, years_that_changed)."""
    try:
        response = requests.get(ATP_INDEX_URL, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        progress(f"Could not reach {ATP_INDEX_URL} ({exc}); nothing to sync.")
        return False, []

    soup = BeautifulSoup(response.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a")]
    wanted_years = _candidate_years()

    changed_years: list[int] = []
    for href in links:
        if not href:
            continue
        match = YEAR_PATTERN.match(href)
        if not match:
            continue
        year = int(match.group(1))
        if year not in wanted_years:
            continue

        try:
            file_response = requests.get(f"{ATP_BASE_URL}{href}", timeout=30)
            file_response.raise_for_status()
        except requests.RequestException as exc:
            progress(f"Failed to fetch {href}: {exc}")
            continue

        ext = ".xlsx" if href.lower().endswith(".xlsx") else ".xls"
        dest_path = RAW_DATA_DIR / f"{year}{ext}"
        new_bytes = file_response.content

        if dest_path.exists() and dest_path.read_bytes() == new_bytes:
            progress(f"{year}: no change.")
            continue

        dest_path.write_bytes(new_bytes)
        changed_years.append(year)
        progress(f"{year}: updated ({len(new_bytes):,} bytes).")

    return bool(changed_years), sorted(changed_years)


def sync_and_retrain(mode: str = "fast", progress: Progress = _noop) -> Optional[object]:
    """Checks for new data; if found, rebuilds the cleaned dataset from the
    refreshed raw files and retrains + activates a new model version.
    Returns the new ModelMeta, or None if nothing had changed."""
    from backend.ml import bootstrap as ml_bootstrap
    from backend.ml import pipeline

    changed, years = fetch_and_check_updates(progress)

    if not changed:
        progress("No new data available - nothing to retrain.")
        _write_sync_state(changed=False, updated_years=[], retrained_version=None)
        return None

    progress(f"New/updated data for {years}; rebuilding the cleaned dataset...")
    for path in (MERGED_CSV, CLEANED_CSV, CLEANED_CSV.with_name("feature_state.pkl")):
        path.unlink(missing_ok=True)

    pipeline.run_full_pipeline(progress=progress)
    meta = ml_bootstrap.train_and_save(mode, progress=progress)

    _write_sync_state(changed=True, updated_years=years, retrained_version=meta.version_id)
    return meta


def _write_sync_state(changed: bool, updated_years: list[int], retrained_version: Optional[str]) -> None:
    SYNC_STATE_FILE.write_text(json.dumps({
        "last_checked": datetime.now().isoformat(),
        "changed": changed,
        "updated_years": updated_years,
        "retrained_version": retrained_version,
    }, indent=2))


def read_sync_state() -> dict:
    if not SYNC_STATE_FILE.exists():
        return {"last_checked": None, "changed": None, "updated_years": [], "retrained_version": None}
    return json.loads(SYNC_STATE_FILE.read_text())
