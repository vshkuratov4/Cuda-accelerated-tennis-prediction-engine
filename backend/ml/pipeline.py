"""Data pipeline: download raw ATP Excel files, merge them into one CSV,
then clean + engineer features. Replaces download_atp_data.py, merge_excels.py
and the generic-cleaning half of clean_data.py (the feature-engineering half
now lives in features.FeatureState).

Every step is idempotent and skips work that's already done, so re-running the
launcher never re-downloads or re-merges data that hasn't changed.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from typing import Callable, Optional

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup

from backend.config import CLEANED_CSV, MERGED_CSV, RAW_DATA_DIR
from backend.ml.features import FeatureState

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


ATP_INDEX_URL = "http://www.tennis-data.co.uk/alldata.php"
ATP_BASE_URL = "http://www.tennis-data.co.uk/"


def download_raw_data(progress: Progress = _noop) -> int:
    """Download every ATP .xls file that isn't already in data/raw. Returns
    the number of files newly downloaded. Never raises on network failure -
    callers should fall back to whatever's already on disk."""
    try:
        response = requests.get(ATP_INDEX_URL, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        progress(f"Could not reach {ATP_INDEX_URL} ({exc}); using local data only.")
        return 0

    soup = BeautifulSoup(response.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a")]
    # tennis-data.co.uk serves ATP (men's) files as "<year>/<year>.xls[x]" (e.g.
    # "2024/2024.xlsx"). WTA (women's) files live under "<year>w/..." on the same
    # index page, so the directory component must be exactly 4 digits to exclude them.
    year_pattern = re.compile(r"^(\d{4})/\d{4}\.xlsx?$", re.IGNORECASE)
    atp_files = [h for h in links if h and year_pattern.match(h)]
    progress(f"Found {len(atp_files)} ATP data files on tennis-data.co.uk")

    downloaded = 0
    for file_name in atp_files:
        year_match = year_pattern.search(file_name)
        ext = ".xlsx" if file_name.lower().endswith(".xlsx") else ".xls"
        save_name = f"{year_match.group(1)}{ext}" if year_match else file_name.replace("/", "_")
        dest_path = RAW_DATA_DIR / save_name

        if dest_path.exists():
            continue

        try:
            file_response = requests.get(f"{ATP_BASE_URL}{file_name}", timeout=30)
            file_response.raise_for_status()
        except requests.RequestException as exc:
            progress(f"Failed to download {file_name}: {exc}")
            continue

        dest_path.write_bytes(file_response.content)
        downloaded += 1
        progress(f"Downloaded {save_name}")

    return downloaded


def merge_raw_data(progress: Progress = _noop) -> pd.DataFrame:
    """Merge every .xls/.xlsx in data/raw into one DataFrame tagged with YEAR,
    and write it to data/processed/final_dataset.csv."""
    paths = sorted(glob.glob(os.path.join(str(RAW_DATA_DIR), "*.xls*")))
    if not paths:
        raise FileNotFoundError(
            f"No .xls/.xlsx files found in {RAW_DATA_DIR}. Provide raw ATP data files "
            "there, or ensure network access to tennis-data.co.uk for auto-download."
        )

    frames = []
    for path in paths:
        df = pd.read_excel(path, sheet_name=0)
        fname = os.path.basename(path)
        year_str = fname.split(".")[0].split()[0]
        try:
            df["YEAR"] = int(year_str)
        except ValueError:
            df["YEAR"] = year_str
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=True)
    merged.to_csv(MERGED_CSV, index=False)
    progress(f"Merged {len(frames)} files -> {merged.shape[0]} rows x {merged.shape[1]} cols")
    return merged


def _generic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Some years' source files have stray leading/trailing whitespace on player
    # names (e.g. "Djokovic N. " vs "Djokovic N."), which would otherwise split
    # one player's Elo/H2H/form history into two separate identities.
    for col in ("Winner", "Loser"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.replace("NR", pd.NA)
    for col in ("WRank", "LRank"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["WRank", "LRank"])

    missing_pct = df.isna().mean()
    to_drop = missing_pct[missing_pct > 0.4].index.tolist()
    df = df.drop(columns=to_drop)

    # Real ATP files occasionally have stray non-numeric entries in odds/points
    # columns (varies by year/bookmaker). Coerce anything we treat as numeric
    # downstream so the column doesn't silently stay `object` dtype with a mix
    # of floats and strings (which breaks arithmetic in build_training_matrix).
    numeric_candidates = [
        "WPts", "LPts", "B365W", "B365L", "Wsets", "Lsets",
        "WRank", "LRank", "YEAR",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include=["number"]).columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in obj_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode[0] if len(mode) else "Unknown")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["YEAR"] = df["YEAR"].astype(int)
    return df


def clean_and_engineer(raw_df: pd.DataFrame, progress: Progress = _noop) -> tuple[pd.DataFrame, FeatureState]:
    progress(f"Cleaning {raw_df.shape[0]} raw rows...")
    df = _generic_clean(raw_df)
    progress(f"Shape after generic cleaning: {df.shape}")

    progress("Engineering Elo / H2H / form / streak features (this replays full match history)...")
    state = FeatureState()
    enriched = state.fit(df)

    enriched.to_csv(CLEANED_CSV, index=False)
    joblib.dump(state, CLEANED_CSV.with_name("feature_state.pkl"))
    progress(f"Saved cleaned dataset to {CLEANED_CSV} ({enriched.shape[0]} rows, {len(state.players)} players)")
    return enriched, state


def data_hash() -> Optional[str]:
    if not CLEANED_CSV.exists():
        return None
    h = hashlib.sha256()
    with open(CLEANED_CSV, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_full_pipeline(progress: Progress = _noop) -> tuple[pd.DataFrame, FeatureState]:
    """Idempotent: only (re)downloads/merges/cleans what's missing."""
    if not any(RAW_DATA_DIR.glob("*.xls*")):
        progress("No raw data found locally - downloading from tennis-data.co.uk...")
        download_raw_data(progress)

    if not any(RAW_DATA_DIR.glob("*.xls*")):
        raise FileNotFoundError(
            f"No raw ATP data available and download failed. Place .xls/.xlsx files "
            f"in {RAW_DATA_DIR} manually and re-run."
        )

    if not MERGED_CSV.exists():
        merged = merge_raw_data(progress)
    else:
        merged = pd.read_csv(MERGED_CSV, low_memory=False)
        progress(f"Reusing existing merged dataset ({merged.shape[0]} rows)")

    if not CLEANED_CSV.exists():
        enriched, state = clean_and_engineer(merged, progress)
    else:
        progress(f"Reusing existing cleaned dataset at {CLEANED_CSV}")
        enriched = pd.read_csv(CLEANED_CSV, low_memory=False)
        state_path = CLEANED_CSV.with_name("feature_state.pkl")
        state = joblib.load(state_path) if state_path.exists() else clean_and_engineer(merged, progress)[1]

    return enriched, state
