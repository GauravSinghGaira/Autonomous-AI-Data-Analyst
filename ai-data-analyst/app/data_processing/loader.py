"""
Dataset loading utilities.

Responsible only for getting a clean pandas DataFrame from an uploaded
CSV or Excel file. No analysis logic lives here (see profiler.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv"}


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be parsed into a DataFrame."""


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV/TSV/Excel file into a DataFrame.

    Raises DatasetLoadError with a human-readable message on failure,
    so calling layers (API/agent) can surface it directly instead of
    leaking a pandas traceback.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetLoadError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
        else:  # .xlsx / .xls
            df = pd.read_excel(file_path)
    except Exception as exc:  # noqa: BLE001 - we deliberately want a single failure path
        logger.exception("Failed to parse dataset at %s", file_path)
        raise DatasetLoadError(f"Could not parse file: {exc}") from exc

    if df.empty:
        raise DatasetLoadError("The uploaded file was parsed but contains no rows.")

    logger.info("Loaded dataset %s -> shape=%s", file_path.name, df.shape)
    return df


def save_upload(file_bytes: bytes, filename: str, upload_dir: str) -> Path:
    """Persist an uploaded file to disk and return its path."""
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = Path(filename).name  # strip any path traversal attempts
    dest = Path(upload_dir) / safe_name
    with open(dest, "wb") as f:
        f.write(file_bytes)
    logger.info("Saved upload to %s (%d bytes)", dest, len(file_bytes))
    return dest
