# src/utils.py
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# -----------------------
# Logging & Environment
# -----------------------

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Configure and return a logger with a standard format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------
# Paths & Basic IO
# -----------------------

@dataclass
class ProjectPaths:
    """
    Helper dataclass to centralise project paths.

    base_dir: root of the project (where README.md lives).
    """
    base_dir: Path

    @property
    def data_raw(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.base_dir / "reports"


def get_project_paths(start_dir: Optional[str | Path] = None) -> ProjectPaths:
    """
    Try to infer the project root (the directory containing README.md).
    If not found, use the current working directory.
    """
    if start_dir is None:
        start_dir = Path.cwd()
    else:
        start_dir = Path(start_dir)

    current = start_dir.resolve()
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists():
            return ProjectPaths(base_dir=parent)

    # fallback
    return ProjectPaths(base_dir=current)


def load_csv(path: str | Path, **read_kwargs) -> pd.DataFrame:
    """
    Convenience wrapper for reading CSVs.
    """
    path = Path(path)
    return pd.read_csv(path, **read_kwargs)


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """
    Save DataFrame as Parquet, creating parent dirs if necessary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
