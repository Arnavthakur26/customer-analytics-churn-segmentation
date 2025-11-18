# src/data_prep.py
from __future__ import annotations

from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for typical tabular churn data.

    Numeric: impute median + standardise
    Categorical: impute most frequent + one-hot encode
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )

    return preprocessor


def stratified_train_val_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test with stratification on target.

    val_size is relative to *full* dataset, not to train.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")

    df_temp, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    relative_val_size = val_size / (1.0 - test_size)

    df_train, df_val = train_test_split(
        df_temp,
        test_size=relative_val_size,
        stratify=df_temp[target_col],
        random_state=random_state,
    )

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def time_based_train_val_test_split(
    df: pd.DataFrame,
    time_col: str,
    target_col: Optional[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split (no shuffling). Sort by time_col and cut at quantiles.

    This ignores stratification; use when temporal leakage is a concern.
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)

    n_test = int(np.floor(n * test_size))
    n_val = int(np.floor(n * val_size))
    n_train = n - n_test - n_val

    df_train = df_sorted.iloc[:n_train]
    df_val = df_sorted.iloc[n_train : n_train + n_val]
    df_test = df_sorted.iloc[n_train + n_val :]

    return df_train, df_val, df_test
