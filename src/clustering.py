# src/clustering.py
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scale_features(
    df_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise feature DataFrame.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_features.values)
    df_scaled = pd.DataFrame(
        scaled_array,
        columns=df_features.columns,
        index=df_features.index,
    )
    return df_scaled, scaler


def kmeans_elbow(
    df_features: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[int, float]:
    """
    Compute inertia for different k to plot elbow curve.
    """
    inertia = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(df_features)
        inertia[k] = km.inertia_
    return inertia


def kmeans_silhouette(
    df_features: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[int, float]:
    """
    Compute silhouette scores for different k.
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(df_features)
        scores[k] = silhouette_score(df_features, labels)
    return scores


def fit_kmeans(
    df_features: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """
    Fit KMeans and return fitted model + labels.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(df_features)
    return km, labels


def characterise_segments(
    df: pd.DataFrame,
    cluster_col: str,
    agg_cols: List[str],
) -> pd.DataFrame:
    """
    Return a summary table of each cluster over selected columns.
    """
    summary = df.groupby(cluster_col)[agg_cols].mean().reset_index()
    summary["count"] = df.groupby(cluster_col)[cluster_col].count().values
    return summary
