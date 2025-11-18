# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


@dataclass
class TrainedModelBundle:
    """
    Container for a trained model and preprocessor.
    """
    model: object
    pipeline: Pipeline
    best_params: Dict


def build_logistic_pipeline(
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Pipeline: preprocessor -> LogisticRegression.
    """
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
    )
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def build_rf_pipeline(
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Pipeline: preprocessor -> RandomForest.
    """
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def build_xgb_pipeline(
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Pipeline: preprocessor -> XGBoost classifier.
    """
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe


def train_with_grid_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> TrainedModelBundle:
    """
    Generic grid search wrapper to train a pipeline.

    param_grid keys must use 'clf__' prefix (e.g. 'clf__C', 'clf__max_depth').
    """
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    best_model = best_pipe.named_steps["clf"]
    best_params = grid.best_params_

    return TrainedModelBundle(
        model=best_model,
        pipeline=best_pipe,
        best_params=best_params,
    )


def evaluate_roc_auc(
    bundle: TrainedModelBundle,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """
    Quick ROC-AUC evaluation for a trained pipeline.
    """
    y_pred_proba = bundle.pipeline.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred_proba)


def save_model_bundle(bundle: TrainedModelBundle, path: str | Path) -> None:
    """
    Save the whole pipeline (incl. preprocessor) with joblib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.pipeline, path)


def load_pipeline(path: str | Path) -> Pipeline:
    """
    Load a saved pipeline.
    """
    path = Path(path)
    return joblib.load(path)
