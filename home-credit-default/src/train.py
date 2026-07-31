"""Model training utilities."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import N_CV_FOLDS, RANDOM_STATE


def make_lightgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def cv_auc(model: Any, X: pd.DataFrame, y: pd.Series, folds: int = N_CV_FOLDS) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return {
        "mean_auc": float(np.mean(scores)),
        "std_auc": float(np.std(scores)),
        "fold_scores": scores.tolist(),
    }


def holdout_auc(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, val_proba))
