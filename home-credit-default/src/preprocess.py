"""Basic preprocessing for the application table."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import ID_COL, TARGET_COL


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    return X, y


def prepare_for_tree_models(X: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categoricals and median-fill numerics for tree boosting models."""
    out = X.copy()
    for col in out.select_dtypes(include=["object", "category"]).columns:
        out[col] = out[col].astype("category").cat.add_categories("Missing").fillna("Missing").cat.codes
    for col in out.select_dtypes(include=["number"]).columns:
        out[col] = out[col].fillna(out[col].median())
    return out


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
