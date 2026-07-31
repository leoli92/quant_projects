"""Starter feature engineering for the application table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a few high-signal ratio features on the main application table."""
    out = df.copy()

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)

    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(out.columns):
        out["CREDIT_TERM"] = out["AMT_CREDIT"] / out["AMT_ANNUITY"].replace(0, np.nan)

    if "DAYS_BIRTH" in out.columns:
        out["AGE_YEARS"] = (-out["DAYS_BIRTH"] / 365.25).astype(float)

    if "DAYS_EMPLOYED" in out.columns:
        employed = out["DAYS_EMPLOYED"].replace(365243, np.nan)
        out["EMPLOYED_YEARS"] = (-employed / 365.25).astype(float)

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out.columns]
    if ext_cols:
        out["EXT_SOURCE_MEAN"] = out[ext_cols].mean(axis=1)
        out["EXT_SOURCE_MISSING"] = out[ext_cols].isna().sum(axis=1)

    out["MISSING_COUNT"] = out.isna().sum(axis=1)
    return out
