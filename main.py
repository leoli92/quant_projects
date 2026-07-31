"""Run initial EDA and baseline models on application_train.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from config import OUTPUTS, RANDOM_STATE, TARGET_COL  # noqa: E402
from features import add_application_features  # noqa: E402
from load_data import load_test, load_train, train_test_summary  # noqa: E402
from preprocess import build_preprocessor, prepare_for_tree_models, split_features_target  # noqa: E402
from train import cv_auc, holdout_auc, make_lightgbm  # noqa: E402


def save_missing_summary(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing.to_csv(path, index=False)
    return missing


def plot_target_distribution(y: pd.Series, path: Path) -> None:
    ax = sns.countplot(x=y.map({0: "Repaid", 1: "Default"}))
    ax.set_title("Target distribution")
    ax.set_xlabel("Outcome")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, path: Path, max_features: int = 20) -> None:
    numeric = df.select_dtypes(include="number")
    if TARGET_COL in numeric.columns:
        top_cols = (
            numeric.corrwith(numeric[TARGET_COL])
            .abs()
            .sort_values(ascending=False)
            .head(max_features)
            .index.tolist()
        )
    else:
        top_cols = numeric.columns[:max_features].tolist()

    corr = numeric[top_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (top numeric features)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    train = add_application_features(load_train())
    test = add_application_features(load_test())

    summary = train_test_summary(train, test)
    print("Dataset summary:")
    print(json.dumps(summary, indent=2))

    save_missing_summary(train, OUTPUTS / "missing_value_summary.csv")
    plot_target_distribution(train[TARGET_COL], OUTPUTS / "target_distribution.png")
    plot_correlation_heatmap(train, OUTPUTS / "correlation_heatmap.png")

    X, y = split_features_target(train)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    from sklearn.pipeline import Pipeline

    logistic_model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    lgbm_model = make_lightgbm()

    X_train_lgbm = prepare_for_tree_models(X_train)
    X_val_lgbm = prepare_for_tree_models(X_val)

    results = {
        "logistic_regression": {
            "cv": cv_auc(logistic_model, X_train, y_train),
            "holdout_auc": holdout_auc(logistic_model, X_train, y_train, X_val, y_val),
        },
        "lightgbm": {
            "cv": cv_auc(lgbm_model, X_train_lgbm, y_train),
            "holdout_auc": holdout_auc(lgbm_model, X_train_lgbm, y_train, X_val_lgbm, y_val),
        },
    }

    with open(OUTPUTS / "baseline_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("Baseline results:")
    print(json.dumps(results, indent=2))
    print(f"Outputs written to {OUTPUTS}")


if __name__ == "__main__":
    main()
