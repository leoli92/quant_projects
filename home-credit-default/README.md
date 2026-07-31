# Home Credit Default Risk

Supervised ML pipeline for predicting loan default risk using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition dataset. The target is binary (`TARGET`), and the primary metric is **AUC-ROC**.

---

## Project Structure

```
home-credit-default/
├── config.py                  # Paths, constants, CV settings
├── data/
│   ├── raw/                   # Kaggle CSVs (gitignored)
│   └── processed/             # Engineered feature tables
├── src/
│   ├── download_data.py       # kagglehub competition download
│   ├── load_data.py           # Read application tables
│   ├── preprocess.py          # Imputation / encoding helpers
│   ├── features.py            # Application-level feature engineering
│   └── train.py               # CV and holdout AUC utilities
├── notebooks/                 # EDA and model experiments
├── outputs/                   # Plots, metrics, submission files
├── main.py                    # EDA + baseline model runner
└── requirements.txt
```

---

## Setup

### 1. Create virtual environment

```bash
cd "/Users/leoli/Desktop/Quant Projects/home-credit-default"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Kaggle credentials

1. Sign in at [kaggle.com](https://www.kaggle.com)
2. Join and accept rules on the [competition page](https://www.kaggle.com/competitions/home-credit-default-risk/rules)
3. Create an API token at [kaggle.com/settings](https://www.kaggle.com/settings)
4. Save it as `~/.kaggle/kaggle.json` (or run `python -c "import kagglehub; kagglehub.login()"`)

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download data

```bash
python src/download_data.py
```

This uses `kagglehub` to download all competition CSVs into `data/raw/`.

### 4. Run baseline pipeline

```bash
python main.py
```

Expected outputs in `outputs/`:
- `missing_value_summary.csv`
- `target_distribution.png`
- `correlation_heatmap.png`
- `baseline_results.json` (logistic regression vs LightGBM CV/holdout AUC)

---

## Dataset Overview

| File | Description |
|---|---|
| `application_train.csv` | Main training table (~307k rows, 122 cols) |
| `application_test.csv` | Kaggle test set (no `TARGET`) |
| `bureau.csv` | External credit bureau history |
| `previous_application.csv` | Past Home Credit applications |
| `installments_payments.csv` | Payment behavior over time |
| `HomeCredit_columns_description.csv` | Column dictionary |

Start with `application_train.csv` only, then join secondary tables for AUC gains.

---

## Roadmap

- [x] Project scaffold + Kaggle download script
- [x] Baseline EDA outputs
- [x] Logistic regression + LightGBM baseline
- [ ] Cross-validation report with ROC / PR curves
- [ ] Join bureau + previous application features
- [ ] Hyperparameter tuning with Optuna
- [ ] SHAP interpretation
- [ ] Model comparison table (RF, XGBoost, CatBoost)
- [ ] Kaggle submission file

---

## Theory Checklist

Binary classification, stratified CV, ROC/AUC, precision-recall, class imbalance, missing-value handling, categorical encoding, gradient boosting, regularization, feature importance, data leakage, and hyperparameter tuning.
