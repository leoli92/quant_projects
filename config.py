"""Project paths and constants for Home Credit Default Risk."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"

COMPETITION = "home-credit-default-risk"
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

TRAIN_FILE = "application_train.csv"
TEST_FILE = "application_test.csv"
COLUMNS_DESC_FILE = "HomeCredit_columns_description.csv"

RANDOM_STATE = 42
N_CV_FOLDS = 5
