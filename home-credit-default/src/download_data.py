"""Download Home Credit Default Risk data via kagglehub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import COMPETITION, DATA_RAW


def _kaggle_credentials_configured() -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "kaggle.json").exists():
        return True
    if (kaggle_dir / "access_token").exists():
        return True
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _print_setup_instructions() -> None:
    print(
        "\nKaggle credentials not found.\n"
        "1. Sign in at https://www.kaggle.com\n"
        "2. Join and accept rules: https://www.kaggle.com/competitions/home-credit-default-risk/rules\n"
        "3. Create an API token at https://www.kaggle.com/settings\n"
        "4. Save it as ~/.kaggle/kaggle.json (or run `python -c \"import kagglehub; kagglehub.login()\"`)\n"
        "5. Run: chmod 600 ~/.kaggle/kaggle.json\n"
        "6. Re-run: python src/download_data.py\n"
    )


def download_competition_data(
    output_dir: Path = DATA_RAW,
    force_download: bool = False,
) -> Path:
    """Download competition files into the project data folder."""
    if not _kaggle_credentials_configured():
        _print_setup_instructions()
        raise FileNotFoundError("Kaggle credentials not configured")

    import shutil

    import kagglehub

    output_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.competition_download(
        COMPETITION,
        force_download=force_download,
    )
    # kagglehub downloads to its own cache; copy files into the project data folder
    src_path = Path(path)
    for f in src_path.iterdir():
        if f.is_file():
            shutil.copy2(f, output_dir / f.name)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Home Credit Kaggle dataset")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files are already cached",
    )
    args = parser.parse_args()

    try:
        path = download_competition_data(force_download=args.force)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    csv_files = sorted(path.glob("*.csv"))
    print(f"Download complete: {path}")
    print(f"CSV files ({len(csv_files)}): {', '.join(f.name for f in csv_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
