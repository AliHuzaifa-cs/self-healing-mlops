"""
Utility: restore luck_features.csv from backup (undo any injected failure).
Run this BEFORE injecting a new failure, or after a demo to reset state.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features.csv"
BACKUP_PATH = PROJECT_ROOT / "data" / "processed" / "luck_features_backup.csv"


def ensure_backup_exists():
    if not BACKUP_PATH.exists():
        shutil.copy(DATA_PATH, BACKUP_PATH)
        print(f"Backup created: {BACKUP_PATH}")
    else:
        print(f"Backup already exists: {BACKUP_PATH}")


def restore():
    if not BACKUP_PATH.exists():
        print("No backup found! Run feature_engineering.py again to regenerate clean data.")
        return
    shutil.copy(BACKUP_PATH, DATA_PATH)
    print(f"Restored {DATA_PATH} from backup — clean state.")


if __name__ == "__main__":
    ensure_backup_exists()
    restore()