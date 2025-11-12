from pathlib import Path
import logging

PROJ_ROOT = Path(__file__).resolve().parents[0]

DATA_DIR = PROJ_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_CLEANED_DIR = DATA_DIR / "cleaned"
DATA_PREPROCESSED_DIR = DATA_DIR / "preprocessed"



