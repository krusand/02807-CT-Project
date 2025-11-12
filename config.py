from pathlib import Path
import logging

PROJ_ROOT = Path(__file__).resolve().parents[0]

DATA_DIR = PROJ_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_CLEANED_DIR = DATA_DIR / "cleaned"
DATA_PREPROCESSED_DIR = DATA_DIR / "preprocessed"

AISLES_PATH = DATA_CLEANED_DIR / "aisles.pq"
DEPARTMENTS_PATH = DATA_CLEANED_DIR / "departments.pq"
ORDER_PRODUCTS__PRIOR_PATH = DATA_CLEANED_DIR / "order_products__prior.pq"
ORDER_PRODUCTS__TRAIN_PATH = DATA_CLEANED_DIR / "order_products__train.pq"
ORDERS_PATH = DATA_CLEANED_DIR / "orders.pq"
PRODUCTS_PATH = DATA_CLEANED_DIR / "products.pq"


