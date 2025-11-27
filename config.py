from pathlib import Path
import logging
import os

PROJ_ROOT = Path(__file__).resolve().parents[0]

DATA_DIR = PROJ_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "0_raw"
DATA_CLEANED_DIR = DATA_DIR / "1_cleaned"
DATA_PREPROCESSED_DIR = DATA_DIR / "2_preprocessed"
OUTPUTS_PATH = PROJ_ROOT / "outputs"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(DATA_RAW_DIR):
    os.makedirs(DATA_RAW_DIR)

if not os.path.exists(DATA_CLEANED_DIR):
    os.makedirs(DATA_CLEANED_DIR)

if not os.path.exists(DATA_PREPROCESSED_DIR):
    os.makedirs(DATA_PREPROCESSED_DIR)

AISLES_PATH = DATA_CLEANED_DIR / "aisles.pq"
DEPARTMENTS_PATH = DATA_CLEANED_DIR / "departments.pq"
ORDER_PRODUCTS__TRAIN_PATH = DATA_CLEANED_DIR / "order_products__train.pq"
ORDER_PRODUCTS__VAL_PATH = DATA_CLEANED_DIR / "order_products__val.pq"
ORDER_PRODUCTS__TEST_PATH = DATA_CLEANED_DIR / "order_products__test.pq"
ORDERS_PATH = DATA_CLEANED_DIR / "orders.pq"
PRODUCTS_PATH = DATA_CLEANED_DIR / "products.pq"

AISLES_PATH_CSV = DATA_RAW_DIR / "aisles.csv"
DEPARTMENTS_PATH_CSV = DATA_RAW_DIR / "departments.csv"
ORDER_PRODUCTS__PRIOR_PATH_CSV = DATA_RAW_DIR / "order_products__prior.csv"
ORDER_PRODUCTS__TRAIN_PATH_CSV = DATA_RAW_DIR / "order_products__train.csv"
ORDERS_PATH_CSV = DATA_RAW_DIR / "orders.csv"
PRODUCTS_PATH_CSV = DATA_RAW_DIR / "products.csv"

UNIQUE_USERS_PATH = DATA_PREPROCESSED_DIR / "unique_users.pq"
AISLE_TOP_PRODUCTS_PATH = DATA_PREPROCESSED_DIR / "aisle_top_products.pq"

CF_ALL_USERS_ALL_ITEMS_EXP_PATH = OUTPUTS_PATH / "cf_all_users_all_items_experiment_results.csv"
CF_ALL_USERS_AISLES_EXP_PATH = OUTPUTS_PATH / "cf_all_users_aisles_experiment_results.csv"
CF_USER_CLUSTERS_ALL_ITEMS_EXP_PATH = OUTPUTS_PATH / "cf_user_clusters_all_items_experiment_results.csv"
CF_USER_CLUSTERS_AISLES_EXP_PATH = OUTPUTS_PATH / "cf_user_clusters_aisles_experiment_results.csv"
MBA_EXP_PATH = OUTPUTS_PATH / "mba_experiment_results.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s\n"
)