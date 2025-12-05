import os
import sys
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import logging

# Add project root (parent of scripts/) to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ORDERS_PATH,
    ORDER_PRODUCTS__TRAIN_PATH,
    DATA_PREPROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def load_train_data() -> pd.DataFrame:
    """Load train orders and order_products in line with the project splits."""
    orders = pd.read_parquet(ORDERS_PATH)
    op_train = pd.read_parquet(ORDER_PRODUCTS__TRAIN_PATH)

    train_order_ids = orders.loc[orders["eval_set"] == "train", "order_id"]
    op_train = op_train[op_train["order_id"].isin(train_order_ids)]

    return op_train


def build_baskets_and_item_counts(op_train: pd.DataFrame):
    """
    Build order-level baskets and global item frequency counts.
    Each basket is a set of product_ids for one order_id.
    """
    baskets_series = (
        op_train.groupby("order_id")["product_id"]
        .apply(set)
    )
    baskets = list(baskets_series)

    item_counts = Counter()
    for b in baskets:
        item_counts.update(b)

    return baskets, item_counts


def run_pass1(
    support_percentages=None,
    bucket_multipliers=None,
    save=True,
):
    """
    PCY Pass 1:
    - Build baskets + item_counts from TRAIN
    - Run grid over (bucket_multipliers × support_percentages)
    - Compute number of frequent buckets per combination
    - Save results to DATA_PREPROCESSED_DIR / 'pcy'
    """
    if support_percentages is None:
        support_percentages = [0.001, 0.002, 0.005, 0.01]  # 0.1%, 0.2%, 0.5%, 1%

    if bucket_multipliers is None:
        bucket_multipliers = [1.0, 1.5, 2.0, 3.0]

    logger.info("Loading train data for PCY Pass 1")
    op_train = load_train_data()

    logger.info("Building baskets and item counts")
    baskets, item_counts = build_baskets_and_item_counts(op_train)
    num_baskets = len(baskets)
    num_unique_items = len(item_counts)

    logger.info(f"Number of baskets: {num_baskets}")
    logger.info(f"Number of unique items: {num_unique_items}")

    # Bucket sizes from multipliers
    bucket_options = [int(num_unique_items * m) for m in bucket_multipliers]
    logger.info(f"Bucket sizes: {bucket_options}")

    # Results matrix: rows = bucket_options, cols = support_percentages
    results = np.zeros((len(bucket_options), len(support_percentages)), dtype=int)

    # Grid search: for each bucket size, one pass over baskets
    for i, num_buckets in enumerate(bucket_options):
        logger.info(f"Running Pass 1 for num_buckets={num_buckets}")

        bucket_counts = np.zeros(num_buckets, dtype=np.int32)

        # Ccount all pairs per bucket size
        for basket in baskets:
            b = sorted(basket)
            for a in range(len(b)):
                ia = b[a]
                for c in range(a + 1, len(b)):
                    ib = b[c]
                    h = (ia * 13 + ib * 7) % num_buckets
                    bucket_counts[h] += 1

        # Different support thresholds reuse the same bucket_counts
        for j, sp in enumerate(support_percentages):
            support_threshold = int(num_baskets * sp)
            frequent_buckets = int((bucket_counts >= support_threshold).sum())
            results[i, j] = frequent_buckets

            logger.info(
                f"num_buckets={num_buckets}, support={sp:.4f} "
                f"→ threshold={support_threshold}, frequent_buckets={frequent_buckets}"
            )

    logger.info("PCY Pass 1 grid search finished")

    if save:
        pcy_dir = DATA_PREPROCESSED_DIR / "pcy"
        os.makedirs(pcy_dir, exist_ok=True)
        save_path = pcy_dir / "pcy_pass1_results.pkl"

        save_data = {
            "num_baskets": num_baskets,
            "num_unique_items": num_unique_items,
            "bucket_multipliers": bucket_multipliers,
            "support_percentages": support_percentages,
            "bucket_options": bucket_options,
            "results": results,
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Pass 1 results saved to {save_path}")

        # Save selected PCY hyperparameters for Pass 2
        chosen_multiplier = 1.5        # from Pass 1 analysis
        chosen_support = 0.005         # 0.5%

        chosen_num_buckets = int(num_unique_items * chosen_multiplier)

        params = {
            "bucket_multiplier": chosen_multiplier,
            "support_percentage": chosen_support,
            "num_buckets": chosen_num_buckets,
            "num_baskets": num_baskets,
            "num_unique_items": num_unique_items,
        }

        params_path = pcy_dir / "pcy_params.pkl"
        with open(params_path, "wb") as f:
            pickle.dump(params, f)

        logger.info(f"Pass 1 selected params saved to {params_path}")

    return {
        "num_baskets": num_baskets,
        "num_unique_items": num_unique_items,
        "bucket_multipliers": bucket_multipliers,
        "support_percentages": support_percentages,
        "bucket_options": bucket_options,
        "results": results,
    }


def run_pass2(save=True):
    """
    PCY Pass 2:
    - Reload TRAIN baskets + item_counts
    - Load chosen PCY hyperparameters from pcy_params.pkl
    - Build bitmap (frequent buckets) for chosen num_buckets & support
    - Count frequent pairs using (frequent singles + bitmap)
    - Save frequent_pairs and item_counts for downstream use
    """
    pcy_dir = DATA_PREPROCESSED_DIR / "pcy"
    params_path = pcy_dir / "pcy_params.pkl"

    if not params_path.exists():
        raise FileNotFoundError(
            f"PCY params not found at {params_path}. "
            "Run Pass 1 first to generate pcy_params.pkl."
        )

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    bucket_multiplier = params["bucket_multiplier"]
    support_percentage = params["support_percentage"]

    logger.info("Loading train data for PCY Pass 2")
    op_train = load_train_data()

    logger.info("Building baskets and item counts")
    baskets, item_counts = build_baskets_and_item_counts(op_train)
    num_baskets = len(baskets)
    num_unique_items = len(item_counts)

    logger.info(f"Number of baskets (Pass 2): {num_baskets}")
    logger.info(f"Number of unique items (Pass 2): {num_unique_items}")

    # Derive final hyperparameters exactly as στο notebook:
    num_buckets = int(num_unique_items * bucket_multiplier)
    support_threshold = int(num_baskets * support_percentage)

    logger.info(
        f"Pass 2 using num_buckets={num_buckets}, "
        f"support={support_percentage:.4f} "
        f"→ threshold={support_threshold}"
    )

    # Frequent single items with respect to final support threshold
    frequent_single_items = {
        item for item, cnt in item_counts.items()
        if cnt >= support_threshold
    }
    logger.info(f"Number of frequent single items: {len(frequent_single_items)}")

    # ---- First sweep: bucket counts (for bitmap) ----
    bucket_counts = np.zeros(num_buckets, dtype=np.int32)

    for basket in baskets:
        # keep only frequent singles to reduce work
        filtered = [it for it in basket if it in frequent_single_items]
        if len(filtered) < 2:
            continue
        filtered.sort()

        for i in range(len(filtered)):
            a = filtered[i]
            for j in range(i + 1, len(filtered)):
                b = filtered[j]
                h = (a * 13 + b * 7) % num_buckets
                bucket_counts[h] += 1

    bitmap = bucket_counts >= support_threshold
    n_frequent_buckets = int(bitmap.sum())
    logger.info(f"Number of frequent buckets in Pass 2: {n_frequent_buckets}")

    # ---- Second sweep: count candidate pairs ----
    pair_counts = Counter()

    for basket in baskets:
        filtered = [it for it in basket if it in frequent_single_items]
        if len(filtered) < 2:
            continue
        filtered.sort()

        for i in range(len(filtered)):
            a = filtered[i]
            for j in range(i + 1, len(filtered)):
                b = filtered[j]
                h = (a * 13 + b * 7) % num_buckets

                # PCY condition: bucket must be frequent
                if not bitmap[h]:
                    continue

                pair_counts[(a, b)] += 1

    # Final frequent pairs
    frequent_pairs = {
        pair: cnt for pair, cnt in pair_counts.items()
        if cnt >= support_threshold
    }
    logger.info(f"Number of frequent pairs: {len(frequent_pairs)}")

    if save:
        os.makedirs(pcy_dir, exist_ok=True)
        save_path = pcy_dir / "pcy_pass2_frequent_pairs.pkl"

        save_data = {
            "frequent_pairs": frequent_pairs,
            "item_counts": dict(item_counts),
            "support_threshold": support_threshold,
            "support_percentage": support_percentage,
            "num_buckets": num_buckets,
            "num_baskets": num_baskets,
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Pass 2 frequent pairs saved to {save_path}")

    return frequent_pairs, item_counts


def main():
    # Full PCY training pipeline: Pass 1 (tuning) + Pass 2 (frequent pairs)
    run_pass1()
    run_pass2()
    print("PCY Pass 1 & Pass 2 completed and results saved.")


if __name__ == "__main__":
    main()
