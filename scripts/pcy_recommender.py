import csv
import os
import sys
import pickle
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

# Allow imports from project root when running as a script
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import *
from utils import *

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading PCY artefacts (Pass 1 parameters + Pass 2 frequent pairs)
# ---------------------------------------------------------------------------

def load_pcy_artifacts():
    """
    Load PCY hyperparameters (from Pass 1) and frequent pairs (from Pass 2).
    Assumes files are stored under DATA_PREPROCESSED_DIR / "pcy".
    """
    pcy_dir = DATA_PREPROCESSED_DIR / "pcy"

    params_path = pcy_dir / "pcy_params.pkl"
    pairs_path = pcy_dir / "pcy_pass2_frequent_pairs.pkl"

    if not params_path.exists():
        raise FileNotFoundError(f"PCY params not found at {params_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"PCY frequent pairs not found at {pairs_path}")

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    with open(pairs_path, "rb") as f:
        pairs_data = pickle.load(f)

    frequent_pairs = pairs_data["frequent_pairs"]      # dict[(i,j)] -> count
    item_counts = pairs_data["item_counts"]            # dict[item] -> count

    num_baskets = params["num_baskets"]

    logger.info(
        "Loaded PCY artefacts: %d frequent pairs, %d items, %d baskets",
        len(frequent_pairs),
        len(item_counts),
        num_baskets,
    )

    return frequent_pairs, item_counts, num_baskets


# ---------------------------------------------------------------------------
# Rule construction from frequent pairs
# ---------------------------------------------------------------------------

def build_rule_dataframe(frequent_pairs, item_counts, num_baskets):
    """
    Convert frequent_pairs + item_counts into a DataFrame with
    support, confidence and lift for both A→B and B→A.
    """
    rows = []

    for (a, b), count_ab in frequent_pairs.items():
        support_ab = count_ab / num_baskets

        support_a = item_counts[a] / num_baskets
        support_b = item_counts[b] / num_baskets

        # Basic guards against division by zero (should not happen in practice)
        if support_a == 0 or support_b == 0:
            continue

        confidence_a_b = support_ab / support_a
        confidence_b_a = support_ab / support_b
        lift_ab = support_ab / (support_a * support_b)

        rows.append(
            {
                "antecedent": a,
                "consequent": b,
                "support": support_ab,
                "confidence": confidence_a_b,
                "lift": lift_ab,
            }
        )
        rows.append(
            {
                "antecedent": b,
                "consequent": a,
                "support": support_ab,
                "confidence": confidence_b_a,
                "lift": lift_ab,
            }
        )

    rules_df = pd.DataFrame(rows)
    logger.info("Constructed %d association rules from frequent pairs", len(rules_df))

    return rules_df


def build_rule_map(rules_df, min_confidence=0.01, min_lift=1.0, top_n=None):
    """
    Build item -> list of (consequent, lift, confidence) sorted by lift.
    Optional thresholds can be used to trim very weak rules.
    """
    rule_map = defaultdict(list)

    mask = (rules_df["confidence"] >= min_confidence) & (rules_df["lift"] >= min_lift)
    filtered = rules_df.loc[mask]

    for _, row in filtered.iterrows():
        a = int(row["antecedent"])
        b = int(row["consequent"])
        lift = float(row["lift"])
        conf = float(row["confidence"])
        rule_map[a].append((b, lift, conf))

    # sort and optionally truncate
    for a, lst in rule_map.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None and len(lst) > top_n:
            rule_map[a] = lst[:top_n]

    logger.info("Built rule_map for %d antecedent items", len(rule_map))

    return rule_map


# ---------------------------------------------------------------------------
# User history and recommenders
# ---------------------------------------------------------------------------

def build_user_history():
    """
    Build user -> set(product_id) from the TRAIN split.
    We combine ORDER_PRODUCTS__TRAIN with ORDERS to get user_id.
    """
    # Load orders and restrict to train split
    orders = pd.read_parquet(ORDERS_PATH)
    orders_train = orders.loc[orders["eval_set"] == "train", ["order_id", "user_id"]]

    # Load order_products for train (already filtered by order_id in data_split)
    op_train = pd.read_parquet(ORDER_PRODUCTS__TRAIN_PATH)

    # Merge to attach user_id to each (order_id, product_id)
    op_train = op_train.merge(orders_train, on="order_id", how="inner")

    # Build user → set of products
    user_baskets = (
        op_train.groupby("user_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    logger.info("Built user history for %d users", len(user_baskets))
    return user_baskets



def recommend_for_user(user_id, user_history, rule_map, popular_items, k=6):
    """
    Return a ranked list of exactly k product_ids for a given user.
    - First uses PCY association rules (lift-based).
    - Then fills up with globally popular items as fallback.
    """
    purchased = user_history.get(user_id, set())
    if not isinstance(purchased, set):
        purchased = set(purchased)

    scores = defaultdict(float)

    # Rule-based recommendations from PCY
    for a in purchased:
        if a not in rule_map:
            continue
        for b, lift, _ in rule_map[a]:
            if b in purchased:
                continue
            scores[b] += lift

    # Sort rule-based candidates by score
    recs = []
    if scores:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recs = [item for item, _ in ranked[:k]]

    # Fallback: fill up to k with popular items not yet recommended or purchased
    if len(recs) < k:
        for item in popular_items:
            if item in purchased:
                continue
            if item in recs:
                continue
            recs.append(item)
            if len(recs) == k:
                break

    return recs[:k]


def build_recs_dict(rule_map, user_history, target_users, popular_items, k=6):
    """
    Build user -> list[product_id] recommendations for a given
    set of target_users. Ensures exactly k recommendations per user
    by using rule-based scores plus popular-item fallback.
    """
    recs_dict = {}

    for user in target_users:
        recs = recommend_for_user(
            user_id=user,
            user_history=user_history,
            rule_map=rule_map,
            popular_items=popular_items,
            k=k,
        )
        recs_dict[user] = recs

    logger.info(
        "Built recommendations for %d users (k=%d)", len(recs_dict), k
    )
    return recs_dict


# ---------------------------------------------------------------------------
# Explainable recommendations (optional, for inspection and examples)
# ---------------------------------------------------------------------------

def load_product_names():
    """Return dict[item_id] -> product_name from products.pq."""
    products = pd.read_parquet(PRODUCTS_PATH)
    return dict(zip(products["product_id"], products["product_name"]))


def recommend_with_explanations(user_id, user_history, rule_map, k=6):
    """
    Same recommender as recommend_for_user, but also returns
    a dict[item_id] -> list of textual explanations.
    Intended for debugging / examples, not for pipeline use.
    """
    id_to_name = load_product_names()
    purchased = user_history.get(user_id)
    if not purchased:
        return [], {}

    scores = defaultdict(float)
    reasons = defaultdict(list)

    for a in purchased:
        if a not in rule_map:
            continue
        for b, lift, conf in rule_map[a]:
            if b in purchased:
                continue
            scores[b] += lift
            reasons[b].append((a, lift, conf))

    if not scores:
        return [], {}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    recs = [item for item, _ in ranked]

    explanations = {}
    for item, _ in ranked:
        parts = []
        for a, lift, conf in reasons[item]:
            parts.append(
                f"because you bought '{id_to_name.get(a, str(a))}' "
                f"(lift={lift:.2f}, conf={conf:.2f})"
            )
        explanations[item] = parts

    return recs, explanations


# ---------------------------------------------------------------------------
# Custom evaluation metrics (precision, coverage, novelty, diversity)
# ---------------------------------------------------------------------------

def build_product_to_users(user_history):
    """item -> set(user_id) mapping used for diversity computation."""
    mapping = defaultdict(set)
    for user, items in user_history.items():
        for item in items:
            mapping[item].add(user)
    return mapping


def evaluate_custom_metrics(
    recs_dict,
    user_history,
    future_baskets,
    product_to_users,
    k=6,
    sample_size=200,
    random_state=42,
):
    """
    Compute precision@k, coverage@k, novelty@k and diversity@k
    on a (possibly sampled) set of users.
    """
    rng = np.random.default_rng(random_state)

    users_with_future = set(future_baskets.keys())
    users_with_recs = set(recs_dict.keys())
    candidate_users = sorted(users_with_future & users_with_recs)

    if not candidate_users:
        logger.warning("No overlap between users in recs and future baskets")
        return {
            "precision": 0.0,
            "coverage": 0.0,
            "novelty": 0.0,
            "diversity": 0.0,
            "n_users": 0,
        }

    if sample_size is not None and sample_size < len(candidate_users):
        eval_users = rng.choice(candidate_users, size=sample_size, replace=False)
    else:
        eval_users = candidate_users

    eval_users = list(eval_users)

    precision_sum = 0.0
    users_with_recommendations = 0
    total_recommended = 0
    total_novel = 0
    diversity_per_user = []

    for user in eval_users:
        recs = recs_dict.get(user, [])[:k]
        future = set(future_baskets.get(user, []))
        history = user_history.get(user, set())

        if recs:
            users_with_recommendations += 1
        else:
            continue  # user contributes 0 to all metrics

        recs_set = set(recs)
        hits = len(recs_set & future)
        precision_sum += hits / k

        total_recommended += len(recs)
        total_novel += sum(1 for item in recs if item not in history)

        # Diversity: average 1 - Jaccard over item pairs in recs
        if len(recs) >= 2:
            pair_scores = []
            for i, j in combinations(recs, 2):
                users_i = product_to_users.get(i, set())
                users_j = product_to_users.get(j, set())
                union = users_i | users_j
                if not union:
                    continue
                inter = users_i & users_j
                jacc = len(inter) / len(union)
                pair_scores.append(1.0 - jacc)
            if pair_scores:
                diversity_per_user.append(float(np.mean(pair_scores)))

    n_eval = len(eval_users)
    if n_eval == 0 or users_with_recommendations == 0 or total_recommended == 0:
        return {
            "precision": 0.0,
            "coverage": 0.0,
            "novelty": 0.0,
            "diversity": 0.0,
            "n_users": n_eval,
        }

    precision = precision_sum / users_with_recommendations
    coverage = users_with_recommendations / n_eval
    novelty = total_novel / total_recommended
    diversity = float(np.mean(diversity_per_user)) if diversity_per_user else 0.0

    return {
        "precision": precision,
        "coverage": coverage,
        "novelty": novelty,
        "diversity": diversity,
        "n_users": n_eval,
    }


# ---------------------------------------------------------------------------
# Main: build rules, generate recs, evaluate with project metrics + custom ones
# ---------------------------------------------------------------------------

def main():
    # 1) Load PCY outputs
    logging.info("Loading PCY artifacts")
    frequent_pairs, item_counts, num_baskets = load_pcy_artifacts()
    
    # Global popularity (for fallback recommendations)
    popular_items = [
        item for item, cnt in sorted(
            item_counts.items(), key=lambda x: x[1], reverse=True
        )
    ]


    # 2) Build rules and rule_map
    logging.info("Building rules and rule_map")
    rules_df = build_rule_dataframe(frequent_pairs, item_counts, num_baskets)
    rule_map = build_rule_map(
        rules_df,
        min_confidence=0.01,
        min_lift=1.0,
        top_n=50,
    )

    # 3) Build user history (TRAIN)
    logging.info("Building user history")
    user_history = build_user_history()

    # 4) Determine target users (validation users) and build recs_dict
    logging.info("Building recs_dict")
    val_ratings_path = DATA_PREPROCESSED_DIR / "val_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq"
    val_ratings = pd.read_parquet(val_ratings_path)
    val_products = construct_test_product_dict(mode="val")  # user -> list[product_id]

    target_users = sorted(val_products.keys())
    k = 6
    recs_dict = build_recs_dict(
            rule_map=rule_map,
            user_history=user_history,
            target_users=target_users,
            popular_items=popular_items,
            k=k,
        )

    # 5) Project-level evaluation: hit-rate@6 and ndcg@6 (aligned with other recommenders)
    logging.info("Evaluating PCY recommendations")
    eval_dict = eval_recs(
        recs_dict=recs_dict,
        rating_df=val_ratings,
        test_products=val_products,
    )
    avg_hr = np.mean([metrics["hit-rate"] for metrics in eval_dict.values()])
    avg_ndcg = np.mean([metrics[f"ndcg@{k}"] for metrics in eval_dict.values()])

    print(f"[PCY] Average hit-rate@{k}: {avg_hr:.6f}")
    print(f"[PCY] Average ndcg@{k}:   {avg_ndcg:.6f}")

    row = ["pcy", 6, avg_hr, avg_ndcg]
    with open(RESULTS_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    # 6) Custom evaluation metrics (precision, coverage, novelty, diversity)
    future_baskets = {u: set(items) for u, items in val_products.items()}
    product_to_users = build_product_to_users(user_history)

    custom_metrics = evaluate_custom_metrics(
        recs_dict=recs_dict,
        user_history=user_history,
        future_baskets=future_baskets,
        product_to_users=product_to_users,
        k=k,
        sample_size=200,
        random_state=42,
    )

    print(
        f"[PCY] Custom metrics on sample of {custom_metrics['n_users']} users "
        f"(k={k}):"
    )
    print(
        f"  precision@{k}: {custom_metrics['precision']:.4f}, "
        f"coverage@{k}: {custom_metrics['coverage']:.4f}, "
        f"novelty@{k}: {custom_metrics['novelty']:.4f}, "
        f"diversity@{k}: {custom_metrics['diversity']:.4f}"
    )


if __name__ == "__main__":
    main()
