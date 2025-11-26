from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any, Hashable
import json

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# -------------------------------------------------------------------
# CONFIG: where your CSVs live + RNG seed
# -------------------------------------------------------------------
DATA_DIR = Path("data")
# must have: order_id,product_id,add_to_cart_order,reordered
ORDER_PRODUCTS_FILE = DATA_DIR / "order_products__prior.csv"
# has: order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
ORDERS_FILE = DATA_DIR / "orders.csv"
# has: product_id,product_name,...
PRODUCTS_FILE = DATA_DIR / "products.csv"

# Global random seed for reproducibility (mainly for any numpy randomness)
SEED = 42
np.random.seed(SEED)


# -------------------------------------------------------------------
# 0) Load & prepare data
# -------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Load data and return a DataFrame with at least:
        user_id, order_id, product_id, product_name, add_to_cart_order, reordered
    """
    order_products = pd.read_csv(ORDER_PRODUCTS_FILE)
    orders = pd.read_csv(ORDERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    # Keep only what we need from orders + products
    orders = orders[["order_id", "user_id"]]
    products = products[["product_id", "product_name"]]

    # Join so each row has user_id, order_id, product_id, product_name
    data = (
        order_products
        .merge(orders, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
    )

    if data["user_id"].isna().any():
        missing = data["user_id"].isna().sum()
        print(f"[WARN] {missing} rows have no user_id after join.")
    if data["product_name"].isna().any():
        missing = data["product_name"].isna().sum()
        print(f"[WARN] {missing} rows have no product_name after join.")

    print("[INFO] Loaded data with columns:", list(data.columns))
    print("[INFO] Rows:", len(data))
    return data


# -------------------------------------------------------------------
# 1) Frequent itemset mining (Apriori only, no rules here)
# -------------------------------------------------------------------
def mine_frequent_itemsets(
    transactions: List[Iterable[Hashable]],
    min_item_support: float = 0.002,
    support_grid: Optional[List[float]] = None,
    max_len: int = 3,
    use_low_memory: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """
    Run Apriori to mine frequent itemsets over the given transactions
    (which in our case are product_name lists).

    Returns:
        freq: DataFrame with 'itemsets' (frozenset of product_name) and 'support'
        chosen_support: the min_support actually used by apriori
    """
    N = len(transactions)
    if N == 0:
        raise ValueError("No transactions provided")

    if verbose:
        print(f"[INFO] Transactions N={N}")

    # 1) Pre-filter rare items BEFORE one-hot encoding
    if min_item_support is not None and min_item_support > 0:
        abs_min_count = max(1, int(np.ceil(min_item_support * N)))
        item_counts = Counter()
        for t in transactions:
            item_counts.update(set(t))  # count once per basket

        keep_items = {item for item, c in item_counts.items() if c >= abs_min_count}
        if verbose:
            U_raw = len(item_counts)
            U_kept = len(keep_items)
            print(
                f"[INFO] Unique items before filtering: {U_raw} "
                f"-> after min_item_support={min_item_support:.4g}: {U_kept}"
            )

        filtered_transactions = [
            [item for item in t if item in keep_items]
            for t in transactions
        ]
    else:
        filtered_transactions = transactions

    # 2) TransactionEncoder -> dense bool DataFrame
    te = TransactionEncoder()
    X_bool = te.fit(filtered_transactions).transform(filtered_transactions)
    X = pd.DataFrame(X_bool, columns=te.columns_, dtype=bool)
    del X_bool

    U = X.shape[1]
    if verbose:
        print(f"[INFO] After filtering & encoding: N={N}, U={U} (bool matrix)")

    # 3) Support ladder: use the lowest support that yields any itemsets
    if support_grid is None:
        base = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, max(1.0 / N, 0.001)]
        support_grid = sorted({s for s in base if s > 0}, reverse=True)
    else:
        support_grid = sorted({s for s in support_grid if s > 0}, reverse=True)

    if verbose:
        print(f"[INFO] Support grid (high -> low): {support_grid}")

    freq = pd.DataFrame()
    chosen_support = None

    last_nonempty = None
    last_s = None

    for s in support_grid:
        f = apriori(
            X,
            min_support=s,
            use_colnames=True,
            max_len=max_len,
            low_memory=use_low_memory,
        )

        lens = f["itemsets"].apply(len)
        pairs = f[lens == 2]

        if verbose:
            print(
                f"[DBG] min_support={s:.4f} -> "
                f"itemsets={len(f)} (pairs={len(pairs)})"
            )

        if len(f) > 0:
            last_nonempty = f
            last_s = s

        del f, pairs

    if last_nonempty is not None:
        freq = last_nonempty.sort_values("support", ascending=False).reset_index(drop=True)
        chosen_support = last_s

    if freq.empty or chosen_support is None:
        raise ValueError(
            "No frequent itemsets found even at lowest support. "
            "Try lowering min_item_support or adding more baskets."
        )

    lens = freq["itemsets"].apply(len)
    n1 = (lens == 1).sum()
    n2 = (lens == 2).sum()
    n3 = (lens == 3).sum()
    if verbose:
        print(
            f"[INFO] Using min_support={chosen_support:.4f}. "
            f"Singles={n1}, Pairs={n2}, Triples={n3}"
        )

    return freq, chosen_support


# -------------------------------------------------------------------
# 2) Association rule generation
# -------------------------------------------------------------------
def build_association_rules(
    freq: pd.DataFrame,
    chosen_support: Optional[float] = None,
    *,
    rule_metric: str = "confidence",
    rule_min_threshold: float = 0.05,
    # strict pruning
    min_lift: Optional[float] = 1.1,
    max_lift: Optional[float] = None,
    min_confidence: Optional[float] = 0.1,
    min_interest: Optional[float] = None,
    prune_by_support: bool = True,
    # soft mode
    min_rules: Optional[int] = None,
    soft_min_confidence: float = 0.02,
    soft_min_lift: float = 1.0,
    soft_prune_by_support: bool = False,
    # global cap
    max_rules: Optional[int] = 100000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build and prune association rules from frequent itemsets.

    max_rules:
        After sorting by ["lift","confidence","support"] (descending),
        keep only the first `max_rules`.
    """
    rules_all = association_rules(
        freq, metric=rule_metric, min_threshold=rule_min_threshold
    ).copy()

    if verbose:
        print(f"[INFO] Raw rules: {len(rules_all)}")

    if rules_all.empty:
        if verbose:
            print(
                "[HINT] No rules produced. You need at least some 2-itemsets; "
                "try lowering min_support or adjusting rule_min_threshold."
            )
        return rules_all

    # interest = confidence - support(consequent)
    lens = freq["itemsets"].apply(len)
    singles = freq[lens == 1][["itemsets", "support"]]
    conseq_support = {
        next(iter(s)): sup for s, sup in zip(singles["itemsets"], singles["support"])
    }

    def get_consequent_support(cs: frozenset) -> float:
        return conseq_support.get(next(iter(cs)), np.nan)

    rules_all["interest"] = rules_all["confidence"] - rules_all["consequents"].apply(
        get_consequent_support
    )

    # strict pruning
    strict_mask = pd.Series(True, index=rules_all.index)

    if min_lift is not None:
        strict_mask &= rules_all["lift"] >= min_lift
    if max_lift is not None:
        strict_mask &= rules_all["lift"] <= max_lift
    if prune_by_support and chosen_support is not None:
        strict_mask &= rules_all["support"] >= chosen_support
    if min_confidence is not None:
        strict_mask &= rules_all["confidence"] >= min_confidence
    if min_interest is not None:
        strict_mask &= rules_all["interest"] >= min_interest

    strict_rules = rules_all[strict_mask].copy()

    if verbose:
        print(f"[INFO] Strict pruning -> {len(strict_rules)} rules")

    # soft mode (optional; often you can just skip by min_rules=None)
    if min_rules is None or min_rules <= 0:
        selected = strict_rules
        mode = "strict-only"
    else:
        if len(strict_rules) >= min_rules:
            selected = strict_rules
            mode = "strict"
        else:
            soft_mask = pd.Series(True, index=rules_all.index)

            if soft_min_lift is not None:
                soft_mask &= rules_all["lift"] >= soft_min_lift
            if soft_min_confidence is not None:
                soft_mask &= rules_all["confidence"] >= soft_min_confidence
            if soft_prune_by_support and chosen_support is not None:
                soft_mask &= rules_all["support"] >= chosen_support

            soft_rules = rules_all[soft_mask].copy()

            if verbose:
                print(
                    f"[INFO] Strict rules ({len(strict_rules)}) < min_rules={min_rules}, "
                    f"using soft pruning -> {len(soft_rules)} rules"
                )

            selected = soft_rules
            mode = "soft"

    selected = selected.sort_values(
        ["lift", "confidence", "support"], ascending=False
    ).reset_index(drop=True)

    if max_rules is not None and max_rules > 0:
        selected = selected.head(max_rules)

    if verbose:
        print(
            f"[INFO] After {mode} pruning & capping: {len(selected)} rules "
            f"(max_rules={max_rules}, min_rules={min_rules})"
        )

    return selected


# -------------------------------------------------------------------
# 3) Helper to build rules_idx (single-item consequents)
# -------------------------------------------------------------------
def build_rules_index(
    rules: pd.DataFrame,
) -> List[Tuple[Tuple[str, ...], str, float, float, float, float]]:
    """
    Turn a rules DataFrame into a convenient index:
    list of (antecedent_tuple, consequent_item_name, lift, confidence, support, interest)
    Only keep rules with single-item consequents.
    """
    rules_idx = []
    for _, r in rules.iterrows():
        A = tuple(sorted(list(r["antecedents"])))
        C = list(r["consequents"])[0] if len(r["consequents"]) == 1 else None
        if C is None:
            continue
        rules_idx.append((A, C, r["lift"], r["confidence"], r["support"], r["interest"]))
    return rules_idx


# -------------------------------------------------------------------
# 4) Train Apriori model on a transactional DataFrame
# -------------------------------------------------------------------
def train_apriori_model(
    data: pd.DataFrame,
    user_col: str = "user_id",
    order_col: str = "order_id",
    item_col: str = "product_name",  # use product_name
    verbose: bool = True,
):
    """
    Train Apriori on all baskets in `data` and return:
      - freq: frequent itemsets (product_name)
      - rules: association rules (product_name)
      - rules_idx: indexed rules for fast lookup
      - popularity: global item popularity (Series) over product_name
    """
    # Build list of transactions (one list per order, of product_name)
    basket_groups = data.groupby(order_col)[item_col].apply(list)
    transactions = basket_groups.tolist()

    # Global item popularity (for fallback recommendations)
    popularity = (
        data[item_col]
        .value_counts()
        .sort_values(ascending=False)
    )

    # 1) Frequent itemsets
    freq, chosen_support = mine_frequent_itemsets(
        transactions=transactions,
        min_item_support=0.002,          # item pre-filter
        support_grid=[0.001],            # candidate supports for Apriori
        max_len=5,                       # up to 5-item sets
        use_low_memory=True,
        verbose=verbose,
    )

    # 2) Rules with “reasonable” thresholds
    rules = build_association_rules(
        freq=freq,
        chosen_support=chosen_support,
        rule_metric="confidence",
        rule_min_threshold=0.05,         # min confidence for generating rules
        min_lift=1.1,                    # positive association
        max_lift=20.0,                   # avoid crazy outliers
        min_confidence=0.1,              # extra confidence filter
        prune_by_support=True,           # rule support >= chosen_support
        min_rules=None,
        max_rules=100000,
        verbose=verbose,
    )

    rules_idx = build_rules_index(rules)

    if verbose:
        print(f"[DBG] rules rows: {len(rules)}")
        print(f"[DBG] rules_idx entries: {len(rules_idx)}")

    return freq, rules, rules_idx, popularity


# -------------------------------------------------------------------
# 5) Recommend items for a given user using rules + item frequencies
# -------------------------------------------------------------------
def recommend_for_user(
    user_id: Any,
    data: pd.DataFrame,
    rules_idx: List[Tuple[Tuple[str, ...], str, float, float, float, float]],
    popularity: pd.Series,
    *,
    user_col: str = "user_id",
    item_col: str = "product_name",  # we recommend names
    k: int = 6,
    subset_k: int = 2,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    Recommend K items (product_name) for a given user_id.

    NOTE: This version ALLOWS recommending items the user has already bought.
    """
    user_data = data[data[user_col] == user_id]
    if user_data.empty:
        raise ValueError(f"No data found for user_id={user_id}")

    # User's item frequency (by product_name)
    user_item_counts = user_data[item_col].value_counts()
    max_count = user_item_counts.max()
    user_item_weight = (user_item_counts / max_count).to_dict()

    # All items the user has ever bought
    user_items = sorted(user_item_counts.index.tolist())
    S = set(user_items)

    best_score = defaultdict(float)
    meta: Dict[Any, Dict[str, Any]] = {}

    # Use rules where antecedent is a small subset of user's items
    for A, C, lift_, conf_, supp_, intr_ in rules_idx:
        if len(A) <= subset_k and set(A).issubset(S):
            # We allow recommending items already in S

            # Base rule score: lift * confidence
            rule_score = lift_ * conf_

            # Antecedent weight: how core these items are for this user
            antecedent_weights = [user_item_weight.get(a, 0.0) for a in A]
            avg_ante_weight = float(np.mean(antecedent_weights)) if antecedent_weights else 0.0

            # Final score: combine rule strength and user-specific weight
            score = rule_score * (1.0 + beta * avg_ante_weight)

            if score > best_score[C]:
                best_score[C] = score
                meta[C] = {
                    "antecedent": A,
                    "lift": lift_,
                    "confidence": conf_,
                    "support": supp_,
                    "interest": intr_,
                    "antecedent_weight": avg_ante_weight,
                }

    # Turn into DataFrame
    rows = [
        (
            item,  # this is product_name
            score,
            meta[item]["antecedent"],
            meta[item]["lift"],
            meta[item]["confidence"],
            meta[item]["support"],
            meta[item]["interest"],
            meta[item]["antecedent_weight"],
        )
        for item, score in best_score.items()
    ]

    recs = pd.DataFrame(
        rows,
        columns=[
            "product_name",     # human-readable name
            "score",
            "because",          # tuple of product_name(s)
            "lift",
            "confidence",
            "support",
            "interest",
            "antecedent_weight",
        ],
    ).sort_values("score", ascending=False)

    # ---- Guarantee exactly k items (if possible) by filling from popularity ----
    already = set(recs["product_name"].tolist())

    if len(recs) < k:
        needed = k - len(recs)
        filler = [
            itm for itm in popularity.index
            if itm not in already      # avoid duplicates in final list
        ][:needed]
        if filler:
            filler_rows = [
                (itm, 0.0, tuple(), np.nan, np.nan, np.nan, np.nan, 0.0)
                for itm in filler
            ]
            filler_df = pd.DataFrame(
                filler_rows,
                columns=[
                    "product_name",
                    "score",
                    "because",
                    "lift",
                    "confidence",
                    "support",
                    "interest",
                    "antecedent_weight",
                ],
            )
            recs = pd.concat([recs, filler_df], ignore_index=True)

    recs = recs.head(k)
    return recs


# -------------------------------------------------------------------
# 5b) Recommend for multiple users (ALL users)
# -------------------------------------------------------------------
def recommend_for_users(
    user_ids: Iterable[Any],
    data: pd.DataFrame,
    rules_idx: List[Tuple[Tuple[str, ...], str, float, float, float, float]],
    popularity: pd.Series,
    *,
    user_col: str = "user_id",
    item_col: str = "product_name",
    k: int = 6,
    subset_k: int = 2,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    Recommend items (product_name) for multiple users.

    Returns a DataFrame with one row per (user_id, recommended product_name),
    including rank per user.
    """
    all_recs = []

    for uid in user_ids:
        print(uid)
        try:
            recs = recommend_for_user(
                user_id=uid,
                data=data,
                rules_idx=rules_idx,
                popularity=popularity,
                user_col=user_col,
                item_col=item_col,
                k=k,
                subset_k=subset_k,
                beta=beta,
            )
        except ValueError:
            # user has no data
            continue

        recs = recs.copy()
        recs.insert(0, "user_id", uid)
        recs["rank"] = np.arange(1, len(recs) + 1)
        all_recs.append(recs)

    if not all_recs:
        return pd.DataFrame(
            columns=[
                "user_id",
                "rank",
                "product_name",
                "score",
                "because",
                "lift",
                "confidence",
                "support",
                "interest",
                "antecedent_weight",
            ]
        )

    all_recs_df = pd.concat(all_recs, ignore_index=True)
    return all_recs_df


# -------------------------------------------------------------------
# 7) Build and save user -> [product_id] dict
# -------------------------------------------------------------------
def build_user_recs_dict(
    multi_recs: pd.DataFrame,
    data: pd.DataFrame,
    k: int = 6,
) -> Dict[Any, List[Any]]:
    """
    Convert multi-user recommendations (by product_name) into a dictionary:

        { user_id: [product_id_1, ..., product_id_k] }

    using the product_id <-> product_name mapping from `data`.
    """
    # Build mapping product_name -> product_id
    # (Assume mostly 1-to-1 in Instacart; if duplicates, first occurrence wins.)
    prod_map = (
        data[["product_id", "product_name"]]
        .dropna()
        .drop_duplicates(subset=["product_name"])
    )
    name_to_id = dict(zip(prod_map["product_name"], prod_map["product_id"]))

    user_recs: Dict[Any, List[Any]] = {}

    # Ensure per-user recommendations are ordered by rank
    for uid, grp in multi_recs.groupby("user_id"):
        grp_sorted = grp.sort_values("rank")
        names = grp_sorted["product_name"].head(k).tolist()
        ids = [name_to_id.get(n) for n in names]
        # Filter out any None (in case of missing mapping)
        ids = [pid for pid in ids if pid is not None]
        user_recs[uid] = ids

    return user_recs


# -------------------------------------------------------------------
# 6) Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load & prepare data from data/
    data = load_data()

    # 2) Train Apriori model on all users (once), using product_name
    freq, rules, rules_idx, popularity = train_apriori_model(
        data,
        user_col="user_id",
        order_col="order_id",
        item_col="product_name",
        verbose=True,
    )

    # 3) Example: recommendations for ONE user (by product_name)
    example_user_id = data["user_id"].dropna().iloc[0]
    single_recs = recommend_for_user(
        user_id=example_user_id,
        data=data,
        rules_idx=rules_idx,
        popularity=popularity,
        user_col="user_id",
        item_col="product_name",
        k=6,
        subset_k=2,
        beta=1.0,
    )
    print(f"\nRecommendations for user {example_user_id}:")
    print(single_recs.to_string(index=False))

    # 4) Recommendations for ALL users (no sampling)
    all_user_ids = data["user_id"].dropna().drop_duplicates()
    print(f"\n[INFO] Generating recommendations for all {len(all_user_ids)} users...")

    multi_recs = recommend_for_users(
        user_ids=all_user_ids,
        data=data,
        rules_idx=rules_idx,
        popularity=popularity,
        user_col="user_id",
        item_col="product_name",
        k=6,
        subset_k=2,
        beta=1.0,
    )



    # 5) Build user_id -> [product_id,...] dict (6 per user) and save it
    user_recs_dict = build_user_recs_dict(
        multi_recs=multi_recs,
        data=data,
        k=6,
    )

    out_path = DATA_DIR / "user_recommendations.json"
    with open(out_path, "w") as f:
        # Convert keys to str so JSON is clean
        json.dump({str(k): v for k, v in user_recs_dict.items()}, f)

    print(f"\n[INFO] Saved recommendations for {len(user_recs_dict)} users to: {out_path}")
