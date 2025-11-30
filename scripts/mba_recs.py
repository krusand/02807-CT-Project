import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import *
from mba_utils import *
from utils import *


def main():
    # Global random seed for reproducibility (mainly for any numpy randomness)
    SEED = 42
    np.random.seed(SEED)

    # loading val ratings and products for evaluation 
    val_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "val_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    val_products = construct_test_product_dict(mode="val")

    # 1) Load & prepare data from data/
    data = load_data()
    all_user_ids = data["user_id"].dropna().drop_duplicates()
    
  
    logging.info("Training Apriori model")
    # 2) Train Apriori model on all users (once), using product_name
    _, _, rules_idx, popularity = train_apriori_model(
        data,
        user_col="user_id",
        order_col="order_id",
        item_col="product_name",
        verbose=False,
        support_grid=[0.01],
        min_confidence=0.1,
    )

    logging.info("Generating recommendations for all users")
    # 3) Recommendations for ALL users (no sampling)
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

    # 4) Build user_id -> [product_id,...] dict (6 per user) and save it
    user_recs_dict = build_user_recs_dict(
        multi_recs=multi_recs,
        data=data,
        k=6,
    )
    
    # 5) Evaluating the recommendations
    eval_dict = eval_recs(recs_dict=user_recs_dict, rating_df=val_ratings, test_products=val_products)
    avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
    avg_ndcg = np.mean([metric_dict[f"ndcg@6"] for metric_dict in eval_dict.values()])
    print(f"Average hit-rate: {avg_hr:.6f}", flush=True)
    print(f"Average ndcg@6: {avg_ndcg:.6f}", flush=True)

    # 6) Saving the evaluation results to a csv file
    row = ["mba", 6, avg_hr, avg_ndcg]
    with open(RESULTS_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


if __name__ == "__main__":
    main()