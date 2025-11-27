import argparse
import csv
import gc
from itertools import product

from tqdm import tqdm

from config import *
from mba_utils import *
from utils import *


def main():
    # Global random seed for reproducibility (mainly for any numpy randomness)
    SEED = 42
    np.random.seed(SEED)

    # argument parser for input parameters
    print("Parsing the input arguments.")
    parser = argparse.ArgumentParser(description="Perform market basket analysis to generate recommendations for all users.")
    parser.add_argument("--support_grids", type=str, required=True, help="Comma-separated sequence of floats: x,y,z")
    parser.add_argument("--min_confidences", type=str, required=False, help="Comma-separated sequence of floats: x,y,z")
    args = parser.parse_args()

    # parsing the input arguments
    support_grids = [[float(grid_val)] for grid_val in args.support_grids.split(",")]
    min_confidences = [float(x) for x in args.min_confidences.split(",")]

    # print out hyperparam values to test
    print(f"Values for support_grid to test: {support_grids}", flush=True)
    print(f"Values for min_confidence to test: {min_confidences}", flush=True)

    # loading val ratings and products for evaluation 
    val_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "val_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    val_products = construct_test_product_dict(mode="val")

    # 1) Load & prepare data from data/
    data = load_data()
    all_user_ids = data["user_id"].dropna().drop_duplicates()
    
    for support_grid, min_confidence in tqdm(list(product(support_grids, min_confidences))):
        print(f"\nTesting combination: support_grid={support_grid}, min_confidence={min_confidence}", flush=True)

        logging.info("Training Apriori model")
        # 2) Train Apriori model on all users (once), using product_name
        _, _, rules_idx, popularity = train_apriori_model(
            data,
            user_col="user_id",
            order_col="order_id",
            item_col="product_name",
            verbose=False,
            support_grid=support_grid,
            min_confidence=min_confidence,
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
        row = [support_grid[0], min_confidence, avg_hr, avg_ndcg]
        with open(MBA_EXP_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        # freeing up some memory
        del rules_idx, popularity, multi_recs, user_recs_dict, eval_dict
        gc.collect()


if __name__ == "__main__":
    main()