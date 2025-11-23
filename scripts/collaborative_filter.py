import os
import pickle as pkl
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import *
from utils import *


def main():
    # loading ratings for each split
    train_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq") 
    val_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "val_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    test_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "test_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    
    # number of recommendations to generate for each user
    n_recs = 6

    logging.info("Starting CF fit")

    scores_dict = get_cf_scores(features=1000, ratings_df_train=train_ratings, rating_df_val=val_ratings)
    logging.info("Ended CF fit")

    logging.info("Starting recs")
    recs_dict = get_recs(scores_dict=scores_dict, n_recs=n_recs)
    logging.info("Ending recs")
    with open(DATA_PREPROCESSED_DIR / "recs.pkl", 'wb') as fp:
        pkl.dump(recs_dict,file=fp)

    logging.info("Saved recs")

    logging.info("Evaluating performance of CF recommender (all items, all users)")
    test_products = construct_test_product_dict(mode="test")
    eval_dict = eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_products)

    # computing average hit-rate and ndcg
    avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
    avg_ndcg = np.mean([metric_dict[f"ndcg@{n_recs}"] for metric_dict in eval_dict.values()])

    logging.info(f"The average hit-rate across all users is {avg_hr:.4f}")
    logging.info(f"The average ndcg@{n_recs} across all users is {avg_ndcg:.4f}")

if __name__ == "__main__":
    main()

