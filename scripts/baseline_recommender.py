import os
import sys 

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import *
import utils as ut


def main():
    top_n = 6
    logging.info(f"Retrieving top {top_n} recommendations for all users")

    # load train data
    df = pd.read_parquet(ORDER_PRODUCTS__TRAIN_PATH)

    # grouping by product_id and counting the occurrence of each product across all orders
    grouped_df = df.groupby(["product_id"])["product_id"].size().reset_index(name="product_id_count")

    # sorting by count in descending order
    sorted_df = grouped_df.sort_values("product_id_count", ascending=False)

    # retrieving the top_n most bough products
    top_n_products = sorted_df.iloc[:top_n, :]["product_id"]

    # loading unique users
    unique_users = pd.read_parquet(UNIQUE_USERS_PATH)

    # dictionary containing top_n recs for each user
    recs_dict = {user: top_n_products.to_list() for user in unique_users["user"]}

    logging.info(f"Evaluating the top {top_n} recommendations")

    # evaluate top_n recs
    test_ratings = pd.read_parquet("data/2_preprocessed/test_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    test_product_dict = ut.construct_test_product_dict(mode="test")
    eval_dict = ut.eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_product_dict)

    # computing average hit-rate and ndcg
    avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
    avg_ndcg = np.mean([metric_dict[f"ndcg@{top_n}"] for metric_dict in eval_dict.values()])

    logging.info(f"The average hit-rate across all users is {avg_hr:.4f}")
    logging.info(f"The average ndcg@{top_n} across all users is {avg_ndcg:.4f}")

if __name__ == "__main__":
    main()