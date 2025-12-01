import csv
import os
import pickle as pkl
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import *
from utils import *


def main():
    # loading ratings for each split
    train_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq") 
    train_ratings_aisles = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_aisle_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq") 
    val_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "val_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    test_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "test_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    
    # top aisle products as dict
    aisle_top_products_df = pd.read_parquet(AISLE_TOP_PRODUCTS_PATH)
    aisle_dict = (aisle_top_products_df
                  .groupby('aisle_id')['product_id']
                  .apply(list)
                  .to_dict()
                  )

    # number of recommendations to generate for each user
    n_recs = 6


    ### CF - ALL USERS ALL ITEMS ###
    logging.info("Starting CF fit - all users and all items")
    pred_ratings, mf = get_cf_scores(features=2000, 
                                     rating_df_train=train_ratings, 
                                     rating_df_val=val_ratings, 
                                     reg=0.001,
                                     bias=False)
    logging.info("Ended CF fit - all users and all items")

    logging.info("Starting recs")
    recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)
    logging.info("Ending recs")
    with open(DATA_PREPROCESSED_DIR / "cf_all_users_all_items_recs.pkl", 'wb') as fp:
        pkl.dump(recs_dict,file=fp)

    logging.info("Saved recs")

    logging.info("Evaluating performance of CF recommender (all items, all users)")
    test_products = construct_test_product_dict(mode="test")
    eval_dict = eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_products)
    
    # computing average hit-rate and ndcg
    avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
    avg_ndcg = np.mean([metric_dict[f"ndcg@{n_recs}"] for metric_dict in eval_dict.values()])

    logging.info(f"The average hit-rate on the test set across all users is {avg_hr:.6f}")
    logging.info(f"The average ndcg@{n_recs} on the test set across all users is {avg_ndcg:.6f}")

    row = ["cf_all_users_all_items", n_recs, avg_hr, avg_ndcg]
    with open(RESULTS_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    ### CF - ALL USERS AISLES ###
    logging.info("Starting CF fit - all users aisles")
    pred_ratings, mf = get_cf_scores(features=500, 
                                     rating_df_train=train_ratings_aisles, 
                                     rating_df_val=val_ratings, 
                                     reg=0.0001,
                                     damping=5,
                                     bias=True,
                                     aisles=True,
                                     aisle_dict=aisle_dict)
    logging.info("Ended CF fit - all users aisles")

    logging.info("Starting recs")
    aisle_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)
    recs_dict = convert_aisle_recs(recs_dict=aisle_recs_dict, aisle_top_products=aisle_dict)
    logging.info("Ending recs")
    with open(DATA_PREPROCESSED_DIR / "cf_all_users_aisles_recs.pkl", 'wb') as fp:
        pkl.dump(recs_dict,file=fp)

    logging.info("Saved recs")

    logging.info("Evaluating performance of CF recommender (all items, all users)")
    test_products = construct_test_product_dict(mode="test")
    eval_dict = eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_products)
    
    # computing average hit-rate and ndcg
    avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
    avg_ndcg = np.mean([metric_dict[f"ndcg@{n_recs}"] for metric_dict in eval_dict.values()])

    logging.info(f"The average hit-rate on the test set across all users is {avg_hr:.6f}")
    logging.info(f"The average ndcg@{n_recs} on the test set across all users is {avg_ndcg:.6f}")

    row = ["cf_all_users_aisles", n_recs, avg_hr, avg_ndcg]
    with open(RESULTS_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)
    

    ### CF - USER CLUSTERS ALL ITEMS ###


    ### CF - USER CLUSTERS AISLES ### 

if __name__ == "__main__":
    main()

