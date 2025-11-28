import argparse
import csv
import gc
from itertools import product
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import *
from utils import *


def main():
    # argument parser for input parameters
    print("Parsing the input arguments.")
    parser = argparse.ArgumentParser(description="Run collaborative filtering experiments with hyperparameter inputs.")
    parser.add_argument("--n_features", type=str, required=True, help="Comma-separated sequence of n_components values: x,y,z")
    parser.add_argument("--reg_vals", type=str, required=True, help="Comma-separated sequence of regularization values: x,y,z")
    parser.add_argument("--damping_vals", type=str, required=False, help="Comma-separated sequence of damping values: x,y,z")
    parser.add_argument("--bias", type=int, required=True, help="0 for no bias terms, 1 for including bias terms")
    parser.add_argument("--n_clusters", type=str, required=False, help="Comma-separated sequence of integers: x,y,z")
    args = parser.parse_args()

    # parsing the input arguments
    n_features_values = [int(x) for x in args.n_features.split(",")]
    reg_values = [float(x) for x in args.reg_vals.split(",")]
    bias_value = bool(args.bias)
    n_clusters_values = [int(x) for x in args.n_clusters.split(",")]
    if bias_value:
        damping_values = [float(x) for x in args.damping_vals.split(",")]
    else:
        damping_values = [None] 
    
    # print out hyperparam values to test
    print(f"Values for n_features to test: {n_features_values}", flush=True)
    print(f"Values for reg to test: {reg_values}", flush=True)
    print(f"Bias: {bias_value}", flush=True)
    print(f"Values for damping to test: {damping_values}", flush=True)

    # loading ratings and products for validation set
    val_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "val_aisle_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    val_products = construct_test_product_dict(mode="val")
    
    # preparing ratings for cluster ratings
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    user_index, _, ratings_matrix = convert_to_user_term_matrix(ratings_long)

    # number of recommendations to generate for each user
    n_recs = 6

    logging.info("Starting CF fit")
    for n_features, reg, damping, n_clusters in tqdm(list(product(n_features_values, reg_values, damping_values, n_clusters_values))):
        logging.info("Started clustering users")
        y_pred = cluster_and_predict_users(ratings_matrix=ratings_matrix, n_clusters=n_clusters)
        user_cluster_preds = assign_cluster_to_users(y_pred=y_pred, ratings_long=ratings_long)
        save_cluster_user_dict(user_cluster_preds=user_cluster_preds)
        save_cluster_ratings(ratings_long=ratings_long, user_cluster_preds=user_cluster_preds)
        train_ratings = pd.read_parquet(TRAIN_CLUSTER_RATINGS_PATH) 
        cluster_user_dict = pd.read_pickle(CLUSTER_USER_DICT_PATH)
        logging.info("Finished clustering users")

        if bias_value:
            print(f"\nTesting combination: features={n_features}, reg={reg}, damping={damping}, n_clusters={n_clusters}", flush=True)
            pred_ratings, mf = get_cf_scores(features=n_features, 
                                             rating_df_train=train_ratings, 
                                             rating_df_val=val_ratings, 
                                             reg=reg,
                                             damping=damping,
                                             bias=bias_value,
                                             user_clusters=True,
                                             cluster_user_dict=cluster_user_dict)

        else:
            print(f"\nTesting combination: features={n_features}, reg={reg}, n_clusters={n_clusters}", flush=True)
            pred_ratings, mf = get_cf_scores(features=n_features, 
                                             rating_df_train=train_ratings, 
                                             rating_df_val=val_ratings, 
                                             reg=reg,
                                             bias=bias_value,
                                             user_clusters=True,
                                             cluster_user_dict=cluster_user_dict)

        logging.info("Ended CF fit")

        logging.info("Starting recs")
        cluster_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)
        recs_dict = convert_user_cluster_recs(cluster_recs_dict=cluster_recs_dict, cluster_user_dict=cluster_user_dict)
                                             
        logging.info("Ending recs")

        logging.info("Evaluating performance on validation set")
        eval_dict = eval_recs(recs_dict=recs_dict, rating_df=val_ratings, test_products=val_products)
        
        # computing average hit-rate and ndcg
        avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
        avg_ndcg = np.mean([metric_dict[f"ndcg@{n_recs}"] for metric_dict in eval_dict.values()])

        logging.info(f"The average hit-rate on the validation set across all users is {avg_hr:.6f}")
        logging.info(f"The average ndcg@{n_recs} on the validation set across all users is {avg_ndcg:.6f}")

        # writing row to csv
        if bias_value:
            row = [n_features, damping, reg, bias_value, avg_hr, avg_ndcg]
        else:
            row = [n_features, "-", reg, bias_value, avg_hr, avg_ndcg]
        with open(CF_USER_CLUSTERS_ALL_ITEMS_EXP_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        # freeing up some memory
        del pred_ratings, mf, recs_dict, eval_dict
        gc.collect()

if __name__ == "__main__":
    main()

