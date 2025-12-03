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

# seed for reproducibility
np.random.seed(51225)

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
    print(f"Values for n_clusters to test: {n_clusters_values}", flush=True)

    # loading ratings and products for validation set
    val_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "val_aisle_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    val_products = construct_test_product_dict(mode="val")
    
    # loading data for computing cluster and aisle ratings
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    _, _, ratings_matrix = convert_to_user_term_matrix(ratings_long)
    products_df = pd.read_csv(PRODUCTS_PATH_CSV)

    # top aisle products as dict
    aisle_top_products_df = pd.read_parquet(AISLE_TOP_PRODUCTS_PATH)
    aisle_dict = (aisle_top_products_df
                  .groupby('aisle_id')['product_id']
                  .apply(list)
                  .to_dict()
                  )

    # number of recommendations to generate for each user
    n_recs = 6

    logging.info("Starting CF fit")
    for n_features, reg, damping, n_clusters in tqdm(list(product(n_features_values, reg_values, damping_values, n_clusters_values))):
        logging.info("Started clustering users")
        y_pred = cluster_and_predict_users(ratings_matrix=ratings_matrix, n_clusters=n_clusters)
        user_cluster_preds = assign_cluster_to_users(y_pred=y_pred, ratings_long=ratings_long)
        cluster_user_dict = save_cluster_user_dict(user_cluster_preds=user_cluster_preds, save=False)
        train_cluster_ratings = save_cluster_ratings(ratings_long=ratings_long, user_cluster_preds=user_cluster_preds, save=False)
        logging.info("Finished clustering users")

        logging.info("Converting product ratings into aisle ratings")
        joined_df = train_cluster_ratings.merge(products_df, how="left", left_on="item", right_on="product_id")
        grp_df = joined_df.groupby(["user", "aisle_id"])["rating"].mean().reset_index()
        train_ratings = grp_df.rename(columns={"aisle_id": "item"})

        if bias_value:
            print(f"\nTesting combination: features={n_features}, reg={reg}, damping={damping}, n_clusters={n_clusters}", flush=True)
            pred_ratings, mf = get_cf_scores(features=n_features, 
                                             rating_df_train=train_ratings, 
                                             rating_df_val=val_ratings, 
                                             reg=reg,
                                             damping=damping,
                                             bias=bias_value,
                                             aisles=True,
                                             aisle_dict=aisle_dict,
                                             user_clusters=True,
                                             cluster_user_dict=cluster_user_dict)

        else:
            print(f"\nTesting combination: features={n_features}, reg={reg}, n_clusters={n_clusters}", flush=True)
            pred_ratings, mf = get_cf_scores(features=n_features, 
                                             rating_df_train=train_ratings, 
                                             rating_df_val=val_ratings, 
                                             reg=reg,
                                             bias=bias_value,
                                             aisles=True,
                                             aisle_dict=aisle_dict,
                                             user_clusters=True,
                                             cluster_user_dict=cluster_user_dict)

        logging.info("Ended CF fit")

        logging.info("Starting recs")
        aisle_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs, d_hondts=True)
        cluster_recs_dict = convert_aisle_recs(recs_dict=aisle_recs_dict, aisle_top_products=aisle_dict)
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
            row = [n_clusters, n_features, damping, reg, bias_value, avg_hr, avg_ndcg]
        else:
            row = [n_clusters, n_features, "-", reg, bias_value, avg_hr, avg_ndcg]
        with open(CF_USER_CLUSTERS_AISLES_EXP_PATH, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        # freeing up some memory
        del pred_ratings, mf, recs_dict, eval_dict
        gc.collect()

if __name__ == "__main__":
    main()

