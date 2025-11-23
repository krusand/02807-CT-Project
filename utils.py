from collections import defaultdict
from typing import Iterable

from lenskit.algorithms.als import BiasedMF
import numpy as np
import pandas as pd

from config import *

import psutil



def get_recs_old(scores_dict: dict, n_recs=6, d_hondts=True) -> dict:
    """
    Generates a list of length n_recs of recommended items, aisles, or clusters for each user.

    Parameters:
    - scores_dict:  Predicted scores output from the cf algorithm.
    - n_recs:       Number of recommended items, aisles, or clusters (6 by default). 
    - d_hondts:     Whether to apply D'Hondts method or not. 

    Returns:
    - recs_dict:    Dictionary containing recommendations per user. 
    """
    recs_dict = defaultdict(list)

    if d_hondts:
        for user, ratings in scores_dict.items():
            for _ in range(n_recs):
                # retrieving index of recommended item (the item with the current highest rating)
                rec_item_idx = ratings.index(max(ratings))
                recs_dict[user].append(rec_item_idx)

                # apply D'Hondts method by halving rating of recommended item
                ratings[rec_item_idx] *= 0.5 

    else:
        users = scores_dict.keys()
        first_user = users[0]
        n_items = len(scores_dict[first_user])

        # checking if there are more unique items than recommendations to generate
        assert n_items >= n_recs, "The number of unique items must equal to n_recs or higher!"

        for user, ratings in scores_dict.items():
            # retrieving indices of recommended items (the n_recs items with the highest rating)
            rec_item_idxs = list(np.argsort(ratings)[::-1][:n_recs])
            recs_dict[user] = rec_item_idxs

    return recs_dict


def get_recs(pred_ratings: np.ndarray, 
             mf: BiasedMF, 
             n_recs=6) -> dict:
    """
    Function to generate the recommendations from the predicted ratings returned by BiasedMF.

    Parameters:
    - pred_ratings:     Array containing the predicted ratings (output from BiasedMF).
    - mf:               Instance of the BiasedMF model.
    - n_recs:           Number of recommendations to generate for each user (6 by default).

    Returns:
    - recs_dict:        Dictionary with users as the keys and lists of recommendations as the values. 
    """
    # obtaining sorted indices for recommendations
    top_idx = np.argpartition(pred_ratings, -n_recs, axis=1)[:, -n_recs:]
    rows = np.arange(pred_ratings.shape[0])[:, None]
    top_idx_sorted = top_idx[rows, np.argsort(-pred_ratings[rows, top_idx], axis=1)]

    # initializing users and items from mf instance
    users = mf.user_index_
    items = mf.item_index_

    # initializing dictionary to store recommendations
    recs_dict = defaultdict(list)

    # retrieving recommendations for each user
    for n, user in enumerate(users):
        recs_idx = top_idx_sorted[n, :].tolist()
        recs = [items[rec_idx] for rec_idx in recs_idx]
        recs_dict[user] = recs
    
    return recs_dict


def construct_test_product_dict(mode: str) -> dict:
    """
    Constructs a dictionary whose keys are user_ids and values are list of product_ids corresponding to the items bought by the validation or test users.

    Parameters:
    - mode:     Equal to "val" or "test" depending on which users to evaluate for. 

    Returns:
    - user_product_dict:    Dictionary containing the products bought in the last order for either the val or test users. 
    """
    orders_df = pd.read_parquet(ORDERS_PATH)   

    if mode == "val":
        orders_df = orders_df[orders_df["eval_set"] == "val"] 
        order_products_df = pd.read_parquet(ORDER_PRODUCTS__VAL_PATH)  
    
    elif mode == "test":
        orders_df = orders_df[orders_df["eval_set"] == "test"] 
        order_products_df = pd.read_parquet(ORDER_PRODUCTS__TEST_PATH)  

    else:
        raise ValueError("The mode parameter must be either 'val' or 'test'.")
    
    val_products_df = order_products_df.merge(orders_df, on="order_id", how="left")[["product_id", "user_id"]]
    user_product_dict = val_products_df.groupby("user_id")["product_id"].apply(list).to_dict()

    return user_product_dict


def eval_recs(recs_dict: dict, rating_df: pd.DataFrame, test_products: dict) -> dict:
    """
    Function to evaluate the generated recommendations in terms of hit-rate and ndcg. 

    Parameters:
    - recs_dict:        Dictionary containing recommendations, user_ids as keys and list of recommended product_ids as values.
    - rating_matrix:    Dataframe containing the ratings for each pair of user and item.
    - test_products:    Dictionary containing the products bought by each user in the val/test period, user_ids as keys and list of product_ids as values.

    Returns:
    - eval_dict:        Dictionary containing the evaluation results on user level, user_ids as keys and dictionaries containing evaluation metrics as values.
    """
    eval_dict = defaultdict(dict)

    # converting the rating_df to a dictionary
    rating_dict = (rating_df
                   .groupby("user")
                   .apply(lambda df: dict(zip(df["item"], df["rating"])))
                   .to_dict()
                   )

    for user, item_list in test_products.items():
        # retrieving recommendations from recs_dict
        user_recs = recs_dict[user]
        n_recs = len(user_recs)

        # converting lists to sets
        item_set = set(item_list)
        recs_set = set(user_recs)

        # finding the number of common items between the recommendations and the items bought
        common_items = len(item_set.intersection(recs_set))

        # appending the hit-rate for the user to the eval_dict
        if common_items >= 1:
            eval_dict[user]["hit-rate"] = 1
        else:
            eval_dict[user]["hit-rate"] = 0

        # capping the ratings for the user at n_recs
        user_ratings = rating_dict[user]
        capped_user_rating_dict = dict(list(user_ratings.items())[:n_recs])

        # computing optimal dcg
        dcg_star = 0
        for j, rating in enumerate(capped_user_rating_dict.values()):
            dcg_star += rating / np.log2(j + 2)   # +2 because position should start at 1
        
        # computing dcg based on recommendations
        dcg = 0
        for j, item in enumerate(user_recs):
            if item in user_ratings:
                rating = user_ratings[item]
                dcg += rating / np.log2(j + 2)
        
        # computing ndcg
        ndcg = dcg / dcg_star

        # appending ndcg for the user to the eval_dict
        eval_dict[user][f"ndcg@{n_recs}"] = ndcg
        
    return eval_dict


def get_cf_scores(features: int, 
                  rating_df_train: pd.DataFrame,
                  rating_df_val: pd.DataFrame,
                  n_epochs=20,
                  reg=0.1,
                  damping=5,
                  bias=True,
                  seed=51225,
                  n_recs=6) -> dict:
    """
    Function to train a BiasedMF model from lenskit and generate predicted scores for each user and item. 
    Lenskit documentation for the BiasedMF model: https://lenskit.org/0.14.4/mf#lenskit.algorithms.svd.BiasedSVD

    Parameters:
    - features:         The number of latent features in the user and item vectors learned by the model.
    - rating_df_train:  The training data containing ratings for the items rated by each user.
    - rating_df_val:    The validation data containing ratings for the items rated by each user. 
    - iterations:       The maximum number of training iterations (default: 20).
    - reg:              Regularization factors, can also be a tuple (ureg, ireg) to specify separate user and item regularization terms (default: 0.1).
    - damping:          Damping factor for the underlying bias (default: 5). 
    - bias:             Whether to include a bias term in the prediction rule or not (default: True). 
    - seed:             Seed for reproducibility purposes (default: 51225).
    - n_recs:           Number of recommendations to generate for each user (6 by default).

    Returns:
    - pred_ratings:      Dictionary containing the predicted score for each pair of user and item.
    """
    # initializing the BiasedMF model
    mf = BiasedMF(features=features, 
                  iterations=n_epochs, 
                  reg=reg, 
                  damping=damping, 
                  bias=bias,
                  rng_spec=seed)
    
    # epoch generator
    epoch_gen = mf.fit_iters(rating_df_train)

    # initializing the pred_ratings and ndcg of the previous epoch 
    pred_ratings = None
    prev_ndcg = -1   # will always be between 0 and 1 when training and evaluating on the val users

    # dictionary containing the items bought for the val users
    val_products = construct_test_product_dict(mode="val")

    # train for n_epochs or until early stopping is triggered 
    for epoch in range(n_epochs):
        logging.info(f"Epoch: {epoch+1}")
        process = psutil.Process()
        logging.info(f"Pre-Memory used: {process.memory_info().rss * 1e-9} GB")
        # run an epoch
        next(epoch_gen)
        logging.info(f"Memory used: {process.memory_info().rss * 1e-9} GB")

        logging.info("Epoch train finished")
        # user matrix of shape [n_users × k]
        U = mf.user_features_
        logging.info(f"Memory used: {process.memory_info().rss * 1e-9} GB")

        # item matrix of shape [n_items × k]
        V = mf.item_features_
        logging.info(f"Memory used: {process.memory_info().rss * 1e-9} GB")

        logging.info("Calculating ratings")
        if bias:
            # retrieving bias terms
            mu = mf.bias.mean_
            ub = mf.bias.user_offsets_.reindex(mf.user_index_).to_numpy()
            ib = mf.bias.item_offsets_.reindex(mf.item_index_).to_numpy()
            ub = ub.reshape(-1, 1)   # shape [n_users, 1]
            ib = ib.reshape(1, -1)   # shape [1, n_items]

            # matrix of predicted ratings: [n_users × n_items]
            pred_ratings = U @ V.T + mu + ub + ib
        
        else:
            pred_ratings = U @ V.T
        logging.info(f"Memory used: {process.memory_info().rss * 1e-9} GB")

        # extract recommendations from pred_ratings
        recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)
        logging.info(f"Memory used: {process.memory_info().rss * 1e-9} GB")

        # evaluate on val set (obtain ndcg@n_recs)
        eval_dict = eval_recs(recs_dict=recs_dict, rating_df=rating_df_val, test_products=val_products)
        avg_ndcg = sum(d[f"ndcg@{n_recs}"] for d in eval_dict.values()) / len(eval_dict)

        # trigger early stopping
        if avg_ndcg < prev_ndcg:
            print(f"Average ndcg@{n_recs} ({avg_ndcg:.4f}) is lower than the previous ndcg@{n_recs} ({prev_ndcg:.4f}).")
            print(f"Early stopping is triggered after {epoch + 1} epochs")
            return prev_ratings

        prev_ndcg = avg_ndcg
        prev_ratings = pred_ratings


    return prev_ratings


