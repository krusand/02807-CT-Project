from collections import defaultdict
import pickle as pkl
from typing import Tuple

from lenskit.algorithms.als import BiasedMF
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from config import *
from sklearn.base import BaseEstimator, TransformerMixin

import psutil


def get_recs(pred_ratings: np.ndarray, 
             mf: BiasedMF, 
             n_recs=6,
             d_hondts=False) -> dict:
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
    if d_hondts:
        scores = pred_ratings.copy()
        n_users, _ = scores.shape

        # array to store indices for each user
        top_idx_sorted = np.zeros((n_users, n_recs), dtype=int)

        for k in range(n_recs):
            # choose item by computing argmax along each row (best item per user)
            chosen = np.argmax(scores, axis=1)
            top_idx_sorted[:, k] = chosen

            # halve the chosen score per user
            scores[np.arange(n_users), chosen] /= 2
    else:
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


def convert_aisle_recs(recs_dict: dict, aisle_top_products: dict) -> dict:
    """
    Convert aisle recommendations into product recommendations by recommending the top products from the corresponding aisle. 

    Parameters:
    - recs_dict:            Dictionary containing aisle recommendations per user.
    - aisle_top_products:   Dictionary containing the top products per aisle.
    
    Returns:
    - product_recs_dict:    Dictionary containing product recommendations per user.
    """
    product_recs_dict = {}

    for user, aisle_recs in recs_dict.items():
        # copying the values such that we can pop the products for each user
        top_dict = {k: v.copy() for k, v in aisle_top_products.items()}

        # retrieving product recommendations
        prod_recs = [top_dict[aisle].pop(0) for aisle in aisle_recs]

        # saving to product_recs_dict
        product_recs_dict[user] = prod_recs
    
    return product_recs_dict


def convert_user_cluster_recs(cluster_recs_dict: dict, cluster_user_dict: dict) -> dict:
    """
    Convert recommendations for each user cluster into recommendations for each user.

    Parameters:
    - cluster_recs_dict:    Recommendations for each user cluster.
    - cluster_user_dict:    Dictionary whose keys are cluster IDs and values are list of user IDs belonging to that cluster.

    Returns:
    - user_recs_dict:       Recommendations for each user. 
    """
    user_recs_dict = {}

    for cluster_id, recs in cluster_recs_dict.items():
        # identify users in cluster
        user_list = cluster_user_dict[cluster_id]

        for user in user_list:
            # recommend recs to all users in cluster
            user_recs_dict[user] = recs

    return user_recs_dict


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
        if dcg_star == 0:
            ndcg = 0
        else:
            ndcg = dcg / dcg_star

        # appending ndcg for the user to the eval_dict
        eval_dict[user][f"ndcg@{n_recs}"] = ndcg
        
    return eval_dict


def get_cf_scores(features: int, 
                  rating_df_train: pd.DataFrame,
                  rating_df_val: pd.DataFrame,
                  n_epochs=20,
                  train_patience=3,
                  reg=0.1,
                  damping=5,
                  bias=True,
                  seed=51225,
                  n_recs=6,
                  aisles=False,
                  aisle_dict=None,
                  user_clusters=False,
                  cluster_user_dict=None,
                  ) -> Tuple[dict, BiasedMF]:
    """
    Function to train a BiasedMF model from lenskit and generate predicted scores for each user and item. 
    Lenskit documentation for the BiasedMF model: https://lenskit.org/0.14.4/mf#lenskit.algorithms.svd.BiasedSVD

    Parameters:
    - features:         The number of latent features in the user and item vectors learned by the model.
    - rating_df_train:  The training data containing ratings for the items rated by each user.
    - rating_df_val:    The validation data containing ratings for the items rated by each user. 
    - n_epochs:         The maximum number of training iterations (default: 20).
    - train_patience:   The number of epochs to run before early stopping can be applied (default: 5).    
    - reg:              Regularization factors, can also be a tuple (ureg, ireg) to specify separate user and item regularization terms (default: 0.1).
    - damping:          Damping factor for the underlying bias (default: 5). 
    - bias:             Whether to include a bias term in the prediction rule or not (default: True). 
    - seed:             Seed for reproducibility purposes (default: 51225).
    - n_recs:           Number of recommendations to generate for each user (6 by default).
    - aisles:           Whether the items are grouped by their aisle IDs or not (default: False).
    - aisle_dict:       Dictionary containing to products for each aisle, only used if aisles=True.
    - user_clusters:    Whether the users are clustered or not (default: False). 

    Returns:
    - pred_ratings:     Dictionary containing the predicted score for each pair of user and item.
    - mf:               Fitted BiasedMF model.  
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
        print(f"Epoch: {epoch}", flush=True)
        logging.info(f"Epoch: {epoch}")
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
        if aisles and user_clusters:
            aisle_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs, d_hondts=True)
            cluster_recs_dict = convert_aisle_recs(recs_dict=aisle_recs_dict, aisle_top_products=aisle_dict)
            recs_dict = convert_user_cluster_recs(cluster_recs_dict=cluster_recs_dict, cluster_user_dict=cluster_user_dict)
        elif aisles:
            aisle_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs, d_hondts=True)
            recs_dict = convert_aisle_recs(recs_dict=aisle_recs_dict, aisle_top_products=aisle_dict)
        elif user_clusters:
            cluster_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)
            recs_dict = convert_user_cluster_recs(cluster_recs_dict=cluster_recs_dict, cluster_user_dict=cluster_user_dict)
        else: 
            recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)
        logging.info(f"Memory used: {process.memory_info().rss * 1e-9} GB")

        # evaluate on val set (obtain ndcg@n_recs)
        eval_dict = eval_recs(recs_dict=recs_dict, rating_df=rating_df_val, test_products=val_products)
        avg_hit_rate = sum(d["hit-rate"] for d in eval_dict.values()) / len(eval_dict)
        avg_ndcg = sum(d[f"ndcg@{n_recs}"] for d in eval_dict.values()) / len(eval_dict)
        print(f"Average hit-rate: {avg_hit_rate:.6f}", flush=True)
        print(f"Average ndcg@{n_recs}: {avg_ndcg:.6f}", flush=True)

        # trigger early stopping
        if avg_ndcg < prev_ndcg and epoch + 1 > train_patience:
            print(f"Average ndcg@{n_recs} ({avg_ndcg:.6f}) is lower than the previous ndcg@{n_recs} ({prev_ndcg:.6f}).")
            print(f"Early stopping is triggered after {epoch + 1} epochs")
            return prev_ratings, mf

        prev_ndcg = avg_ndcg
        prev_ratings = pred_ratings


    return prev_ratings, mf


def convert_to_user_term_matrix(ratings):
    """Converts from long to wide in CSR format. Input must have column names ['user', 'item', 'rating']"""
    users = ratings["user"].unique()
    item = ratings["item"].unique()
    shape = (len(users), len(item))

    # Create indices for users and movies
    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    product_cat = CategoricalDtype(categories=sorted(item), ordered=True)
    user_index = ratings["user"].astype(user_cat).cat.codes
    product_index = ratings["item"].astype(product_cat).cat.codes

    # Conversion via COO matrix
    coo = sparse.coo_matrix((ratings["rating"], (user_index, product_index)), shape=shape)
    ratings_matrix = coo.tocsr()
    return user_index, product_index, ratings_matrix

def cluster_and_predict_users(ratings_matrix, n_clusters=50, n_representative_features=50):

    svd = TruncatedSVD(n_components=n_representative_features)
    X_svd = svd.fit_transform(ratings_matrix)

    km = KMeans(n_clusters=n_clusters)
    y_pred = km.fit_predict(X_svd)

    return y_pred

def assign_cluster_to_users(y_pred, ratings_long):

    user_cluster_preds = (ratings_long[["user"]]
                    .drop_duplicates()
                    .assign(cluster=y_pred)
    )

    return user_cluster_preds

def save_cluster_user_dict(user_cluster_preds):

    cluster_user_df = (user_cluster_preds
                        .groupby(["cluster"])
                        ["user"]
                        .unique()
                        .reset_index()
                        .assign(user = lambda x: x["user"].tolist()))
    
    cluster_user_dict = {i: row["user"].tolist() for i,row in cluster_user_df.iterrows()}
    with open(DATA_PREPROCESSED_DIR / "cluster_user_dict.pkl", "wb") as fp:
        pkl.dump(cluster_user_dict, fp)

    

def save_cluster_ratings(ratings_long, user_cluster_preds):

    ratings_clusters = (ratings_long
        .merge(user_cluster_preds, on='user', how='left')
        .groupby(["cluster", "item"])
        .agg(rating = ('rating', 'mean'))
        .reset_index()
    )
    ratings_clusters.columns = ["user", "item", "rating"]
    ratings_clusters.to_parquet(DATA_PREPROCESSED_DIR / "train_cluster_ratings.pq")



class DOWEncoder(BaseEstimator, TransformerMixin):
    """Encodes an int denoting DOW (1-7), to polar coordinates, allowing cyclic features representation"""
    def __init__(self, return_pd = True) -> pd.DataFrame:
        self.return_pd = return_pd
        return

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def encode_day(day):
            rad = 2*np.pi*day / 7
            return np.round(np.cos(rad),3), np.round(np.sin(rad),3)
        X["X_day_dir"], X["Y_day_dir"] = zip(*X["order_dow"].apply(encode_day))
        X = X.drop("order_dow",axis=1)
        if self.return_pd:
            return X.copy()
        return X


class HourOfDayEncoder(BaseEstimator, TransformerMixin):
    """Encodes an int denoting hour (0-24), to polar coordinates, allowing cyclic features representation"""
    def __init__(self, return_pd = True):
        self.return_pd = return_pd
        return

    def fit(self, X, y=None):
        return self
    
    def transform(self, X) -> pd.DataFrame:
        def encode_hour(HourOfDay: int):
            rad = 2*np.pi*HourOfDay / 24
            return np.round(np.cos(rad),3), np.round(np.sin(rad),3)
        X["X_hour_dir"], X["Y_hour_dir"] = zip(*X["order_hour_of_day"].apply(encode_hour))
        X = X.drop("order_hour_of_day",axis=1)
        if self.return_pd:
            return X.copy()
        return X