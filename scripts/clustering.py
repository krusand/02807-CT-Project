import sys
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import numpy as np

from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse

import pickle as pkl 


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import *

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



def main():
    logging.info("Started clustering users")
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
    user_index, _, ratings_matrix = convert_to_user_term_matrix(ratings_long)
    y_pred = cluster_and_predict_users(ratings_matrix=ratings_matrix)
    user_cluster_preds = assign_cluster_to_users(y_pred=y_pred, ratings_long=ratings_long)
    save_cluster_user_dict(user_cluster_preds=user_cluster_preds)
    save_cluster_ratings(ratings_long=ratings_long, user_cluster_preds=user_cluster_preds)
    logging.info("Finished clustering users")


if __name__ == '__main__':
    main()