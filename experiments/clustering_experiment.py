
import sys 
import os
import pickle as pkl

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score

import pandas as pd
import numpy as np

from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse

from tqdm import tqdm

from collections import defaultdict

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



ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

user_index, _, ratings_matrix = convert_to_user_term_matrix(ratings_long)

component_experiments = dict()

for n_components in range(50, 2500, 25):
    print(f"{n_components = }", flush=True)
    component_experiments[n_components] = {'db_scores': [],
                                           'cluster_sizes': []}
    
    svd = TruncatedSVD(n_components=n_components)
    X_svd = svd.fit_transform(ratings_matrix)

    for n_clusters in range(2,2500, 25):
        print(f"{n_clusters = }", flush=True)
        for i in tqdm(range(15)):
            km = KMeans(n_clusters = n_clusters)
            y_pred_svd = km.fit_predict(X_svd)
            component_experiments[n_components]["db_scores"].append(davies_bouldin_score(X_svd, y_pred_svd))
            component_experiments[n_components]["cluster_sizes"].append(np.unique(y_pred_svd, return_counts=True)[1].tolist())


with open(CLUSTER_EXP_PATH, "wb") as fp:
    pkl.dump(component_experiments, fp)