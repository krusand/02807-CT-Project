from collections import defaultdict
import pickle as pkl
from typing import Tuple
import warnings

from lenskit.algorithms.als import BiasedMF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import davies_bouldin_score

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



class KMeans_self_implemented(BaseEstimator, ClassifierMixin):
    """
        KMeans clustering using Lloyd's algoritm. 

        Parameters
        ----------
        n_clusters : {int} The number of clusters KMeans should find.

        init_method : {str} 
        The centroid initialisation method to use.
            'random' : picks {n_clusters} points randomly from X.
            'kmeans++' : picks {n_clusters} points sequentially
                         by weighted sampling according to
                         squared euclidean distance from data
                         point to cluster centroid
        
        max_iter : {int} The maximum amount of iterations to optimize.
        
        seed : {int} Control the seed to allow determinism of output

        Returns
        -------
        None
        """
    def __init__(self, n_clusters=3, init_method='kmeans++', max_iter=200, seed=51225):
        """
        Initialises KMeans clustering parameters

        Parameters
        ----------
        n_clusters : {int} The number of clusters KMeans should find.

        init_method : {str} 
        The centroid initialisation method to use.
            'random' : picks {n_clusters} points randomly from X.
            'kmeans++' : picks {n_clusters} points sequentially
                         by weighted sampling according to
                         squared euclidean distance from data
                         point to cluster centroid
        
        max_iter : {int} The maximum amount of iterations to optimize.

        seed : {int} Control the seed to allow determinism of output

        Returns
        -------
        None
        """
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self._n_features_in = None
        self.seed = seed

    def _init_clusters(self, size: int):
        """
        Assigns initial random clusters for all observations in data set

        Parameters
        ----------
        size : {int} Number of samples to generate. 
        
        Returns
        -------
        y : {array-like} of shape (size, )
        """
        np.random.seed(self.seed)
        return np.random.randint(low=0, high=self.n_clusters, size=size)
    
    def _init_centroids(self, X):
        """
        Initialises centroids using either 'random' or 'kmeans++'. 
        
        'random': Choses n_clusters random data points from X
        'kmeans++': Sequentially picks n_clusters weighted random data points. 
                    Reweighs after each iteration. Data points are weighted 
                    according to their squared euclidean distance to the 
                    closest centroid. A higher value means a higher probability
                    of being chosen as the next cluster centroid.
                    This allows for better initial clustering assignment, 
                    lowering the occurences of getting stuck in local minima.

        Parameters
        ----------
        X : {array-like} of shape (n_samples_X, n_features)
            An array where each row is a sample and each column is a feature.
        
        Returns
        -------
        init_vals : {array-like} of shape (n_clusters, n_features)
        """
        np.random.seed(self.seed)

        if self.init_method == 'random':
            print("Doing random initialisation of cluster centroids")
            init_vals = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        elif self.init_method == 'kmeans++':
            print("Doing kmeans++ initialisation of cluster centroids")
            # https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
            centroids = [X[np.random.choice(X.shape[0], size=1, replace=False)]]
            n_clusters_picked = 1

            while n_clusters_picked < self.n_clusters:
                distances_arr_fast = euclidean_distances(X, np.concatenate(centroids))**2
                minimum_dist_idx = np.argmin(distances_arr_fast, axis=1)

                X_shortest = distances_arr_fast[np.arange(minimum_dist_idx.shape[0]), minimum_dist_idx]
                sample_weight = (X_shortest) / np.sum(X_shortest)
                centroids.append(X[np.random.choice(X.shape[0], size=1, p=sample_weight)])
                n_clusters_picked += 1
            init_vals = np.concatenate(centroids, axis=0)
        return init_vals
    
    def _calculate_cluster_centroid(self, X,y):
        """
        Calculates the cluster centroid for each cluster in y

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features). The data points
        y : {array-like} of shape (n_samples, ). The previous cluster assignments.

        Returns
        -------
        centroids : {array-like} of shape (n_clusters, ). The new cluster centroids. 
        """
        centroids = []

        for i in np.unique(y):
            X_gp = X[np.where(y == i)[0]]
            centroids.append(X_gp.mean(axis=0))
        centroids = np.stack(centroids)
        return centroids
    
    def _predict_cluster(self, X, centroids):    
        """
        Predicts new cluster assignment by picking for each point, which cluster,
        the point has the smallest euclicean distance to.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features). The data points.
        centroids : {array-like} of shape (n_clusters, ). The centroids of each cluster.
        Returns
        -------
        y_new : {array_like} of shape (n_clusters, ). The new cluster assignments
        """  
        distances_arr_fast = euclidean_distances(X, centroids)**2
        y_new = np.argmin(distances_arr_fast, axis=1)
        
        return y_new
    

    def _plot_clusters(self, X, y, centroids):
        """
        Plots the clusters and their centroids. Can be used for debugging. 
        Only plots the 2 first dimensions of X
        
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features). If n_features
            is larger than two, only the two first features are used
        y : {array-like} of shape (n_samples, ). The cluster assignments
        centroids : {array-like} of shape (n_clusters, ). The cluster centroid coordinates.

        Returns
        -------
        None
        """
        cmap = plt.get_cmap("tab20")

        for i in np.unique(y):
            X_gp = X[np.where(y == i)[0]]    
            plt.scatter(X_gp[:,0], X_gp[:,1], c=cmap(i))
        
        for centroid in centroids:
            plt.scatter(centroid[0], centroid[1], c='black')
        plt.show()



    def fit(self, X):
        """
        Sklearn fits the KMeans model using the parameters specified in init. 
        Uses an iterative algorithm (Lloyd's algorithm).

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
        
        Returns
        -------
        None
        """
        assert self.n_clusters <= X.shape[0], f"Number of clusters {self.n_clusters} must be less than number of data points {X.shape[0]}"
        if (self.n_clusters == X.shape[0]): warnings.warn(f"The same number of clusters {self.n_clusters} as data points are used {X.shape[0]}")

        y_prev = self._init_clusters(X.shape[0])
        centroids = self._init_centroids(X)

        # assignment step
        y_new = self._predict_cluster(X, centroids)

        for i in range(self.max_iter): # Termination rule
            print(i, self.max_iter)
            if i == self.max_iter-1: warnings.warn(f"Reached max iterations={self.max_iter}, increase max_iter parameter")
            if (y_prev == y_new).all(): # Termination rule
                print("stationary clusters, breaking")
                break

            y_prev = y_new

            # Update step
            centroids = self._calculate_cluster_centroid(X, y_new)

            # Assignment step
            y_new = self._predict_cluster(X, centroids)
            # self._plot_clusters(X, y_new, centroids)

            print(100*np.sum(y_prev == y_new).sum() / y_prev.shape[0])

        self.centroids = centroids
        self._n_features_in = X.shape[1]
        self._is_fitted = True
        
    
    def predict(self, X):
        """
        Sklearn predict method which calls _predict_cluster to assign clusters
        to data points provided in X 

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
        
        Returns
        -------
        y : {array-like} of shape (n_samples, ). New predictions.
        """
        check_is_fitted(self)
        assert X.shape[1] == self._n_features_in, f"Number of features in {self._n_features_in} must be the same as number of prediction features {X.shape[1]}"
        y = self._predict_cluster(X, self.centroids)
        return y

    def fit_predict(self, X):
        """
        Sklearn fit_predict which calls fit and then predict

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
        
        Returns
        -------
        y : {array-like} of shape (n_samples, ). New predictions.
        """
        
        self.fit(X)
        y = self.predict(X)
        return y

    def __sklearn_is_fitted__(self):
        """
        Method to check whether class has been fitted using .fit()
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


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
    """Uses SVD to create a latent representation of the rating matrix, and clusters using these features"""
    svd = TruncatedSVD(n_components=n_representative_features)
    X_svd = svd.fit_transform(ratings_matrix)

    km = KMeans_self_implemented(n_clusters=n_clusters)
    y_pred = km.fit_predict(X_svd)
    dbs = davies_bouldin_score(X_svd, y_pred)
    print(f"Davies-bouldin-score: {dbs}")
    return y_pred



def assign_cluster_to_users(y_pred, ratings_long):

    user_cluster_preds = (ratings_long[["user"]]
                    .drop_duplicates()
                    .assign(cluster=y_pred)
    )

    return user_cluster_preds


def save_cluster_user_dict(user_cluster_preds, save=True):

    cluster_user_df = (user_cluster_preds
                        .groupby(["cluster"])
                        ["user"]
                        .unique()
                        .reset_index()
                        .assign(user = lambda x: x["user"].tolist()))
    
    cluster_user_dict = {i: row["user"].tolist() for i,row in cluster_user_df.iterrows()}

    if save:
        with open(DATA_PREPROCESSED_DIR / "cluster_user_dict.pkl", "wb") as fp:
            pkl.dump(cluster_user_dict, fp)
    else:
        return cluster_user_dict

    
def save_cluster_ratings(ratings_long, user_cluster_preds, save=True):

    ratings_clusters = (ratings_long
        .merge(user_cluster_preds, on='user', how='left')
        .groupby(["cluster", "item"])
        .agg(rating = ('rating', 'mean'))
        .reset_index()
    )
    ratings_clusters.columns = ["user", "item", "rating"]
    if save:
        ratings_clusters.to_parquet(DATA_PREPROCESSED_DIR / "train_cluster_ratings.pq")
    else: 
        return ratings_clusters



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