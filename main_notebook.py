#!/usr/bin/env python
# coding: utf-8

# # Instacart Grocery Recommendations 
# This notebook contains the code for our exam project in 02807 Computational Tools for Data Science.  
# 
# The goal of our project is to implement and evaluate multiple recommenders for groceries. We are working with the [Instacart Online Grocery Basket Analysis Dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/data?select=order_products__prior.csv), which includes approximately 50,000 products, 200,000 users, and 3.4 million orders. 
# 
# We use the following algorithms for our recommenders:
# - KMeans, clustering
# - Apriori, market basket analysis
# - Collaborative Filtering (CF), recommender system
# 
# Our project introduces the following recommenders:
# - Baseline: Top-n recommender
# - Apriori Recommender
# - CF on full user-item matrix (CF_Full)
# - CF on user-aisle matrix (CF_A)
# - CF with user clusters and all items (CF_C)
# - CF with user clusters and aisles (CF_AC)
# 
# Contributors:
# - Andreas Kruse Svenningsen (s253844)
# - Frederik Winther Bæk (s214618)
# - Georgios Loulakis (s252920)
# - Sebastian Nygaard Wærling (s254120)

# # Environment setup
# 
# To run this notebook, a python environment must be setup first. We use python 3.10 for this project.
# 
# We provide a short concise guide for setting up the environtment. 
# To setup the environment we will use `uv`, a fast python package manager and python version manager written in rust. Information about `uv` can be found here on their [Github](https://github.com/astral-sh/uv) or their [Docs](https://docs.astral.sh/uv/).
# 
# Start by cloning our repository [https://github.com/krusand/02807-CT-Project](https://github.com/krusand/02807-CT-Project). 
# 
# Using HTTPS: `git clone https://github.com/krusand/02807-CT-Project.git`
# 
# Change into the clone directory: `cd 02807-CT-Project`
# 
# 
# 

# 
# ## MacOS / Linux
# 
# On MacOS / Linux, `uv` can be installed using either
# 
# brew (Recommended for MacOS): `brew install uv` \
# curl: `curl -LsSf https://astral.sh/uv/install.sh | sh` \
# wget: `wget -qO- https://astral.sh/uv/install.sh | sh`
# 
# If prompted, run `source $HOME/.local/bin/env`
# 
# Afterwards, install the specific python version.\
# In this project we use python 3.10
# 
# Install using
# `uv python install 3.10`
# 
# Sync dependencies to our environment using
# `uv sync`
# 
# **The environment is now setup**
# 
# To run scripts from the command line
# `uv run {script_name}.py`
# 
# In jupyter notebooks, use the environment in
# `.venv/bin/python`
# 
# Activate the environment using CLI
# `source .venv/bin/activate`
# 

# 
# ## Windows
# 
# On Windows, `uv` can be installed using
# 
# `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
# 
# Afterwards, we install the specific python version.\
# In this project we use python 3.10
# 
# Install using
# `uv python install 3.10`
# 
# Sync dependencies to our environment using
# `uv sync`
# 
# **The environment is now setup**
# 
# To run scripts from the command line
# `uv run {script_name}.py`
# 
# In jupyter notebooks, use the environment in
# `.venv/bin/python`
# 
# Activate the environment using CLI
# `.venv/Scripts/activate`
# 

# # Code

# In[ ]:


# imports
from collections import Counter, defaultdict
import csv
from itertools import combinations
import os
from pathlib import Path
import pickle as pkl
from typing import Iterable, List, Optional, Tuple, Dict, Any, Hashable
import warnings

import kagglehub
from lenskit.algorithms.als import BiasedMF
import logging
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse
import shutil
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm


# The following code blocks follow the execution order of scripts specified in the `pipeline.py` script. Some of the recommenders require a substantial amount of memory (RAM) and have been executed using the DTU HPC. We have decided to exclude those part from this notebook. It will be stated when code has been excluded.  

# ## Config
# The following code block defines configurational variables, mostly paths to store data. 

# In[ ]:


PROJ_ROOT = Path(".").resolve()

DATA_DIR = PROJ_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "0_raw"
DATA_CLEANED_DIR = DATA_DIR / "1_cleaned"
DATA_PREPROCESSED_DIR = DATA_DIR / "2_preprocessed"
OUTPUTS_PATH = PROJ_ROOT / "outputs"
RESULTS_DIR = PROJ_ROOT / "results"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(DATA_RAW_DIR):
    os.makedirs(DATA_RAW_DIR)

if not os.path.exists(DATA_CLEANED_DIR):
    os.makedirs(DATA_CLEANED_DIR)

if not os.path.exists(DATA_PREPROCESSED_DIR):
    os.makedirs(DATA_PREPROCESSED_DIR)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

AISLES_PATH = DATA_CLEANED_DIR / "aisles.pq"
DEPARTMENTS_PATH = DATA_CLEANED_DIR / "departments.pq"
ORDER_PRODUCTS__TRAIN_PATH = DATA_CLEANED_DIR / "order_products__train.pq"
ORDER_PRODUCTS__VAL_PATH = DATA_CLEANED_DIR / "order_products__val.pq"
ORDER_PRODUCTS__TEST_PATH = DATA_CLEANED_DIR / "order_products__test.pq"
ORDERS_PATH = DATA_CLEANED_DIR / "orders.pq"
PRODUCTS_PATH = DATA_CLEANED_DIR / "products.pq"

AISLES_PATH_CSV = DATA_RAW_DIR / "aisles.csv"
DEPARTMENTS_PATH_CSV = DATA_RAW_DIR / "departments.csv"
ORDER_PRODUCTS__PRIOR_PATH_CSV = DATA_RAW_DIR / "order_products__prior.csv"
ORDER_PRODUCTS__TRAIN_PATH_CSV = DATA_RAW_DIR / "order_products__train.csv"
ORDERS_PATH_CSV = DATA_RAW_DIR / "orders.csv"
PRODUCTS_PATH_CSV = DATA_RAW_DIR / "products.csv"

UNIQUE_USERS_PATH = DATA_PREPROCESSED_DIR / "unique_users.pq"
AISLE_TOP_PRODUCTS_PATH = DATA_PREPROCESSED_DIR / "aisle_top_products.pq"
CLUSTER_USER_DICT_PATH = DATA_PREPROCESSED_DIR / "cluster_user_dict.pkl"
TRAIN_CLUSTER_RATINGS_PATH = DATA_PREPROCESSED_DIR / "train_cluster_ratings.pq"

CF_ALL_USERS_ALL_ITEMS_EXP_PATH = OUTPUTS_PATH / "cf_all_users_all_items_experiment_results.csv"
CF_ALL_USERS_AISLES_EXP_PATH = OUTPUTS_PATH / "cf_all_users_aisles_experiment_results.csv"
CF_USER_CLUSTERS_ALL_ITEMS_EXP_PATH = OUTPUTS_PATH / "cf_user_clusters_all_items_experiment_results.csv"
CF_USER_CLUSTERS_AISLES_EXP_PATH = OUTPUTS_PATH / "cf_user_clusters_aisles_experiment_results.csv"
MBA_EXP_PATH = OUTPUTS_PATH / "mba_experiment_results.csv"
CLUSTER_EXP_PATH = OUTPUTS_PATH / "cluster_experiment_results.pkl"

RESULTS_PATH = PROJ_ROOT / "results/results.csv"


# ## 1. Download Dataset
# The following functions are used to download the Instacart dataset from Kaggle through kagglehub. 

# In[ ]:


# UTILS

def download_dataset() -> str:
    """
    Downloads Instacart dataset from Kaggle through kagglehub, 
    and returns path to downloaded dataset

    Parameters
    ----------

    Returns
    -------
    path (str): Path to downloaded dataset
    
    """
    path = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")

    print("Path to dataset files:", path)
    return path

def move_dataset_from_cache_to_folder(path_to_cache: str, path_to_folder: str) -> None:
    """
    Moves downloaded dataset from {path_to_cache} to {path_to_folder}. 

    Parameters
    ----------

    path_to_cache (str): Path to downloaded dataset

    path_to_folder (int, float): Path to dataset destination

    
    Returns
    -------
    """
    shutil.copytree(path_to_cache, path_to_folder, dirs_exist_ok=True)
    shutil.rmtree(path_to_folder / "data", ignore_errors=True)

def convert_to_parquet() -> None:
    """Converts csv files into parquet files and saves them to DATA_CLEANED_DIR"""

    for file in tqdm(os.listdir(DATA_RAW_DIR)):
        file_name, file_extension = file.split(".")
        file_extension = "."+(file_extension)
        pd.read_csv(DATA_RAW_DIR / (file_name + file_extension)).to_parquet(DATA_CLEANED_DIR / (file_name + ".pq"))


# 
# Now we can use the functions to download the dataset:

# In[ ]:


path_to_cache = download_dataset()
move_dataset_from_cache_to_folder(path_to_cache=path_to_cache, path_to_folder=DATA_RAW_DIR)
convert_to_parquet()


# w## 2. Data Split
# The following code block saves the raw data to parquet files and divides the orders into a train, validation, and test set. We define the test set to be the last order of a user. The validation set is the second last, and the train set is every other order by the user. 
# The three sets are saved as parquet files named `order_products__{train|val|test}`.  

# In[ ]:


# load data
orders_df = pd.read_csv(ORDERS_PATH_CSV)
op_prior = pd.read_csv(ORDER_PRODUCTS__PRIOR_PATH_CSV)
op_train = pd.read_csv(ORDER_PRODUCTS__TRAIN_PATH_CSV)

# remove test orders
orders_df = orders_df[orders_df["eval_set"] != "test"]

# sorting to ensure correct ordering
orders_df = orders_df.sort_values(["user_id", "order_number"])

# helper column counting number of orders per user
orders_df["n_orders"] = orders_df.groupby("user_id")["order_number"].transform("max")

# assign split labels (order 1,...,n-2: train, order n-1: val, order n: test)
orders_df["eval_set_new"] = "train"
orders_df.loc[orders_df["order_number"] == orders_df["n_orders"], "eval_set_new"] = "test"
orders_df.loc[orders_df["order_number"] == orders_df["n_orders"] - 1, "eval_set_new"] = "val"

# drop n_orders and make eval_set_new the new eval_set column
orders_df["eval_set"] = orders_df["eval_set_new"]
orders_df = orders_df.drop(columns=["eval_set_new", "n_orders"])

# save orders_df to parquet
orders_df.to_parquet(ORDERS_PATH)

# concatenate order_products data
op_combined = pd.concat([op_prior, op_train])

# order_ids in each split
train_orders = orders_df[orders_df["eval_set"]=="train"]["order_id"]
val_orders = orders_df[orders_df["eval_set"]=="val"]["order_id"]
test_orders = orders_df[orders_df["eval_set"]=="test"]["order_id"]

# order products for each split
op_train_new = op_combined[op_combined["order_id"].isin(train_orders)]
op_val_new = op_combined[op_combined["order_id"].isin(val_orders)]
op_test_new = op_combined[op_combined["order_id"].isin(test_orders)]

# saving to parquet
op_train_new.to_parquet(ORDER_PRODUCTS__TRAIN_PATH)
op_val_new.to_parquet(ORDER_PRODUCTS__VAL_PATH)
op_test_new.to_parquet(ORDER_PRODUCTS__TEST_PATH)


# ## 3. Calculate Rating

# The following code block contains function utils used to calculate ratings. 
# 
# Ratings are used later in both collaborative filtering, and clustering of the users. 
# 
# We combine three ratings: *product_frequency*, *product_recency*, *TF-IDF*. Each rating is weighted by 1/3 in the overall rating matrix. Each individual rating is scaled to be within 0 and 1. Since product frequency is already scaled between 0 and 1, we will not scale this rating.

# In[ ]:


# UTILS
def calculate_user_product_frequency(merged_df: pd.DataFrame) -> None: 
    """
    Calculates product frequency pr. user. The frequency is calculated as 
    the number of unique times a product was bought, divided by the total
    number of orders. 

    Parameters
    ----------

    merged_df (pd.DataFrame): A DataFrame with the merge between orders
                              and order_products, merged on 'order_id'

    Returns
    -------

    None
    """
    logging.info("")

    # Number of unique times a product was bought, pr. user
    bui = (
        merged_df.groupby(['user_id', 'product_id'])['order_id']
        .nunique()
        .reset_index(name='Bui')
    )

    # Number of unique orders
    bu = (
        merged_df.groupby(['user_id'])['order_id']
        .nunique()
        .reset_index(name='Bu')
    )

    freq_df = pd.merge(bui, bu, on='user_id', how='left')
    freq_df['freq_ui'] = freq_df['Bui'] / freq_df['Bu']
    file_path = DATA_PREPROCESSED_DIR / "user_product_frequency.pq"
    freq_df.to_parquet(file_path, index=False)
    logging.info(f"Saved user_product_frequency to {file_path}")



def calculate_user_product_recency(merged_df: pd.DataFrame, lam: float = 0.0015) -> None:
    """
    Calculates product recency pr. user. 
   
    Parameters
    ----------

    merged_df (pd.DataFrame): A DataFrame with the merge between orders
                              and order_products, merged on 'order_id'
    lam (float, int): exponential paramater, a higher value means more focus on recent items

    Returns
    -------

    None
    """
    logging.info("")
    # replace NaN for first orders
    merged_df['days_since_prior_order'] = merged_df['days_since_prior_order'].fillna(0)

    # sort to compute cumulative time for each user
    merged_df = merged_df.sort_values(['user_id', 'order_number'])

    # cumulative days since first order
    merged_df['cum_days'] = merged_df.groupby('user_id')['days_since_prior_order'].cumsum()

    # total days each user is active
    merged_df['total_days'] = merged_df.groupby('user_id')['days_since_prior_order'].transform('sum')

    # age_days = time until last order
    merged_df['age_days'] = merged_df['total_days'] - merged_df['cum_days']

    # exponential weight based on recency
    merged_df['weight'] = np.exp(-lam * merged_df['age_days'])

    
    freq = (
        merged_df.groupby(['user_id', 'product_id'])['weight']
        .sum()
        .reset_index(name='score')
    )

    file_path = DATA_PREPROCESSED_DIR / "user_product_recency.pq"

    freq.to_parquet(file_path, index=False)
    logging.info(f"Saved user_product_recency to {file_path}")

    freq['recency_score_min_maxed'] = (
        (freq['score'] - freq['score'].min()) /
        (freq['score'].max() - freq['score'].min())
    )

    file_path = DATA_PREPROCESSED_DIR / "user_product_recency_min_max_scaled.pq"

    freq.to_parquet(file_path, index=False)
    logging.info(f"Saved user_product_recency min_max_scaled to {file_path}")



def calculate_tf_idf(merged_df: pd.DataFrame, orders: pd.DataFrame) -> None:
    """
    Calculates TF.IDF pr. user. In this context, each 'term' is an 'item',
    and each 'document' is an 'order'. TF.IDF gives an idea of when a product 
    is important for one user, taking into account how popular it is among all users. 
    This effectively finds each users 'specialty' purchase. The TF term measures the 
    frequency of each product across all products bought. 
    The IDF term measures the inverse popularity across all users. 
   
    Parameters
    ----------

    merged_df (pd.DataFrame): A DataFrame with the merge between orders
                              and order_products, merged on 'order_id'
    orders (pd.DataFrame): A DataFrame containing orders (ORDERS_PATH)

    Returns
    -------

    None
    """
    logging.info("")
    tf = (merged_df
          .groupby(['user_id', 'product_id'])
          .size()
          .reset_index(name='purchase_count')
        )
    
    total_products_per_user = (merged_df
                               .groupby('user_id')
                               .size()
                               .reset_index(name='total_products')
                               )

    tf = pd.merge(tf, total_products_per_user, on='user_id', how='left')
    
    tf['tf'] = tf['purchase_count'] / tf['total_products']

    # IDF = log(total users / users who bought this product)
    total_users = orders['user_id'].nunique()
    users_per_product = merged_df.groupby('product_id')['user_id'].nunique().reset_index(name='users_who_bought')

    # Avoid division by zero
    users_per_product['users_who_bought'] = users_per_product['users_who_bought'] + 1e-5
    users_per_product['idf'] = np.log(total_users / users_per_product['users_who_bought'])

    tf_idf = pd.merge(
        tf[['user_id', 'product_id', 'tf']],
        users_per_product[['product_id', 'idf']],
        on='product_id',
        how='left'
    )

    tf_idf['tfidf_score'] = tf_idf['tf'] * tf_idf['idf']


    final_tfidf_scores = tf_idf[['user_id', 'product_id', 'tfidf_score']]
    
    file_path = DATA_PREPROCESSED_DIR / "user_product_tfidf.pq"

    final_tfidf_scores.to_parquet(file_path, index=False)
    logging.info(f"Saved user_product_tfidf to {file_path}")

    # min-max scaling
    final_tfidf_scores['tfidf_score_min_maxed'] = (
        (final_tfidf_scores['tfidf_score'] - final_tfidf_scores['tfidf_score'].min()) /
        (final_tfidf_scores['tfidf_score'].max() - final_tfidf_scores['tfidf_score'].min())
    )

    file_path = DATA_PREPROCESSED_DIR / "user_product_tfidf_min_max_scaled.pq"

    final_tfidf_scores.to_parquet(file_path, index=False)
    logging.info(f"Saved user_product_tfidf min_max_scaled to {file_path}")

def combine_ratings(mode: str) -> None:
    """Combines frequency, recency and TF.IDF into one rating through a equally weighted linear combination"""
    logging.info("")
    frequency = pd.read_parquet(DATA_PREPROCESSED_DIR / "user_product_frequency.pq")
    recency = pd.read_parquet(DATA_PREPROCESSED_DIR / "user_product_recency_min_max_scaled.pq")
    tfidf = pd.read_parquet(DATA_PREPROCESSED_DIR / "user_product_tfidf_min_max_scaled.pq")

    merged = (
        frequency.merge(recency, on=['user_id', 'product_id'], how='inner')
                 .merge(tfidf, on=['user_id', 'product_id'], how='inner')
    )

    w_freq, w_rec, w_tfidf = ((1/3), (1/3), (1/3))

    merged['ranke_ui'] = (
        w_freq * merged['freq_ui'] +
        w_rec * merged['recency_score_min_maxed'] +
        w_tfidf * merged['tfidf_score_min_maxed']
    )

    final_ratings = merged[['user_id', 'product_id', 'ranke_ui']]
    final_ratings.columns = ["user", "item", 'rating']

    filename_parquet = f"{mode}_ratings_w_freq-{w_freq:.2f}_w_rec-{w_rec:.2f}_w_tfidf-{w_tfidf:.2f}.pq"
    file_path = DATA_PREPROCESSED_DIR / filename_parquet
    final_ratings.to_parquet(file_path, index=False)
    logging.info(f"Saved ratings to {file_path}")

def save_sparse_matrix() -> None:
    """Helper function to save to a csr matrix which is better optimised when a lot of values are 0"""
    logging.info("")
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

    users = ratings_long["user"].unique()
    products = ratings_long["item"].unique()
    shape = (len(users), len(products))

    # Create indices for users and movies
    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    product_cat = CategoricalDtype(categories=sorted(products), ordered=True)
    user_index = ratings_long["user"].astype(user_cat).cat.codes
    product_index = ratings_long["item"].astype(product_cat).cat.codes

    # Conversion via COO matrix
    coo = sparse.coo_matrix((ratings_long["rating"], (user_index, product_index)), shape=shape)
    ratings_matrix = coo.tocsr()

    with open(DATA_PREPROCESSED_DIR / "ratings_csr_matrix.pkl", 'wb') as fp:
        pkl.dump(ratings_matrix, file=fp)

def save_unique_users() -> None:
    """Helper function to save unique users in the rating matrix"""
    logging.info("")
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

    users = ratings_long["user"].drop_duplicates().to_frame().reset_index()

    file_path = DATA_PREPROCESSED_DIR / "unique_users.pq"
    users.to_parquet(file_path, index=False)
    logging.info(f"Saved ratings to {file_path}")

def save_unique_products() -> None:
    """Helper function to save unique products in the rating matrix"""
    logging.info("")
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

    products = ratings_long["item"].drop_duplicates().to_frame().reset_index()

    file_path = DATA_PREPROCESSED_DIR / "unique_products.pq"
    products.to_parquet(file_path, index=False)
    logging.info(f"Saved ratings to {file_path}")

def save_aisle_ratings(mode: str) -> None:
    """Helper function to group by aisle and aggregate each group with the mean, using the rating matrix"""
    logging.info("")
    # loading products and ratings dataframes
    products_df = pd.read_csv(PRODUCTS_PATH_CSV)
    ratings_df = pd.read_parquet(f"{DATA_PREPROCESSED_DIR}/{mode}_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

    # left joining the products dataframe onto the ratings dataframe
    joined_df = ratings_df.merge(products_df, how="left", left_on="item", right_on="product_id")

    # grouping by user and aisle_id and computing the average rating per (user, aisle_id)
    grp_df = joined_df.groupby(["user", "aisle_id"])["rating"].mean().reset_index()
    grp_df = grp_df.rename(columns={"aisle_id": "item"})

    # saving the aisle ratings
    file_path = f"{DATA_PREPROCESSED_DIR}/{mode}_aisle_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq"
    grp_df.to_parquet(file_path, index=False)
    logging.info(f"Saved ratings to {file_path}")

def save_aisle_top_products() -> None:
    """
    Helper function to group by aisle and aggregate using count to find top products within each aisle.
    Is used for recommending things within each aisle.
    """
    logging.info("")
    # loading order products train and products dataframes
    op_train = pd.read_parquet(ORDER_PRODUCTS__TRAIN_PATH)
    products_df = pd.read_csv(PRODUCTS_PATH_CSV)

    # left joining products_df onto op_train
    merged_df = op_train.merge(products_df, how="left", on="product_id")

    # counting occurrence for each product per aisle
    agg_df = (merged_df
              .groupby(["product_id", "aisle_id"])
              .size()
              .reset_index(name="count")
              .sort_values(['aisle_id', 'count'], ascending=[True, False])
              )

    # saving dictionary to parquet
    file_path = AISLE_TOP_PRODUCTS_PATH # Fixed
    agg_df.to_parquet(file_path, index=False)
    logging.info(f"Saved aisle top products to {file_path}")


# And this code block actually calculates the ratings:

# In[ ]:


# path to order_products for each set
path_dict = {"train": ORDER_PRODUCTS__TRAIN_PATH,
             "val": ORDER_PRODUCTS__VAL_PATH,
             "test": ORDER_PRODUCTS__TEST_PATH,
             }

# loading orders
orders = pd.read_parquet(ORDERS_PATH)

# saving ratings and aisle ratings for each set {train, val, test}
for mode, path in path_dict.items():
    order_products = pd.read_parquet(path)
    
    merged_df = pd.merge(
        order_products,
        orders,
        on="order_id",
        how="inner"
    )

    calculate_user_product_frequency(merged_df=merged_df.copy())
    calculate_user_product_recency(merged_df=merged_df.copy(), lam=0.0015)
    calculate_tf_idf(merged_df=merged_df.copy(), orders=orders.copy())
    combine_ratings(mode=mode)

    if mode == "train":
        save_unique_users()
        save_unique_products()
        save_aisle_top_products()

    save_aisle_ratings(mode=mode)


# ## 4. Clustering
# 
# Clustering is performed to try and downsize the rating matrix. We cluster users. In the dataset there are ~200,000 unique users and about ~50,000 unique products. This leads to a rating matrix of size $\mathbb{R}^{200,000 \times 50,000}$, which is massive. It is about
# $$
# 200,000 \times 50,000 \times 4 \text{ bytes} \times 10^{-9} = 40 \text{GB}
# $$
# 
# Which is a lot for commodity hardware. That is also the reason why we cannot run the full rating matrix on our own computers, and must instead use the HPC. 
# 
# If clustering gives good results at a fraction of the size (approximately 30 MB to 500 MB, depending on cluster size), this would be a viable option. 
# 
# We use KMeans clustering for this. We implement our own version of KMeans clustering, which we have implemented as a class, inheriting from sklearns base classes. 

# In[ ]:


# UTILS
class KMeans_self_implemented(BaseEstimator, ClassifierMixin):
    """
        KMeans clustering using Lloyd's algoritm. There are two initialisation
        strategies. We recommend 'kmeans++', because it is more consistent,
        and less likely to be stuck in local minimas.

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


# The following code block contains functions used to perform clustering:

# In[ ]:


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

    return y_pred


def assign_cluster_to_users(y_pred, ratings_long):
    """Assigns the predicted cluster to the rating DataFrame"""
    user_cluster_preds = (ratings_long[["user"]]
                    .drop_duplicates()
                    .assign(cluster=y_pred)
    )

    return user_cluster_preds


def save_cluster_user_dict(user_cluster_preds, save=True):
    """Helper function to save a dictionary with clusters as keys and list of users as values"""
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
    """Calculates the mean rating for each cluster and saves to disk"""
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


# This code block saves the generated clusters and their ratings across the users belonging to the cluster. The clusters are also generated within the relevant CF configurations further down. We are just saving the clusters in case one wants to explore them and/or play around with the number of generated clusters.

# In[ ]:


# loading train ratings
ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

# converting to a sparse coo matrix
_, _, ratings_matrix = convert_to_user_term_matrix(ratings_long)

# clustering users (50 clusters by default)
y_pred = cluster_and_predict_users(ratings_matrix=ratings_matrix)

# assigning users to clusters
user_cluster_preds = assign_cluster_to_users(y_pred=y_pred, ratings_long=ratings_long)

# saving cluster_user_dict and cluster_ratings
save_cluster_user_dict(user_cluster_preds=user_cluster_preds)
save_cluster_ratings(ratings_long=ratings_long, user_cluster_preds=user_cluster_preds)


# ## 5. Baseline Recommender

# The following functions are used to construct and evaluate our baseline recommender. The baseline is a top_n recommender, recommending the top_n most bought products across the entire dataset. We will use this baseline to compare against our later recommenders to see if they are an improvement.

# In[ ]:


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


# Now we can define and evaluate the baseline recommender:

# In[ ]:


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
test_product_dict = construct_test_product_dict(mode="test")
eval_dict = eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_product_dict)

# computing average hit-rate and ndcg
avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
avg_ndcg = np.mean([metric_dict[f"ndcg@{top_n}"] for metric_dict in eval_dict.values()])

logging.info(f"The average hit-rate across all users is {avg_hr:.4f}")
logging.info(f"The average ndcg@{top_n} across all users is {avg_ndcg:.4f}")

# saving results to results.csv
row = ["baseline", top_n, avg_hr, avg_ndcg]
with open(RESULTS_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(row)


# ## 6. Collaborative Filtering

# The following code block loads data preliminaries for constructing the CF recommenders. These include ratings, aisle data, and the products data.

# In[ ]:


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

# preparing ratings for cluster ratings
ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
_, _, ratings_matrix = convert_to_user_term_matrix(ratings_long)

# loading product data containing aisle information
products_df = pd.read_csv(PRODUCTS_PATH_CSV)

# number of recommendations to generate for each user
n_recs = 6


# Additionally, we need to define the following functions:

# In[ ]:


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
        
        # run an epoch
        next(epoch_gen)

        # user matrix of shape [n_users × k]
        U = mf.user_features_

        # item matrix of shape [n_items × k]
        V = mf.item_features_

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


# ### CF_Full
# This is the first CF configuration trying to predict the full rating matrix of size `n_users` $\times$ `n_items`.

# In[ ]:


# CF fit
pred_ratings, mf = get_cf_scores(features=2000, 
                                 rating_df_train=train_ratings, 
                                 rating_df_val=val_ratings, 
                                 reg=0.001,
                                 bias=False)

# retrieving recommendations
recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)

# saving recommendations
with open(DATA_PREPROCESSED_DIR / "cf_all_users_all_items_recs.pkl", 'wb') as fp:
    pkl.dump(recs_dict,file=fp)

# evaluating on test set
test_products = construct_test_product_dict(mode="test")
eval_dict = eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_products)

# computing average hit-rate and ndcg
avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
avg_ndcg = np.mean([metric_dict[f"ndcg@{n_recs}"] for metric_dict in eval_dict.values()])

print(f"The average hit-rate on the test set across all users is {avg_hr:.6f}")
print(f"The average ndcg@{n_recs} on the test set across all users is {avg_ndcg:.6f}")

# saving results to csv file
row = ["cf_all_users_all_items", n_recs, avg_hr, avg_ndcg]
with open(RESULTS_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(row)


# ### CF_A
# This is the second CF configuration using all users and product aisles instead of each individual product. This reduces the amount of columns, thus reducing the overall size of the rating matrix.

# In[ ]:


# CF fit
pred_ratings, mf = get_cf_scores(features=500, 
                                 rating_df_train=train_ratings_aisles, 
                                 rating_df_val=val_ratings, 
                                 reg=0.0001,
                                 damping=5,
                                 bias=True,
                                 aisles=True,
                                 aisle_dict=aisle_dict)

# retrieving recommendations
aisle_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)
recs_dict = convert_aisle_recs(recs_dict=aisle_recs_dict, aisle_top_products=aisle_dict)

# saving recommendations
with open(DATA_PREPROCESSED_DIR / "cf_all_users_aisles_recs.pkl", 'wb') as fp:
    pkl.dump(recs_dict,file=fp)

# evaluating on test set
test_products = construct_test_product_dict(mode="test")
eval_dict = eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_products)

# computing average hit-rate and ndcg
avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
avg_ndcg = np.mean([metric_dict[f"ndcg@{n_recs}"] for metric_dict in eval_dict.values()])

print(f"The average hit-rate on the test set across all users is {avg_hr:.6f}")
print(f"The average ndcg@{n_recs} on the test set across all users is {avg_ndcg:.6f}")

# saving results to csv
row = ["cf_all_users_aisles", n_recs, avg_hr, avg_ndcg]
with open(RESULTS_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(row)


# ### CF_C
# And now onto the third CF configuration using user clusters and all products. This reduces the number of rows of the rating matrix, reducing the overall size of the matrix.
# 

# In[ ]:


# cluster users
y_pred = cluster_and_predict_users(ratings_matrix=ratings_matrix, n_clusters=100)
user_cluster_preds = assign_cluster_to_users(y_pred=y_pred, ratings_long=ratings_long)
cluster_user_dict = save_cluster_user_dict(user_cluster_preds=user_cluster_preds, save=False)
train_ratings = save_cluster_ratings(ratings_long=ratings_long, user_cluster_preds=user_cluster_preds, save=False)

# CF fit
pred_ratings, mf = get_cf_scores(features=2000, 
                                    rating_df_train=train_ratings, 
                                    rating_df_val=val_ratings, 
                                    reg=0.001,
                                    damping=5,
                                    bias=True,
                                    user_clusters=True,
                                    cluster_user_dict=cluster_user_dict)

# retrieving recommendations
cluster_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs)
recs_dict = convert_user_cluster_recs(cluster_recs_dict=cluster_recs_dict, cluster_user_dict=cluster_user_dict)

# saving recommendations
with open(DATA_PREPROCESSED_DIR / "cf_user_clusters_all_items_recs.pkl", 'wb') as fp:
    pkl.dump(recs_dict,file=fp)

# evaluating on test set
test_products = construct_test_product_dict(mode="test")
eval_dict = eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_products)

# computing average hit-rate and ndcg
avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
avg_ndcg = np.mean([metric_dict[f"ndcg@{n_recs}"] for metric_dict in eval_dict.values()])

print(f"The average hit-rate on the test set across all users is {avg_hr:.6f}")
print(f"The average ndcg@{n_recs} on the test set across all users is {avg_ndcg:.6f}")

# saving results to csv
row = ["cf_user_clusters_all_items", n_recs, avg_hr, avg_ndcg]
with open(RESULTS_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(row)


# ### CF_AC
# This is the final CF configuration using both user clusters and product aisles, resulting in the smallest rating matrix to predict among all four CF configurations.

# In[ ]:


# cluster users
y_pred = cluster_and_predict_users(ratings_matrix=ratings_matrix, n_clusters=200)
user_cluster_preds = assign_cluster_to_users(y_pred=y_pred, ratings_long=ratings_long)
cluster_user_dict = save_cluster_user_dict(user_cluster_preds=user_cluster_preds, save=False)
train_cluster_ratings = save_cluster_ratings(ratings_long=ratings_long, user_cluster_preds=user_cluster_preds, save=False)

# preparing train ratings
joined_df = train_cluster_ratings.merge(products_df, how="left", left_on="item", right_on="product_id")
grp_df = joined_df.groupby(["user", "aisle_id"])["rating"].mean().reset_index()
train_ratings = grp_df.rename(columns={"aisle_id": "item"})

# CF fit
pred_ratings, mf = get_cf_scores(features=500, 
                                    rating_df_train=train_ratings, 
                                    rating_df_val=val_ratings, 
                                    reg=0.0001,
                                    damping=5,
                                    bias=True,
                                    aisles=True,
                                    aisle_dict=aisle_dict,
                                    user_clusters=True,
                                    cluster_user_dict=cluster_user_dict)

# retrieving recommendations
aisle_recs_dict = get_recs(pred_ratings=pred_ratings, mf=mf, n_recs=n_recs, d_hondts=True)
cluster_recs_dict = convert_aisle_recs(recs_dict=aisle_recs_dict, aisle_top_products=aisle_dict)
recs_dict = convert_user_cluster_recs(cluster_recs_dict=cluster_recs_dict, cluster_user_dict=cluster_user_dict)

# saving recommendations
with open(DATA_PREPROCESSED_DIR / "cf_user_clusters_aisles_recs.pkl", 'wb') as fp:
    pkl.dump(recs_dict,file=fp)

# evaluating on test set
test_products = construct_test_product_dict(mode="test")
eval_dict = eval_recs(recs_dict=recs_dict, rating_df=test_ratings, test_products=test_products)

# computing average hit-rate and ndcg
avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
avg_ndcg = np.mean([metric_dict[f"ndcg@{n_recs}"] for metric_dict in eval_dict.values()])

print(f"The average hit-rate on the test set across all users is {avg_hr:.6f}")
print(f"The average ndcg@{n_recs} on the test set across all users is {avg_ndcg:.6f}")

# saving results to csv
row = ["cf_user_clusters_aisles", n_recs, avg_hr, avg_ndcg]
with open(RESULTS_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(row)


# ## 7. Apriori

# Apriori uses a different approach, analyzing frequent co-purchase patterns. The result is association rules, which we use to recommend products to each user. 
# 
# We use the following functions for the apriori recommender:

# In[ ]:


def load_data() -> pd.DataFrame:
    """
    Load data and return a DataFrame with at least:
        user_id, order_id, product_id, product_name, add_to_cart_order, reordered
    """
    order_products = pd.read_parquet(ORDER_PRODUCTS__TRAIN_PATH)
    orders = pd.read_csv(ORDERS_PATH_CSV)
    products = pd.read_csv(PRODUCTS_PATH_CSV)

    # Keep only what we need from orders + products
    orders = orders[["order_id", "user_id"]]
    products = products[["product_id", "product_name"]]

    # Join so each row has user_id, order_id, product_id, product_name
    data = (
        order_products
        .merge(orders, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
    )

    if data["user_id"].isna().any():
        missing = data["user_id"].isna().sum()
        print(f"[WARN] {missing} rows have no user_id after join.")
    if data["product_name"].isna().any():
        missing = data["product_name"].isna().sum()
        print(f"[WARN] {missing} rows have no product_name after join.")

    print("[INFO] Loaded data with columns:", list(data.columns))
    print("[INFO] Rows:", len(data))
    return data


def mine_frequent_itemsets(
    transactions: List[Iterable[Hashable]],
    min_item_support: float = 0.002,
    support_grid: Optional[List[float]] = None,
    max_len: int = 3,
    use_low_memory: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """
    Run Apriori to mine frequent itemsets over the given transactions
    (which in our case are product_name lists).

    Returns:
        freq: DataFrame with 'itemsets' (frozenset of product_name) and 'support'
        chosen_support: the min_support actually used by apriori
    """
    N = len(transactions)
    if N == 0:
        raise ValueError("No transactions provided")

    if verbose:
        print(f"[INFO] Transactions N={N}")

    # 1) Pre-filter rare items BEFORE one-hot encoding
    if min_item_support is not None and min_item_support > 0:
        abs_min_count = max(1, int(np.ceil(min_item_support * N)))
        item_counts = Counter()
        for t in transactions:
            item_counts.update(set(t))  # count once per basket

        keep_items = {item for item, c in item_counts.items() if c >= abs_min_count}
        if verbose:
            U_raw = len(item_counts)
            U_kept = len(keep_items)
            print(
                f"[INFO] Unique items before filtering: {U_raw} "
                f"-> after min_item_support={min_item_support:.4g}: {U_kept}"
            )

        filtered_transactions = [
            [item for item in t if item in keep_items]
            for t in transactions
        ]
    else:
        filtered_transactions = transactions

    # 2) TransactionEncoder -> dense bool DataFrame
    te = TransactionEncoder()
    X_bool = te.fit(filtered_transactions).transform(filtered_transactions)
    X = pd.DataFrame(X_bool, columns=te.columns_, dtype=bool)
    del X_bool

    U = X.shape[1]
    if verbose:
        print(f"[INFO] After filtering & encoding: N={N}, U={U} (bool matrix)")

    # 3) Support ladder: use the lowest support that yields any itemsets
    if support_grid is None:
        base = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, max(1.0 / N, 0.001)]
        support_grid = sorted({s for s in base if s > 0}, reverse=True)
    else:
        support_grid = sorted({s for s in support_grid if s > 0}, reverse=True)

    if verbose:
        print(f"[INFO] Support grid (high -> low): {support_grid}")

    freq = pd.DataFrame()
    chosen_support = None

    last_nonempty = None
    last_s = None

    for s in support_grid:
        f = apriori(
            X,
            min_support=s,
            use_colnames=True,
            max_len=max_len,
            low_memory=use_low_memory,
        )

        lens = f["itemsets"].apply(len)
        pairs = f[lens == 2]

        if verbose:
            print(
                f"[DBG] min_support={s:.4f} -> "
                f"itemsets={len(f)} (pairs={len(pairs)})"
            )

        if len(f) > 0:
            last_nonempty = f
            last_s = s

        del f, pairs

    if last_nonempty is not None:
        freq = last_nonempty.sort_values("support", ascending=False).reset_index(drop=True)
        chosen_support = last_s

    if freq.empty or chosen_support is None:
        raise ValueError(
            "No frequent itemsets found even at lowest support. "
            "Try lowering min_item_support or adding more baskets."
        )

    lens = freq["itemsets"].apply(len)
    n1 = (lens == 1).sum()
    n2 = (lens == 2).sum()
    n3 = (lens == 3).sum()
    if verbose:
        print(
            f"[INFO] Using min_support={chosen_support:.4f}. "
            f"Singles={n1}, Pairs={n2}, Triples={n3}"
        )

    return freq, chosen_support


def build_association_rules(
    freq: pd.DataFrame,
    chosen_support: Optional[float] = None,
    *,
    rule_metric: str = "confidence",
    rule_min_threshold: float = 0.05,
    # strict pruning
    min_lift: Optional[float] = 1.1,
    max_lift: Optional[float] = None,
    min_confidence: Optional[float] = 0.1,
    min_interest: Optional[float] = None,
    prune_by_support: bool = True,
    # soft mode
    min_rules: Optional[int] = None,
    soft_min_confidence: float = 0.02,
    soft_min_lift: float = 1.0,
    soft_prune_by_support: bool = False,
    # global cap
    max_rules: Optional[int] = 100000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build and prune association rules from frequent itemsets.

    max_rules:
        After sorting by ["lift","confidence","support"] (descending),
        keep only the first `max_rules`.
    """
    rules_all = association_rules(
        freq, metric=rule_metric, min_threshold=rule_min_threshold
    ).copy()

    if verbose:
        print(f"[INFO] Raw rules: {len(rules_all)}")

    if rules_all.empty:
        if verbose:
            print(
                "[HINT] No rules produced. You need at least some 2-itemsets; "
                "try lowering min_support or adjusting rule_min_threshold."
            )
        return rules_all

    # interest = confidence - support(consequent)
    lens = freq["itemsets"].apply(len)
    singles = freq[lens == 1][["itemsets", "support"]]
    conseq_support = {
        next(iter(s)): sup for s, sup in zip(singles["itemsets"], singles["support"])
    }

    def get_consequent_support(cs: frozenset) -> float:
        return conseq_support.get(next(iter(cs)), np.nan)

    rules_all["interest"] = rules_all["confidence"] - rules_all["consequents"].apply(
        get_consequent_support
    )

    # strict pruning
    strict_mask = pd.Series(True, index=rules_all.index)

    if min_lift is not None:
        strict_mask &= rules_all["lift"] >= min_lift
    if max_lift is not None:
        strict_mask &= rules_all["lift"] <= max_lift
    if prune_by_support and chosen_support is not None:
        strict_mask &= rules_all["support"] >= chosen_support
    if min_confidence is not None:
        strict_mask &= rules_all["confidence"] >= min_confidence
    if min_interest is not None:
        strict_mask &= rules_all["interest"] >= min_interest

    strict_rules = rules_all[strict_mask].copy()

    if verbose:
        print(f"[INFO] Strict pruning -> {len(strict_rules)} rules")

    # soft mode (optional; often you can just skip by min_rules=None)
    if min_rules is None or min_rules <= 0:
        selected = strict_rules
        mode = "strict-only"
    else:
        if len(strict_rules) >= min_rules:
            selected = strict_rules
            mode = "strict"
        else:
            soft_mask = pd.Series(True, index=rules_all.index)

            if soft_min_lift is not None:
                soft_mask &= rules_all["lift"] >= soft_min_lift
            if soft_min_confidence is not None:
                soft_mask &= rules_all["confidence"] >= soft_min_confidence
            if soft_prune_by_support and chosen_support is not None:
                soft_mask &= rules_all["support"] >= chosen_support

            soft_rules = rules_all[soft_mask].copy()

            if verbose:
                print(
                    f"[INFO] Strict rules ({len(strict_rules)}) < min_rules={min_rules}, "
                    f"using soft pruning -> {len(soft_rules)} rules"
                )

            selected = soft_rules
            mode = "soft"

    selected = selected.sort_values(
        ["lift", "confidence", "support"], ascending=False
    ).reset_index(drop=True)

    if max_rules is not None and max_rules > 0:
        selected = selected.head(max_rules)

    if verbose:
        print(
            f"[INFO] After {mode} pruning & capping: {len(selected)} rules "
            f"(max_rules={max_rules}, min_rules={min_rules})"
        )

    return selected


def build_rules_index(
    rules: pd.DataFrame,
) -> List[Tuple[Tuple[str, ...], str, float, float, float, float]]:
    """
    Turn a rules DataFrame into a convenient index:
    list of (antecedent_tuple, consequent_item_name, lift, confidence, support, interest)
    Only keep rules with single-item consequents.
    """
    rules_idx = []
    for _, r in rules.iterrows():
        A = tuple(sorted(list(r["antecedents"])))
        C = list(r["consequents"])[0] if len(r["consequents"]) == 1 else None
        if C is None:
            continue
        rules_idx.append((A, C, r["lift"], r["confidence"], r["support"], r["interest"]))
    return rules_idx


def train_apriori_model(
    data: pd.DataFrame,
    user_col: str = "user_id",
    order_col: str = "order_id",
    item_col: str = "product_name",  # use product_name
    verbose: bool = True,
    min_item_support = 0.002,
    support_grid=[0.001],
    rule_min_threshold=0.05,
    min_confidence=0.1
):
    """
    Train Apriori on all baskets in `data` and return:
      - freq: frequent itemsets (product_name)
      - rules: association rules (product_name)
      - rules_idx: indexed rules for fast lookup
      - popularity: global item popularity (Series) over product_name
    """
    # Build list of transactions (one list per order, of product_name)
    basket_groups = data.groupby(order_col)[item_col].apply(list)
    transactions = basket_groups.tolist()

    # Global item popularity (for fallback recommendations)
    popularity = (
        data[item_col]
        .value_counts()
        .sort_values(ascending=False)
    )

    # 1) Frequent itemsets
    freq, chosen_support = mine_frequent_itemsets(
        transactions=transactions,
        min_item_support=min_item_support,  # item pre-filter
        support_grid=support_grid,          # candidate supports for Apriori
        max_len=5,                          # up to 5-item sets
        use_low_memory=True,
        verbose=verbose,
    )

    # 2) Rules with “reasonable” thresholds
    rules = build_association_rules(
        freq=freq,
        chosen_support=chosen_support,
        rule_metric="confidence",
        rule_min_threshold=rule_min_threshold,  # min confidence for generating rules
        min_lift=1.1,                           # positive association
        max_lift=20.0,                          # avoid crazy outliers
        min_confidence=min_confidence,           # extra confidence filter
        prune_by_support=True,                  # rule support >= chosen_support
        min_rules=None,
        max_rules=100000,
        verbose=verbose,
    )

    rules_idx = build_rules_index(rules)

    if verbose:
        print(f"[DBG] rules rows: {len(rules)}")
        print(f"[DBG] rules_idx entries: {len(rules_idx)}")

    return freq, rules, rules_idx, popularity


def recommend_for_user(
    user_id: Any,
    data: pd.DataFrame,
    rules_idx: List[Tuple[Tuple[str, ...], str, float, float, float, float]],
    popularity: pd.Series,
    *,
    user_col: str = "user_id",
    item_col: str = "product_name",  # we recommend names
    k: int = 6,
    subset_k: int = 2,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    Recommend K items (product_name) for a given user_id.

    NOTE: This version ALLOWS recommending items the user has already bought.
    """
    user_data = data[data[user_col] == user_id]
    if user_data.empty:
        raise ValueError(f"No data found for user_id={user_id}")

    # User's item frequency (by product_name)
    user_item_counts = user_data[item_col].value_counts()
    max_count = user_item_counts.max()
    user_item_weight = (user_item_counts / max_count).to_dict()

    # All items the user has ever bought
    user_items = sorted(user_item_counts.index.tolist())
    S = set(user_items)

    best_score = defaultdict(float)
    meta: Dict[Any, Dict[str, Any]] = {}

    # Use rules where antecedent is a small subset of user's items
    for A, C, lift_, conf_, supp_, intr_ in rules_idx:
        if len(A) <= subset_k and set(A).issubset(S):
            # We allow recommending items already in S

            # Base rule score: lift * confidence
            rule_score = lift_ * conf_

            # Antecedent weight: how core these items are for this user
            antecedent_weights = [user_item_weight.get(a, 0.0) for a in A]
            avg_ante_weight = float(np.mean(antecedent_weights)) if antecedent_weights else 0.0

            # Final score: combine rule strength and user-specific weight
            score = rule_score * (1.0 + beta * avg_ante_weight)

            if score > best_score[C]:
                best_score[C] = score
                meta[C] = {
                    "antecedent": A,
                    "lift": lift_,
                    "confidence": conf_,
                    "support": supp_,
                    "interest": intr_,
                    "antecedent_weight": avg_ante_weight,
                }

    # Turn into DataFrame
    rows = [
        (
            item,  # this is product_name
            score,
            meta[item]["antecedent"],
            meta[item]["lift"],
            meta[item]["confidence"],
            meta[item]["support"],
            meta[item]["interest"],
            meta[item]["antecedent_weight"],
        )
        for item, score in best_score.items()
    ]

    recs = pd.DataFrame(
        rows,
        columns=[
            "product_name",     # human-readable name
            "score",
            "because",          # tuple of product_name(s)
            "lift",
            "confidence",
            "support",
            "interest",
            "antecedent_weight",
        ],
    ).sort_values("score", ascending=False)

    # ---- Guarantee exactly k items (if possible) by filling from popularity ----
    already = set(recs["product_name"].tolist())

    if len(recs) < k:
        needed = k - len(recs)
        filler = [
            itm for itm in popularity.index
            if itm not in already      # avoid duplicates in final list
        ][:needed]
        if filler:
            filler_rows = [
                (itm, 0.0, tuple(), np.nan, np.nan, np.nan, np.nan, 0.0)
                for itm in filler
            ]
            filler_df = pd.DataFrame(
                filler_rows,
                columns=[
                    "product_name",
                    "score",
                    "because",
                    "lift",
                    "confidence",
                    "support",
                    "interest",
                    "antecedent_weight",
                ],
            )
            recs = pd.concat([recs, filler_df], ignore_index=True)

    recs = recs.head(k)
    return recs


def recommend_for_users(
    user_ids: Iterable[Any],
    data: pd.DataFrame,
    rules_idx: List[Tuple[Tuple[str, ...], str, float, float, float, float]],
    popularity: pd.Series,
    *,
    user_col: str = "user_id",
    item_col: str = "product_name",
    k: int = 6,
    subset_k: int = 2,
    beta: float = 1.0,
) -> pd.DataFrame:
    """
    Recommend items (product_name) for multiple users.

    Returns a DataFrame with one row per (user_id, recommended product_name),
    including rank per user.
    """
    all_recs = []

    for uid in user_ids:
        print(uid)
        try:
            recs = recommend_for_user(
                user_id=uid,
                data=data,
                rules_idx=rules_idx,
                popularity=popularity,
                user_col=user_col,
                item_col=item_col,
                k=k,
                subset_k=subset_k,
                beta=beta,
            )
        except ValueError:
            # user has no data
            continue

        recs = recs.copy()
        recs.insert(0, "user_id", uid)
        recs["rank"] = np.arange(1, len(recs) + 1)
        all_recs.append(recs)

    if not all_recs:
        return pd.DataFrame(
            columns=[
                "user_id",
                "rank",
                "product_name",
                "score",
                "because",
                "lift",
                "confidence",
                "support",
                "interest",
                "antecedent_weight",
            ]
        )

    all_recs_df = pd.concat(all_recs, ignore_index=True)
    return all_recs_df


def build_user_recs_dict(
    multi_recs: pd.DataFrame,
    data: pd.DataFrame,
    k: int = 6,
) -> Dict[Any, List[Any]]:
    """
    Convert multi-user recommendations (by product_name) into a dictionary:

        { user_id: [product_id_1, ..., product_id_k] }

    using the product_id <-> product_name mapping from `data`.
    """
    # Build mapping product_name -> product_id
    # (Assume mostly 1-to-1 in Instacart; if duplicates, first occurrence wins.)
    prod_map = (
        data[["product_id", "product_name"]]
        .dropna()
        .drop_duplicates(subset=["product_name"])
    )
    name_to_id = dict(zip(prod_map["product_name"], prod_map["product_id"]))

    user_recs: Dict[Any, List[Any]] = {}

    # Ensure per-user recommendations are ordered by rank
    for uid, grp in multi_recs.groupby("user_id"):
        grp_sorted = grp.sort_values("rank")
        names = grp_sorted["product_name"].head(k).tolist()
        ids = [name_to_id.get(n) for n in names]
        # Filter out any None (in case of missing mapping)
        ids = [pid for pid in ids if pid is not None]
        user_recs[int(uid)] = ids

    return user_recs


# Now we are ready to train and evaluate the apriori recommender:

# In[ ]:


# Global random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# loading val ratings and products for evaluation 
logging.info("Loading validation data")
val_ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "val_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")
val_products = construct_test_product_dict(mode="val")

# 1) Load & prepare data from data/
logging.info("Loading and preparing data")
data = load_data()
all_user_ids = data["user_id"].dropna().drop_duplicates()


logging.info("Training Apriori model")
# 2) Train Apriori model on all users (once), using product_name
_, _, rules_idx, popularity = train_apriori_model(
    data,
    user_col="user_id",
    order_col="order_id",
    item_col="product_name",
    verbose=False,
    support_grid=[0.01],
    min_confidence=0.1,
)

logging.info("Generating recommendations for all users")
# 3) Recommendations for ALL users (no sampling)
multi_recs = recommend_for_users(
    user_ids=all_user_ids,
    data=data,
    rules_idx=rules_idx,
    popularity=popularity,
    user_col="user_id",
    item_col="product_name",
    k=6,
    subset_k=2,
    beta=1.0,
)

# 4) Build user_id -> [product_id,...] dict (6 per user) and save it
logging.info("Building dictionary containing recommendations for each user")
user_recs_dict = build_user_recs_dict(
    multi_recs=multi_recs,
    data=data,
    k=6,
)

# 5) Evaluating the recommendations
logging.info("Evaluating the recommendations")
eval_dict = eval_recs(recs_dict=user_recs_dict, rating_df=val_ratings, test_products=val_products)
avg_hr = np.mean([metric_dict["hit-rate"] for metric_dict in eval_dict.values()])
avg_ndcg = np.mean([metric_dict[f"ndcg@6"] for metric_dict in eval_dict.values()])
print(f"Average hit-rate: {avg_hr:.6f}", flush=True)
print(f"Average ndcg@6: {avg_ndcg:.6f}", flush=True)

# 6) Saving the evaluation results to a csv file
row = ["mba", 6, avg_hr, avg_ndcg]
with open(RESULTS_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(row)


# ## 8. PCY

# This is our final recommender, the PCY recommender. Once again, we must first define a series of functions:

# In[ ]:


def load_train_data() -> pd.DataFrame:
    """Load train orders and order_products in line with the project splits."""
    orders = pd.read_parquet(ORDERS_PATH)
    op_train = pd.read_parquet(ORDER_PRODUCTS__TRAIN_PATH)

    train_order_ids = orders.loc[orders["eval_set"] == "train", "order_id"]
    op_train = op_train[op_train["order_id"].isin(train_order_ids)]

    return op_train


def build_baskets_and_item_counts(op_train: pd.DataFrame):
    """
    Build order-level baskets and global item frequency counts.
    Each basket is a set of product_ids for one order_id.
    """
    baskets_series = (
        op_train.groupby("order_id")["product_id"]
        .apply(set)
    )
    baskets = list(baskets_series)

    item_counts = Counter()
    for b in baskets:
        item_counts.update(b)

    return baskets, item_counts


def run_pass1(
    support_percentages=None,
    bucket_multipliers=None,
    save=True,
):
    """
    PCY Pass 1:
    - Build baskets + item_counts from TRAIN
    - Run grid over (bucket_multipliers × support_percentages)
    - Compute number of frequent buckets per combination
    - Save results to DATA_PREPROCESSED_DIR / 'pcy'
    """
    if support_percentages is None:
        support_percentages = [0.001, 0.002, 0.005, 0.01]  # 0.1%, 0.2%, 0.5%, 1%

    if bucket_multipliers is None:
        bucket_multipliers = [1.0, 1.5, 2.0, 3.0]

    logging.info("Loading train data for PCY Pass 1")
    op_train = load_train_data()

    logging.info("Building baskets and item counts")
    baskets, item_counts = build_baskets_and_item_counts(op_train)
    num_baskets = len(baskets)
    num_unique_items = len(item_counts)

    logging.info(f"Number of baskets: {num_baskets}")
    logging.info(f"Number of unique items: {num_unique_items}")

    # Bucket sizes from multipliers
    bucket_options = [int(num_unique_items * m) for m in bucket_multipliers]
    logging.info(f"Bucket sizes: {bucket_options}")

    # Results matrix: rows = bucket_options, cols = support_percentages
    results = np.zeros((len(bucket_options), len(support_percentages)), dtype=int)

    # Grid search: for each bucket size, one pass over baskets
    for i, num_buckets in enumerate(bucket_options):
        logging.info(f"Running Pass 1 for num_buckets={num_buckets}")

        bucket_counts = np.zeros(num_buckets, dtype=np.int32)

        # Ccount all pairs per bucket size
        for basket in baskets:
            b = sorted(basket)
            for a in range(len(b)):
                ia = b[a]
                for c in range(a + 1, len(b)):
                    ib = b[c]
                    h = (ia * 13 + ib * 7) % num_buckets
                    bucket_counts[h] += 1

        # Different support thresholds reuse the same bucket_counts
        for j, sp in enumerate(support_percentages):
            support_threshold = int(num_baskets * sp)
            frequent_buckets = int((bucket_counts >= support_threshold).sum())
            results[i, j] = frequent_buckets

            logging.info(
                f"num_buckets={num_buckets}, support={sp:.4f} "
                f"→ threshold={support_threshold}, frequent_buckets={frequent_buckets}"
            )

    logging.info("PCY Pass 1 grid search finished")

    if save:
        pcy_dir = DATA_PREPROCESSED_DIR / "pcy"
        os.makedirs(pcy_dir, exist_ok=True)
        save_path = pcy_dir / "pcy_pass1_results.pkl"

        save_data = {
            "num_baskets": num_baskets,
            "num_unique_items": num_unique_items,
            "bucket_multipliers": bucket_multipliers,
            "support_percentages": support_percentages,
            "bucket_options": bucket_options,
            "results": results,
        }

        with open(save_path, "wb") as f:
            pkl.dump(save_data, f)

        logging.info(f"Pass 1 results saved to {save_path}")

        # Save selected PCY hyperparameters for Pass 2
        chosen_multiplier = 1.5        # from Pass 1 analysis
        chosen_support = 0.005         # 0.5%

        chosen_num_buckets = int(num_unique_items * chosen_multiplier)

        params = {
            "bucket_multiplier": chosen_multiplier,
            "support_percentage": chosen_support,
            "num_buckets": chosen_num_buckets,
            "num_baskets": num_baskets,
            "num_unique_items": num_unique_items,
        }

        params_path = pcy_dir / "pcy_params.pkl"
        with open(params_path, "wb") as f:
            pkl.dump(params, f)

        logging.info(f"Pass 1 selected params saved to {params_path}")

    return {
        "num_baskets": num_baskets,
        "num_unique_items": num_unique_items,
        "bucket_multipliers": bucket_multipliers,
        "support_percentages": support_percentages,
        "bucket_options": bucket_options,
        "results": results,
    }


def run_pass2(save=True):
    """
    PCY Pass 2:
    - Reload TRAIN baskets + item_counts
    - Load chosen PCY hyperparameters from pcy_params.pkl
    - Build bitmap (frequent buckets) for chosen num_buckets & support
    - Count frequent pairs using (frequent singles + bitmap)
    - Save frequent_pairs and item_counts for downstream use
    """
    pcy_dir = DATA_PREPROCESSED_DIR / "pcy"
    params_path = pcy_dir / "pcy_params.pkl"

    if not params_path.exists():
        raise FileNotFoundError(
            f"PCY params not found at {params_path}. "
            "Run Pass 1 first to generate pcy_params.pkl."
        )

    with open(params_path, "rb") as f:
        params = pkl.load(f)

    bucket_multiplier = params["bucket_multiplier"]
    support_percentage = params["support_percentage"]

    logging.info("Loading train data for PCY Pass 2")
    op_train = load_train_data()

    logging.info("Building baskets and item counts")
    baskets, item_counts = build_baskets_and_item_counts(op_train)
    num_baskets = len(baskets)
    num_unique_items = len(item_counts)

    logging.info(f"Number of baskets (Pass 2): {num_baskets}")
    logging.info(f"Number of unique items (Pass 2): {num_unique_items}")

    # Derive final hyperparameters exactly as στο notebook:
    num_buckets = int(num_unique_items * bucket_multiplier)
    support_threshold = int(num_baskets * support_percentage)

    logging.info(
        f"Pass 2 using num_buckets={num_buckets}, "
        f"support={support_percentage:.4f} "
        f"→ threshold={support_threshold}"
    )

    # Frequent single items with respect to final support threshold
    frequent_single_items = {
        item for item, cnt in item_counts.items()
        if cnt >= support_threshold
    }
    logging.info(f"Number of frequent single items: {len(frequent_single_items)}")

    # ---- First sweep: bucket counts (for bitmap) ----
    bucket_counts = np.zeros(num_buckets, dtype=np.int32)

    for basket in baskets:
        # keep only frequent singles to reduce work
        filtered = [it for it in basket if it in frequent_single_items]
        if len(filtered) < 2:
            continue
        filtered.sort()

        for i in range(len(filtered)):
            a = filtered[i]
            for j in range(i + 1, len(filtered)):
                b = filtered[j]
                h = (a * 13 + b * 7) % num_buckets
                bucket_counts[h] += 1

    bitmap = bucket_counts >= support_threshold
    n_frequent_buckets = int(bitmap.sum())
    logging.info(f"Number of frequent buckets in Pass 2: {n_frequent_buckets}")

    # ---- Second sweep: count candidate pairs ----
    pair_counts = Counter()

    for basket in baskets:
        filtered = [it for it in basket if it in frequent_single_items]
        if len(filtered) < 2:
            continue
        filtered.sort()

        for i in range(len(filtered)):
            a = filtered[i]
            for j in range(i + 1, len(filtered)):
                b = filtered[j]
                h = (a * 13 + b * 7) % num_buckets

                # PCY condition: bucket must be frequent
                if not bitmap[h]:
                    continue

                pair_counts[(a, b)] += 1

    # Final frequent pairs
    frequent_pairs = {
        pair: cnt for pair, cnt in pair_counts.items()
        if cnt >= support_threshold
    }
    logging.info(f"Number of frequent pairs: {len(frequent_pairs)}")

    if save:
        os.makedirs(pcy_dir, exist_ok=True)
        save_path = pcy_dir / "pcy_pass2_frequent_pairs.pkl"

        save_data = {
            "frequent_pairs": frequent_pairs,
            "item_counts": dict(item_counts),
            "support_threshold": support_threshold,
            "support_percentage": support_percentage,
            "num_buckets": num_buckets,
            "num_baskets": num_baskets,
        }

        with open(save_path, "wb") as f:
            pkl.dump(save_data, f)

        logging.info(f"Pass 2 frequent pairs saved to {save_path}")

    return frequent_pairs, item_counts


def load_pcy_artifacts():
    """
    Load PCY hyperparameters (from Pass 1) and frequent pairs (from Pass 2).
    Assumes files are stored under DATA_PREPROCESSED_DIR / "pcy".
    """
    pcy_dir = DATA_PREPROCESSED_DIR / "pcy"

    params_path = pcy_dir / "pcy_params.pkl"
    pairs_path = pcy_dir / "pcy_pass2_frequent_pairs.pkl"

    if not params_path.exists():
        raise FileNotFoundError(f"PCY params not found at {params_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"PCY frequent pairs not found at {pairs_path}")

    with open(params_path, "rb") as f:
        params = pkl.load(f)

    with open(pairs_path, "rb") as f:
        pairs_data = pkl.load(f)

    frequent_pairs = pairs_data["frequent_pairs"]      # dict[(i,j)] -> count
    item_counts = pairs_data["item_counts"]            # dict[item] -> count

    num_baskets = params["num_baskets"]

    logging.info(
        "Loaded PCY artefacts: %d frequent pairs, %d items, %d baskets",
        len(frequent_pairs),
        len(item_counts),
        num_baskets,
    )

    return frequent_pairs, item_counts, num_baskets


def build_rule_dataframe(frequent_pairs, item_counts, num_baskets):
    """
    Convert frequent_pairs + item_counts into a DataFrame with
    support, confidence and lift for both A→B and B→A.
    """
    rows = []

    for (a, b), count_ab in frequent_pairs.items():
        support_ab = count_ab / num_baskets

        support_a = item_counts[a] / num_baskets
        support_b = item_counts[b] / num_baskets

        # Basic guards against division by zero (should not happen in practice)
        if support_a == 0 or support_b == 0:
            continue

        confidence_a_b = support_ab / support_a
        confidence_b_a = support_ab / support_b
        lift_ab = support_ab / (support_a * support_b)

        rows.append(
            {
                "antecedent": a,
                "consequent": b,
                "support": support_ab,
                "confidence": confidence_a_b,
                "lift": lift_ab,
            }
        )
        rows.append(
            {
                "antecedent": b,
                "consequent": a,
                "support": support_ab,
                "confidence": confidence_b_a,
                "lift": lift_ab,
            }
        )

    rules_df = pd.DataFrame(rows)
    logging.info("Constructed %d association rules from frequent pairs", len(rules_df))

    return rules_df


def build_rule_map(rules_df, min_confidence=0.01, min_lift=1.0, top_n=None):
    """
    Build item -> list of (consequent, lift, confidence) sorted by lift.
    Optional thresholds can be used to trim very weak rules.
    """
    rule_map = defaultdict(list)

    mask = (rules_df["confidence"] >= min_confidence) & (rules_df["lift"] >= min_lift)
    filtered = rules_df.loc[mask]

    for _, row in filtered.iterrows():
        a = int(row["antecedent"])
        b = int(row["consequent"])
        lift = float(row["lift"])
        conf = float(row["confidence"])
        rule_map[a].append((b, lift, conf))

    # sort and optionally truncate
    for a, lst in rule_map.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None and len(lst) > top_n:
            rule_map[a] = lst[:top_n]

    logging.info("Built rule_map for %d antecedent items", len(rule_map))

    return rule_map


def build_user_history():
    """
    Build user -> set(product_id) from the TRAIN split.
    We combine ORDER_PRODUCTS__TRAIN with ORDERS to get user_id.
    """
    # Load orders and restrict to train split
    orders = pd.read_parquet(ORDERS_PATH)
    orders_train = orders.loc[orders["eval_set"] == "train", ["order_id", "user_id"]]

    # Load order_products for train (already filtered by order_id in data_split)
    op_train = pd.read_parquet(ORDER_PRODUCTS__TRAIN_PATH)

    # Merge to attach user_id to each (order_id, product_id)
    op_train = op_train.merge(orders_train, on="order_id", how="inner")

    # Build user → set of products
    user_baskets = (
        op_train.groupby("user_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    logging.info("Built user history for %d users", len(user_baskets))
    return user_baskets


def recommend_for_user(user_id, user_history, rule_map, popular_items, k=6):
    """
    Return a ranked list of exactly k product_ids for a given user.
    - First uses PCY association rules (lift-based).
    - Then fills up with globally popular items as fallback.
    """
    purchased = user_history.get(user_id, set())
    if not isinstance(purchased, set):
        purchased = set(purchased)

    scores = defaultdict(float)

    # Rule-based recommendations from PCY
    for a in purchased:
        if a not in rule_map:
            continue
        for b, lift, _ in rule_map[a]:
            if b in purchased:
                continue
            scores[b] += lift

    # Sort rule-based candidates by score
    recs = []
    if scores:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recs = [item for item, _ in ranked[:k]]

    # Fallback: fill up to k with popular items not yet recommended or purchased
    if len(recs) < k:
        for item in popular_items:
            if item in purchased:
                continue
            if item in recs:
                continue
            recs.append(item)
            if len(recs) == k:
                break

    return recs[:k]


def build_recs_dict(rule_map, user_history, target_users, popular_items, k=6):
    """
    Build user -> list[product_id] recommendations for a given
    set of target_users. Ensures exactly k recommendations per user
    by using rule-based scores plus popular-item fallback.
    """
    recs_dict = {}

    for user in target_users:
        recs = recommend_for_user(
            user_id=user,
            user_history=user_history,
            rule_map=rule_map,
            popular_items=popular_items,
            k=k,
        )
        recs_dict[user] = recs

    logging.info(
        "Built recommendations for %d users (k=%d)", len(recs_dict), k
    )
    return recs_dict


def load_product_names():
    """Return dict[item_id] -> product_name from products.pq."""
    products = pd.read_parquet(PRODUCTS_PATH)
    return dict(zip(products["product_id"], products["product_name"]))


def recommend_with_explanations(user_id, user_history, rule_map, k=6):
    """
    Same recommender as recommend_for_user, but also returns
    a dict[item_id] -> list of textual explanations.
    Intended for debugging / examples, not for pipeline use.
    """
    id_to_name = load_product_names()
    purchased = user_history.get(user_id)
    if not purchased:
        return [], {}

    scores = defaultdict(float)
    reasons = defaultdict(list)

    for a in purchased:
        if a not in rule_map:
            continue
        for b, lift, conf in rule_map[a]:
            if b in purchased:
                continue
            scores[b] += lift
            reasons[b].append((a, lift, conf))

    if not scores:
        return [], {}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    recs = [item for item, _ in ranked]

    explanations = {}
    for item, _ in ranked:
        parts = []
        for a, lift, conf in reasons[item]:
            parts.append(
                f"because you bought '{id_to_name.get(a, str(a))}' "
                f"(lift={lift:.2f}, conf={conf:.2f})"
            )
        explanations[item] = parts

    return recs, explanations


def build_product_to_users(user_history):
    """item -> set(user_id) mapping used for diversity computation."""
    mapping = defaultdict(set)
    for user, items in user_history.items():
        for item in items:
            mapping[item].add(user)
    return mapping


def evaluate_custom_metrics(
    recs_dict,
    user_history,
    future_baskets,
    product_to_users,
    k=6,
    sample_size=200,
    random_state=42,
):
    """
    Compute precision@k, coverage@k, novelty@k and diversity@k
    on a (possibly sampled) set of users.
    """
    rng = np.random.default_rng(random_state)

    users_with_future = set(future_baskets.keys())
    users_with_recs = set(recs_dict.keys())
    candidate_users = sorted(users_with_future & users_with_recs)

    if not candidate_users:
        logging.warning("No overlap between users in recs and future baskets")
        return {
            "precision": 0.0,
            "coverage": 0.0,
            "novelty": 0.0,
            "diversity": 0.0,
            "n_users": 0,
        }

    if sample_size is not None and sample_size < len(candidate_users):
        eval_users = rng.choice(candidate_users, size=sample_size, replace=False)
    else:
        eval_users = candidate_users

    eval_users = list(eval_users)

    precision_sum = 0.0
    users_with_recommendations = 0
    total_recommended = 0
    total_novel = 0
    diversity_per_user = []

    for user in eval_users:
        recs = recs_dict.get(user, [])[:k]
        future = set(future_baskets.get(user, []))
        history = user_history.get(user, set())

        if recs:
            users_with_recommendations += 1
        else:
            continue  # user contributes 0 to all metrics

        recs_set = set(recs)
        hits = len(recs_set & future)
        precision_sum += hits / k

        total_recommended += len(recs)
        total_novel += sum(1 for item in recs if item not in history)

        # Diversity: average 1 - Jaccard over item pairs in recs
        if len(recs) >= 2:
            pair_scores = []
            for i, j in combinations(recs, 2):
                users_i = product_to_users.get(i, set())
                users_j = product_to_users.get(j, set())
                union = users_i | users_j
                if not union:
                    continue
                inter = users_i & users_j
                jacc = len(inter) / len(union)
                pair_scores.append(1.0 - jacc)
            if pair_scores:
                diversity_per_user.append(float(np.mean(pair_scores)))

    n_eval = len(eval_users)
    if n_eval == 0 or users_with_recommendations == 0 or total_recommended == 0:
        return {
            "precision": 0.0,
            "coverage": 0.0,
            "novelty": 0.0,
            "diversity": 0.0,
            "n_users": n_eval,
        }

    precision = precision_sum / users_with_recommendations
    coverage = users_with_recommendations / n_eval
    novelty = total_novel / total_recommended
    diversity = float(np.mean(diversity_per_user)) if diversity_per_user else 0.0

    return {
        "precision": precision,
        "coverage": coverage,
        "novelty": novelty,
        "diversity": diversity,
        "n_users": n_eval,
    }


# Now we can first train the PCY algorithm:

# In[ ]:


# training passes
run_pass1()
run_pass2()
print("PCY Pass 1 & Pass 2 completed and results saved.")


# And then generate and evaluate some recommendations:

# In[ ]:


# 1) Load PCY outputs
logging.info("Loading PCY artifacts")
frequent_pairs, item_counts, num_baskets = load_pcy_artifacts()

# Global popularity (for fallback recommendations)
popular_items = [
    item for item, cnt in sorted(
        item_counts.items(), key=lambda x: x[1], reverse=True
    )
]

# 2) Build rules and rule_map
logging.info("Building rules and rule_map")
rules_df = build_rule_dataframe(frequent_pairs, item_counts, num_baskets)
rule_map = build_rule_map(
    rules_df,
    min_confidence=0.01,
    min_lift=1.0,
    top_n=50,
)

# 3) Build user history (TRAIN)
logging.info("Building user history")
user_history = build_user_history()

# 4) Determine target users (validation users) and build recs_dict
logging.info("Building recs_dict")
test_ratings_path = DATA_PREPROCESSED_DIR / "test_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq"
test_ratings = pd.read_parquet(test_ratings_path)
test_products = construct_test_product_dict(mode="test")  # user -> list[product_id]

target_users = sorted(test_products.keys())
k = 6
recs_dict = build_recs_dict(
        rule_map=rule_map,
        user_history=user_history,
        target_users=target_users,
        popular_items=popular_items,
        k=k,
    )

# 5) Project-level evaluation: hit-rate@6 and ndcg@6 (aligned with other recommenders)
logging.info("Evaluating PCY recommendations")
eval_dict = eval_recs(
    recs_dict=recs_dict,
    rating_df=test_ratings,
    test_products=test_products,
)
avg_hr = np.mean([metrics["hit-rate"] for metrics in eval_dict.values()])
avg_ndcg = np.mean([metrics[f"ndcg@{k}"] for metrics in eval_dict.values()])

print(f"[PCY] Average hit-rate@{k}: {avg_hr:.6f}")
print(f"[PCY] Average ndcg@{k}:   {avg_ndcg:.6f}")

row = ["pcy", k, avg_hr, avg_ndcg]
with open(RESULTS_PATH, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(row)

# 6) Custom evaluation metrics (precision, coverage, novelty, diversity)
future_baskets = {u: set(items) for u, items in test_products.items()}
product_to_users = build_product_to_users(user_history)

custom_metrics = evaluate_custom_metrics(
    recs_dict=recs_dict,
    user_history=user_history,
    future_baskets=future_baskets,
    product_to_users=product_to_users,
    k=k,
    sample_size=200,
    random_state=42,
)

print(
    f"[PCY] Custom metrics on sample of {custom_metrics['n_users']} users "
    f"(k={k}):"
)
print(
    f"precision@{k}: {custom_metrics['precision']:.4f}, "
    f"coverage@{k}: {custom_metrics['coverage']:.4f}, "
    f"novelty@{k}: {custom_metrics['novelty']:.4f}, "
    f"diversity@{k}: {custom_metrics['diversity']:.4f}"
)


# # 9. Results

# Finally we can compare our final models

# In[75]:


overall_results = pd.read_csv(RESULTS_PATH)
print(overall_results.sort_values("hit_rate", ascending=False))

