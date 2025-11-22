import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
import scipy.sparse as sparse
from pandas.api.types import CategoricalDtype
import pickle as pkl


from config import *


def calculate_user_product_frequency(merged_df: pd.DataFrame) -> None: 
    logging.info("")
    bui = (
        merged_df.groupby(['user_id', 'product_id'])['order_id']
        .nunique()
        .reset_index(name='Bui')
    )

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

    sample_user_id = merged_df['user_id'].iloc[0]
    sample = (merged_df
              .loc[merged_df['user_id'] == sample_user_id
                   , ['user_id', 'order_number', 'days_since_prior_order', 'cum_days', 'total_days', 'age_days', 'weight']
                   ])
    
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
    logging.info("")
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

    users = ratings_long["user_id"].unique()
    products = ratings_long["product_id"].unique()
    shape = (len(users), len(products))

    # Create indices for users and movies
    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    product_cat = CategoricalDtype(categories=sorted(products), ordered=True)
    user_index = ratings_long["user_id"].astype(user_cat).cat.codes
    product_index = ratings_long["product_id"].astype(product_cat).cat.codes

    # Conversion via COO matrix
    coo = sparse.coo_matrix((ratings_long["ranke_ui"], (user_index, product_index)), shape=shape)
    ratings_matrix = coo.tocsr()

    with open(DATA_PREPROCESSED_DIR / "ratings_csr_matrix.pkl", 'wb') as fp:
        pkl.dump(ratings_matrix, file=fp)

def save_unique_users() -> None:
    logging.info("")
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

    users = ratings_long["user"].drop_duplicates().to_frame().reset_index()

    file_path = DATA_PREPROCESSED_DIR / "unique_users.pq"
    users.to_parquet(file_path, index=False)
    logging.info(f"Saved ratings to {file_path}")

def save_unique_products() -> None:
    logging.info("")
    ratings_long = pd.read_parquet(DATA_PREPROCESSED_DIR / "train_ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq")

    products = ratings_long["item"].drop_duplicates().to_frame().reset_index()

    file_path = DATA_PREPROCESSED_DIR / "unique_products.pq"
    products.to_parquet(file_path, index=False)
    logging.info(f"Saved ratings to {file_path}")

def main():
    path_dict = {"train": ORDER_PRODUCTS__TRAIN_PATH,
                 "val": ORDER_PRODUCTS__VAL_PATH,
                 "test": ORDER_PRODUCTS__TEST_PATH,
                 }
    
    orders = pd.read_parquet(ORDERS_PATH)

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

if __name__ == "__main__":
    main()


