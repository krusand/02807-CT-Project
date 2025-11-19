
from config import *
from utils import *

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
import scipy.sparse as sparse
import pickle as pkl

def main():
    ratings = pd.read_parquet(DATA_PREPROCESSED_DIR / "ratings_w_freq-0.33_w_rec-0.33_w_tfidf-0.33.pq") 
    users = pd.read_parquet(DATA_PREPROCESSED_DIR / "unique_users.pq")["user_id"].to_list()
    products = pd.read_parquet(DATA_PREPROCESSED_DIR / "unique_products.pq")["product_id"].to_list()
    
    logging.info("Starting CF")

    get_cf_scores(features=1000, rating_df = ratings, users=users, items=products)
    logging.info("Ended CF")

if __name__ == "__main__":
    main()

