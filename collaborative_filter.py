
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
    users = pd.read_parquet(DATA_PREPROCESSED_DIR / "unique_users.pq")["user"].to_list()
    products = pd.read_parquet(DATA_PREPROCESSED_DIR / "unique_products.pq")["item"].to_list()
    
    logging.info("Starting CF fit")

    scores_dict = get_cf_scores(features=1000, rating_df = ratings, users=users, items=products, iterations=2)
    logging.info("Ended CF fit")

    logging.info("Starting recs")
    recs_dict = get_recs(scores_dict=scores_dict, n_recs=6)
    logging.info("Ending recs")
    with open(DATA_PREPROCESSED_DIR / "recs.pkl", 'wb') as fp:
        pkl.dump(recs_dict,file=fp)

    logging.info("Saved recs")

if __name__ == "__main__":
    main()

