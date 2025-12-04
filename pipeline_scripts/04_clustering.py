import sys
import os

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import *
from utils import *


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