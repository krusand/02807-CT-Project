from collections import defaultdict
from typing import Iterable

from lenskit.algorithms.als import BiasedMF
import numpy as np
import pandas as pd

def get_cf_scores(features: int, 
                  rating_df: pd.DataFrame,
                  users: Iterable,
                  items: Iterable,
                  iterations=20,
                  reg=0.1,
                  damping=5,
                  bias=True,
                  seed=51225)-> dict:
    """
    Function to train a BiasedMF model from lenskit and generate predicted scores for each user and item. 
    Lenskit documentation for the BiasedMF model: https://lenskit.org/0.14.4/mf#lenskit.algorithms.svd.BiasedSVD

    Parameters:
    - features:         The number of latent features in the user and item vectors learned by the model.
    - rating_df:        The training data containing ratings for the items rated by each user.
    - users:            An iterable containing the ID of all users.
    - items:            An iterable containing the ID of all items. 
    - iterations:       The number of training iterations (default: 20).
    - reg:              Regularization factors, can also be a tuple (ureg, ireg) to specify separate user and item regularization terms (default: 0.1).
    - damping:          Damping factor for the underlying bias (default: 5). 
    - bias:             Whether to include a bias term in the prediction rule or not (default: True). 
    - seed:             Seed for reproducibility purposes (default: 51225).

    Returns:
    - scores_dict:      Dictionary containing the predicted score for each pair of user and item.
    """

    # initializing dictionary containing predicted scores for each user and item
    scores_dict = {}

    # initializing the BiasedMF model
    mf = BiasedMF(features=features, 
                  iterations=iterations, 
                  reg=reg, 
                  damping=damping, 
                  bias=bias,
                  rng_spec=seed)
    
    # fitting the BiasedMF model
    mf.fit(rating_df)

    # generating scores for each user and saving to the scores_dict
    for user in users:
        scores = list(mf.predict_for_user(user, items))
        scores_dict[user] = scores

    return scores_dict


def get_recs(scores_dict, n_recs, d_hondts=True) -> dict:
    """
    Generates a list of length n_recs of recommended items, aisles, or clusters for each user.

    Parameters:
    - scores_dict:  Predicted scores output from the cf algorithm.
    - n_recs:       Number of recommended items, aisles, or clusters. 
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