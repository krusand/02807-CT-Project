from typing import Iterable

from lenskit.algorithms.als import BiasedMF
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