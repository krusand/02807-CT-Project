from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

from config import *
from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
dbscan = DBSCAN()

orders = pd.read_parquet(ORDERS_PATH)

df = orders.groupby(["user_id"]).size().reset_index(name="n_orders")
X = df["n_orders"]


user_product_frequency = pd.read_parquet(DATA_PREPROCESSED_DIR / "user_product_frequency.pq")
user_product_frequency_grouped = user_product_frequency.groupby(["user_id"])["freq_ui"].mean().reset_index(name='avg_freq_u')
user_product_recency_min_max_scaled = pd.read_parquet(DATA_PREPROCESSED_DIR / "user_product_recency_min_max_scaled.pq")
upr_grouped = user_product_recency_min_max_scaled.groupby(["user_id"])["recency_score_min_maxed"].mean().reset_index(name='avg_score')

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(upr_grouped["avg_score"], X, user_product_frequency_grouped["avg_freq_u"])
plt.show()
