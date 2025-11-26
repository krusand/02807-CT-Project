from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram


from umap import UMAP

import pandas as pd
import numpy as np

from config import *
from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from tqdm import tqdm

from functools import reduce

from utils import HourOfDayEncoder, DOWEncoder

orders = pd.read_parquet(ORDERS_PATH)
orders_products = pd.read_parquet(ORDER_PRODUCTS__TRAIN_PATH)
orders_orders_products = orders.merge(orders_products, how="inner", on="order_id")
products = pd.read_parquet(PRODUCTS_PATH)
departments = pd.read_parquet(DEPARTMENTS_PATH)
aisle = pd.read_parquet(AISLES_PATH)

order_all_info = (
    orders_orders_products.merge(products, how="inner", on="product_id")
    .merge(departments, how="inner", on="department_id")
    .merge(aisle, how="inner", on="aisle_id")
)


# Number of orders a user has made: _are they a shopper who orders a lot, or a shopper with fewer orders?_

n_orders = orders.groupby(["user_id"]).size().reset_index(name="n_orders_u")

# Avg number of products in baskets: _do they shop small baskets or large baskets? -> family or single_
products_pr_order = (
    orders_orders_products.groupby(["user_id", "order_id"])["product_id"]
    .size()
    .reset_index(name="n_products_pr_order")
    .groupby(["user_id"])["n_products_pr_order"]
    .mean()
    .reset_index(name="avg_products_pr_order")
)

# User frequency: _how often does the user shop?_

order_frequency = (
    orders.groupby(["user_id"])["days_since_prior_order"]
    .mean()
    .reset_index(name="avg_days_since_prior_order")
)

# Reorder frequency: Same product types?
reorders = (
    orders_orders_products.groupby(["user_id"])
    .agg(n_products=("reordered", "size"), n_reorders=("reordered", "sum"))
    .reset_index()
    .assign(frac_reorders=lambda x: x["n_reorders"] / x["n_products"])
)

# Avg hour of day
hour_enc = HourOfDayEncoder(return_pd=True)
avg_order_of_hour = (
    hour_enc.fit_transform(orders)
    .groupby(["user_id"])
    .agg(avg_x_hour=("X_hour_dir", "mean"), avg_y_hour=("Y_hour_dir", "mean"))
    .reset_index()
)


# Avg dow

dow_enc = DOWEncoder(return_pd=True)
avg_order_dow = (
    dow_enc.fit_transform(X=orders)
    .groupby(["user_id"])
    .agg(avg_x_dow=("X_day_dir", "mean"), avg_y_dow=("Y_day_dir", "mean"))
    .reset_index()
)


# Fraction in departments:

products_pr_user = (
    orders_orders_products.groupby(["user_id"])["product_id"]
    .size()
    .reset_index(name="n_products_pr_user")
)


products_pr_user_pr_department = (
    order_all_info.groupby(["user_id", "department_id"])
    .size()
    .reset_index(name="n_products")
)

frac_department = (
    products_pr_user_pr_department.merge(products_pr_user, on="user_id", how="inner")
    .assign(
        frac_products_in_department=lambda x: x["n_products"] / x["n_products_pr_user"]
    )[["user_id", "department_id", "frac_products_in_department"]]
    .pivot(
        index="user_id", columns="department_id", values="frac_products_in_department"
    )
    .fillna(0)
)

frac_department.columns = [f"dep_id_{i}" for i in frac_department.columns]
frac_department = frac_department.reset_index()

feature_dfs = [
    n_orders,
    products_pr_order,
    order_frequency,
    reorders,
    avg_order_dow,
    avg_order_of_hour,
    frac_department,
]


def get_feature_columns(feature_dfs: list[pd.DataFrame]) -> list:
    feature_cols = []
    for df in feature_dfs:
        df_column_set = set(df.columns.tolist())
        df_column_set.remove("user_id")
        for col in list(df_column_set):
            feature_cols.append(col)

    return feature_cols


feature_columns = get_feature_columns(feature_dfs)
# merge all feature dfs, and select all columns which are not 'user_id' in those dfs
X = reduce(lambda x, y: pd.merge(x, y, on="user_id"), feature_dfs)[feature_columns]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"Total features: {len(feature_columns)}")

metric = "euclidean"
linkage = "ward"
agg_clust = AgglomerativeClustering(n_clusters=3, linkage=linkage, metric=metric)

agg_clust.fit_predict(X_scaled)
