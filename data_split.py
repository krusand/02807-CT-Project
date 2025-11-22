import pandas as pd

from config import *

def main():
    logging.info("Loading data")

    # load data
    orders_df = pd.read_csv(ORDERS_PATH_CSV)
    op_prior = pd.read_csv(ORDER_PRODUCTS__PRIOR_PATH_CSV)
    op_train = pd.read_csv(ORDER_PRODUCTS__TRAIN_PATH_CSV)

    logging.info("Assigning split labels to orders")

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

    logging.info("Creating parquet file for each split")

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

if __name__ == "__main__":
    main()
