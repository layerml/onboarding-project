# Develop your feature in a Python file by implementing the build_feature function
# Ensure that you have the same ID column across all of your features in the featureset.
# Layer joins your singular features using that ID column.

from typing import Any
from layer import Dataset

def build_feature(
        orders_dataset_layer: Dataset("orders_dataset"),
        customers_dataset_layer: Dataset("customers_dataset")
) -> Any:

    # Convert Layer datasets into pandas data frames
    orders_df = orders_dataset_layer.to_pandas()
    customers_df = customers_dataset_layer.to_pandas()

    # We will only take "delivered" orders into consideration during our analysis
    orders_df = orders_df[orders_df.ORDER_STATUS == "delivered"]

    # Join 2 pandas data frames
    users_df = orders_df.merge(customers_df, left_on='CUSTOMER_ID', right_on='CUSTOMER_ID', how='left')

    # Add a new column order_rank: For each user, rank the orders with respect to their purchase time
    users_df["ORDER_RANK"] = users_df.groupby('CUSTOMER_UNIQUE_ID')['ORDER_PURCHASE_TIMESTAMP'].rank(method='first')

    # Filter out only the first orders of users in the dataset
    users_df = users_df[users_df["ORDER_RANK"] == 1.0].drop(columns=['ORDER_RANK'])

    # Rename columns and select only the columns to be returned
    first_order_timestamp = users_df.rename(columns={"ORDER_PURCHASE_TIMESTAMP": "FIRST_ORDER_TIMESTAMP"})[['CUSTOMER_UNIQUE_ID', 'FIRST_ORDER_TIMESTAMP']]

    return first_order_timestamp
