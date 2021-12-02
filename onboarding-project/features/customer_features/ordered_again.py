"""
Develop your feature in a Python file by implementing the build_feature function
Ensure that you have the same ID column across all of your features in the featureset.
Layer joins your singular features using that ID column.
"""
from typing import Any
from layer import Dataset
import numpy as np

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

    # # Add a new column total_orders: For each user, compute total number of orders
    users_df["TOTAL_ORDERS"] = users_df.groupby('CUSTOMER_UNIQUE_ID')['CUSTOMER_ID'].transform('count')

    # Add a new binary column ordered_again: If a customer orders again after its first purchase
    users_df['ORDERED_AGAIN'] = np.where(users_df['TOTAL_ORDERS'] > 1, 1, 0)

    # Select only the columns to be returned
    ordered_again = users_df[['CUSTOMER_UNIQUE_ID', 'ORDERED_AGAIN','TOTAL_ORDERS']]

    return ordered_again
