from typing import Any
from layer import Dataset
import pandas as pd

def build_feature(orders_layer_df: Dataset("orders_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    orders_df = orders_layer_df.to_pandas()

    # Compute a new feature: PAYMENT_APPROVEMENT_WAITING
    orders_df["PAYMENT_APPROVEMENT_WAITING"] = (orders_df.ORDER_APPROVED_AT - orders_df.ORDER_PURCHASE_TIMESTAMP).dt.days

    # Select only the id column and the feature column to be returned
    payment_approvement_waiting = orders_df[['ORDER_ID', 'PAYMENT_APPROVEMENT_WAITING']]

    return payment_approvement_waiting
