from typing import Any
from layer import Dataset
import pandas as pd


def build_feature(orders_layer_df: Dataset("orders_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    orders_df = orders_layer_df.to_pandas()

    # Compute a new feature: DELIVERED_CARRIER_WAITING
    orders_df["DELIVERED_CARRIER_WAITING"] = (orders_df.ORDER_DELIVERED_CARRIER_DATE - orders_df.ORDER_APPROVED_AT).dt.days

    # Select only the id column and the feature column to be returned
    delivered_carrier_waiting = orders_df[['ORDER_ID', 'DELIVERED_CARRIER_WAITING']]

    return delivered_carrier_waiting
