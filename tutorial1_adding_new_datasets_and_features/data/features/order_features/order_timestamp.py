from typing import Any
from layer import Dataset

def build_feature(orders_layer_df: Dataset("orders_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    orders_df = orders_layer_df.to_pandas()

    # Select only the id column and the feature column to be returned
    order_timestamp = orders_df[['ORDER_ID', 'ORDER_PURCHASE_TIMESTAMP']]

    return order_timestamp
