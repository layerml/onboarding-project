from typing import Any
from layer import Dataset

def build_feature(orders_layer_df: Dataset("orders_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    orders_df = orders_layer_df.to_pandas()

    # Select only the id column and the feature column to be returned
    status = orders_df[['ORDER_ID', 'ORDER_STATUS']]

    return status
