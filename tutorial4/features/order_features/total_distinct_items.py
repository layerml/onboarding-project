from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    items_df = items_layer_df.to_pandas()

    # Compute a new feature: TOTAL_DISTINCT_ITEMS and return only id and relevant feature columns
    total_distinct_items = items_df.groupby('ORDER_ID', as_index=False).agg(TOTAL_DISTINCT_ITEMS=("PRODUCT_ID", "nunique"))

    return total_distinct_items
