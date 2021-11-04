from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    items_df = items_layer_df.to_pandas()

    # Compute a new feature: TOTAL_ITEMS and return ORDER_ID and the relevant feature
    total_items = items_df.groupby('ORDER_ID', as_index=False).agg(TOTAL_ITEMS=("PRODUCT_ID", "count"))

    return total_items
