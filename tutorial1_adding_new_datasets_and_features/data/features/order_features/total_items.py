from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset")) -> Any:
    items_df = items_layer_df.to_pandas()

    total_items = items_df.groupby('ORDER_ID', as_index=False).agg(TOTAL_ITEMS=("PRODUCT_ID", "count"))

    return total_items
