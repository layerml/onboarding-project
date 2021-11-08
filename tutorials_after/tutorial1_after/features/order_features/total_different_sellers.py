from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    items_df = items_layer_df.to_pandas()

    # Compute a new feature: TOTAL_DIFFERENT_SELLERS and return only id and relevant feature columns
    total_different_sellers = items_df.groupby('ORDER_ID', as_index=False).agg(TOTAL_DIFFERENT_SELLERS=("SELLER_ID", "nunique"))

    return total_different_sellers
