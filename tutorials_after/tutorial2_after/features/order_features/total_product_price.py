from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    items_df = items_layer_df.to_pandas()

    # Compute a new feature: TOTAL_PRODUCT_PRICE and return id and relevant feature column
    total_product_price = items_df.groupby('ORDER_ID', as_index=False).agg(TOTAL_PRODUCT_PRICE=("PRICE", "sum"))

    return total_product_price
