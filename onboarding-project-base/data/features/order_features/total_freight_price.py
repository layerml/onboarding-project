"""
Develop your feature in a Python file by implementing the build_feature function
Ensure that you have the same ID column across all of your features in the featureset.
Layer joins your singular features using that ID column.
"""
from typing import Any
from layer import Dataset

def build_feature(
        items_layer_df: Dataset("items_dataset")
) -> Any:

    # Convert Layer Dataset into pandas data frame
    items_df = items_layer_df.to_pandas()

    # Compute a new feature: FREIGHT_VALUE
    total_freight_price = items_df.groupby('ORDER_ID', as_index=False).agg(TOTAL_FREIGHT_PRICE=("FREIGHT_VALUE", "sum"))

    return total_freight_price
