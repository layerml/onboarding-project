from typing import Any
from layer import Dataset

def build_feature(orders_layer_df: Dataset("orders_dataset")) -> Any:
    orders_df = orders_layer_df.to_pandas()

    status = orders_df[['ORDER_ID', 'ORDER_STATUS']]

    return status
