from typing import Any
from layer import Featureset
import numpy as np

def build_feature(order_features_layer: Featureset("order_features_tutorial3")) -> Any:

    order_features_df = order_features_layer.to_pandas()

    df = order_features_df.assign(IS_MULTI_ITEMS=np.where(order_features_df.TOTAL_ITEMS > 1.0, 1, 0))

    is_multi_items = df[["ORDER_ID","IS_MULTI_ITEMS"]]

    return is_multi_items
