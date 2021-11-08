from typing import Any
from layer import Featureset

def build_feature(order_general_features_layer: Featureset("order_general_features")) -> Any:
    order_general_features_df = order_general_features_layer.to_pandas()

    df = order_general_features_df.assign(TOTAL_WAITING=order_general_features_df.PAYMENT_APPROVEMENT_WAITING.astype(int) + order_general_features_df.DELIVERED_CARRIER_WAITING.astype(int))

    total_waiting = df[["ORDER_ID","TOTAL_WAITING"]]

    return total_waiting
