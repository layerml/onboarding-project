from typing import Any
from layer import Featureset

def build_feature(order_features_layer: Featureset("order_features_tutorial5")) -> Any:
    order_features_df = order_features_layer.to_pandas()

    df = order_features_df.assign(TOTAL_WAITING=order_features_df.PAYMENT_APPROVEMENT_WAITING + order_features_df.DELIVERED_CARRIER_WAITING)

    total_waiting = df[["ORDER_ID","TOTAL_WAITING"]]

    return total_waiting
