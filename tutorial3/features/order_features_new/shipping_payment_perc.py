from typing import Any
from layer import Featureset

def build_feature(order_features_layer: Featureset("order_features_tutorial3")) -> Any:

    order_features_df = order_features_layer.to_pandas()

    df = order_features_df\
        .assign(SHIPPING_PAYMENT_PERC=round((order_features_df.TOTAL_FREIGHT_PRICE / (order_features_df.TOTAL_PRODUCT_PRICE + order_features_df.TOTAL_FREIGHT_PRICE)) * 100,2))

    shipping_payment_perc = df[["ORDER_ID","SHIPPING_PAYMENT_PERC"]]

    return shipping_payment_perc