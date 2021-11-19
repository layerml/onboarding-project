from typing import Any
from layer import Featureset

def build_feature(order_items_features_layer: Featureset("order_items_features")) -> Any:
    order_items_features_df = order_items_features_layer.to_pandas()

    df = order_items_features_df.assign(SHIPPING_PAYMENT_PERC=(order_items_features_df.TOTAL_FREIGHT_PRICE / (order_items_features_df.TOTAL_PRODUCT_PRICE + order_items_features_df.TOTAL_FREIGHT_PRICE)) * 100)

    shipping_payment_perc = df[["ORDER_ID","SHIPPING_PAYMENT_PERC"]]

    return shipping_payment_perc
