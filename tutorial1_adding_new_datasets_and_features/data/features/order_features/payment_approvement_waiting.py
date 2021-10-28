from typing import Any
from layer import Dataset
import pandas as pd

def build_feature(orders_layer_df: Dataset("orders_dataset")) -> Any:
    orders_df = orders_layer_df.to_pandas()

    orders_df.ORDER_PURCHASE_TIMESTAMP_TMP = pd.to_datetime(orders_df.ORDER_PURCHASE_TIMESTAMP)
    orders_df.ORDER_APPROVED_AT = pd.to_datetime(orders_df.ORDER_APPROVED_AT)

    orders_df["PAYMENT_APPROVEMENT_WAITING"] = (orders_df.ORDER_APPROVED_AT - orders_df.ORDER_PURCHASE_TIMESTAMP_TMP).dt.days

    orders_general = orders_df[['ORDER_ID', 'PAYMENT_APPROVEMENT_WAITING']]

    return orders_general
