from typing import Any
from layer import Dataset
import pandas as pd


def build_feature(orders_layer_df: Dataset("orders_dataset")) -> Any:
    orders_df = orders_layer_df.to_pandas()

    orders_df.ORDER_APPROVED_AT = pd.to_datetime(orders_df.ORDER_APPROVED_AT)
    orders_df.ORDER_DELIVERED_CARRIER_DATE = pd.to_datetime(orders_df.ORDER_DELIVERED_CARRIER_DATE)

    orders_df["DELIVERED_CARRIER_WAITING"] = (orders_df.ORDER_DELIVERED_CARRIER_DATE - orders_df.ORDER_APPROVED_AT).dt.days

    delivered_carrier_waiting = orders_df[['ORDER_ID', 'DELIVERED_CARRIER_WAITING']]

    return delivered_carrier_waiting
