from typing import Any
from layer import Dataset
import pandas as pd


def build_feature(orders_layer_df: Dataset("orders_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    orders_df = orders_layer_df.to_pandas()

    # Compute a new feature: DELIVERED_CARRIER_WAITING -> Number of days between order approvement date and the date carrier picked up the order to deliver
    orders_df["DELIVERED_CARRIER_WAITING"] = (orders_df.ORDER_DELIVERED_CARRIER_DATE - orders_df.ORDER_APPROVED_AT).dt.days

    # Select only the columns to be returned
    delivered_carrier_waiting = orders_df[['ORDER_ID', 'DELIVERED_CARRIER_WAITING']]

    return delivered_carrier_waiting
