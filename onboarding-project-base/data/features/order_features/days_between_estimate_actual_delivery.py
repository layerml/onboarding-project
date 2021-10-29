"""
Develop your feature in a Python file by implementing the build_feature function
Ensure that you have the same ID column across all of your features in the featureset.
Layer joins your singular features using that ID column.
"""
from typing import Any
from layer import Dataset

def build_feature(
        orders_layer_df: Dataset("orders_dataset")
) -> Any:

    # Convert Layer Dataset into pandas data frame
    orders_df = orders_layer_df.to_pandas()

    # Compute a new feature: DAYS_BETWEEN_ESTIMATE_ACTUAL_DELIVERY
    orders_df["DAYS_BETWEEN_ESTIMATE_ACTUAL_DELIVERY"] = (orders_df.ORDER_ESTIMATED_DELIVERY_DATE - orders_df.ORDER_DELIVERED_CUSTOMER_DATE).dt.days

    # Select only the columns to be returned
    days_between_estimate_actual_delivery = orders_df[['ORDER_ID', 'DAYS_BETWEEN_ESTIMATE_ACTUAL_DELIVERY']]

    return days_between_estimate_actual_delivery
