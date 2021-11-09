from typing import Any
from layer import Dataset

def build_feature(payment_layer_df: Dataset("payments_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    payments_df = payment_layer_df.to_pandas()

    # Compute a new feature: NO_DISTINCT_PAYMENT_TYPES
    number_of_distinct_payment_types = payments_df[["ORDER_ID","PAYMENT_TYPE"]]\
        .groupby(['ORDER_ID'], as_index=False)\
        .agg(NO_DISTINCT_PAYMENT_TYPES=("PAYMENT_TYPE", "nunique"))

    return number_of_distinct_payment_types
