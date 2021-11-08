from typing import Any
from layer import Dataset

def build_feature(payment_layer_df: Dataset("payments_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    payments_df = payment_layer_df.to_pandas()

    # Compute a new feature: TOTAL_PAYMENT
    total_payment = payments_df[["ORDER_ID","PAYMENT_VALUE"]]\
        .groupby(['ORDER_ID'], as_index=False)\
        .agg(TOTAL_PAYMENT=("PAYMENT_VALUE", "sum"))

    return total_payment
