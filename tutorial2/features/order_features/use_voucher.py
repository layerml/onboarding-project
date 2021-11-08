from typing import Any
from layer import Dataset
import numpy as np

def build_feature(payment_layer_df: Dataset("payments_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    payments_df = payment_layer_df.to_pandas()

    # Compute a new feature: TOTAL_PAYMENT & return id and relevant columns
    use_voucher = payments_df[["ORDER_ID","PAYMENT_TYPE"]]\
        .assign(IS_VOUCHER=np.where(payments_df['PAYMENT_TYPE'] == 'voucher', 1, 0))\
        .groupby(['ORDER_ID'], as_index=False)\
        .agg(USE_VOUCHER=("IS_VOUCHER", "max"))

    return use_voucher
