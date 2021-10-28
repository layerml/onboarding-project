from typing import Any
from layer import Dataset
import numpy as np

def build_feature(payment_layer_df: Dataset("payments_dataset")) -> Any:
    payments_df = payment_layer_df.to_pandas()

    # Add a column to sum payment values by payment types
    payments_df['PAYMENT_TYPE_TOTAL'] = payments_df.groupby(['ORDER_ID', 'PAYMENT_TYPE'])['PAYMENT_VALUE'].transform('sum')

    # Add a column to show maximum amount payment by a payment type
    payments_df['PAYMENT_TYPE_TOTAL_MAX'] = payments_df.groupby(['ORDER_ID'])['PAYMENT_TYPE_TOTAL'].transform('max')

    # Add a column to indicate if it is the main payment_type for this order or not (NAN)
    payments_df['MAIN_PAYMENT_TYPE'] = np.where(payments_df['PAYMENT_TYPE_TOTAL_MAX'] == payments_df['PAYMENT_TYPE_TOTAL'], payments_df.PAYMENT_TYPE, np.NaN)

    payments_df.MAIN_PAYMENT_TYPE = payments_df.MAIN_PAYMENT_TYPE.astype(str)

    # Cleaning on Payments Dataset: Aggregate payments on order id
    main_payment_type = payments_df[["ORDER_ID","MAIN_PAYMENT_TYPE"]]\
        .groupby(['ORDER_ID'], as_index=False)\
        .agg(MAIN_PAYMENT_TYPE=("MAIN_PAYMENT_TYPE", "first"))

    return main_payment_type
