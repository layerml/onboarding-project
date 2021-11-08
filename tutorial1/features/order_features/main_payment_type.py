# The 'payments_df' dataset has number of rows associated to an order as many as the number of payment types the order has.
# MAIN_PAYMENT_TYPE: The payment type of the order which has the biggest share of the total payment. (If an order has 1 payment type, then that's the MAIN_PAYMENT_TYPE.)

from typing import Any
from layer import Dataset
import numpy as np

def build_feature(payment_layer_df: Dataset("payments_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    payments_df = payment_layer_df.to_pandas()

    # Add a column 'PAYMENT_TYPE_TOTAL' to sum payment values by payment types
    payments_df['PAYMENT_TYPE_TOTAL'] = payments_df.groupby(['ORDER_ID', 'PAYMENT_TYPE'])['PAYMENT_VALUE'].transform('sum')

    # Add a column 'PAYMENT_TYPE_TOTAL_MAX' to show maximum amount payment by a payment type
    payments_df['PAYMENT_TYPE_TOTAL_MAX'] = payments_df.groupby(['ORDER_ID'])['PAYMENT_TYPE_TOTAL'].transform('max')

    # Add a column to indicate if it is the main payment_type for this order or not (NAN)
    payments_df['MAIN_PAYMENT_TYPE'] = np.where(payments_df['PAYMENT_TYPE_TOTAL_MAX'] == payments_df['PAYMENT_TYPE_TOTAL'], payments_df.PAYMENT_TYPE, np.NaN)

    # Pick the payment type with highest total price as the main payment type of the order ('first' in aggregation returns first non-null value)
    main_payment_type = payments_df\
        .groupby(['ORDER_ID'], as_index=False)\
        .agg(MAIN_PAYMENT_TYPE=("MAIN_PAYMENT_TYPE", "first"))

    return main_payment_type
