from typing import Any
from layer import Dataset
import numpy as np

def build_feature(payment_layer_df: Dataset("payments_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    payments_df = payment_layer_df.to_pandas()

    # Cleaning on Payments Dataset: Aggregate payments on order id
    number_of_total_installments = payments_df[["ORDER_ID","PAYMENT_TYPE"]]\
        .assign(NON_VOUCHER_INSTALLMENTS=np.where(payments_df['PAYMENT_TYPE'] == 'voucher', 0, 1))\
        .groupby(['ORDER_ID'], as_index=False)\
        .agg(TOTAL_NON_VOUCHER_INSTALLMENTS=("NON_VOUCHER_INSTALLMENTS", "sum"))

    # Change number of installments to 1 for payments done by only vouchers because each payment must have at least 1 installment
    number_of_total_installments["TOTAL_NON_VOUCHER_INSTALLMENTS"] = np.where(
        number_of_total_installments['TOTAL_NON_VOUCHER_INSTALLMENTS'] == 0, 1,
        number_of_total_installments['TOTAL_NON_VOUCHER_INSTALLMENTS'])

    return number_of_total_installments
