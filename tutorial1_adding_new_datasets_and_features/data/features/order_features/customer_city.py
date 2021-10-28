from typing import Any
from layer import Dataset

def build_feature(orders_layer_df: Dataset("orders_dataset"), customers_layer_df: Dataset("customers_dataset")) -> Any:
    orders_df = orders_layer_df.to_pandas()
    customers_df = customers_layer_df.to_pandas()

    orders_customers_df = orders_df.merge(customers_df, left_on='CUSTOMER_ID', right_on='CUSTOMER_ID', how='left')

    customer_city = orders_customers_df[['ORDER_ID', 'CUSTOMER_CITY']]

    return customer_city
