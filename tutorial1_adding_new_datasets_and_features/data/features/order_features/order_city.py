from typing import Any
from layer import Dataset

def build_feature(orders_layer_df: Dataset("orders_dataset"), customers_layer_df: Dataset("customers_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    orders_df = orders_layer_df.to_pandas()
    customers_df = customers_layer_df.to_pandas()

    # Merge 2 pandas dataframes
    orders_customers_df = orders_df.merge(customers_df, left_on='CUSTOMER_ID', right_on='CUSTOMER_ID', how='left')

    # ORDER_CITY: Select the city of the customer the order is placed by.
    order_city = orders_customers_df.rename(columns={"CUSTOMER_CITY": "ORDER_CITY"})[['ORDER_ID', 'ORDER_CITY']]

    return order_city
