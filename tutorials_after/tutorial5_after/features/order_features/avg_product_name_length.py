from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset"),products_layer_df: Dataset("products_dataset")) -> Any:
    # Convert Layer Dataset into pandas dataframe
    items_df = items_layer_df.to_pandas()
    products_df = products_layer_df.to_pandas()

    # Join 2 dataframes
    orders_products_df = items_df.merge(products_df, left_on='PRODUCT_ID', right_on='PRODUCT_ID', how='left')

    # Compute a new feature: PRODUCT_NAME_LENGHT and return only ORDER ID and PRODUCT_NAME_LENGHT
    avg_product_name_length = orders_products_df\
        .groupby('ORDER_ID', as_index=False)\
        .agg(AVG_PRODUCT_NAME_LENGTH=("PRODUCT_NAME_LENGHT", "mean"))

    return avg_product_name_length
