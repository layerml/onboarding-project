from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset"),products_layer_df: Dataset("products_dataset")) -> Any:
    items_df = items_layer_df.to_pandas()
    products_df = products_layer_df.to_pandas()

    orders_products_df = items_df.merge(products_df, left_on='PRODUCT_ID', right_on='PRODUCT_ID', how='left')

    avg_product_description_length = orders_products_df\
        .groupby('ORDER_ID', as_index=False)\
        .agg(AVG_PRODUCT_DESCRIPTION_LENGTH=("PRODUCT_DESCRIPTION_LENGHT", "mean"))

    return avg_product_description_length
