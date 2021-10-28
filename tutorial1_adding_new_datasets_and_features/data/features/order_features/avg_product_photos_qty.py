from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset"),products_layer_df: Dataset("products_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    items_df = items_layer_df.to_pandas()
    products_df = products_layer_df.to_pandas()

    # Merge 2 pandas dataframes
    orders_products_df = items_df.merge(products_df, left_on='PRODUCT_ID', right_on='PRODUCT_ID', how='left')

    # Compute a new feature: AVG_PRODUCT_PHOTOS_QTY - Take mean of product photo quantities since there could be multiple products in o single order
    avg_product_photos_qty = orders_products_df\
        .groupby('ORDER_ID', as_index=False)\
        .agg(AVG_PRODUCT_PHOTOS_QTY=("PRODUCT_PHOTOS_QTY", "mean"))

    return avg_product_photos_qty
