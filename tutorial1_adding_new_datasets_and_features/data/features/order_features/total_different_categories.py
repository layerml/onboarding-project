from typing import Any
from layer import Dataset

def to_list_of_set(x):
    return list(set(x))

def build_feature(items_layer_df: Dataset("items_dataset"),products_layer_df: Dataset("products_dataset")) -> Any:
    items_df = items_layer_df.to_pandas()
    products_df = products_layer_df.to_pandas()

    orders_products_df = items_df.merge(products_df, left_on='PRODUCT_ID', right_on='PRODUCT_ID', how='left')

    total_different_categories = orders_products_df.groupby('ORDER_ID').agg(PRODUCTS_CATEGORY_LIST=('PRODUCT_CATEGORY_NAME', to_list_of_set))
    total_different_categories['NO_DIFFERENT_CATEGORIES'] = total_different_categories['PRODUCTS_CATEGORY_LIST'].str.len()
    total_different_categories.drop(columns=["PRODUCTS_CATEGORY_LIST"])

    return total_different_categories
