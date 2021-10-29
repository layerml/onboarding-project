from typing import Any
from layer import Dataset

def to_list_of_set(x):
    return list(set(x))

def build_feature(items_layer_df: Dataset("items_dataset"),sellers_layer_df: Dataset("sellers_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    items_df = items_layer_df.to_pandas()
    sellers_df = sellers_layer_df.to_pandas()

    # Merge 2 pandas dataframes
    orders_sellers_df = items_df.merge(sellers_df, left_on='SELLER_ID', right_on='SELLER_ID', how='left')

    # State list of all different sellers in a single order
    seller_state_list = orders_sellers_df.groupby('ORDER_ID').agg(SELLERS_STATE_LIST=('SELLER_STATE', to_list_of_set))

    return seller_state_list
