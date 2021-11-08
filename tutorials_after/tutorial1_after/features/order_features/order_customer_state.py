from typing import Any
from layer import Dataset

def build_feature(orders_layer_df: Dataset("orders_dataset"), customers_layer_df: Dataset("customers_dataset")) -> Any:
    # Convert Layer Dataset into pandas dataframe
    orders_df = orders_layer_df.to_pandas()
    customers_df = customers_layer_df.to_pandas()

    # Join 2 dataframes
    orders_customers_df = orders_df.merge(customers_df, left_on='CUSTOMER_ID', right_on='CUSTOMER_ID', how='left')

    # Customer State is a nominal variable not an ordinal variable. Therefore, it is better to convert this column using OneHotEncoding in the model stage.
    # However since there are many states in this column, it's not good practice to encode a nominal variable with too many levels into one-hot version. Article:[https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769]
    # Therefore, after checking on the distribution of people over the states, we decided to use only the top 10 states and call the rest of the states as "other"
    top_10_states = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA","DF", "ES","GO"]
    orders_customers_df['ORDER_CUSTOMER_STATE'] = orders_customers_df['CUSTOMER_STATE'].apply(
        lambda category: category if category in top_10_states else 'other')

    order_customer_state = orders_customers_df[['ORDER_ID', 'ORDER_CUSTOMER_STATE']]

    return order_customer_state
