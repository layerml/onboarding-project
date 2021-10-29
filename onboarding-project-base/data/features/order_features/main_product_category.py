"""
Develop your feature in a Python file by implementing the build_feature function
Ensure that you have the same ID column across all of your features in the featureset.
Layer joins your singular features using that ID column.
"""
from typing import Any
from layer import Dataset
import numpy as np

def build_feature(
        items_layer_df: Dataset("items_dataset"),
        products_layer_df: Dataset("products_dataset"),
        category_name_translation_layer_df: Dataset("category_name_translation_dataset")
) -> Any:

    # Convert Layer Dataset into pandas data frame
    items_df = items_layer_df.to_pandas()
    products_df = products_layer_df.to_pandas()
    category_translation_df = category_name_translation_layer_df.to_pandas()

    # Join items and products pandas dataframes : items_products_joined
    items_products_joined = items_df.merge(products_df, left_on='PRODUCT_ID', right_on='PRODUCT_ID', how='left')

    # Join items_products_joined with category_translation pandas dataframe (translate category names from portuguese to english)
    all_joined_df = items_products_joined.merge(category_translation_df, left_on='PRODUCT_CATEGORY_NAME', right_on='PRODUCT_CATEGORY_NAME' ,how='left')

    # MAIN PRODUCT CATEGORY: Since an order might have different products from different categories, the category with the highest total payment will be the main category of the order.

    # CATEGORY_TOTAL_PAYMENT: Total payment for each category. (In case of having multiple categories in an order)
    all_joined_df['CATEGORY_TOTAL_PRICE'] = all_joined_df.groupby(['ORDER_ID', 'PRODUCT_CATEGORY_NAME_ENGLISH'])['PRICE'].transform('sum')

    # CATEGORY_TOTAL_MAX: Total maximum price among categories of the order
    all_joined_df['CATEGORY_TOTAL_MAX'] = all_joined_df.groupby(['ORDER_ID'])['CATEGORY_TOTAL_PRICE'].transform('max')

    # Pick the category with highest total price as the main category of the order ('first' in aggregation returns first non-null value)
    all_joined_df['MAIN_PRODUCT_CATEGORY'] = np.where(all_joined_df['CATEGORY_TOTAL_PRICE'] == all_joined_df['CATEGORY_TOTAL_MAX'],all_joined_df.PRODUCT_CATEGORY_NAME_ENGLISH, np.NaN)
    main_product_category = all_joined_df\
        .groupby('ORDER_ID', as_index=False)\
        .agg(MAIN_PRODUCT_CATEGORY=("MAIN_PRODUCT_CATEGORY", "first"))

    # Main Product Category is a nominal variable not an ordinal variable. Therefore, it is better to convert this column using OneHotEncoding in the model stage.
    # There are many categories in the dataset. It's not good practice to encode a nominal variable with too many levels into one-hot version.
    # Article:[https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769]
    # Therefore, we used the top 10 categories here and call the rest of the categories as "other"
    top_10_categories = ["bed_bath_table", "sports_leisure", "health_beauty","computers_accessories","furniture_decor","housewares","watches_gifts","telephony","auto","toys"]
    main_product_category['MAIN_PRODUCT_CATEGORY'] = main_product_category['MAIN_PRODUCT_CATEGORY'].apply(lambda category: category if category in top_10_categories else 'other')

    return main_product_category
