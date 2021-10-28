from typing import Any
from layer import Dataset
import pandas as pd


def build_feature(reviews_layer_df: Dataset("reviews_dataset")) -> Any:
    reviews_df = reviews_layer_df.to_pandas()

    # Add new column time between review creation and answer
    reviews_df['REVIEW_CREATION_DATE'] = pd.to_datetime(reviews_df.REVIEW_CREATION_DATE)
    reviews_df['REVIEW_ANSWER_TIMESTAMP'] = pd.to_datetime(reviews_df.REVIEW_ANSWER_TIMESTAMP)
    reviews_df["REVIEW_RESPONSE_DAYS"] = (reviews_df.REVIEW_ANSWER_TIMESTAMP - reviews_df.REVIEW_CREATION_DATE).dt.days

    # There are multiple reviews for a single order - Only take into account the latest review record
    reviews_df['LATEST_REVIEW_TIMES'] = reviews_df.groupby(['ORDER_ID'])['REVIEW_ANSWER_TIMESTAMP'].transform('max')

    ### Fetch only latest review
    reviews_df = reviews_df[reviews_df['LATEST_REVIEW_TIMES'] == reviews_df['REVIEW_ANSWER_TIMESTAMP']]

    review_response_days = reviews_df[['ORDER_ID', 'REVIEW_RESPONSE_DAYS']]

    return review_response_days
