from typing import Any
from layer import Dataset
import pandas as pd

def build_feature(reviews_layer_df: Dataset("reviews_dataset")) -> Any:
    reviews_df = reviews_layer_df.to_pandas()

    reviews_df['REVIEW_ANSWER_TIMESTAMP'] = pd.to_datetime(reviews_df.REVIEW_ANSWER_TIMESTAMP)

    # There are multiple reviews for a single order - Only take into account the latest review record
    reviews_df['LATEST_REVIEW_TIMES'] = reviews_df.groupby(['ORDER_ID'])['REVIEW_ANSWER_TIMESTAMP'].transform('max')

    ### Fetch only latest review
    reviews_df = reviews_df[reviews_df['LATEST_REVIEW_TIMES'] == reviews_df['REVIEW_ANSWER_TIMESTAMP']]

    review_score = reviews_df[['ORDER_ID', 'REVIEW_SCORE']]

    return review_score
