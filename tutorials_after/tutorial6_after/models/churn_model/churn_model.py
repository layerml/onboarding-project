"""
This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. In order to build a model, every ML project
should have a model file like this one which implements train_model function.
"""
from typing import Any

import numpy as np
from layer import Featureset, Train, Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime
import layer

def train_model(
        train: Train,
        order_features_base: Featureset("order_features_tutorial6"),
        order_high_level_features: Featureset("order_features_tutorial6_new"),
        customer_features: Featureset("customer_features_tutorial6"),
) -> Any:

    # FEATURES GENERATION
    # Convert the Layer featuresets to pandas dataframes
    order_features_base = order_features_base.to_pandas().dropna()
    order_high_level_features = order_high_level_features.to_pandas().dropna()
    # Merge 2 featuresets
    order_features = order_features_base.merge(order_high_level_features, left_on='ORDER_ID', right_on='ORDER_ID',how='left')

    # LABEL GENERATION: <<Definition of Churn>>: A user who has not ordered again at least in the next 365 days after its first purchase.
    # Convert the Layer featureset to pandas dataframe
    customer_features = customer_features.to_pandas()
    # Filter the users who did not order again in next 365 days after their first purchases ("2018-10-17" is the latest date in the data)
    order_silence_period = 365
    dataset_max_date = datetime.date(2018, 10, 17)
    customer_not_ordered_again = customer_features[(customer_features.ORDERED_AGAIN == 0) & (customer_features.FIRST_ORDER_TIMESTAMP.dt.date + datetime.timedelta(days=order_silence_period) < dataset_max_date)]
    # Use all the users who ordered again
    customer_ordered_again = customer_features.loc[(customer_features.ORDERED_AGAIN == 1)]
    # Concat 2 data frames and add a new label column: CHURNED
    customers_features_filtered = pd.concat([customer_not_ordered_again, customer_ordered_again])
    customers_features_filtered['CHURNED'] = 1  # Create a label column 'churned' with all 1s.
    customers_features_filtered.loc[customers_features_filtered['ORDERED_AGAIN'] == 1, 'CHURNED'] = 0  # Change the label to 0 if the customer has ordered again
    customers_labels_df = customers_features_filtered[['CUSTOMER_UNIQUE_ID', 'FIRST_ORDER_ID', 'CHURNED']]

    # FINAL TRAINING DATA GENERATION
    # Fetch only the first order features of users and drop excluded and na columns from the final dataframe
    excluded_cols = ['CUSTOMER_UNIQUE_ID', 'FIRST_ORDER_ID', 'ORDER_ID', 'ORDER_PURCHASE_TIMESTAMP']
    training_data_df = customers_labels_df.merge(order_features, left_on='FIRST_ORDER_ID', right_on='ORDER_ID', how='left') \
        .drop(columns=excluded_cols) \
        .dropna()

    # USE ANOTHER MODEL: order_review_score to add a new feature: predicted_order_review_scores
    # Select the subset of input feature column names for the order_review_score model
    feature_columns_names = ['ORDER_STATUS', 'MAIN_PRODUCT_CATEGORY', 'MAIN_PAYMENT_TYPE',
                             'DAYS_BETWEEN_ESTIMATE_ACTUAL_DELIVERY','AVG_PRODUCT_NAME_LENGTH',
                             'AVG_PRODUCT_DESCRIPTION_LENGTH', 'AVG_PRODUCT_PHOTOS_QTY',
                             'IS_MULTI_ITEMS', 'SHIPPING_PAYMENT_PERC', 'TOTAL_WAITING']

    training_data_for_review_score_model = training_data_df[feature_columns_names]
    # Fetch the order_review_score model
    order_review_model = layer.get_model("order_review_model").trained_model_object
    # Make inferences by using the model
    predicted_order_review_scores = order_review_model.predict(training_data_for_review_score_model)
    # Append the predicted column back to the training dataframe
    training_data_df['PREDICTED_ORDER_REVIEW_SCORE']=predicted_order_review_scores

    # MODEL FITTING
    # Define all paramaters
    # Parameters for data split
    test_size_fraction = 0.33
    random_seed = 42
    # Model Parameters
    learning_rate = 0.01
    max_depth = 6
    max_features = 'sqrt'
    min_samples_leaf = 10
    n_estimators = 100
    subsample = 0.8
    random_state = 42

    # Layer logging all parameters
    train.log_parameters({"test_size": test_size_fraction,
                          "train_test_split_seed": random_seed,
                          "learning_rate": learning_rate,
                          "max_depth": max_depth,
                          "max_features": max_features,
                          "min_samples_leaf": min_samples_leaf,
                          "n_estimators": n_estimators,
                          "subsample": subsample
                          })

    # Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(training_data_df.drop(columns=['CHURNED']),
                                                        training_data_df.CHURNED,
                                                        test_size=test_size_fraction,
                                                        random_state=random_seed)
    # Layer logging model signature
    train.register_input(X_train)
    train.register_output(Y_train)

    # DEFINE PIPELINE STEPS
    # Pre-processing: One-hot encoding on a categorical variable: MAIN_PRODUCT_CATEGORY
    categorical_cols = ['MAIN_PRODUCT_CATEGORY', 'MAIN_PAYMENT_TYPE', 'ORDER_CUSTOMER_CITY', 'ORDER_CUSTOMER_STATE','ORDER_STATUS']
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
                                    remainder='passthrough')
    # Model: Define a Gradient Boosting Classifier
    model = GradientBoostingClassifier(learning_rate=learning_rate,
                                       max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf,
                                       n_estimators=n_estimators,
                                       subsample=subsample,
                                       random_state=random_state)

    # FIT THE PIPELINE
    pipeline = Pipeline(steps=[('t', transformer), ('m', model)])
    pipeline.fit(X_train, Y_train)

    # MODEL EVALUATION
    # Predict probabilities of target 1:Churn
    probs = pipeline.predict_proba(X_test)[:, 1]
    # Calculate average precision and area under the receiver operating characteric curve (ROC AUC)
    avg_precision = average_precision_score(Y_test, probs, pos_label=1)
    auc = roc_auc_score(Y_test, probs)

    # Layer logging performance metrics
    train.log_metric("Average Precision Score", avg_precision)
    train.log_metric("ROC AUC Score", auc)

    return pipeline



