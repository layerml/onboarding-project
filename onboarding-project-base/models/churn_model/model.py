"""
This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. In order to build a model, every ML project
should have a model file like this one which implements train_model function.
"""
from typing import Any
from layer import Featureset, Train
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



def train_model(
        train: Train,
        order_features: Featureset("olist_order_features"),
        customer_features: Featureset("olist_customer_features")
) -> Any:

    # Fetch features
    features_df = order_features.to_pandas()

    # Label Generation Process
    customer_churn_targets_df = customer_features.to_pandas()
    # Find users who have not ordered anything within the last 365 days. Use only those users as churn users. (Target Variable Column: RETENTION)
    order_silence_period = 365
    dataset_max_date = customer_churn_targets_df.FIRST_ORDER_TIMESTAMP.dt.date.max()
    df_with_labels_0 = customer_churn_targets_df.loc[(customer_churn_targets_df.RETENTION == 0) & ((dataset_max_date - customer_churn_targets_df.FIRST_ORDER_TIMESTAMP.dt.date).dt.days > order_silence_period)]
    # Use all non-churn users who have ordered more than once
    df_with_labels_1 = customer_churn_targets_df.loc[(customer_churn_targets_df.RETENTION == 1)]
    # Final data frame with labels
    labels_df = pd.concat([df_with_labels_0, df_with_labels_1])

    # Model Data Generation: Merge features with labels & dropping irrelevant id columns and NA rows
    model_data_df = labels_df.merge(features_df, left_on='FIRST_ORDER_ID', right_on='ORDER_ID',how='left')\
        .drop(columns=['ORDER_ID', 'FIRST_ORDER_TIMESTAMP','CUSTOMER_UNIQUE_ID', 'FIRST_ORDER_ID'])\
        .dropna()

    # Model Data Split: train & test data
    test_size_fraction = 0.33
    random_seed = 42
    X_train, X_test, Y_train, Y_test = train_test_split(model_data_df.drop(columns=['RETENTION']),
                                                        model_data_df.RETENTION,
                                                        test_size=test_size_fraction,
                                                        random_state=random_seed)
    # Data split parameters & model signature logging
    train.log_parameter("test_size", test_size_fraction)
    train.log_parameter("train_test_split_seed", random_seed)
    train.register_input(X_train)
    train.register_output(Y_train)

    # Model Fitting
    # Step I: Pre-processing: One-hot encoding on a categorical variable: MAIN_PRODUCT_CATEGORY
    categorical_cols = ['MAIN_PRODUCT_CATEGORY']
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],remainder='passthrough')
    # Step II: Fitting a Gradient Boosting Classifier
    learning_rate = 0.01
    max_depth = 6
    max_features = 'sqrt'
    min_samples_leaf = 10
    n_estimators = 100
    subsample = 0.8

    model = GradientBoostingClassifier(learning_rate=learning_rate,
                                       max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf,
                                       n_estimators=n_estimators,
                                       subsample=subsample)

    #Model parameters logging
    train.log_parameter("learning_rate", learning_rate)
    train.log_parameter("max_depth", max_depth)
    train.log_parameter("max_features", max_features)
    train.log_parameter("min_samples_leaf", min_samples_leaf)
    train.log_parameter("n_estimators", n_estimators)
    train.log_parameter("subsample", 0.8)


    # Pipeline definition
    pipeline = Pipeline(steps=[('t', transformer), ('m', model)])
    pipeline.fit(X_train, Y_train)

    # Model evaluation
    probs = pipeline.predict_proba(X_test)
    # Keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # Calculate precisions, recalls and fscores for different thresholds
    precisions, recalls, thresholds = precision_recall_curve(Y_test, probs)
    f1_scores = 2 * recalls * precisions / (recalls + precisions)
    # Find the threshold for the best f-score and precision-recall-f1score values at that particular threshold
    threshold_for_best_f1 = thresholds[np.argmax(f1_scores)]
    best_f1_score = np.max(f1_scores)
    precision_for_the_threshold = precisions[np.argmax(f1_scores)]
    recall_for_the_threshold = recalls[np.argmax(f1_scores)]

    # Model performance metrics logging
    train.log_metric("Threshold for the Best F1 Score", threshold_for_best_f1)
    train.log_metric("Best F1 Score", best_f1_score)
    train.log_metric("Precision for the same threshold", precision_for_the_threshold)
    train.log_metric("Recall for the same threshold", recall_for_the_threshold)

    return pipeline




