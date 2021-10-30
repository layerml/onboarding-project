"""
This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. In order to build a model, every ML project
should have a model file like this one which implements train_model function.
"""
from typing import Any
from layer import Featureset, Train
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_model(
        train: Train,
        order_features: Featureset("olist_order_features"),
        customer_features: Featureset("olist_customer_features")
) -> Any:

    # Fetch order features: Convert the Layer featureset to pandas dataframe
    features_df = order_features.to_pandas()

    # Label Generation Process
    # 1. Fetch customer features: Convert the Layer featureset to pandas dataframe (Target Variable Column: RETENTION)
    customer_churn_targets_df = customer_features.to_pandas()
    # 2. Find users who have not ordered anything within the last 365 days. Use only those users as churn users.
    order_silence_period = 365
    dataset_max_date = customer_churn_targets_df.FIRST_ORDER_TIMESTAMP.dt.date.max()
    df_with_labels_0 = customer_churn_targets_df.loc[(customer_churn_targets_df.RETENTION == 0) & ((dataset_max_date - customer_churn_targets_df.FIRST_ORDER_TIMESTAMP.dt.date).dt.days > order_silence_period)]
    # 3. Use all non-churn users who have ordered more than once
    df_with_labels_1 = customer_churn_targets_df.loc[(customer_churn_targets_df.RETENTION == 1)]
    # 4. Merge 2 data frames to create the final data frame with both labels
    labels_df = pd.concat([df_with_labels_0, df_with_labels_1])

    # Training Data Generation: Fetch only first order features of users
    # Dropping irrelevant id columns and NA rows
    training_data_df = labels_df.merge(features_df, left_on='FIRST_ORDER_ID', right_on='ORDER_ID',how='left')\
        .drop(columns=['ORDER_ID', 'FIRST_ORDER_TIMESTAMP','CUSTOMER_UNIQUE_ID', 'FIRST_ORDER_ID'])\
        .dropna()

    # Define all parameters
    # 1.Parameters for train-test split
    test_size_fraction = 0.33
    random_seed = 42
    # 2. Model Parameters
    learning_rate = 0.01
    max_depth = 6
    max_features = 'sqrt'
    min_samples_leaf = 10
    n_estimators = 100
    subsample = 0.8


    X_train, X_test, Y_train, Y_test = train_test_split(training_data_df.drop(columns=['RETENTION']),
                                                        training_data_df.RETENTION,
                                                        test_size=test_size_fraction,
                                                        random_state=random_seed)
    # Layer model signature logging
    train.register_input(X_train)
    train.register_output(Y_train)

    # Pipeline Steps
    # Step I: Pre-processing: One-hot encoding on a categorical variable: MAIN_PRODUCT_CATEGORY
    categorical_cols = ['MAIN_PRODUCT_CATEGORY']
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],remainder='passthrough')
    # Step II: Define a Gradient Boosting Classifier
    model = GradientBoostingClassifier(learning_rate=learning_rate,
                                       max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf,
                                       n_estimators=n_estimators,
                                       subsample=subsample)

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

    # Pipeline definition and fit
    pipeline = Pipeline(steps=[('t', transformer), ('m', model)])
    pipeline.fit(X_train, Y_train)

    # Model evaluation
    # 1. Predict probabilities of target 1:Non-churn
    probs = pipeline.predict_proba(X_test)[:,1]
    # 2. Calculate average precision and area under the receiver operating characteric curve (ROC AUC)
    avg_precision = average_precision_score(Y_test, probs, pos_label=1)
    auc = roc_auc_score(Y_test, probs)

    # Layer logging performance metrics
    train.log_metric("Average Precision Score", avg_precision)
    train.log_metric("ROC AUC Score", auc)

    return pipeline




