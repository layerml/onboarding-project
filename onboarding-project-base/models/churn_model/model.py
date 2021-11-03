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
import datetime

def train_model(
        train: Train,
        order_features: Featureset("olist_order_features"),
        customer_features: Featureset("olist_customer_features")
) -> Any:

    # STEP I: TRAINING DATA GENERATION PROCESS
    # 1. Fetch order features: Convert the Layer featureset to pandas dataframe
    order_features = order_features.to_pandas()

    # 2. Label Generation Process
    # 2.1. Fetch customer features: Convert the Layer featureset to pandas dataframe
    customer_features = customer_features.to_pandas()
    # 2.2. Filter the users who did not order again by using a time period of at least 365 days after their first purchases (comparing with the max date in the data --> "2018-10-17")
    order_silence_period = 365
    dataset_max_date = datetime.date(2018, 10, 17)
    customer_not_ordered_again = customer_features[(customer_features.ORDERED_AGAIN == 0) & (customer_features.FIRST_ORDER_TIMESTAMP.dt.date + datetime.timedelta(days=order_silence_period) < dataset_max_date)]
    # 2.3. Use all the users who ordered again
    customer_ordered_again = customer_features.loc[(customer_features.ORDERED_AGAIN == 1)]
    # 2.4. Merge 2 data frames and add a new label column: CHURNED
    # <<Definition of Churn>>: A user who has not ordered again at least in the next 365 days after its first purchase.
    customers_features_filtered = pd.concat([customer_not_ordered_again, customer_ordered_again])
    customers_features_filtered['CHURNED'] = 1 # Create a label column 'churned' with all 1s.
    customers_features_filtered.loc[customers_features_filtered['ORDERED_AGAIN'] == 1, 'CHURNED'] = 0 # Change the label to 0 if the customer ordered again
    customers_labels_df = customers_features_filtered[['CUSTOMER_UNIQUE_ID','FIRST_ORDER_ID','CHURNED']]

    # 3. Final Training Data: Fetch only the first order features of users and the label column
    # Drop irrelevant id columns and NA rows
    training_data_df = customers_labels_df.merge(order_features, left_on='FIRST_ORDER_ID', right_on='ORDER_ID',how='left')\
        .drop(columns=['CUSTOMER_UNIQUE_ID', 'FIRST_ORDER_ID','ORDER_ID']) \
        .dropna()

    #STEP II: MODEL FITTING
    # 1. Define all paramaters
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

    # 2. Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(training_data_df.drop(columns=['CHURNED']),
                                                        training_data_df.CHURNED,
                                                        test_size=test_size_fraction,
                                                        random_state=random_seed)
    # Layer logging model signature
    train.register_input(X_train)
    train.register_output(Y_train)

    # 3. Pipeline Steps
    # Pre-processing: One-hot encoding on a categorical variable: MAIN_PRODUCT_CATEGORY
    categorical_cols = ['MAIN_PRODUCT_CATEGORY']
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],remainder='passthrough')
    # Model: Define a Gradient Boosting Classifier
    model = GradientBoostingClassifier(learning_rate=learning_rate,
                                       max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf,
                                       n_estimators=n_estimators,
                                       subsample=subsample,
                                       random_state=random_state)

    # 4. Pipeline fit
    pipeline = Pipeline(steps=[('t', transformer), ('m', model)])
    pipeline.fit(X_train, Y_train)

    # STEP III: MODEL EVALUATION
    # 1. Predict probabilities of target 1:Churn
    probs = pipeline.predict_proba(X_test)[:,1]
    # 2. Calculate average precision and area under the receiver operating characteric curve (ROC AUC)
    avg_precision = average_precision_score(Y_test, probs, pos_label=1)
    auc = roc_auc_score(Y_test, probs)

    # Layer logging performance metrics
    train.log_metric("Average Precision Score", avg_precision)
    train.log_metric("ROC AUC Score", auc)

    return pipeline




