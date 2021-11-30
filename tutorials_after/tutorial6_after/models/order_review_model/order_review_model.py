"""
This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. In order to build a model, every ML project
should have a model file like this one which implements train_model function.
"""
from typing import Any
from layer import Featureset, Train
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

def train_model(
        train: Train,
        order_features_base: Featureset("order_features_tutorial6"),
        order_high_level_features: Featureset("order_features_tutorial6_new")
) -> Any:

    # Step 1: TRAINING DATA GENERATION PROCESS
    # Step 1.1 Fetch order features: Convert the Layer featureset to pandas dataframe
    order_features_base = order_features_base.to_pandas().dropna()
    order_features_base_subset = order_features_base[["ORDER_ID","REVIEW_SCORE","ORDER_STATUS","MAIN_PRODUCT_CATEGORY","MAIN_PAYMENT_TYPE","DAYS_BETWEEN_ESTIMATE_ACTUAL_DELIVERY","AVG_PRODUCT_NAME_LENGTH","AVG_PRODUCT_DESCRIPTION_LENGTH","AVG_PRODUCT_PHOTOS_QTY"]]
    print("HELLO1 ", order_features_base_subset.columns)
    order_high_level_features = order_high_level_features.to_pandas().dropna()
    print("HELLO2 ", order_high_level_features.columns)
    order_features_all = order_features_base_subset.merge(order_high_level_features, left_on='ORDER_ID', right_on='ORDER_ID', how='left')


    # Step 1.2 Label Generation Process
    # 1.2.1. Fetch customer features: Convert the Layer featureset to pandas dataframe

    # 3. Final Training Data: Fetch only the first order features of users and drop excluded and na columns from the final dataframe
    excluded_cols = ['ORDER_ID']
    training_data_df = order_features_all\
        .drop(columns=excluded_cols) \
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
    X_train, X_test, Y_train, Y_test = train_test_split(training_data_df.drop(columns=['REVIEW_SCORE']),
                                                        training_data_df.REVIEW_SCORE,
                                                        test_size=test_size_fraction,
                                                        random_state=random_seed)
    # Layer logging model signature
    train.register_input(X_train)
    train.register_output(Y_train)

    # 3. Pipeline Steps
    # Pre-processing: One-hot encoding on a categorical variable: MAIN_PRODUCT_CATEGORY
    categorical_cols = ['MAIN_PRODUCT_CATEGORY','MAIN_PAYMENT_TYPE','ORDER_STATUS']
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],remainder='passthrough')
    # Model: Define a Gradient Boosting Classifier
    model = GradientBoostingRegressor(learning_rate=learning_rate,
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
    yhat_test = pipeline.predict(X_test)
    # 2. Calculate average precision and area under the receiver operating characteric curve (ROC AUC)
    r2score = r2_score(Y_test, yhat_test)

    # Layer logging performance metrics
    train.log_metric("R2 Score", r2score)

    return pipeline




