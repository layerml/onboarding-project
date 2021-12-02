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

    # TRAINING DATA GENERATION
    # Convert the Layer featureset to pandas dataframe and select relevant columns (our target variable REVIEW_SCORE is also among these features)
    order_features_base = order_features_base.to_pandas().dropna()
    selected_columns = ["ORDER_ID", "REVIEW_SCORE", "ORDER_STATUS", "MAIN_PRODUCT_CATEGORY", "MAIN_PAYMENT_TYPE",
                        "DAYS_BETWEEN_ESTIMATE_ACTUAL_DELIVERY", "AVG_PRODUCT_NAME_LENGTH",
                        "AVG_PRODUCT_DESCRIPTION_LENGTH", "AVG_PRODUCT_PHOTOS_QTY"]
    order_features_base_subset = order_features_base[selected_columns]
    # Convert the Layer featureset to pandas dataframe
    order_high_level_features = order_high_level_features.to_pandas().dropna()
    # Merge 2 featuresets
    order_features_all = order_features_base_subset.merge(order_high_level_features, left_on='ORDER_ID',right_on='ORDER_ID', how='left')

    # Final Training Data: Drop excluded columns and NA rows from the data
    excluded_cols = ['ORDER_ID']
    training_data_df = order_features_all \
        .drop(columns=excluded_cols) \
        .dropna()

    # MODEL FITTING
    # Define all paramaters
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
    X_train, X_test, Y_train, Y_test = train_test_split(training_data_df.drop(columns=['REVIEW_SCORE']),
                                                        training_data_df.REVIEW_SCORE,
                                                        test_size=test_size_fraction,
                                                        random_state=random_seed)
    # Layer logging model signature
    train.register_input(X_train)
    train.register_output(Y_train)

    # DEFINE PIPELINE STEPS
    # Pre-processing: One-hot encoding on categorical variables
    categorical_cols = ['MAIN_PRODUCT_CATEGORY', 'MAIN_PAYMENT_TYPE', 'ORDER_STATUS']
    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],remainder='passthrough')
    # Model: Define a Gradient Boosting Classifier
    model = GradientBoostingRegressor(learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      max_features=max_features,
                                      min_samples_leaf=min_samples_leaf,
                                      n_estimators=n_estimators,
                                      subsample=subsample,
                                      random_state=random_state)

    # FIT PIPELINE
    pipeline = Pipeline(steps=[('t', transformer), ('m', model)])
    pipeline.fit(X_train, Y_train)

    # MODEL EVALUATION
    # Predict review scores of orders
    yhat_test = pipeline.predict(X_test)
    # Calculate R2 Score
    r2score = r2_score(Y_test, yhat_test)

    # Layer logging performance metrics
    train.log_metric("R2 Score", r2score)

    return pipeline



