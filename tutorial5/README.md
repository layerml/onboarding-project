# Tutorial V: How to build a different model by making use of the same sef of features


## What you will learn in this tutorial?

Imagine that you have a set of features you previously extracted from your data for a churn model. 
These are mostly order-related features. One of these features is order review score which is a number 
between 1 and 5. (1:Least satified - 5:Most satisfied)

You think that this order review score is a quite important and useful data point. However, you realized that you have this order review score for only a small portion of orders. 
Thus, you wanted to fit an ML model using the order review score as your target variable to predict the review score
of an order based on its other features.

Since your colleagues have already extracted lots of order features before, you would like to skip 'Feature Engineering' part
and use some of existing features for your model. 

In this tutorial, you will see how easy to make use of these existing features to build a new model on Layer
and shorten project development time drastically.


## Step I: Clone the tutorial repo
To check out the Tutorial V, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project.git
2. cd onboarding-project/tutorial5
```

To build the whole project for the first time:
```commandline
layer start
```


## Step II: Create a new source file for your new model
Go to the /models directory and create a new directory under it.
Name this directory: 'order_review_model'. Under this new directory, there must be 3 separate files:
- **order_review.py**: python source file for your model
- **requirements.txt**: A txt file that lists python packages and their respective versions required for your model
- **order_review.yaml**: A yaml file to define your model on Layer


Create a new file _order_review.py_ under the directory _/models/order_review_model/_ 
```python
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
        order_features_base: Featureset("order_features_trial"),
        order_high_level_features: Featureset("order_high_level_features_trial2")
) -> Any:

    # Step 1: TRAINING DATA GENERATION PROCESS
    # Step 1.1 Fetch order features: Convert the Layer featureset to pandas dataframe
    order_features_base = order_features_base.to_pandas().dropna()
    order_features_base_subset = order_features_base[["ORDER_ID","REVIEW_SCORE","ORDER_STATUS","MAIN_PRODUCT_CATEGORY","MAIN_PAYMENT_TYPE","DAYS_BETWEEN_ESTIMATE_ACTUAL_DELIVERY","AVG_PRODUCT_NAME_LENGTH","AVG_PRODUCT_DESCRIPTION_LENGTH","AVG_PRODUCT_PHOTOS_QTY"]]

    order_high_level_features = order_high_level_features.to_pandas().dropna()

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
```
## Step III: Create a new requirements.txt file for your model
Create a text file _requirements.txt_ under the directory _/models/order_review_model/_  and copy the content below and paste it into this file
```commandline
scikit-learn==1.0
```

## Step IV: Create a new yaml file for your model
Create a yaml file _order_review_ under the directory _/models/order_review_model/_  and copy the content below and paste it into this file
```yaml
# Layer Onboarding Project
#
# Any directory includes an `model.yaml` will be treated as a ml model project.
# In this `yaml` file, we will define the attributes of our model.
# For more information on Model Configuration: https://docs.beta.layer.co/docs/modelcatalog/modelyml

apiVersion: 1

# Name and description of our model
name: "order_review_model"
description: "Order Review Score Prediction Model"
type: model

training:
  name: order_review_training
  description: "Order Review Score Training"

  # The source model definition file with a `train_model` method
  entrypoint: model.py

  # File includes the required python libraries with their correct versions
  environment: requirements.txt

  # Name of the predefined fabric config for model training.
  # Documentation (https://docs.beta.layer.co/docs/reference/fabrics)
  fabric: "f-medium"
```