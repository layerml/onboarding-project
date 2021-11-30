# Tutorial III: How to build your first model using Layer Entities


## What you will learn in this tutorial?

Imagine that you have done enough on feature engineering part 
and ready to train your first baseline model using your Layer featuresets.

In this tutorial, you will learn:
- How to fetch featuresets in the `train_model` function signature: `order_features_tutorial3` & `customer_features_tutorial3`
- How to log model parameters on Layer using `train.log_parameters`
- How to log model metrics on Layer using `train.log_metric`

## Step I: Clone the tutorial repo
To check out the Tutorial III, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project-and-tutorials.git
2. cd onboarding-project-and-tutorials/tutorial3
```

To build the featuresets, run:
```commandline
layer start
```

## Step II: Create files in the project file structure

**Create model source file: churn_model.py**

Copy the code block below and paste it into the churn_model.py file
```python
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
        order_features_base: Featureset("order_features_tutorial3"),
        customer_features: Featureset("customer_features_tutorial3")
) -> Any:

    # Step 1: TRAINING DATA GENERATION PROCESS
    # Step 1.1 Fetch order features: Convert the Layer featureset to pandas dataframe
    order_features_base = order_features_base.to_pandas().dropna()

    # Step 1.2 Label Generation Process
    # 1.2.1. Fetch customer features: Convert the Layer featureset to pandas dataframe
    customer_features = customer_features.to_pandas().dropna()
    # 1.2.2. Filter the users who did not order again by using a time period of at least 365 days after their first purchases (comparing with the max date in the data --> "2018-10-17")
    order_silence_period = 365
    dataset_max_date = datetime.date(2018, 10, 17)
    customer_not_ordered_again = customer_features[(customer_features.ORDERED_AGAIN == 0) & (customer_features.FIRST_ORDER_TIMESTAMP.dt.date + datetime.timedelta(days=order_silence_period) < dataset_max_date)]
    # 1.2.3. Use all the users who ordered again
    customer_ordered_again = customer_features.loc[(customer_features.ORDERED_AGAIN == 1)]
    # 1.2.4. Merge 2 data frames and add a new label column: CHURNED
    # <<Definition of Churn>>: A user who has not ordered again at least in the next 365 days after its first purchase.
    customers_features_filtered = pd.concat([customer_not_ordered_again, customer_ordered_again])
    customers_features_filtered['CHURNED'] = 1 # Create a label column 'churned' with all 1s.
    customers_features_filtered.loc[customers_features_filtered['ORDERED_AGAIN'] == 1, 'CHURNED'] = 0 # Change the label to 0 if the customer has ordered again
    customers_labels_df = customers_features_filtered[['CUSTOMER_UNIQUE_ID','FIRST_ORDER_ID','CHURNED']]

    # 3. Final Training Data: Fetch only the first order features of users and drop excluded and na columns from the final dataframe
    excluded_cols = ['CUSTOMER_UNIQUE_ID', 'FIRST_ORDER_ID','ORDER_ID','ORDER_PURCHASE_TIMESTAMP','ORDER_STATUS']
    training_data_df = customers_labels_df.merge(order_features_base, left_on='FIRST_ORDER_ID', right_on='ORDER_ID',how='left')\
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
    X_train, X_test, Y_train, Y_test = train_test_split(training_data_df.drop(columns=['CHURNED']),
                                                        training_data_df.CHURNED,
                                                        test_size=test_size_fraction,
                                                        random_state=random_seed)
    # Layer logging model signature
    train.register_input(X_train)
    train.register_output(Y_train)

    # 3. Pipeline Steps
    # Pre-processing: One-hot encoding on a categorical variable: MAIN_PRODUCT_CATEGORY
    categorical_cols = ['MAIN_PRODUCT_CATEGORY','MAIN_PAYMENT_TYPE','ORDER_CUSTOMER_CITY','ORDER_CUSTOMER_STATE']
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
```

**Create requirements.txt**

Copy the block below and paste it into the requirements.txt file

```text
scikit-learn==1.0
```

**Create model's yaml file**

Copy the block below and paste it into the churn_model.yaml file
```yaml
# Layer Onboarding Project
#
# Any directory includes an `model.yaml` will be treated as a ml model project.
# In this `yaml` file, we will define the attributes of our model.
# For more information on Model Configuration: https://docs.beta.layer.co/docs/modelcatalog/modelyml

apiVersion: 1
type: model

# Name and description of our model
name: "churn_model_tutorial3"
description: "Churn Prediction Model"

training:
  name: churn_model_training_trial
  description: "Churn Prediction Model Training"

  # The source model definition file with a `train_model` method
  entrypoint: model.py

  # File includes the required python libraries with their correct versions
  environment: requirements.txt

  # Name of the predefined fabric config for model training.
  # Documentation (https://docs.beta.layer.co/docs/reference/fabrics)
  fabric: "f-medium"
```

### Step III: Build your baseline model for the first time
```commandline
layer start model churn_model_tutorial3
```

Congratulations, you have just completed the tutorial. Now, you know how to build your first baseline ML model using existing Featuresets you generated before. 

To check if you are done correct, go and check the Tutorial 3's after project:
```commandline
cd onboarding-project-and-tutorials/tutorials_after/tutorial3_after
```




