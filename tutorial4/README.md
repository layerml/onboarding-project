# Tutorial IV: How to build different versions of the same model


## What you will learn in this tutorial?

Imagine that you have a baseline model previously trained. 
That's the model you deployed on your production before and running at the moment. 
After a while, you wanted to revisit this model and try to make some improvement on its performance. 


Here is the list of possible ways you have in mind improving your model:
- **Parameter change**: Make a change in the model parameters without any changes on the features
  - You have been using the Gradient Boosting Classifier from Sklearn for this problem. 
  You are wondering what if a change in the model parameters will make a positive impact on the model performance.
  - You would like to try a different value for the `max_depth` parameter of the model.
  

- **Data change**: Make use of another featureset without any changes on the model parameters
  - You have a new featureset now: `order_features_high_level`, since the last time you trained this model.
  - Let's try a new set of features which mixes the features from the `order_features` as well as the `order_high_level_features` featuresets.



Total of 3 combinations you will try on your model. 
In other words, you will have 3 different versions of the model to compare their performances.
- _Version 1.1 (Baseline)_: Only use the `order_features_tutorial4` featureset & `max_depth=6`
- _Version 2.1 (Change in model parameter)_: Change the default value of model parameter: `max_depth=10`
- _Version 3.1 (Change in model features)_: Use a mixed set of features from both `order_features_tutorial4` and `order_features_tutorial4_new` featureset. [Keep the paramater `max_depth=10` the same]


## Step I: Clone the tutorial repo
To check out the Tutorial IV, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project-and-tutorials.git
2. cd onboarding-project-and-tutorials/tutorial4
```

## Step II: Train the model version 1.1 [Baseline]
To build the whole project for the first time:
```commandline
layer start
```

To see the model version 1.1 on Web UI:
> Click on the model link on the CLI

## Step III: Train the model version 2.1 [Change in model parameters]
Go and find the parameter `max_depth = 6` in the model.py source file and change its value:
```commandline
max_depth = 10
```

To re-build the model after the parameter change: 
_[No need to build the whole project all over again]_
```commandline
layer start model churn_model
```

To see the model version 2.1 on Web UI:
> Click on the model link on the CLI 

## Step IV: Train the model version 3.1 [Change in model features]
To fetch the new featureset, change function signature of the `train_model` in the **/models/churn_model/model.py** file
```python
def train_model(
        train: Train,
        order_features_base: Featureset("order_features_trial"),
        order_high_level_features: Featureset("order_high_level_features_trial2"),
        customer_features: Featureset("customer_features_trial")
) -> Any:
```

To merge 2 different order featuresets, change the  _'Step 1.1'_ in the **/models/churn_model/model.py** file as shown below
```python
...
    # STEP I: TRAINING DATA GENERATION PROCESS
    # 1. Fetch order features: Convert the Layer featureset to pandas dataframe
    order_features_base = order_features_base.to_pandas()
    order_high_level_features = order_high_level_features.to_pandas()
    order_features = order_features_base.merge(order_high_level_features, left_on='ORDER_ID', right_on='ORDER_ID', how='left')
...
```


To re-build the model after the parameter change: 
_[No need to build the whole project all over again]_
```commandline
layer start model churn_model
```

To see the model version 3.1 on Web UI:
> Click on the model link on the CLI 

Congratulations, you have just completed the tutorial. Now, you know how to build different versions of the same ML model and compare them on Layer.

To check if you are done correct, go and check the Tutorial 3's after project:
```commandline
cd onboarding-project-and-tutorials/tutorials_after/tutorial4_after
```