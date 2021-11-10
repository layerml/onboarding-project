# Tutorial III: How to build different versions of the same model

## Install and run
To check out the Tutorial II, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project.git
2. cd onboarding-project/tutorial3
```

To build the project:
```commandline
layer start
```

To build a new version of the same model after making any change in the model source code:
```commandline
layer start model churn_model
```

## What you will learn in this tutorial?

### The Case
Imagine that you have a baseline model previously trained. 
That's the model you deployed on your production before and running at the moment. 
After a while, you wanted to revisit this model and try to make some improvement on its performance. 


Here is the list of possible ways you have in mind improving your model:
- Make use of another featureset as well:
  - You have a new featureset now: `order_features_high_level`, since the last time you trained this model.
  - Let's try a new set of features which mixes the features from the `order_features` featureset as well as the `order_high_level_features` featureset.

- Make some changes in the model parameters:
  - You have been using the Gradient Boosting Classifier from Sklearn for this problem. 
  You are wondering what if a change in the model parameters will make a positive impact on the model performance.
  - You would like to try a different value for the `max_depth` parameter of the model.

Total of 4 combinations you will try on our model. 
In other words, we will have 4 different versions of the model to compare their performances.
- Combination I: Only use the `order_features` featureset & `max_depth=6`
- Combination II: Only use the `order_features` featureset & `max_depth=10`
- Combination III: Use a mixed set of features from both `order_features` and `order_features_high_level` featureset & `max_depth=6`
- Combination IV: Use a mixed set of features from both `order_features` and `order_features_high_level` featureset & `max_depth=10`


