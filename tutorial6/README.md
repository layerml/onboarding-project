# Tutorial VI: How to use an existing model's outcome as a new input to another model


## What you will learn in this tutorial?

Let's say you are the developer of the Churn model and one of your colleagues developed and released the Order Review Score model recently. 
Youn think that predicted order review score could be a good input data point for your churn model. Therefore, you want to revisit your model
source code and use Order Review Score model's outcome as one of your features in the Churn model. 

In this tutorial, you will learn how easy for you and your colleagues to share and reuse features and models on Layer.


## Step I: Clone the tutorial repo
To check out the Tutorial VI, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project-and-tutorials.git
2. cd onboarding-project-and-tutorials/tutorial6
```

To build existing Layer entities in this tutorial for the first time, run:
```commandline
layer start
```


## Step II: Change model source file
- Go to the file: _/models/order_review_model/order_review_model.py_ in the project.


- Copy the code block below and paste it into the source file where you see '# << PASTE HERE >> #' comment 


```python
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
```
## Step III: Run the layer command on CLI to build new version of the Churn model
To rebuild updated model, run:
```commandline
layer model start churn_model_tutorial6
```

## Step IV: Model performance comparisons on Layer Web UI

Go to the model page on the Layer Web UI to compare performance of the latest version with the previous versions to see if we improve
the model performance metrics by adding a new feature with the help of another ML model.

Congratulations, you have just completed the tutorial. Now, you know how to use an existing model's outcome as a new input to another model on Layer.

To check if you are done correct, go and check the Tutorial 6's after project:
```commandline
cd onboarding-project-and-tutorials/tutorials_after/tutorial6_after
```