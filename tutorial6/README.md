# Tutorial VI: How to use an existing model's outcome as a new input to another model


## What you will learn in this tutorial?

Let's say you are the developer of the Churn model and one of your colleagues developed and released the Order Review Score model recently. 
Youn think that predicted order review score could be a good input data point for your churn model. Therefore, you want to revisit your model
source code and use Order Review Score model's outcome as one of your features in the Churn model. 

In this tutorial, you will learn how easy for you and your colleagues to share and reuse features and models on Layer.


## Step I: Clone the tutorial repo
To check out the Tutorial VI, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project.git
2. cd onboarding-project/tutorial6
```

To build the whole project for the first time:
```commandline
layer start
```


## Step II: Change model source file
- Go to the _/models/order_review_model/_ directory and open the model.py file.


- Copy the code block below and paste it into the source file where you see '#PASTE HERE' comment 


```python
    # 4. Use another model: order_review_score to add a new input (feature): predicted_order_review_scores
    # 4.1 Select the subset of input feature column names for the order_review_score model
    feature_columns_names = ['ORDER_STATUS', 'MAIN_PRODUCT_CATEGORY', 'MAIN_PAYMENT_TYPE',
                             'DAYS_BETWEEN_ESTIMATE_ACTUAL_DELIVERY','AVG_PRODUCT_NAME_LENGTH',
                             'AVG_PRODUCT_DESCRIPTION_LENGTH', 'AVG_PRODUCT_PHOTOS_QTY',
                             'IS_MULTI_ITEMS', 'SHIPPING_PAYMENT_PERC', 'TOTAL_WAITING']

    training_data_for_review_score_model = training_data_df[feature_columns_names]
    # 4.2 Fetch the order_review_score model
    order_review_model = layer.get_model("order_review_model").trained_model_object
    # 4.3 Make inferences by using the model
    predicted_order_review_scores = order_review_model.predict(training_data_for_review_score_model)
    # 4.4 Append the predicted column back to the training dataframe
    training_data_df['PREDICTED_ORDER_REVIEW_SCORE']=predicted_order_review_scores
```
## Step III: Run the layer command on CLI to build new version of the Churn model
Create a text file _requirements.txt_ under the directory _/models/order_review_model/_  and copy the content below and paste it into this file
```commandline
layer model start churn_model_trial
```

## Step IV: Model performance comparisons on Layer Web UI

Go to the model page on the Layer Web UI to compare performance of the latest version with the previous versions to see if we improve
the model performance metrics by adding a new feature with the help of another ML model.
