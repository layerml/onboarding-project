# Tutorial I: How to Add New Source Datasets and Features into an Layer Project

## Step I: Add a New Dataset

> In this step, you will learn how to define one of your external source tables as a Layer Dataset.

> There is a new table named ***olist_reviews*** for you that resides on our public Snowflake database.
In order to define a new Layer Dataset for this table, first create a new directory in the project 
> directory under the **/tutorial1/data** and name it ***reviews_dataset***. 
>
> In the new directory 'reviews_dataset', create a **dataset.yaml** file and copy the block below and paste it into this yaml file.

```yaml
# For more information on Dataset Configuration: https://docs.beta.layer.co/docs/datacatalog/datasets

apiVersion: 1

# Unique name of this dataset which will be used in this project to refer to this dataset
name: "reviews_dataset"
type: source
description: "This dataset includes data about the order reviews."

materialization:
    target: layer-public-datasets
    table_name: "olist_reviews"
```

> That's it, you are done. Congratulations! You just defined a source table on your database as a Layer Dataset.

## Step II: Add New Features

> In this step, you will learn how to create 2 new Layer Features and add them into the existing Layer Featureset: ***order_features***

> We will add 2 new features into the project:
> - Review Score
> - Total Items
> 
> For each feature, we will first create its respective python source file and then define it into the featureset along with its description.

> Create ***review_score.py*** and ***total_items.py*** python source files and add them in the project directory under the **/tutorial1/features/order_features**. 
> 
>Copy and paste the code blocks below into the respective files.

***review_score.py***
```python
from typing import Any
from layer import Dataset

def build_feature(reviews_layer_df: Dataset("reviews_dataset")) -> Any:
    # Convert Layer Datasets into pandas dataframes
    reviews_df = reviews_layer_df.to_pandas()

    # For some reason, there might be multiple reviews for a single order in the data - Only take into account the latest review record
    # Compute a new feature: LATEST_REVIEW_TIMES
    reviews_df['LATEST_REVIEW_TIMES'] = reviews_df.groupby(['ORDER_ID'])['REVIEW_ANSWER_TIMESTAMP'].transform('max')

    # Fetch only latest review
    reviews_df = reviews_df[reviews_df['LATEST_REVIEW_TIMES'] == reviews_df['REVIEW_ANSWER_TIMESTAMP']]

    # Return only ORDER_ID and REVIEW_SCORE columns
    review_score = reviews_df[['ORDER_ID', 'REVIEW_SCORE']]

    return review_score
```
***total_items.py***
```python
from typing import Any
from layer import Dataset

def build_feature(items_layer_df: Dataset("items_dataset")) -> Any:
    # Convert Layer Dataset into pandas data frame
    items_df = items_layer_df.to_pandas()

    # Compute a new feature: TOTAL_ITEMS and return ORDER_ID and the relevant feature
    total_items = items_df.groupby('ORDER_ID', as_index=False).agg(TOTAL_ITEMS=("PRODUCT_ID", "count"))

    return total_items
```

> Now, we will add the feature definitions below into the featureset yaml file: **/tutorial1/features/order_features/dataset.yaml** under the 'features' section. 
>
> <ins>Note:</ins> In this tutorial, we don't need any new Python packages to be installed for these 2 new features. Therefore, we will use the file as it is. 
> However, whenever you use new Python packages for your newly added features, make sure that you add the packages in the **requirements.txt**.

**Feature Definition: review_score**
```yaml
  name: review_score
  description: "Review rating of the order between 1 and 5."
  source: review_score.py
  environment: requirements.txt
```

**Feature Definition: total_items**
```yaml
  name: total_items
  description: "Total number of items in the order."
  source: total_items.py
  environment: requirements.txt
```
