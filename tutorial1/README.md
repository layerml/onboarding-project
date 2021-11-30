# Tutorial I: How to add new Datasets into your Layer project & build new Features 

## What you will learn in this tutorial?
Assume that your team have many features previously extracted from several different data sources. 
In the project folder structure, you can see the whole list of existing order-related features
under the _/tutorial1/features/order_features/_ once you build the Tutorial I's base project in the Step I below. 

Let's say, you've found another useful data source on your warehouse lately. 
You would like to define that table as a Layer dataset to be able to extract some more features out of it.

At the end of this tutorial, you will learn how to define a source table on your warehouse as a new Layer Dataset and add brand-new features into your existing Layer Featureset.


## Step I: Fetch the repo
To check out the Tutorial I's base project, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project-and-tutorials.git
2. cd onboarding-project-and-tutorials/tutorial1
```
To build the whole project:
```commandline
layer start
```


## Step II : Add a new Layer Dataset

- In this step, you will learn how to define one of your external source tables on your data warehouse as a Layer Dataset.


- There is a new table named ***olist_reviews*** resides on the Layer's public database on Snowflake. Let's say, we would like use this table for our Layer Project.


- In order to define a new Layer Dataset entity for the table, first create a new file **'/tutorial1/data/reviews_dataset.yaml'**.


> Copy the block below and paste it into the dataset.yaml file
```yaml
# For more information on Dataset Configuration: https://docs.beta.layer.co/docs/datacatalog/datasets

apiVersion: 1
type: dataset

# Unique name of this dataset which will be used in this project to refer to this dataset
name: "reviews_dataset"
description: "This dataset includes data about the order reviews."

materialization:
    target: layer-public-datasets
    table_name: "olist_reviews"
```

## Step III: Add new features into an existing Layer Featureset

### Creating feature source files

- In this step, you will learn how to create 2 new Layer Features and add them into the existing Layer Featureset: ***order_features***


- We will add 2 new features into the project:
  - Review Score
  - Total Items
 

- For each feature, we will first create its respective python source file.
  - Create ***review_score.py*** and ***total_items.py*** python source files 
  - Add them in the project directory under the **/tutorial1/features/order_features**. 

**How newly added files look in the project folder structure**
```
.
├──tutorial1  
│   ├── features
│   │   ├── order_features
│   │   │   ├── review_score.py
│   │   │   ├── total_items.py
```
 
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
### Adding feature definitions into the featureset
- Now, we will add the feature definitions into the featureset yaml file: **/tutorial1/features/order_features/order_features.yaml** 


>Copy the feature definitions below and paste them into their respective places in the yaml file: "# review_score feature definition goes here" & "# total_items feature definition goes here"

***Feature Definition: review_score***
```yaml
  name: review_score
  description: "Review rating of the order between 1 and 5."
  source: review_score.py
  environment: requirements.txt
```

***Feature Definition: total_items***
```yaml
  name: total_items
  description: "Total number of items in the order."
  source: total_items.py
  environment: requirements.txt
```
> That's it, you are done. Congratulations! You just learned how to define a source table as a Layer Dataset entity and how to add new features into your existing Layer Featureset.

To check if you are done correct, go and check the Tutorial 1's after project:
```commandline
cd onboarding-project-and-tutorials/tutorials_after/tutorial1_after
```