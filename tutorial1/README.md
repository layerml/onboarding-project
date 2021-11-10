# Tutorial I: How to add new Datasets into your Layer project & build new Features 

## Install and run
To check out the Tutorial I, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project.git
2. cd onboarding-project/tutorial1
```

To build the project:
```commandline
layer start
```

To rebuild the featureset after the following steps below:
```commandline
layer start featureset order_features
```


## Step I : Add a new Layer Dataset

- In this step, you will learn how to define one of your external source tables as a Layer Dataset.


- There is a table named ***olist_reviews*** that resides on the Layer's public database on Snowflake. Let's say, we would like use this table for our Layer Project.


- In order to define a new Layer Dataset entity for the table, first create a new directory under the **'/tutorial1/data'** and name it ***reviews_dataset***. 


- In the new directory '/tutorial1/data/reviews_dataset', create a **dataset.yaml** file.

**How newly added files look in the project directory tree**
.
├──tutorial1  
│   ├── data
│   │   ├── reviews_dataset
│   │   │   ├── dataset.yaml


> Copy the block below and paste it into the dataset.yaml file
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

## Step II: Add new features into an existing Layer Featureset

### Creating feature source files

- In this step, you will learn how to create 2 new Layer Features and add them into the existing Layer Featureset: ***order_features***


- We will add 2 new features into the project:
  - Review Score
  - Total Items
 

- For each feature, we will first create its respective python source file.
  - Create ***review_score.py*** and ***total_items.py*** python source files 
  - Add them in the project directory under the **/tutorial1/features/order_features**. 

**How newly added files look in the project directory tree**
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
- Now, we will add the feature definitions below into the featureset yaml file: **/tutorial1/features/order_features/dataset.yaml** 


- Add the feature definitions below under the 'features' section in the yaml file. 

> <ins>Note:</ins> In this tutorial, we don't need any new Python packages to be installed for these 2 new features. Therefore, we will use the file as it is. 
> However, whenever you use new Python packages for your newly added features, make sure that you add the packages in the **requirements.txt**.

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
> That's it, you are done. Congratulations! You just learned how define a source table as a Layer Dataset entity and how to add new features into your existing Layer Featureset.
