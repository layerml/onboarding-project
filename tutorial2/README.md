# Tutorial II: How to Add A New Featureset Which Depends On An Existing Featureset

## Step I: Changes In Your Project Directory

> In this step, you will learn and create the files required for adding a new layer featureset into your existing project.

> We will add a new featureset into the project which contains 3 features:
> - is_multi_items
> - shipping_payment_perc
> - total_waiting

> Go to the project tree and create a new directory for your new featureset. 
> Since this featureset will be based on the existing ***order_features*** featureset, 
> let's name it ***order_features_high_level*** for the sake of clarity.
>
> In the new directory 'order_features_high_level',create respective **python source files (.py)** for the 3 features above then create a **dataset.yaml** file and a **requirements.txt** file.
 
>Copy and paste the code blocks below into the respective files.

***is_multi_items.py***
```python
from typing import Any
from layer import Featureset
import numpy as np

def build_feature(order_items_features_layer: Featureset("order_features")) -> Any:
    order_items_features_df = order_items_features_layer.to_pandas()

    df = order_items_features_df.assign(IS_MULTI_ITEMS=np.where(order_items_features_df.TOTAL_ITEMS > 1.0, 1, 0))

    is_multi_items = df[["ORDER_ID","IS_MULTI_ITEMS"]]

    return is_multi_items
```
***shipping_payment_perc.py***
```python
from typing import Any
from layer import Featureset

def build_feature(order_items_features_layer: Featureset("order_features")) -> Any:
    order_items_features_df = order_items_features_layer.to_pandas()

    df = order_items_features_df.assign(SHIPPING_PAYMENT_PERC=(order_items_features_df.TOTAL_FREIGHT_PRICE / (order_items_features_df.TOTAL_PRODUCT_PRICE + order_items_features_df.TOTAL_FREIGHT_PRICE)) * 100)

    shipping_payment_perc = df[["ORDER_ID","SHIPPING_PAYMENT_PERC"]]

    return shipping_payment_perc
```

***total_waiting.py***
```python
from typing import Any
from layer import Featureset

def build_feature(order_general_features_layer: Featureset("order_general_features")) -> Any:
    order_general_features_df = order_general_features_layer.to_pandas()

    df = order_general_features_df.assign(TOTAL_WAITING=order_general_features_df.PAYMENT_APPROVEMENT_WAITING.astype(int) + order_general_features_df.DELIVERED_CARRIER_WAITING.astype(int))

    total_waiting = df[["ORDER_ID","TOTAL_WAITING"]]

    return total_waiting

```

***requirements.txt***
```text
numpy==1.21.1
```

***dataset.yaml***
```yaml
# In this `yaml` file, we will define the attributes of our featureset.

# For more information on Featureset Configuration: https://docs.beta.layer.co/docs/datacatalog/featuresets

apiVersion: 1

type: featureset

name: "order_high_level_features"
description: "High level features about orders based on other order featuresets."


features:
    - name: is_multi_items
      description: "A binary columns that indicates if the order has multiple items (1) or not (0)."
      source: is_multi_items.py
      environment: requirements.txt
    - name: shipping_payment_perc
      description: "Percentage of the shipping cost over total payment."
      source: shipping_payment_perc.py
      environment: requirements.txt
    - name: total_waiting
      description: "Total number of days the customer had to wait for payment approvement and carrier to pick up the items of the order."
      source: total_waiting.py
      environment: requirements.txt

materialization:
    target: olist-ecommerce-datasets
```

> That's it, you are done. Congratulations! You just defined a source table on your database as a Layer Dataset.

## Step II: Add New Features

> In this step, you will learn how to create 2 new Layer Features and add them into the existing Layer Featureset: ***order_features***

> We will create and add 2 features below into our project:
> - Review Score
> - Total Items
> 
> For each feature, we will first create its respective python source file and then define it into the featureset along with its description.

> Create ***review_score.py*** and ***total_items.py*** python source files and add them in the project directory under the **/tutorial1/features/order_features**. 
 
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
