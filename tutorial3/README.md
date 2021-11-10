# Tutorial III: How to build different versions of a model

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
> Imagine that you have a baseline model previously trained. 
> That's the model you deployed on your production before and running at the moment. 
> After a while, you wanted to revisit this model and try to make some improvement on its performance. 


> Here is the list of possible ways you have in mind improving your model:
>- Make use of another featureset as well:
>  - You have a new featureset now: `order_features_high_level`, since the last time you trained this model.
>  - Let's try a new set of features which mixes the features from the `order_features` featureset as well as the `order_high_level_features` featureset.
>
>- Make some changes in the model parameters:
>  - You have been using the Gradient Boosting Classifier from Sklearn for this problem. 
     You are wondering what if a change in the model parameters will make a positive impact on the model performance.
>  - You would like to try a different value for the `max_depth` parameter of the model.


- First create a new directory for your new featureset:
  - Since this featureset will be based on the existing ***order_features*** featureset, 
  let's name it ***order_features_high_level*** for the sake of clarity.
  - We will add a new featureset into the project which contains 3 features:
    - is_multi_items
    - shipping_payment_perc
    - total_waiting


- In the new directory 'order_features_high_level':
  - create respective **python source files (.py)** for the 3 features above 
  - create a **dataset.yaml** file for featureset definition 
  - create a **requirements.txt** file for listing required python packages.

**How newly added files look in the project directory tree**
```
.
├──tutorial2  
│   ├── features
│   │   ├── order_high_level_features
│   │   │   ├── is_multi_items.py
│   │   │   ├── shipping_payment_perc.py
│   │   │   ├── total_waiting.py
│   │   │   ├── requirements.txt
│   │   │   ├── dataset.yaml
```
## Put relevant contents into these files
>Copy the code blocks below and paste them into the respective files.

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
description: "Higher level features about orders based on other order featureset: order_features"


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

> That's it, you are done. Congratulations! You just learned how to create a new featureset based on another featureset and add it into your Layer project.

