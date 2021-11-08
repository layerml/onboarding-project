# Tutorial II: How to create a new Layer Featureset based on another Featureset

## Step I: Changes In Your Project Directory

> In this step, you will create the files required for adding a new layer featureset into your existing project.

> We will add a new featureset into the project which contains 3 features:
> - is_multi_items
> - shipping_payment_perc
> - total_waiting

> Go to the project tree and create a new directory for your new featureset. 
> Since this featureset will be based on the existing ***order_features*** featureset, 
> let's name it ***order_features_high_level*** for the sake of clarity.
>
> In the new directory 'order_features_high_level',create respective **python source files (.py)** for the 3 features above then create a **dataset.yaml** file and a **requirements.txt** file.

## Step II: Fill The New Files With The Relevant Contents
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

