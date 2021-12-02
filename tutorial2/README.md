# Tutorial II: How to build a new Featureset based on another Featureset

## What you will learn in this tutorial?
- In this step, you will learn how to create a new Layer Featureset based on another existing Featureset.


- Once you build the Tutorial II's base project in the step I below, you will see that you already have a Layer Featureset named _order_features_tutorial2_ :
  - Since the new featureset will be based on the existing ***order_features_tutorial2***, 
  let's name the new featureset ***order_features_tutorial2_new*** for the sake of clarity.
  - This featureset will contain 3 new order features:
    - is_multi_items
    - shipping_payment_perc
    - total_waiting


- In the new directory '/features/order_features_tutorial2_new':
  - We will create **python source files (.py)** for all the 3 features above 
  - We will create a **yaml** file for the featureset definition 
  - We will create a **requirements.txt** file for listing required python packages.

**How newly added files look in the project directory tree**
```
.
├──tutorial2  
│   ├── features
│   │   ├── order_features_new
│   │   │   ├── is_multi_items.py
│   │   │   ├── shipping_payment_perc.py
│   │   │   ├── total_waiting.py
│   │   │   ├── requirements.txt
│   │   │   ├── order_features_new.yaml
```

## Step I: Clone the repo
To check out the Tutorial II's base project, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project-and-tutorials.git
2. cd onboarding-project-and-tutorials/tutorial2
```

To build existing Layer entities in this tutorial for the first time, run:
```commandline
layer start
```

## Step II: Create python source files
>Copy the code blocks below and paste them into the respective files.

***is_multi_items.py***
```python
from typing import Any
from layer import Featureset
import numpy as np

def build_feature(order_features_layer: Featureset("order_features_tutorial2")) -> Any:
    order_features_df = order_features_layer.to_pandas()

    df = order_features_df.assign(IS_MULTI_ITEMS=np.where(order_features_df.TOTAL_ITEMS > 1.0, 1, 0))

    is_multi_items = df[["ORDER_ID","IS_MULTI_ITEMS"]]

    return is_multi_items
```
***shipping_payment_perc.py***
```python
from typing import Any
from layer import Featureset

def build_feature(order_features_layer: Featureset("order_features_tutorial2")) -> Any:
    order_features_df = order_features_layer.to_pandas()

    df = order_features_df.assign(SHIPPING_PAYMENT_PERC=(order_features_df.TOTAL_FREIGHT_PRICE / (order_features_df.TOTAL_PRODUCT_PRICE + order_features_df.TOTAL_FREIGHT_PRICE)) * 100)

    shipping_payment_perc = df[["ORDER_ID","SHIPPING_PAYMENT_PERC"]]

    return shipping_payment_perc
```

***total_waiting.py***
```python
from typing import Any
from layer import Featureset

def build_feature(order_features_layer: Featureset("order_features_tutorial2")) -> Any:
    order_features_df = order_features_layer.to_pandas()

    df = order_features_df.assign(TOTAL_WAITING=order_features_df.PAYMENT_APPROVEMENT_WAITING.astype(int) + order_features_df.DELIVERED_CARRIER_WAITING.astype(int))

    total_waiting = df[["ORDER_ID","TOTAL_WAITING"]]

    return total_waiting
```
## Step II: Create requirements.txt file
>Copy the block below and paste it into the requirements.txt file.

***requirements.txt***
```text
numpy==1.21.1
```

## Step III: Create yaml file
>Copy the block below and paste it into the order_features_tutorial2_new.yaml file.

***order_features_tutorial2_new.yaml***
```yaml
# In this `yaml` file, we will define the attributes of our featureset.

# For more information on Featureset Configuration: https://docs.beta.layer.co/docs/datacatalog/featuresets

apiVersion: 1
type: featureset

# Unique name for the featureset which will be used in this project to refer to this featureset
name: "order_features_tutorial2_new"
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
    target: layer-public-datasets
```
## Step IV: Build the new features
> To build the new featureset, run:

```commandline
layer start featureset order_features_tutorial2_new
```


Congratulations, you have just completed the tutorial. Now, you know how to create a new featureset based on another featureset and add it into your project on Layer.
To check if you are done correct, go and check the Tutorial 2's after project:
```commandline
cd onboarding-project-and-tutorials/tutorials_after/tutorial2_after
```
