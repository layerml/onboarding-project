# Tutorial I: How to Add New Source Datasets and Features into an Layer Project

## Step I: Add a New Dataset

> In this step, you will learn how to define one of your external source tables as a Layer Dataset.

> There is a new table named ***'olist_reviews'*** for you that resides on our public Snowflake database.
In order to define a new Layer Dataset for this table, first create a new directory in the project directory tree under **/tutorial1_before/data** and name it ***'reviews_dataset'***. 
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

> In this step, you will learn how to create 2 new Layer Features and add them into an existing Layer Featureset.

> We will create and add these 2 features:
> - Review Score
> - Total Items
> 
> For each feature, we will first create a respective python source file and add them in the project directoy tree under **/tutorial1_before/features/order_features**. 
> Create ***'review_score.py'*** and ***'total_distinct_items.py'***