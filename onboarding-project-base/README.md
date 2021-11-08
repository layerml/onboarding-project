# Onboarding Project

## Install and run
To check out the Tutorial I, run:
```commandline
1. layer clone https://github.com/layerml/onboarding-project.git
2. cd onboarding-project/onboarding-project-base
```

To build the project:
```commandline
layer start
```

## Onboarding Base Project Directory Tree
```
.
├── .layer
├── data
│   ├── category_name_translation_dataset  
│   │   ├── dataset.yaml         
│   ├── customers_dataset  
│   │   ├── dataset.yaml
│   ├── items_dataset  
│   │   ├── dataset.yaml
│   ├── orders_dataset  
│   │   ├── dataset.yaml
│   ├── payments_dataset  
│   │   ├── dataset.yaml
│   ├── products_dataset  
│   │   ├── dataset.yaml
├── features
│   ├── customer_features
│   │   ├── first_order_id.py  
│   │   ├── first_order_timestamp.py
│   │   ├── ordered_again.py
│   │   ├── dataset.yaml
│   │   ├── requirements.txt         
│   ├── order_features 
│   │   ├── days_between_delivery_and_purchase.py  
│   │   ├── days_between_estimate_actual_delivery.py
│   │   ├── main_product_category.py
│   │   ├── total_freight.py
│   │   ├── total_payment.py
│   │   ├── dataset.yaml
│   │   ├── requirements.txt  
├── models
│   └── churn_model
│       ├── model.py              
│       ├── model.yaml                
│       └── requirements.txt   
└── README.md
```