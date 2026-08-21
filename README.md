# exp_learning_telco_churn

This project is my assignment work on predicting customer churn using the Telco Customer Churn dataset. I explored a few different classification approaches, including Logistic Regression, Random Forest, and a simple Artificial Neural Network (ANN), to see which one does the best job of identifying customers who are likely to churn.

## About

This was built as part of a learning assignment. The dataset (`Telco_customer_churn.xlsx`) was provided by my teacher and is used here only for educational purposes.

## Project Structure
exp_learning_telco_churn/
│
├── dataset/
│   └── Telco_customer_churn.xlsx     # Raw dataset (provided by teacher)
│
├── images/                           # Saved plots/visualizations
│
├── models/                           # Trained/saved models
│
├── src/
│   ├── __init__.py
│   ├── load_data.py                  # Loads dataset & prepares train/test split
│   ├── metrics.py                    # Evaluation metrics & plots (confusion matrix, ROC-AUC)
│   ├── pipeline.py                   # Preprocessing pipeline (OneHotEncoder + StandardScaler)
│   └── pipeline_tree.py              # Preprocessing pipeline for tree-based models (OrdinalEncoder)
│
├── tuner_results/                    # Hyperparameter tuning results
│
├── eda.ipynb                         # Exploratory Data Analysis
├── processing.ipynb                  # Data preprocessing/cleaning
├── LogisticRegression.ipynb          # Logistic Regression model
├── RandomForest.ipynb                # Random Forest model
├── ann_model.ipynb                   # Artificial Neural Network model
│
├── LICENSE
└── README.md


## Dataset

The dataset used is the Telco Customer Churn dataset. It contains customer demographic information, account details, and the services each customer signed up for, along with whether or not that customer churned.

This dataset was given to me by my teacher as part of the assignment, so all credit for the data itself goes to them.

Some of the key columns used for modeling include:

Demographics: Gender, Senior Citizen, Partner, Dependents

Services: Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies

Account info: Contract, Paperless Billing, Payment Method, Tenure Months, Monthly Charges, Total Charges

Target variable: Churn Value

A few columns like CustomerID, Count, Country, State, Lat Long, Latitude, Longitude, Zip Code, City, Churn Score, Churn Reason, Churn Label, and CLTV were dropped during preprocessing since they were not useful for the modeling task.

## Workflow

1. EDA (eda.ipynb): Looked at the distributions, correlations, and general churn patterns in the data.
2. Preprocessing (processing.ipynb, src/load_data.py): Cleaned the data, handled missing values in Total Charges, and split everything into train and test sets.
3. Pipelines (src/pipeline.py, src/pipeline_tree.py): Built two preprocessing pipelines.
   pipeline.py uses one hot encoding for categorical features along with standard scaling for numerical features, mainly used for Logistic Regression.
   pipeline_tree.py uses ordinal encoding for categorical features, used for tree based models like Random Forest.
4. Modeling: Trained and tested three different models in LogisticRegression.ipynb, RandomForest.ipynb, and ann_model.ipynb.
5. Evaluation (src/metrics.py): Calculated accuracy, precision, recall, F1 score, and ROC-AUC score, along with confusion matrix and ROC curve plots for each model.

## Evaluation Metrics

Each model was evaluated using the following metrics.

Accuracy
Precision
Recall
F1 Score
ROC-AUC Score
Confusion Matrix
ROC Curve

## Tech Stack

Python
pandas and numpy
scikit-learn
matplotlib and seaborn
Jupyter Notebook

## Getting Started

1. Clone the repository.
2. Place the Telco_customer_churn.xlsx file inside the dataset folder.
3. Install the required dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn.
4. Run the notebooks in this order: eda.ipynb, then processing.ipynb, then the model notebooks.

## License
MIT LICENSE

## Acknowledgements
Dataset provided by my teacher for this assignment.

>> This is an educational project for my assignment.