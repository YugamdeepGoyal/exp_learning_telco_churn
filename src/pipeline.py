import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_preprocessor():
    cat_cols = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Multiple Lines", "Internet Service", "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing", "Payment Method"]
    scaling_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]


    encoder = Pipeline([
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    scaler = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("encoder_col", encoder, cat_cols),
        ("scaling_col", scaler, scaling_cols)
    ])

    return preprocessor
