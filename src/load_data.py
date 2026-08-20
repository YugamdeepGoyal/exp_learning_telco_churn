import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os


def get_dataset():
    # Base or Parent directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Writing path to the dataset
    DATASET_PATH = BASE_DIR / "dataset" / "Telco_customer_churn.xlsx"

    # Creating folder to store images
    IMG_FOLDER = BASE_DIR / "images"
    os.makedirs(IMG_FOLDER, exist_ok=True)

    # Creating folder to store ready to use models
    MODEL_FOLDER = BASE_DIR / "models"
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"File not found at the {DATASET_PATH}")

    return pd.read_excel(DATASET_PATH)


def prepare_df(df):
    df = df.copy()
    df = df.drop(
        columns=[
            "CustomerID",
            "Count",
            "Country",
            "State",
            "Lat Long",
            "Latitude",
            "Longitude",
            "Zip Code",
            "City",
            "Churn Score",
            "Churn Reason",
            "Churn Label",
            "CLTV",
        ]
    )
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"] = df["Total Charges"].fillna(0)
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    return X_train, X_test, y_train, y_test
