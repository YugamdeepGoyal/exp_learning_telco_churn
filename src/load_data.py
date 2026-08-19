import numpy as np
import pandas as pd
from pathlib import Path
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
