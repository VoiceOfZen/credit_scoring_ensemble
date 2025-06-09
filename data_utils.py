import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
DATA_FILE = "default_of_credit_card_clients.xls"

def download_dataset():
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def load_and_prepare_data():
    download_dataset()

    # Load Excel file
    df = pd.read_excel(DATA_FILE, header=1)

    # Drop ID column if present
    df.drop("ID", axis=1, inplace=True)

    # Define features and target
    X = df.drop("default payment next month", axis=1)
    y = df["default payment next month"]

    # Feature names for plotting
    feature_names = X.columns.tolist()

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names
