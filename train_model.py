import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
<<<<<<< HEAD

# Define paths to datasets and target column names
datasets = {
    "asthma": {"path": "asthma/asthma.csv", "target": "target"},  # Adjust target column name
=======
datasets = {
    "asthma": {"path": "asthma/asthma.csv", "target": "target"},  
>>>>>>> d6ae26a2cb0a79581a456fa75cba338c054f25f0
    "cancer": {"path": "cancer/cancer.csv", "target": "target"},
    "diabetes": {"path": "diabetes/diabetes.csv", "target": "target"},
    "stroke": {"path": "stroke/stroke.csv", "target": "target"}
}

<<<<<<< HEAD
# Function to train and save model
def train_and_save_model(disease, file_path, target_column):
    # Load dataset
=======

def train_and_save_model(disease, file_path, target_column):
>>>>>>> d6ae26a2cb0a79581a456fa75cba338c054f25f0
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Dataset for {disease} not found at {file_path}")
        return

<<<<<<< HEAD
    # Debugging: Print columns and first few rows
    print(f"\nColumns in {disease} dataset: {list(data.columns)}")
    print(f"First few rows of {disease} dataset:\n{data.head()}")

    # Check if target column exists
=======
    print(f"\nColumns in {disease} dataset: {list(data.columns)}")
    print(f"First few rows of {disease} dataset:\n{data.head()}")

>>>>>>> d6ae26a2cb0a79581a456fa75cba338c054f25f0
    if target_column not in data.columns:
        print(f"Target column '{target_column}' not found in {disease} dataset. Available columns: {list(data.columns)}")
        return

<<<<<<< HEAD
    # Extract features and target
    X = data.drop(columns=[target_column])  # Features
    y = data[target_column]  # Target

    # Debugging: Check unique values in y
    print(f"Unique values in target ({target_column}) for {disease}: {y.unique()}")

    # Ensure target is binary (0/1) for classification
=======
    X = data.drop(columns=[target_column]) 
    y = data[target_column]  

    print(f"Unique values in target ({target_column}) for {disease}: {y.unique()}")

>>>>>>> d6ae26a2cb0a79581a456fa75cba338c054f25f0
    if not set(y.unique()).issubset({0, 1}):
        print(f"Target column for {disease} should be binary (0/1), but found: {y.unique()}")
        return

<<<<<<< HEAD
    # Handle categorical columns if any
=======
>>>>>>> d6ae26a2cb0a79581a456fa75cba338c054f25f0
    for column in X.columns:
        if X[column].dtype == "object":
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])

<<<<<<< HEAD
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate model
    accuracy = model.score(X_test, y_test)
    print(f"{disease} model accuracy: {accuracy:.2f}")

    # Save the model
    joblib.dump(model, f"{disease}_model.pkl")
    print(f"Saved {disease}_model.pkl")

# Train and save models for each disease
=======
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"{disease} model accuracy: {accuracy:.2f}")

    joblib.dump(model, f"{disease}_model.pkl")
    print(f"Saved {disease}_model.pkl")

>>>>>>> d6ae26a2cb0a79581a456fa75cba338c054f25f0
for disease, info in datasets.items():
    print(f"Training model for {disease}...")
    train_and_save_model(disease, info["path"], info["target"])

<<<<<<< HEAD
print("All models trained and saved!")
=======
print("All models trained and saved!")
>>>>>>> d6ae26a2cb0a79581a456fa75cba338c054f25f0
