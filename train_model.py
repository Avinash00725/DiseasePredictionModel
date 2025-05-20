import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
datasets = {
    "asthma": {"path": "asthma/asthma.csv", "target": "target"},  
    "cancer": {"path": "cancer/cancer.csv", "target": "target"},
    "diabetes": {"path": "diabetes/diabetes.csv", "target": "target"},
    "stroke": {"path": "stroke/stroke.csv", "target": "target"}
}


def train_and_save_model(disease, file_path, target_column):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Dataset for {disease} not found at {file_path}")
        return

    print(f"\nColumns in {disease} dataset: {list(data.columns)}")
    print(f"First few rows of {disease} dataset:\n{data.head()}")

    if target_column not in data.columns:
        print(f"Target column '{target_column}' not found in {disease} dataset. Available columns: {list(data.columns)}")
        return

    X = data.drop(columns=[target_column]) 
    y = data[target_column]  

    print(f"Unique values in target ({target_column}) for {disease}: {y.unique()}")

    if not set(y.unique()).issubset({0, 1}):
        print(f"Target column for {disease} should be binary (0/1), but found: {y.unique()}")
        return

    for column in X.columns:
        if X[column].dtype == "object":
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"{disease} model accuracy: {accuracy:.2f}")

    joblib.dump(model, f"{disease}_model.pkl")
    print(f"Saved {disease}_model.pkl")

for disease, info in datasets.items():
    print(f"Training model for {disease}...")
    train_and_save_model(disease, info["path"], info["target"])

print("All models trained and saved!")
