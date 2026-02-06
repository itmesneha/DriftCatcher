import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import mlflow
import mlflow.sklearn


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
ARTIFACT_DIR = "artifacts"
MODEL_NAME = "ddos_random_forest"
RANDOM_STATE = 42

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# -----------------------------
# Load data
# -----------------------------
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    # Strip column whitespace (CRITICAL for CICIDS)
    df.columns = df.columns.str.strip()

    # Drop rows with missing labels
    df = df.dropna(subset=["Label"])

    # Binary label
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    
    print(f"Data loaded from {csv_path} with shape {df.shape}")
    return df



# -----------------------------
# Preprocess
# -----------------------------
def get_feature_columns(df):
    return [
        col for col in df.columns
        if col != "Label" and df[col].dtype != "object"
    ]

def clean_numeric_features(df, feature_cols):
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    return df


# -----------------------------
# Save training statistics
# -----------------------------
def compute_training_stats(df, feature_cols):
    stats = {}

    for col in feature_cols:
        values = df[col].values

        stats[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "quantiles": [
                float(q) for q in np.quantile(
                    values,
                    q=np.linspace(0, 1, 11)  # deciles
                )
            ]
        }

    return stats

def save_training_stats(stats, path="ARTIFACT_DIR"):
    path = os.path.join(ARTIFACT_DIR, "training_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Training statistics saved to {path}")


# -----------------------------
# Train
# -----------------------------
def train(df):
    feature_cols = get_feature_columns(df)
    df = clean_numeric_features(df, feature_cols)
    
    X = df[feature_cols]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    training_stats = compute_training_stats(X_train, feature_cols)
    save_training_stats(training_stats)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print("ROC-AUC:", auc)
        print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    train(df)
