"""Quick training without MLflow"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.universal_loader import UniversalDataLoader
from config.dataset_config import load_dataset_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
import json

# Load data
config = load_dataset_config("CICIDS2017")
loader = UniversalDataLoader(config)
X_train, X_test, y_train, y_test = loader.load_and_preprocess(
    "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    test_size=0.2
)

# Train
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nResults:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save model
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save stats
stats = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "n_features": X_train.shape[1],
    "feature_names": loader.feature_columns
}
with open("artifacts/training_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n✅ Model saved to artifacts/model.pkl")
print("✅ Stats saved to artifacts/training_stats.json")
