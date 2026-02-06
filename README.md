New attack appears →
Feature distributions shift:
  - Flow Duration mean changes
  - SYN flag ratios change
  - Packet rate distribution changes

Agent observes:
  - KL divergence ↑
  - PSI ↑
  - Confidence entropy ↑

Agent decides:
  - This is not DDOS-like drift
  - Trigger retraining
  - Pull new CSVs
  - Train new model version

uv run dvc add data/raw/
git add data/raw.dvc
git commit -m "Update raw data"

We intentionally separated drift detection from retraining decisions. Drift metrics are deterministic, but the retraining agent reasons over time, memory, and operational constraints like cooldowns and data availability.

Tech stack (hackathon-friendly)

Data

CICIDS2017

UNSW-NB15

Or even synthetic logs (acceptable!)

Model

Logistic Regression / XGBoost / Random Forest

Or Isolation Forest for anomaly detection

👉 Simplicity = good engineering judgment

MLOps

MLflow for experiment tracking

Dockerized training + inference

FastAPI inference endpoint

Simple CI pipeline (even pseudo)

Monitoring

Prediction distribution

Drift detection (feature mean shifts)

Alert volume over time

1. Detect drift (PSI) → New data distribution changed
2. Collect & label new data → Get ground truth for recent samples
3. Retrain on OLD + NEW data → Model learns new patterns
4. Deploy updated model → Now handles both old and new patterns