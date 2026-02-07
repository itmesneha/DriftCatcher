# MLflow Logging Summary

## What Gets Logged to MLflow

### 1. Training Runs (`training/train.py`)
**Experiment:** Default (or specify with `mlflow.set_experiment()`)

**Metrics:**
- `roc_auc` - ROC-AUC score
- `accuracy` - Overall accuracy
- `precision` - Precision score
- `recall` - Recall score
- `f1_score` - F1 score
- `true_positives`, `true_negatives`, `false_positives`, `false_negatives`

**Parameters:**
- `model_type` - "RandomForest"
- `n_estimators` - Number of trees
- `test_size` - Train/test split ratio

**Artifacts:**
- Trained model (sklearn format)
- Registered under model name: `ddos_random_forest`

**View:**
```bash
uv run mlflow ui
# Go to http://localhost:5000
# Look under "Default" experiment or model registry
```

---

### 2. Drift Detection (`monitoring/drift_detector.py`)
**Experiment:** `drift_monitoring`

**Metrics:**
- `overall_psi` - Average PSI across all features
- `n_drifted_features` - Count of features with significant drift
- `total_features` - Total features checked
- `psi_rank_1_<feature>` to `psi_rank_10_<feature>` - Top 10 drifted features with their PSI scores

**Tags:**
- `action_recommended` - "retrain", "alert", or "wait"
- `timestamp` - When drift check was performed

**Artifacts:**
- `drift_feature_psi.json` - Complete PSI scores for all features

**Triggered by:**
- Running: `uv run python agent/retrain_agent.py check --data <file>`
- Or: `uv run python monitoring/drift_detector.py <file>`

**View in MLflow:**
```bash
uv run mlflow ui
# Navigate to "drift_monitoring" experiment
# Compare PSI scores over time
# Filter by action_recommended tag
```

---

### 3. Performance Monitoring (`monitoring/performance_monitor.py`)
**Experiment:** Default (uses run_name="production_monitoring")

**Metrics:**
- `prod_accuracy` - Production accuracy
- `prod_precision` - Production precision
- `prod_recall` - Production recall
- `prod_f1_score` - Production F1 score
- `prod_false_positive_rate` - FP rate
- `prod_false_negative_rate` - FN rate
- `prod_roc_auc` - Production ROC-AUC (if probabilities available)

**Requires:**
- CSV with columns: `y_true`, `y_pred`, `y_prob` (optional)

**Triggered by:**
```bash
uv run python monitoring/performance_monitor.py predictions.csv
```

**View in MLflow:**
```bash
uv run mlflow ui
# Look for runs named "production_monitoring"
# Compare production metrics vs training metrics
```

---

## MLflow UI Overview

### Dashboard View
```
Experiments
├── Default
│   ├── Training Runs (train.py)
│   └── Performance Monitoring (performance_monitor.py)
└── drift_monitoring
    └── Drift Detection Runs (drift_detector.py)

Models
└── ddos_random_forest
    ├── Version 1 (initial)
    ├── Version 2 (after retrain)
    └── ...
```

### Key Comparisons You Can Make

**1. Drift Over Time:**
```
Select: drift_monitoring experiment
Chart: overall_psi vs timestamp
→ See when drift spikes occur
```

**2. Training vs Production:**
```
Compare:
- Training: roc_auc, accuracy, precision, recall
- Production: prod_roc_auc, prod_accuracy, prod_precision, prod_recall
→ Detect performance degradation
```

**3. Model Versions:**
```
Go to: Models → ddos_random_forest
→ Compare metrics across versions
→ See which training data each version used
```

**4. Retrain Impact:**
```
Before retrain: drift_monitoring shows high PSI
After retrain: 
  - New training run logged
  - Next drift check shows lower PSI
  - Production performance improves
```

---

## Useful MLflow Queries

### Find High Drift Events
```python
from mlflow import MlflowClient

client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["drift_monitoring"],
    filter_string="metrics.overall_psi > 0.2"
)
```

### Compare Model Versions
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
versions = client.search_model_versions("name='ddos_random_forest'")

for v in versions:
    run = client.get_run(v.run_id)
    print(f"Version {v.version}: AUC = {run.data.metrics['roc_auc']}")
```

### Track Retraining Frequency
```bash
# Count retrains by checking training runs
mlflow runs list --experiment-name Default | grep train.py | wc -l
```

---

## What's NOT Logged to MLflow

**Agent State:**
- Cooldown status
- Last retrain time
- Total checks/retrains/alerts
→ These are in `agent/agent_state.json`

**Agent Logs:**
- Detailed event logs
- Error messages
- Action decisions
→ These are in `agent/logs/agent_YYYYMMDD.log`

**Raw Data:**
- Training data files
- Production data files
→ These should be versioned with DVC

---

## Best Practices

1. **Regular Checks:**
   - Schedule drift checks (hourly/daily)
   - Each check creates a new MLflow run in drift_monitoring

2. **Performance Tracking:**
   - When labels become available, log to performance_monitor
   - Compare against training baseline

3. **Experiment Organization:**
   - Keep training in Default experiment
   - Keep drift monitoring separate
   - Use tags to mark production vs test runs

4. **Model Registry:**
   - Use version tags: "production", "staging", "archived"
   - Add notes about what data each version was trained on

5. **Cleanup:**
   - Periodically archive old runs
   - Keep last 100 drift checks
   - Maintain key milestone models

---

## Viewing Complete Timeline

To see the full ML lifecycle in MLflow:

```bash
uv run mlflow ui
```

**Timeline view:**
1. Initial training → Model v1 registered
2. Drift detection runs (daily) → PSI stable
3. Week 2: PSI spike → Alert triggered
4. Manual retrain with new data → Model v2 registered
5. Performance monitoring → Metrics improve
6. Drift detection → PSI drops back to normal
7. Continue monitoring...

All tracked automatically in MLflow! 🎉
