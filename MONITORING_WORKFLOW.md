# DriftCatcher: Complete ML Monitoring Workflow

## Overview

DriftCatcher implements a complete ML monitoring system with **two complementary approaches**:

1. **Data Drift Detection (PSI)** - Proactive, no labels needed
2. **Performance Monitoring** - Reactive, requires labels

---

## The Complete Workflow

### 1. Initial Training
```bash
# Train initial model
uv run python training/train.py

# This creates:
# - Trained model (logged to MLflow)
# - training_stats.json (baseline distributions)
```

### 2. Production Monitoring Phase

#### A. Data Drift Detection (Continuous)
```bash
# Check new unlabeled data for drift
uv run python agent/retrain_agent.py check --data data/production/batch_001.csv
```

**What happens:**
- Computes PSI for each feature against training baseline
- If drift detected → Sends alert
- If severe drift → Recommends retraining

**Key Point:** This runs **immediately** when new data arrives, no labels needed!

#### B. Performance Monitoring (When labels available)
```bash
# After incident investigation, you get labels
# Create CSV with columns: y_true, y_pred, y_prob

uv run python monitoring/performance_monitor.py data/production/labeled_predictions.csv
```

**What happens:**
- Evaluates actual model performance
- Checks if accuracy/precision/recall dropped
- Logs to MLflow for tracking

---

## The Retraining Decision

### Scenario 1: Drift Detected (No Labels Yet)
```
1. PSI shows drift → Alert sent
2. Investigate: Is this a new attack pattern?
3. Label recent production samples
4. Manual retrain with new data
```

### Scenario 2: Performance Degradation (Labels Available)
```
1. Performance monitoring shows accuracy drop
2. You already have labeled production data
3. Manual retrain immediately
```

---

## Retraining with New Data

### ❌ Wrong Way (Current Issue)
```bash
# This just recreates the same model!
uv run python training/train.py
```

### ✅ Right Way (Fixed)
```bash
# Retrain on OLD + NEW labeled data
uv run python training/train.py \
  --base-data data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
  --new-data data/labeled/new_attacks_batch1.csv data/labeled/new_attacks_batch2.csv
```

**Or use the agent:**
```bash
uv run python agent/retrain_agent.py retrain \
  --new-labeled-data data/labeled/new_attacks_batch1.csv
```

This combines:
- Original training data (old patterns)
- New labeled data (new patterns)
- Model learns BOTH → Handles drift

---

## Practical Example Workflow

### Day 1: Initial Deployment
```bash
uv run python training/train.py
# Model v1 deployed
```

### Week 2: Drift Alert
```bash
# Automatic check
uv run python agent/retrain_agent.py check --data data/production/week2_batch.csv

# Output: PSI = 0.25 → "RETRAIN RECOMMENDED"
```

### Week 2: Human Investigation
```
1. Review alerts: "Packet size distribution changed"
2. Investigate: New DDoS attack type discovered
3. Label 1000 samples from week 2 data → save to data/labeled/week2_samples.csv
```

### Week 2: Retrain
```bash
uv run python agent/retrain_agent.py retrain \
  --new-labeled-data data/labeled/week2_samples.csv

# Model v2 deployed - now handles both old and new attack patterns
```

### Week 3: Performance Check (After Labels)
```bash
# After security team confirms attacks/benign traffic
uv run python monitoring/performance_monitor.py data/production/week3_labeled.csv

# Output: Accuracy = 0.96, Precision = 0.94 ✓
```

---

## Monitoring Commands Summary

### Drift Detection
```bash
# Auto check with agent
uv run python agent/retrain_agent.py check --data <new_data.csv>

# Check agent status
uv run python agent/retrain_agent.py status

# Manual drift check only
uv run python monitoring/drift_detector.py <new_data.csv>
```

### Performance Monitoring
```bash
# Evaluate predictions against labels
uv run python monitoring/performance_monitor.py <predictions.csv>
```

### Retraining
```bash
# Manual retrain with new labeled data
uv run python agent/retrain_agent.py retrain \
  --new-labeled-data data/labeled/batch1.csv data/labeled/batch2.csv

# Or directly
uv run python training/train.py \
  --new-data data/labeled/batch1.csv data/labeled/batch2.csv

# Force retrain even in cooldown
uv run python agent/retrain_agent.py retrain \
  --new-labeled-data data/labeled/batch1.csv \
  --force
```

---

## Key Concepts

### Why Both Drift Detection AND Performance Monitoring?

| | Data Drift (PSI) | Performance Monitoring |
|---|---|---|
| **When** | Immediately | Days/weeks later |
| **Input** | Unlabeled data | Labeled data |
| **Detects** | Distribution changes | Actual errors |
| **Action** | Early warning | Confirmation |
| **Frequency** | Every batch | When labels available |

### Why Retrain on OLD + NEW Data?

**Problem:** If you retrain only on new data:
- Model forgets old patterns (catastrophic forgetting)
- Breaks on previously working cases

**Solution:** Combine datasets:
```python
df_old = load("old_training.csv")  # Original patterns
df_new = load("new_labeled.csv")   # New patterns discovered
df_combined = pd.concat([df_old, df_new])
train(df_combined)  # Model handles BOTH
```

### Data Labeling Strategy

After drift detected:
1. **Sample recent data** (e.g., 500-1000 samples)
2. **Prioritize diverse samples** (different time periods, sources)
3. **Get expert labels** (security team confirms attacks)
4. **Save as CSV** with same schema as training data
5. **Retrain** with combined dataset

---

## Automation Options

### Schedule Periodic Checks (cron)
```bash
# Check every hour
0 * * * * cd /path/to/DriftCatcher && uv run python agent/retrain_agent.py check --data /path/to/latest_batch.csv
```

### Continuous Monitoring Service
```python
# monitoring/service.py
while True:
    latest_batch = fetch_latest_data()
    agent.check_and_act(latest_batch)
    time.sleep(3600)  # Check hourly
```

### MLflow Integration
- All training runs logged automatically
- Compare model versions
- Track drift over time
- Performance trends visualization

```bash
uv run mlflow ui
# Open http://localhost:5000
```

---

## Best Practices

1. **Set appropriate thresholds:**
   - PSI > 0.2 for critical retraining
   - Accuracy < 0.90 for alerts

2. **Maintain cooldown periods:**
   - Prevent retraining too frequently
   - Default: 24h between retrains, 6h between alerts

3. **Label strategically:**
   - Don't label everything
   - Focus on samples where drift detected
   - Quality over quantity

4. **Version everything:**
   - Use DVC for data versions
   - MLflow for model versions
   - Git for code versions

5. **Monitor both metrics:**
   - Drift = leading indicator
   - Performance = lagging indicator
   - Use both for complete picture

---

## Troubleshooting

### "Retrain recommended but model performs fine"
- False positive from PSI
- Data shifted but model still works
- Consider tuning PSI thresholds

### "Model degraded but no drift detected"
- Concept drift (Y changes, not X)
- PSI only catches feature distribution changes
- Performance monitoring caught it correctly

### "Retrained but performance still bad"
- Need more diverse labeled data
- Model capacity issue (try different algorithm)
- Features not capturing new patterns
