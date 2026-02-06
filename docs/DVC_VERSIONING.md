# DVC Baseline Versioning Strategy

## Overview
DriftCatcher uses **DVC (Data Version Control)** to automatically version the baseline training data after each successful model retraining. This ensures the model accumulates production learnings over time.

## How It Works

### 1. Initial Setup
```bash
# Initial baseline created from Friday DDos dataset
cp data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv data/baseline.csv

# Track with DVC
uv run dvc add data/baseline.csv
git add data/.gitignore data/baseline.csv.dvc
git commit -m "Initial baseline v1"
```

### 2. Automatic Versioning During Retraining

When the Planning Agent executes retraining:

```python
# 1. Model retrains on: baseline.csv + new_data.csv
# 2. After successful training, combines datasets:
combined_data = pd.concat([baseline, new_data])
combined_data.to_csv('data/baseline.csv', index=False)

# 3. Versions the updated baseline with DVC:
subprocess.run(["uv", "run", "dvc", "add", "data/baseline.csv"])

# 4. Git tracks the .dvc file (not the large CSV)
# Run: git add data/baseline.csv.dvc && git commit
```

### 3. Data Accumulation Strategy

**Problem Solved:**
- ❌ **Before:** Each retrain used same baseline → lost previous production data
  - Retrain 1: `baseline.csv (2017)` + `new1.csv`
  - Retrain 2: `baseline.csv (2017)` + `new2.csv` ← **Lost new1!**

- ✅ **Now:** Baseline grows with each retrain
  - Retrain 1: `baseline.csv (2017)` + `new1.csv` → `baseline.csv (2017+new1)`
  - Retrain 2: `baseline.csv (2017+new1)` + `new2.csv` → `baseline.csv (2017+new1+new2)`

### 4. Benefits

1. **Version Control:** Git tracks baseline versions via `.dvc` files
2. **Storage Efficiency:** DVC deduplicates data (only stores differences)
3. **Easy Rollback:** `git checkout v1.0 data/baseline.csv.dvc && dvc checkout`
4. **Collaboration:** Push baseline versions to DVC remote (S3, GCS, etc.)
5. **Production-Ready:** Model learns from all historical production data

## Workflow Commands

### Check Current Baseline Version
```bash
# View DVC-tracked baseline
cat data/baseline.csv.dvc

# Check Git history of baseline versions
git log --oneline data/baseline.csv.dvc
```

### Rollback to Previous Baseline
```bash
# Find the commit with desired baseline
git log data/baseline.csv.dvc

# Checkout that version
git checkout <commit-hash> data/baseline.csv.dvc

# Pull the actual data
uv run dvc checkout data/baseline.csv
```

### Push to Remote Storage (Optional)
```bash
# Configure DVC remote (one-time setup)
uv run dvc remote add -d storage s3://my-bucket/driftcatcher

# Push baseline versions to remote
uv run dvc push
```

### Manual Baseline Update
```bash
# Update baseline with new data
cp data/uploads/new_attacks.csv data/baseline.csv

# Version it
uv run dvc add data/baseline.csv
git add data/baseline.csv.dvc
git commit -m "Update baseline with new attack patterns"
```

## Integration Points

### PlanningAgent._tool_retrain_model
```python
# After successful retraining:
# 1. Combine base + new data
combined_df = pd.concat([base_df] + new_dfs)

# 2. Save as new baseline
combined_df.to_csv('data/baseline.csv', index=False)

# 3. Version with DVC (automatic)
subprocess.run(["uv", "run", "dvc", "add", "data/baseline.csv"])
```

### API Digital Twin
```python
# Always uses latest DVC-tracked baseline
plan_context = {
    'base_data_path': 'data/baseline.csv',  # DVC-tracked
    'latest_data_path': new_data_path
}
```

## Monitoring

Track baseline evolution in MLflow:
- **Experiment:** `Model Training`
- **Metrics:** Training samples increase over retrains
- **Tags:** Baseline version from git commit

## Production Deployment

For production systems:
1. Set up DVC remote storage (S3/GCS/Azure)
2. Enable auto-staging: `dvc config core.autostage true`
3. CI/CD pipeline commits baseline updates after validation
4. Model registry tracks which baseline version was used

## Example: Complete Retrain Cycle

```bash
# 1. Upload new production data via dashboard
# 2. Digital twin detects drift, decides to retrain
# 3. PlanningAgent executes:
#    - retrain_model(base='data/baseline.csv', new='data/uploads/new.csv')
#    - Combines datasets → saves to data/baseline.csv
#    - Runs: dvc add data/baseline.csv
# 4. Commit the updated .dvc file:
git add data/baseline.csv.dvc
git commit -m "chore: Update baseline after production retrain"
# 5. Next retrain automatically uses updated baseline ✅
```

## Comparison with Alternatives

| Strategy | Storage | Pros | Cons |
|----------|---------|------|------|
| **DVC Versioning** (Current) | Git + DVC | Full history, efficient storage, rollback | Requires DVC setup |
| Manual baseline_v1.csv, v2.csv | Git LFS | Simple | Poor storage efficiency |
| Single baseline, no versioning | Local | Very simple | Can't rollback, no history |
| S3 with timestamps | S3 | Cloud storage | Manual tracking, no Git integration |

## Future Enhancements

- **Data Quality Checks:** Validate combined baseline before versioning
- **Automatic Cleanup:** Remove old baseline versions based on retention policy
- **Stratified Sampling:** Keep balanced baseline if it grows too large
- **A/B Testing:** Track which baseline version performs best
