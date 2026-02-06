# Baseline Data Versioning Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DVC BASELINE VERSIONING                             │
│                     (Sliding Window Data Strategy)                           │
└─────────────────────────────────────────────────────────────────────────────┘

INITIAL STATE (v1)
==================
┌──────────────────────┐
│  baseline.csv (74MB) │  ← Friday DDos dataset (2017)
│  [DVC tracked]       │
└──────────────────────┘
         │
         └─► baseline.csv.dvc (96B)  [Git tracked]
         └─► md5: b2b2764e4c8a...


PRODUCTION CYCLE 1: Drift Detected
====================================
1. Upload: new_data_1.csv (Production Monday traffic)
   
2. Digital Twin Detects:
   ┌────────────────────┐
   │ Drift PSI: 0.25    │
   │ Action: RETRAIN    │
   └────────────────────┘

3. Retraining:
   baseline.csv (2017) + new_data_1.csv (Production)
   ↓
   Combined: 150MB total

4. Automatic Baseline Update:
   ┌──────────────────────┐
   │  baseline.csv (150MB)│  ← 2017 + Production Week 1
   │  [DVC updated]       │
   └──────────────────────┘
         │
         └─► baseline.csv.dvc (96B)  [New MD5 hash]
         └─► md5: f3a9c82d1b7f...

5. Git Commit:
   $ git add data/baseline.csv.dvc
   $ git commit -m "Update baseline after retrain 1"
   [main abc1234] Update baseline after retrain 1


PRODUCTION CYCLE 2: More Drift
================================
1. Upload: new_data_2.csv (Production Tuesday traffic)

2. Retraining Now Uses Updated Baseline:
   baseline.csv (2017 + Week 1) + new_data_2.csv (Week 2)
   ↓
   Combined: 200MB total
   ✅ INCLUDES LEARNINGS FROM WEEK 1 (not lost!)

3. Baseline Updated Again:
   ┌──────────────────────┐
   │  baseline.csv (200MB)│  ← 2017 + Week 1 + Week 2
   │  [DVC updated]       │
   └──────────────────────┘
         │
         └─► baseline.csv.dvc (96B)  [New MD5 hash]
         └─► md5: 7e2b5c9a3d8e...

4. Git History Now Shows:
   $ git log --oneline data/baseline.csv.dvc
   def5678 Update baseline after retrain 2
   abc1234 Update baseline after retrain 1
   initial Initial baseline v1


ROLLBACK SCENARIO
=================
Model v3 performs worse after Week 2 data:

1. Rollback baseline to Week 1 version:
   $ git checkout abc1234 data/baseline.csv.dvc
   $ uv run dvc checkout data/baseline.csv

2. Baseline now restored to 150MB (2017 + Week 1)

3. Retrain with rollback baseline:
   baseline.csv (2017 + Week 1) + new_data_3.csv
   ↓
   New model trained on correct data


DATA FLOW COMPARISON
=====================

❌ WITHOUT DVC VERSIONING:
   Retrain 1: baseline (2017) + new_1 → model v1
   Retrain 2: baseline (2017) + new_2 → model v2  [Lost new_1! 😱]
   Retrain 3: baseline (2017) + new_3 → model v3  [Lost new_1 & new_2! 😱😱]

✅ WITH DVC VERSIONING:
   Retrain 1: baseline (2017) + new_1 → baseline (2017+new_1) → model v1
   Retrain 2: baseline (2017+new_1) + new_2 → baseline (2017+new_1+new_2) → model v2
   Retrain 3: baseline (2017+new_1+new_2) + new_3 → baseline (cumulative) → model v3


STORAGE EFFICIENCY
===================

Git Repository:
┌─────────────────────────────────┐
│ data/baseline.csv.dvc (96B)     │  ← Only 96 bytes per version!
│ - v1: md5 + metadata            │
│ - v2: md5 + metadata            │
│ - v3: md5 + metadata            │
└─────────────────────────────────┘

DVC Cache (Local or S3):
┌─────────────────────────────────┐
│ .dvc/cache/                     │
│ ├─ b2/b2764e... (74MB)  ← v1    │
│ ├─ f3/a9c82d... (150MB) ← v2    │  Only stores DIFFERENCES
│ └─ 7e/2b5c9a... (200MB) ← v3    │  (DVC deduplication)
└─────────────────────────────────┘


MLFLOW INTEGRATION
===================

Each retraining logs to MLflow:
┌─────────────────────────────────────────┐
│ Experiment: Model Training              │
│ ├─ Run 1 (v1)                           │
│ │  ├─ metric: training_samples = 230K   │
│ │  ├─ param: baseline_version = abc1234 │
│ │  └─ tag: baseline_md5 = b2b2764e...   │
│ ├─ Run 2 (v2)                           │
│ │  ├─ metric: training_samples = 450K   │  ← Grows!
│ │  ├─ param: baseline_version = def5678 │
│ │  └─ tag: baseline_md5 = f3a9c82d...   │
│ └─ Run 3 (v3)                           │
│    ├─ metric: training_samples = 650K   │  ← Accumulates
│    ├─ param: baseline_version = ghi9012 │
│    └─ tag: baseline_md5 = 7e2b5c9a...   │
└─────────────────────────────────────────┘


PRODUCTION DEPLOYMENT
======================

CI/CD Pipeline:
1. Model retrains in production environment
2. Validation passes (accuracy > 90%)
3. PlanningAgent updates baseline + runs `dvc add`
4. Automated commit:
   └─► git add data/baseline.csv.dvc
   └─► git commit -m "chore: Update baseline after production retrain"
   └─► git push origin main
5. Next deployment pulls latest baseline version
6. Continuous learning loop! 🔄


BENEFITS SUMMARY
=================

✅ Full Version History
   - Every baseline version tracked in Git
   - Complete audit trail of data evolution

✅ Efficient Storage
   - Git only stores 96-byte .dvc files
   - DVC handles large CSVs with deduplication

✅ Easy Rollback
   - Revert to any previous baseline in seconds
   - Test model performance on different data versions

✅ Production Data Accumulation
   - Model learns from ALL historical data
   - No data loss between retraining cycles

✅ Collaboration Ready
   - Team shares data via DVC remote (S3, GCS)
   - No need to commit large files to Git

✅ MLOps Integration
   - Track baseline version in MLflow experiments
   - Reproducible model training

✅ Cost Effective
   - S3/GCS storage cheaper than expanding Git repo
   - Pay only for unique data chunks
```
