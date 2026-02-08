# DriftCatcher

**Autonomous ML Model Lifecycle Management with Agentic AI**

DriftCatcher is an intelligent system that autonomously monitors, reasons about, and responds to ML model drift using LLM-powered agentic reasoning. Built for production environments, it combines drift detection, autonomous decision-making, and automated retraining in a complete MLOps pipeline.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc)](https://dvc.org/)

---

## 🌟 Key Features

### 🤖 Agentic AI System
- **LLM-Powered Reasoning**: Uses OpenRouter (Liquid LFM 2.5) for intelligent drift response decisions
- **Contextual Decision Making**: Considers retraining cost, deployment risk, time constraints, and current accuracy
- **Multi-Agent Architecture**: Drift Detector → Reasoning Engine → Planning Agent → Execution
- **Policy-Based Fallback**: Rule-based reasoning when LLM unavailable

### 📊 Drift Detection
- **PSI (Population Stability Index)** monitoring across all features
- **Feature-level drift analysis** with configurable thresholds
- **Automated drift logging** to MLflow for historical tracking
- **Real-time drift visualization** in dashboard

### 🔄 Automated Retraining Pipeline
- **Digital Twin Simulator**: Test retraining strategies without production impact
- **Validation Gating**: Models must pass accuracy thresholds before deployment
- **Baseline Data Versioning**: DVC-based data accumulation across retraining cycles
- **MLflow Integration**: Complete experiment tracking and model registry

### 📈 Production-Ready Features
- **DVC Data Versioning**: Automatic baseline versioning after each retrain
- **Comprehensive Logging**: All decisions and executions logged to MLflow
- **RESTful API**: FastAPI backend with OpenAPI documentation
- **Interactive Dashboard**: Streamlit UI for monitoring and control
- **Deployment Safety**: Validation-based deployment gating

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DRIFTCATCHER SYSTEM                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Streamlit  │◄────►│   FastAPI    │◄────►│    MLflow    │
│   Dashboard  │      │      API     │      │   Tracking   │
│  (Port 8501) │      │  (Port 8000) │      │  (Port 5001) │
└──────────────┘      └──────┬───────┘      └──────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
         ┌──────▼─────┐ ┌────▼───────┐ ┌──▼─────────┐
         │   Drift    │ │  Reasoning │ │  Planning  │
         │  Detector  │ │   Engine   │ │   Agent    │
         │    (PSI)   │ │   (LLM)    │ │  (Tools)   │
         └────────────┘ └────────────┘ └──────┬─────┘
                                              │
                    ┌─────────────────────────┼─────────────┐
                    │                         │             │
             ┌──────▼──────┐         ┌────────▼────┐  ┌─────▼─────┐
             │   Retrain   │         │  Validate   │  │  Deploy   │
             │    Model    │────────►│   Model     │─►│   Model   │
             └──────┬──────┘         └─────────────┘  └───────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  DVC Baseline       │
         │  Versioning         │
         │  (Git + DVC Cache)  │
         └─────────────────────┘
```

### Components

1. **Drift Detector** (`monitoring/drift_detector.py`)
   - Calculates PSI for each feature
   - Identifies drifted features above threshold
   - Logs results to MLflow `drift_monitoring` experiment

2. **Reasoning Engine** (`agent/AgenticReasoningEngine.py`)
   - LLM-based decision making (OpenRouter API)
   - Considers: drift severity, costs, risks, timing
   - Actions: MONITOR, RETRAIN, RETRAIN_URGENT
   - Logs to MLflow `agentic_reasoning` experiment

3. **Planning Agent** (`agent/PlanningAgent.py`)
   - Creates multi-step execution plans
   - Tools: check_drift, retrain_model, validate_model, deploy_model
   - Handles dependencies and error recovery
   - Logs to MLflow `agentic_planning` experiment

4. **FastAPI Backend** (`api/main.py`)
   - RESTful API endpoints
   - Digital twin simulation
   - Model management (reload, info)
   - MLflow experiment queries

5. **Streamlit Dashboard** (`dashboard/app.py`)
   - Real-time monitoring
   - Drift visualization
   - Agent activity tracking
   - Digital twin simulator

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- Git with DVC support
- OpenRouter API key (optional, for LLM reasoning)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DriftCatcher.git
cd DriftCatcher

# Install dependencies with uv
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY (optional)

# Pull DVC-tracked data
uv run dvc pull

# Train initial model
uv run python training/train.py --base-data data/baseline.csv
```

### Running the System

**Terminal 1: MLflow Tracking Server**
```bash
uv run mlflow ui
# Access at http://localhost:5001
```

**Terminal 2: FastAPI Backend**
```bash
uv run python api/main.py
# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Terminal 3: Streamlit Dashboard**
```bash
uv run streamlit run dashboard/app.py
# Access at http://localhost:8501
```

---

## 🐳 Docker Deployment (Recommended)

### Pull from Docker Hub

```bash
# Pull the latest image
docker pull yourusername/driftcatcher:latest

# Or build locally
docker build -t driftcatcher:latest -f docker/Dockerfile .
```

### Quick Start with Docker Compose

```bash
# Navigate to docker directory
cd docker

# Set your OpenRouter API key (required for LLM reasoning)
export OPENROUTER_API_KEY="sk-or-v1-your-api-key-here"

# Start all services (API, Dashboard, MLflow, PostgreSQL)
docker compose up -d

# Check service status
docker ps
```

**Access the services:**
- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000 (Docs at http://localhost:8000/docs)
- **MLflow UI**: http://localhost:5001

### Memory Configuration

The Docker setup is optimized for memory efficiency:

```yaml
# docker/docker-compose.yml
services:
  driftcatcher-api:
    deploy:
      resources:
        limits:
          memory: 6G      # Maximum memory
        reservations:
          memory: 3G      # Reserved memory
  
  driftcatcher-dashboard:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

**Adjust limits** if working with larger datasets:
```bash
# Edit docker/docker-compose.yml
# Increase memory limits under deploy.resources.limits.memory

# Restart services
docker compose down
docker compose up -d
```

### Initial Dataset Setup

The container includes a sampled baseline (50k rows) for demo purposes. To use your own dataset:

```bash
# 1. Copy your dataset into the container
docker cp /path/to/your/baseline.csv docker-driftcatcher-api-1:/app/data/baseline.csv

# 2. Verify the dataset
docker exec docker-driftcatcher-api-1 ls -lh /app/data/baseline.csv

# 3. Train initial model (or use the Digital Twin Simulator in the dashboard)
docker exec docker-driftcatcher-api-1 bash -c "cd /app && uv run python training/train_universal.py --base-data data/baseline.csv"
```

### Persisting Data

**Mount volumes** to persist data between container restarts:

```yaml
# docker/docker-compose.yml (already configured)
volumes:
  - ./data:/app/data          # Your datasets
  - ./mlruns:/app/mlruns      # MLflow experiments
  - postgres_data:/var/lib/postgresql/data  # MLflow backend
```

Data is stored in:
- `./data/` - Baseline datasets and uploads
- `./mlruns/` - MLflow experiment tracking
- `postgres_data` - MLflow metadata (Docker volume)

---

## 📊 Working with Different Datasets

DriftCatcher is **dataset agnostic** and works with any tabular CSV data. Here's how to configure it for your dataset:

### Dataset Configuration

Create or modify a dataset config in `config/dataset_config.py`:

```python
from config.dataset_config import DatasetConfig, DATASETS

# Add your dataset configuration
DATASETS["YourDataset"] = DatasetConfig(
    name="YourDataset",
    label_column="target",           # Your target/label column name
    feature_columns=None,            # None = auto-detect all numeric columns
    binary_classification=True,      # True for binary, False for multi-class
    description="Your dataset description"
)
```

**Auto-Detection Features:**
- **Label Column**: Automatically looks for 'Label', 'label', 'target', 'class', 'y'
- **Feature Columns**: All numeric columns except the label
- **Binary Classification**: Converts labels to 0/1 automatically

### Using Your Dataset

**Option 1: Via Docker Dashboard (Recommended)**

1. Navigate to **Digital Twin Simulator** tab
2. Upload your CSV file (any structure)
3. Click **Run Digital Twin Simulation**
4. The system will:
   - Auto-detect label and features
   - Detect drift
   - Make retraining decisions
   - Validate and deploy models

**Option 2: Via Training Script**

```bash
# Using Docker
docker exec docker-driftcatcher-api-1 bash -c "cd /app && \
  uv run python training/train_universal.py \
    --base-data data/your_baseline.csv \
    --dataset YourDataset"

# Or locally
uv run python training/train_universal.py \
  --base-data data/your_baseline.csv \
  --new-data data/your_new_data.csv \
  --dataset YourDataset
```

**Option 3: Via API**

```bash
curl -X POST "http://localhost:8000/agent/digital-twin" \
  -F "file=@your_data.csv"
```

### Dataset Requirements

Your CSV should have:
- **Numeric features**: Most ML-relevant columns
- **Label column**: Target variable for classification
- **Header row**: Column names in the first row
- **No missing critical data**: Clean or handle NaNs beforehand

**Example CSV structure:**
```csv
feature1,feature2,feature3,...,target
1.2,0.5,3.4,...,0
0.8,1.2,2.1,...,1
...
```

### Memory-Efficient Training

For large datasets (>100k rows), the system automatically:
- Samples each file to 15k rows before combining
- Maintains baseline at 50k rows maximum
- Uses chunked processing for PSI calculation

**Manual sampling** if needed:
```bash
docker exec docker-driftcatcher-api-1 bash -c "cd /app && uv run python -c \"
import pandas as pd
df = pd.read_csv('data/your_large_file.csv')
sampled = df.sample(n=50000, random_state=42)
sampled.to_csv('data/your_large_file_sampled.csv', index=False)
print(f'Sampled {len(df)} to {len(sampled)} rows')
\""
```

### Built-in Dataset Configs

```python
# config/dataset_config.py
DATASETS = {
    "CICIDS2017": DatasetConfig(
        name="CICIDS2017",
        label_column="Label",
        feature_columns=None,  # Auto-detect 78 features
        binary_classification=True
    ),
    "Generic": DatasetConfig(
        name="Generic",
        label_column=None,     # Auto-detect
        feature_columns=None,  # Auto-detect
        binary_classification=True
    )
}
```

Use `"Generic"` for any dataset with auto-detection:
```bash
uv run python training/train_universal.py \
  --base-data data/your_data.csv \
  --dataset Generic
```

---

## 🧠 Machine Learning Pipeline

### Model Architecture

**Algorithm**: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Max Depth**: Unlimited (trees grow until leaves are pure)
- **Random State**: 42 (reproducibility)
- **Parallelization**: All available CPU cores (`n_jobs=-1`)

**Why Random Forest?**
- Handles high-dimensional network traffic features (78 features)
- Robust to outliers and missing values
- Provides feature importance for drift analysis
- Excellent performance on CICIDS2017 dataset (95%+ accuracy)
- Low false positive rate critical for network security

### Training Process

#### 1. Data Loading & Cleaning

```python
# training/train.py
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    
    # Critical: Strip whitespace from CICIDS column names
    df.columns = df.columns.str.strip()
    
    # Drop rows with missing labels
    df = df.dropna(subset=["Label"])
    
    # Binary classification: BENIGN (0) vs ATTACK (1)
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    
    return df
```

**Feature Engineering**:
- 78 network flow features extracted by CICFlowMeter
- Flow Duration, Packet Length Statistics, Inter-Arrival Times
- Protocol flags, Header lengths, Flow IAT metrics
- Forward/Backward segment sizes, subflow metrics

#### 2. Feature Preprocessing

```python
def clean_numeric_features(df, feature_cols):
    # Replace infinite values with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN in numeric features
    df = df.dropna(subset=feature_cols)
    
    return df
```

**Handling Edge Cases**:
- Infinite values from division by zero (e.g., packets per second with 0 duration)
- Missing values from incomplete network captures
- Maintains data integrity for accurate drift detection

#### 3. Training Statistics Computation

```python
def compute_training_stats(df, feature_cols):
    stats = {}
    
    for col in feature_cols:
        values = df[col].values
        
        # Compute decile quantiles for binning (PSI calculation)
        quantiles = np.quantile(values, q=np.linspace(0, 1, 11))
        
        # Compute actual distribution in training data
        hist, _ = np.histogram(values, bins=quantiles)
        expected_dist = (hist / hist.sum()).tolist()
        
        stats[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "quantiles": quantiles.tolist(),
            "expected_dist": expected_dist  # For PSI drift detection
        }
    
    return stats
```

**Saved to**: `artifacts/training_stats.json` (used by drift detector)

#### 4. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80% train, 20% test
    stratify=y,         # Maintain class balance
    random_state=42     # Reproducibility
)
```

**Stratification**: Critical for imbalanced datasets (more benign than attack traffic)

#### 5. Model Training & MLflow Logging

```python
mlflow.set_experiment('Model Training')

with mlflow.start_run():
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Log comprehensive metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    mlflow.log_metric("true_positives", tp)
    mlflow.log_metric("false_positives", fp)
    mlflow.log_metric("true_negatives", tn)
    mlflow.log_metric("false_negatives", fn)
    
    # Save model to MLflow registry
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="ddos_random_forest"
    )
    
    # Also save to artifacts/ for API
    pickle.dump(model, open("artifacts/model.pkl", "wb"))
```

**Typical Performance**:
- **Accuracy**: 95-98%
- **Precision**: 96-99% (low false positives)
- **Recall**: 94-97% (catches most attacks)
- **F1-Score**: 95-97%
- **ROC-AUC**: 0.98-0.99

#### 6. Retraining with Combined Data

```bash
# Retrain with base + new production data
uv run python training/train.py \
  --base-data data/baseline.csv \
  --new-data data/uploads/monday_traffic.csv data/uploads/tuesday_traffic.csv

# Process:
# 1. Load baseline.csv (existing training data)
# 2. Load each new CSV from production
# 3. Concatenate all datasets
# 4. Retrain model on combined data
# 5. Update baseline.csv with combined data (via DVC)
```

### Holdout Validation Set

**Critical for production**: Never use test set for validation during retraining. Create a separate holdout set.

#### Creating the Holdout Set

```bash
# Generate comprehensive holdout from all CICIDS CSVs
uv run python preprocessing/create_holdout.py \
  --data-dir data/raw \
  --output data/processed/holdout.csv \
  --sample-fraction 0.15
```

#### Holdout Generation Strategy

```python
# preprocessing/create_holdout.py

1. Sample from ALL 8 CICIDS2017 CSVs
   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   - Tuesday-WorkingHours.pcap_ISCX.csv
   - Wednesday-workingHours.pcap_ISCX.csv
   - Monday-WorkingHours.pcap_ISCX.csv
   - Friday-WorkingHours-Morning.pcap_ISCX.csv

2. Stratified Sampling (15% from each file)
   - Maintains original label distribution within each file
   - Ensures attack type diversity (DDoS, PortScan, Web Attacks, etc.)
   - Prevents class imbalance in holdout

3. Data Cleaning
   - Replace inf with NaN (from division by zero in feature extraction)
   - Drop rows with NaN values in numeric columns
   - Typically removes ~2-3% of data
   - Results in clean validation set without edge cases

4. Shuffling
   - Mix samples from different days/scenarios
   - Prevents temporal bias
   - Ensures diverse mini-batches if used in batched validation
```

#### Holdout Characteristics

**Size**: ~15% × 8 files = ~200,000-300,000 samples (depending on CSV sizes)

**Label Distribution** (example):
```
BENIGN:           180,000 (70%)
DDoS:              45,000 (18%)
PortScan:          15,000 (6%)
Web Attack XSS:     8,000 (3%)
Infiltration:       5,000 (2%)
Other attacks:      2,000 (1%)
```

**Key Properties**:
- ✅ **Diverse**: Represents all attack types from all days
- ✅ **Balanced**: Maintains realistic class distribution
- ✅ **Clean**: No inf/NaN values that could cause validation errors
- ✅ **Independent**: Never seen during training
- ✅ **Shuffled**: Mixed temporal patterns
- ✅ **Stratified**: Each attack type properly represented

#### Validation During Retraining

```python
# agent/PlanningAgent.py - _tool_validate_model()

def _tool_validate_model(self, holdout_path: str) -> Dict:
    # Load newly trained model
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load holdout dataset
    holdout = pd.read_csv(holdout_path)
    holdout.columns = holdout.columns.str.strip()
    
    # Prepare features
    feature_cols = [c for c in holdout.columns 
                   if c != 'Label' and holdout[c].dtype != 'object']
    
    X_holdout = holdout[feature_cols]
    y_holdout = holdout['Label']
    
    # Predict
    y_pred = model.predict(X_holdout)
    
    # Calculate metrics
    accuracy = accuracy_score(y_holdout, y_pred)
    precision = precision_score(y_holdout, y_pred)
    recall = recall_score(y_holdout, y_pred)
    f1 = f1_score(y_holdout, y_pred)
    
    # Validation threshold: 90% accuracy
    validation_passed = accuracy >= 0.90
    
    return {
        'validation_passed': validation_passed,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

**Validation Gating**:
- Model must achieve ≥90% accuracy on holdout before deployment
- Prevents degraded models from reaching production
- Dashboard shows validation metrics and blocks deployment on failure

**Deployment Decision Flow**:
```
Retrain → Validate on Holdout → Check Accuracy
                                     ↓
                          Accuracy ≥ 90%? 
                          ↙         ↘
                        YES          NO
                         ↓            ↓
                    Deploy to    Block Deployment
                    Production   Show Error in UI
```

---

## 📖 Usage

### 1. Check Drift via API

```bash
curl -X POST "http://localhost:8000/drift/check" \
  -F "file=@data/new_traffic.csv"
```

### 2. Run Digital Twin Simulation

```python
import requests

with open('data/new_traffic.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/agent/digital-twin',
        files={'file': f}
    )
    
result = response.json()
print(f"Action: {result['simulation']['steps']['2_reasoning']['action']}")
```

### 3. Monitor via Dashboard

1. Navigate to http://localhost:8501
2. Go to "🎭 Digital Twin Simulator" tab
3. Upload CSV file with new traffic data
4. Click "Run Digital Twin Simulation"
5. Review drift detection, reasoning, and execution results

### 4. Check MLflow Experiments

Navigate to http://localhost:5001 to view:
- **drift_monitoring**: All drift detection runs
- **agentic_reasoning**: LLM reasoning decisions
- **agentic_planning**: Plan executions
- **Model Training**: Training runs and metrics

---

## 🏭 Production Deployment

### Architecture Options

#### Option 1: Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: driftcatcher-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: driftcatcher-api
  template:
    metadata:
      labels:
        app: driftcatcher-api
    spec:
      containers:
      - name: api
        image: driftcatcher:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: driftcatcher-secrets
              key: openrouter-api-key
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5001"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: driftcatcher-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: driftcatcher-api
```

#### Option 2: Docker Compose (Single Server)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5001:5000"
    volumes:
      - mlflow-data:/mlflow
      - ./artifacts:/artifacts
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/artifacts
    command: mlflow server --host 0.0.0.0 --port 5000

  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
      - dvc-cache:/app/.dvc/cache
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000

  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
      - MLFLOW_BASE_URL=http://mlflow:5000
    depends_on:
      - api
    command: streamlit run dashboard/app.py --server.port 8501

volumes:
  mlflow-data:
  dvc-cache:
```

### Production Checklist

#### 1. **Data Versioning Setup**

```bash
# Configure DVC remote (S3 example)
dvc remote add -d production s3://driftcatcher-prod/data
dvc remote modify production region us-west-2
dvc remote modify production access_key_id ${AWS_ACCESS_KEY}
dvc remote modify production secret_access_key ${AWS_SECRET_KEY}

# Enable auto-staging
dvc config core.autostage true

# Push initial baseline
dvc push
```

#### 2. **MLflow Remote Tracking**

```python
# Set remote MLflow server
export MLFLOW_TRACKING_URI=https://mlflow.yourcompany.com

# Or use managed service
export MLFLOW_TRACKING_URI=databricks://your-workspace
```

#### 3. **Secret Management**

```bash
# Using Kubernetes secrets
kubectl create secret generic driftcatcher-secrets \
  --from-literal=openrouter-api-key=${OPENROUTER_API_KEY} \
  --from-literal=aws-access-key=${AWS_ACCESS_KEY} \
  --from-literal=aws-secret-key=${AWS_SECRET_KEY}

# Using AWS Secrets Manager
aws secretsmanager create-secret \
  --name driftcatcher/openrouter \
  --secret-string "${OPENROUTER_API_KEY}"
```

#### 4. **Monitoring & Alerting**

```yaml
# prometheus/alerts.yml
groups:
- name: driftcatcher_alerts
  rules:
  - alert: HighDriftDetected
    expr: drift_psi_score > 0.2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High drift detected (PSI > 0.2)"
      
  - alert: ModelRetrainingFailed
    expr: retrain_failure_count > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Model retraining failed"
      
  - alert: ValidationAccuracyLow
    expr: validation_accuracy < 0.90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Model validation accuracy below threshold"
```

#### 5. **CI/CD Pipeline**

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run tests
        run: uv run pytest tests/
      
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t driftcatcher:${{ github.sha }} .
      - name: Push to registry
        run: |
          docker tag driftcatcher:${{ github.sha }} your-registry/driftcatcher:latest
          docker push your-registry/driftcatcher:latest
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/driftcatcher-api \
            api=your-registry/driftcatcher:${{ github.sha }}
```

#### 6. **Automated Retraining Schedule**

```python
# scheduler/retrain_job.py
from apscheduler.schedulers.blocking import BlockingScheduler
import requests

def check_and_retrain():
    """Scheduled job to check drift and trigger retraining"""
    # Check latest production data
    response = requests.get('http://api:8000/drift/latest')
    drift = response.json()
    
    if drift['overall_psi'] > 0.2:
        # Trigger digital twin
        with open('/data/production/latest.csv', 'rb') as f:
            requests.post(
                'http://api:8000/agent/digital-twin',
                files={'file': f}
            )

scheduler = BlockingScheduler()
scheduler.add_job(check_and_retrain, 'cron', hour=2, minute=0)  # Daily at 2 AM
scheduler.start()
```

#### 7. **Backup & Disaster Recovery**

```bash
# Backup script
#!/bin/bash

# Backup MLflow database
sqlite3 mlruns/mlflow.db ".backup '/backup/mlflow-$(date +%Y%m%d).db'"

# Backup DVC cache
aws s3 sync .dvc/cache s3://driftcatcher-backup/dvc-cache/

# Backup model artifacts
aws s3 sync artifacts/ s3://driftcatcher-backup/artifacts/

# Backup baseline versions
git push backup main
dvc push backup
```

### Production Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| Drift PSI | > 0.2 | Alert + consider retraining |
| Model Accuracy | < 90% | Block deployment |
| API Latency (p95) | > 500ms | Scale up |
| Retraining Duration | > 30min | Optimize pipeline |
| Validation Failures | > 2 consecutive | Manual review |
| LLM Response Time | > 5s | Use fallback policy |

### Scaling Considerations

**Horizontal Scaling:**
- API: 3-5 replicas behind load balancer
- Dashboard: 2-3 replicas for high availability
- MLflow: Single instance (or managed service)

**Vertical Scaling:**
- API: 2-4 GB RAM, 1-2 CPU cores
- Training: 8-16 GB RAM, 4-8 CPU cores (or GPU)
- MLflow: 4-8 GB RAM, 2-4 CPU cores

**Storage:**
- DVC Cache: S3/GCS with lifecycle policies
- MLflow: PostgreSQL for production (not SQLite)
- Artifacts: S3/GCS with versioning enabled

---

## 📦 DVC Data Versioning

DriftCatcher uses DVC to automatically version baseline training data:

```bash
# View baseline version history
git log --oneline data/baseline.csv.dvc

# Rollback to previous baseline
git checkout <commit-hash> data/baseline.csv.dvc
dvc checkout data/baseline.csv

# Push new baseline version
dvc add data/baseline.csv
git add data/baseline.csv.dvc
git commit -m "Update baseline after production retrain"
dvc push
```

**How it works:**
1. After successful retraining, combined data saved as new baseline
2. `dvc add` automatically versions the updated baseline
3. Git tracks tiny .dvc file (96 bytes), DVC manages large CSV
4. Next retrain uses latest baseline → continuous learning!

See [docs/DVC_VERSIONING.md](docs/DVC_VERSIONING.md) for detailed workflow.

---

## 📊 MLflow Experiments

### Drift Monitoring
- **Metrics**: overall_psi, n_drifted_features, total_features
- **Params**: threshold, method
- **Tags**: data_source, timestamp

### Agentic Reasoning
- **Metrics**: confidence_score
- **Params**: action, drift_psi, model
- **Tags**: reasoning_type, context

### Agentic Planning
- **Metrics**: total_steps, completed_steps, failed_steps
- **Params**: goal, urgency
- **Tags**: plan_id, execution_status

### Model Training
- **Metrics**: accuracy, precision, recall, f1_score, training_samples
- **Params**: base_data_path, n_estimators, max_depth
- **Artifacts**: model.pkl, training_stats.json

---

## 🔧 Configuration

### Environment Variables

```bash
# .env
OPENROUTER_API_KEY=your_key_here          # LLM reasoning (optional)
MLFLOW_TRACKING_URI=http://localhost:5001 # MLflow server
API_BASE_URL=http://localhost:8000        # FastAPI
```

### System Configuration

```python
# config/settings.py
DRIFT_THRESHOLDS = {
    'low': 0.1,    # PSI < 0.1: No drift
    'high': 0.2    # PSI > 0.2: Severe drift
}

VALIDATION_THRESHOLDS = {
    'accuracy': 0.90,
    'precision': 0.85,
    'recall': 0.85,
    'f1_score': 0.85
}

RETRAINING_SCHEDULE = {
    'check_interval': '1h',
    'max_frequency': 'daily',
    'maintenance_window': '02:00-04:00'
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
uv sync --all-extras

# Install pre-commit hooks
pre-commit install

# Run linters
uv run ruff check .
uv run black .
uv run mypy .
```

---

## 📚 Documentation

- [DVC Versioning Workflow](docs/DVC_VERSIONING.md)
- [DVC Workflow Diagrams](docs/DVC_WORKFLOW_DIAGRAM.md)
- [API Documentation](http://localhost:8000/docs)
- [Architecture Deep Dive](docs/)

---

## 🛣️ Roadmap

- [ ] A/B testing infrastructure
- [ ] Multi-model ensemble support
- [ ] Automated hyperparameter tuning
- [ ] Real-time streaming drift detection
- [ ] Custom LLM fine-tuning on decision history
- [ ] Integration with popular ML platforms (SageMaker, Vertex AI)
- [ ] Advanced deployment strategies (blue-green, canary)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [MLflow](https://mlflow.org/) for experiment tracking
- Data versioning with [DVC](https://dvc.org/)
- LLM reasoning via [OpenRouter](https://openrouter.ai/)
- Dashboard powered by [Streamlit](https://streamlit.io/)
- API built with [FastAPI](https://fastapi.tiangolo.com/)

---

**Built for production ML teams who want autonomous, intelligent drift management.** 🚀