# DriftCatcher - Low Level Design (LLD)

**Version**: 1.0  
**Purpose**: Hackathon build guide for parallel team development  
**Goal**: Zero merge conflicts through strict module separation

---

## 🎯 System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    DriftCatcher System                            │
│                                                                   │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐              │
│  │  Frontend  │◄─►│  Backend   │◄─►│   MLflow   │              │
│  │ (Streamlit)│   │  (FastAPI) │   │  Tracking  │              │
│  └────────────┘   └─────┬──────┘   └────────────┘              │
│                          │                                        │
│         ┌────────────────┼────────────────┐                     │
│         │                │                │                      │
│    ┌────▼─────┐   ┌─────▼──────┐  ┌──────▼──────┐             │
│    │  Drift   │   │ Reasoning  │  │  Planning   │             │
│    │ Detector │   │  Engine    │  │   Agent     │             │
│    └──────────┘   └────────────┘  └──────┬──────┘             │
│                                           │                      │
│                    ┌──────────────────────┼──────┐              │
│                    │                      │      │              │
│             ┌──────▼──────┐        ┌─────▼──┐  ┌▼──────┐      │
│             │   Training  │        │  DVC   │  │MLflow │      │
│             │   Module    │        │  Data  │  │Model  │      │
│             └─────────────┘        └────────┘  └───────┘      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Module Structure (Team Assignment)

### Module 1: Data Layer (Person A)
**Folder**: `data/` + `preprocessing/`  
**Responsibility**: Data handling, preprocessing, DVC integration

```
data/
├── raw/                    # Raw CICIDS CSVs (DVC tracked)
├── processed/             # Processed datasets
│   └── holdout.csv       # Validation set
├── uploads/              # User uploaded files
├── baseline.csv          # DVC tracked baseline
└── baseline.csv.dvc      # DVC metadata

preprocessing/
├── __init__.py
├── data_loader.py        # Load & clean CSVs
├── feature_engineer.py   # Feature extraction
└── create_holdout.py     # Generate holdout set
```

**Interface Contract**:
```python
# preprocessing/data_loader.py

class DataLoader:
    """Loads and cleans CICIDS2017 data"""
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Load CSV with cleaned columns
        Returns: DataFrame with stripped columns, dropped NaN labels
        """
        pass
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Returns list of numeric feature columns (excluding Label)"""
        pass
    
    def clean_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf with NaN, drop rows with NaN"""
        pass
    
    def binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert labels to binary: BENIGN=0, ATTACK=1"""
        pass
```

**Key Files**:
- `preprocessing/data_loader.py` - CSV loading
- `preprocessing/create_holdout.py` - Holdout generation
- `data/.gitignore` - Ignore patterns
- `.dvc/config` - DVC configuration

---

### Module 2: Training Pipeline (Person B)
**Folder**: `training/`  
**Responsibility**: Model training, statistics, MLflow logging

```
training/
├── __init__.py
├── train.py              # Main training script
├── model_builder.py      # Model instantiation
└── stats_computer.py     # Training statistics for drift
```

**Interface Contract**:
```python
# training/model_builder.py

class ModelBuilder:
    """Builds and trains Random Forest model"""
    
    def build_model(self) -> RandomForestClassifier:
        """Returns configured RF model"""
        pass
    
    def train(self, X_train, y_train) -> RandomForestClassifier:
        """Trains model, returns fitted model"""
        pass
    
    def evaluate(self, model, X_test, y_test) -> Dict:
        """
        Returns metrics dict:
        {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1_score': float,
            'roc_auc': float,
            'confusion_matrix': [[tn, fp], [fn, tp]]
        }
        """
        pass

# training/stats_computer.py

class StatsComputer:
    """Computes training statistics for drift detection"""
    
    def compute_stats(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """
        Computes per-feature statistics
        Returns: {
            'feature_name': {
                'mean': float,
                'std': float,
                'quantiles': List[float],  # 11 quantiles (deciles)
                'expected_dist': List[float]  # Distribution for PSI
            }
        }
        """
        pass
    
    def save_stats(self, stats: Dict, path: str = "artifacts/training_stats.json"):
        """Save stats to JSON file"""
        pass
```

**Key Files**:
- `training/train.py` - CLI entry point
- `training/model_builder.py` - Model logic
- `training/stats_computer.py` - Statistics computation
- `artifacts/model.pkl` - Saved model (output)
- `artifacts/training_stats.json` - Stats (output)

---

### Module 3: Drift Detection (Person C)
**Folder**: `monitoring/`  
**Responsibility**: PSI calculation, drift detection, MLflow logging

```
monitoring/
├── __init__.py
├── drift_detector.py     # Main drift detection
├── psi_calculator.py     # PSI computation
└── mlflow_logger.py      # MLflow drift logging
```

**Interface Contract**:
```python
# monitoring/drift_detector.py

class DriftDetector:
    """Detects drift using PSI"""
    
    def __init__(self, stats_path: str = "artifacts/training_stats.json"):
        """Load training statistics"""
        pass
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict:
        """
        Calculate drift for new data
        Returns: {
            'overall_psi': float,
            'n_drifted_features': int,
            'total_features': int,
            'feature_psi': Dict[str, float],
            'drifted_features': List[Tuple[str, float]],
            'top_drifted_features': List[Tuple[str, float]]  # Top 10
        }
        """
        pass

# monitoring/psi_calculator.py

class PSICalculator:
    """Calculates Population Stability Index"""
    
    def calculate_psi(
        self, 
        expected_dist: List[float], 
        actual_dist: List[float]
    ) -> float:
        """
        Calculate PSI between two distributions
        PSI = Σ (actual - expected) * ln(actual / expected)
        """
        pass
    
    def bin_data(self, values: np.ndarray, quantiles: List[float]) -> List[float]:
        """Bin data into quantile buckets, return distribution"""
        pass
```

**Key Files**:
- `monitoring/drift_detector.py` - Main detector class
- `monitoring/psi_calculator.py` - PSI math
- `monitoring/mlflow_logger.py` - MLflow integration

**Dependencies**:
- Reads: `artifacts/training_stats.json` (from Person B)
- Writes: MLflow experiment `drift_monitoring`

---

### Module 4: Reasoning Engine (Person D)
**Folder**: `agent/`  
**Responsibility**: LLM reasoning, action decisions, context handling

```
agent/
├── __init__.py
├── AgenticReasoningEngine.py   # Main reasoning
├── llm_client.py                # OpenRouter API
├── rule_engine.py               # Fallback rules
└── logs/
    └── reasoning_decisions.jsonl
```

**Interface Contract**:
```python
# agent/AgenticReasoningEngine.py

class AgenticReasoningEngine:
    """Makes drift response decisions"""
    
    def __init__(self, model: str = "liquid/lfm-2.5-1.2b-instruct:free"):
        """Initialize with LLM model or rule-based fallback"""
        pass
    
    def reason_about_action(
        self, 
        drift_results: Dict,  # From DriftDetector
        context: Dict         # Operational context
    ) -> Dict:
        """
        Decide action based on drift and context
        
        Args:
            drift_results: Output from DriftDetector.detect_drift()
            context: {
                'time_since_last_retrain': str,
                'retraining_cost': str,  # 'low', 'medium', 'high'
                'deployment_risk': str,
                'current_accuracy': float
            }
        
        Returns: {
            'action': str,  # 'MONITOR', 'RETRAIN', 'RETRAIN_URGENT'
            'confidence': float,
            'reasoning': str,
            'risk_assessment': str,
            'context_considered': Dict
        }
        """
        pass

# agent/llm_client.py

class LLMClient:
    """OpenRouter API client"""
    
    def __init__(self, api_key: str, model: str):
        pass
    
    def chat_completion(self, system: str, user: str) -> str:
        """Send chat completion request, return response text"""
        pass
```

**Key Files**:
- `agent/AgenticReasoningEngine.py` - Decision logic
- `agent/llm_client.py` - API wrapper
- `agent/rule_engine.py` - Fallback logic
- `agent/logs/reasoning_decisions.jsonl` - Decision log (output)

**Dependencies**:
- Reads: Drift results from Person C
- Writes: MLflow experiment `agentic_reasoning`

---

### Module 5: Planning Agent (Person E)
**Folder**: `agent/`  
**Responsibility**: Multi-step planning, tool execution, validation

```
agent/
├── PlanningAgent.py         # Main planner
├── tools/
│   ├── __init__.py
│   ├── retrain_tool.py      # Retraining
│   ├── validate_tool.py     # Validation
│   └── deploy_tool.py       # Deployment
└── logs/
    ├── planning_plans.jsonl
    └── planning_executions.jsonl
```

**Interface Contract**:
```python
# agent/PlanningAgent.py

class PlanStep:
    step_id: int
    tool_name: str
    description: str
    params: Dict
    dependencies: List[int]
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    result: Optional[Dict]

class PlanningAgent:
    """Creates and executes multi-step plans"""
    
    def create_plan(self, goal: str, context: Dict) -> List[PlanStep]:
        """
        Create execution plan based on goal
        
        Args:
            goal: 'RETRAIN' or 'RETRAIN_URGENT'
            context: {
                'latest_data_path': str,
                'base_data_path': str,
                'holdout_path': str,
                'target_accuracy': float,
                'urgency': str
            }
        
        Returns: List of PlanSteps
        """
        pass
    
    def execute_plan(self, plan: List[PlanStep], dry_run: bool = False) -> Dict:
        """
        Execute plan steps respecting dependencies
        
        Returns: {
            'plan_id': str,
            'total_steps': int,
            'completed_steps': int,
            'failed_steps': int,
            'step_results': List[Dict],
            'final_status': str
        }
        """
        pass

# agent/tools/retrain_tool.py

def retrain_model(base_data: str, new_data: List[str]) -> Dict:
    """
    Retrain model on combined data
    
    Returns: {
        'success': bool,
        'output': str,
        'baseline_updated': bool
    }
    """
    pass

# agent/tools/validate_tool.py

def validate_model(holdout_path: str) -> Dict:
    """
    Validate model on holdout set
    
    Returns: {
        'validation_passed': bool,
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1_score': float
    }
    """
    pass

# agent/tools/deploy_tool.py

def deploy_model(model_version: str, validation_result: Dict) -> Dict:
    """
    Deploy model to MLflow registry
    
    Returns: {
        'deployed': bool,
        'version': str,
        'model_name': str,
        'run_id': str,
        'reason': str  # If not deployed
    }
    """
    pass
```

**Key Files**:
- `agent/PlanningAgent.py` - Planning logic
- `agent/tools/retrain_tool.py` - Retraining
- `agent/tools/validate_tool.py` - Validation
- `agent/tools/deploy_tool.py` - Deployment
- `agent/logs/planning_*.jsonl` - Logs (output)

**Dependencies**:
- Reads: Reasoning decision from Person D
- Calls: Training from Person B
- Writes: MLflow experiment `agentic_planning`

---

### Module 6: FastAPI Backend (Person F)
**Folder**: `api/`  
**Responsibility**: REST API, endpoints, request handling

```
api/
├── __init__.py
├── main.py              # FastAPI app
├── routers/
│   ├── __init__.py
│   ├── drift.py         # Drift endpoints
│   ├── agent.py         # Agent endpoints
│   ├── model.py         # Model management
│   └── mlflow_proxy.py  # MLflow queries
├── models.py            # Pydantic schemas
└── dependencies.py      # Shared dependencies
```

**Interface Contract**:
```python
# api/models.py (Pydantic schemas)

class DriftCheckResponse(BaseModel):
    status: str
    drift_results: Dict
    timestamp: str

class DigitalTwinRequest(BaseModel):
    # File uploaded via multipart/form-data
    pass

class DigitalTwinResponse(BaseModel):
    status: str
    message: str
    simulation: Dict[str, Any]
    timestamp: str

class AgentStatusResponse(BaseModel):
    status: str
    reasoning_engine: Dict
    planning_agent: Dict

# api/routers/drift.py

@router.post("/drift/check")
async def check_drift(file: UploadFile) -> DriftCheckResponse:
    """
    Check drift on uploaded CSV
    1. Save uploaded file
    2. Call DriftDetector.detect_drift()
    3. Log to MLflow
    4. Return results
    """
    pass

# api/routers/agent.py

@router.post("/agent/digital-twin")
async def run_digital_twin(file: UploadFile) -> DigitalTwinResponse:
    """
    Run complete digital twin simulation
    1. Save file to data/uploads/
    2. Detect drift (Person C)
    3. Get reasoning decision (Person D)
    4. Create and execute plan (Person E)
    5. Return complete results
    """
    pass

@router.get("/agent/status")
async def get_agent_status() -> AgentStatusResponse:
    """Return agent status and learning summary"""
    pass

# api/routers/model.py

@router.get("/model/info")
async def get_model_info():
    """Return current model version and type"""
    pass

@router.post("/model/reload")
async def reload_model(source: str, version: str = "latest"):
    """Reload model from file or MLflow"""
    pass
```

**Key Files**:
- `api/main.py` - FastAPI app initialization
- `api/routers/drift.py` - Drift endpoints
- `api/routers/agent.py` - Agent endpoints
- `api/routers/model.py` - Model management
- `api/models.py` - Pydantic schemas

**Dependencies**:
- Calls: All modules (Person A-E)
- Port: 8000

---

### Module 7: Streamlit Dashboard (Person G)
**Folder**: `dashboard/`  
**Responsibility**: UI, visualization, monitoring

```
dashboard/
├── app.py               # Main dashboard
├── components/
│   ├── __init__.py
│   ├── drift_tab.py     # Drift monitoring tab
│   ├── performance_tab.py # Model performance
│   ├── agent_tab.py     # Agent activity
│   └── simulator_tab.py # Digital twin
└── utils.py             # API client helpers
```

**Interface Contract**:
```python
# dashboard/utils.py

class APIClient:
    """Client for FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        pass
    
    def check_drift(self, file) -> Dict:
        """Upload file, check drift"""
        pass
    
    def run_digital_twin(self, file) -> Dict:
        """Run digital twin simulation"""
        pass
    
    def get_agent_status(self) -> Dict:
        """Get agent status"""
        pass
    
    def get_mlflow_experiments(self) -> List[Dict]:
        """Get MLflow experiments"""
        pass
    
    def get_mlflow_runs(self, experiment: str, limit: int) -> List[Dict]:
        """Get runs for experiment"""
        pass

# dashboard/components/drift_tab.py

def render_drift_tab(api_client: APIClient):
    """
    Render drift monitoring tab
    - File uploader
    - Drift check button
    - PSI visualization
    - Feature-level drift table
    - History plot
    """
    pass

# dashboard/components/simulator_tab.py

def render_simulator_tab(api_client: APIClient):
    """
    Render digital twin simulator tab
    - File uploader
    - Run simulation button
    - 4-step results display:
      1. Drift detection
      2. Agent reasoning
      3. Plan creation
      4. Execution results
    """
    pass
```

**Key Files**:
- `dashboard/app.py` - Main entry point
- `dashboard/components/*.py` - Tab components
- `dashboard/utils.py` - API helpers

**Dependencies**:
- Calls: FastAPI backend (Person F)
- Port: 8501

---

## 🔌 Interface Definitions (Critical for Parallel Work)

### 1. Data Flow Interfaces

```python
# Person A → Person B
training_data: pd.DataFrame
# Columns: [78 numeric features] + ['Label']

# Person B → Person C  
training_stats: Dict[str, Dict[str, Any]]
# Saved to: artifacts/training_stats.json

# Person C → Person D
drift_results: Dict = {
    'overall_psi': float,
    'n_drifted_features': int,
    'total_features': int,
    'feature_psi': Dict[str, float],
    'drifted_features': List[Tuple[str, float]],
    'top_drifted_features': List[Tuple[str, float]]
}

# Person D → Person E
reasoning_decision: Dict = {
    'action': str,  # 'MONITOR', 'RETRAIN', 'RETRAIN_URGENT'
    'confidence': float,
    'reasoning': str,
    'risk_assessment': str,
    'context_considered': Dict
}

# Person E → Person F
plan_execution_result: Dict = {
    'plan_id': str,
    'total_steps': int,
    'completed_steps': int,
    'failed_steps': int,
    'step_results': List[Dict],
    'final_status': str
}
```

### 2. File I/O Contracts

```python
# Input Files (Created by setup script)
data/raw/*.csv                    # CICIDS2017 CSVs (Person A)
data/baseline.csv                 # Initial baseline (Person A)
.env                             # API keys (All)

# Intermediate Files (Must be created in order)
artifacts/training_stats.json    # Person B → Person C
artifacts/model.pkl              # Person B → Person E, F
data/processed/holdout.csv       # Person A → Person E
data/uploads/*.csv               # Person F → All (runtime)

# Output Files
agent/logs/*.jsonl               # Person D, E
mlruns/*                        # All (via MLflow)
```

### 3. Environment Variables

```bash
# .env (shared by all)
OPENROUTER_API_KEY=sk-or-...    # Person D
MLFLOW_TRACKING_URI=http://localhost:5001  # All
API_BASE_URL=http://localhost:8000  # Person G
```

---

## 🏗️ Development Workflow

### Phase 1: Setup (30 minutes)
**Everyone works together**

1. Clone repository
2. Create virtual environment: `uv sync`
3. Download CICIDS2017 dataset
4. Create `.env` file
5. Initialize DVC: `dvc init`
6. Create initial file structure (empty files)

### Phase 2: Parallel Development (4-6 hours)

#### Sprint 1: Core Components (2 hours)

**Person A**: Data layer
- Implement `DataLoader` class
- Create `create_holdout.py` script
- Run: Generate initial baseline and holdout
- **Deliverable**: `data/baseline.csv`, `data/processed/holdout.csv`

**Person B**: Training pipeline
- Implement `ModelBuilder` class
- Implement `StatsComputer` class
- Create `train.py` CLI
- Run: Train initial model
- **Deliverable**: `artifacts/model.pkl`, `artifacts/training_stats.json`

**Person C**: Drift detection (blocked by Person B's stats)
- Implement `PSICalculator` class
- Implement `DriftDetector` class
- Unit test with mock stats
- **Deliverable**: `monitoring/drift_detector.py`

**Person D**: Reasoning engine
- Implement `LLMClient` class
- Implement `AgenticReasoningEngine` class
- Unit test with mock drift results
- **Deliverable**: `agent/AgenticReasoningEngine.py`

**Person E**: Planning agent (blocked by Person B's training)
- Implement `PlanningAgent` class
- Implement tool functions
- Unit test with mock context
- **Deliverable**: `agent/PlanningAgent.py`

**Person F**: FastAPI backend (blocked by Person B, C, D, E)
- Create FastAPI app structure
- Implement Pydantic schemas
- Create placeholder endpoints (return mock data)
- **Deliverable**: `api/main.py` (runnable with mocks)

**Person G**: Streamlit dashboard (blocked by Person F)
- Create dashboard structure
- Implement API client
- Create tab components with mock data
- **Deliverable**: `dashboard/app.py` (runnable with mocks)

#### Sprint 2: Integration (2 hours)

**Person C**: 
- Replace mock stats with real `training_stats.json`
- Test with actual data
- Integrate MLflow logging

**Person D**:
- Replace mock drift with real drift results
- Test with Person C's output
- Integrate MLflow logging

**Person E**:
- Replace mocks with real tools
- Test retraining with Person B's train.py
- Test validation with holdout
- Integrate MLflow logging

**Person F**:
- Replace all mock endpoints with real integrations
- Wire up Person C, D, E modules
- Test complete digital twin flow

**Person G**:
- Wire up real API endpoints
- Test with Person F's backend
- Polish UI

#### Sprint 3: Testing & Polish (1-2 hours)

**Everyone**:
- End-to-end testing
- Bug fixes
- Documentation
- Demo preparation

---

## 🧪 Testing Strategy

### Unit Tests (Each person tests their module)

```python
# tests/test_data_loader.py (Person A)
def test_load_csv():
    loader = DataLoader()
    df = loader.load_csv("test_data.csv")
    assert 'Label' in df.columns
    assert df['Label'].isin([0, 1]).all()

# tests/test_drift_detector.py (Person C)
def test_detect_drift():
    detector = DriftDetector("mock_stats.json")
    results = detector.detect_drift(mock_df)
    assert 'overall_psi' in results
    assert results['overall_psi'] >= 0

# tests/test_reasoning_engine.py (Person D)
def test_reason_about_action():
    engine = AgenticReasoningEngine()
    decision = engine.reason_about_action(mock_drift, mock_context)
    assert decision['action'] in ['MONITOR', 'RETRAIN', 'RETRAIN_URGENT']
```

### Integration Tests (After Sprint 2)

```python
# tests/test_integration.py
def test_complete_flow():
    # 1. Upload file
    response = client.post("/drift/check", files={"file": test_csv})
    assert response.status_code == 200
    
    # 2. Digital twin
    response = client.post("/agent/digital-twin", files={"file": test_csv})
    assert response.status_code == 200
    assert 'simulation' in response.json()
```

---

## 🚫 Merge Conflict Prevention

### File Ownership Rules

| Person | Primary Files | Never Touch |
|--------|--------------|-------------|
| A | `preprocessing/*`, `data/*` | `training/*`, `agent/*`, `api/*`, `dashboard/*` |
| B | `training/*` | `preprocessing/*`, `agent/*`, `api/*`, `dashboard/*` |
| C | `monitoring/*` | `preprocessing/*`, `training/*`, `agent/*`, `api/*` |
| D | `agent/AgenticReasoningEngine.py`, `agent/llm_client.py` | `preprocessing/*`, `training/*`, `monitoring/*`, `api/*` |
| E | `agent/PlanningAgent.py`, `agent/tools/*` | `preprocessing/*`, `training/*`, `monitoring/*`, `api/*` |
| F | `api/*` | `preprocessing/*`, `training/*`, `agent/*`, `dashboard/*` |
| G | `dashboard/*` | All other folders |

### Git Workflow

```bash
# Each person works on their own branch
git checkout -b feature/data-layer        # Person A
git checkout -b feature/training          # Person B
git checkout -b feature/drift-detection   # Person C
git checkout -b feature/reasoning         # Person D
git checkout -b feature/planning          # Person E
git checkout -b feature/backend           # Person F
git checkout -b feature/dashboard         # Person G

# Commit frequently to your branch
git add <your-files-only>
git commit -m "feat: implement X"

# When ready to integrate (after Sprint 1)
git push origin feature/your-feature
# Create PR → merge to main
```

### Shared Files (Requires Coordination)

- `pyproject.toml` - **Person F** coordinates dependencies
- `README.md` - **Person F** writes final version
- `.env.example` - **Person F** maintains
- `.gitignore` - **Person A** maintains

---

## 📊 Module Dependency Graph

```
    [A: Data]
        ↓
    [B: Training] ─────────────┐
        ↓                      ↓
 [training_stats.json]    [model.pkl]
        ↓                      ↓
    [C: Drift]            [E: Planning]
        ↓                      ↑
   [drift_results]             │
        ↓                      │
    [D: Reasoning] ────────────┘
        ↓
   [decision]
        ↓
┌───[F: Backend]────┐
│                   │
↓                   ↓
[MLflow]      [G: Dashboard]
```

**Critical Path**: A → B → C → D → E → F → G

**Parallelizable**:
- A, B can work independently (both read CSVs)
- C, D, E can develop with mocks while waiting for B
- F, G can develop UI/API structure with mocks

---

## 🎯 Hackathon Timeline (8 hours)

```
Hour 0-1:   Setup, architecture review, task assignment
Hour 1-3:   Sprint 1 - Core components (with mocks)
Hour 3:     Quick sync - resolve blockers
Hour 3-5:   Sprint 2 - Integration (replace mocks)
Hour 5-6:   Sprint 3 - Testing & polish
Hour 6-7:   Demo preparation, slides
Hour 7-8:   Buffer for bugs, final touches
```

---

## 📝 Quick Reference: Mock Data

For development before integration, use these mocks:

```python
# Mock drift results (Person C's output)
MOCK_DRIFT = {
    'overall_psi': 0.25,
    'n_drifted_features': 15,
    'total_features': 78,
    'feature_psi': {'Flow Duration': 0.35, ...},
    'drifted_features': [('Flow Duration', 0.35), ...],
    'top_drifted_features': [('Flow Duration', 0.35), ...]
}

# Mock context (Person F builds this)
MOCK_CONTEXT = {
    'time_since_last_retrain': '7 days',
    'retraining_cost': 'medium',
    'deployment_risk': 'low',
    'current_accuracy': 0.95
}

# Mock reasoning decision (Person D's output)
MOCK_DECISION = {
    'action': 'RETRAIN',
    'confidence': 85.0,
    'reasoning': 'Moderate drift detected...',
    'risk_assessment': 'low',
    'context_considered': MOCK_CONTEXT
}

# Mock training stats (Person B's output)
MOCK_STATS = {
    'Flow Duration': {
        'mean': 12345.6,
        'std': 5678.9,
        'min': 0.0,
        'max': 999999.0,
        'quantiles': [0, 100, 200, ...],
        'expected_dist': [0.1, 0.1, 0.1, ...]
    }
}
```

---

## ✅ Definition of Done (Each Module)

### Person A (Data Layer)
- [ ] `DataLoader` class implemented
- [ ] `create_holdout.py` generates holdout set
- [ ] `data/baseline.csv` exists
- [ ] `data/processed/holdout.csv` exists
- [ ] Unit tests pass

### Person B (Training)
- [ ] `train.py` CLI works with `--base-data` and `--new-data`
- [ ] `artifacts/model.pkl` created
- [ ] `artifacts/training_stats.json` created
- [ ] MLflow experiment logs training run
- [ ] Unit tests pass

### Person C (Drift Detection)
- [ ] `DriftDetector` class calculates PSI correctly
- [ ] Works with real `training_stats.json`
- [ ] MLflow experiment logs drift runs
- [ ] Unit tests pass

### Person D (Reasoning)
- [ ] `AgenticReasoningEngine` makes decisions
- [ ] LLM integration works (with fallback)
- [ ] MLflow experiment logs reasoning runs
- [ ] Unit tests pass

### Person E (Planning)
- [ ] `PlanningAgent` creates valid plans
- [ ] All tools work (retrain, validate, deploy)
- [ ] Dependency handling works correctly
- [ ] MLflow experiment logs plan executions
- [ ] Unit tests pass

### Person F (Backend)
- [ ] All endpoints return correct schemas
- [ ] Digital twin endpoint works end-to-end
- [ ] Health check endpoint works
- [ ] OpenAPI docs accessible at /docs
- [ ] Integration tests pass

### Person G (Dashboard)
- [ ] All 4 tabs render without errors
- [ ] Digital twin simulator works
- [ ] Charts and visualizations display correctly
- [ ] Connected to real API (not mocks)
- [ ] UI/UX polished

---

## 🎓 Success Criteria

**Minimum Viable Product (MVP)**:
1. ✅ Upload CSV → Detect drift → Show results
2. ✅ Digital twin simulation runs end-to-end
3. ✅ Dashboard displays all tabs without errors
4. ✅ MLflow logs all experiments

**Demo-Ready**:
1. ✅ All MVPs +
2. ✅ LLM reasoning working (not just fallback)
3. ✅ Validation gating prevents bad deployments
4. ✅ DVC baseline versioning implemented
5. ✅ Professional UI with good UX

**Hackathon Winner**:
1. ✅ All Demo-Ready +
2. ✅ Complete production deployment docs
3. ✅ Docker Compose setup
4. ✅ Comprehensive README
5. ✅ Live demo with multiple scenarios
6. ✅ Clear architectural diagrams

---

## 📞 Communication Protocols

### Slack Channels
- `#general` - Team coordination
- `#blockers` - Urgent issues
- `#integration` - Interface questions
- `#demo-prep` - Final hours

### Status Updates (Every 2 hours)
```
Format:
- Done: [X, Y, Z]
- Working on: [A]
- Blocked by: [Person B needs to finish stats.json]
- ETA: [30 minutes]
```

### Integration Points (Coordination Required)
1. **Hour 2**: Person B finishes training → Person C can integrate
2. **Hour 3**: Person C, D finish → Person E can integrate
3. **Hour 4**: Person E finishes → Person F can integrate
4. **Hour 5**: Person F finishes → Person G can integrate

---

## 🔥 Troubleshooting Guide

### Common Issues

**Person C**: "Can't find training_stats.json"
- Solution: Wait for Person B to run training, or use mock stats

**Person F**: "Module not found error"
- Solution: Run `uv sync` to install dependencies

**Person G**: "Cannot connect to API"
- Solution: Check FastAPI is running on port 8000

**Anyone**: "MLflow experiment not found"
- Solution: Run `mlflow ui` first, creates experiments automatically

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual modules
if __name__ == "__main__":
    # Test your module here
    pass
```

---

## 🏆 Best Practices

1. **Commit often**: Every 30 minutes to your branch
2. **Test locally**: Before pushing
3. **Use type hints**: Makes integration easier
4. **Document interfaces**: Docstrings for all public methods
5. **Handle errors**: Try/except with meaningful messages
6. **Log everything**: Use Python logging module
7. **Mock first**: Don't wait for dependencies
8. **Communicate**: Post updates every 2 hours

Good luck! 🚀
