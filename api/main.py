"""
FastAPI Backend for DriftCatcher Agentic AI System
Provides inference and agent management endpoints
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pickle
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.drift_detector import DriftDetector
from monitoring.performance_monitor import PerformanceMonitor
from agent.AgenticReasoningEngine import AgenticReasoningEngine
from agent.PlanningAgent import PlanningAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="DriftCatcher API",
    description="Agentic AI system for autonomous ML model lifecycle management",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
MODEL_PATH = "artifacts/model.pkl"
MODEL_NAME = "ddos_random_forest"
TRAINING_STATS_PATH = "artifacts/training_stats.json"

model = None
current_model_version = None
drift_detector = None
reasoning_engine = None
planning_agent = None


# Pydantic models for requests/responses
class PredictRequest(BaseModel):
    features: Dict[str, float]

class PredictBatchRequest(BaseModel):
    data: List[Dict[str, float]]

class DriftCheckRequest(BaseModel):
    csv_data: Optional[str] = None  # Base64 encoded or path

class PlanRequest(BaseModel):
    goal: str
    context: Dict

class ReasoningRequest(BaseModel):
    drift_results: Dict
    context: Dict

def load_model_from_file(path: str = MODEL_PATH):
    """Load model from pickle file"""
    global model, current_model_version
    
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        current_model_version = f"file:{path}"
        logger.info(f"✅ Model loaded from {path}")
        return True
    else:
        logger.warning(f"⚠️  No model found at {path}")
        return False


def load_model_from_mlflow(model_name: str = MODEL_NAME, version: str = "latest"):
    """Load model from MLflow Model Registry"""
    global model, current_model_version
    
    try:
        if version == "latest":
            # Get latest version number
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.warning(f"⚠️  No versions found for model {model_name}")
                return False
            latest_version = max([int(v.version) for v in versions])
            model_uri = f"models:/{model_name}/{latest_version}"
        else:
            model_uri = f"models:/{model_name}/{version}"
        
        model = mlflow.sklearn.load_model(model_uri)
        current_model_version = model_uri
        logger.info(f"✅ Model loaded from MLflow: {model_uri}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model from MLflow: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model and initialize components on startup"""
    global model, drift_detector, reasoning_engine, planning_agent
    
    logger.info("🚀 Starting DriftCatcher API...")
    
    # Try loading model from file first (fastest)
    if not load_model_from_file():
        # Fallback to MLflow if file doesn't exist
        logger.info("Trying to load from MLflow...")
        load_model_from_mlflow()
    
    # Initialize drift detector
    if os.path.exists(TRAINING_STATS_PATH):
        drift_detector = DriftDetector(TRAINING_STATS_PATH)
        logger.info(f"✅ Drift detector initialized")
    else:
        logger.warning(f"⚠️  No training stats found at {TRAINING_STATS_PATH}")
    
    # Initialize agents
    reasoning_engine = AgenticReasoningEngine()
    planning_agent = PlanningAgent(use_reasoning_engine=True)
    logger.info("✅ Agents initialized")
    
    logger.info("🎉 DriftCatcher API ready!")


# Health check
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DriftCatcher API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_version": current_model_version,
        "drift_detector_ready": drift_detector is not None,
        "agents_ready": reasoning_engine is not None and planning_agent is not None
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "model": model is not None,
            "drift_detector": drift_detector is not None,
            "reasoning_engine": reasoning_engine is not None,
            "planning_agent": planning_agent is not None
        }
    }


# Inference endpoints
@app.post("/predict")
async def predict(request: PredictRequest):
    """Make a single prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])
        
        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()
        
        return {
            "prediction": int(prediction),
            "probability": probability,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: PredictBatchRequest):
    """Make batch predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Predict
        predictions = model.predict(df).tolist()
        probabilities = model.predict_proba(df).tolist()
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Drift detection endpoints
@app.post("/drift/check")
async def check_drift(file: UploadFile = File(...)):
    """Check for drift in uploaded CSV data"""
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        logger.info(f"Checking drift on {len(df)} samples")
        
        # Detect drift
        results = drift_detector.detect_drift(df)
        
        # Log to MLflow
        drift_detector.log_drift_to_mlflow(results)
        
        return {
            "drift_results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Drift check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/status")
async def drift_status():
    """Get current drift detection status"""
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    return {
        "detector_ready": True,
        "training_stats_loaded": True,
        "thresholds": {
            "low": 0.1,
            "high": 0.2
        }
    }


# Agent endpoints
@app.post("/agent/reason")
async def agent_reason(request: ReasoningRequest):
    """Get reasoning engine decision on drift"""
    if reasoning_engine is None:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    
    try:
        logger.info("Agent reasoning request received")
        
        # Get reasoning decision
        decision = reasoning_engine.reason_about_action(
            drift_results=request.drift_results,
            context=request.context
        )
        
        return {
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/plan")
async def agent_plan(request: PlanRequest):
    """Create a multi-step plan to achieve a goal"""
    if planning_agent is None:
        raise HTTPException(status_code=503, detail="Planning agent not initialized")
    
    try:
        logger.info(f"Planning request: {request.goal}")
        
        # Create plan
        plan = planning_agent.create_plan(request.goal, request.context)
        
        # Convert plan to serializable format
        plan_dict = [
            {
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "description": step.description,
                "dependencies": step.dependencies,
                "status": step.status.value
            }
            for step in plan
        ]
        
        return {
            "goal": request.goal,
            "plan": plan_dict,
            "total_steps": len(plan_dict),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Planning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/plan/execute")
async def agent_execute_plan(request: PlanRequest):
    """Create and execute a plan"""
    if planning_agent is None:
        raise HTTPException(status_code=503, detail="Planning agent not initialized")
    
    try:
        logger.info(f"Executing plan for goal: {request.goal}")
        
        # Create plan
        plan = planning_agent.create_plan(request.goal, request.context)
        
        # Execute plan (dry run for API)
        results = planning_agent.execute_plan(plan, dry_run=True)
        
        return {
            "goal": request.goal,
            "execution_results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Plan execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/digital-twin")
async def digital_twin_simulation(file: UploadFile = File(...)):
    """
    Digital twin simulation: Upload CSV and see what agent would decide
    Complete flow: drift check → reasoning → planning → execution (dry run)
    Each agent logs to its own MLflow experiment automatically
    """
    if not all([drift_detector, reasoning_engine, planning_agent]):
        raise HTTPException(status_code=503, detail="Agents not fully initialized")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        logger.info(f"🎭 Digital Twin: Simulating on {len(df)} samples")
        
        # Step 1: Drift detection (logs to drift_monitoring experiment)
        logger.info("Step 1: Running drift detection...")
        drift_results = drift_detector.detect_drift(df)
        drift_detector.log_drift_to_mlflow(drift_results)
        
        # Step 2: Reasoning engine decision (logs to agentic_reasoning experiment)
        logger.info("Step 2: Consulting reasoning engine...")
        context = {
            "time_since_last_retrain": "7 days",
            "retraining_cost": "medium",
            "deployment_risk": "low",
            "current_accuracy": 0.95,
            "simulation": True
        }
        decision = reasoning_engine.reason_about_action(drift_results, context)
        
        # Step 3: Create and execute plan if needed (logs to agentic_planning experiment)
        plan = None
        execution_results = None
        action = decision.get('action', 'MONITOR').upper()  # Normalize to uppercase
        
        if action in ['RETRAIN', 'RETRAIN_URGENT']:
            logger.info(f"Step 3: Creating execution plan for action '{action}'...")
            
            plan_context = {
                'latest_data_path': f'simulation_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                'base_data_path': 'data/raw/baseline.csv',
                'holdout_path': 'data/holdout.csv',
                'target_accuracy': 0.95,
                'urgency': 'high' if action == 'RETRAIN_URGENT' else 'normal'
            }
            
            # Create plan (logs plan creation to MLflow)
            plan_steps = planning_agent.create_plan("maintain_accuracy_above_0.95", plan_context)
            
            # Execute plan in dry-run mode
            logger.info("Step 4: Executing plan (dry run)...")
            execution_results = planning_agent.execute_plan(plan_steps, dry_run=True)
            
            # Serialize plan for response
            plan = [
                {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "dependencies": step.dependencies,
                    "status": step.status.value if hasattr(step.status, 'value') else str(step.status)
                }
                for step in plan_steps
            ]
        else:
            logger.info(f"Step 3: Action is '{action}', no plan needed")
        
        # Create comprehensive summary
        simulation_summary = {
            "data_samples": len(df),
            "timestamp": datetime.now().isoformat(),
            "steps": {
                "1_drift_detection": {
                    "overall_psi": drift_results['overall_psi'],
                    "n_drifted_features": drift_results['n_drifted_features'],
                    "total_features": drift_results['total_features'],
                    "top_drifted_features": list(drift_results['feature_psi'].items())[:5]
                },
                "2_reasoning": {
                    "action": decision.get('action', 'MONITOR'),
                    "reasoning": decision.get('reasoning', 'N/A'),
                    "confidence": decision.get('confidence', 0.0),
                    "context_considered": context
                },
                "3_planning": {
                    "plan_created": plan is not None,
                    "total_steps": len(plan) if plan else 0,
                    "plan_details": plan
                },
                "4_execution": {
                    "executed": execution_results is not None,
                    "dry_run": True,
                    "results": execution_results
                }
            }
        }
        
        logger.info("✅ Digital twin simulation complete")
        
        return {
            "status": "success",
            "simulation": simulation_summary,
            "message": f"Simulation complete: {action} recommended based on PSI={drift_results['overall_psi']:.3f}"
        }
        
    except Exception as e:
        logger.error(f"Digital twin simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# MLflow integration
@app.get("/mlflow/experiments")
async def get_experiments():
    """Get MLflow experiments"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        logger.error(f"MLflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/runs/{experiment_name}")
async def get_runs(experiment_name: str, limit: int = 10):
    """Get recent runs from an MLflow experiment"""
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
        
        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"]
        )
        
        # Convert to dict and replace NaN/inf with None for JSON serialization
        import numpy as np
        import math
        runs_dict = runs.to_dict(orient='records')
        
        # Clean up NaN/inf values
        def clean_value(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        
        runs_dict = [
            {k: clean_value(v) for k, v in run.items()}
            for run in runs_dict
        ]
        
        return {
            "experiment": experiment_name,
            "runs": runs_dict
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MLflow runs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent status and logs
@app.get("/agent/status")
async def agent_status():
    """Get agent status and learning summary"""
    if reasoning_engine is None:
        raise HTTPException(status_code=503, detail="Reasoning engine not initialized")
    
    try:
        summary = reasoning_engine.get_learning_summary()
        
        return {
            "status": "operational",
            "reasoning_engine": {
                "learning_summary": summary,
                "use_llm": reasoning_engine.use_llm,
                "model": reasoning_engine.model if reasoning_engine.use_llm else None
            },
            "planning_agent": {
                "total_plans": len(planning_agent.execution_history) if planning_agent else 0
            }
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@app.get("/model/info")
async def model_info():
    """Get current model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    return {
        "model_loaded": True,
        "version": current_model_version,
        "model_type": type(model).__name__
    }


@app.post("/model/reload")
async def reload_model(source: str = "file", version: str = "latest"):
    """
    Reload model from file or MLflow
    
    Args:
        source: 'file' or 'mlflow'
        version: version number or 'latest' (only for MLflow)
    """
    try:
        if source == "file":
            success = load_model_from_file()
        elif source == "mlflow":
            success = load_model_from_mlflow(version=version)
        else:
            raise HTTPException(status_code=400, detail="Invalid source. Use 'file' or 'mlflow'")
        
        if success:
            return {
                "status": "success",
                "message": f"Model reloaded from {source}",
                "version": current_model_version
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/logs/recent")
async def agent_recent_logs(limit: int = 10):
    """Get recent agent decision logs"""
    try:
        log_file = Path("agent/logs/reasoning_decisions.jsonl")
        
        if not log_file.exists():
            return {"logs": []}
        
        import json
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        logs = [json.loads(line) for line in lines[-limit:]]
        
        return {
            "logs": logs,
            "count": len(logs)
        }
    except Exception as e:
        logger.error(f"Logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
