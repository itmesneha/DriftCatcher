"""
Streamlit Dashboard for DriftCatcher Agentic AI System
Real-time monitoring and control interface
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
MLFLOW_BASE_URL = "http://localhost:5001"

st.set_page_config(
    page_title="DriftCatcher - Agentic AI Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .status-healthy {
        color: #10b981;
        font-weight: bold;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: bold;
    }
    .status-critical {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def check_api_health():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.ok else None
    except:
        return False, None

def get_mlflow_experiments():
    """Get MLflow experiments from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/mlflow/experiments", timeout=5)
        return response.json().get("experiments", []) if response.ok else []
    except:
        return []

def get_mlflow_runs(experiment_name, limit=10):
    """Get runs for an experiment"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/mlflow/runs/{experiment_name}",
            params={"limit": limit},
            timeout=5
        )
        return response.json().get("runs", []) if response.ok else []
    except:
        return []

def get_agent_status():
    """Get agent status"""
    try:
        response = requests.get(f"{API_BASE_URL}/agent/status", timeout=5)
        return response.json() if response.ok else None
    except:
        return None

def get_recent_logs(limit=10):
    """Get recent agent logs"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/agent/logs/recent",
            params={"limit": limit},
            timeout=5
        )
        return response.json().get("logs", []) if response.ok else []
    except:
        return []

def check_drift(file):
    """Upload file and check drift"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_BASE_URL}/drift/check", files=files, timeout=30)
        return response.json() if response.ok else None
    except Exception as e:
        st.error(f"Error checking drift: {e}")
        return None

def run_digital_twin(file):
    """Run digital twin simulation"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_BASE_URL}/agent/digital-twin", files=files, timeout=60)
        
        if response.ok:
            return response.json()
        else:
            st.error(f"❌ API Error {response.status_code}: {response.text[:500]}")
            return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The simulation may be taking too long.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure it's running on port 8000.")
        return None
    except Exception as e:
        st.error(f"❌ Error running simulation: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Sidebar
st.sidebar.markdown("## 🎯 DriftCatcher")
st.sidebar.markdown("### Agentic AI System")

# Health check
api_healthy, health_data = check_api_health()
if api_healthy:
    st.sidebar.success("✅ API Connected")
    if health_data:
        components = health_data.get("components", {})
        st.sidebar.markdown("**Components:**")
        st.sidebar.markdown(f"- Model: {'✅' if components.get('model') else '❌'}")
        st.sidebar.markdown(f"- Drift Detector: {'✅' if components.get('drift_detector') else '❌'}")
        st.sidebar.markdown(f"- Reasoning Engine: {'✅' if components.get('reasoning_engine') else '❌'}")
        st.sidebar.markdown(f"- Planning Agent: {'✅' if components.get('planning_agent') else '❌'}")
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.markdown("Start API: `uv run python api/main.py`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Links:**")
st.sidebar.markdown(f"- [API Docs]({API_BASE_URL}/docs)")
st.sidebar.markdown(f"- [MLflow UI]({MLFLOW_BASE_URL})")

# Main content
st.markdown('<h1 class="main-header">DriftCatcher Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Autonomous ML Model Lifecycle Management with Agentic AI**")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance",
    "🔍 Drift Monitoring",
    "🤖 Agent Activity",
    "🎭 Digital Twin Simulator"
])

# Tab 1: Model Performance
with tab1:
    st.header("Model Performance Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("MLflow Experiments")
        
        experiments = get_mlflow_experiments()
        if experiments:
            exp_names = [exp["name"] for exp in experiments]
            selected_exp = st.selectbox("Select Experiment:", exp_names)
            
            if selected_exp:
                runs = get_mlflow_runs(selected_exp, limit=20)
                
                if runs:
                    st.markdown(f"**Recent Runs: {len(runs)}**")
                    
                    # Check if this is drift monitoring or model training
                    if "drift" in selected_exp.lower():
                        # Drift monitoring metrics
                        run_data = []
                        for run in runs:
                            run_data.append({
                                "run_id": run.get("run_id", "")[:8],
                                "start_time": run.get("start_time", 0),
                                "overall_psi": run.get("metrics.overall_psi"),
                                "n_drifted": run.get("metrics.n_drifted_features"),
                                "total_features": run.get("metrics.total_features")
                            })
                        
                        if run_data:
                            df_runs = pd.DataFrame(run_data)
                            df_runs = df_runs.sort_values("start_time")
                            
                            # Plot drift metrics over time
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df_runs["run_id"],
                                y=df_runs["overall_psi"],
                                mode='lines+markers',
                                name='Overall PSI',
                                line=dict(color='#f59e0b', width=2)
                            ))
                            
                            # Add threshold lines
                            fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Low")
                            fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="High")
                            
                            fig.update_layout(
                                title="Drift Monitoring Over Time",
                                xaxis_title="Run ID",
                                yaxis_title="PSI Score",
                                hovermode='x unified',
                                height=400
                            )
                            st.plotly_chart(fig, width='stretch')
                            
                            # Show recent runs table
                            st.dataframe(
                                df_runs[["run_id", "overall_psi", "n_drifted", "total_features"]],
                                width='stretch'
                            )
                    else:
                        # Model training metrics
                        run_data = []
                        for run in runs:
                            run_data.append({
                                "run_id": run.get("run_id", "")[:8],
                                "start_time": run.get("start_time", 0),
                                "accuracy": run.get("metrics.accuracy"),
                                "f1_score": run.get("metrics.f1_score"),
                                "precision": run.get("metrics.precision"),
                                "recall": run.get("metrics.recall")
                            })
                        
                        if run_data:
                            df_runs = pd.DataFrame(run_data)
                            df_runs = df_runs.sort_values("start_time")
                            
                            # Plot metrics over time
                            fig = go.Figure()
                            if "accuracy" in df_runs.columns and df_runs["accuracy"].notna().any():
                                fig.add_trace(go.Scatter(
                                    x=df_runs["run_id"],
                                    y=df_runs["accuracy"],
                                    mode='lines+markers',
                                    name='Accuracy',
                                    line=dict(color='#667eea', width=2)
                                ))
                            if "f1_score" in df_runs.columns and df_runs["f1_score"].notna().any():
                                fig.add_trace(go.Scatter(
                                    x=df_runs["run_id"],
                                    y=df_runs["f1_score"],
                                    mode='lines+markers',
                                    name='F1 Score',
                                    line=dict(color='#764ba2', width=2)
                                ))
                            
                            fig.update_layout(
                                title="Model Performance Over Time",
                                xaxis_title="Run ID",
                                yaxis_title="Score",
                                hovermode='x unified',
                                height=400
                            )
                            st.plotly_chart(fig, width='stretch')
                            
                            # Show recent runs table
                            st.dataframe(
                                df_runs[["run_id", "accuracy", "f1_score", "precision", "recall"]],
                                width='stretch'
                            )
                else:
                    st.info("No runs found for this experiment")
        else:
            st.info("No experiments found. Train a model first.")
    
    with col2:
        st.subheader("Current Model Status")
        
        try:
            model_info = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
            if model_info.ok:
                info = model_info.json()
                st.markdown(f"**Version:** `{info.get('version', 'Unknown')}`")
                st.markdown(f"**Type:** `{info.get('model_type', 'Unknown')}`")
                
                # Model reload
                st.markdown("---")
                st.markdown("**Model Management**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔄 Reload from File"):
                        response = requests.post(
                            f"{API_BASE_URL}/model/reload",
                            params={"source": "file"}
                        )
                        if response.ok:
                            st.success("✅ Model reloaded from file")
                        else:
                            st.error("❌ Failed to reload model")
                
                with col_b:
                    if st.button("📦 Reload from MLflow"):
                        response = requests.post(
                            f"{API_BASE_URL}/model/reload",
                            params={"source": "mlflow", "version": "latest"}
                        )
                        if response.ok:
                            st.success("✅ Model reloaded from MLflow")
                        else:
                            st.error("❌ Failed to reload model")
            else:
                st.warning("⚠️ Model not loaded")
        except:
            st.error("❌ Cannot connect to API")

# Tab 2: Drift Monitoring
with tab2:
    st.header("Drift Detection & Monitoring")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Data for Drift Check")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file to check for drift",
            type=["csv"],
            key="drift_upload"
        )
        
        if uploaded_file:
            if st.button("🔍 Check Drift", key="check_drift_btn"):
                with st.spinner("Analyzing drift..."):
                    result = check_drift(uploaded_file)
                    
                    if result:
                        st.success("✅ Drift analysis complete")
                        
                        drift_results = result.get("drift_results", {})
                        
                        # Overall status
                        overall_psi = drift_results.get("overall_psi", 0)
                        n_drifted = drift_results.get("n_drifted_features", 0)
                        
                        # Determine drift level based on PSI thresholds
                        if overall_psi > 0.2:
                            drift_level = "SEVERE"
                            color = "🔴"
                        elif overall_psi > 0.1:
                            drift_level = "MODERATE"
                            color = "🟡"
                        else:
                            drift_level = "NONE"
                            color = "🟢"
                        
                        status_col1, status_col2, status_col3 = st.columns(3)
                        with status_col1:
                            st.metric("Overall PSI", f"{overall_psi:.4f}")
                        with status_col2:
                            st.metric("Drifted Features", n_drifted)
                        with status_col3:
                            st.metric("Drift Level", f"{color} {drift_level}")
                        
                        # Show reasoning engine recommendation
                        st.markdown("---")
                        st.info("💡 For action recommendation, use the **Agent Activity** tab or **Digital Twin Simulator**")
                        
                        # Feature-level drift
                        st.markdown("---")
                        st.subheader("Feature-Level Drift")
                        
                        feature_psi = drift_results.get("feature_psi", {})
                        drifted_features_list = drift_results.get("drifted_features", [])
                        drifted_feature_names = [feat[0] for feat in drifted_features_list]
                        
                        if feature_psi:
                            drift_df = pd.DataFrame([
                                {
                                    "Feature": feat, 
                                    "PSI": psi, 
                                    "Drifted": feat in drifted_feature_names
                                }
                                for feat, psi in feature_psi.items()
                            ]).sort_values("PSI", ascending=False)
                            
                            # Plot top drifted features
                            fig = px.bar(
                                drift_df.head(10),
                                x="PSI",
                                y="Feature",
                                orientation='h',
                                color="Drifted",
                                color_discrete_map={True: "#ef4444", False: "#10b981"},
                                title="Top 10 Features by PSI Score"
                            )
                            st.plotly_chart(fig, width="stretch")
                            
                            # Show table
                            st.dataframe(drift_df, width="stretch")
    
    with col2:
        st.subheader("Drift Monitoring History")
        
        # Get drift monitoring runs from MLflow
        drift_runs = get_mlflow_runs("drift_monitoring", limit=20)
        
        if drift_runs:
            # Extract drift data
            drift_history = []
            for run in drift_runs:
                # MLflow returns flattened keys like "metrics.overall_psi"
                start_time = run.get("start_time", 0)
                # Handle both timestamp formats (ISO string or ms)
                try:
                    if isinstance(start_time, str):
                        timestamp = pd.to_datetime(start_time)
                    else:
                        timestamp = pd.to_datetime(start_time, unit='ms')
                except:
                    timestamp = pd.Timestamp.now()
                
                drift_history.append({
                    "timestamp": timestamp,
                    "overall_psi": run.get("metrics.overall_psi", 0),
                    "n_drifted": run.get("metrics.n_drifted_features", 0)
                })
            
            df_drift = pd.DataFrame(drift_history).sort_values("timestamp")
            
            # Plot PSI over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_drift["timestamp"],
                y=df_drift["overall_psi"],
                mode='lines+markers',
                name='Overall PSI',
                line=dict(color='#f59e0b', width=2),
                fill='tozeroy'
            ))
            
            # Add threshold lines
            fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Low Threshold")
            fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="High Threshold")
            
            fig.update_layout(
                title="PSI Score Over Time",
                xaxis_title="Time",
                yaxis_title="PSI Score",
                height=400
            )
            st.plotly_chart(fig, width="stretch")
            
            # Show recent drift checks
            st.dataframe(df_drift, width="stretch")
        else:
            st.info("No drift monitoring history. Upload a CSV to start monitoring.")

# Tab 3: Agent Activity
with tab3:
    st.header("Agentic AI Activity Monitor")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Agent Status")
        
        agent_status = get_agent_status()
        
        if agent_status:
            st.success("✅ Agents Operational")
            
            # Reasoning engine status
            st.markdown("**🧠 Reasoning Engine**")
            reasoning = agent_status.get("reasoning_engine", {})
            summary = reasoning.get("learning_summary", {})
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Total Decisions", summary.get("total_decisions", 0))
            with metric_col2:
                st.metric("LLM Enabled", "✅" if reasoning.get("use_llm") else "❌")
            
            if reasoning.get("use_llm"):
                st.markdown(f"**Model:** `{reasoning.get('model', 'Unknown')}`")
            
            # Planning agent status
            st.markdown("---")
            st.markdown("**📋 Planning Agent**")
            planning = agent_status.get("planning_agent", {})
            st.metric("Total Plans", planning.get("total_plans", 0))
        else:
            st.warning("⚠️ Cannot fetch agent status")
    
    with col2:
        st.subheader("Recent Agent Decisions")
        
        logs = get_recent_logs(limit=20)
        
        if logs:
            for log in reversed(logs):  # Show most recent first
                timestamp = log.get("timestamp", "Unknown")
                action = log.get("action", "Unknown")
                reasoning = log.get("reasoning", "No reasoning provided")
                
                # Color-code by action
                if action == "MONITOR":
                    emoji = "🟢"
                    color = "#10b981"
                elif action == "RETRAIN":
                    emoji = "🟡"
                    color = "#f59e0b"
                elif action == "RETRAIN_URGENT":
                    emoji = "🔴"
                    color = "#ef4444"
                else:
                    emoji = "⚪"
                    color = "#6b7280"
                
                with st.expander(f"{emoji} {action} - {timestamp}"):
                    st.markdown(f"**Action:** {action}")
                    st.markdown(f"**Reasoning:** {reasoning}")
                    
                    drift = log.get("drift_results", {})
                    if drift:
                        st.markdown(f"**PSI:** {drift.get('overall_psi', 0):.4f}")
                        st.markdown(f"**Drifted Features:** {drift.get('n_drifted_features', 0)}")
        else:
            st.info("No recent decisions. Upload data to trigger agent reasoning.")
    
    # Agent runs from MLflow
    st.markdown("---")
    st.subheader("Agent Run History")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Reasoning Runs**")
        reasoning_runs = get_mlflow_runs("agentic_reasoning", limit=10)
        if reasoning_runs:
            st.metric("Total Runs", len(reasoning_runs))
            
            # Count actions (MLflow returns flattened params like "params.action")
            actions = [run.get("params.action") for run in reasoning_runs if run.get("params.action")]
            if actions:
                action_counts = pd.Series(actions).value_counts()
                
                fig = px.pie(
                    values=action_counts.values,
                    names=action_counts.index,
                    title="Action Distribution"
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No actions logged yet")
        else:
            st.info("No reasoning runs yet")
    
    with col_b:
        st.markdown("**Planning Runs**")
        planning_runs = get_mlflow_runs("agentic_planning", limit=10)
        if planning_runs:
            st.metric("Total Plans", len(planning_runs))
            
            # Extract success metrics
            success_data = []
            for run in planning_runs:
                # MLflow returns flattened keys like "metrics.total_steps"
                success_data.append({
                    "run": run.get("run_id", "")[:8],
                    "total_steps": run.get("metrics.total_steps", 0),
                    "completed": run.get("metrics.completed_steps", 0),
                    "failed": run.get("metrics.failed_steps", 0)
                })
            
            if success_data:
                df_plans = pd.DataFrame(success_data)
                
                fig = go.Figure(data=[
                    go.Bar(name='Completed', x=df_plans['run'], y=df_plans['completed'], marker_color='#10b981'),
                    go.Bar(name='Failed', x=df_plans['run'], y=df_plans['failed'], marker_color='#ef4444')
                ])
                fig.update_layout(barmode='stack', title="Plan Execution Status", height=300)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No planning runs yet")

# Tab 4: Digital Twin Simulator
with tab4:
    st.header("🎭 Digital Twin Simulator")
    st.markdown("**Upload CSV data to simulate complete agent decision flow**")
    st.markdown("Flow: Drift Detection → Reasoning Engine → Planning Agent")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Data")
        
        sim_file = st.file_uploader(
            "Upload CSV for digital twin simulation",
            type=["csv"],
            key="sim_upload"
        )
        
        if sim_file:
            # Preview data
            preview_df = pd.read_csv(sim_file)
            st.markdown(f"**Data Preview:** {len(preview_df)} samples, {len(preview_df.columns)} features")
            st.dataframe(preview_df.head(), width="stretch")
            
            # Reset file pointer
            sim_file.seek(0)
            
            if st.button("🎭 Run Digital Twin Simulation", type="primary", key="run_sim_btn"):
                st.info("🔄 Starting simulation...")
                
                with st.spinner("Running complete simulation flow..."):
                    # Reset file pointer again before sending
                    sim_file.seek(0)
                    result = run_digital_twin(sim_file)
                    
                    if result:
                        st.success("✅ Simulation request successful!")
                        
                        # Store in session state for display in col2
                        st.session_state.sim_result = result
                        st.rerun()
                    else:
                        st.error("❌ Simulation failed. Check error message above.")
    
    with col2:
        st.subheader("Simulation Results")
        
        if "sim_result" in st.session_state:
            result = st.session_state.sim_result
            
            # Show status message
            if result.get("status") == "success":
                st.success(result.get("message", "Simulation complete"))
            
            sim = result.get("simulation", {})
            steps = sim.get("steps", {})
            
            # Step 1: Drift Detection
            st.markdown("### 1️⃣ Drift Detection")
            drift = steps.get("1_drift_detection", {})
            
            metric1, metric2, metric3 = st.columns(3)
            with metric1:
                st.metric("Data Samples", sim.get("data_samples", 0))
            with metric2:
                st.metric("Overall PSI", f"{drift.get('overall_psi', 0):.4f}")
            with metric3:
                st.metric("Drifted Features", drift.get("n_drifted_features", 0))
            
            # Show top drifted features
            top_features = drift.get("top_drifted_features", [])
            if top_features:
                with st.expander("View Top Drifted Features"):
                    for feat_name, psi in top_features:
                        st.markdown(f"- **{feat_name}**: PSI = {psi:.4f}")
            
            # Step 2: Agent Decision
            st.markdown("### 2️⃣ Agent Reasoning")
            decision = steps.get("2_reasoning", {})
            
            action = decision.get("action", "Unknown")
            reasoning = decision.get("reasoning", "No reasoning")
            confidence = decision.get("confidence", 0.0)
            
            if action == "MONITOR":
                st.success(f"✅ **Action:** {action}")
            elif action == "RETRAIN":
                st.warning(f"⚠️ **Action:** {action}")
            elif action == "RETRAIN_URGENT":
                st.error(f"🚨 **Action:** {action}")
            else:
                st.info(f"ℹ️ **Action:** {action}")
            
            st.markdown(f"**Reasoning:** {reasoning}")
            st.markdown(f"**Confidence:** {confidence}%")
            
            # Show context considered
            context = decision.get("context_considered", {})
            if context:
                with st.expander("Context Considered"):
                    for key, value in context.items():
                        st.markdown(f"- **{key}**: {value}")
            
            # Step 3: Planning
            st.markdown("### 3️⃣ Planning")
            planning = steps.get("3_planning", {})
            plan = planning.get("plan_details")
            
            if planning.get("plan_created") and plan:
                st.info(f"📋 Plan created with {planning.get('total_steps', 0)} steps")
                
                for i, step in enumerate(plan, 1):
                    status = step.get('status', 'unknown')
                    status_emoji = {
                        'pending': '⏳',
                        'completed': '✅',
                        'failed': '❌',
                        'in_progress': '⚙️'
                    }.get(status, '❓')
                    
                    with st.expander(f"{status_emoji} Step {i}: {step.get('description', 'Unknown')}"):
                        st.markdown(f"**Tool:** `{step.get('tool_name', 'Unknown')}`")
                        st.markdown(f"**Description:** {step.get('description', 'N/A')}")
                        st.markdown(f"**Status:** {status}")
                        deps = step.get('dependencies', [])
                        if deps:
                            st.markdown(f"**Dependencies:** Steps {', '.join(map(str, deps))}")
            else:
                st.success("✅ No plan needed - monitoring only")
            
            # Step 4: Execution
            st.markdown("### 4️⃣ Execution")
            execution = steps.get("4_execution", {})
            
            if execution.get("executed"):
                st.info("🎭 **Dry Run Mode** - No actual changes made")
                results = execution.get("results", {})
                if results:
                    for step_id, result in results.items():
                        # Result might be a string or dict
                        if isinstance(result, dict):
                            status = result.get('status', 'unknown')
                        else:
                            status = str(result)
                        st.markdown(f"- **{step_id}**: {status}")
            else:
                st.info("No execution needed for MONITOR action")
            
            # Clear button
            st.markdown("---")
            if st.button("🗑️ Clear Results"):
                del st.session_state.sim_result
                st.rerun()
        else:
            st.info("Upload a CSV file and run simulation to see results here")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**DriftCatcher v1.0**")
with col2:
    st.markdown(f"**API:** {API_BASE_URL}")
with col3:
    st.markdown(f"**MLflow:** {MLFLOW_BASE_URL}")
