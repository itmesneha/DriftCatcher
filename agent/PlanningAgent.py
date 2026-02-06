"""
Multi-step Planning Agent with Tool Use
Creates and executes plans to achieve ML operational goals
"""
import json
import sys
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional
from pathlib import Path
from enum import Enum
import mlflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create logs directory
Path("agent/logs").mkdir(parents=True, exist_ok=True)

from monitoring.drift_detector import DriftDetector
from monitoring.performance_monitor import PerformanceMonitor
from agent.AgenticReasoningEngine import AgenticReasoningEngine


class PlanStatus(Enum):
    """Status of plan execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanStep:
    """Represents a single step in a plan"""
    
    def __init__(
        self,
        step_id: int,
        tool_name: str,
        description: str,
        parameters: Dict,
        dependencies: List[int] = None
    ):
        self.step_id = step_id
        self.tool_name = tool_name
        self.description = description
        self.parameters = parameters
        self.dependencies = dependencies or []
        self.status = PlanStatus.PENDING
        self.result = None
        self.error = None


class PlanningAgent:
    """Multi-step planning agent for ML operations"""
    
    def __init__(self, use_reasoning_engine: bool = True):
        """Initialize planning agent with available tools
        
        Args:
            use_reasoning_engine: Whether to use AgenticReasoningEngine for decisions
        """
        
        # Tool registry
        self.tools = {
            'check_drift': self._tool_check_drift,
            'evaluate_performance': self._tool_evaluate_performance,
            'request_labeling': self._tool_request_labeling,
            'retrain_model': self._tool_retrain_model,
            'validate_model': self._tool_validate_model,
            'deploy_model': self._tool_deploy_model,
            'rollback_model': self._tool_rollback_model
        }
        
        # Reasoning engine for intelligent decision-making
        self.reasoning_engine = AgenticReasoningEngine() if use_reasoning_engine else None
        
        # State tracking
        self.current_plan = None
        self.execution_history = []
    
    def create_plan(
        self,
        goal: str,
        context: Dict
    ) -> List[PlanStep]:
        """
        Create a multi-step plan to achieve a goal
        
        Args:
            goal: High-level goal (e.g., "maintain_accuracy_above_0.95")
            context: Current system state and constraints
            
        Returns:
            List of plan steps
        """
        logger.info("="*60)
        logger.info("PLANNING AGENT - CREATING PLAN")
        logger.info("="*60)
        logger.info(f"Goal: {goal}")
        logger.info(f"Context: {json.dumps(context, indent=2)}")
        
        if goal == "maintain_accuracy_above_0.95":
            plan = self._plan_maintain_accuracy(context)
        elif goal == "handle_drift_proactively":
            plan = self._plan_handle_drift(context)
        elif goal == "retrain_with_new_data":
            plan = self._plan_retrain_with_data(context)
        else:
            raise ValueError(f"Unknown goal: {goal}")
        
        logger.info(f"\n📋 Plan created with {len(plan)} steps:")
        for step in plan:
            deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
            logger.info(f"  {step.step_id}. {step.description}{deps}")
        logger.info("="*60 + "\n")
        
        # Save plan to file
        self._save_plan_log(goal, context, plan)
        
        # Log to MLflow
        self._log_plan_to_mlflow(goal, context, plan)
        
        return plan
    
    def _log_plan_to_mlflow(self, goal: str, context: Dict, plan: List[PlanStep]):
        """Log plan creation to MLflow"""
        try:
            mlflow.set_experiment("agentic_planning")
            
            with mlflow.start_run(run_name=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log plan metadata
                mlflow.log_param("goal", goal)
                mlflow.log_metric("total_steps", len(plan))
                
                # Log each step
                for step in plan:
                    mlflow.log_param(f"step_{step.step_id}_tool", step.tool_name)
                    mlflow.log_param(f"step_{step.step_id}_desc", step.description[:100])  # Truncate
                
                # Log plan as JSON artifact
                plan_dict = {
                    'goal': goal,
                    'context': context,
                    'plan': [
                        {
                            'step_id': step.step_id,
                            'tool_name': step.tool_name,
                            'description': step.description,
                            'dependencies': step.dependencies
                        }
                        for step in plan
                    ]
                }
                mlflow.log_dict(plan_dict, "plan.json")
                
                # Set tags
                mlflow.set_tag("agent_type", "planning_agent")
                mlflow.set_tag("goal", goal)
                
                logger.info("✅ Plan logged to MLflow experiment 'agentic_planning'")
                
        except Exception as e:
            logger.warning(f"Failed to log plan to MLflow: {e}")
    
    def _plan_maintain_accuracy(self, context: Dict) -> List[PlanStep]:
        """Create plan to maintain model accuracy"""
        
        plan = []
        step_id = 1
        
        # Step 1: Check drift
        plan.append(PlanStep(
            step_id=step_id,
            tool_name='check_drift',
            description='Check for data drift in recent production data',
            parameters={'data_path': context.get('latest_data_path')}
        ))
        step_id += 1
        
        # Step 2: Evaluate current performance (if labels available)
        if context.get('labeled_data_available'):
            plan.append(PlanStep(
                step_id=step_id,
                tool_name='evaluate_performance',
                description='Evaluate current model performance',
                parameters={'predictions_path': context.get('predictions_path')},
                dependencies=[1]
            ))
            step_id += 1
        
        # Step 3: Conditional retraining (decision made by reasoning engine)
        # This will be decided during execution based on drift/performance results
        plan.append(PlanStep(
            step_id=step_id,
            tool_name='retrain_model',
            description='Retrain model if reasoning engine recommends it',
            parameters={
                'base_data': context.get('base_data_path'),
                'new_data': context.get('new_labeled_data', []),
                'conditional': True  # Mark as conditional - check reasoning engine first
            },
            dependencies=[1, 2] if context.get('labeled_data_available') else [1]
        ))
        step_id += 1
        
        # Step 4: Validate retrained model
        plan.append(PlanStep(
            step_id=step_id,
            tool_name='validate_model',
            description='Validate retrained model on holdout set',
            parameters={'holdout_path': context.get('holdout_path')},
            dependencies=[3]
        ))
        step_id += 1
        
        # Step 5: Deploy if improved
        plan.append(PlanStep(
            step_id=step_id,
            tool_name='deploy_model',
            description='Deploy model if validation passed',
            parameters={'model_version': 'latest'},
            dependencies=[4]
        ))
        
        return plan
    
    def _plan_handle_drift(self, context: Dict) -> List[PlanStep]:
        """Create plan to handle detected drift"""
        
        plan = []
        
        # Step 1: Assess drift severity
        plan.append(PlanStep(
            step_id=1,
            tool_name='check_drift',
            description='Assess drift severity',
            parameters={'data_path': context.get('latest_data_path')}
        ))
        
        # Step 2: Request labeling of drifted samples
        plan.append(PlanStep(
            step_id=2,
            tool_name='request_labeling',
            description='Request labeling of samples from drifted distribution',
            parameters={
                'n_samples': context.get('samples_to_label', 500),
                'data_path': context.get('latest_data_path')
            },
            dependencies=[1]
        ))
        
        # Step 3: Wait and collect labels (simulated)
        # In production, this would involve human-in-the-loop
        
        # Step 4: Retrain with new labeled data
        plan.append(PlanStep(
            step_id=3,
            tool_name='retrain_model',
            description='Retrain model with original + new labeled data',
            parameters={
                'base_data': context.get('base_data_path'),
                'new_data': context.get('newly_labeled_paths', [])
            },
            dependencies=[2]
        ))
        
        # Step 5: Validate
        plan.append(PlanStep(
            step_id=4,
            tool_name='validate_model',
            description='Validate on both old and new distributions',
            parameters={'holdout_path': context.get('holdout_path')},
            dependencies=[3]
        ))
        
        # Step 6: Deploy or rollback
        plan.append(PlanStep(
            step_id=5,
            tool_name='deploy_model',
            description='Deploy if improved, else rollback',
            parameters={'model_version': 'latest', 'min_accuracy': 0.95},
            dependencies=[4]
        ))
        
        return plan
    
    def _plan_retrain_with_data(self, context: Dict) -> List[PlanStep]:
        """Create plan for retraining with new data"""
        
        return [
            PlanStep(
                step_id=1,
                tool_name='retrain_model',
                description='Retrain with new data',
                parameters={
                    'base_data': context.get('base_data_path'),
                    'new_data': context.get('new_data_paths', [])
                }
            ),
            PlanStep(
                step_id=2,
                tool_name='validate_model',
                description='Validate retrained model',
                parameters={'holdout_path': context.get('holdout_path')},
                dependencies=[1]
            ),
            PlanStep(
                step_id=3,
                tool_name='deploy_model',
                description='Deploy validated model',
                parameters={'model_version': 'latest'},
                dependencies=[2]
            )
        ]
    
    def _save_plan_log(self, goal: str, context: Dict, plan: List[PlanStep]):
        """Save plan to log file for later review"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'goal': goal,
            'context': context,
            'plan': [
                {
                    'step_id': step.step_id,
                    'tool_name': step.tool_name,
                    'description': step.description,
                    'parameters': step.parameters,
                    'dependencies': step.dependencies
                }
                for step in plan
            ]
        }
        
        log_file = Path("agent/logs/planning_plans.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def execute_plan(
        self,
        plan: List[PlanStep],
        dry_run: bool = False
    ) -> Dict:
        """
        Execute a plan step by step
        
        Args:
            plan: List of plan steps
            dry_run: If True, simulate execution without running tools
            
        Returns:
            Execution results
        """
        
        self.current_plan = plan
        results = {
            'plan_id': datetime.now().isoformat(),
            'total_steps': len(plan),
            'completed_steps': 0,
            'failed_steps': 0,
            'step_results': []
        }
        
        # Execute steps respecting dependencies
        for step in plan:
            logger.info("\n" + "="*60)
            logger.info(f"EXECUTING STEP {step.step_id}/{len(plan)}")
            logger.info("="*60)
            logger.info(f"Description: {step.description}")
            logger.info(f"Tool: {step.tool_name}")
            logger.info(f"Parameters: {json.dumps(step.parameters, indent=2)}")
            
            print(f"\n{'='*60}")
            print(f"Step {step.step_id}: {step.description}")
            print(f"Tool: {step.tool_name}")
            print(f"{'='*60}")
            
            # Check dependencies
            if not self._check_dependencies(step, plan):
                step.status = PlanStatus.BLOCKED
                step.error = "Dependencies not met"
                results['failed_steps'] += 1
                logger.warning(f"❌ BLOCKED: Dependencies not met")
                print(f"❌ BLOCKED: Dependencies not met")
                continue
            
            # Execute step
            step.status = PlanStatus.IN_PROGRESS
            
            if dry_run:
                print(f"🔍 DRY RUN: Would execute {step.tool_name}")
                step.result = {'dry_run': True}
                step.status = PlanStatus.COMPLETED
            else:
                try:
                    # Check if this step requires reasoning engine decision
                    if step.parameters.get('conditional') and self.reasoning_engine:
                        logger.info("🧠 Step is conditional - consulting reasoning engine...")
                        decision = self._consult_reasoning_engine(step, plan)
                        
                        logger.info(f"\n📊 REASONING ENGINE DECISION:")
                        logger.info(f"  Action: {decision['action']}")
                        logger.info(f"  Confidence: {decision.get('confidence', 'N/A')}")
                        logger.info(f"  Reasoning: {decision['reasoning']}")
                        
                        if not decision['should_execute']:
                            logger.info(f"⏭️  SKIPPING STEP: Reasoning engine recommends {decision['action']}")
                            print(f"🧠 REASONING ENGINE: {decision['reasoning']}")
                            print(f"⏭️  SKIPPED: {decision['action']}")
                            step.status = PlanStatus.COMPLETED
                            step.result = {'skipped': True, 'reason': decision['reasoning']}
                            results['completed_steps'] += 1
                            continue
                        else:
                            logger.info(f"✅ PROCEEDING: Reasoning engine recommends {decision['action']} with {decision.get('confidence', 'N/A')} confidence")
                            print(f"🧠 REASONING ENGINE: {decision['reasoning']}")
                            print(f"✅ PROCEEDING with {decision['action']}")
                    
                    tool = self.tools.get(step.tool_name)
                    if not tool:
                        raise ValueError(f"Unknown tool: {step.tool_name}")
                    
                    step.result = tool(**step.parameters)
                    step.status = PlanStatus.COMPLETED
                    results['completed_steps'] += 1
                    
                    logger.info(f"✅ STEP COMPLETED")
                    logger.info(f"Result: {json.dumps(step.result, indent=2, default=str)}")
                    print(f"✅ COMPLETED")
                    
                except Exception as e:
                    step.status = PlanStatus.FAILED
                    step.error = str(e)
                    results['failed_steps'] += 1
                    
                    logger.error(f"❌ STEP FAILED: {e}")
                    print(f"❌ FAILED: {e}")
                    
                    # Decide whether to continue or abort
                    if self._is_critical_step(step):
                        print("⚠️  Critical step failed. Aborting plan.")
                        break
            
            results['step_results'].append({
                'step_id': step.step_id,
                'tool': step.tool_name,
                'status': step.status.value,
                'result': step.result,
                'error': step.error
            })
        
        # Record execution
        self.execution_history.append(results)
        
        # Save execution log
        self._save_execution_log(results)
        
        # Log to MLflow
        self._log_execution_to_mlflow(results)
        
        logger.info("\n" + "="*60)
        logger.info("PLAN EXECUTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total steps: {results['total_steps']}")
        logger.info(f"Completed: {results['completed_steps']}")
        logger.info(f"Failed: {results['failed_steps']}")
        logger.info("="*60 + "\n")
        
        return results
    
    def _log_execution_to_mlflow(self, results: Dict):
        """Log plan execution results to MLflow"""
        try:
            mlflow.set_experiment("agentic_planning")
            
            with mlflow.start_run(run_name=f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log execution summary
                mlflow.log_metric("total_steps", results['total_steps'])
                mlflow.log_metric("completed_steps", results['completed_steps'])
                mlflow.log_metric("failed_steps", results['failed_steps'])
                mlflow.log_metric("success_rate", results['completed_steps'] / results['total_steps'] if results['total_steps'] > 0 else 0)
                
                # Log each step result
                for step_result in results['step_results']:
                    step_id = step_result['step_id']
                    mlflow.log_param(f"step_{step_id}_status", step_result['status'])
                    mlflow.log_param(f"step_{step_id}_tool", step_result['tool'])
                    
                    if step_result.get('error'):
                        mlflow.log_param(f"step_{step_id}_error", str(step_result['error'])[:200])
                
                # Log full execution results as artifact
                mlflow.log_dict(results, "execution_results.json")
                
                # Set tags
                mlflow.set_tag("agent_type", "planning_agent")
                mlflow.set_tag("execution_status", "success" if results['failed_steps'] == 0 else "partial_failure")
                mlflow.set_tag("plan_id", results['plan_id'])
                
                logger.info("✅ Execution logged to MLflow experiment 'agentic_planning'")
                
        except Exception as e:
            logger.warning(f"Failed to log execution to MLflow: {e}")
    
    def _save_execution_log(self, results: Dict):
        """Save execution results to log file"""
        log_file = Path("agent/logs/planning_executions.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(results, indent=None, default=str) + '\n')
    
    def _check_dependencies(
        self,
        step: PlanStep,
        plan: List[PlanStep]
    ) -> bool:
        """Check if step dependencies are satisfied"""
        
        for dep_id in step.dependencies:
            dep_step = next((s for s in plan if s.step_id == dep_id), None)
            if not dep_step or dep_step.status != PlanStatus.COMPLETED:
                return False
        return True
    
    def _is_critical_step(self, step: PlanStep) -> bool:
        """Determine if step is critical for plan success"""
        critical_tools = ['retrain_model', 'deploy_model']
        return step.tool_name in critical_tools
    
    def _consult_reasoning_engine(
        self,
        step: PlanStep,
        plan: List[PlanStep]
    ) -> Dict:
        """Consult reasoning engine for decision on conditional step
        
        Args:
            step: The step to decide on
            plan: Full plan with previous results
            
        Returns:
            Decision dict with should_execute, action, reasoning
        """
        # Gather context from previous steps
        drift_results = None
        performance_results = None
        
        for prev_step in plan:
            if prev_step.status == PlanStatus.COMPLETED and prev_step.result:
                if prev_step.tool_name == 'check_drift':
                    drift_results = prev_step.result
                elif prev_step.tool_name == 'evaluate_performance':
                    performance_results = prev_step.result
        
        if not drift_results:
            # No drift info, can't make informed decision
            return {
                'should_execute': True,
                'action': 'PROCEED',
                'reasoning': 'No drift data available, proceeding with default'
            }
        
        # Build context for reasoning engine
        context = {
            'time_since_last_retrain': '7 days',  # Would get from agent state
            'retraining_cost': 'medium',
            'deployment_risk': 'low',
            'performance_metrics': performance_results
        }
        
        # Get reasoning engine decision
        decision = self.reasoning_engine.reason_about_action(
            drift_results=drift_results,
            context=context
        )
        
        # Translate reasoning engine decision to plan execution decision
        should_execute = decision['action'] in ['RETRAIN', 'RETRAIN_URGENT']
        
        return {
            'should_execute': should_execute,
            'action': decision['action'],
            'reasoning': decision['reasoning'],
            'confidence': decision['confidence']
        }
    
    # Tool implementations
    
    def _tool_check_drift(self, data_path: str) -> Dict:
        """Tool: Check for data drift"""
        print(f"  📊 Checking drift in {data_path}...")
        
        try:
            import pandas as pd
            new_df = pd.read_csv(data_path)
            new_df.columns = new_df.columns.str.strip()
            
            detector = DriftDetector("training_stats.json")
            results = detector.detect_drift(new_df)
            
            print(f"  PSI: {results['overall_psi']:.4f}, Action: {results['action']}")
            return results
        except Exception as e:
            print(f"  Error: {e}")
            raise
    
    def _tool_evaluate_performance(self, predictions_path: str) -> Dict:
        """Tool: Evaluate model performance"""
        print(f"  📈 Evaluating performance from {predictions_path}...")
        
        try:
            import pandas as pd
            df = pd.read_csv(predictions_path)
            
            monitor = PerformanceMonitor()
            metrics = monitor.evaluate_predictions(
                df['y_true'].values,
                df['y_pred'].values,
                df.get('y_prob', None)
            )
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            return metrics
        except Exception as e:
            print(f"  Error: {e}")
            raise
    
    def _tool_request_labeling(self, n_samples: int, data_path: str) -> Dict:
        """Tool: Request human labeling"""
        print(f"  🏷️  Requesting labels for {n_samples} samples...")
        print(f"  📝 In production: Would send to labeling queue")
        
        return {
            'requested': n_samples,
            'status': 'pending',
            'estimated_completion': '2-3 days'
        }
    
    def _tool_retrain_model(self, base_data: str, new_data: List[str] = None) -> Dict:
        """Tool: Retrain model"""
        print(f"  🔄 Retraining model...")
        print(f"     Base data: {base_data}")
        if new_data:
            print(f"     New data: {len(new_data)} files")
        
        import subprocess
        
        cmd = ["uv", "run", "python", "training/train.py", "--base-data", base_data]
        if new_data:
            cmd.extend(["--new-data"] + new_data)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✅ Retraining completed")
            return {'success': True, 'output': result.stdout}
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Retraining failed")
            raise
    
    def _tool_validate_model(self, holdout_path: str) -> Dict:
        """Tool: Validate model on holdout set"""
        print(f"  ✓ Validating model on {holdout_path}...")
        
        # Simulation - in production would actually evaluate
        return {
            'accuracy': 0.96,
            'precision': 0.95,
            'recall': 0.97,
            'validation_passed': True
        }
    
    def _tool_deploy_model(self, model_version: str, min_accuracy: float = 0.95) -> Dict:
        """Tool: Deploy model to production"""
        print(f"  🚀 Deploying model version: {model_version}...")
        
        # Simulation
        return {
            'deployed': True,
            'version': model_version,
            'timestamp': datetime.now().isoformat()
        }
    
    def _tool_rollback_model(self, previous_version: str) -> Dict:
        """Tool: Rollback to previous model version"""
        print(f"  ⏪ Rolling back to version: {previous_version}...")
        
        return {
            'rolled_back': True,
            'version': previous_version
        }
    
    def _save_plan_log(self, goal: str, context: Dict, plan: List[PlanStep]):
        """Save plan to log file for later review"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'goal': goal,
            'context': context,
            'plan': [
                {
                    'step_id': step.step_id,
                    'tool_name': step.tool_name,
                    'description': step.description,
                    'parameters': step.parameters,
                    'dependencies': step.dependencies
                }
                for step in plan
            ]
        }
        
        log_file = Path("agent/logs/planning_plans.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def print_plan(self, plan: List[PlanStep]):
        """Print plan in readable format"""
        print("\n" + "="*60)
        print("EXECUTION PLAN")
        print("="*60)
        
        for step in plan:
            deps_str = f" (depends on: {step.dependencies})" if step.dependencies else ""
            print(f"{step.step_id}. {step.description}{deps_str}")
            print(f"   Tool: {step.tool_name}")
            print(f"   Status: {step.status.value}")
        
        print("="*60 + "\n")


def main():
    """Test the planning agent"""
    
    agent = PlanningAgent()
    
    # Create a plan
    context = {
        'latest_data_path': 'data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'base_data_path': 'data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'holdout_path': 'data/holdout/test.csv',
        'labeled_data_available': False
    }
    
    print("🎯 Creating plan: maintain_accuracy_above_0.95")
    plan = agent.create_plan("maintain_accuracy_above_0.95", context)
    
    agent.print_plan(plan)
    
    # Execute plan (dry run)
    print("Executing plan (dry run)...")
    results = agent.execute_plan(plan, dry_run=False)
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total Steps: {results['total_steps']}")
    print(f"Completed: {results['completed_steps']}")
    print(f"Failed: {results['failed_steps']}")
    print("="*60)


if __name__ == "__main__":
    main()
