# Agentic Architecture: How Components Work Together

## System Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    PlanningAgent                        │
│              (Strategic Multi-Step Planner)             │
│                                                          │
│  • Creates plans: "maintain_accuracy_above_0.95"        │
│  • Orchestrates tools in sequence                       │
│  • Manages dependencies between steps                   │
│                                                          │
│  CONSULTS ↓                                             │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│            AgenticReasoningEngine                       │
│         (Tactical Decision Reasoning)                   │
│                                                          │
│  • LLM-powered reasoning about specific decisions       │
│  • Considers: drift severity, context, trade-offs       │
│  • Learns from outcomes (meta-learning)                 │
│  • Returns: action + confidence + reasoning             │
│                                                          │
│  USES ↓                                                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│              Monitoring Components                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │DriftDetector │  │Performance   │  │Training      │ │
│  │(PSI-based)   │  │Monitor       │  │Pipeline      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Interaction Flow

### Example: "Maintain Accuracy Above 0.95"

```
1. USER GOAL
   ↓
   "I want to maintain model accuracy above 0.95"

2. PLANNING AGENT (Strategic)
   ↓
   Creates plan:
   Step 1: check_drift → drift_detector.detect_drift()
   Step 2: evaluate_performance → performance_monitor.evaluate()
   Step 3: retrain_model (CONDITIONAL) → training.train()
   Step 4: validate_model → validate()
   Step 5: deploy_model → deploy()

3. EXECUTION - Step 1 (check_drift)
   ↓
   DriftDetector runs, returns:
   {
     'overall_psi': 0.23,
     'action': 'RETRAIN',
     'n_drifted_features': 5,
     'top_drifted': [...]
   }

4. EXECUTION - Step 3 (CONDITIONAL - retrain_model)
   ↓
   PlanningAgent: "This step is conditional, should I execute?"
   ↓
   Consults AgenticReasoningEngine:
   
   REASONING ENGINE receives:
   - drift_results: {'overall_psi': 0.23, ...}
   - context: {'time_since_last_retrain': '7 days', ...}
   
   ↓ (LLM REASONING)
   
   LLM considers:
   • PSI 0.23 > threshold 0.2 (HIGH drift)
   • 5 features drifted significantly
   • Last retrain was 7 days ago (not recent)
   • Retraining cost: medium
   • Risk of not retraining: model degradation
   
   ↓
   
   Returns decision:
   {
     'action': 'RETRAIN',
     'confidence': 0.85,
     'reasoning': 'High drift detected (PSI 0.23) with 5 features 
                   drifted. Sufficient time since last retrain. 
                   Benefits outweigh costs.',
     'risk_assessment': 'medium'
   }

5. PLANNING AGENT
   ↓
   "Reasoning engine recommends RETRAIN with 85% confidence"
   → Executes Step 3: retrain_model()
   → Continues with Step 4, 5...

6. META-LEARNING
   ↓
   After execution completes:
   reasoning_engine.record_outcome({
     'decision': 'RETRAIN',
     'actual_result': {'accuracy_improved': True, 'new_accuracy': 0.97}
   })
   ↓
   Adapts thresholds based on outcome
```

## Key Benefits of Integration

### 1. **Separation of Concerns**

- **PlanningAgent**: "What steps are needed to achieve this goal?"
- **AgenticReasoningEngine**: "Given this situation, should I take this action?"

### 2. **Intelligent Decision Points**

Without reasoning engine:
```python
if drift_psi > 0.2:  # Hard-coded threshold
    retrain()
```

With reasoning engine:
```python
decision = reasoning_engine.reason_about_action(drift_results, context)
if decision['action'] == 'RETRAIN' and decision['confidence'] > 0.7:
    retrain()
    reasoning_engine.record_outcome(result)  # Learn from outcome
```

### 3. **Context-Aware Decisions**

The reasoning engine considers:
- **Drift severity** (PSI score)
- **Time since last retrain** (avoid thrashing)
- **Operational context** (cost, urgency, risk)
- **Historical outcomes** (learned optimal policies)

### 4. **Explainable Actions**

```python
{
  'action': 'RETRAIN',
  'confidence': 0.85,
  'reasoning': 'High drift detected (PSI 0.23) affecting 5 critical 
                features. Last retrain was 7 days ago, providing 
                sufficient cooldown. Historical data shows retraining 
                at this drift level improves accuracy by avg 3.2%. 
                Medium cost justified by high benefit.'
}
```

## Code Examples

### Basic Usage

```python
# Initialize both agents
planner = PlanningAgent(use_reasoning_engine=True)

# Create a plan
context = {
    'latest_data_path': 'data/new_batch.csv',
    'base_data_path': 'data/training.csv',
    'holdout_path': 'data/processed/holdout.csv'
}

plan = planner.create_plan("maintain_accuracy_above_0.95", context)

# Execute with intelligent decision-making
results = planner.execute_plan(plan)
```

### What Happens Under the Hood

```python
# During execution, at conditional step:
def execute_plan(plan):
    for step in plan:
        if step.parameters.get('conditional'):
            # Gather context from previous steps
            drift_results = previous_drift_step.result
            
            # Consult reasoning engine
            decision = self.reasoning_engine.reason_about_action(
                drift_results=drift_results,
                context={'time_since_retrain': '7 days', ...}
            )
            
            if decision['action'] in ['RETRAIN', 'RETRAIN_URGENT']:
                execute_tool(step)  # Go ahead with retraining
            else:
                skip_step(step)  # Skip, reasoning says not needed
```

## When to Use Each Component

### Use ReasoningEngine Directly

When you need to make a **single decision**:
- "Should I retrain now given this drift?"
- "Is this alert urgent or can it wait?"
- "Should I request more labeled samples?"

```python
from agent.AgenticReasoningEngine import AgenticReasoningEngine

engine = AgenticReasoningEngine()
decision = engine.reason_about_action(drift_results, context)
```

### Use PlanningAgent

When you need to **achieve a complex goal** with multiple steps:
- "Maintain model accuracy above 95%"
- "Handle detected drift proactively"
- "Retrain safely with validation"

```python
from agent.PlanningAgent import PlanningAgent

planner = PlanningAgent()
plan = planner.create_plan("maintain_accuracy_above_0.95", context)
planner.execute_plan(plan)
```

### Use Both (Recommended)

The PlanningAgent **automatically uses** the ReasoningEngine for conditional steps:

```python
# This uses both!
planner = PlanningAgent(use_reasoning_engine=True)
plan = planner.create_plan("maintain_accuracy_above_0.95", context)
results = planner.execute_plan(plan)

# The planner will consult the reasoning engine at decision points
```

## Real-World Scenario

### Scenario: Drift Detected on Production Traffic

```
9:00 AM - Drift monitoring runs
         → DriftDetector: PSI = 0.18 (MODERATE)

9:01 AM - PlanningAgent creates plan: "handle_drift_proactively"
         → Step 1: check_drift ✓
         → Step 2: request_labeling (conditional)
         → Step 3: retrain_model (conditional)
         → Step 4: validate_model
         → Step 5: deploy_model

9:02 AM - Step 2 decision point
         → ReasoningEngine: "PSI 0.18 is moderate. Only 3 features 
                            drifted. We retrained 2 days ago. 
                            RECOMMEND: ALERT but don't retrain yet.
                            Request 200 samples for labeling."
         → PlanningAgent: Executes request_labeling(200)

9:05 AM - Step 3 decision point  
         → ReasoningEngine: "Still early, labels not ready.
                            RECOMMEND: SKIP retraining for now."
         → PlanningAgent: Skips retrain_model

2 DAYS LATER

9:00 AM - Drift monitoring runs again
         → DriftDetector: PSI = 0.24 (HIGH)

9:01 AM - PlanningAgent creates new plan
         → Step 1: check_drift ✓
         → Step 3: retrain_model (conditional)

9:02 AM - Step 3 decision point
         → ReasoningEngine: "PSI increased to 0.24 (HIGH). 
                            4 days since last retrain. 
                            200 labeled samples available.
                            RECOMMEND: RETRAIN_URGENT."
         → PlanningAgent: Executes retrain_model with new data
         
9:15 AM - Model retrained, validated, deployed ✓
```

## Summary

| Component | Role | Scope | Intelligence |
|-----------|------|-------|--------------|
| **AgenticReasoningEngine** | Tactical decision-maker | Single decisions | LLM reasoning + meta-learning |
| **PlanningAgent** | Strategic planner | Multi-step goals | Plan creation + orchestration |
| **Together** | Autonomous ML ops | Complex scenarios | Strategic planning + intelligent execution |

The **PlanningAgent** is the "project manager" that knows the big picture.
The **AgenticReasoningEngine** is the "expert advisor" that makes smart calls at critical moments.

Together, they create a truly agentic system that can:
- ✅ Plan complex multi-step operations
- ✅ Make context-aware decisions
- ✅ Learn from outcomes
- ✅ Explain its reasoning
- ✅ Operate autonomously
