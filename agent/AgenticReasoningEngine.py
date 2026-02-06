"""
Agentic Reasoning Engine with LLM-based decision making
Learns optimal policies and reasons about complex trade-offs
"""
import os
import json
import requests
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import mlflow

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory
Path("agent/logs").mkdir(parents=True, exist_ok=True)


class AgenticReasoningEngine:
    """LLM-powered reasoning engine for drift response decisions"""
    
    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        learning_rate: float = 0.1
    ):
        """
        Initialize agentic reasoning engine
        
        Args:
            model: OpenRouter model to use
            learning_rate: Learning rate for threshold adaptation
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.use_llm = self.api_key is not None
        
        # Meta-learning state
        self.psi_threshold_low = 0.1
        self.psi_threshold_high = 0.2
        self.learning_rate = learning_rate
        self.outcome_history = []
        
        if not self.use_llm:
            print("⚠️  OpenRouter API key not found. Using rule-based fallback.")
            print("   Set OPENROUTER_API_KEY in .env for LLM reasoning.")
        # Use cheap, fast model for demo
        self.model = "meta-llama/llama-3.1-8b-instruct:free"  # FREE!
        # Or upgrade: "anthropic/claude-3.5-sonnet" for better reasoning
    
    def reason_about_action(
        self,
        drift_results: Dict,
        context: Dict
    ) -> Dict:
        """
        Reason about what action to take given drift and context
        
        Args:
            drift_results: Drift detection results
            context: Operational context (budget, time since retrain, etc.)
            
        Returns:
            Decision dictionary with action, confidence, and reasoning
        """
        logger.info("="*60)
        logger.info("AGENTIC REASONING ENGINE - DECISION REQUEST")
        logger.info("="*60)
        logger.info(f"Drift PSI: {drift_results.get('overall_psi', 'N/A')}")
        logger.info(f"Drifted features: {drift_results.get('n_drifted_features', 'N/A')}")
        logger.info(f"Context: {context}")
        
        if self.use_llm:
            decision = self._llm_reasoning(drift_results, context)
        else:
            decision = self._rule_based_reasoning(drift_results, context)
        
        # Log decision
        logger.info("\n" + "="*60)
        logger.info("DECISION MADE")
        logger.info("="*60)
        logger.info(f"Action: {decision['action']}")
        logger.info(f"Confidence: {decision['confidence']:.2f}")
        logger.info(f"Reasoning: {decision['reasoning']}")
        logger.info(f"Risk: {decision.get('risk_assessment', 'N/A')}")
        logger.info("="*60 + "\n")
        
        # Save to file for review
        self._save_decision_log(drift_results, context, decision)
        
        # Log to MLflow
        self._log_decision_to_mlflow(drift_results, context, decision)
        
        return decision
    
    def _log_decision_to_mlflow(
        self,
        drift_results: Dict,
        context: Dict,
        decision: Dict
    ):
        """Log reasoning engine decision to MLflow"""
        try:
            mlflow.set_experiment("agentic_reasoning")
            
            with mlflow.start_run(run_name=f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log input metrics
                mlflow.log_metric("drift_psi", drift_results.get('overall_psi', 0))
                mlflow.log_metric("n_drifted_features", drift_results.get('n_drifted_features', 0))
                
                # Log decision
                mlflow.log_param("action", decision['action'])
                mlflow.log_metric("confidence", decision['confidence'])
                mlflow.log_param("reasoning_type", decision.get('reasoning_type', 'rule-based'))
                
                if decision.get('model'):
                    mlflow.log_param("llm_model", decision['model'])
                
                # Log thresholds (meta-learning state)
                mlflow.log_metric("psi_threshold_low", self.psi_threshold_low)
                mlflow.log_metric("psi_threshold_high", self.psi_threshold_high)
                
                # Log context
                mlflow.log_param("time_since_retrain", context.get('time_since_last_retrain', 'unknown'))
                mlflow.log_param("retraining_cost", context.get('retraining_cost', 'unknown'))
                
                # Log full reasoning as text artifact
                mlflow.log_text(decision['reasoning'], "reasoning.txt")
                
                # Log full decision as JSON
                mlflow.log_dict({
                    'drift_results': drift_results,
                    'context': context,
                    'decision': decision,
                    'thresholds': {
                        'psi_low': self.psi_threshold_low,
                        'psi_high': self.psi_threshold_high
                    }
                }, "decision_details.json")
                
                # Set tags for easy filtering
                mlflow.set_tag("agent_type", "reasoning_engine")
                mlflow.set_tag("action", decision['action'])
                mlflow.set_tag("reasoning_type", decision.get('reasoning_type', 'rule-based'))
                
                logger.info("✅ Decision logged to MLflow experiment 'agentic_reasoning'")
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def _llm_reasoning(
        self,
        drift_results: Dict,
        context: Dict
    ) -> Dict:
        """Use LLM to reason about the situation"""
        
        prompt = self._build_reasoning_prompt(drift_results, context)
        
        logger.info("🧠 Using LLM-based reasoning...")
        logger.info(f"Model: {self.model}")
        logger.debug(f"\nPrompt sent to LLM:\n{prompt}\n")
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/driftcatcher",
                    "X-Title": "DriftCatcher ML Agent"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            llm_response = result['choices'][0]['message']['content']
            logger.info(f"\n📝 LLM Response:\n{llm_response}\n")
            
            decision = self._parse_llm_response(llm_response)
            decision['reasoning_type'] = 'llm'
            decision['model'] = self.model
            
            logger.info("✅ LLM reasoning successful")
            return decision
            
        except Exception as e:
            logger.warning(f"⚠️  LLM reasoning failed: {e}")
            logger.info("   Falling back to rule-based reasoning")
            return self._rule_based_reasoning(drift_results, context)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt defining agent's role"""
        return """You are an autonomous ML operations agent managing a DDoS detection model in production.

Your responsibilities:
- Monitor data drift and model performance
- Decide when to retrain, alert, or wait
- Balance model accuracy, cost, and operational constraints
- Reason about trade-offs and uncertainties

Key principles:
- Missing attacks (false negatives) is WORSE than false alarms
- Retraining is expensive but necessary when drift is severe
- Consider time, budget, and recent performance
- Be decisive but explain your reasoning clearly

You must provide structured decisions that can be parsed and executed."""
    
    def _build_reasoning_prompt(
        self,
        drift_results: Dict,
        context: Dict
    ) -> str:
        """Build detailed reasoning prompt"""
        
        # Format drifted features
        drifted_features_str = "\n".join([
            f"  - {feature}: PSI = {psi:.4f}"
            for feature, psi in drift_results['drifted_features'][:5]
        ]) if drift_results['drifted_features'] else "  None"
        
        prompt = f"""**Current Situation:**

Drift Metrics:
- Overall PSI Score: {drift_results['overall_psi']:.4f}
- Drifted Features: {drift_results['n_drifted_features']}/{drift_results['total_features']}
- Automatic Recommendation: {drift_results['action'].upper()}

Operational Context:
- Hours Since Last Retrain: {context.get('hours_since_retrain', 'Never')}
- Can Retrain Now: {context.get('can_retrain', True)}
- Remaining Budget: ${context.get('remaining_budget', 1000)}
- Recent Accuracy: {context.get('recent_accuracy', 'Unknown')}

Top Drifted Features:
{drifted_features_str}

Learned Thresholds (adaptive):
- PSI Low Threshold: {self.psi_threshold_low:.3f}
- PSI High Threshold: {self.psi_threshold_high:.3f}

**Decision Options:**

1. RETRAIN
   - Trigger immediate model retraining
   - Cost: $50, Time: ~2 hours
   - Requires: Available budget + cooldown expired
   
2. ALERT
   - Send alert to operations team
   - Cost: $0, requires manual review
   - Use when: Drift moderate but uncertain
   
3. WAIT
   - Continue monitoring
   - Cost: $0, but risk of degradation
   - Use when: Drift minimal

**Constraints:**
- Must wait 24h between retrains (cooldown)
- Production SLA: 95%+ accuracy required
- CRITICAL: False negatives (missed attacks) > False positives

**Task:**
Analyze the situation and decide ONE action. Consider:
- Is drift severe enough to impact accuracy?
- Is it better to act now or gather more data?
- What are the risks of each option?

Provide your response in this EXACT format:
ACTION: [RETRAIN|ALERT|WAIT]
CONFIDENCE: [0-100]
REASONING: [2-3 sentences explaining your decision]
RISK: [HIGH|MEDIUM|LOW]"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse structured LLM response into decision dict"""
        
        decision = {
            'action': 'wait',
            'confidence': 50,
            'reasoning': 'Unable to parse LLM response',
            'risk': 'medium',
            'raw_response': response
        }
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('ACTION:'):
                    action = line.split(':', 1)[1].strip().lower()
                    if action in ['retrain', 'alert', 'wait']:
                        decision['action'] = action
                
                elif line.startswith('CONFIDENCE:'):
                    conf_str = line.split(':', 1)[1].strip().rstrip('%')
                    try:
                        decision['confidence'] = int(conf_str)
                    except ValueError:
                        pass
                
                elif line.startswith('REASONING:'):
                    decision['reasoning'] = line.split(':', 1)[1].strip()
                
                elif line.startswith('RISK:'):
                    risk = line.split(':', 1)[1].strip().lower()
                    if risk in ['high', 'medium', 'low']:
                        decision['risk'] = risk
        
        except Exception as e:
            print(f"⚠️  Error parsing LLM response: {e}")
        
        return decision
    
    def _rule_based_reasoning(
        self,
        drift_results: Dict,
        context: Dict
    ) -> Dict:
        """Fallback rule-based reasoning with adaptive thresholds"""
        
        overall_psi = drift_results['overall_psi']
        n_drifted = drift_results['n_drifted_features']
        can_retrain = context.get('can_retrain', True)
        
        # Use learned thresholds
        if overall_psi > self.psi_threshold_high or n_drifted > 5:
            if can_retrain:
                return {
                    'action': 'retrain',
                    'confidence': 85,
                    'reasoning': f'High drift detected (PSI: {overall_psi:.3f}). Immediate retraining recommended.',
                    'risk': 'high',
                    'reasoning_type': 'rule_based_adaptive'
                }
            else:
                return {
                    'action': 'alert',
                    'confidence': 75,
                    'reasoning': 'High drift but in retrain cooldown. Alerting for manual review.',
                    'risk': 'high',
                    'reasoning_type': 'rule_based_adaptive'
                }
        
        elif overall_psi > self.psi_threshold_low or n_drifted > 0:
            return {
                'action': 'alert',
                'confidence': 70,
                'reasoning': f'Moderate drift detected (PSI: {overall_psi:.3f}). Monitoring recommended.',
                'risk': 'medium',
                'reasoning_type': 'rule_based_adaptive'
            }
        
        else:
            return {
                'action': 'wait',
                'confidence': 90,
                'reasoning': f'Minimal drift (PSI: {overall_psi:.3f}). Continue normal operations.',
                'risk': 'low',
                'reasoning_type': 'rule_based_adaptive'
            }
    
    def record_outcome(
        self,
        decision: Dict,
        drift_psi: float,
        outcome: Dict
    ):
        """
        Record outcome of a decision for meta-learning
        
        Args:
            decision: The decision that was made
            drift_psi: PSI at time of decision
            outcome: Results after action (accuracy change, cost, etc.)
        """
        outcome_record = {
            'timestamp': datetime.now().isoformat(),
            'action': decision['action'],
            'drift_psi': drift_psi,
            'thresholds': {
                'low': self.psi_threshold_low,
                'high': self.psi_threshold_high
            },
            'outcome': outcome
        }
        
        self.outcome_history.append(outcome_record)
        
        # Keep last 100 outcomes
        if len(self.outcome_history) > 100:
            self.outcome_history = self.outcome_history[-100:]
    
    def adapt_thresholds(self):
        """
        Meta-learning: Adapt PSI thresholds based on outcome history
        
        Learning strategy:
        - If we retrained and performance didn't improve much → increase threshold (less aggressive)
        - If we waited and performance degraded → decrease threshold (more aggressive)
        """
        
        if len(self.outcome_history) < 5:
            return  # Need more data
        
        recent_outcomes = self.outcome_history[-10:]
        
        # Analyze retrain decisions
        retrains = [o for o in recent_outcomes if o['action'] == 'retrain']
        for retrain in retrains:
            accuracy_gain = retrain['outcome'].get('accuracy_gain', 0)
            drift_psi = retrain['drift_psi']
            
            if accuracy_gain < 0.01:  # Retrained but minimal improvement
                # Threshold was too low, increase it
                adjustment = self.learning_rate * (drift_psi - self.psi_threshold_high)
                self.psi_threshold_high = min(0.3, self.psi_threshold_high + adjustment)
        
        # Analyze wait decisions
        waits = [o for o in recent_outcomes if o['action'] == 'wait']
        for wait in waits:
            accuracy_loss = wait['outcome'].get('accuracy_loss', 0)
            drift_psi = wait['drift_psi']
            
            if accuracy_loss > 0.05:  # Waited and performance dropped significantly
                # Threshold was too high, decrease it
                adjustment = self.learning_rate * (self.psi_threshold_high - drift_psi)
                self.psi_threshold_high = max(0.1, self.psi_threshold_high - adjustment)
        
        # Keep low threshold at half of high
        self.psi_threshold_low = self.psi_threshold_high / 2
        
        print(f"📊 Thresholds adapted: Low={self.psi_threshold_low:.3f}, High={self.psi_threshold_high:.3f}")
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress"""
        return {
            'current_thresholds': {
                'low': self.psi_threshold_low,
                'high': self.psi_threshold_high
            },
            'total_outcomes': len(self.outcome_history),
            'recent_actions': [o['action'] for o in self.outcome_history[-10:]],
            'using_llm': self.use_llm,
            'model': self.model if self.use_llm else 'rule_based'
        }


def main():
    """Test the reasoning engine"""
    
    # Initialize engine
    engine = AgenticReasoningEngine()
    
    # Example drift scenario
    drift_results = {
        'overall_psi': 0.25,
        'n_drifted_features': 3,
        'total_features': 78,
        'drifted_features': [
            ('Flow Duration', 0.35),
            ('Packet Length Mean', 0.28),
            ('Flow Bytes/s', 0.22)
        ],
        'action': 'retrain'
    }
    
    context = {
        'hours_since_retrain': 48,
        'can_retrain': True,
        'remaining_budget': 500,
        'recent_accuracy': 0.94
    }
    
    # Get decision
    print("🤖 Reasoning about drift...")
    decision = engine.reason_about_action(drift_results, context)
    
    print("\n" + "="*60)
    print("AGENTIC DECISION")
    print("="*60)
    print(f"Action: {decision['action'].upper()}")
    print(f"Confidence: {decision['confidence']}%")
    print(f"Risk Level: {decision['risk'].upper()}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"Method: {decision.get('reasoning_type', 'unknown')}")
    print("="*60)
    
    # Simulate outcome and adapt
    if decision['action'] == 'retrain':
        outcome = {'accuracy_gain': 0.03, 'cost': 50}
        engine.record_outcome(decision, drift_results['overall_psi'], outcome)
        engine.adapt_thresholds()
    
    # Show learning summary
    print("\n📊 Learning Summary:")
    summary = engine.get_learning_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
