"""
Automated Retraining Agent with State Management
Monitors drift and triggers retraining intelligently
"""
import json
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.drift_detector import DriftDetector


class RetrainAgent:
    """Agent that monitors drift and manages model retraining"""
    
    def __init__(
        self,
        training_stats_path: str = "artifacts/training_stats.json",
        state_file: str = "agent/agent_state.json",
        min_retrain_interval_hours: int = 24,
        alert_cooldown_hours: int = 6
    ):
        """
        Initialize retraining agent
        
        Args:
            training_stats_path: Path to training statistics
            state_file: Path to save agent state
            min_retrain_interval_hours: Minimum hours between retrains
            alert_cooldown_hours: Minimum hours between alerts
        """
        self.training_stats_path = training_stats_path
        self.state_file = state_file
        self.min_retrain_interval = timedelta(hours=min_retrain_interval_hours)
        self.alert_cooldown = timedelta(hours=alert_cooldown_hours)
        
        # Initialize drift detector
        self.drift_detector = DriftDetector(training_stats_path)
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Create logs directory
        os.makedirs("agent/logs", exist_ok=True)
    
    def _load_state(self) -> dict:
        """Load agent state from disk"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                # Convert ISO strings back to datetime
                if 'last_retrain_time' in state and state['last_retrain_time']:
                    state['last_retrain_time'] = datetime.fromisoformat(state['last_retrain_time'])
                if 'last_alert_time' in state and state['last_alert_time']:
                    state['last_alert_time'] = datetime.fromisoformat(state['last_alert_time'])
                return state
        
        # Default state
        return {
            'last_retrain_time': None,
            'last_alert_time': None,
            'total_retrains': 0,
            'total_alerts': 0,
            'total_checks': 0,
            'drift_history': []
        }
    
    def _save_state(self):
        """Save agent state to disk"""
        state_copy = self.state.copy()
        
        # Convert datetime to ISO format for JSON
        if state_copy['last_retrain_time']:
            state_copy['last_retrain_time'] = state_copy['last_retrain_time'].isoformat()
        if state_copy['last_alert_time']:
            state_copy['last_alert_time'] = state_copy['last_alert_time'].isoformat()
        
        # Keep only last 100 drift history entries
        state_copy['drift_history'] = state_copy['drift_history'][-100:]
        
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state_copy, f, indent=2)
    
    def _can_retrain(self) -> bool:
        """Check if enough time has passed since last retrain"""
        if self.state['last_retrain_time'] is None:
            return True
        
        time_since_retrain = datetime.now() - self.state['last_retrain_time']
        return time_since_retrain >= self.min_retrain_interval
    
    def _can_alert(self) -> bool:
        """Check if enough time has passed since last alert"""
        if self.state['last_alert_time'] is None:
            return True
        
        time_since_alert = datetime.now() - self.state['last_alert_time']
        return time_since_alert >= self.alert_cooldown
    
    def _log_event(self, event_type: str, message: str, drift_results: dict = None):
        """Log event to file"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'message': message
        }
        
        if drift_results:
            log_entry['drift_results'] = {
                'overall_psi': drift_results['overall_psi'],
                'n_drifted_features': drift_results['n_drifted_features'],
                'action': drift_results['action']
            }
        
        # Append to daily log file
        log_file = f"agent/logs/agent_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"[{timestamp}] {event_type}: {message}")
    
    def _trigger_retrain(self, new_data_paths: list = None):
        """
        Trigger model retraining
        
        Args:
            new_data_paths: List of paths to new labeled data files to include in retraining
        """
        if new_data_paths:
            self._log_event("RETRAIN_START", f"Starting retraining with {len(new_data_paths)} new dataset(s)...")
        else:
            self._log_event("RETRAIN_START", "Starting model retraining (base data only)...")
        
        try:
            # Build command - use train_universal.py with MLflow logging
            cmd = ["uv", "run", "python", "training/train_universal.py", "--base-data", "data/baseline.csv"]
            
            # Add new data paths if provided
            if new_data_paths:
                cmd.extend(["--new-data"] + new_data_paths)
            
            # Run training script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Update state
            self.state['last_retrain_time'] = datetime.now()
            self.state['total_retrains'] += 1
            self._save_state()
            
            self._log_event("RETRAIN_SUCCESS", f"Model retrained successfully (Total retrains: {self.state['total_retrains']})")
            return True
            
        except subprocess.CalledProcessError as e:
            self._log_event("RETRAIN_FAILED", f"Retraining failed: {e.stderr}")
            return False
    
    def _send_alert(self, drift_results: dict):
        """Send drift alert"""
        message = (
            f"Drift detected! Overall PSI: {drift_results['overall_psi']:.4f}, "
            f"Drifted features: {drift_results['n_drifted_features']}"
        )
        
        self._log_event("ALERT", message, drift_results)
        
        # Update state
        self.state['last_alert_time'] = datetime.now()
        self.state['total_alerts'] += 1
        self._save_state()
        
        # TODO: Add actual alerting (email, Slack, PagerDuty, etc.)
        # For now, just log it
    
    def check_and_act(self, new_data_path: str) -> dict:
        """
        Check for drift and take appropriate action
        
        Args:
            new_data_path: Path to new/live data CSV
            
        Returns:
            Dictionary with action taken and drift results
        """
        self.state['total_checks'] += 1
        
        # Load new data
        self._log_event("CHECK_START", f"Checking drift for: {new_data_path}")
        
        try:
            new_df = pd.read_csv(new_data_path)
            new_df.columns = new_df.columns.str.strip()
        except Exception as e:
            self._log_event("ERROR", f"Failed to load data: {str(e)}")
            return {'action': 'error', 'message': str(e)}
        
        # Detect drift
        drift_results = self.drift_detector.detect_drift(new_df)
        
        # Log drift to MLflow
        try:
            self.drift_detector.log_drift_to_mlflow(drift_results)
        except Exception as e:
            self._log_event("WARNING", f"Could not log drift to MLflow: {e}")
        
        # Record drift in history
        drift_history_entry = {
            'timestamp': datetime.now().isoformat(),
            'overall_psi': drift_results['overall_psi'],
            'n_drifted_features': drift_results['n_drifted_features'],
            'action_recommended': drift_results['action']
        }
        self.state['drift_history'].append(drift_history_entry)
        
        # Decide action based on drift and state
        action_taken = None
        
        if drift_results['action'] == 'retrain':
            if self._can_retrain():
                self._trigger_retrain()
                action_taken = 'retrained'
            else:
                time_since = datetime.now() - self.state['last_retrain_time']
                hours_left = (self.min_retrain_interval - time_since).total_seconds() / 3600
                message = f"Retrain needed but in cooldown ({hours_left:.1f}h remaining)"
                self._log_event("RETRAIN_COOLDOWN", message, drift_results)
                
                # Send alert instead
                if self._can_alert():
                    self._send_alert(drift_results)
                    action_taken = 'alert_sent'
                else:
                    action_taken = 'cooldown'
        
        elif drift_results['action'] == 'alert':
            if self._can_alert():
                self._send_alert(drift_results)
                action_taken = 'alert_sent'
            else:
                self._log_event("ALERT_COOLDOWN", "Alert needed but in cooldown period")
                action_taken = 'cooldown'
        
        else:  # wait
            self._log_event("CHECK_OK", f"No significant drift detected (PSI: {drift_results['overall_psi']:.4f})")
            action_taken = 'no_action'
        
        self._save_state()
        
        return {
            'action': action_taken,
            'drift_results': drift_results,
            'state': self.get_status()
        }
    
    def get_status(self) -> dict:
        """Get agent status"""
        return {
            'total_checks': self.state['total_checks'],
            'total_retrains': self.state['total_retrains'],
            'total_alerts': self.state['total_alerts'],
            'last_retrain_time': self.state['last_retrain_time'].isoformat() if self.state['last_retrain_time'] else None,
            'last_alert_time': self.state['last_alert_time'].isoformat() if self.state['last_alert_time'] else None,
            'can_retrain': self._can_retrain(),
            'can_alert': self._can_alert()
        }
    
    def print_status(self):
        """Print agent status"""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("RETRAINING AGENT STATUS")
        print("="*60)
        print(f"Total Checks:    {status['total_checks']}")
        print(f"Total Retrains:  {status['total_retrains']}")
        print(f"Total Alerts:    {status['total_alerts']}")
        print(f"Last Retrain:    {status['last_retrain_time'] or 'Never'}")
        print(f"Last Alert:      {status['last_alert_time'] or 'Never'}")
        print(f"Can Retrain:     {'Yes' if status['can_retrain'] else 'No (in cooldown)'}")
        print(f"Can Alert:       {'Yes' if status['can_alert'] else 'No (in cooldown)'}")
        print("="*60 + "\n")


def main():
    """CLI interface for the agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Retraining Agent")
    parser.add_argument('command', choices=['check', 'status', 'retrain'], help='Command to run')
    parser.add_argument('--data', help='Path to new data CSV (required for check)')
    parser.add_argument('--new-labeled-data', nargs='+', help='Path(s) to new labeled data for retraining')
    parser.add_argument('--min-retrain-hours', type=int, default=24, help='Minimum hours between retrains')
    parser.add_argument('--alert-cooldown-hours', type=int, default=6, help='Minimum hours between alerts')
    parser.add_argument('--force', action='store_true', help='Force retrain even if in cooldown')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = RetrainAgent(
        min_retrain_interval_hours=args.min_retrain_hours,
        alert_cooldown_hours=args.alert_cooldown_hours
    )
    
    if args.command == 'status':
        agent.print_status()
        
        # Show recent drift history
        if agent.state['drift_history']:
            print("Recent Drift History (last 5):")
            for entry in agent.state['drift_history'][-5:]:
                print(f"  {entry['timestamp']}: PSI={entry['overall_psi']:.4f}, "
                      f"Drifted={entry['n_drifted_features']}, "
                      f"Action={entry['action_recommended']}")
    
    elif args.command == 'retrain':
        # Manual retrain with new labeled data
        if not args.new_labeled_data:
            print("Error: --new-labeled-data is required for retrain command")
            print("Example: python agent/retrain_agent.py retrain --new-labeled-data data/labeled/batch1.csv data/labeled/batch2.csv")
            sys.exit(1)
        
        if not args.force and not agent._can_retrain():
            print("Error: Cannot retrain yet (in cooldown period)")
            print("Use --force to override, or wait until cooldown expires")
            agent.print_status()
            sys.exit(1)
        
        print(f"Triggering manual retrain with {len(args.new_labeled_data)} new dataset(s)...")
        success = agent._trigger_retrain(new_data_paths=args.new_labeled_data)
        
        if success:
            print("✅ Retraining completed successfully!")
        else:
            print("❌ Retraining failed. Check logs for details.")
    
    elif args.command == 'check':
        if not args.data:
            print("Error: --data is required for check command")
            sys.exit(1)
        
        result = agent.check_and_act(args.data)
        
        print("\n" + "="*60)
        print("AGENT ACTION SUMMARY")
        print("="*60)
        print(f"Action Taken: {result['action']}")
        
        if 'drift_results' in result:
            agent.drift_detector.print_report(result['drift_results'])
        
        agent.print_status()


if __name__ == "__main__":
    main()
