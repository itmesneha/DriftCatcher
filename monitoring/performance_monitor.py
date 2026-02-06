"""
Model Performance Monitoring
Tracks actual model performance when labels are available
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import mlflow


class PerformanceMonitor:
    """Monitors model performance with labeled production data"""
    
    # Performance thresholds
    ACCURACY_THRESHOLD = 0.90
    PRECISION_THRESHOLD = 0.85
    RECALL_THRESHOLD = 0.85
    
    def __init__(self, performance_log_path: str = "monitoring/performance_log.json"):
        """
        Initialize performance monitor
        
        Args:
            performance_log_path: Path to save performance logs
        """
        self.performance_log_path = performance_log_path
    
    def evaluate_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray = None
    ) -> dict:
        """
        Evaluate model predictions against ground truth
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC)
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred))
        }
        
        # Add AUC if probabilities provided
        if y_prob is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        # Calculate rates
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def log_performance(self, metrics: dict):
        """Log performance metrics to file"""
        with open(self.performance_log_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def check_degradation(self, metrics: dict) -> dict:
        """
        Check if performance has degraded below thresholds
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Dictionary with degradation status and recommendations
        """
        issues = []
        
        if metrics['accuracy'] < self.ACCURACY_THRESHOLD:
            issues.append(f"Accuracy dropped to {metrics['accuracy']:.3f} (threshold: {self.ACCURACY_THRESHOLD})")
        
        if metrics['precision'] < self.PRECISION_THRESHOLD:
            issues.append(f"Precision dropped to {metrics['precision']:.3f} (threshold: {self.PRECISION_THRESHOLD})")
        
        if metrics['recall'] < self.RECALL_THRESHOLD:
            issues.append(f"Recall dropped to {metrics['recall']:.3f} (threshold: {self.RECALL_THRESHOLD})")
        
        # High false positive rate is critical for DDoS detection
        if metrics['false_positive_rate'] > 0.10:
            issues.append(f"High false positive rate: {metrics['false_positive_rate']:.3f}")
        
        # High false negative rate means missing attacks
        if metrics['false_negative_rate'] > 0.05:
            issues.append(f"High false negative rate: {metrics['false_negative_rate']:.3f}")
        
        return {
            'degraded': len(issues) > 0,
            'issues': issues,
            'recommendation': 'retrain' if len(issues) > 0 else 'continue'
        }
    
    def log_to_mlflow(self, metrics: dict, run_name: str = "production_monitoring"):
        """Log performance metrics to MLflow"""
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("prod_accuracy", metrics['accuracy'])
            mlflow.log_metric("prod_precision", metrics['precision'])
            mlflow.log_metric("prod_recall", metrics['recall'])
            mlflow.log_metric("prod_f1_score", metrics['f1_score'])
            mlflow.log_metric("prod_false_positive_rate", metrics['false_positive_rate'])
            mlflow.log_metric("prod_false_negative_rate", metrics['false_negative_rate'])
            
            if 'roc_auc' in metrics:
                mlflow.log_metric("prod_roc_auc", metrics['roc_auc'])
    
    def print_report(self, metrics: dict, degradation: dict):
        """Print performance report"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE REPORT")
        print("="*60)
        print(f"Timestamp:       {metrics['timestamp']}")
        print(f"Samples:         {metrics['n_samples']}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f} {'✓' if metrics['accuracy'] >= self.ACCURACY_THRESHOLD else '✗ LOW'}")
        print(f"  Precision:     {metrics['precision']:.4f} {'✓' if metrics['precision'] >= self.PRECISION_THRESHOLD else '✗ LOW'}")
        print(f"  Recall:        {metrics['recall']:.4f} {'✓' if metrics['recall'] >= self.RECALL_THRESHOLD else '✗ LOW'}")
        print(f"  F1-Score:      {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
        
        print(f"\nError Rates:")
        print(f"  FP Rate:       {metrics['false_positive_rate']:.4f}")
        print(f"  FN Rate:       {metrics['false_negative_rate']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TN: {cm['true_negatives']:>6}  FP: {cm['false_positives']:>6}")
        print(f"  FN: {cm['false_negatives']:>6}  TP: {cm['true_positives']:>6}")
        
        print(f"\nStatus: {'⚠️  DEGRADED' if degradation['degraded'] else '✅ HEALTHY'}")
        
        if degradation['degraded']:
            print(f"\nIssues Detected:")
            for issue in degradation['issues']:
                print(f"  - {issue}")
            print(f"\nRecommendation: {degradation['recommendation'].upper()}")
        
        print("="*60 + "\n")


def main():
    """Example usage with labeled production data"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: uv run python monitoring/performance_monitor.py <predictions_csv>")
        print("\nCSV should have columns: y_true, y_pred, y_prob (optional)")
        sys.exit(1)
    
    predictions_path = sys.argv[1]
    
    # Load predictions and true labels
    print(f"Loading predictions from {predictions_path}...")
    df = pd.read_csv(predictions_path)
    
    if 'y_true' not in df.columns or 'y_pred' not in df.columns:
        print("Error: CSV must contain 'y_true' and 'y_pred' columns")
        sys.exit(1)
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    y_prob = df['y_prob'].values if 'y_prob' in df.columns else None
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Evaluate
    print("Evaluating model performance...")
    metrics = monitor.evaluate_predictions(y_true, y_pred, y_prob)
    
    # Check for degradation
    degradation = monitor.check_degradation(metrics)
    
    # Log
    monitor.log_performance(metrics)
    
    # Log to MLflow (optional)
    try:
        monitor.log_to_mlflow(metrics)
        print("Logged to MLflow successfully")
    except Exception as e:
        print(f"Warning: Could not log to MLflow: {e}")
    
    # Print report
    monitor.print_report(metrics, degradation)


if __name__ == "__main__":
    main()
