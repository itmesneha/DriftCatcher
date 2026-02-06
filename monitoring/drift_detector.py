"""
Drift Detection using Population Stability Index (PSI)
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


class DriftDetector:
    """Detects data drift using PSI (Population Stability Index)"""
    
    # PSI thresholds (industry standard)
    PSI_THRESHOLD_LOW = 0.1   # < 0.1: No significant change
    PSI_THRESHOLD_MED = 0.2   # 0.1-0.2: Moderate change, monitor
    PSI_THRESHOLD_HIGH = 0.2  # > 0.2: Significant change, action needed
    
    def __init__(self, training_stats_path: str):
        """
        Initialize drift detector with training statistics
        
        Args:
            training_stats_path: Path to training_stats.json
        """
        with open(training_stats_path, 'r') as f:
            self.training_stats = json.load(f)
        
        self.feature_names = list(self.training_stats.keys())
    
    def compute_psi(self, expected_dist: np.ndarray, actual_dist: np.ndarray) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        
        Args:
            expected_dist: Distribution from training data (baseline)
            actual_dist: Distribution from new data
            
        Returns:
            PSI value
        """
        # Avoid division by zero
        expected_dist = np.where(expected_dist == 0, 0.0001, expected_dist)
        actual_dist = np.where(actual_dist == 0, 0.0001, actual_dist)
        
        psi_value = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
        
        return psi_value
    
    def compute_feature_psi(self, feature_name: str, new_data: np.ndarray) -> float:
        """
        Compute PSI for a single feature
        
        Args:
            feature_name: Name of the feature
            new_data: New/live data for this feature
            
        Returns:
            PSI value for the feature
        """
        stats = self.training_stats[feature_name]
        quantiles = np.array(stats['quantiles'])
        
        # Use quantiles as bin edges
        bins = quantiles
        
        # Expected distribution: uniform across bins (10 bins = 10% each)
        n_bins = len(bins) - 1
        expected_dist = np.full(n_bins, 1.0 / n_bins)
        
        # Actual distribution: bin the new data
        counts, _ = np.histogram(new_data, bins=bins)
        actual_dist = counts / len(new_data)
        
        psi = self.compute_psi(expected_dist, actual_dist)
        
        return psi
    
    def detect_drift(self, new_df: pd.DataFrame) -> Dict:
        """
        Detect drift across all features
        
        Args:
            new_df: DataFrame with new/live data
            
        Returns:
            Dictionary with drift results
        """
        psi_scores = {}
        drifted_features = []
        
        for feature in self.feature_names:
            if feature not in new_df.columns:
                continue
            
            # Get clean numeric data
            feature_data = new_df[feature].replace([np.inf, -np.inf], np.nan).dropna().values
            
            if len(feature_data) == 0:
                continue
            
            psi = self.compute_feature_psi(feature, feature_data)
            psi_scores[feature] = psi
            
            if psi > self.PSI_THRESHOLD_HIGH:
                drifted_features.append((feature, psi))
        
        # Aggregate drift score (mean PSI across all features)
        overall_psi = np.mean(list(psi_scores.values()))
        
        # Determine action
        action = self._determine_action(overall_psi, len(drifted_features))
        
        return {
            'overall_psi': overall_psi,
            'feature_psi': psi_scores,
            'drifted_features': sorted(drifted_features, key=lambda x: x[1], reverse=True),
            'action': action,
            'n_drifted_features': len(drifted_features),
            'total_features': len(psi_scores)
        }
    
    def _determine_action(self, overall_psi: float, n_drifted: int) -> str:
        """
        Determine recommended action based on drift scores
        
        Args:
            overall_psi: Overall drift score
            n_drifted: Number of features with significant drift
            
        Returns:
            Action recommendation: 'retrain', 'alert', or 'wait'
        """
        if overall_psi > self.PSI_THRESHOLD_HIGH or n_drifted > 5:
            return 'retrain'
        elif overall_psi > self.PSI_THRESHOLD_LOW or n_drifted > 0:
            return 'alert'
        else:
            return 'wait'
    
    def print_report(self, drift_results: Dict):
        """Print drift detection report"""
        print("\n" + "="*60)
        print("DRIFT DETECTION REPORT")
        print("="*60)
        print(f"Overall PSI: {drift_results['overall_psi']:.4f}")
        print(f"Drifted Features: {drift_results['n_drifted_features']}/{drift_results['total_features']}")
        print(f"Recommended Action: {drift_results['action'].upper()}")
        print("\n" + "-"*60)
        
        if drift_results['drifted_features']:
            print("Top Drifted Features:")
            for feature, psi in drift_results['drifted_features'][:10]:
                print(f"  {feature:40s}: PSI = {psi:.4f}")
        else:
            print("No significant drift detected.")
        
        print("="*60 + "\n")


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: uv run python monitoring/drift_detector.py <path_to_new_data.csv>")
        sys.exit(1)
    
    new_data_path = sys.argv[1]
    
    # Load new data
    print(f"Loading new data from {new_data_path}...")
    new_df = pd.read_csv(new_data_path)
    new_df.columns = new_df.columns.str.strip()
    
    # Initialize detector
    print("Initializing drift detector...")
    detector = DriftDetector("training_stats.json")
    
    # Detect drift
    print("Computing drift scores...")
    results = detector.detect_drift(new_df)
    
    # Print report
    detector.print_report(results)
    
    # Save results
    output_path = "monitoring/drift_report.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full report saved to {output_path}")


if __name__ == "__main__":
    main()
