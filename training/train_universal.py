"""
Universal Training Script for DriftCatcher
Works with any tabular dataset using UniversalDataLoader
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import mlflow
import mlflow.sklearn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.universal_loader import UniversalDataLoader
from config.dataset_config import load_dataset_config, PREDEFINED_CONFIGS


def train_model(
    csv_path: str,
    dataset_name: str = "Generic",
    model_name: str = None,
    artifact_dir: str = "artifacts",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
):
    """
    Train a Random Forest model on any dataset
    
    Args:
        csv_path: Path to CSV file
        dataset_name: Name of predefined config or 'Generic' for auto-detection
        model_name: Name for MLflow model (defaults to dataset_name + '_model')
        artifact_dir: Directory to save artifacts
        test_size: Test set size
        n_estimators: Number of trees in Random Forest
        max_depth: Max depth of trees
        random_state: Random seed
    """
    os.makedirs(artifact_dir, exist_ok=True)
    
    if model_name is None:
        model_name = f"{dataset_name.lower()}_model"
    
    print(f"\n{'='*80}")
    print(f"🎯 DriftCatcher Universal Training")
    print(f"{'='*80}")
    print(f"📊 Dataset: {csv_path}")
    print(f"📝 Config: {dataset_name}")
    print(f"🏷️  Model: {model_name}")
    print(f"{'='*80}\n")
    
    # Load dataset configuration
    config = load_dataset_config(dataset_name)
    print(f"✅ Loaded config: {config.name}")
    print(f"   Label column: {config.label_column}")
    print(f"   Feature columns: {'Auto-detect' if config.feature_columns is None else f'{len(config.feature_columns)} specified'}")
    print(f"   Binary classification: {config.binary_classification}")
    
    # Load and preprocess data
    print("\n📥 Loading data...")
    loader = UniversalDataLoader(config)
    
    # Load full dataset first
    X_full, y_full = loader.load_csv(csv_path, clean_features=True)
    
    # Create holdout set (15% for validation by planning agent)
    from sklearn.model_selection import train_test_split
    X_remain, X_holdout, y_remain, y_holdout = train_test_split(
        X_full.values, y_full.values,
        test_size=0.15,
        random_state=random_state,
        stratify=y_full
    )
    
    print(f"\n📦 Holdout Set Created:")
    print(f"   Holdout samples: {X_holdout.shape[0]:,} (15%)")
    print(f"   Training pool: {X_remain.shape[0]:,} (85%)")
    print(f"   Holdout distribution: {dict(zip(*np.unique(y_holdout, return_counts=True)))}")
    
    # Save holdout set
    holdout_dir = os.path.join(artifact_dir, "holdout")
    os.makedirs(holdout_dir, exist_ok=True)
    
    holdout_X_path = os.path.join(holdout_dir, "X_holdout.npy")
    holdout_y_path = os.path.join(holdout_dir, "y_holdout.npy")
    
    np.save(holdout_X_path, X_holdout)
    np.save(holdout_y_path, y_holdout)
    print(f"💾 Holdout set saved to: {holdout_dir}/")
    
    # Split remaining data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_remain, y_remain,
        test_size=test_size / (1 - 0.15),  # Adjust test_size for remaining data
        random_state=random_state,
        stratify=y_remain
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Set MLflow tracking URI with proper headers
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    print(f"\n🔗 Connecting to MLflow at: {mlflow_uri}")
    
    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Start MLflow run
    mlflow.set_experiment("model_retraining")
    
    with mlflow.start_run(run_name=f"train_{model_name}"):
        # Log parameters
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("csv_path", csv_path)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        
        # Train model
        print(f"\n🔨 Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)
        print("✅ Model trained!")
        
        # Evaluate on test set
        print("\n📊 Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if config.binary_classification else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary' if config.binary_classification else 'weighted')
        recall = recall_score(y_test, y_pred, average='binary' if config.binary_classification else 'weighted')
        f1 = f1_score(y_test, y_pred, average='binary' if config.binary_classification else 'weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        if config.binary_classification and y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
            mlflow.log_metric("roc_auc", auc)
        else:
            auc = None
        
        # Print results
        print(f"\n{'='*80}")
        print("📈 Model Performance")
        print(f"{'='*80}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        if auc is not None:
            print(f"ROC AUC:   {auc:.4f}")
        print(f"{'='*80}\n")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        model_path = os.path.join(artifact_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\n💾 Model saved to: {model_path}")
        mlflow.log_artifact(model_path)
        
        # Log model to MLflow (try-except to handle version compatibility issues)
        try:
            mlflow.sklearn.log_model(model, model_name)
            print(f"✅ Model logged to MLflow as '{model_name}'")
        except Exception as e:
            print(f"⚠️  Warning: Could not log model to MLflow registry: {e}")
            print(f"   Model artifact still saved to: {model_path}")
        
        # Save model metadata to separate file
        model_metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "csv_path": csv_path,
            "n_features": X_train.shape[1],
            "feature_names": loader.feature_columns,
            "n_train_samples": int(X_train.shape[0]),
            "n_test_samples": int(X_test.shape[0]),
            "n_holdout_samples": int(X_holdout.shape[0]),
            "holdout_path": holdout_dir,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(auc) if auc is not None else None,
            "hyperparameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state
            },
            "dataset_config": {
                "label_column": config.label_column,
                "binary_classification": config.binary_classification,
                "psi_threshold_low": config.psi_threshold_low,
                "psi_threshold_high": config.psi_threshold_high
            }
        }
        
        metadata_path = os.path.join(artifact_dir, "model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)
        print(f"💾 Model metadata saved to: {metadata_path}")
        mlflow.log_artifact(metadata_path)
        
        # Save feature distribution for drift detection (same format as original train.py)
        # This goes to training_stats.json for DriftDetector compatibility
        feature_stats = {}
        for col in loader.feature_columns:
            col_data = X_train[:, loader.feature_columns.index(col)]
            
            # Compute quantiles for binning (10 bins)
            quantiles = np.percentile(col_data, np.linspace(0, 100, 11))
            
            # Compute expected distribution (histogram counts)
            counts, _ = np.histogram(col_data, bins=quantiles)
            expected_dist = counts / len(col_data)
            
            feature_stats[col] = {
                "mean": float(np.mean(col_data)),
                "std": float(np.std(col_data)),
                "min": float(np.min(col_data)),
                "max": float(np.max(col_data)),
                "median": float(np.median(col_data)),
                "quantiles": quantiles.tolist(),
                "expected_dist": expected_dist.tolist()
            }
        
        # Save to training_stats.json (for DriftDetector - same as original train.py)
        training_stats_path = os.path.join(artifact_dir, "training_stats.json")
        with open(training_stats_path, "w") as f:
            json.dump(feature_stats, f, indent=2)
        print(f"💾 Training stats saved to: {training_stats_path}")
        mlflow.log_artifact(training_stats_path)
        
        print(f"\n✅ Training complete! Model registered in MLflow as '{model_name}'")
        print(f"🔗 View results in MLflow UI: http://localhost:5001\n")
        
        return model, model_metadata


def main():
    parser = argparse.ArgumentParser(description="Universal model training for DriftCatcher")
    parser.add_argument("csv_path", nargs="?", help="Path to CSV file (or use --base-data)")
    parser.add_argument("--base-data", help="Path to base/baseline CSV file")
    parser.add_argument("--new-data", nargs="+", help="Path(s) to new CSV file(s) to combine with base data")
    parser.add_argument(
        "--dataset",
        default="CICIDS2017",
        choices=list(PREDEFINED_CONFIGS.keys()) + ["Generic"],
        help="Dataset configuration name"
    )
    parser.add_argument("--model-name", help="Name for the model (default: {dataset}_model)")
    parser.add_argument("--artifact-dir", default="artifacts", help="Directory to save artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=10, help="Max tree depth")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Determine CSV path - either single file or combined base + new data
    if args.base_data:
        import pandas as pd
        import tempfile
        
        # Load base data
        base_df = pd.read_csv(args.base_data)
        print(f"Loaded base data: {args.base_data} with {len(base_df)} rows")
        
        # Combine with new data if provided
        if args.new_data:
            new_dfs = [pd.read_csv(nd) for nd in args.new_data]
            print(f"Loaded {len(args.new_data)} new dataset(s) with {sum(len(df) for df in new_dfs)} total rows")
            combined_df = pd.concat([base_df] + new_dfs, ignore_index=True)
        else:
            combined_df = base_df
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        combined_df.to_csv(temp_file.name, index=False)
        csv_path = temp_file.name
        print(f"Combined dataset: {len(combined_df)} rows saved to {csv_path}")
    elif args.csv_path:
        csv_path = args.csv_path
    else:
        raise ValueError("Must provide either csv_path or --base-data argument")
    
    train_model(
        csv_path=csv_path,
        dataset_name=args.dataset,
        model_name=args.model_name,
        artifact_dir=args.artifact_dir,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
