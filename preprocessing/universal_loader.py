"""
Universal Data Loader for DriftCatcher
Auto-detects features and labels from any tabular dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
from config.dataset_config import DatasetConfig


class UniversalDataLoader:
    """
    Universal data loader that works with any tabular dataset.
    
    Auto-detection logic:
    1. Label column: Uses config.label_column or looks for common patterns:
       - Column named: 'label', 'class', 'target', 'y', 'outcome'
       - Last column as fallback
    
    2. Feature columns: Uses config.feature_columns or:
       - All numeric columns except the label column
       - Automatically excludes: ID columns, timestamp columns, non-numeric columns
    
    3. Label processing:
       - Binary classification: Converts to 0/1 (normal=0, anomaly/fraud=1)
       - Multi-class: Encodes string labels to integers
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize loader with dataset configuration.
        
        Args:
            config: DatasetConfig with label_column, feature_columns, etc.
        """
        self.config = config
        self.feature_columns = None
        self.label_mapping = None
    
    def detect_label_column(self, df: pd.DataFrame) -> str:
        """
        Auto-detect the label column if not specified in config.
        
        Logic:
        1. Check for common label column names (case-insensitive)
        2. Use the last column as fallback
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the label column
        """
        if self.config.label_column:
            return self.config.label_column
        
        # Common label column patterns (case-insensitive)
        common_patterns = ['label', 'class', 'target', 'y', 'outcome', 
                          'attack', 'fraud', 'anomaly', 'intrusion']
        
        for col in df.columns:
            if col.lower().strip() in common_patterns:
                print(f"🔍 Auto-detected label column: '{col}'")
                return col
        
        # Fallback: use last column
        label_col = df.columns[-1]
        print(f"⚠️  No standard label column found, using last column: '{label_col}'")
        return label_col
    
    def detect_feature_columns(self, df: pd.DataFrame, label_col: str) -> List[str]:
        """
        Auto-detect feature columns if not specified in config.
        
        Logic:
        1. Exclude the label column
        2. Exclude non-numeric columns (or convert if possible)
        3. Exclude ID-like columns (all unique values)
        4. Exclude timestamp columns
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            List of feature column names
        """
        if self.config.feature_columns:
            return self.config.feature_columns
        
        # Start with all columns except label
        feature_cols = [col for col in df.columns if col != label_col]
        
        # Filter numeric columns only
        numeric_cols = []
        for col in feature_cols:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's an ID column (all unique or mostly unique)
                if len(df) > 0 and df[col].nunique() / len(df) > 0.95:
                    print(f"⏭️  Skipping ID-like column: '{col}' ({df[col].nunique()} unique values)")
                    continue
                numeric_cols.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col])
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    print(f"⏭️  Skipping non-numeric column: '{col}'")
        
        if not numeric_cols:
            raise ValueError("No numeric feature columns found! Check your dataset.")
        
        print(f"✅ Auto-detected {len(numeric_cols)} feature columns")
        return numeric_cols
    
    def process_labels(self, y: pd.Series) -> np.ndarray:
        """
        Process labels based on classification type.
        
        For binary classification:
        - Converts to 0/1 (normal=0, anomaly/attack/fraud=1)
        - Detects normal labels: 'benign', 'normal', 'legitimate', 'no', '0'
        
        For multi-class:
        - Encodes string labels to integers
        
        Args:
            y: Label Series
            
        Returns:
            Processed labels as numpy array
        """
        if self.config.binary_classification:
            # Define normal/benign patterns
            normal_patterns = ['benign', 'normal', 'legitimate', 'no', 'negative', '0']
            
            if self.config.normal_label:
                # Use specified normal label
                y_binary = (y != self.config.normal_label).astype(int)
                print(f"📊 Binary classification: '{self.config.normal_label}' → 0, others → 1")
            else:
                # Auto-detect normal label
                unique_labels = y.unique()
                
                # Check if already binary (0/1)
                if set(unique_labels).issubset({0, 1, '0', '1'}):
                    y_binary = pd.to_numeric(y).astype(int)
                    print(f"📊 Binary classification: Already 0/1 encoded")
                else:
                    # Find normal label from unique values
                    normal_label = None
                    for label in unique_labels:
                        if str(label).lower().strip() in normal_patterns:
                            normal_label = label
                            break
                    
                    if normal_label is not None:
                        y_binary = (y != normal_label).astype(int)
                        print(f"📊 Binary classification: '{normal_label}' → 0, others → 1")
                    else:
                        # Fallback: first unique value is normal
                        normal_label = unique_labels[0]
                        y_binary = (y != normal_label).astype(int)
                        print(f"⚠️  Assuming '{normal_label}' is normal class → 0, others → 1")
            
            return y_binary.values
        else:
            # Multi-class classification
            if pd.api.types.is_numeric_dtype(y):
                print(f"📊 Multi-class classification: {y.nunique()} classes (already numeric)")
                return y.values
            else:
                # Encode string labels
                self.label_mapping = {label: idx for idx, label in enumerate(y.unique())}
                y_encoded = y.map(self.label_mapping)
                print(f"📊 Multi-class classification: {len(self.label_mapping)} classes")
                print(f"   Label mapping: {self.label_mapping}")
                return y_encoded.values
    
    def clean_numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean numeric features: handle inf, NaN, and convert to float.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Replace inf with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Count NaNs
        nan_counts = X.isna().sum()
        if nan_counts.sum() > 0:
            print(f"⚠️  Found {nan_counts.sum()} NaN values:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"     {col}: {count}")
            
            # Fill NaNs with column mean
            X = X.fillna(X.mean())
            print(f"✅ Filled NaNs with column means")
        
        # Ensure all numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        return X
    
    def load_csv(
        self, 
        csv_path: str,
        clean_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load CSV and separate features and labels.
        
        Args:
            csv_path: Path to CSV file
            clean_features: Whether to clean numeric features
            
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        print(f"\n📥 Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        
        # Detect label column
        label_col = self.detect_label_column(df)
        print(f"   Label column detected: '{label_col}'")
        print(f"   Available columns: {list(df.columns[:5])}... ({len(df.columns)} total)")
        print(f"   Label column in dataframe: {label_col in df.columns}")
        
        if label_col not in df.columns:
            raise ValueError(f"Detected label column '{label_col}' not found in dataframe! Available columns: {list(df.columns)}")
        
        try:
            print(f"   Label column type: {df[label_col].dtype}")
            print(f"   Sample labels: {list(df[label_col].head())}")
            print(f"   Null labels: {df[label_col].isna().sum()}")
        except Exception as e:
            print(f"   ⚠️  Error accessing label column '{label_col}': {type(e).__name__}: {e}")
            print(f"   Column names in df: {list(df.columns)}")
            print(f"   DataFrame dtypes: {df.dtypes.to_dict()}")
            raise
        
        # Drop rows with missing labels
        initial_rows = len(df)
        df = df.dropna(subset=[label_col])
        if len(df) < initial_rows:
            print(f"⚠️  Dropped {initial_rows - len(df)} rows with missing labels")
        
        print(f"   Rows after dropping nulls: {len(df)}")
        
        # Detect feature columns
        self.feature_columns = self.detect_feature_columns(df, label_col)
        
        print(f"   Detected {len(self.feature_columns)} feature columns")
        if len(self.feature_columns) == 0:
            raise ValueError(f"No feature columns detected! All columns were filtered out. Available columns: {list(df.columns)}")
        
        # Extract features and labels
        X = df[self.feature_columns].copy()
        y = df[label_col].copy()
        
        print(f"   Extracted X shape: {X.shape}, y shape: {y.shape}")
        
        # Clean features
        if clean_features:
            X = self.clean_numeric_features(X)
        
        # Validate we still have data after cleaning
        if len(X) == 0:
            raise ValueError(f"No data remaining after cleaning! Original CSV had {initial_rows} rows, all were removed during preprocessing.")
        
        # Process labels
        y_processed = self.process_labels(y)
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Features: {X.shape[1]} columns")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Label distribution: {dict(zip(*np.unique(y_processed, return_counts=True)))}")
        
        return X, pd.Series(y_processed)
    
    def load_and_preprocess(
        self,
        csv_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        clean_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CSV, preprocess, and split into train/test sets.
        
        Args:
            csv_path: Path to CSV file
            test_size: Fraction of data for test set
            random_state: Random seed
            clean_features: Whether to clean numeric features
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X, y = self.load_csv(csv_path, clean_features=clean_features)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"\n📊 Train/Test Split:")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Test:  {X_test.shape[0]} samples")
        print(f"   Split: {100*(1-test_size):.0f}/{100*test_size:.0f}")
        
        return X_train, X_test, y_train, y_test
