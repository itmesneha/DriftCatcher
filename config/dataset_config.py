"""
Dataset Configuration for Universal Data Loading
Defines standard configurations for common datasets
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Configuration for a dataset"""
    name: str = Field(description="Dataset name")
    label_column: str = Field(description="Name of the label column")
    feature_columns: Optional[List[str]] = Field(
        default=None, 
        description="List of feature columns. If None, auto-detect."
    )
    normal_label: Optional[str] = Field(
        default=None,
        description="Label value representing normal/benign class. If None, auto-detect."
    )
    binary_classification: bool = Field(
        default=True,
        description="Whether this is binary or multi-class classification"
    )
    psi_threshold_low: float = Field(
        default=0.1,
        description="PSI threshold for low drift"
    )
    psi_threshold_high: float = Field(
        default=0.2,
        description="PSI threshold for high drift"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the dataset"
    )


# Predefined dataset configurations
PREDEFINED_CONFIGS = {
    "CICIDS2017": DatasetConfig(
        name="CICIDS2017",
        label_column="Label",
        feature_columns=None,  # Auto-detect (too many to list)
        normal_label="BENIGN",
        binary_classification=True,
        psi_threshold_low=0.1,
        psi_threshold_high=0.2,
        description="CICIDS2017 network intrusion detection dataset"
    ),
    
    "KDD99": DatasetConfig(
        name="KDD99",
        label_column="label",
        feature_columns=None,  # Auto-detect
        normal_label="normal.",
        binary_classification=True,
        psi_threshold_low=0.1,
        psi_threshold_high=0.2,
        description="KDD Cup 1999 network intrusion detection dataset"
    ),
    
    "NSL-KDD": DatasetConfig(
        name="NSL-KDD",
        label_column="label",
        feature_columns=None,  # Auto-detect
        normal_label="normal",
        binary_classification=True,
        psi_threshold_low=0.1,
        psi_threshold_high=0.2,
        description="NSL-KDD network intrusion detection dataset (improved KDD99)"
    ),
    
    "CreditCard": DatasetConfig(
        name="CreditCard",
        label_column="Class",
        feature_columns=None,  # Auto-detect (V1-V28 + Amount)
        normal_label="0",
        binary_classification=True,
        psi_threshold_low=0.1,
        psi_threshold_high=0.2,
        description="Kaggle Credit Card Fraud Detection dataset"
    ),
    
    "Generic": DatasetConfig(
        name="Generic",
        label_column="",  # Will be auto-detected
        feature_columns=None,  # Auto-detect
        normal_label=None,  # Auto-detect
        binary_classification=True,
        psi_threshold_low=0.1,
        psi_threshold_high=0.2,
        description="Generic dataset with auto-detection"
    )
}

# Export for easy access
DATASET_CONFIGS = PREDEFINED_CONFIGS


def load_dataset_config(name: str) -> DatasetConfig:
    """
    Load a predefined dataset configuration by name.
    
    Args:
        name: Dataset name (e.g., 'CICIDS2017', 'KDD99', 'Generic')
        
    Returns:
        DatasetConfig object
        
    Raises:
        ValueError: If dataset name not found
    """
    if name not in PREDEFINED_CONFIGS:
        available = ", ".join(PREDEFINED_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}"
        )
    
    return PREDEFINED_CONFIGS[name]


def save_dataset_config(config: DatasetConfig, filepath: str):
    """
    Save a dataset configuration to JSON file.
    
    Args:
        config: DatasetConfig object
        filepath: Path to save JSON file
    """
    import json
    
    with open(filepath, "w") as f:
        json.dump(config.model_dump(), f, indent=2)
    
    print(f"✅ Dataset config saved to: {filepath}")


def load_custom_config(filepath: str) -> DatasetConfig:
    """
    Load a custom dataset configuration from JSON file.
    
    Args:
        filepath: Path to JSON config file
        
    Returns:
        DatasetConfig object
    """
    import json
    
    with open(filepath, "r") as f:
        config_dict = json.load(f)
    
    return DatasetConfig(**config_dict)
