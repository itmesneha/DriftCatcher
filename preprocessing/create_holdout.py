"""
Create comprehensive holdout dataset from all CICIDS2017 CSVs

Best practices:
- Sample from ALL datasets for diverse attack representation
- Stratified sampling by label to maintain distribution
- 15% from each file for representative coverage
- Shuffle to mix different days/scenarios
"""
import pandas as pd
import os
from pathlib import Path
import argparse


def create_holdout(
    data_dir: str = "data/raw",
    output_path: str = "data/processed/holdout.csv",
    sample_fraction: float = 0.15,
    random_state: int = 42
):
    """
    Create stratified holdout set from all CSVs
    
    Args:
        data_dir: Directory containing CICIDS2017 CSVs
        output_path: Path to save holdout dataset
        sample_fraction: Fraction to sample from each file (0.15 = 15%)
        random_state: Random seed for reproducibility
    """
    
    print("Creating comprehensive holdout set from all CICIDS2017 CSVs...")
    print("="*60)
    
    # Find all CSV files
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️  No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Sample from each CSV to create diverse holdout
    holdout_samples = []
    
    for csv_file in csv_files:
        print(f"\n📁 Processing: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            
            if 'Label' not in df.columns:
                print(f"  ⚠️  No 'Label' column, skipping")
                continue
            
            print(f"  Total samples: {len(df):,}")
            print(f"  Label distribution:")
            for label, count in df['Label'].value_counts().items():
                pct = 100 * count / len(df)
                print(f"    {label}: {count:,} ({pct:.1f}%)")
            
            # Stratified sampling by label to maintain distribution
            sampled = df.groupby('Label', group_keys=False).apply(
                lambda x: x.sample(
                    frac=min(sample_fraction, len(x) / max(1, len(x))),
                    random_state=random_state
                )
            )
            
            print(f"  → Sampled {len(sampled):,} for holdout")
            holdout_samples.append(sampled)
            
        except Exception as e:
            print(f"  ❌ Error processing {csv_file.name}: {e}")
            continue
    
    if not holdout_samples:
        print("\n❌ No samples collected. Exiting.")
        return
    
    # Combine all samples
    print(f"\n🔗 Combining samples from {len(holdout_samples)} files...")
    holdout = pd.concat(holdout_samples, ignore_index=True)
    
    # Clean data: Remove inf and NaN values
    print("🧹 Cleaning data...")
    import numpy as np
    
    # Get numeric columns (exclude Label and any object columns)
    numeric_cols = holdout.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in numeric_cols:
        numeric_cols.remove('Label')
    
    initial_size = len(holdout)
    
    # Replace inf with NaN
    holdout[numeric_cols] = holdout[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with any NaN values
    holdout = holdout.dropna(subset=numeric_cols)
    
    cleaned_count = initial_size - len(holdout)
    print(f"  Removed {cleaned_count:,} rows with inf/NaN values ({100*cleaned_count/initial_size:.2f}%)")
    print(f"  Final clean dataset: {len(holdout):,} rows")
    
    # Shuffle to mix different days/scenarios
    print("🔀 Shuffling to mix attack types and days...")
    holdout = holdout.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    holdout.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "="*60)
    print(f"✅ Created comprehensive holdout: {output_path}")
    print(f"   Total samples: {len(holdout):,}")
    print(f"\n   Label distribution:")
    
    label_counts = holdout['Label'].value_counts()
    for label, count in label_counts.items():
        pct = 100 * count / len(holdout)
        print(f"     {label}: {count:,} ({pct:.1f}%)")
    
    print(f"\n   Attack type diversity: {len(label_counts)} unique labels")
    print(f"   CSV sources: {len(holdout_samples)} files")
    print("="*60)
    
    return holdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create stratified holdout dataset from CICIDS2017 CSVs"
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Directory containing source CSV files'
    )
    parser.add_argument(
        '--output',
        default='data/processed/holdout.csv',
        help='Output path for holdout dataset'
    )
    parser.add_argument(
        '--sample-fraction',
        type=float,
        default=0.15,
        help='Fraction to sample from each file (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    create_holdout(
        data_dir=args.data_dir,
        output_path=args.output,
        sample_fraction=args.sample_fraction,
        random_state=args.random_state
    )
