import json
import numpy as np
import pandas as pd

# Load the data that was used for training
df = pd.read_csv('data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df.columns = df.columns.str.strip()

# Load model metadata to get feature names
metadata = json.load(open('artifacts/training_stats.json'))
feature_cols = metadata['feature_names']

# Compute feature stats with quantiles (same as original train.py)
stats = {}
for col in feature_cols:
    if col not in df.columns:
        continue
    values = pd.to_numeric(df[col], errors='coerce').dropna().values
    if len(values) == 0:
        continue
    
    quantiles = np.percentile(values, np.linspace(0, 100, 11))
    counts, _ = np.histogram(values, bins=quantiles)
    expected_dist = (counts / len(values)).tolist()
    
    stats[col] = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'quantiles': quantiles.tolist(),
        'expected_dist': expected_dist
    }

# Save model metadata separately
with open('artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Overwrite training_stats.json with feature stats
with open('artifacts/training_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f'✅ Regenerated training_stats.json with {len(stats)} features')
print(f'✅ Model metadata saved to model_metadata.json')
