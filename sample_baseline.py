"""
Sample baseline.csv to 50k rows without loading entire file into memory
"""
import pandas as pd
import os

print("Sampling baseline.csv...")

# Read in chunks and sample from each chunk proportionally
chunk_size = 10000
sample_size = 50000
total_rows = 755664  # Known from previous check

# Calculate how many rows to sample from each chunk
chunks_needed = total_rows // chunk_size
sample_per_chunk = sample_size // chunks_needed

print(f"Total rows: {total_rows}")
print(f"Target sample: {sample_size}")
print(f"Sampling ~{sample_per_chunk} rows per {chunk_size}-row chunk")

sampled_chunks = []
for i, chunk in enumerate(pd.read_csv('data/baseline.csv', chunksize=chunk_size)):
    if i == 0:
        print(f"First chunk shape: {chunk.shape}")
    
    # Sample from this chunk
    n_sample = min(sample_per_chunk, len(chunk))
    if len(sampled_chunks) * sample_per_chunk < sample_size:
        sampled = chunk.sample(n=n_sample, random_state=42)
        sampled_chunks.append(sampled)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size} rows, collected {len(sampled_chunks) * sample_per_chunk} samples")

# Combine all sampled chunks
print("Combining sampled chunks...")
combined = pd.concat(sampled_chunks, ignore_index=True)
print(f"Combined shape: {combined.shape}")

# If we got more than needed, sample down
if len(combined) > sample_size:
    print(f"Sampling down from {len(combined)} to {sample_size}")
    combined = combined.sample(n=sample_size, random_state=42)

# Backup original
print("Creating backup...")
os.rename('data/baseline.csv', 'data/baseline.csv.backup')

# Save sampled version
print(f"Saving new baseline with {len(combined)} rows...")
combined.to_csv('data/baseline.csv', index=False)

print(f"✅ Done! Baseline reduced from {total_rows} to {len(combined)} rows")
print(f"   Backup saved as: data/baseline.csv.backup")
