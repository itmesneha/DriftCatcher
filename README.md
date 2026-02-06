New attack appears →
Feature distributions shift:
  - Flow Duration mean changes
  - SYN flag ratios change
  - Packet rate distribution changes

Agent observes:
  - KL divergence ↑
  - PSI ↑
  - Confidence entropy ↑

Agent decides:
  - This is not DDOS-like drift
  - Trigger retraining
  - Pull new CSVs
  - Train new model version

uv run dvc add data/raw/
git add data/raw.dvc
git commit -m "Update raw data"
