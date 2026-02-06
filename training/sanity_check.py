import json

with open("artifacts/training_stats.json") as f:
    stats = json.load(f)

print(list(stats.keys())[:5])
print(stats[list(stats.keys())[0]])