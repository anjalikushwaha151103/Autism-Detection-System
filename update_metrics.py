import json
import csv
import os
import sys

# Results from terminal evaluation
TUNED_METRICS = {
    "accuracy": 0.9429,
    "precision": 0.9461,
    "recall": 0.9429,
    "f1": 0.9428,
    "sensitivity": 0.9000,
    "specificity": 0.9857
}

results_dir = r"d:\Autism Detection System\Autism-Detection-System\results"
json_path = os.path.join(results_dir, "model_comparison.json")
csv_path = os.path.join(results_dir, "model_comparison.csv")

# 1. Update JSON
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data["resnet50"] = TUNED_METRICS
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print("Updated model_comparison.json")

# 2. Update CSV
if os.path.exists(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Model"] == "resnet50":
                row["Accuracy"] = str(TUNED_METRICS["accuracy"])
                row["Precision"] = str(TUNED_METRICS["precision"])
                row["Recall"] = str(TUNED_METRICS["recall"])
                row["F1 Score"] = str(TUNED_METRICS["f1"])
                row["Sensitivity"] = str(TUNED_METRICS["sensitivity"])
                row["Specificity"] = str(TUNED_METRICS["specificity"])
            rows.append(row)
            
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Sensitivity", "Specificity"])
        writer.writeheader()
        writer.writerows(rows)
    print("Updated model_comparison.csv")
