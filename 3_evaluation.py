"""
3_evaluation.py
---------------
SageMaker Processing Step Script (run after training)

What it does:
- Loads the trained model artifact from /opt/ml/processing/model/
- Runs it against the held-out test set
- Writes a JSON evaluation report to /opt/ml/processing/evaluation/
  → The pipeline reads this report to decide whether to register the model
"""

import os
import json
import tarfile
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_TAR  = "/opt/ml/processing/model/model.tar.gz"   # SM wraps model in a tar
TEST_DIR   = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Extract & load model ────────────────────────────────────────────────────
print("Extracting model archive...")
with tarfile.open(MODEL_TAR) as tar:
    tar.extractall("/opt/ml/processing/model/")

model = joblib.load("/opt/ml/processing/model/model.joblib")
print("Model loaded ✓")

# ── 2. Load test data ──────────────────────────────────────────────────────────
test_df = pd.read_csv(f"{TEST_DIR}/test.csv")
X_test  = test_df.drop("target", axis=1)
y_test  = test_df["target"]
print(f"Test rows: {len(test_df)}")

# ── 3. Predict & compute metrics ───────────────────────────────────────────────
preds = model.predict(X_test)

metrics = {
    "accuracy":  round(accuracy_score(y_test, preds), 4),
    "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
    "recall":    round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
    "f1":        round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
}

print("Evaluation results:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# ── 4. Save report ─────────────────────────────────────────────────────────────
# SageMaker's ConditionStep can read this file to gate model registration
report = {"metrics": metrics}
report_path = os.path.join(OUTPUT_DIR, "evaluation.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"Report saved → {report_path}  ✓")
