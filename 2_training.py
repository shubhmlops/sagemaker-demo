"""
2_training.py
-------------
SageMaker Training Step Script

What it does:
- Reads train/validation CSVs from /opt/ml/input/data/
- Trains a simple Random Forest classifier
- Saves the trained model to /opt/ml/model/  (SageMaker tars & uploads this to S3)

Hyperparameters come from SM environment variables (set in the pipeline definition).
"""

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── SM standard directory layout ───────────────────────────────────────────────
TRAIN_DIR  = "/opt/ml/input/data/train"
VAL_DIR    = "/opt/ml/input/data/validation"
MODEL_DIR  = "/opt/ml/model"
PARAM_FILE = "/opt/ml/input/config/hyperparameters.json"   # set by SageMaker

os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load hyperparameters ────────────────────────────────────────────────────
# SageMaker writes these from the Estimator's hyperparameters= dict
if os.path.exists(PARAM_FILE):
    with open(PARAM_FILE) as f:
        hp = json.load(f)
else:
    hp = {}

n_estimators = int(hp.get("n_estimators", 100))
max_depth    = int(hp.get("max_depth", 5))
print(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}")

# ── 2. Load data ───────────────────────────────────────────────────────────────
print("Loading training data...")
train_df = pd.read_csv(f"{TRAIN_DIR}/train.csv")
val_df   = pd.read_csv(f"{VAL_DIR}/validation.csv")

X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
X_val,   y_val   = val_df.drop("target", axis=1),   val_df["target"]

print(f"  Train: {len(train_df)} rows  |  Val: {len(val_df)} rows")

# ── 3. Train model ─────────────────────────────────────────────────────────────
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42,
    n_jobs=-1,          # use all CPU cores
)
model.fit(X_train, y_train)

# ── 4. Quick validation check ──────────────────────────────────────────────────
val_preds    = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {val_accuracy:.4f}")

# ── 5. Save model ──────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "model.joblib")
joblib.dump(model, model_path)
print(f"Model saved → {model_path}  ✓")
