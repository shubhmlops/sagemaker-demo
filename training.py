"""
2_training.py
-------------
SageMaker Training Step Script

What it does:
- Reads train/validation CSVs from /opt/ml/input/data/
- Trains a simple Random Forest classifier
- Saves the trained model to /opt/ml/model/  (SageMaker tars & uploads this to S3)
- Copies itself into model/code/ so the serving container can find model_fn / predict_fn

Hyperparameters come from SM environment variables (set in the pipeline definition).
"""

import os
import io
import json
import shutil
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# Inference handlers — loaded by the serving container on startup.
# These must be at module level so they are importable without executing
# any training code.
# ─────────────────────────────────────────────────────────────────────────────

def model_fn(model_dir):
    """Load the model from disk — called once when the container starts."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type):
    """Deserialise the incoming request into a DataFrame."""
    if content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body), header=None)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run inference and return predictions."""
    return model.predict(input_data)


# ─────────────────────────────────────────────────────────────────────────────
# Training — only runs when SageMaker executes this script directly.
# Guarded by __main__ so importing this file for inference never triggers it.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TRAIN_DIR  = "/opt/ml/input/data/train"
    VAL_DIR    = "/opt/ml/input/data/validation"
    MODEL_DIR  = "/opt/ml/model"
    PARAM_FILE = "/opt/ml/input/config/hyperparameters.json"

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load hyperparameters ────────────────────────────────────────────────
    if os.path.exists(PARAM_FILE):
        with open(PARAM_FILE) as f:
            hp = json.load(f)
    else:
        hp = {}

    n_estimators = int(hp.get("n_estimators", 100))
    max_depth    = int(hp.get("max_depth", 5))
    print(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}")

    # ── 2. Load data ───────────────────────────────────────────────────────────
    print("Loading training data...")
    train_df = pd.read_csv(f"{TRAIN_DIR}/train.csv")
    val_df   = pd.read_csv(f"{VAL_DIR}/validation.csv")

    X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
    X_val,   y_val   = val_df.drop("target", axis=1),   val_df["target"]

    print(f"  Train: {len(train_df)} rows  |  Val: {len(val_df)} rows")

    # ── 3. Train model ─────────────────────────────────────────────────────────
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── 4. Quick validation check ──────────────────────────────────────────────
    val_preds    = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # ── 5. Save model ──────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}  ✓")

    # ── 6. Copy this script into model/code/ ───────────────────────────────────
    # SageMaker tars /opt/ml/model/ into model.tar.gz and uploads to S3.
    # The serving container extracts it and looks for SAGEMAKER_PROGRAM here.
    code_dir = os.path.join(MODEL_DIR, "code")
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(code_dir, "training.py"))
    print(f"Inference script copied → {code_dir}/training.py  ✓")
