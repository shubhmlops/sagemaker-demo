"""
1_preprocessing.py
------------------
SageMaker Processing Step Script

What it does:
- Reads raw CSV data from S3 (mounted at /opt/ml/processing/input)
- Cleans and prepares features
- Splits into train/validation/test sets
- Saves outputs to /opt/ml/processing/output (SageMaker uploads these back to S3)

Run locally:  python 1_preprocessing.py  (needs a sample data.csv in the same folder)
Run on SM:    Used automatically by the pipeline (see 4_pipeline.py)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths SageMaker injects automatically ──────────────────────────────────────
INPUT_DIR  = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

os.makedirs(f"{OUTPUT_DIR}/train",      exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/validation", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/test",       exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(f"{INPUT_DIR}/data.csv")
print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

# ── 2. Basic cleaning ──────────────────────────────────────────────────────────
print("Cleaning data...")
df = df.dropna()                          # drop rows with missing values
df = df.drop_duplicates()                 # remove exact duplicates

# ── 3. Feature engineering ─────────────────────────────────────────────────────
# Assumes the last column is the target label named "target"
feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols]
y = df["target"]

# Normalise numeric features so they're on the same scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# ── 4. Split: 70 % train | 15 % validation | 15 % test ────────────────────────
print("Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp,   y_temp, test_size=0.50, random_state=42)

# ── 5. Save outputs ────────────────────────────────────────────────────────────
def save(X, y, folder, name):
    df_out = X.copy()
    df_out["target"] = y.values
    path = f"{OUTPUT_DIR}/{folder}/{name}.csv"
    df_out.to_csv(path, index=False)
    print(f"  Saved {len(df_out)} rows → {path}")

save(X_train, y_train, "train",      "train")
save(X_val,   y_val,   "validation", "validation")
save(X_test,  y_test,  "test",       "test")

print("Preprocessing complete ✓")
