# train.py
import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_processing import (
    load_data,
    prepare_training_data,
    time_based_split
)
from utils import *

df = load_data(data_path)
X, y, feature_cols, df_fe = prepare_training_data(df)
X_train, X_val, y_train, y_val = time_based_split(X, y)

## SageMaker location - /opt/ml/model/model.pkl  and model.tar.gz in S3 Location

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def create_features(df):
    df = df.sort_values("date")

    # Time features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Lag features
    for lag in [1, 3, 6, 12]:
        df[f"lag_{lag}"] = df["units_sold"].shift(lag)

    # Rolling mean
    df["rolling_3"] = df["units_sold"].shift(1).rolling(3).mean()
    df["rolling_6"] = df["units_sold"].shift(1).rolling(6).mean()

    return df.dropna()


def main():
    args = parse_args()

    # Load data
    path = os.path.join(args.train_dir, "drug_demand.csv")
    df = pd.read_csv(path, parse_dates=["date"])

    # Feature engineering
    df = create_features(df)

    features = [
        "price", "promo_flag", "month", "year",
        "lag_1", "lag_3", "lag_6", "lag_12",
        "rolling_3", "rolling_6"
    ]

    X = df[features]
    y = df["units_sold"]

    # Time-based split (NOT random)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {mae:.2f}")

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # you can use save_model function from utils to save the model in S3 location with versioning enabled.


if __name__ == "__main__":
    main()
