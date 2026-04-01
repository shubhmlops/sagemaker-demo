# data_processing.py
import pandas as pd


TARGET_COL = "units_sold"


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw demand data
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])   # You can use S3 Location.
    df = df.sort_values(["drug_name", "date"]).reset_index(drop=True)
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features
    """
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def create_lag_features(
    df: pd.DataFrame,
    lags=(1, 3, 6, 12),
    rolling_windows=(3, 6),
) -> pd.DataFrame:
    """
    Create lag and rolling features per drug
    """
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby("drug_name")[TARGET_COL]
            .shift(lag)
        )

    for window in rolling_windows:
        df[f"rolling_{window}"] = (
            df.groupby("drug_name")[TARGET_COL]
            .shift(1)
            .rolling(window)
            .mean()
        )

    return df


def prepare_training_data(df: pd.DataFrame):
    """
    Full feature engineering pipeline for training
    """
    df = create_time_features(df)
    df = create_lag_features(df)

    # Drop rows with NaNs created by lags
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "price",
        "promo_flag",
        "month",
        "year",
        "lag_1",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_3",
        "rolling_6",
    ]

    X = df[feature_cols]
    y = df[TARGET_COL]

    return X, y, feature_cols, df


def time_based_split(X, y, split_ratio=0.8):
    """
    Time-based train/validation split (no shuffling)
    """
    split_idx = int(len(X) * split_ratio)

    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]

    return X_train, X_val, y_train, y_val


def prepare_inference_history(df: pd.DataFrame, min_history=12) -> pd.DataFrame:
    """
    Ensure we have enough history for forecasting
    """
    if len(df) < min_history:
        raise ValueError(
            f"Need at least {min_history} months of data for inference"
        )

    df = create_time_features(df)
    df = create_lag_features(df)

    df = df.dropna().reset_index(drop=True)
    return df
