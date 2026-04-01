import os
import json
import pickle
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_lag_variables(df, lag_terms, lag_feature):
    for lag_val in lag_terms:
        df[f"{lag_feature}_LAG{lag_val}"] = df[lag_feature].shift(lag_val)
    return df


# ============================================================
# MODEL LOADING
# ============================================================

def model_fn(model_dir):
    """
    Load Prophet model from SageMaker model directory
    """
    logger.info("Listing model directory contents:")
    for root, _, files in os.walk(model_dir):
        logger.info(f"{root}: {files}")

    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info("Model loaded successfully")
    return model


# ============================================================
# INPUT PARSING
# ============================================================

def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)

    if "data" not in payload:
        raise ValueError("Request JSON must contain 'data'")

    return {
        "data": pd.DataFrame(payload["data"]),
        "regressors": payload.get("regressors"),
        "regressors_used": payload.get("regressors_used"),
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_fn(inputs, model):
    data_df = inputs["data"]
    regressors = inputs["regressors"]
    regressors_used = inputs["regressors_used"]

    logger.info("Raw input:")
    logger.info(data_df.head())

    # Validate columns
    missing = set(regressors) - set(data_df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    data_df = data_df[regressors]

    # Lag features (only for numeric trx)
    data_df = create_lag_variables(data_df, [1], "MARKET_TRX")
    data_df = create_lag_variables(data_df, [1, 2], "CGRP_TRX")

    data_df.drop(columns=["MARKET_TRX", "CGRP_TRX"], inplace=True)
    data_df["ds"] = pd.to_datetime(data_df["ds"])

    data_df = data_df.iloc[2:]
    data_df.fillna(0, inplace=True)

    # Validate regressors_used
    missing_used = set(regressors_used) - set(data_df.columns)
    if missing_used:
        raise ValueError(f"Missing regressors_used columns: {missing_used}")

    forecast = model.predict(data_df[["ds"] + regressors_used])

    print(forecast)

    return forecast[["ds", "yhat"]].rename(
        columns={"yhat": "prediction"}
    )


# ============================================================
# OUTPUT
# ============================================================

def output_fn(prediction, response_content_type):
    if response_content_type != "application/json":
        raise ValueError("Unsupported response content type")

    result = prediction.to_json(
        orient="records",
        date_format="iso",
        indent=4
    )

    print(result)

    return result

# # ============================================================
# # HANDLER (FOR LOCAL / CUSTOM SERVER)
# # ============================================================

def handler(data):
    global _MODEL_CACHE

    if "_MODEL_CACHE" not in globals():
        _MODEL_CACHE = model_fn("/opt/ml/model")

    request_body = data.body.decode("utf-8")
    input_df = input_fn(request_body, "application/json")

    prediction = predict_fn(input_df, _MODEL_CACHE)
    return output_fn(prediction, "application/json")