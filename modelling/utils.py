import pandas as pd
import numpy as np
import pickle
import os
import snowflake.connector
#from config import *
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import re
import logging
import sys
from snowflake.connector.pandas_tools import write_pandas
import boto3
from io import BytesIO
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime
from functools import lru_cache

# utils.py
import boto3
import joblib
import os


def save_model(model, model_dir, bucket=None, s3_key=None):
    """
    Save model locally for SageMaker AND optionally upload to S3
    """
    os.makedirs(model_dir, exist_ok=True)

    local_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, local_path)

    print(f"Model saved locally at {local_path}")

    if bucket and s3_key:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, s3_key)
        print(f"Model uploaded to s3://{bucket}/{s3_key}")

    return local_path



@lru_cache(maxsize=1)
def get_env_from_ssm(
    param_name="/demo/env",
    region="us-east-1"
):
    ssm = boto3.client("ssm", region_name=region)

    response = ssm.get_parameter(
        Name=param_name,
        WithDecryption=False
    )
    return response["Parameter"]["Value"]


def get_ssm_parameter(name, with_decryption=False):
    ssm = boto3.client("ssm", region_name="us-east-1")
    response = ssm.get_parameter(
        Name=name,
        WithDecryption=with_decryption
    )
    return response["Parameter"]["Value"]


def get_secret(secret_name: str, region: str):
    client = boto3.client("secretsmanager", region_name=region)

    response = client.get_secret_value(SecretId=secret_name)

    if "SecretString" in response:
        return json.loads(response["SecretString"])
    else:
        raise ValueError("SecretBinary is not supported")


def load_csv_data(file_path, parse_dates=None):


def save_model(model, name, path="", timestamp=None):
    """
    Save model both locally and to S3.
    If timestamp is provided, use it so multiple files go in same folder.
    """
    # 1️⃣ Local directory
    model_dir = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    local_path = os.path.join(model_dir, name)

    ext = name.split(".")[-1].lower()

    # 2️⃣ Save locally
    if ext == "pkl":
        with open(local_path, "wb") as f:
            pickle.dump(model, f)
    elif ext == "json":
        if isinstance(model, dict):
            model = json.dumps(model, indent=2)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(model)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Resolve env strictly from SSM
    env = get_env_from_ssm()    

    # 3️⃣ Upload to S3
    s3 = boto3.client("s3")
    bucket = f"demo-{env}-sagemaker-data"
    base_prefix = "models"

    subfolder = name.split("_model")[0] if "_model" in name else name.split(".")[0]

    # 👉 Use the same timestamp if provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    s3_prefix = f"{base_prefix}/{subfolder}/{timestamp}/"
    s3_key = s3_prefix + name

    s3.upload_file(local_path, bucket, s3_key)

    return timestamp

def save_model_area(model, name, path="", timestamp=None):
    """
    Save model both locally and to S3.
    If timestamp is provided, use it so multiple files go in same folder.
    """
    # 1️⃣ Local directory
    model_dir = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    local_path = os.path.join(model_dir, name)

    ext = name.split(".")[-1].lower()

    # 2️⃣ Save locally
    if ext == "pkl":
        with open(local_path, "wb") as f:
            pickle.dump(model, f)
    elif ext == "json":
        if isinstance(model, dict):
            model = json.dumps(model, indent=2)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(model)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Resolve env strictly from SSM
    env = get_env_from_ssm()    

    # 3️⃣ Upload to S3
    s3 = boto3.client("s3")
    bucket = f"demo-{env}-sagemaker-data"
    base_prefix = "models/area_level"

    subfolder = name.split("_model")[0] if "_model" in name else name.split(".")[0]

    # 👉 Use the same timestamp if provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    s3_prefix = f"{base_prefix}/{subfolder}/{timestamp}/"
    s3_key = s3_prefix + name

    s3.upload_file(local_path, bucket, s3_key)

    return timestamp

def save_model_weekly(model, name, path="", timestamp=None):
    """
    Save model both locally and to S3.
    If timestamp is provided, use it so multiple files go in same folder.
    """
    # 1️⃣ Local directory
    model_dir = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    local_path = os.path.join(model_dir, name)

    ext = name.split(".")[-1].lower()

    # 2️⃣ Save locally
    if ext == "pkl":
        with open(local_path, "wb") as f:
            pickle.dump(model, f)
    elif ext == "json":
        if isinstance(model, dict):
            model = json.dumps(model, indent=2)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(model)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Resolve env strictly from SSM
    env = get_env_from_ssm()    

    # 3️⃣ Upload to S3
    s3 = boto3.client("s3")
    bucket = f"demo-{env}-sagemaker-data"
    base_prefix = "models/weekly"

    subfolder = name.split("_model")[0] if "_model" in name else name.split(".")[0]

    # 👉 Use the same timestamp if provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    s3_prefix = f"{base_prefix}/{subfolder}/{timestamp}/"
    s3_key = s3_prefix + name

    s3.upload_file(local_path, bucket, s3_key)

    return timestamp

def load_model_area(name, timestamp=None):
    import os, pickle, boto3

    model_dir = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    local_path = os.path.join(model_dir, name)

    if os.path.exists(local_path):
        print(f"✅ Loaded model from local path: {local_path}")
        with open(local_path, "rb") as f:
            return pickle.load(f)

    print(f"⚠️ Local model not found, searching S3 for {name}...")

    env = get_env_from_ssm()
    bucket = f"demo-{env}-sagemaker-data"
    s3 = boto3.client("s3")

    prefix = "models/area_level/"

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    found_key = None

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # exact match only on filename
            if key.split("/")[-1] == name:
                if timestamp:
                    if f"/{timestamp}/" in key:
                        found_key = key
                        break
                else:
                    found_key = key
                    break

        if found_key:
            break

    if not found_key:
        raise FileNotFoundError(f"❌ Could not find {name} under s3://{bucket}/{prefix}")

    local_tmp_path = os.path.join("/tmp", name)
    s3.download_file(bucket, found_key, local_tmp_path)

    print(f"✅ Found model: s3://{bucket}/{found_key}")

    with open(local_tmp_path, "rb") as f:
        return pickle.load(f)
    

def load_model_weekly(name, timestamp=None):
    import os, pickle, boto3

    model_dir = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    local_path = os.path.join(model_dir, name)

    if os.path.exists(local_path):
        print(f"✅ Loaded model from local path: {local_path}")
        with open(local_path, "rb") as f:
            return pickle.load(f)

    print(f"⚠️ Local model not found, searching S3 for {name}...")

    env = get_env_from_ssm()
    bucket = f"demo-{env}-sagemaker-data"
    prefix = "models/weekly/"

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    candidates = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.split("/")[-1] == name:
                if timestamp:
                    if f"/{timestamp}/" in key:
                        candidates.append(obj)
                else:
                    candidates.append(obj)

    if not candidates:
        raise FileNotFoundError(f"❌ Could not find {name} under s3://{bucket}/{prefix}")

    # choose latest by LastModified
    latest_obj = max(candidates, key=lambda x: x["LastModified"])
    found_key = latest_obj["Key"]

    print(f"✅ Found latest model: s3://{bucket}/{found_key}")

    local_download_path = os.path.join(model_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    s3.download_file(bucket, found_key, local_download_path)

    with open(local_download_path, "rb") as f:
        return pickle.load(f)

env = get_env_from_ssm()
def load_latest_json_for_model(model_name, bucket=f"demo-{env}-sagemaker-data"):
    s3 = boto3.client("s3")

    # 1️⃣ Path: models/market_trx/
    model_prefix = f"models/{model_name}/"

    # 2️⃣ List timestamp folders inside model
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=model_prefix, Delimiter="/")

    timestamp_folders = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]

    if not timestamp_folders:
        raise FileNotFoundError(f"No timestamp folders found for model '{model_name}'")

    # 3️⃣ Choose latest folder
    latest_prefix = sorted(timestamp_folders)[-1]

    # 4️⃣ Inside latest folder, find JSON file
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=latest_prefix)
    contents = resp.get("Contents", [])
    json_files = [obj["Key"] for obj in contents if obj["Key"].endswith(".json")]

    if not json_files:
        raise FileNotFoundError(f"No JSON file found inside '{latest_prefix}' for '{model_name}'")

    json_key = json_files[0]

    # 5️⃣ Read JSON
    json_obj = s3.get_object(Bucket=bucket, Key=json_key)
    raw_text = json_obj["Body"].read().decode("utf-8")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text  # fallback: return raw string
    
import json

env = get_env_from_ssm()
def load_all_latest_jsons(bucket=f"demo-{env}-sagemaker-data", base_prefix="models"):
    s3 = boto3.client("s3")

    result = {}

    # 1️⃣ Get all model folders under models/
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{base_prefix}/", Delimiter="/")
    model_folders = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]

    for folder in model_folders:
        model_name = folder.split("/")[-2]  # e.g., market_trx

        # 2️⃣ Get timestamp folders for this model
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=folder, Delimiter="/")
        timestamp_folders = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]

        if not timestamp_folders:
            continue  # skip if no folders exist

        # 3️⃣ Pick latest timestamp folder
        latest_prefix = sorted(timestamp_folders)[-1]

        # 4️⃣ Get JSON files inside the latest folder
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=latest_prefix)
        json_files = [obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith(".json")]

        if not json_files:
            continue  # skip if no json in this model folder

        json_key = json_files[0]

        # 5️⃣ Read JSON content
        obj = s3.get_object(Bucket=bucket, Key=json_key)
        raw_text = obj["Body"].read().decode("utf-8")

        try:
            json_content = json.loads(raw_text)
        except json.JSONDecodeError:
            json_content = raw_text

        result[model_name] = json_content

    return result


def load_model(name, timestamp=None):
    """
    Load model from local (/opt/ml/model) or from S3.
    Automatically detects model subfolder (based on file name prefix).
    
    Args:
        name: model filename, e.g. 'market_trx_model.pkl'
        timestamp: optional timestamp folder in S3, e.g. '2025-11-06_14-32-10'
                   If None, loads from the latest timestamp.
    """
    # 1️⃣ Try local first (works in SageMaker inference container)
    model_dir = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    local_path = os.path.join(model_dir, name)

    if os.path.exists(local_path):
        print(f"✅ Loaded model from local path: {local_path}")
        with open(local_path, "rb") as f:
            return pickle.load(f)

    print(f"⚠️ Local model not found at {local_path}, attempting to load from S3...")

    # 2️⃣ Auto-detect subfolder from filename
    # Example: "market_trx_model.pkl" → subfolder "market_trx"
    base_prefix = "models"
    subfolder = name.split("_model")[0] if "_model" in name else name.split(".")[0]
    s3_prefix = f"{base_prefix}/{subfolder}"

    # Resolve env strictly from SSM
    env = get_env_from_ssm()  

    # 3️⃣ S3 setup
    s3 = boto3.client("s3")
    bucket = f"demo-{env}-sagemaker-data"

    # 4️⃣ Find latest timestamp if not provided
    if timestamp is None:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{s3_prefix}/", Delimiter="/")
        timestamps = sorted(
            [p['Prefix'].split('/')[-2] for p in response.get('CommonPrefixes', []) if len(p['Prefix'].split('/')) > 1]
        )
        if not timestamps:
            raise FileNotFoundError(f"No timestamped folders found under s3://{bucket}/{s3_prefix}/")
        timestamp = timestamps[-1]

    # 5️⃣ Build full key and download
    s3_key = f"{s3_prefix}/{timestamp}/{name}"
    local_tmp_path = os.path.join("/tmp", name)

    try:
        s3.download_file(bucket, s3_key, local_tmp_path)
        print(f"✅ Downloaded model from S3: s3://{bucket}/{s3_key}")
        with open(local_tmp_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise FileNotFoundError(f"❌ Failed to load model from S3 (key={s3_key}): {e}")
    

def get_snowflake_connector(secret_name="snowflake/credentials", region_name="us-east-1"):
    """
    Fetch Snowflake credentials from AWS Secrets Manager and return a live connection.
    The secret must contain keys: user, password, account, warehouse, database, schema, role
    """
    # 1️⃣ Initialize Secrets Manager client
    client = boto3.client("secretsmanager", region_name=region_name)

    # 2️⃣ Retrieve the secret value
    secret_value = client.get_secret_value(SecretId=secret_name)
    secret_dict = json.loads(secret_value["SecretString"])

    # 3️⃣ Connect to Snowflake using secrets
    conn = snowflake.connector.connect(
        user=secret_dict["user"],
        password=secret_dict["password"],
        account=secret_dict["account"],
        warehouse=secret_dict.get("warehouse"),
        database=secret_dict.get("database"),
        schema=secret_dict.get("schema"),
        role=secret_dict.get("role"),
    )
    logging.info("✅ Connected to Snowflake.")
    return conn


def retrieve_df_from_snowflake(
    conn,
    table_name,
    limit=None,
    start_date=None,
    end_date=None,
    feature_ids=None,
    latest_only=True,
    feature_id_name = 'FEATURE_ID',
    indicator_id=None
):


def save_df_to_snowflake(conn, df, table_name):


def plot_actual_predicted(train_df, train_forecast, test_df, test_forecast,name_of_plot = "Actual vs Predicted (Train/Test)"):