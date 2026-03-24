"""
Lambda Helper Functions for SageMaker Step Functions Pipeline
Contains 3 Lambda functions:
  1. evaluate_model_metrics  - checks training metric vs threshold
  2. check_endpoint_exists   - returns whether endpoint already exists
  3. register_model          - registers model in SageMaker Model Registry
"""

import boto3
import json


# ─────────────────────────────────────────────
# 1. evaluate_model_metrics/lambda_function.py
# ─────────────────────────────────────────────
def evaluate_model_metrics(event, context):
    """
    Checks whether the training job's metric meets the required threshold.

    Expected event keys:
      - training_job_name (str)
      - metric_name       (str)  e.g. "validation:auc"
      - threshold         (float)

    Returns:
      - meets_threshold (bool)
      - accuracy        (float)
      - metric_name     (str)
    """
    sm = boto3.client("sagemaker")

    training_job_name = event["training_job_name"]
    metric_name       = event["metric_name"]
    threshold         = float(event["threshold"])

    response = sm.describe_training_job(TrainingJobName=training_job_name)

    # Locate the target metric in the final metric data list
    final_metrics = response.get("FinalMetricDataList", [])
    accuracy = None
    for metric in final_metrics:
        if metric["MetricName"] == metric_name:
            accuracy = metric["Value"]
            break

    if accuracy is None:
        raise ValueError(
            f"Metric '{metric_name}' not found in training job '{training_job_name}'. "
            f"Available metrics: {[m['MetricName'] for m in final_metrics]}"
        )

    meets_threshold = accuracy >= threshold
    print(f"Metric '{metric_name}': {accuracy:.4f} | Threshold: {threshold} | Pass: {meets_threshold}")

    return {
        "meets_threshold": meets_threshold,
        "accuracy":        round(accuracy, 4),
        "metric_name":     metric_name,
        "threshold":       threshold,
    }


# ─────────────────────────────────────────────
# 2. check_endpoint_exists/lambda_function.py
# ─────────────────────────────────────────────
def check_endpoint_exists(event, context):
    """
    Checks whether a SageMaker endpoint exists and is InService.

    Expected event keys:
      - endpoint_name (str)

    Returns:
      - exists (bool)
      - status (str | None)
    """
    sm = boto3.client("sagemaker")
    endpoint_name = event["endpoint_name"]

    try:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status   = response["EndpointStatus"]
        print(f"Endpoint '{endpoint_name}' exists with status: {status}")
        return {"exists": True, "status": status}

    except sm.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint '{endpoint_name}' does not exist.")
            return {"exists": False, "status": None}
        raise  # re-raise unexpected errors


# ─────────────────────────────────────────────
# 3. register_model/lambda_function.py
# ─────────────────────────────────────────────
def register_model(event, context):
    """
    Registers a trained model in the SageMaker Model Registry.

    Expected event keys:
      - model_name           (str)
      - training_job_name    (str)
      - model_package_group  (str)
      - accuracy             (float)

    Returns:
      - model_package_arn (str)
      - status            (str)
    """
    sm = boto3.client("sagemaker")

    model_name          = event["model_name"]
    training_job_name   = event["training_job_name"]
    model_package_group = event["model_package_group"]
    accuracy            = float(event["accuracy"])

    # Retrieve model artifact location from the training job
    training_response  = sm.describe_training_job(TrainingJobName=training_job_name)
    model_data_url     = training_response["ModelArtifacts"]["S3ModelArtifacts"]
    training_image_uri = training_response["AlgorithmSpecification"]["TrainingImage"]

    # Register the model package
    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group,
        ModelPackageDescription=f"Model trained by job {training_job_name}",
        InferenceSpecification={
            "Containers": [
                {
                    "Image":        training_image_uri,
                    "ModelDataUrl": model_data_url,
                }
            ],
            "SupportedContentTypes":       ["text/csv"],
            "SupportedResponseMIMETypes":  ["text/csv"],
        },
        ModelApprovalStatus="Approved",
        CustomerMetadataProperties={
            "accuracy":            str(accuracy),
            "training_job_name":   training_job_name,
            "model_name":          model_name,
        },
    )

    model_package_arn = response["ModelPackageArn"]
    print(f"Model registered: {model_package_arn}")

    return {
        "model_package_arn": model_package_arn,
        "status":            "Registered",
    }