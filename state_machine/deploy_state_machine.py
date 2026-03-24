"""
Deploy the Step Functions State Machine using boto3 (Python SDK)
Run this script once to create/update the state machine in AWS.

Prerequisites:
  pip install boto3
  AWS credentials configured (aws configure or env vars)
"""

import json
import boto3

# ── Configuration ──────────────────────────────────────────────────────────────
ACCOUNT_ID  = "your-account-id"
REGION      = "us-east-1"
ROLE_ARN    = f"arn:aws:iam::{ACCOUNT_ID}:role/StepFunctionsExecutionRole"
STATE_MACHINE_NAME = "SageMakerMLPipeline"

# Load the state machine definition
with open("step_function_definition.json") as f:
    DEFINITION = json.dumps(json.load(f))  # minified JSON string


def create_or_update_state_machine():
    sf_client = boto3.client("stepfunctions", region_name=REGION)

    # Check if it already exists
    existing_arn = None
    paginator = sf_client.get_paginator("list_state_machines")
    for page in paginator.paginate():
        for sm in page["stateMachines"]:
            if sm["name"] == STATE_MACHINE_NAME:
                existing_arn = sm["stateMachineArn"]
                break

    if existing_arn:
        print(f"Updating existing state machine: {existing_arn}")
        sf_client.update_state_machine(
            stateMachineArn=existing_arn,
            definition=DEFINITION,
            roleArn=ROLE_ARN,
        )
        print("✅ State machine updated.")
        return existing_arn

    print("Creating new state machine...")
    response = sf_client.create_state_machine(
        name=STATE_MACHINE_NAME,
        definition=DEFINITION,
        roleArn=ROLE_ARN,
        type="STANDARD",
        loggingConfiguration={
            "level": "ERROR",
            "includeExecutionData": True,
            "destinations": [
                {
                    "cloudWatchLogsLogGroup": {
                        "logGroupArn": f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:/aws/states/{STATE_MACHINE_NAME}:*"
                    }
                }
            ],
        },
        tracingConfiguration={"enabled": True},
    )
    arn = response["stateMachineArn"]
    print(f"✅ State machine created: {arn}")
    return arn


def start_pipeline_execution(state_machine_arn: str, run_name: str = "manual-run"):
    sf_client = boto3.client("stepfunctions", region_name=REGION)

    response = sf_client.start_execution(
        stateMachineArn=state_machine_arn,
        name=run_name,
        input=json.dumps({
            "pipeline_config": {
                "bucket":       "your-bucket",
                "prefix":       "ml-pipeline",
                "model_prefix": "xgboost-classifier",
            }
        }),
    )
    print(f"🚀 Execution started: {response['executionArn']}")
    return response["executionArn"]


# ── IAM Policy (attach to StepFunctionsExecutionRole) ─────────────────────────
IAM_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SageMakerFullAccess",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateProcessingJob",
                "sagemaker:DescribeProcessingJob",
                "sagemaker:StopProcessingJob",
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:CreateModel",
                "sagemaker:DescribeModel",
                "sagemaker:DeleteModel",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:CreateEndpoint",
                "sagemaker:DescribeEndpoint",
                "sagemaker:UpdateEndpoint",
                "sagemaker:CreateModelPackage",
                "sagemaker:ListModelPackages",
            ],
            "Resource": "*",
        },
        {
            "Sid": "LambdaInvoke",
            "Effect": "Allow",
            "Action": ["lambda:InvokeFunction"],
            "Resource": [
                f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:evaluate-model-metrics",
                f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:check-endpoint-exists",
                f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:register-model",
            ],
        },
        {
            "Sid": "SNSPublish",
            "Effect": "Allow",
            "Action": ["sns:Publish"],
            "Resource": f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:ml-pipeline-notifications",
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogDelivery",
                "logs:PutLogEvents",
                "logs:GetLogDelivery",
            ],
            "Resource": "*",
        },
        {
            "Sid": "XRayTracing",
            "Effect": "Allow",
            "Action": ["xray:PutTraceSegments", "xray:GetSamplingRules"],
            "Resource": "*",
        },
    ],
}


if __name__ == "__main__":
    arn = create_or_update_state_machine()
    print(f"\nState Machine ARN: {arn}")
    print("\nTo start a pipeline run:")
    print(f'  execution_arn = start_pipeline_execution("{arn}", "run-001")')
    print("\nIAM Policy (save as iam_policy.json and attach to your role):")
    print(json.dumps(IAM_POLICY, indent=2))