# Step Functions 

Serverless service that allows you to build and manage workflows for Distributed applications, automate processes, orchestrate microservices and create data and ml pipelines.

# SageMaker ML Pipeline — AWS Step Functions

A production-grade ML pipeline that automates preprocessing → training → evaluation → deployment using AWS Step Functions + SageMaker.

---

## Pipeline Architecture

```
S3 Raw Data
    │
    ▼
┌─────────────────────┐
│  PreprocessingJob   │  SageMaker Processing Job
│  (Data Cleaning)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    TrainingJob      │  SageMaker Training (XGBoost)
│   (XGBoost Model)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  EvaluateAccuracy   │  Lambda → checks AUC ≥ threshold
└────────┬────────────┘
         │
    ┌────┴────┐
    │ Choice  │  AUC ≥ 0.75?
    └────┬────┘
   YES   │   NO → ModelAccuracyInsufficient (Fail)
         ▼
┌─────────────────────┐
│    CreateModel      │  SageMaker Model
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ CreateEndpointConfig│  With data capture enabled
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  CheckEndpointExists│  Lambda
└────────┬────────────┘
         │
    ┌────┴──────┐
    │  Exists?  │
    └────┬──────┘
   YES   │   NO
    ▼         ▼
Update     Create
Endpoint   Endpoint
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────────┐
│  RegisterModel      │  Lambda → SageMaker Model Registry
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   NotifySuccess     │  SNS notification
└────────┬────────────┘
         │
         ▼
      SUCCEED
```

---

## Files

| File | Description |
|------|-------------|
| `step_function_definition.json` | Step Functions state machine (ASL) |
| `lambda_helpers.py` | 3 Lambda functions used by the pipeline |
| `deploy_state_machine.py` | Python script to deploy + run the pipeline |

---

## Setup Steps

### 1. Replace placeholders
Search and replace `your-account-id` and `your-bucket` across all files.

### 2. Deploy the 3 Lambda functions
Package each function from `lambda_helpers.py` into its own Lambda:

- `evaluate-model-metrics`  → `evaluate_model_metrics()`
- `check-endpoint-exists`   → `check_endpoint_exists()`
- `register-model`          → `register_model()`

### 3. Create IAM role for Step Functions
Create role `StepFunctionsExecutionRole` and attach the policy printed by `deploy_state_machine.py`.

### 4. Create IAM role for SageMaker
Create role `SageMakerExecutionRole` with:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or scoped to your bucket)

### 5. Create SNS topic
```bash
aws sns create-topic --name ml-pipeline-notifications
aws sns subscribe --topic-arn <ARN> --protocol email --notification-endpoint you@example.com
```

### 6. Deploy the state machine
```bash
pip install boto3
python deploy_state_machine.py
```

### 7. Start a pipeline run
```python
from deploy_state_machine import start_pipeline_execution
start_pipeline_execution("<state-machine-arn>", "run-001")
```

---

## Key Features

- ✅ **Auto create or update** endpoint (no manual check needed)
- ✅ **Accuracy gate** — stops deployment if model underperforms
- ✅ **Data capture** — logs all inference inputs/outputs to S3
- ✅ **Model Registry** — every deployed model is versioned
- ✅ **SNS alerts** — notified on success or failure
- ✅ **X-Ray tracing** + CloudWatch logging built-in

---

## Customization

| What to change | Where |
|----------------|-------|
| ML framework (PyTorch, TF) | `TrainingJob.AlgorithmSpecification.TrainingImage` |
| Instance type | `ResourceConfig.InstanceType` |
| Accuracy threshold | `EvaluateModelAccuracy.Payload.threshold` |
| Endpoint instance | `ProductionVariants.InstanceType` |
| Add auto-scaling | Add `ApplicationAutoScaling` after endpoint creation |
