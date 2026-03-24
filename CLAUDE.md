# SageMaker Demo — Abalone Age Classification Pipeline

## Project Summary
End-to-end MLOps pipeline on AWS SageMaker that trains a Random Forest model to classify
abalone age (Young / Middle / Old) from physical shell measurements. Built as a team demo
for production-style MLOps workflows.

## Pipeline Flow
```
[1 Preprocess] → [2 Train] → [3 Evaluate] → [Condition: F1 ≥ 0.80?]
                                                       │
                                               YES → Register Model
                                               NO  → (skip)

After pipeline succeeds:
  → Deploy model to endpoint (with data capture ON)
  → Run baseline job on training data
  → Create hourly drift monitoring schedule
```

## Key Files
| File | Purpose |
|------|---------|
| `pipeline.py` | Main entrypoint — defines and runs the full pipeline |
| `training.py` | Trains Random Forest + contains inference handlers (model_fn, predict_fn) |
| `preprocessing.py` | Cleans data, normalises features, splits train/val/test |
| `evaluation.py` | Scores model on test set, writes evaluation.json |
| `buildspec.yml` | AWS CodeBuild spec — uploads data.csv to S3 if present, then runs pipeline |

## Rules — Important
- **DO NOT touch the `state_machine/` folder** — it is kept as a standalone demo example only
- All pipeline work goes through `pipeline.py` exclusively
- The `state_machine/` folder uses Step Functions + XGBoost; `pipeline.py` uses SageMaker Pipelines + Random Forest — they are separate and intentionally different

## Running the Pipeline
```bash
ENV=dev python pipeline.py
```

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `dev` | Environment name (dev / uat / prod) |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET` | SageMaker default bucket | S3 bucket for all artifacts |
| `INPUT_DATA_KEY` | `data/data.csv` | S3 key for input data |
| `ACCURACY_THRESHOLD` | `0.80` | Minimum F1 score to register model |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role | IAM role for SageMaker |

## AWS Resource Names
- **Endpoint:** `abalone-age-endpoint-{env}`
- **Model group:** `AbaloneModelPackageGroup`
- **Monitoring schedule:** `abalone-data-monitor-{env}`
- **Pipeline name:** `AbalonePipeline-{env}`

## S3 Structure
```
s3://<bucket>/
  data/data.csv                          ← raw input
  training/output/                       ← model artifacts
  monitoring/
    data-capture/<endpoint-name>/        ← captured inference traffic
    baseline/<env>/                      ← baseline statistics & constraints
    reports/<env>/                       ← hourly drift reports
```

## Model Deployment Notes
- Deployment uses boto3 directly (not model.deploy()) to reliably handle create vs update
- Endpoint config and model names are timestamped to avoid name collisions on re-runs
- `training.py` copies itself to `/opt/ml/model/code/` during training so the serving
  container can find it via `SAGEMAKER_PROGRAM=training.py`
- Data capture is ON at 100% sampling — feeds the monitoring schedule

## Dataset — Abalone (UCI)
4,177 rows, 9 features. Target is age group:
- `0` = Young  (rings ≤ 8)
- `1` = Middle (rings 9–11)
- `2` = Old    (rings ≥ 12)
