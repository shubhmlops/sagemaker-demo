# SageMaker Pipeline Demo

A minimal end-to-end ML pipeline for team demos using the **Abalone dataset**.  
Each file is intentionally short and heavily commented.

## Dataset — Abalone

📌 What is the Abalone Dataset?

The Abalone dataset is a classic dataset used in machine learning.

👉 Goal:

Predict the age of an abalone (a type of sea snail)

👉 Why is it tricky?

You can’t directly measure age easily.
Instead, scientists:

Cut the shell

Count the number of rings


`data.csv` — 4,177 rows, 9 features, multi-class classification problem.

| Column | Type | Description |
|---|---|---|
| `sex` | int | 0 = Male, 1 = Female, 2 = Infant |
| `length` | float | Longest shell measurement (mm) |
| `diameter` | float | Perpendicular to length (mm) |
| `height` | float | Height with meat in shell (mm) |
| `whole_weight` | float | Whole abalone weight (g) |
| `shucked_weight` | float | Weight of meat (g) |
| `viscera_weight` | float | Gut weight after bleeding (g) |
| `shell_weight` | float | Shell weight after drying (g) |
| `rings` | int | Raw age indicator (like tree rings) |
| `target` | int | **Age group: 0 = Young, 1 = Middle, 2 = Old** |

The original dataset is a regression problem (predicting `rings`).  
Here it's framed as **3-class classification** on age group — works with the pipeline out of the box.

**Target distribution:**
```
0 (Young,  rings ≤ 8)  → 1983 samples
1 (Middle, rings 9-11) →  867 samples
2 (Old,    rings ≥ 12) → 1327 samples
```

## File Map

```
data.csv             → Abalone dataset (4177 rows, ready to upload to S3)
preprocessing.py   → cleans data, normalises features, splits train/val/test
training.py        → trains a Random Forest classifier, saves model artifact
evaluation.py      → scores model on test set, writes evaluation.json
pipeline.py        → wires everything into a SageMaker Pipeline
```

## Pipeline Flow

```
[1 Preprocess] ──► [2 Train] ──► [3 Evaluate] ──► [Condition: F1 ≥ 0.80?]
                                                         │
                                                   Yes ──► Register Model
                                                   No  ──► (skip)
```

## Quick Start - Local

```bash
pip install sagemaker scikit-learn boto3

# Upload the sample data to S3:
aws s3 cp data.csv s3://<your-bucket>/data/data.csv

# Run the pipeline
python pipeline.py
```

## SageMaker
```
Run Codepipeline with Latest
BuildSpec will run the Pipeline.py and it will call the Sagemaker execution.
Check the Execution in SageMaker Studio - Pipelines Section
```

## What each SageMaker concept does

| Concept | Purpose |
|---|---|
| `ProcessingStep` | Run a Python script on a managed container (good for ETL & eval) |
| `TrainingStep` | Run training on a managed container, saves model.tar.gz to S3 |
| `ConditionStep` | Branch the pipeline based on a metric value |
| `RegisterModel` | Add the model to the SageMaker Model Registry for deployment |
| `ParameterString/Float` | Pipeline-level inputs you can override per execution |

## Extending for Claude Code

Later you can drop Claude Code in to:
- Auto-generate feature engineering logic in `preprocessing.py`
- Suggest better hyperparameters in `pipeline.py`
- Explain what each ring count means biologically and why it maps to age
- Explain any step in plain English
