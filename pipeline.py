import os
import boto3
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics
import sagemaker.inputs


# ── Config from environment ─────────────────────────────
role = os.environ["SAGEMAKER_ROLE_ARN"]
bucket = os.environ["S3_BUCKET"]
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

pipeline_name = os.environ.get("PIPELINE_NAME", "MyMLPipeline")
data_key = os.environ.get("INPUT_DATA_KEY", "data/data.csv")
threshold = float(os.environ.get("ACCURACY_THRESHOLD", "0.80"))
approval = os.environ.get("MODEL_APPROVAL_STATUS", "PendingManualApproval")

session = sagemaker.Session(boto_session=boto3.Session(region_name=region))


# ── Parameters ──────────────────────────────────────────
input_data_uri = ParameterString(
    name="InputDataUri",
    default_value=f"s3://{bucket}/{data_key}",
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value=approval,
)

accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=threshold,
)


# ── Step 1: Preprocess ─────────────────────────────────
preprocessor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    sagemaker_session=session,
)

step_preprocess = ProcessingStep(
    name="PreprocessData",
    processor=preprocessor,
    inputs=[
        ProcessingInput(
            source=input_data_uri,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test"),
    ],
    code="preprocessing.py",
)


# ── Step 2: Train ──────────────────────────────────────
estimator = SKLearn(
    entry_point="training.py",
    framework_version="1.2-1",
    instance_type="ml.m5.xlarge",
    role=role,
    sagemaker_session=session,
    hyperparameters={"n_estimators": 100, "max_depth": 5},
)

step_train = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
        ),
        "validation": sagemaker.inputs.TrainingInput(
            step_preprocess.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
        ),
    },
)


# ── Step 3: Evaluate ───────────────────────────────────
evaluator = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    sagemaker_session=session,
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

step_evaluate = ProcessingStep(
    name="EvaluateModel",
    processor=evaluator,
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    code="evaluation.py",
    property_files=[evaluation_report],
)


# ── Step 4: Condition + Register ───────────────────────
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=step_evaluate.properties.ProcessingOutputConfig.Outputs[
            "evaluation"
        ].S3Output.S3Uri,
        content_type="application/json",
    )
)

step_register = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    approval_status=model_approval_status,
    model_metrics=model_metrics,
)

condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_evaluate.name,
        property_file="EvaluationReport",
        json_path="metrics.f1",
    ),
    right=accuracy_threshold,
)

step_condition = ConditionStep(
    name="CheckModelF1",
    conditions=[condition],
    if_steps=[step_register],
    else_steps=[],
)


# ── Pipeline ───────────────────────────────────────────
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[input_data_uri, model_approval_status, accuracy_threshold],
    steps=[step_preprocess, step_train, step_evaluate, step_condition],
    sagemaker_session=session,
)

pipeline.upsert(role_arn=role)
print(f"Pipeline '{pipeline_name}' upserted successfully ✓")

execution = pipeline.start()
print(f"Execution ARN: {execution.arn}")