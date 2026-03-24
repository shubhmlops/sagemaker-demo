"""
pipeline.py
-------------
SageMaker Pipeline Definition — Abalone Age Classification

Pipeline steps (run sequentially via depends_on):
  1. PreprocessData  → preprocessing.py  (SKLearn ProcessingStep)
  2. TrainModel      → training.py        (SKLearn TrainingStep)
  3. EvaluateModel   → evaluation.py      (SKLearn ProcessingStep)
  4. RegisterModel   → SageMaker Model Registry (if F1 ≥ threshold)

After pipeline succeeds → deploys the latest approved model to an endpoint.

Usage:
    ENV=dev python3 pipeline.py
"""

import os
import sys
import time
import boto3
import sagemaker
from sagemaker import Session
from sagemaker.model import ModelPackage
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
import sagemaker.inputs

# ── Base directory (all scripts live next to this file) ───────────────────────
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def endpoint_exists(endpoint_name: str, region: str) -> bool:
    """Check if a SageMaker endpoint already exists (to decide create vs update)."""
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        return True
    except sm.exceptions.ClientError as e:
        if "Could not find" in str(e) or "does not exist" in str(e):
            return False
        raise


def get_latest_approved_model_package(model_group: str, region: str) -> str:
    """Return the ARN of the most recently approved model in a model package group."""
    sm = boto3.client("sagemaker", region_name=region)
    resp = sm.list_model_packages(
        ModelPackageGroupName=model_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    if not resp["ModelPackageSummaryList"]:
        raise ValueError(f"No Approved model package found in group: {model_group}")
    return resp["ModelPackageSummaryList"][0]["ModelPackageArn"]


def deploy_latest_model(env: str, region: str, endpoint_name: str, role_arn: str,
                         model_group: str = "AbaloneModelPackageGroup"):
    """Deploy (or update) the latest approved model to a real-time endpoint."""
    session = Session(boto3.Session(region_name=region))
    model_package_arn = get_latest_approved_model_package(model_group, region)

    print(f"Deploying model package: {model_package_arn}")

    model = ModelPackage(
        role=role_arn,
        model_package_arn=model_package_arn,
        sagemaker_session=session,
    )

    exists = endpoint_exists(endpoint_name, region)
    print(f"Endpoint '{endpoint_name}' already exists: {exists}")

    model.deploy(
        endpoint_name=endpoint_name,
        instance_type="ml.m5.large",
        initial_instance_count=1,
        update_endpoint=exists,   # update in-place if endpoint exists
    )
    print(f"✅ Endpoint ready: {endpoint_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline definition
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline(env: str, region: str):
    """Build and return the SageMaker Pipeline object."""

    pipeline_session = PipelineSession()

    # Config — in a real project pull these from Secrets Manager or SSM
    role_arn  = os.environ.get("SAGEMAKER_ROLE_ARN", sagemaker.get_execution_role())
    bucket    = os.environ.get("S3_BUCKET", sagemaker.Session().default_bucket())
    data_key  = os.environ.get("INPUT_DATA_KEY", "data/data.csv")
    threshold = float(os.environ.get("ACCURACY_THRESHOLD", "0.80"))

    print(f"Env      : {env}")
    print(f"Region   : {region}")
    print(f"Role     : {role_arn}")
    print(f"Data     : s3://{bucket}/{data_key}")

    # ── Pipeline parameters (can be overridden per execution) ─────────────────
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{bucket}/{data_key}",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="Approved",    # auto-approve for demo; use PendingManualApproval in prod
    )
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=threshold,
    )

    # ── STEP 1 — Preprocessing ────────────────────────────────────────────────
    preprocessor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role_arn,
        sagemaker_session=pipeline_session,
    )

    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=preprocessor,
        inputs=[
            ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train",      source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/output/validation"),
            ProcessingOutput(output_name="test",       source="/opt/ml/processing/output/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocessing.py"),
    )

    # ── STEP 2 — Training ─────────────────────────────────────────────────────
    estimator = SKLearn(
        entry_point=os.path.join(BASE_DIR, "training.py"),
        framework_version="1.2-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=role_arn,
        sagemaker_session=pipeline_session,
        output_path=f"s3://{bucket}/training/output/",
        hyperparameters={"n_estimators": 100, "max_depth": 5},
    )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train":      sagemaker.inputs.TrainingInput(
                step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
            "validation": sagemaker.inputs.TrainingInput(
                step_preprocess.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
            ),
        },
        depends_on=[step_preprocess],   # explicit ordering (mirrors reference pattern)
    )

    # ── STEP 3 — Evaluation ───────────────────────────────────────────────────
    evaluator = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role_arn,
        sagemaker_session=pipeline_session,
    )

    eval_report = PropertyFile(
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
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluation.py"),
        property_files=[eval_report],
        depends_on=[step_train],
    )

    # ── STEP 4 — Register Model (gated by F1 score) ───────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_evaluate.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    step_register = RegisterModel(
        name="RegisterAbaloneModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        model_package_group_name="AbaloneModelPackageGroup",
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        depends_on=[step_evaluate],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
    )

    # Only register if F1 ≥ threshold
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

    # ── Assemble pipeline ─────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=f"AbalonePipeline-{env}",
        parameters=[input_data_uri, model_approval_status, accuracy_threshold],
        steps=[step_preprocess, step_train, step_evaluate, step_condition],
        sagemaker_session=pipeline_session,
    )

    return pipeline, role_arn


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ENV    = os.getenv("ENV", "dev")       # dev | uat | prod
    REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # 1. Build & upsert pipeline definition
    pipeline, role_arn = get_pipeline(ENV, REGION)
    pipeline.upsert(role_arn=role_arn)
    print("✅ Pipeline upserted")

    # 2. Start execution
    execution = pipeline.start()
    print("⏳ Pipeline execution started:", execution.arn)

    # 3. Poll until done (CodePipeline needs a blocking call to know pass/fail)
    while True:
        desc   = execution.describe()
        status = desc["PipelineExecutionStatus"]
        print(f"   Status: {status}")

        if status in ("Succeeded", "Failed", "Stopped"):
            break

        time.sleep(20)

    print(f"🔵 Final status: {status}")

    # 4. Fail the CodeBuild step if pipeline didn't succeed
    if status != "Succeeded":
        print(f"❌ Pipeline failed with status: {status}")
        sys.exit(1)

    # 5. Deploy the latest approved model
    print("✅ Pipeline succeeded — deploying model...")
    deploy_latest_model(
        env=ENV,
        region=REGION,
        endpoint_name=f"abalone-age-endpoint-{ENV}",
        role_arn=role_arn,
        model_group="AbaloneModelPackageGroup",
    )

    sys.exit(0)