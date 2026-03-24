"""
pipeline.py
-------------
SageMaker Pipeline Definition — Abalone Age Classification

Pipeline steps (run sequentially via depends_on):
  1. PreprocessData  → preprocessing.py  (SKLearn ProcessingStep)
  2. TrainModel      → training.py        (SKLearn TrainingStep)
  3. EvaluateModel   → evaluation.py      (SKLearn ProcessingStep)
  4. RegisterModel   → SageMaker Model Registry (if F1 ≥ threshold)

After pipeline succeeds:
  → Deploys the latest approved model to an endpoint (with data capture ON)
  → Runs a baseline job on training data to learn "normal" feature distributions
  → Creates an hourly monitoring schedule to detect data drift

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
from sagemaker.model_monitor import DefaultModelMonitor, DataCaptureConfig
from sagemaker.model_monitor.dataset_format import DatasetFormat
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
                         bucket: str, model_group: str = "AbaloneModelPackageGroup"):
    """Deploy (or update) the latest approved model to a real-time endpoint.

    Data capture is enabled so all inference inputs/outputs are saved to S3.
    These captured requests feed the model monitoring schedule.
    """
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

    # Delete stale endpoint config if it exists — SageMaker keeps configs around
    # even after an endpoint is deleted, which blocks CreateEndpointConfig on redeploy.
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Deleted stale endpoint config: {endpoint_name}")
    except sm.exceptions.ClientError as e:
        if "Could not find" not in str(e) and "does not exist" not in str(e):
            raise

    # Capture 100% of requests/responses → stored in S3 for drift monitoring
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=f"s3://{bucket}/monitoring/data-capture/{endpoint_name}",
    )

    model.deploy(
        endpoint_name=endpoint_name,
        instance_type="ml.m5.large",
        initial_instance_count=1,
        update_endpoint=exists,
        data_capture_config=data_capture_config,
    )
    print(f"✅ Endpoint ready: {endpoint_name}")


def setup_model_monitoring(env: str, region: str, endpoint_name: str, role_arn: str,
                            bucket: str, baseline_data_uri: str):
    """Set up SageMaker Model Monitor on the deployed endpoint.

    Steps:
      1. Run a baseline job — analyses the training data to learn expected
         feature distributions (mean, std, min/max, nulls etc.)
      2. Create an hourly monitoring schedule — compares live captured traffic
         against that baseline and flags violations in CloudWatch.
    """
    session = Session(boto3.Session(region_name=region))

    monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=session,
    )

    baseline_results_uri = f"s3://{bucket}/monitoring/baseline/{env}/"

    # Step 1 — Baseline: learn what "normal" looks like from training data
    print("Running baseline job — this analyses training data distributions...")
    monitor.suggest_baseline(
        baseline_dataset=baseline_data_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_results_uri,
        wait=True,
        logs=False,
    )
    print(f"✅ Baseline complete. Results at: {baseline_results_uri}")

    # Step 2 — Monitoring schedule: hourly drift check against the baseline
    schedule_name = f"abalone-data-monitor-{env}"

    # Delete existing schedule first so we always get a fresh one with the new baseline
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.delete_monitoring_schedule(MonitoringScheduleName=schedule_name)
        print(f"Deleted existing monitoring schedule: {schedule_name}")
        time.sleep(10)  # wait for deletion to propagate
    except sm.exceptions.ResourceNotFound:
        pass

    monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=endpoint_name,
        output_s3_uri=f"s3://{bucket}/monitoring/reports/{env}/",
        statistics=monitor.baseline_statistics(),
        constraints=monitor.suggested_constraints(),
        schedule_cron_expression="cron(0 * ? * * *)",   # every hour
    )
    print(f"✅ Monitoring schedule active: {schedule_name} (runs hourly)")
    print("   Violations will appear in CloudWatch under /aws/sagemaker/Endpoints")


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
    ENV      = os.getenv("ENV", "dev")
    REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    BUCKET   = os.getenv("S3_BUCKET", sagemaker.Session().default_bucket())
    DATA_KEY = os.getenv("INPUT_DATA_KEY", "data/data.csv")

    ENDPOINT_NAME = f"abalone-age-endpoint-{ENV}"

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

    # 5. Deploy the latest approved model (data capture enabled)
    print("✅ Pipeline succeeded — deploying model...")
    deploy_latest_model(
        env=ENV,
        region=REGION,
        endpoint_name=ENDPOINT_NAME,
        role_arn=role_arn,
        bucket=BUCKET,
        model_group="AbaloneModelPackageGroup",
    )

    # 6. Set up model monitoring — baseline + hourly drift schedule
    print("Setting up model monitoring...")
    setup_model_monitoring(
        env=ENV,
        region=REGION,
        endpoint_name=ENDPOINT_NAME,
        role_arn=role_arn,
        bucket=BUCKET,
        baseline_data_uri=f"s3://{BUCKET}/{DATA_KEY}",  # training data as baseline
    )

    sys.exit(0)