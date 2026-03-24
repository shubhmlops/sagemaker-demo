from sagemaker.workflow.steps import TrainingStep
from sagemaker.network import NetworkConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker import ModelPackage, Session
from utils import *
import os
import time
import json
import boto3
from sagemaker.workflow.parameters import ParameterString

run_scope = ParameterString(
    name="RunScope",
    default_value="all"   # all | monthly | area | weekly
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

session = PipelineSession()

###################### Inference Endpoint Step ####################

def endpoint_exists(endpoint_name: str, region: str) -> bool:
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        return True
    except sm.exceptions.ClientError as e:
        if "Could not find" in str(e) or "does not exist" in str(e):
            return False
        raise


def get_latest_approved_model_package(group_name, region):
    sm = boto3.client("sagemaker", region_name=region)

    resp = sm.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1
    )

    if not resp["ModelPackageSummaryList"]:
        raise ValueError("No Approved model package found")

    return resp["ModelPackageSummaryList"][0]["ModelPackageArn"]

def deploy_latest_model(env, region, endpoint_name, role_arn, model_group="DemoModelPackageGroup"):
    session = Session(boto3.Session(region_name=region))
    model_package_arn = get_latest_approved_model_package(model_group, region)

    print(f"Deploying model: {model_package_arn}")

    model = ModelPackage(
        role=role_arn,
        model_package_arn=model_package_arn,
        sagemaker_session=session
    )

    exists = endpoint_exists(endpoint_name, region)

    print(f"Endpoint exists? {exists}")

    return model.deploy(
        endpoint_name=endpoint_name,
        instance_type="ml.c4.large",
        initial_instance_count=1,
        update_endpoint=exists  # Only update if it already exists
    )


###################### Inference Endpoint Step ####################


def get_pipeline(env: str, region: str, selected_scope: str = "all"):
    session = PipelineSession()

    secret_name = f"{env}/sagemaker/pipeline-config-vars"
    config = get_secret(secret_name, region)

    role_arn = config["execution_role_arn"]
    subnets = config["subnets"]
    security_groups = config["security_group_ids"]
    ecr_image_uri = config["ecr_image_uri"]

    assert isinstance(subnets, list)
    assert isinstance(security_groups, list)


    Demo_JOBS = [
        {
            "name": "DataProcessing",
            "entry_point": "Demo_monthly_data_loading.py",
            "register_model": False,
            "scope": "monthly"
        },
        {
            "name": "DataPull",
            "entry_point": "data_pull.sql",
            "register_model": False,
            "scope": "monthly"
        },
        {
            "name": "ModelTraining",
            "entry_point": "model_training.py",
            "register_model": True,
            "model_group": "DemoFeatureModelPackageGroup",
            "scope": "monthly"
        },
        {
            "name": "ModelInference",
            "entry_point": "model_inference.py",
            "register_model": True,
            "model_group": "DemoMainModelPackageGroup",
            "scope": "monthly"
        }
    ]

    filtered_jobs = [
        job for job in Demo_JOBS
        if selected_scope == "all" or job["scope"] == selected_scope
    ]

    steps = []
    previous_step = None

    for job in filtered_jobs:
        estimator = PyTorch(
            image_uri=ecr_image_uri,
            entry_point=os.path.join(BASE_DIR, job["entry_point"]),
            source_dir=BASE_DIR,
            role=role_arn,
            instance_type="ml.m5.large",
            instance_count=1,
            output_path=f"s3://lundbeck-{env}-sagemaker-data/training/output/Demo/{job['name']}/",
            sagemaker_session=session,
            subnets=subnets,
            security_group_ids=security_groups,
            environment={
                "SAGEMAKER_PROGRAM": job["entry_point"],
                "ENV": env
            }
        )

        train_step = TrainingStep(
            name=f"{job['name']}Step",
            estimator=estimator,
            depends_on=[previous_step] if previous_step else None
        )

        steps.append(train_step)

        if job.get("register_model", False):
            register_step = RegisterModel(
                name=f"Register{job['name']}Model",
                estimator=estimator,
                model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                content_types=["application/json"],
                response_types=["application/json"],
                model_package_group_name=job["model_group"],
                approval_status="PendingManualApproval",
                depends_on=[train_step]
            )
            steps.append(register_step)
            previous_step = register_step
        else:
            previous_step = train_step

    pipeline = Pipeline(
        name=f"LundbeckDemoPipelineTraining-{env}",
        steps=steps,
        sagemaker_session=session
    )

    return pipeline, role_arn

if __name__ == "__main__":
    ENV = os.getenv("ENV")   # dev | uat | prod
    REGION = "us-east-1"
    pipeline, role_arn = get_pipeline(ENV, REGION, selected_scope="all")
    pipeline.upsert(role_arn=role_arn)
    execution = pipeline.start()

    print("✅ Pipeline execution started:", execution.arn)

    print("⏳ Waiting for pipeline execution to finish...")


    while True:
        desc = execution.describe()
        status = desc["PipelineExecutionStatus"]
        print("Current Status:", status)

        if status in ("Succeeded", "Failed", "Stopped"):
            break

        time.sleep(20)

    print("🔵 Final Status:", status)


    if status != "Succeeded":
        print(f"❌ SageMaker Pipeline failed with status: {status}")
        sys.exit(1)   # <-- makes CodePipeline FAIL
    else:
        print("✅ SageMaker Pipeline succeeded")

        deploy_latest_model(
                env=ENV,
                region=REGION,
                endpoint_name=f"Demo-endpoint-inference-{ENV}-auto",
                role_arn=role_arn,
                model_group="DemoMainModelPackageGroup"
            )
        sys.exit(0)