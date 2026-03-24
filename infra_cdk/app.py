#!/usr/bin/env python3
import aws_cdk as cdk
from infra_resources.vpc_stack import VpcStack
from sagemaker_resources.sagemaker_stack_dev import SageMakerStack
from infra_resources.code_connection import GitHubConnectionStack
from pipelines.cdk_sagemaker_pipeline_vypepti.vypeti_model_build import SageMakerPipelineStack
from infra_resources.ssm import ParameterStoreStack


common_env = cdk.Environment(region="us-east-1")
account_dev = "864683085375"

AWS_REGION = "us-east-1"
ENVIRONMENT = "dev"

CODE_STAR_CONNECTION_ARN = "arn:aws:codeconnections:us-east-1:{account_dev}:connection/0dd62ab7-e6d7-4585-a41b-8d4a1aba0823"

app = cdk.App()

# GitHub connection
connection_stack = GitHubConnectionStack(app, "GitHubConnectionStack", env=common_env)

# VPC
vpc_stack = VpcStack(app, "VpcStack", env=common_env)


sagemaker_stack = SageMakerStack(
    app,
    "SageMakerStack",
    vpc_id=Fn.import_value("VpcId"),
    private_subnet_ids=[
        Fn.import_value("PrivateSubnet1Id"),
        Fn.import_value("PrivateSubnet2Id"),
    ],
    env=cdk.Environment(region="us-east-1")
)

sagemaker_stack.add_dependency(vpc_stack)

github_owner = ""
github_repo = "repo_name"
codestar_connection_arn = "arn:aws:codeconnections:us-east-1:{account_dev}:connection/0dd62ab7-e6d7-4585-a41b-8d4a1aba0823"
github_branch = "release/modelling_1.0.0"  # or whichever branch you want to watch
github_folder = "modelling/"

# 💡 Stack definition - ModelBuildMonthly
sagemaker_pipeline_stack = SageMakerPipelineStack(
    app,
    "SageMakerPipelineStack",
    github_owner=github_owner,
    github_repo=github_repo,
    codestar_connection_arn=codestar_connection_arn,
    github_branch=github_branch,
    github_folder=github_folder,
    env=cdk.Environment(
        region="us-east-1"
    )
)

ParameterStoreStack(
    app,
    "ParameterStoreStack",
    env=common_env
)

app.synth()