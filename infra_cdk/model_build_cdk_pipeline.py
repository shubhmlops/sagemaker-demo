from aws_cdk import (
    Stack,
    aws_codepipeline as codepipeline,
    aws_codepipeline_actions as cp_actions,
    aws_codebuild as codebuild,
    aws_iam as iam,
    aws_ec2 as ec2,
    CfnOutput,
    Tags,
)
from constructs import Construct


class SageMakerPipelineStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        github_owner: str,
        github_repo: str,
        codestar_connection_arn: str,
        github_branch: str = "branch_name",
        github_folder: str = "modelling/..",
        env_name: str = "dev",
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        source_output = codepipeline.Artifact()
        build_output = codepipeline.Artifact()

        vpc_id = "vpc-xxxxxxxxxxxxxxxxxx"
        subnet_ids = [
            "subnet-xxxxxxxxxxxx",
            "subnet-xxxxxxxxxxxx",
        ]
        security_group_ids = [
            "sg-xxxxxxxxxxxx",
        ]

        vpc = ec2.Vpc.from_vpc_attributes(
            self,
            "Vpc",
            vpc_id=vpc_id,
            availability_zones=self.availability_zones,
            private_subnet_ids=subnet_ids,
        )

        subnet_selection = ec2.SubnetSelection(
            subnets=[
                ec2.Subnet.from_subnet_id(
                    self,
                    f"Subnet{i}",
                    subnet_id,
                )
                for i, subnet_id in enumerate(subnet_ids)
            ]
        )

        security_groups = [
            ec2.SecurityGroup.from_security_group_id(
                self,
                f"SG{i}",
                sg_id,
            )
            for i, sg_id in enumerate(security_group_ids)
        ]

        # 3️⃣ CodeBuild Project
        build_project = codebuild.PipelineProject(
            self,
            "SageMakerPipelineBuild",
            project_name="Sample-MLOPs",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.from_code_build_image_id(
                    "aws/codebuild/amazonlinux-x86_64-standard:5.0"
                ),
                privileged=True,
                environment_variables={
                    "ENV": codebuild.BuildEnvironmentVariable(value=env_name),
                    "AWS_REGION": codebuild.BuildEnvironmentVariable(value=self.region),
                },
            ),
            vpc=vpc,
            subnet_selection=subnet_selection,
            security_groups=security_groups,
            build_spec=codebuild.BuildSpec.from_source_filename(
                f"{github_folder}buildspec.yml"
            ),
        )

        Tags.of(build_project).add("Project", "ModelBuild-Monthly")
        Tags.of(build_project).add("Environment", env_name)

        # 4️⃣ IAM Permissions (SageMaker / ECR / S3)
        build_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:*",
                    "iam:GetRole",
                    "iam:PassRole",
                    "sts:AssumeRole",
                    "ecr:*",
                    "s3:*",
                    "logs:*",
                ],
                resources=["*"],
            )
        )

        # 5️⃣ Required EC2 permissions for VPC CodeBuild
        build_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "ec2:CreateNetworkInterface",
                    "ec2:DescribeNetworkInterfaces",
                    "ec2:DeleteNetworkInterface",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeSecurityGroups",
                    "ec2:DescribeVpcs",
                ],
                resources=["*"],
            )
        )

        # 6️⃣ Source Stage
        source_action = cp_actions.CodeStarConnectionsSourceAction(
            action_name="GitHub_Source",
            connection_arn=codestar_connection_arn,
            owner=github_owner,
            repo=github_repo,
            branch=github_branch,
            output=source_output,
            trigger_on_push=True,
            code_build_clone_output=True,
        )

        # 7️⃣ Build Stage
        build_action = cp_actions.CodeBuildAction(
            action_name="Build_SageMaker_Pipeline",
            project=build_project,
            input=source_output,
            outputs=[build_output],
        )

        # 8️⃣ CodePipeline
        pipeline = codepipeline.Pipeline(
            self,
            "SageMakerPipeline",
            pipeline_name="Sample-MLOPs-ModelBuild",
            stages=[
                codepipeline.StageProps(
                    stage_name="Source",
                    actions=[source_action],
                ),
                codepipeline.StageProps(
                    stage_name="Build",
                    actions=[build_action],
                ),
            ],
        )

        # 9️⃣ Output
        CfnOutput(
            self,
            "PipelineName",
            value=pipeline.pipeline_name,
        )