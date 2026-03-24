from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_s3 as s3,
    CfnOutput,
    RemovalPolicy,
)
from constructs import Construct


class SageMakerStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc_id: str,
        private_subnet_ids: list[str],
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # -----------------------------
        # 🪣 S3 bucket for SageMaker
        # -----------------------------
        sagemaker_bucket = s3.Bucket(
            self,
            "SageMakerBucket",
            bucket_name=f"sagemaker-domain-bucket-{self.account}-{self.region}",
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        # -----------------------------
        # IAM Role for SageMaker
        # -----------------------------
        sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
            ]
        )

        # Grant role access to S3 bucket
        sagemaker_bucket.grant_read_write(sagemaker_role)

        # -----------------------------
        # 🧠 SageMaker Studio Domain
        # -----------------------------
        domain = sagemaker.CfnDomain(
            self,
            "SageMakerStudioDomain",
            auth_mode="IAM",
            domain_name="StudioDomainMLOPs",
            subnet_ids=private_subnet_ids,
            vpc_id=vpc_id,
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=sagemaker_role.role_arn,
                jupyter_server_app_settings=sagemaker.CfnDomain.JupyterServerAppSettingsProperty(),
                kernel_gateway_app_settings=sagemaker.CfnDomain.KernelGatewayAppSettingsProperty(
                    default_resource_spec=sagemaker.CfnDomain.ResourceSpecProperty(
                        instance_type="ml.t3.medium"
                    )
                )
            )
        )

        # -----------------------------
        # 👥 SageMaker Studio Users
        # -----------------------------
        users = ["MLOPs", "Devs"]
        for username in users:
            sagemaker.CfnUserProfile(
                self,
                f"{username}UserProfile",
                domain_id=domain.attr_domain_id,
                user_profile_name=username,
                user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                    execution_role=sagemaker_role.role_arn,
                    jupyter_server_app_settings=sagemaker.CfnUserProfile.JupyterServerAppSettingsProperty(),
                    kernel_gateway_app_settings=sagemaker.CfnUserProfile.KernelGatewayAppSettingsProperty(
                        default_resource_spec=sagemaker.CfnUserProfile.ResourceSpecProperty(
                            instance_type="ml.t3.medium"
                        )
                    )
                )
            )

        # -----------------------------
        # 📤 Outputs
        # -----------------------------
        CfnOutput(self, "DomainID", value=domain.attr_domain_id)
        CfnOutput(self, "VPCID", value=vpc_id)
        CfnOutput(self, "PrivateSubnets", value=",".join(private_subnet_ids))
        CfnOutput(self, "SageMakerRole", value=sagemaker_role.role_arn)
        CfnOutput(self, "SageMakerBucketName", value=sagemaker_bucket.bucket_name)