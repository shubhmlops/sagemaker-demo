import json
from aws_cdk import (
    Stack,
    aws_ssm as ssm,
)
from constructs import Construct
from pathlib import Path

class ParameterStoreStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # # Load JSON
        # params_path = Path(__file__).parent / "ssm-params-uat.json"

        # ✅ Go one level UP from infra_resources to project root
        params_path = Path(__file__).resolve().parents[1] / "ssm-params.json"

        with open(params_path) as f:
            parameter_values = json.load(f)

        for name, value in parameter_values.items():
            ssm.StringParameter(
                self,
                name.replace("/", "_"),  # logical ID must not contain "/"
                parameter_name=name,
                string_value=value,
                tier=ssm.ParameterTier.STANDARD
            )