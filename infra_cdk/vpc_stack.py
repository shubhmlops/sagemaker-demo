from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    Tags,
    CfnOutput,
)
from constructs import Construct


class VpcStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)


        vpc = ec2.Vpc(
            self,
            "sample-vpc-{env}",
            ip_addresses=ec2.IpAddresses.cidr("10.0.0.0/16"),
            max_azs=2,
            subnet_configuration=[],   # disables auto subnets
        )

        Tags.of(vpc).add("Name", "sample-vpc-{env}")

        # -----------------------------
        # Public Subnets
        # -----------------------------
        public_subnet_1 = ec2.CfnSubnet(
            self,
            "sample-subnet-{env}-01",
            vpc_id=vpc.vpc_id,
            cidr_block="10.0.1.0/24",
            availability_zone="us-east-1a",
            map_public_ip_on_launch=True
        )
        Tags.of(public_subnet_1).add("Name", "sample-subnet-public-{env}-01")

        public_subnet_2 = ec2.CfnSubnet(
            self,
            "sample-subnet-{env}-02",
            vpc_id=vpc.vpc_id,
            cidr_block="10.0.2.0/24",
            availability_zone="us-east-1b",
            map_public_ip_on_launch=True
        )
        Tags.of(public_subnet_2).add("Name", "sample-subnet-public-{env}-02")

        # -----------------------------
        # Private Subnets
        # -----------------------------
        private_subnet_1 = ec2.CfnSubnet(
            self,
            "sample-subnet-{env}-03",
            vpc_id=vpc.vpc_id,
            cidr_block="10.0.3.0/24",
            availability_zone="us-east-1a",
            map_public_ip_on_launch=False
        )
        Tags.of(private_subnet_1).add("Name", "sample-subnet-private-{env}-01")

        private_subnet_2 = ec2.CfnSubnet(
            self,
            "sample-subnet-{env}-04",
            vpc_id=vpc.vpc_id,
            cidr_block="10.0.4.0/24",
            availability_zone="us-east-1b",
            map_public_ip_on_launch=False
        )
        Tags.of(private_subnet_2).add("Name", "sample-subnet-private-{env}-02")

        # =============================
        # 3. Internet Gateway + Attach
        # =============================
        igw = ec2.CfnInternetGateway(self, "InternetGateway")
        Tags.of(igw).add("Name", "sample-igw")

        igw_attach = ec2.CfnVPCGatewayAttachment(
            self,
            "IGWAttachment",
            vpc_id=vpc.vpc_id,
            internet_gateway_id=igw.ref
        )

        # =============================
        #  Public Route Table + Route
        # =============================
        public_rt = ec2.CfnRouteTable(
            self,
            "public-route-table",
            vpc_id=vpc.vpc_id
        )

        # Add Name tag to Public Route Table
        Tags.of(public_rt).add("Name", "public-route-table-{env}")

        # Internet Route for Public Subnets
        public_route = ec2.CfnRoute(
            self,
            "public-default-route",
            route_table_id=public_rt.ref,
            destination_cidr_block="0.0.0.0/0",
            gateway_id=igw.ref
        )

        public_route.add_dependency(igw_attach)

        ec2.CfnSubnetRouteTableAssociation(
            self,
            "public-subnet-01-association",
            subnet_id=public_subnet_1.ref,
            route_table_id=public_rt.ref
        )

        ec2.CfnSubnetRouteTableAssociation(
            self,
            "public-subnet-02-association",
            subnet_id=public_subnet_2.ref,
            route_table_id=public_rt.ref
        )

        # =============================
        # 5. NAT Gateway for private egress
        # =============================
        eip = ec2.CfnEIP(self, "NatEip", domain="vpc")
        Tags.of(eip).add("Name", "sample-eip-{env}")

        nat_gw = ec2.CfnNatGateway(
            self,
            "nat-gateway",
            subnet_id=public_subnet_1.ref,
            allocation_id=eip.attr_allocation_id
        )
        nat_gw.add_dependency(public_subnet_1)
        nat_gw.add_dependency(eip)

        Tags.of(nat_gw).add("Name", "sample-natgw")

        # -----------------------------
        # Private Route Table
        # -----------------------------
        private_rt = ec2.CfnRouteTable(
            self,
            "private-route-table",
            vpc_id=vpc.vpc_id
        )

        # Add Name tag to Private Route Table
        Tags.of(private_rt).add("Name", "private-route-table-{env}")

        private_route = ec2.CfnRoute(
            self,
            "PrivateDefaultRoute",
            route_table_id=private_rt.ref,
            destination_cidr_block="0.0.0.0/0",
            nat_gateway_id=nat_gw.ref
        )

        private_route.add_dependency(nat_gw)

        ec2.CfnSubnetRouteTableAssociation(
            self,
            "private-subnet-1-association",
            subnet_id=private_subnet_1.ref,
            route_table_id=private_rt.ref
        )

        ec2.CfnSubnetRouteTableAssociation(
            self,
            "private-subnet-2-association",
            subnet_id=private_subnet_2.ref,
            route_table_id=private_rt.ref
        )

        # -----------------------------
        # Outputs
        # -----------------------------
        CfnOutput(self, "VpcId", value=vpc.vpc_id, export_name="VpcId")
        CfnOutput(self, "PublicSubnet1Id", value=public_subnet_1.ref, export_name="PublicSubnet1Id")
        CfnOutput(self, "PublicSubnet2Id", value=public_subnet_2.ref, export_name="PublicSubnet2Id")
        CfnOutput(self, "PrivateSubnet1Id", value=private_subnet_1.ref, export_name="PrivateSubnet1Id")
        CfnOutput(self, "PrivateSubnet2Id", value=private_subnet_2.ref, export_name="PrivateSubnet2Id")