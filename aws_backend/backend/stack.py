import os
import urllib.parse

from aws_cdk import core

import aws_cdk.aws_certificatemanager as certificatemanager
import aws_cdk.aws_cognito as cognito
import aws_cdk.aws_ec2 as ec2
import aws_cdk.aws_ecs as ecs
import aws_cdk.aws_ecs_patterns as ecs_patterns
import aws_cdk.aws_ecr_assets as ecr_assets
import aws_cdk.aws_elasticloadbalancingv2 as elb
import aws_cdk.aws_route53 as route53


APP_DNS_NAME = "q-and-aid.com"
HOSTED_ZONE_ID = "Z09644041TWEBPC10I0YZ"
HOSTED_ZONE_NAME = "q-and-aid.com"


class FargateStack(core.Stack):
    def __init__(self, scope: core.Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Get the hosted Zone and create a certificate for our domain

        hosted_zone = route53.HostedZone.from_hosted_zone_attributes(
            self,
            "HostedZone",
            hosted_zone_id=HOSTED_ZONE_ID,
            zone_name=HOSTED_ZONE_NAME,
        )

        cert = certificatemanager.DnsValidatedCertificate(
            self, "Certificate", hosted_zone=hosted_zone, domain_name=APP_DNS_NAME
        )

        # Set up a new VPC

        vpc = ec2.Vpc(self, "med-qaid-vpc", max_azs=2)

        # Set up an ECS Cluster for fargate

        cluster = ecs.Cluster(self, "med-qaid-cluster", vpc=vpc)

        # Define the Docker Image for our container (the CDK will do the build and push for us!)
        docker_image = ecr_assets.DockerImageAsset(
            self,
            "med-qaid-app",
            directory=os.path.join(os.path.dirname(__file__), "..", "src"),
        )

        # Define the fargate service + ALB

        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "FargateService",
            cluster=cluster,
            certificate=cert,
            domain_name=f"{APP_DNS_NAME}",
            domain_zone=hosted_zone,
            cpu=2048,
            memory_limit_mib=16384,
            task_image_options={
                "image": ecs.ContainerImage.from_docker_image_asset(docker_image),
                "environment": {"PORT": "80",},
            },
        )

        # Allow 10 seconds for in flight requests before termination, the default of 5 minutes is much too high.
        fargate_service.target_group.set_attribute(
            key="deregistration_delay.timeout_seconds", value="10"
        )
