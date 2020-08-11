#!/usr/bin/env python3

from aws_cdk import core

from backend.stack import FargateStack


app = core.App()
FargateStack(app, "med-qaid-core-backend-v3", env={"region": "eu-central-1"})

app.synth()
