#!/usr/bin/env python3

from aws_cdk import core

from backend.stack import FargateStack


app = core.App()
FargateStack(app, "med-qaid-backend", env={"region": "eu-central-1"})

app.synth()
