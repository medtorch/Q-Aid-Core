import pytest

import numpy as np
from PIL import Image
from torchvision import models

import pytorchxai.xai.gradient_guided_backprop as ctx
from pytorchxai.xai.utils import preprocess_image


def create_image(width=244, height=244):
    width = int(width)
    height = int(height)

    rgb_array = np.random.rand(height, width, 3) * 255
    image = Image.fromarray(rgb_array.astype("uint8")).convert("RGB")
    prep = preprocess_image(image)

    return image, prep


@pytest.mark.parametrize(
    "model", [models.alexnet(pretrained=True), models.vgg19(pretrained=True)]
)
def test_sanity(model):
    generator = ctx.GuidedBackprop(model)
    assert generator is not None


@pytest.mark.parametrize(
    "model", [models.alexnet(pretrained=True), models.vgg19(pretrained=True)]
)
def test_generate_gradients(model):
    generator = ctx.GuidedBackprop(model)

    _, test_image = create_image()
    test_target = 55

    guided_grads = generator.generate_gradients(test_image, test_target)

    assert guided_grads.shape == (3, 224, 224)


@pytest.mark.parametrize(
    "model", [models.alexnet(pretrained=True), models.vgg19(pretrained=True)]
)
def test_generate(model):
    generator = ctx.GuidedBackprop(model)

    test_image, test_input = create_image()
    test_target = 55

    output = generator.generate(test_image, test_input, test_target)

    expected = [
        "guided_grads_colored",
        "guided_grads_grayscale",
        "guided_grads_grayscale_grad_times_image",
        "saliency_maps_positive",
        "saliency_maps_negative",
    ]

    for check in expected:
        assert check in output
        if "grayscale" in check:
            assert output[check].shape == (1, 224, 224)
        else:
            assert output[check].shape == (3, 224, 224)
