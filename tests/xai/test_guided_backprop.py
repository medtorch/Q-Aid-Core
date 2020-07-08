import pytest

from tests.xai.utils import create_image
from torchvision import models

from pytorchxai.xai.gradient_guided_backprop import GuidedBackprop


@pytest.mark.parametrize(
    "model", [models.alexnet(pretrained=True), models.vgg19(pretrained=True)]
)
def test_sanity(model):
    generator = GuidedBackprop(model)
    assert generator is not None


@pytest.mark.parametrize(
    "model", [models.alexnet(pretrained=True), models.vgg19(pretrained=True)]
)
def test_generate_gradients(model):
    generator = GuidedBackprop(model)

    _, test_image, test_target = create_image()

    guided_grads = generator.generate_gradients(test_image, test_target)

    assert guided_grads.shape == (3, 224, 224)


@pytest.mark.parametrize(
    "model", [models.alexnet(pretrained=True), models.vgg19(pretrained=True)]
)
def test_generate(model):
    generator = GuidedBackprop(model)

    test_image, test_input, test_target = create_image()

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
