import pytest

from .utils import MODELS, create_image

from pytorchxai.xai.gradient_vanilla_backprop import VanillaBackprop


@pytest.mark.parametrize("model", MODELS)
def test_sanity(model):
    generator = VanillaBackprop(model)
    assert generator is not None


@pytest.mark.parametrize("model", MODELS)
def test_generate_gradients(model):
    generator = VanillaBackprop(model)

    _, test_image, test_target = create_image()

    guided_grads = generator.generate_gradients(test_image, test_target)

    assert guided_grads.shape == (3, 224, 224)


@pytest.mark.parametrize("model", MODELS)
def test_generate(model):
    generator = VanillaBackprop(model)

    test_image, test_input, test_target = create_image()

    output = generator.generate(test_image, test_input, test_target)

    expected = [
        "vanilla_colored_backpropagation",
        "vanilla_grayscale_backpropagation",
        "vanilla_grayscale_grad_times_image",
    ]

    for check in expected:
        assert check in output
        if "grayscale" in check:
            assert output[check].shape == (1, 224, 224)
        else:
            assert output[check].shape == (3, 224, 224)
