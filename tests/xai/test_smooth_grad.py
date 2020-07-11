import pytest

from .utils import MODELS, create_image

from pytorchxai.xai.gradient_smooth_grad import SmoothGrad


@pytest.mark.parametrize("model", MODELS)
def test_sanity(model):
    generator = SmoothGrad(model)
    assert generator is not None


@pytest.mark.parametrize("model", MODELS)
def test_generate_smooth_grad(model):
    generator = SmoothGrad(model)

    _, test_image, test_target = create_image()

    guided_grads = generator.generate_smooth_grad(test_image, test_target, 5, 4)

    assert guided_grads.shape == (3, 224, 224)


@pytest.mark.parametrize("model", MODELS)
def test_generate(model):
    generator = SmoothGrad(model)

    test_image, test_input, test_target = create_image()

    output = generator.generate(test_image, test_input, test_target)

    expected = [
        "smooth_grad_colored",
        "smooth_grad_grayscale",
    ]

    for check in expected:
        assert check in output
        if "grayscale" in check:
            assert output[check].shape == (1, 224, 224)
        else:
            assert output[check].shape == (3, 224, 224)
