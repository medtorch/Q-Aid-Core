import pytest

from .utils import MODELS, create_image

from pytorchxai.xai.gradient_integrated_grad import IntegratedGradients


@pytest.mark.parametrize("model", MODELS)
def test_sanity(model):
    generator = IntegratedGradients(model)
    assert generator is not None


@pytest.mark.parametrize("model", MODELS)
def test_generate(model):
    generator = IntegratedGradients(model)

    test_image, test_input, test_target = create_image()

    output = generator.generate(test_image, test_input, test_target)

    expected = [
        "integrated_gradients",
        "integrated_gradients_times_image",
    ]

    for check in expected:
        assert check in output
        assert output[check].shape == (1, 224, 224)


@pytest.mark.parametrize("model", MODELS)
def test_generate_images_on_linear_path(model):
    generator = IntegratedGradients(model)

    _, test_input, _ = create_image()

    imgs = generator.generate_images_on_linear_path(test_input, 10)

    assert len(imgs) == 11
    for img in imgs:
        assert img.shape == (1, 3, 224, 224)


@pytest.mark.parametrize("model", MODELS)
def test_generate_integrated_gradients(model):
    generator = IntegratedGradients(model)

    _, test_input, test_target = create_image()

    output = generator.generate_integrated_gradients(test_input, test_target, 2)

    assert output.shape == (3, 224, 224)
