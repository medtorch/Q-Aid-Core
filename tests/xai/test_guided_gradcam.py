import pytest

from .utils import MODELS, create_image

from pytorchxai.xai.gradient_guided_gradcam import GuidedGradCam


@pytest.mark.parametrize("model", MODELS)
def test_sanity(model):
    generator = GuidedGradCam(model)
    assert generator is not None


@pytest.mark.parametrize("model", MODELS)
def test_generate(model):
    generator = GuidedGradCam(model)

    test_image, test_input, test_target = create_image()

    output = generator.generate(test_image, test_input, test_target)

    expected = [
        "guided_gradcam",
        "guided_gradcam_grayscale",
    ]

    for check in expected:
        assert check in output
        if "grayscale" in check:
            assert output[check].shape == (1, 224, 224)
        else:
            assert output[check].shape == (3, 224, 224)
