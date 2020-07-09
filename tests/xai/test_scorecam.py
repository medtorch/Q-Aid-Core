import pytest

from tests.xai.utils import MODELS, create_image

from pytorchxai.xai.gradient_scorecam import ScoreCam


@pytest.mark.parametrize("model", MODELS[:1])
def test_sanity(model):
    generator = ScoreCam(model)
    assert generator is not None


@pytest.mark.parametrize("model", MODELS[:1])
def test_generate_cam(model):
    generator = ScoreCam(model)

    _, test_image, test_target = create_image()

    guided_grads = generator.generate_cam(test_image, test_target)

    assert guided_grads.shape == (224, 224)


@pytest.mark.parametrize("model", MODELS[:1])
def test_generate(model):
    generator = ScoreCam(model)

    test_image, test_input, test_target = create_image()

    output = generator.generate(test_image, test_input, test_target)

    expected = [
        "scorecam_heatmap",
        "scorecam_heatmap_on_image",
        "scorecam_grayscale",
    ]

    for check in expected:
        assert check in output
        print(check)
        if "grayscale" in check:
            assert output[check].shape == (1, 224, 224)
        else:
            assert output[check].shape == (
                4,
                224,
                224,
            )  # scorecam adds alpha to the image
