import pytest

from tests.xai.utils import MODELS, create_image

from pytorchxai.xai.gradient_gradcam import CamExtractor, GradCam


@pytest.mark.parametrize("model", MODELS)
def test_sanity(model):
    generator = GradCam(model)
    assert generator is not None


@pytest.mark.parametrize("model", MODELS)
def test_generate_cam(model):
    generator = GradCam(model)

    _, test_image, test_target = create_image()

    guided_grads = generator.generate_cam(test_image, test_target)

    assert guided_grads.shape == (224, 224)


@pytest.mark.parametrize("model", MODELS)
def test_generate(model):
    generator = GradCam(model)

    test_image, test_input, test_target = create_image()

    output = generator.generate(test_image, test_input, test_target)

    expected = [
        "gradcam_heatmap",
        "gradcam_heatmap_on_image",
        "gradcam_grayscale",
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
            )  # gradcam adds alpha to the image


@pytest.mark.parametrize("model", MODELS)
def test_extractor(model):
    extractor = CamExtractor(model)
    _, test_input, _ = create_image()

    conv_output, model_output = extractor.forward_pass(test_input)

    assert model_output.shape[1] == list(model.children())[-1][-1].out_features
    assert conv_output is not None
