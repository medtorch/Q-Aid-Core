"""
GradCAM helps vizualing which parts of an input image trigger the predicted class, by backpropagating the gradients to the last convolutional layer, producing a coarse heatmap.

Guided GradCAM is obtained by fusing GradCAM with Guided Backpropagation via element-wise multiplication, and results in a heatmap highliting much finer details.

This technique is only useful for inspecting an already trained network, not for training it, as the backpropagation on ReLU will be changed for computing the Guided Backpropagation.

[1] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE International Conference on Computer Vision. 2017.
"""
import numpy as np

from pytorchxai.xai.gradient_gradcam import GradCam
from pytorchxai.xai.gradient_guided_backprop import GuidedBackprop
from pytorchxai.xai.utils import convert_to_grayscale, normalize_gradient


class GuidedGradCam:
    def __init__(self, model):
        self.model = model

        self.gradcam = GradCam(model)
        self.gbp = GuidedBackprop(model)

    def generate(self, orig_image, input_image, target_class):
        """
            Guided gradcam is just pointwise multiplication of the cam mask and
            the guided backprop mask.

            Args:
                orig_image: Original resized image.
                input_image: Preprocessed input image.
                target_class: Expected category.
            Returns:
                Colored and grayscale gradients for the guided Grad-CAM.
        """
        cam = self.gradcam.generate_cam(input_image, target_class)
        guided_grads = self.gbp.generate_gradients(input_image, target_class)

        cam_gb = np.multiply(cam, guided_grads)

        guided_gradcam = normalize_gradient(cam_gb)
        grayscale_cam_gb = convert_to_grayscale(cam_gb)
        guided_gradcam_grayscale = normalize_gradient(grayscale_cam_gb)

        return {
            "guided_gradcam": guided_gradcam,
            "guided_gradcam_grayscale": guided_gradcam_grayscale,
        }
