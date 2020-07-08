import numpy as np

from pytorchxai.xai.gradient_gradcam import GradCam
from pytorchxai.xai.gradient_guided_backprop import GuidedBackprop
from pytorchxai.xai.utils import convert_to_grayscale, normalize_gradient


class GuidedGradCam:
    def __init__(self, model):
        self.model = model

        self.gradcam = GradCam(model)
        self.gbp = GuidedBackprop(model)

    def generate_cam(self, grad_cam_mask, guided_backprop_mask):
        """
            Guided grad cam is just pointwise multiplication of cam mask and
            guided backprop mask

        Args:
            grad_cam_mask (np_arr): Class activation map mask
            guided_backprop_mask (np_arr):Guided backprop mask
        """
        cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
        return cam_gb

    def generate(self, orig_image, input_image, target_class):
        cam = self.gradcam.generate_cam(input_image, target_class)
        guided_grads = self.gbp.generate_gradients(input_image, target_class)
        cam_gb = self.generate_cam(cam, guided_grads)

        guided_gradcam = normalize_gradient(cam_gb)
        grayscale_cam_gb = convert_to_grayscale(cam_gb)
        guided_gradcam_grayscale = normalize_gradient(grayscale_cam_gb)

        return {
            "guided_gradcam": guided_gradcam,
            "guided_gradcam_grayscale": guided_gradcam_grayscale,
        }
