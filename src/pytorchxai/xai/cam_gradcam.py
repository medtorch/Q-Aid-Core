"""
Gradient-weighted Class Activation Mapping(Grad-CAM) is an algorithm that can be used to visualize the class activation maps of a Convolutional Neural Network.

Algorithm details:
 - The algorithm finds the final convolutional layer in the network.
 - It examines the gradient information flowing into that layer.
 - The output of Grad-CAM is a heatmap visualization for a given class label.

[1] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE International Conference on Computer Vision. 2017.
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from pytorchxai.xai.cam_utils import CamExtractor
from pytorchxai.xai.utils import apply_colormap_on_image


class GradCam:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_class=None):
        """
            Does a full forward pass on the model and generates the activations maps.
            Args:
                input_image: input image
                target_class: optional target class
            Returns:
                The activations maps.
        """
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()

        model_output.backward(gradient=one_hot_output, retain_graph=True)
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        target = conv_output.data.numpy()[0]

        weights = np.mean(guided_gradients, axis=(1, 2))

        cam = np.ones(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        cam = (
            np.uint8(
                Image.fromarray(cam).resize(
                    (input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS,
                )
            ) / 255
        )
        return cam

    def generate(self, orig_image, input_image, target_class=None):
        """
            Generates and returns the activations maps.

            Args:
                orig_image: Original resized image.
                input_image: Preprocessed input image.
                target_class: Expected category.
            Returns:
                Colored and grayscale Grad-Cam heatmaps.
                Heatmap over the original image
        """
        cam = self.generate_cam(input_image, target_class)
        heatmap, heatmap_on_image = apply_colormap_on_image(orig_image, cam, "hsv")
        return {
            "gradcam_heatmap": T.ToTensor()(heatmap),
            "gradcam_heatmap_on_image": T.ToTensor()(heatmap_on_image),
            "gradcam_grayscale": T.ToTensor()(cam),
        }
