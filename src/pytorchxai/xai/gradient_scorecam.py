"""Score-CAM is a gradient-free visualization method, extended from Grad-CAM and Grad-CAM++. It achieves better visual performance and fairness for interpreting the decision making process.

Algorithm details:
 - For an input image, it extracts and saves the K activation maps from the last convolutional layer.
 - The activation maps are normalized using the maximum and minimum for each map.
 - The activations maps are multiplied by the original image to create K images.
 - Each of the generated images becomes an input to the CNN, and the probability of the target class (probability through softmax rather than score) is calculated.
 - The K probability values are regarded as the importance levels of the K Activation Maps, and the importance levels are multiplied by the activation maps, and then added together to obtain the ScoreCAM.
 -

[1] Wang et al. "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"

"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from pytorchxai.xai.gradient_cam import CamExtractor
from pytorchxai.xai.utils import apply_colormap_on_image


class ScoreCam:
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

        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :], 0), 0)
            # Upsampling to input size
            saliency_map = F.interpolate(
                saliency_map, size=(224, 224), mode="bilinear", align_corners=False,
            )
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (
                saliency_map.max() - saliency_map.min()
            )
            # Get the target score
            w = F.softmax(
                self.extractor.forward_pass(input_image * norm_saliency_map)[1], dim=1,
            )[0][target_class]
            cam += w.data.numpy() * target[i, :, :].data.numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
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
                Colored and grayscale ScoreCam heatmaps.
                Heatmap over the original image
        """

        cam = self.generate_cam(input_image, target_class)
        heatmap, heatmap_on_image = apply_colormap_on_image(orig_image, cam, "hsv")
        return {
            "scorecam_heatmap": T.ToTensor()(heatmap),
            "scorecam_heatmap_on_image": T.ToTensor()(heatmap_on_image),
            "scorecam_grayscale": T.ToTensor()(cam),
        }
