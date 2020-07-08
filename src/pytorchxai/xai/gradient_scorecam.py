import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from pytorchxai.xai.utils import apply_colormap_on_image


class CamExtractor:
    """
        Extracts cam features from the model
    """

    def __init__(self, model):
        self.model = model
        self.last_conv = None
        for module_pos, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.Conv2d):
                self.last_conv = module_pos
        if not self.last_conv:
            raise ("invalid input model")

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if module_pos == self.last_conv:
                conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class ScoreCam:
    """
        Produces class activation map
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
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
        cam = self.generate_cam(input_image, target_class)
        heatmap, heatmap_on_image = apply_colormap_on_image(orig_image, cam, "hsv")
        return {
            "scorecam_heatmap": T.ToTensor()(heatmap),
            "scorecam_heatmap_on_image": T.ToTensor()(heatmap_on_image),
            "scorecam_grayscale": T.ToTensor()(cam),
        }
