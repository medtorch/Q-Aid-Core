import numpy as np
import torch
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

        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if module_pos == self.last_conv:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
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


class GradCam:
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
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(
            guided_gradients, axis=(1, 2)
        )  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = (
            np.uint8(
                Image.fromarray(cam).resize(
                    (input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS
                )
            )
            / 255
        )
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam

    def generate(self, orig_image, input_image, target_class=None):
        cam = self.generate_cam(input_image, target_class)
        heatmap, heatmap_on_image = apply_colormap_on_image(orig_image, cam, "hsv")
        return {
            "gradcam_heatmap": T.ToTensor()(heatmap),
            "gradcam_heatmap_on_image": T.ToTensor()(heatmap_on_image),
            "gradcam_grayscale": T.ToTensor()(cam),
        }
