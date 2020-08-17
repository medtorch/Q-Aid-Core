"""
Integrated Gradients aims to explain the relationship between a model's predictions in terms of its features. It has many use cases including understanding feature importances, identifying data skew, and debugging model performance.

Algorithm details:
 - It constructs a sequence of images interpolating from a baseline (black) to the actual image.
 - It averages the gradients across these images.

Other use cases:
 - Text Classification
 - Language translation
 - Search Ranking

[1] Sundararajan, Taly, Yan et al. "Axiomatic Attribution for Deep Networks.", Proceedings of International Conference on Machine Learning (ICML), 2017.
[2] https://github.com/ankurtaly/Integrated-Gradients
"""
import numpy as np
import torch

from .utils import convert_to_grayscale, normalize_gradient


class IntegratedGradients:
    """
        Produces gradients generated with integrated gradients from the image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None

        self.model.eval()
        self._hook_layers()

    def _hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        """
            Generates "steps" intermediary images.

            Args:
                input_image: Preprocessed input image.
                steps: Numbers of intermediary images.
            Returns:
                An array of "steps" images.
        """
        step_list = np.arange(steps + 1) / steps

        return [input_image * step for step in step_list]

    def generate_gradients(self, input_image, target_class):
        """
            Generates the gradients for the given model and image.

            Args:
                input_image: Preprocessed input image.
                target_class: Expected category.
            Returns:
                The gradients.
        """
        model_output = self.model(input_image)
        self.model.zero_grad()

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        model_output.backward(gradient=one_hot_output)

        return self.gradients.data.numpy()[0]

    def generate_integrated_gradients(self, input_image, target_class, steps):
        """
            Generates "steps" intermediary images and generates gradients for all of them. Returns the average of the gradients.

            Args:
                input_image: Preprocessed input image.
                target_class: Expected category.
                steps: Numbers of intermediay images.
            Returns:
                The gradients' average.
        """
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = np.zeros(input_image.size())

        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            integrated_grads = integrated_grads + single_integrated_grad / steps

        return integrated_grads[0]

    def generate(self, orig_image, input_image, target_class):
        """
            Generates heatmaps using the integrated gradient method.

            Args:
                orig_image: The original image.
                input_image: Preprocessed input image.
                target_class: Expected category.
            Returns:
                The heatmaps.
        """
        integrated_grads = self.generate_integrated_gradients(
            input_image, target_class, 5
        )
        grayscale_integrated_grads = normalize_gradient(
            convert_to_grayscale(integrated_grads)
        )

        grad_times_image = integrated_grads[0] * input_image.detach().numpy()[0]
        grad_times_image = convert_to_grayscale(grad_times_image)
        grad_times_image = normalize_gradient(grad_times_image)

        return {
            "integrated_gradients": grayscale_integrated_grads,
            "integrated_gradients_times_image": grad_times_image,
        }
