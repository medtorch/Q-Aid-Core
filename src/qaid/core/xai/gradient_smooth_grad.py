"""
SmoothGrad technique is adding some Gaussian noise to the original image and calculating gradients multiple times and averaging the results.
It can augment other sensitivity techniques, such as: vanilla gradients, integrated gradients, guided backpropagation or GradCam.


[1] D. Smilkov, N. Thorat, N. Kim, F. Viégas, M. Wattenberg. "SmoothGrad: removing noise by adding noise", 2017.
"""

import numpy as np
import torch
from torch.autograd import Variable

from .gradient_vanilla_backprop import VanillaBackprop
from .utils import convert_to_grayscale, normalize_gradient


class SmoothGrad:
    def __init__(self, model):
        self.model = model

        self.backprop = VanillaBackprop(model)

    def generate_smooth_grad(
        self, prep_img, target_class, param_n, param_sigma_multiplier
    ):
        """
            Generates smooth gradients of given backprop type: vanilla or guided.
        Args:
            prep_img (torch Variable): preprocessed image.
            target_class (int): target class of imagenet
            param_n (int): Amount of images used to smooth gradient.
            param_sigma_multiplier (int): Sigma multiplier when calculating std of noise.
        Returns:
            The gradients.
        """
        # Generate an empty image/matrix
        smooth_grad = np.zeros(prep_img.size()[1:])

        mean = 0
        sigma = (
            param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
        )
        for x in range(param_n):
            # Generate noise
            noise = Variable(
                prep_img.data.new(prep_img.size()).normal_(mean, sigma ** 2)
            )
            # Add noise to the image
            noisy_img = prep_img + noise
            # Calculate gradients
            vanilla_grads = self.backprop.generate_gradients(noisy_img, target_class)
            # Add gradients to smooth_grad
            smooth_grad = smooth_grad + vanilla_grads
        # Average it out
        smooth_grad = smooth_grad / param_n
        return smooth_grad

    def generate(self, orig_image, input_image, target_class):
        """
            Generates and returns multiple sensitivy heatmaps, based on SmoothGrad technique.
            Args:
                orig_image: Original resized image.
                input_image: Preprocessed input image.
                target_class: Expected category.
            Returns:
                Colored and grayscale gradients for the SmoothGrad backpropagation.
        """
        param_n = 5
        param_sigma_multiplier = 4
        smooth_grad = self.generate_smooth_grad(
            input_image, target_class, param_n, param_sigma_multiplier
        )

        color_smooth_grad = normalize_gradient(smooth_grad)

        grayscale_smooth_grad = convert_to_grayscale(smooth_grad)
        grayscale_smooth_grad = normalize_gradient(grayscale_smooth_grad)

        return {
            "smooth_grad_colored": color_smooth_grad,
            "smooth_grad_grayscale": grayscale_smooth_grad,
        }
