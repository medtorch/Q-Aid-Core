import numpy as np
import torch
from torch.autograd import Variable

from pytorchxai.xai.gradient_vanilla_backprop import VanillaBackprop
from pytorchxai.xai.utils import convert_to_grayscale, normalize_gradient


class SmoothGrad:
    def __init__(self, model):
        self.model = model

        self.backprop = VanillaBackprop(model)

    def generate_smooth_grad(
        self, prep_img, target_class, param_n, param_sigma_multiplier
    ):
        """
            Generates smooth gradients of given Backprop type.
            You can use this with both vanilla and guided backprop
        Args:
            prep_img (torch Variable): preprocessed image
            target_class (int): target class of imagenet
            param_n (int): Amount of images used to smooth gradient
            param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
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
        param_n = 50
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
