"""
Guided Backpropagation(guided saliency) generates heat maps that are intended to provide insight into what aspects of an input image a convolutional neural network is using to make a prediction, focusing on what image features the neuron detects, not in what kind of stuff it doesn’t detect.

For that, we backpropagate positive error signals – i.e. we set the negative gradients to zero. This is the application of the ReLU to the error signal itself during the backward pass.

Like vanilla backpropagation, we also restrict ourselves to only positive inputs.
"""
import torch
from torch.nn import ReLU

from pytorchxai.xai.utils import (
    convert_to_grayscale,
    get_positive_negative_saliency,
    normalize_gradient
)


class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []

        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        """
            Method for registering a hook to the first layer
        """

        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Updates relu activation functions so that:
                - they store the output in the forward pass.
                - they set the negative gradients to zero.
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
                If there is a negative gradient, change it to zero
            """
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0
            )
            del self.forward_relu_outputs[-1]
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
                Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        """
            Generates the gradients using guided backpropagation from the given model and image.

            Args:
                input_image: Preprocessed input image.
                target_class: Expected category.
            Returns:
                The gradients computed using the guided backpropagation.
        """
        model_output = self.model(input_image)
        self.model.zero_grad()

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        model_output.backward(gradient=one_hot_output)

        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

    def generate(self, orig_image, input_image, target_class):
        """
            Generates and returns multiple saliency maps, based on guided backpropagation.

            Args:
                input_image: Preprocessed input image.
                target_class: Expected category.
            Returns:
                Colored and grayscale gradients for the guided backpropagation.
                Positive and negative saliency maps.
                Grayscale gradients multiplied with the image itself.
        """
        guided_grads = self.generate_gradients(input_image, target_class)

        color_guided_grads = normalize_gradient(guided_grads)
        grayscale_guided_grads = normalize_gradient(convert_to_grayscale(guided_grads))

        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        pos_sal_grads = normalize_gradient(pos_sal)
        neg_sal_grads = normalize_gradient(neg_sal)

        grad_times_image = guided_grads[0] * input_image.detach().numpy()[0]
        grad_times_image = convert_to_grayscale(grad_times_image)
        grad_times_image = normalize_gradient(grad_times_image)

        return {
            "guided_grads_colored": color_guided_grads,
            "guided_grads_grayscale": grayscale_guided_grads,
            "guided_grads_grayscale_grad_times_image": grad_times_image,
            "saliency_maps_positive": pos_sal_grads,
            "saliency_maps_negative": neg_sal_grads,
        }
