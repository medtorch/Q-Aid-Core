import torch

from pytorchxai.xai.utils import convert_to_grayscale, normalize_gradient


class VanillaBackprop:
    """
        Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None

        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

    def generate(self, input_image, target_class):
        vanilla_grads = self.generate_gradients(input_image, target_class)

        color_vanilla_bp = normalize_gradient(vanilla_grads)

        grayscale_vanilla_bp = convert_to_grayscale(vanilla_grads)
        grayscale_vanilla_bp = normalize_gradient(grayscale_vanilla_bp)

        grad_times_image = vanilla_grads[0] * input_image.detach().numpy()[0]
        grad_times_image = convert_to_grayscale(grad_times_image)
        grad_times_image = normalize_gradient(grad_times_image)

        return {
            "vanilla_colored_backpropagation": color_vanilla_bp,
            "vanilla_grayscale_backpropagation": grayscale_vanilla_bp,
            "vanilla_grayscale_grad_times_image": grad_times_image,
        }
