"""
Class Activation Mapping helpers
"""
import torch


class CamExtractor:
    def __init__(self, model):
        self.model = model
        self.last_conv = None
        for module_pos, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.Conv2d):
                self.last_conv = module_pos
        if not self.last_conv:
            raise ("invalid input model")

        self.gradients = None

    def _save_gradient(self, grad):
        self.gradients = grad

    def _forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
            Args:
                x: input image
            Returns:
                The output of the last convolutional layer.
                The output of the model.
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)
            if module_pos == self.last_conv:
                x.register_hook(self._save_gradient)
                conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
            Args:
                x: input image
            Returns:
                The output of the last convolutional layer.
                The output of the model.
        """
        conv_output, x = self._forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return conv_output, x
